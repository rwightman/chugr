import json
from typing import Optional, Callable

import numpy as np
import torch
from torchvision import tv_tensors

from chugr import ImageFeatureInfo, FeatureInfo
from chugr.doc import DocProcessor, DEFAULT_DOC_FEAT
from chugr.doc.doc_processor import get_next_valid_page_index, _get_value, _logger
from chugr.doc.regions import PatchLabelCfg, LineTokenizer, line_polys, token_polys, patch_labels, empty_patch_labels
from chugr.doc.synth_text import SynthTextCfg, SynthTextRenderer


class DocReadProcessor(DocProcessor):
    """ Process documents w/ OCR annotation for reading tasks.
    """

    def __init__(
            self,
            image_process_fn: Optional[Callable] = None,
            text_process_fn: Optional[Callable] = None,
            image_input_feat: ImageFeatureInfo = DEFAULT_DOC_FEAT,
            text_input_feat: FeatureInfo = FeatureInfo('text_input', input_key='pages'),
            text_target_feat: FeatureInfo = FeatureInfo('text_target', input_key=None),
            line_break: str = '\n',
            page_sampling: str = 'random',
            render_dpi: int = 150,
            squeeze_pages: bool = True,
            expand_pages: bool = False,
            flatten_json: bool = True,
            seed: int = 0,
            synth_prob: float = 0.,
            synth_cfg: Optional[SynthTextCfg] = None,
            patch_label_cfg: Optional[PatchLabelCfg] = None,
            patch_size: Optional[int] = None,
            tokenizer: Optional[Callable] = None,
    ):
        """
        Args:
            synth_prob: probability of replacing a sample's page image w/ a synthetic rendering of (part of) its
                annotation text (target == rendered text, no layout annotations used, page image not decoded)
            synth_cfg: synthetic text rendering config
            patch_label_cfg: enables patch level text labels ('patch_label_index', 'patch_label') from token regions
                (line polys sliced per token), the image transforms are called w/ the regions (torchvision v2)
            patch_size: model patch size for the patch labels
            tokenizer: tokenizer for the per token regions (same as the text targets, w/ a decode method)
        """
        super().__init__(
            image_process_fn=image_process_fn,
            text_process_fn=text_process_fn,
            image_input_feat=image_input_feat,
            text_input_feat=text_input_feat,
            text_target_feat=text_target_feat,
            render_dpi=render_dpi,
            page_sampling=page_sampling,
            squeeze_pages=squeeze_pages,
            expand_pages=expand_pages,
            flatten_json=flatten_json,
            seed=seed,
        )
        self.line_break = line_break
        assert page_sampling in ('random', 'first', 'all_valid', 'all')
        self.synth_prob = synth_prob
        self.synth = SynthTextRenderer(synth_cfg) if synth_prob > 0 else None
        self.patch_label_cfg = patch_label_cfg
        if patch_label_cfg is not None:
            assert patch_size and tokenizer is not None, 'patch labels require the patch size and a tokenizer'
            self.patch_size = patch_size
            self.line_tokenizer = LineTokenizer(tokenizer)
            blank_id = patch_label_cfg.blank_id
            self.blank_id = tokenizer.pad_token_id if blank_id is None else blank_id
        self._label_rng = None
        self._label_rng_worker = None

    @property
    def label_rng(self) -> np.random.Generator:
        # separate from the page / synth sampling generator so enabling labels doesn't change the data sampled
        from torch.utils.data import get_worker_info
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else -1
        if self._label_rng is None or worker_id != self._label_rng_worker:
            seed = (worker_info.seed if worker_info is not None else 0) + self.seed
            self._label_rng = np.random.default_rng([seed, 0x5eed])
            self._label_rng_worker = worker_id
        return self._label_rng

    def _preprocess_page(self, page, page_info):
        if self.patch_label_cfg is None or page_info is None:
            return self.image_process_fn(page), {}
        polys = page_info['polys']
        if polys is None:
            out = self.image_process_fn(page)
            index, label = empty_patch_labels(self.patch_label_cfg)
            return out, {'patch_label_index': index, 'patch_label': label}
        w, h = page.size
        scale = np.array([w, h], dtype=np.float32)
        ids, line_idx, spans = self.line_tokenizer(page_info['lines'])
        tok = token_polys(polys, line_idx, spans) * scale
        lines = polys * scale
        regions = tv_tensors.KeyPoints(
            torch.from_numpy(np.concatenate([tok, lines]).astype(np.float32)), canvas_size=(h, w))
        out, regions = self.image_process_fn(page, regions)
        regions = regions.as_subclass(torch.Tensor).numpy()
        if isinstance(out, dict):
            coord = out['patch_coord']
        else:
            gh, gw = out.shape[-2] // self.patch_size, out.shape[-1] // self.patch_size
            coord = torch.stack(torch.meshgrid(torch.arange(gh), torch.arange(gw), indexing='ij'), -1).reshape(-1, 2)
        index, label = patch_labels(
            regions[:len(tok)],
            ids,
            regions[len(tok):],
            coord,
            self.patch_size,
            self.patch_label_cfg,
            self.blank_id,
            self.label_rng,
        )
        return out, {'patch_label_index': index, 'patch_label': label}

    def __call__(self, sample):
        if self.synth is not None and self.generator.random() < self.synth_prob:
            return self._synth_sample(sample)
        return super().__call__(sample)

    def _process_text(self, text):
        """ Tokenize text (if text_process_fn set), returns (text input, text target or None). """
        if self.text_process_fn is None:
            return text, None
        processed = self.text_process_fn(text)
        assert self.text_input_name in processed, \
            f"Text input name '{self.text_input_name}' not found in processed sample."
        target = processed.get(self.text_target_name) if self.text_target_name else None
        if self.text_target_feat is not None and target is None:
            assert False, f"Expected a text target named '{self.text_target_name}' in processed sample."
        return processed[self.text_input_name], target

    def _synth_sample(self, sample):
        if 'json' in sample:
            if isinstance(sample['json'], bytes):
                sample['json'] = json.loads(sample['json'])
            if self.flatten_json:
                sample.update(sample.pop('json'))
        anno = _get_value(self.text_input_key, sample)
        assert anno, f"No annotation found with keys ({self.text_input_key})."
        pages = [p['lines'] if 'lines' in p else p for p in anno]
        valid = [i for i, p in enumerate(pages) if p.get('text')]
        if not valid:
            raise RuntimeError("No text on any page, skipping sample...")
        page_index = self.generator.choice(valid)
        page_info = None
        if self.patch_label_cfg is not None:
            image, lines, boxes = self.synth(pages[page_index]['text'], rng=self.generator, return_boxes=True)
            b = np.array(boxes, dtype=np.float32).reshape(-1, 4) / np.array(image.size * 2, dtype=np.float32)
            polys = np.stack([b[:, [0, 1]], b[:, [2, 1]], b[:, [2, 3]], b[:, [0, 3]]], 1)
            page_info = [{'lines': lines, 'polys': polys}]
        else:
            image, lines = self.synth(pages[page_index]['text'], rng=self.generator)
        if self.image_input_feat.image_mode and image.mode != self.image_input_feat.image_mode:
            image = image.convert(self.image_input_feat.image_mode)
        images, extras = self._preprocess_image_pages([image], page_info)
        image = images[0]
        text_input, text_target = self._process_text(self.line_break.join(lines))

        output = {
            self.image_input_name: image,
            self.text_input_name: text_input,
            '_parse': {
                'num_pages': len(anno),
                'page_indices': [page_index],
                'page_text': [lines],
                'synthetic': True,
            },
        }
        if text_target is not None:
            output[self.text_target_name] = text_target
        if extras is not None:
            output.update(extras[0])
        if not self.squeeze_pages:
            for k in (self.image_input_name, self.text_input_name, self.text_target_name, *(extras or [{}])[0]):
                if k in output:
                    output[k] = [output[k]]
        return output

    def _process_anno_pages(self, anno):
        assert isinstance(anno, (list, tuple)), f"Annotation should be a list of pages"
        num_pages = len(anno)
        if not num_pages:
            raise RuntimeError("Empty annotation. Skipping...")

        # FIXME for initial behaviour we will randomly sample one of N pages
        # TODO determine if we want to train in multi-page mode, use another sampling strategy?
        page_indices = []
        try:
            if self.page_sampling == 'random':
                n_wanted_pages = min(1, num_pages)  # TODO increase for multi-page processing, rand start+end?
                current_index = self.generator.randrange(-1, num_pages - 1)
                for _ in range(n_wanted_pages):
                    current_index = get_next_valid_page_index(current_index, num_pages, anno)
                    page_indices.append(current_index)
            elif self.page_sampling == 'first':
                current_index = get_next_valid_page_index(-1, num_pages, anno)
                page_indices.append(current_index)
            elif self.page_sampling == 'all_valid':
                current_index = -1
                for _ in range(num_pages):
                    current_index = get_next_valid_page_index(current_index, num_pages, anno)
                    page_indices.append(current_index)
            elif self.page_sampling == 'all':
                page_indices = list(range(num_pages))
        except RuntimeError:
            pass

        if not page_indices:
            raise RuntimeError("No valid annotated pages. Skipping...")

        text_pages = []
        tokenized_text_pages = []
        target_pages = []
        image_info = []
        for current_index in page_indices:
            # FIXME currently encoding each page separately with own start/end tokens.
            #  For multi-age should consider encoding in one sequence w/ page-break tokens.
            anno_page = anno[current_index]
            if 'lines' in anno_page:
                # Two supported formats right now
                # {
                #     'pages': [
                #         {
                #             'text': [],  # these are lines
                #             'bbox': [],
                #         }
                #     ]
                # }
                #
                # OR
                #
                # {
                #     'pages': [
                #         {
                #             'lines': {
                #                 'text': [],
                #                 'bbox': [],
                #             },
                #             'words': {
                #                 'text': [],
                #                 'bbox': [],
                #             }
                #         }
                #     ]
                # }
                #
                #
                anno_page = anno_page['lines']

            # Currently page text is created by concatenating lines of text with a CR line break
            # Additions could involve:
            # * using different line-break tokens between lines
            # * using word-level bbox anno information to mask works and construct partial lines
            # * group lines into blocks (or use block annos) and treat blocks / paragraphs of text and
            if not anno_page["text"]:
                raise RuntimeError("No text on page, skipping sample...")

            text = self.line_break.join(anno_page["text"])

            # tokenize w/ and generate training target if enabled
            text_input, text_target = self._process_text(text)
            tokenized_text_pages.append(text_input)
            if text_target is not None:
                target_pages.append(text_target)

            text_pages.append(anno_page["text"])  # unencoded text added as lines
            if self.patch_label_cfg is not None:
                polys = line_polys(anno_page, self.patch_label_cfg.region_source)
                image_info.append({'lines': anno_page['text'], 'polys': polys})

        gt_parse = {
            'num_pages': num_pages,  # total # of pages in doc
            'page_indices': page_indices,  # page indices sampled
            'page_text': text_pages,  # text of sampled page indices pages[].lines[]
        }
        if image_info:
            gt_parse['image_info'] = image_info  # per page regions for the image preprocessing (popped)

        output = {
            self.text_input_name: tokenized_text_pages,
            '_parse': gt_parse,
        }
        if target_pages:
            output[self.text_target_name] = target_pages

        return output

    def _decode_anno(self, sample):
        anno = _get_value(self.text_input_key, sample)
        assert anno is not None, f"No annotation found with keys ({self.text_input_key})."

        try:
            page_anno = self._process_anno_pages(anno)
        except Exception as exn:
            _logger.error(f'Issue processing annotation for {sample["__url__"]}, {sample["__key__"]}.')
            #_logger.error(json.dumps(anno, indent=4))
            raise exn

        # extract info from the _parse
        info = page_anno.get('_parse', {})
        page_indices = info.get('page_indices', [0])  # the samples page indices
        num_anno_pages = info.get('num_pages', 1)

        return page_anno, page_indices, num_anno_pages

    def _expand_anno(self, anno, count: int):
        expanded_annos = []
        for i in range(count):
            sample = {}
            for k, v in anno.items():
                if k == '_parse':
                    gt_parse = {}
                    gt_parse['num_pages'] = v['num_pages']
                    gt_parse['page_indices'] = [v['page_indices'][i]]
                    gt_parse['page_text'] = [v['page_text'][i]]
                    sample[k] = gt_parse
                else:
                    sample[k] = v[i] if isinstance(v, (list, tuple)) else v
            expanded_annos.append(sample)
        return expanded_annos
