import json
from typing import Optional, Callable

from chugr import ImageFeatureInfo, FeatureInfo
from chugr.doc import DocProcessor, DEFAULT_DOC_FEAT
from chugr.doc.doc_processor import get_next_valid_page_index, _get_value, _logger
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
    ):
        """
        Args:
            synth_prob: probability of replacing a sample's page image w/ a synthetic rendering of (part of) its
                annotation text (target == rendered text, no layout annotations used, page image not decoded)
            synth_cfg: synthetic text rendering config
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
        image, lines = self.synth(pages[page_index]['text'], rng=self.generator)
        if self.image_input_feat.image_mode and image.mode != self.image_input_feat.image_mode:
            image = image.convert(self.image_input_feat.image_mode)
        image = self._preprocess_image_pages([image])[0]
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
        if not self.squeeze_pages:
            for k in (self.image_input_name, self.text_input_name, self.text_target_name):
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

        gt_parse = {
            'num_pages': num_pages,  # total # of pages in doc
            'page_indices': page_indices,  # page indices sampled
            'page_text': text_pages,  # text of sampled page indices pages[].lines[]
        }

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

        # TODO support 'image info' to relay details such as text bbox, layout
        # page_image_info = info.get('image_info', None)
        # if page_image_info is not None:
        #     assert len(page_image_info) == len(page_indices)

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
