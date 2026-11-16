import abc
import json
import logging
import random
from typing import Callable, Dict, List, Optional, Tuple

from chugr.common import FeatureInfo, ImageFeatureInfo
from chugr.wds import decode_image_pages, decode_pdf_pages


from .constants import (
    DEFAULT_DOC_FEAT,
)

_logger = logging.getLogger(__name__)


def get_next_valid_page_index(
        current_index: int,
        num_pages: int,
        page_annos: list,
        retries: int = 10,
        wanted: str = 'lines',
):
    """
    Get the index of the next valid page which contains text. If it doesn't find any non-empty page
    after 'retries' attempts, it raises a RuntimeError.

    Parameters:
    current_index (int): Current page index.
    num_pages (int): Total number of pages.
    page_annos (list): List of page annotations.
    retries (int): Number of maximum retries for a given document.

    Returns:
    int: The index of the next non-empty page.
    """
    for _ in range(retries):
        # Get the next index, wrap around to 0 if it exceeds num_pages (in case of random init)
        current_index = (current_index + 1) % num_pages
        anno_page = page_annos[current_index]
        anno_page = anno_page.get(wanted, anno_page)  # use 'lines' / 'words' level if exists
        if anno_page["text"]:
            return current_index
    raise RuntimeError(f"No non-empty page found after {retries} attempts")


def _get_value(keys, sample):
    if isinstance(keys, (list, tuple)):
        value = None
        for k in keys:
            if (value := sample.get(k, None)) is not None:
                break
        return value
    else:
        return sample.get(keys, None)


class DocProcessor:
    """ Process documents w/ OCR annotation for reading tasks.
    """

    def __init__(
            self,
            image_process_fn: Optional[Callable] = None,
            text_process_fn: Optional[Callable] = None,
            image_input_feat: ImageFeatureInfo = DEFAULT_DOC_FEAT,
            text_input_feat: FeatureInfo = FeatureInfo('text_input', input_key='pages'),
            text_target_feat: FeatureInfo = FeatureInfo('text_target', input_key=None),
            page_sampling: str = 'random',
            render_dpi: int = 150,
            squeeze_pages: bool = True,
            expand_pages: bool = False,
            flatten_json: bool = True,
            seed: int = 0,
    ):
        """

        Args:
            image_process_fn:
            text_process_fn:
            page_sampling:
            render_dpi:
            seed:
        """
        self.image_process_fn = image_process_fn
        self.text_process_fn = text_process_fn

        self.image_input_feat = image_input_feat
        self.image_input_name = image_input_feat.output_name
        self.image_input_key = image_input_feat.input_key.split(';')
        self.text_input_feat = text_input_feat
        self.text_input_name = text_input_feat.output_name
        self.text_input_key = text_input_feat.input_key.split(';') if text_input_feat.input_key else None
        self.text_target_feat = text_target_feat
        self.text_target_name = text_target_feat.output_name if text_target_feat is not None else None

        self.page_sampling = page_sampling
        self.render_dpi = render_dpi
        self.squeeze_pages = squeeze_pages
        self.expand_pages = expand_pages
        self.flatten_json = flatten_json
        self.seed = seed
        self._generator = random.Random(seed)
        self._generator_worker = None

    @property
    def generator(self) -> random.Random:
        # (re)seed per dataloader worker so workers don't produce identical random sequences
        from torch.utils.data import get_worker_info
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else -1
        if worker_id != self._generator_worker:
            if worker_info is not None:
                self._generator = random.Random(worker_info.seed + self.seed)
            self._generator_worker = worker_id
        return self._generator

    def _preprocess_image_pages(self, decoded_pages, page_image_info=None):
        """ Preprocess decoded page images, returns (pages, per page extra outputs or None).

        With per page image info (e.g. annotation regions for region aware transforms / labels), each page goes
        through `_preprocess_page` which may return extra outputs (dict) to merge into the sample.
        """
        if self.image_process_fn is None:
            return decoded_pages, None
        if page_image_info is None:
            return [self.image_process_fn(dp) for dp in decoded_pages], None
        assert len(page_image_info) == len(decoded_pages)
        outputs = [self._preprocess_page(dp, pi) for dp, pi in zip(decoded_pages, page_image_info)]
        return [o[0] for o in outputs], [o[1] for o in outputs]

    def _preprocess_page(self, page, page_info):
        """ Preprocess one page image w/ its page info, returns (page, extra outputs dict). """
        return self.image_process_fn(page), {}

    def _decode_image_pages(
            self,
            sample,
            ext,
            page_indices,
            num_anno_pages,
    ):
        image_mode = self.image_input_feat.image_mode

        decoded_pages, num_image_pages = decode_image_pages(
            sample[ext],
            image_mode=image_mode,
            page_indices=page_indices,
        )
        if num_image_pages != num_anno_pages:
            _logger.warning(
                f'Mismatch between num image and num annotation pages {num_image_pages} != {num_anno_pages}'
                f' for sample {sample["__url__"]}, {sample["__key__"]}.')

        return decoded_pages, num_image_pages

    def _decode_pdf_pages(
            self,
            sample,
            ext,
            page_indices,
            num_anno_pages,
    ):
        image_mode = self.image_input_feat.image_mode
        decoded_pages, num_image_pages = decode_pdf_pages(
            sample[ext],
            image_mode=image_mode,
            page_indices=page_indices,
            render_dpi=self.render_dpi,
        )
        if num_anno_pages is not None and num_image_pages != num_anno_pages:
            _logger.warning(
                f'Mismatch between num image and num annotation pages {num_image_pages} != {num_anno_pages}'
                f' for sample {sample["__url__"]}, {sample["__key__"]}.')

        return decoded_pages, num_image_pages

    @abc.abstractmethod
    def _decode_anno(self, sample) -> Tuple[Dict, List[int], int]:
        pass

    def _expand_anno(self, anno, count: int):
        expanded_annos = [
            {k: v[i] if isinstance(v, (list, tuple)) else v for k, v in anno.items()}
            for i in range(count)
        ]
        return expanded_annos

    def _squeeze_anno(self, anno):
        anno = {k: v[0] if isinstance(v, (list, tuple)) else v for k, v in anno.items()}
        return anno

    def __call__(self, sample):
        if 'json' in sample and isinstance(sample['json'], bytes):
            # decode json if present and in undecoded state
            sample['json'] = json.loads(sample['json'])

        if self.flatten_json and 'json' in sample:
            # flatten json into sample
            sample.update(sample.pop('json'))

        # FIXME separate decode & preprocess interfaces

        # decode page annotations / text
        page_anno, page_indices, num_anno_pages = self._decode_anno(sample)
        # per page info for the image preprocessing (e.g. regions), not part of the output
        page_image_info = page_anno.get('_parse', {}).pop('image_info', None)

        # decode page images
        page_images = []
        for ext in self.image_input_key:
            if ext in sample:
                if ext == 'pdf':
                    images, num_image_pages = self._decode_pdf_pages(
                        sample,
                        ext,
                        page_indices,
                        num_anno_pages,
                    )
                else:
                    images, num_image_pages = self._decode_image_pages(
                        sample,
                        ext,
                        page_indices,
                        num_anno_pages,
                    )
                page_images.extend(images)
                # process one document type per doc, should be ordered by priority
                break

        assert len(page_images), 'No page images present'
        page_images, page_extras = self._preprocess_image_pages(page_images, page_image_info)
        if page_extras is not None and page_extras[0]:
            # extra per page outputs (e.g. patch labels) are added to the annotation as per page lists
            page_anno.update({k: [e[k] for e in page_extras] for k in page_extras[0]})

        if self.expand_pages and len(page_images) > 1:
            # expand pages and page annotations into multiple samples (return list of sample dicts)
            page_anno = self._expand_anno(page_anno, len(page_images))
            decoded = [{self.image_input_name: pi, **pa} for pi, pa in zip(page_images, page_anno)]
        else:
            if self.squeeze_pages and len(page_images) == 1:
                # squeeze page & annotation lists into singular items
                page_images = page_images[0]
                page_anno = self._squeeze_anno(page_anno)
            decoded = {self.image_input_name: page_images, **page_anno}

        return decoded


