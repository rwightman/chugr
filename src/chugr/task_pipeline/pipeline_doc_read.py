from dataclasses import dataclass, field
from typing import Callable, Optional

import webdataset as wds

from chugr.common import DataTaskCfg, FeatureInfo, ImageFeatureInfo
from chugr.doc import DocReadProcessor, DEFAULT_DOC_FEAT
from chugr.doc.synth_text import SynthTextCfg
from chugr.wds import get_error_handler


@dataclass
class DataTaskDocReadCfg(DataTaskCfg):
    image_input_feat: ImageFeatureInfo = DEFAULT_DOC_FEAT
    text_input_feat: FeatureInfo = FeatureInfo('text_input', input_key='pages')
    text_target_feat: Optional[FeatureInfo] = FeatureInfo('text_target', input_key=None)
    page_sampling: str = 'random'
    render_dpi: int = 150
    synth_prob: float = 0.  # probability of a synthetic text rendering (of the sample's text) replacing the page
    synth_cfg: Optional[SynthTextCfg] = None


def create_processor_doc_read(cfg: DataTaskDocReadCfg, image_process_fn: Optional[Callable] = None):
    return DocReadProcessor(
        image_process_fn=image_process_fn if image_process_fn is not None else cfg.image_process_fn,
        text_process_fn=cfg.text_process_fn,
        image_input_feat=cfg.image_input_feat,
        text_input_feat=cfg.text_input_feat,
        text_target_feat=cfg.text_target_feat,
        page_sampling=cfg.page_sampling,
        render_dpi=cfg.render_dpi,
        flatten_json=cfg.flatten_json,
        synth_prob=cfg.synth_prob,
        synth_cfg=cfg.synth_cfg,
    )


def build_task_pipeline_doc_read(
        cfg: DataTaskDocReadCfg,
):
    handler = get_error_handler(cfg.error_handler)
    pipe = []

    # document decoding & pre-processing done together, there is coupling in random page
    # selection and in the future, masking of image and/or text
    pipe += [wds.map(create_processor_doc_read(cfg), handler=handler)]

    if cfg.output_tuple:
        # NOTE in this mode we lose '_parse' key and would need to derive from target
        # Unless we add support for parse as the last tuple element?
        if cfg.text_target_feat is not None:
            pipe += [
                wds.to_tuple(
                    cfg.image_input_feat.output_name,
                    cfg.text_input_feat.output_name,
                    cfg.text_target_feat.output_name,
                )
            ]
        else:
            pipe += [
                wds.to_tuple(
                    cfg.image_input_feat.output_name,
                    cfg.text_input_feat.output_name,
                )
            ]
    return pipe
