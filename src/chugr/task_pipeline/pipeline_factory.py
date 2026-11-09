from chugr.common import DataTaskCfg

from .pipeline_doc_read import build_task_pipeline_doc_read, create_processor_doc_read, DataTaskDocReadCfg
from .pipeline_doc_vqa import build_task_pipeline_doc_vqa, create_processor_doc_vqa, DataTaskDocVqaCfg
from .pipeline_image_text import build_task_pipeline_image_text, DataTaskImageTextCfg
# from .donut_gtparse_pipe import
from .pipeline_manual import build_task_pipeline_manual, DataTaskManualCfg

_cfg_to_create = {
    DataTaskDocReadCfg: build_task_pipeline_doc_read,
    DataTaskDocVqaCfg: build_task_pipeline_doc_vqa,
    DataTaskImageTextCfg: build_task_pipeline_image_text,
    DataTaskManualCfg: build_task_pipeline_manual,
}


def create_task_pipeline(cfg: DataTaskCfg):
    create_fn = _cfg_to_create[type(cfg)]
    return create_fn(cfg)


_cfg_to_processor = {
    DataTaskDocReadCfg: create_processor_doc_read,
    DataTaskDocVqaCfg: create_processor_doc_vqa,
}


def create_task_processor(cfg: DataTaskCfg, image_process_fn=None):
    """ Per-sample processor (decode + image / text processing) for tasks that support NaFlex bucketed batching. """
    if type(cfg) not in _cfg_to_processor:
        raise ValueError(f'NaFlex batching is not supported for task cfg {type(cfg).__name__}.')
    return _cfg_to_processor[type(cfg)](cfg, image_process_fn=image_process_fn)
