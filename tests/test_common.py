from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import torch

from chugr.common import DataArg, DataCfg, ImageInputCfg, Serializable, SourceSpec, collate, parse_args
from chugr.text import text_input_to_target


@dataclass
class _Opt:
    lr: float = 1e-3
    betas: Optional[Tuple[float, float]] = None


@dataclass
class _TaskA:
    opt: _Opt = field(default_factory=_Opt)
    name: str = 'a'


@dataclass
class _TaskB:
    opt: _Opt = field(default_factory=_Opt)
    depth: int = 2


@dataclass
class _Cfg(Serializable):
    data: DataArg = field(default_factory=lambda: DataArg(source=''))
    task: Serializable = None
    tags: Optional[List[str]] = None
    flag: bool = False


def test_parse_args_nested_and_subgroups():
    subgroups = {'task': {'a': _TaskA, 'b': _TaskB}}
    cfg = parse_args(
        _Cfg,
        argv=['--task', 'b', '--task.opt.lr', '0.5', '--task.opt.betas', '0.9', '0.95', '--task.depth', '7',
              '--data.source', 'foo.tar', '--data.num-samples', '100', '--flag', 'true', '--tags', 'x', 'y'],
        subgroups=subgroups,
    )
    assert isinstance(cfg.task, _TaskB)
    assert cfg.task.opt.lr == 0.5 and cfg.task.opt.betas == (0.9, 0.95) and cfg.task.depth == 7
    assert cfg.data.source == 'foo.tar' and cfg.data.num_samples == 100
    assert cfg.flag is True and cfg.tags == ['x', 'y']

    cfg = parse_args(_Cfg, argv=[], subgroups=subgroups, subgroup_defaults={'task': 'a'})
    assert isinstance(cfg.task, _TaskA) and cfg.task.opt.lr == 1e-3


def test_serializable_roundtrip(tmp_path):
    cfg = ImageInputCfg(size=(64, 48), mode='L', transform_type='doc_naflex', patch_size=16, max_seq_len=128)
    p = tmp_path / 'cfg.yaml'
    cfg.save(p)
    cfg2 = ImageInputCfg.load(p)
    assert cfg2 == cfg
    assert isinstance(cfg2.size, tuple)


def test_data_cfg_from_arg():
    arg = DataArg(source='a.tar::b.tar', split='train', num_samples='10::20', sampling_weight='1::2', batch_size=4)
    cfg = DataCfg.from_arg(arg)
    assert cfg.num_samples == 30
    assert isinstance(cfg.source, list) and len(cfg.source) == 2
    assert cfg.source[1].sampling_weight == 2.0 and cfg.source[0].split == 'train'
    cfg = DataCfg.from_arg(DataArg(source='ds', split='validation', num_samples=5))
    assert isinstance(cfg.source, SourceSpec) and cfg.source.split == 'validation' and cfg.num_samples == 5


def test_collate_naflex_padding():
    batch = [
        {'image_input': {'patches': torch.ones(5, 12), 'patch_coord': torch.zeros(5, 2, dtype=torch.long),
                         'patch_valid': torch.ones(5, dtype=torch.bool)}, 'text': torch.zeros(3), '_parse': 1},
        {'image_input': {'patches': torch.ones(11, 12), 'patch_coord': torch.zeros(11, 2, dtype=torch.long),
                         'patch_valid': torch.ones(11, dtype=torch.bool)}, 'text': torch.zeros(3), '_parse': 2},
    ]
    out = collate(batch)
    assert out['image_input']['patches'].shape == (2, 16, 12)  # padded to multiple of 8
    assert out['image_input']['patch_valid'].sum(1).tolist() == [5, 11]
    assert out['_parse'] == [1, 2]


def test_text_input_to_target_masks_prompt():
    ids = torch.tensor([10, 11, 12, 99, 13, 14, 2, 1, 1])  # 99 == prompt end, 1 == pad
    target = text_input_to_target(ids, prompt_end_token=99, pad_token_id=1)
    assert target.tolist() == [-100, -100, -100, -100, 13, 14, 2, -100, -100]
