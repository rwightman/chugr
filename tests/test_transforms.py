import pytest
import torch
from PIL import Image

import chugr


def _page(w=850, h=1100):
    return Image.new('L', (w, h), color=255)


@pytest.mark.parametrize('is_training', [True, False])
def test_doc_naflex_respects_budget(is_training):
    cfg = chugr.ImageInputCfg(size=None, mode='L', transform_type='doc_naflex', patch_size=16, max_seq_len=1024)
    t = chugr.create_image_preprocessor(cfg, is_training=is_training)
    out = t(_page())
    assert set(out.keys()) == {'patches', 'patch_coord', 'patch_valid'}
    n = out['patches'].shape[0]
    assert n <= 1024 and n > 900  # aspect preserved, budget mostly used
    assert out['patches'].shape[1] == 16 * 16
    # coords cover a portrait grid
    hy, wx = out['patch_coord'].max(0).values.tolist()
    assert hy > wx


@pytest.mark.parametrize('transform_type', ['doc_basic', 'doc_better', 'doc_nougat'])
@pytest.mark.parametrize('is_training', [True, False])
def test_doc_fixed_size(transform_type, is_training):
    cfg = chugr.ImageInputCfg(size=(256, 192), mode='L', transform_type=transform_type)
    t = chugr.create_image_preprocessor(cfg, is_training=is_training)
    out = t(_page(1700, 2200))
    assert out.shape == (1, 256, 192)


def _aug_ops(t):
    return [op for op in t.transforms if type(op).__name__ == 'RandomApply']


@pytest.mark.parametrize('transform_type', ['doc_better', 'doc_naflex'])
def test_doc_aug_cfg(transform_type):
    kw = dict(patch_size=16, max_seq_len=1024, size=None) if transform_type == 'doc_naflex' else dict(size=(256, 192))
    cfg = chugr.ImageInputCfg(mode='L', transform_type=transform_type, **kw)
    t = chugr.create_image_preprocessor(cfg, is_training=True)
    assert len(_aug_ops(t)) == 7  # defaults, every op enabled

    # overrides as a DocAugCfg or a dict, zero probabilities drop the op
    aug = chugr.DocAugCfg(bitmap_prob=0.5, elastic_prob=0.)
    for aug_cfg in (aug, {'bitmap_prob': 0.5, 'elastic_prob': 0.}):
        ops = _aug_ops(chugr.create_image_preprocessor(cfg, is_training=True, aug_cfg=aug_cfg))
        assert len(ops) == 6 and ops[0].p == 0.5

    # prob_scale scales every op, 0 == no augmentation
    ops = _aug_ops(chugr.create_image_preprocessor(cfg, is_training=True, aug_cfg=chugr.DocAugCfg(prob_scale=2.)))
    assert ops[0].p == pytest.approx(0.1)
    t = chugr.create_image_preprocessor(cfg, is_training=True, aug_cfg=chugr.DocAugCfg(prob_scale=0.))
    assert not _aug_ops(t)
    assert t(_page()) is not None

    # the generic image aug cfg doesn't apply to the doc transforms
    with pytest.raises(AssertionError):
        chugr.create_image_preprocessor(cfg, is_training=True, aug_cfg=chugr.ImageAugCfg())

