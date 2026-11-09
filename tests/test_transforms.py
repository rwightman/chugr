import pytest
import torch
from PIL import Image

import chug


def _page(w=850, h=1100):
    return Image.new('L', (w, h), color=255)


@pytest.mark.parametrize('is_training', [True, False])
def test_doc_naflex_respects_budget(is_training):
    cfg = chug.ImageInputCfg(size=None, mode='L', transform_type='doc_naflex', patch_size=16, max_seq_len=1024)
    t = chug.create_image_preprocessor(cfg, is_training=is_training)
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
    cfg = chug.ImageInputCfg(size=(256, 192), mode='L', transform_type=transform_type)
    t = chug.create_image_preprocessor(cfg, is_training=is_training)
    out = t(_page(1700, 2200))
    assert out.shape == (1, 256, 192)
