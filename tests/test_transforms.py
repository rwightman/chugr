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


# torchvision v2 region (poly / box) handling

import numpy as np
import PIL.ImageDraw
import PIL.ImageFilter
from torchvision import tv_tensors
from torchvision.transforms import InterpolationMode, v2
from torchvision.transforms.v2 import functional as TF

from chugr.image import AlignLongAxis, Bitmap, CenterCropOrPad, CropMargin, Dilation, Erosion, \
    NormalizeOrientation, RandomCropOrPad, ResizeKeepRatio, ResizeToSeqLen
from chugr.image.build_transforms_doc import _doc_aug_ops

_LINE = (200, 300, 600, 340)  # x0, y0, x1, y1 of a solid 'text line'


def _line_page(w=850, h=1100):
    img = _page(w, h)
    PIL.ImageDraw.Draw(img).rectangle((_LINE[0], _LINE[1], _LINE[2] - 1, _LINE[3] - 1), fill=0)
    x0, y0, x1, y1 = _LINE
    polys = tv_tensors.KeyPoints(
        torch.tensor([[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]], dtype=torch.float32), canvas_size=(h, w))
    boxes = tv_tensors.BoundingBoxes(torch.tensor([[x0, y0, x1, y1]], dtype=torch.float32), format='XYXY',
                                     canvas_size=(h, w))
    return img, polys, boxes


def _poly_ink_check(img, polys, margin=4):
    """ Fraction of the ink within `margin` px of the poly, ink area / poly area. """
    ink = np.asarray(img) < 128
    m = PIL.Image.new('L', img.size, 0)
    PIL.ImageDraw.Draw(m).polygon([tuple(c) for c in polys.reshape(-1, 2).tolist()], fill=255)
    area = (np.asarray(m) > 0).sum()
    near = np.asarray(m.filter(PIL.ImageFilter.MaxFilter(2 * margin + 1))) > 0
    return (ink & near).sum() / max(ink.sum(), 1), ink.sum() / max(area, 1)


def _geometric_ops():
    aug = chugr.DocAugCfg(prob_scale=100.)  # every op p=1
    ops = {type(o.transforms[0]).__name__ + str(i): o for i, o in enumerate(_doc_aug_ops(aug, InterpolationMode.BICUBIC, 255))}
    ops.update(
        resize_seq=ResizeToSeqLen(16, 1024, random_seq_len_prob=1.),
        resize_keep=ResizeKeepRatio((1024, 768), longest=1, random_scale_prob=1., random_aspect_prob=1.),
        random_crop_or_pad=RandomCropOrPad((1000, 1000), fill=255),
        center_crop_or_pad=CenterCropOrPad((900, 1200), fill=255),
        crop_margin=CropMargin(),
        align_long_axis=AlignLongAxis((768, 1024)),
    )
    return ops


@pytest.mark.parametrize('name', list(_geometric_ops().keys()))
def test_regions_follow_ink(name):
    op = _geometric_ops()[name]
    img, polys, boxes = _line_page()
    for seed in range(3):
        torch.manual_seed(seed)
        out, p2, b2 = op(img, polys, boxes)
        assert isinstance(p2, tv_tensors.KeyPoints) and isinstance(b2, tv_tensors.BoundingBoxes)
        assert p2.canvas_size == tuple(TF.get_size(out)) == b2.canvas_size
        if isinstance(op, v2.RandomApply) and isinstance(op.transforms[0], (Bitmap, Erosion, Dilation)):
            assert torch.equal(p2, polys) and torch.equal(b2, boxes)  # pixel ops leave regions alone
            continue
        # elastic bends the ink inside the poly (corners only), so allow a few px
        within, area_ratio = _poly_ink_check(out, p2)
        assert within > 0.97 and 0.85 < area_ratio < 1.15, f'{name} seed {seed}: {within:.3f} {area_ratio:.3f}'
        # box covers the poly
        x0, y0, x1, y1 = b2[0].tolist()
        q = p2.reshape(-1, 2)
        assert x0 <= q[:, 0].min() + 1 and y0 <= q[:, 1].min() + 1 and x1 >= q[:, 0].max() - 1 and y1 >= q[:, 1].max() - 1


@pytest.mark.parametrize('angle', [0, 90, 180, 270])
def test_normalize_orientation(angle):
    img, polys, _ = _line_page()
    rot_img = TF.rotate(img, angle, expand=True)
    rot_polys = TF.rotate(polys, angle, expand=True)  # corners keep their reading order
    out, p2 = NormalizeOrientation()(rot_img, rot_polys)
    assert out.size == img.size
    assert np.array_equal(np.asarray(out), np.asarray(img))
    assert torch.allclose(p2, polys, atol=1e-3)
    assert NormalizeOrientation()(rot_img) is rot_img  # no polys, no-op


@pytest.mark.parametrize('is_training', [True, False])
def test_doc_naflex_regions(is_training):
    cfg = chugr.ImageInputCfg(size=None, mode='L', transform_type='doc_naflex', patch_size=16, max_seq_len=1024)
    t = chugr.create_image_preprocessor(cfg, is_training=is_training)
    img, polys, _ = _line_page()
    torch.manual_seed(0)
    out, p2 = t(img, polys)
    torch.manual_seed(0)
    out_img_only = t(img)
    assert torch.equal(out['patches'], out_img_only['patches'])  # regions don't change the image path
    gh, gw = (out['patch_coord'].max(0).values + 1).tolist()
    assert p2.canvas_size == (gh * 16, gw * 16)
    if not is_training:
        scale = torch.tensor([gw * 16 / 850, gh * 16 / 1100])
        assert torch.allclose(p2, polys * scale, atol=0.01)


def test_image_basic_modes():
    for kw in (dict(resize_mode='longest'), dict(resize_mode='squash'), dict(resize_mode='shortest')):
        cfg = chugr.ImageInputCfg(size=(224, 224), mode='RGB', transform_type='image_basic', **kw)
        assert chugr.create_image_preprocessor(cfg, is_training=False)(_page().convert('RGB')).shape == (3, 224, 224)
    cfg = chugr.ImageInputCfg(size=(224, 224), mode='RGB', transform_type='image_basic', interpolation='random')
    aug = chugr.ImageAugCfg(color_jitter=(.4, .4, .4, .1), color_jitter_prob=.8, grayscale_prob=.2, gaussian_blur_prob=.5)
    t = chugr.create_image_preprocessor(cfg, is_training=True, aug_cfg=aug)
    names = [type(o).__name__ for o in t.transforms]
    assert names.count('RandomApply') == 2 and 'RandomGrayscale' in names
    assert t(_page().convert('RGB')).shape == (3, 224, 224)
