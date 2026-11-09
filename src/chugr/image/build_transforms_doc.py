import random
from typing import Optional

import torchvision.transforms.functional as F
from torchvision import transforms

from chugr.common import ImageInputCfg, ImageAugCfg

from .transforms_torch import AlignLongAxis, Bitmap, Erosion, Dilation, CropMargin

# NOTE, chugr currently depends on some time aug impl, this should be flipped if timm ends up
# leveraging chugr data pipelines.
from timm.data import str_to_interp_mode, ResizeKeepRatio, CenterCropOrPad, RandomCropOrPad
from timm.data.naflex_transforms import Patchify, get_image_size_for_seq


_DOC_AUG_DEFAULTS = dict(
    scale_prob=0.05,
    scale_range=(0.85, 1.04),
    ratio_prob=0.05,
    ratio_range=(.9, 1.11),
    bitmap_prob=0.05,
    erosion_dilation_prob=0.02,
    shear_prob=0.05,
    shear_range_x=(0, 3.),
    shear_range_y=(-3, 0),
    shift_scale_rotate_prob=0.03,
    shift_range_x=0.04,
    shift_range_y=0.03,
    rotate_range=3,
    elastic_prob=0.04,
    elastic_alpha=50.,
    elastic_sigma=12.,
    brightness_contrast_prob=0.04,
    brightness_range=0.1,
    contrast_range=0.1,
    gaussian_blur_prob=0.03,
    gaussian_blur_kernel=3,
)


def _doc_aug_ops(params, interpolation_mode, fill):
    """ Torchvision based document augmentations applied to an image already at (roughly) target size. """
    return [
        transforms.RandomApply([Bitmap()], p=params['bitmap_prob']),
        transforms.RandomApply([
            transforms.RandomChoice([
                Erosion(3),
                Dilation(3),
            ])],
            p=params['erosion_dilation_prob']
        ),
        transforms.RandomApply([
            transforms.RandomAffine(
                degrees=0,
                shear=params['shear_range_x'] + params['shear_range_y'],
                interpolation=interpolation_mode,
                fill=fill,
            )],
            p=params['shear_prob'],
        ),
        transforms.RandomApply([
            transforms.RandomAffine(
                degrees=params['rotate_range'],
                translate=(params['shift_range_x'], params['shift_range_y']),
                interpolation=interpolation_mode,
                fill=fill,
            )],
            p=params['shift_scale_rotate_prob'],
        ),
        transforms.RandomApply([
            transforms.ElasticTransform(
                alpha=params['elastic_alpha'],
                sigma=params['elastic_sigma'],
                interpolation=interpolation_mode,
                fill=fill,
            )],
            p=params['elastic_prob'],
        ),
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=params['brightness_range'],
                contrast=params['contrast_range'],
            )],
            p=params['brightness_contrast_prob'],
        ),
        transforms.RandomApply([
            transforms.GaussianBlur(
                params['gaussian_blur_kernel'],
                sigma=(0.1, 0.8),
            )],
            p=params['gaussian_blur_prob'],
        ),
    ]


def build_transforms_doc_basic(
        input_cfg: ImageInputCfg,
        is_training: bool = False,
        do_normalize: bool = True,
        aug_cfg: Optional[ImageAugCfg] = None,
        composed: bool = True,
):
    # minimal fixed size doc transforms, resize keeping aspect + pad (random pad position for train)
    image_size = input_cfg.size
    interpolation_mode = str_to_interp_mode(input_cfg.interpolation)

    pp = []

    if input_cfg.crop_margin:
        pp += [CropMargin()]

    if input_cfg.align_long_axis:
        pp += [AlignLongAxis(image_size, interpolation=interpolation_mode)]

    pp += [ResizeKeepRatio(image_size, longest=1, interpolation=input_cfg.interpolation)]
    if is_training:
        pp += [RandomCropOrPad(image_size, fill=input_cfg.fill_color)]
    else:
        pp += [CenterCropOrPad(image_size, fill=input_cfg.fill_color)]

    pp += [transforms.ToTensor()]

    if do_normalize:
        pp += [transforms.Normalize(input_cfg.mean, input_cfg.std)]

    return transforms.Compose(pp) if composed else pp


def build_transforms_doc_better(
        input_cfg: ImageInputCfg,
        is_training: bool = False,
        do_normalize: bool = True,
        aug_cfg: Optional[ImageAugCfg] = None,
        composed: bool = True,
):
    # an improved torchvision + custom op transforms (no albumentations)
    image_size = input_cfg.size
    interpolation_mode = str_to_interp_mode(input_cfg.interpolation)
    pp = []

    if input_cfg.crop_margin:
        pp += [CropMargin()]

    if input_cfg.align_long_axis:
        pp += [AlignLongAxis(image_size, interpolation=interpolation_mode)]

    if is_training:
        # FIXME merge defaults w/ aug_cfg
        params = dict(_DOC_AUG_DEFAULTS)
        pp += [
            ResizeKeepRatio(
                image_size,
                longest=1,
                interpolation=input_cfg.interpolation,
                random_scale_prob=params['scale_prob'],
                random_scale_range=params['scale_range'],
                random_aspect_prob=params['ratio_prob'],
                random_aspect_range=params['ratio_range'],
            ),
        ]
        pp += _doc_aug_ops(params, interpolation_mode, input_cfg.fill_color)
        pp += [
            RandomCropOrPad(image_size, fill=input_cfg.fill_color),
            transforms.CenterCrop(image_size),
        ]
    else:
        pp += [
            ResizeKeepRatio(image_size, longest=1, interpolation=input_cfg.interpolation),
            CenterCropOrPad(image_size, fill=input_cfg.fill_color),
        ]

    pp += [transforms.ToTensor()]

    if do_normalize:
        pp += [transforms.Normalize(input_cfg.mean, input_cfg.std)]

    return transforms.Compose(pp) if composed else pp


class ResizeToSeqLen:
    """ Resize (preserving aspect) to the largest size whose patch count fits within a sequence length budget.

    Optionally (train) sample a reduced budget to vary the resolution.
    """

    def __init__(
            self,
            patch_size: int,
            max_seq_len: int,
            interpolation: str = 'bicubic',
            random_seq_len_prob: float = 0.,
            random_seq_len_range=(0.5, 1.0),
            max_ratio: Optional[float] = None,
    ):
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len
        self.interpolation = str_to_interp_mode(interpolation)
        self.random_seq_len_prob = random_seq_len_prob
        self.random_seq_len_range = random_seq_len_range
        self.max_ratio = max_ratio

    def __call__(self, img):
        w, h = F.get_image_size(img)
        seq_len = self.max_seq_len
        if self.random_seq_len_prob > 0 and random.random() < self.random_seq_len_prob:
            seq_len = max(1, int(seq_len * random.uniform(*self.random_seq_len_range)))
        _, (th, tw) = get_image_size_for_seq(
            (h, w),
            patch_size=self.patch_size,
            max_seq_len=seq_len,
            divisible_by_patch=True,
            max_ratio=self.max_ratio,
        )
        return F.resize(img, [th, tw], interpolation=self.interpolation, antialias=True)

    def __repr__(self):
        return (f'{self.__class__.__name__}(patch_size={self.patch_size}, max_seq_len={self.max_seq_len}, '
                f'random_seq_len_prob={self.random_seq_len_prob})')


def build_transforms_doc_naflex(
        input_cfg: ImageInputCfg,
        is_training: bool = False,
        do_normalize: bool = True,
        aug_cfg: Optional[ImageAugCfg] = None,
        composed: bool = True,
):
    """ Native aspect, variable resolution (NaFlex) document transforms.

    The page is resized (aspect preserved) to fit a patch budget (input_cfg.max_seq_len), augmented, then
    split into a flat sequence of patches w/ (y, x) patch coordinates. Output is a dict that the chugr
    collate fn pads to the longest sequence in the batch.
    """
    assert input_cfg.patch_size, 'patch_size must be set in ImageInputCfg for naflex transforms.'
    assert input_cfg.max_seq_len, 'max_seq_len must be set in ImageInputCfg for naflex transforms.'
    interpolation_mode = str_to_interp_mode(input_cfg.interpolation)
    pp = []

    if input_cfg.crop_margin:
        pp += [CropMargin()]

    if is_training:
        params = dict(_DOC_AUG_DEFAULTS)
        params.update(random_seq_len_prob=0.0, random_seq_len_range=(0.6, 1.0))
        pp += [
            ResizeToSeqLen(
                input_cfg.patch_size,
                input_cfg.max_seq_len,
                interpolation=input_cfg.interpolation,
                random_seq_len_prob=params['random_seq_len_prob'],
                random_seq_len_range=params['random_seq_len_range'],
            )
        ]
        pp += _doc_aug_ops(params, interpolation_mode, input_cfg.fill_color)
    else:
        pp += [
            ResizeToSeqLen(
                input_cfg.patch_size,
                input_cfg.max_seq_len,
                interpolation=input_cfg.interpolation,
            )
        ]

    pp += [transforms.ToTensor()]

    if do_normalize:
        pp += [transforms.Normalize(input_cfg.mean, input_cfg.std)]

    pp += [Patchify(input_cfg.patch_size)]

    return transforms.Compose(pp) if composed else pp


def build_transforms_doc_nougat(
        input_cfg: ImageInputCfg,
        is_training: bool = False,
        do_normalize: bool = True,
        aug_cfg: Optional[ImageAugCfg] = None,
        composed: bool = True,
):
    import albumentations as alb
    from chugr.image.transforms_alb import BitmapAlb, ErosionAlb, DilationAlb, AlbWrapper, CropMarginCv2

    # albumentations + custom opencv transforms from nougat
    image_size = input_cfg.size
    if input_cfg.interpolation == 'bilinear':
        interpolation_mode = 1
    else:
        interpolation_mode = 2  # bicubic
    border_mode = 0
    fill = input_cfg.fill_color
    if isinstance(fill, (list, tuple)) and len(fill) == 1:
        fill = fill[0]

    tv_pp = []
    alb_pp = []

    if input_cfg.crop_margin:
        tv_pp += [CropMarginCv2()]

    if input_cfg.align_long_axis:
        tv_pp += [AlignLongAxis(image_size)]

    if is_training:
        # FIXME merge defaults w/ aug_cfg
        params = dict(
            scale_range=(0.85, 1.03),  # done as part of shift_scale_rotate
            bitmap_prob=0.05,
            erosion_dilation_prob=0.02,
            erosion_dilation_scale=(2, 3),
            shear_prob=0.03,
            shear_range_x=(0, 3.),
            shear_range_y=(-3, 0),
            shift_scale_rotate_prob=0.03,
            shift_range_x=(0, 0.04),
            shift_range_y=(0, 0.03),
            rotate_range=2.,
            grid_distort_prob=0.04,
            grid_distort_range=0.05,
            elastic_prob=0.04,
            elastic_alpha=50.,
            elastic_sigma=12.,
            brightness_contrast_prob=0.03,
            brightness_range=0.1,
            constrast_range=0.1,
            gaussian_noise_prob=0.08,
            gaussian_noise_std=(0., 0.0175),  # ~ variance 20 in [0, 255] pixel units
            gaussian_blur_prob=0.03,
            gaussian_blur_kernel_range=(3, 3),
            image_compression_prob=0.1,
        )

        tv_pp += [
            # this should be equivalent to initial resize & pad in Donut prepare_input()
            ResizeKeepRatio(image_size, longest=1, interpolation=input_cfg.interpolation),
            RandomCropOrPad(image_size, fill=input_cfg.fill_color),
        ]

        alb_pp += [
            BitmapAlb(p=params['bitmap_prob']),
            alb.OneOf([
                    ErosionAlb(params['erosion_dilation_scale']),
                    DilationAlb(params['erosion_dilation_scale'])
                ],
                p=params['erosion_dilation_prob']
            ),
            alb.Affine(
                shear={
                    "x": params['shear_range_x'],
                    "y": params['shear_range_y']
                },
                border_mode=border_mode,
                fill=fill,
                p=params['shear_prob']
            ),
            alb.Affine(
                translate_percent={
                    "x": params['shift_range_x'],
                    "y": params['shift_range_y'],
                },
                scale=params['scale_range'],
                rotate=(-params['rotate_range'], params['rotate_range']),
                interpolation=interpolation_mode,
                border_mode=border_mode,
                fill=fill,
                p=params['shift_scale_rotate_prob'],
            ),
            alb.GridDistortion(
                distort_limit=params['grid_distort_range'],
                interpolation=interpolation_mode,
                border_mode=border_mode,
                fill=fill,
                p=params['grid_distort_prob'],
            ),
            alb.Compose(
                [
                    alb.Affine(
                        translate_px=(0, 5),
                        border_mode=border_mode,
                        fill=fill,
                        p=1.0,
                    ),
                    alb.ElasticTransform(
                        alpha=params['elastic_alpha'],
                        sigma=params['elastic_sigma'],
                        border_mode=border_mode,
                        fill=fill,
                        p=1.0,
                    ),
                ],
                p=params['elastic_prob'],
            ),
            alb.RandomBrightnessContrast(
                brightness_limit=params['brightness_range'],
                contrast_limit=params['constrast_range'],
                p=params['brightness_contrast_prob'],
            ),
            alb.ImageCompression(
                quality_range=(95, 100),
                p=params['image_compression_prob'],
            ),
            alb.GaussNoise(
                std_range=params['gaussian_noise_std'],
                p=params['gaussian_noise_prob']
            ),
            alb.GaussianBlur(
                blur_limit=params['gaussian_blur_kernel_range'],
                p=params['gaussian_blur_prob'],
            ),
        ]
    else:
        # inference / eval
        tv_pp += [
            ResizeKeepRatio(image_size, longest=1, interpolation=input_cfg.interpolation),
            CenterCropOrPad(image_size, fill=input_cfg.fill_color),
        ]

    if alb_pp:
        # FIXME leave alb uncomposed too if composed=False?
        tv_pp += [AlbWrapper(alb.Compose(alb_pp))]

    tv_pp += [transforms.ToTensor()]
    if do_normalize:
        tv_pp += [transforms.Normalize(input_cfg.mean, input_cfg.std)]

    return transforms.Compose(tv_pp) if composed else tv_pp
