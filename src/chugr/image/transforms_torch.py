""" chugr image transforms (torchvision v2)

Geometric transforms draw their random parameters once per call (make_params) and apply them to every input
(images, masks, bounding boxes, keypoints) through the v2 functional kernels, so region annotations passed w/
the image (e.g. `t(img, polys)`) stay aligned with it. Pixel transforms only touch images, any other input passes
through unchanged. Called w/ a single image, every transform behaves like its torchvision v1 era counterpart.
"""
import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision import tv_tensors
from torchvision.transforms import InterpolationMode, v2
from torchvision.transforms.v2 import functional as F
from torchvision.transforms.v2._utils import is_pure_tensor

# NOTE, chugr currently depends on some timm aug / NaFlex impl, this should be flipped if timm ends up
# leveraging chugr data pipelines.
from timm.data import str_to_interp_mode
from timm.data.naflex_transforms import get_image_size_for_seq, patchify_image

_RANDOM_INTERPOLATION = (InterpolationMode.BILINEAR, InterpolationMode.BICUBIC)


def _is_image(inpt) -> bool:
    return isinstance(inpt, (Image.Image, tv_tensors.Image)) or is_pure_tensor(inpt)


def _query_image(flat_inputs: List[Any]):
    for inpt in flat_inputs:
        if _is_image(inpt):
            return inpt
    raise TypeError('No image found in the transform inputs.')


def _fill(inpt, fill):
    """ Fill for padded / exposed areas: the image fill for images, 0 (background) for masks, unused otherwise. """
    if isinstance(inpt, tv_tensors.Mask):
        return 0
    return fill if _is_image(inpt) else None


def _interp_modes(interpolation: Union[str, InterpolationMode]) -> Tuple[InterpolationMode, ...]:
    if interpolation == 'random':
        return _RANDOM_INTERPOLATION
    if isinstance(interpolation, str):
        return str_to_interp_mode(interpolation),
    return interpolation,


def _setup_size(size) -> Tuple[int, int]:
    if isinstance(size, (int, float)):
        return int(size), int(size)
    if len(size) == 1:
        return size[0], size[0]
    assert len(size) == 2, 'Size should be (h, w).'
    return tuple(size)


def _crop_or_pad(inpt, top: int, left: int, height: int, width: int, fill=0, padding_mode: str = 'constant'):
    """ Crop and / or pad to (height, width) w/ the window at (top, left), negative offsets pad. """
    image_height, image_width = F.get_size(inpt)
    right = left + width
    bottom = top + height
    if left < 0 or top < 0 or right > image_width or bottom > image_height:
        padding_ltrb = [
            max(-left + min(0, right), 0),
            max(-top + min(0, bottom), 0),
            max(right - max(image_width, left), 0),
            max(bottom - max(image_height, top), 0),
        ]
        inpt = F.pad(inpt, padding_ltrb, fill=_fill(inpt, fill), padding_mode=padding_mode)
    return F.crop(inpt, max(top, 0), max(left, 0), height, width)


class PixelTransform(v2.Transform):
    """ Base for transforms that change image pixel values only, other inputs pass through. """

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if not _is_image(inpt):
            return inpt
        return self.transform_image(inpt, params)

    def transform_image(self, img, params: Dict[str, Any]):
        raise NotImplementedError


class Bitmap(PixelTransform):
    """ Binarize-ish, pixels below the threshold go to black (PIL images). """

    def __init__(self, threshold: int = 200):
        super().__init__()
        self.threshold = threshold
        self.lut = [0 if i < threshold else i for i in range(256)]

    def transform_image(self, img, params):
        assert isinstance(img, Image.Image), 'Bitmap expects PIL images.'
        lut = self.lut * 3 if img.mode == 'RGB' else self.lut
        return img.point(lut)


class _MinMaxFilter(PixelTransform):
    minimum = True

    def __init__(self, scale: Union[int, Sequence[int]] = 3):
        super().__init__()
        if isinstance(scale, (tuple, list)):
            assert len(scale) == 2
            self.scale = tuple(scale)
        else:
            self.scale = (scale, scale)

    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        return dict(kernel_size=random.choice(self.scale))

    def transform_image(self, img, params):
        kernel_size = params['kernel_size']
        if isinstance(img, Image.Image):
            f = ImageFilter.MinFilter if self.minimum else ImageFilter.MaxFilter
            return img.filter(f(kernel_size))
        x = img.as_subclass(torch.Tensor)
        sign = -1 if self.minimum else 1
        x = sign * torch.nn.functional.max_pool2d(sign * x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        return tv_tensors.wrap(x, like=img) if isinstance(img, tv_tensors.Image) else x


class Erosion(_MinMaxFilter):
    """ Grow dark (ink) regions, min filter w/ a randomly chosen kernel size from scale. """
    minimum = True


class Dilation(_MinMaxFilter):
    """ Shrink dark (ink) regions, max filter w/ a randomly chosen kernel size from scale. """
    minimum = False


class ConvertColor(PixelTransform):
    def __init__(self, mode: str = 'RGB'):
        super().__init__()
        self.mode = mode

    def transform_image(self, img, params):
        assert isinstance(img, Image.Image), 'ConvertColor expects PIL images.'
        return img.convert(self.mode)


def python_find_non_zero(image: np.array):
    """This is a reimplementation of a findNonZero function equivalent to cv2."""
    non_zero_indices = np.column_stack(np.nonzero(image))
    idxvec = non_zero_indices[:, [1, 0]]
    idxvec = idxvec.reshape(-1, 1, 2)
    return idxvec


def python_bounding_rect(coordinates):
    """This is a reimplementation of a BoundingRect function equivalent to cv2."""
    min_values = np.min(coordinates, axis=(0, 1)).astype(int)
    max_values = np.max(coordinates, axis=(0, 1)).astype(int)
    x_min, y_min = min_values[0], min_values[1]
    width = max_values[0] - x_min + 1
    height = max_values[1] - y_min + 1
    return x_min, y_min, width, height


class CropMargin(v2.Transform):
    """ Crop to the bounding rect of the (contrast normalized) content darker than gray_threshold. """

    def __init__(self, gray_threshold: int = 200):
        super().__init__()
        self.gray_threshold = gray_threshold

    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        image = _query_image(flat_inputs)
        # FIXME check tensor vs PIL and convert as needed, this is assuming PIL right now
        assert isinstance(image, Image.Image), 'CropMargin expects PIL images.'
        data = np.array(image.convert('L')).astype(np.uint8)
        max_val = data.max()
        min_val = data.min()
        if max_val == min_val:
            return dict(crop=None)
        data = (data - min_val) / (max_val - min_val) * 255
        gray = data < self.gray_threshold
        coords = python_find_non_zero(gray)
        x_min, y_min, width, height = python_bounding_rect(coords)
        return dict(crop=(int(y_min), int(x_min), int(height), int(width)))

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if params['crop'] is None:
            return inpt
        return F.crop(inpt, *params['crop'])


class AlignLongAxis(v2.Transform):
    """ Rotate (-90) so the long image axis matches the long axis of the target (h, w) input_size. """

    def __init__(self, input_size, interpolation=InterpolationMode.BICUBIC):
        super().__init__()
        self.input_size = input_size
        self.interpolation = interpolation

    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        img_height, img_width = F.get_size(_query_image(flat_inputs))
        rotate = (
            (self.input_size[0] > self.input_size[1] and img_width > img_height) or
            (self.input_size[0] < self.input_size[1] and img_width < img_height)
        )
        return dict(rotate=rotate)

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if not params['rotate']:
            return inpt
        return F.rotate(inpt, angle=-90, expand=True, interpolation=self.interpolation)


class NormalizeOrientation(v2.Transform):
    """ Rotate by a multiple of 90 degrees so text lines read left to right, upright.

    The orientation comes from the region polys passed w/ the image (tv_tensors.KeyPoints [M, 4, 2], 4 point line /
    token polygons w/ corners in reading order: top left, top right, bottom right, bottom left, as in the IDL / PDFA
    OCR 'poly' annotations), the length weighted circular mean of the line direction rounded to the nearest 90
    degrees. Without polys the transform does nothing.
    """

    def __init__(self, interpolation=InterpolationMode.BICUBIC):
        super().__init__()
        self.interpolation = interpolation

    @staticmethod
    def poly_angle(polys: torch.Tensor) -> Optional[float]:
        """ Line direction (degrees, image coords w/ y down, 0 = left to right) of [M, 4, 2] polys. """
        polys = polys.reshape(-1, 4, 2).float()
        if not len(polys):
            return None
        e = (polys[:, 1] - polys[:, 0]) + (polys[:, 2] - polys[:, 3])  # top + bottom edges, length weighted
        theta = torch.atan2(e[:, 1], e[:, 0])
        w = e.norm(dim=-1)
        if w.sum() <= 0:
            return None
        return math.degrees(math.atan2((w * theta.sin()).sum().item(), (w * theta.cos()).sum().item()))

    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        polys = [
            i.as_subclass(torch.Tensor) for i in flat_inputs
            if isinstance(i, tv_tensors.KeyPoints) and i.ndim >= 2 and i.shape[-2] == 4
        ]
        angle = self.poly_angle(torch.cat([p.reshape(-1, 4, 2) for p in polys])) if polys else None
        # F.rotate angles are counter-clockwise (as displayed), a line pointing down (+90) needs a +90 rotation
        k = 0 if angle is None else round(angle / 90.) % 4
        return dict(angle=90. * k)

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if not params['angle']:
            return inpt
        return F.rotate(inpt, angle=params['angle'], expand=True, interpolation=self.interpolation)


class ResizeKeepRatio(v2.Transform):
    """ Resize keeping the aspect ratio (shortest, longest, or a mix of the two edges vs the target size).

    Optional random scale and aspect jitter (train). A torchvision v2 version of the timm transform.
    """

    def __init__(
            self,
            size,
            longest: float = 0.,
            interpolation: Union[str, InterpolationMode] = 'bilinear',
            random_scale_prob: float = 0.,
            random_scale_range: Tuple[float, float] = (0.85, 1.05),
            random_scale_area: bool = False,
            random_aspect_prob: float = 0.,
            random_aspect_range: Tuple[float, float] = (0.9, 1.11),
    ):
        super().__init__()
        self.size = _setup_size(size)
        self.interpolation = _interp_modes(interpolation)
        self.longest = float(longest)
        self.random_scale_prob = random_scale_prob
        self.random_scale_range = random_scale_range
        self.random_scale_area = random_scale_area
        self.random_aspect_prob = random_aspect_prob
        self.random_aspect_range = random_aspect_range

    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        img_h, img_w = img_size = F.get_size(_query_image(flat_inputs))
        target_h, target_w = self.size
        ratio_h = img_h / target_h
        ratio_w = img_w / target_w
        ratio = max(ratio_h, ratio_w) * self.longest + min(ratio_h, ratio_w) * (1. - self.longest)

        if self.random_scale_prob > 0 and random.random() < self.random_scale_prob:
            ratio_factor = random.uniform(*self.random_scale_range)
            if self.random_scale_area:
                # make ratio factor equivalent to RRC area crop where < 1.0 = area zoom,
                # otherwise like affine scale where < 1.0 = linear zoom out
                ratio_factor = 1. / math.sqrt(ratio_factor)
            ratio_factor = (ratio_factor, ratio_factor)
        else:
            ratio_factor = (1., 1.)

        if self.random_aspect_prob > 0 and random.random() < self.random_aspect_prob:
            log_aspect = (math.log(self.random_aspect_range[0]), math.log(self.random_aspect_range[1]))
            aspect_factor = math.sqrt(math.exp(random.uniform(*log_aspect)))
            ratio_factor = (ratio_factor[0] / aspect_factor, ratio_factor[1] * aspect_factor)

        size = [round(x * f / ratio) for x, f in zip(img_size, ratio_factor)]
        interpolation = random.choice(self.interpolation) if len(self.interpolation) > 1 else self.interpolation[0]
        return dict(size=size, interpolation=interpolation)

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return F.resize(inpt, params['size'], interpolation=params['interpolation'], antialias=True)


class CenterCropOrPad(v2.Transform):
    """ Center crop and / or pad to size (h, w). """

    def __init__(self, size, fill=0, padding_mode: str = 'constant'):
        super().__init__()
        self.size = _setup_size(size)
        self.fill = fill
        self.padding_mode = padding_mode

    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        image_height, image_width = F.get_size(_query_image(flat_inputs))
        crop_height, crop_width = self.size
        # pad (left / top get the smaller half), then center crop, matches timm center_crop_or_pad
        pad_left = (crop_width - image_width) // 2 if crop_width > image_width else 0
        pad_top = (crop_height - image_height) // 2 if crop_height > image_height else 0
        padded_h = max(image_height, crop_height)
        padded_w = max(image_width, crop_width)
        top = int(round((padded_h - crop_height) / 2.0)) - pad_top
        left = int(round((padded_w - crop_width) / 2.0)) - pad_left
        return dict(top=top, left=left)

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return _crop_or_pad(
            inpt, params['top'], params['left'], *self.size, fill=self.fill, padding_mode=self.padding_mode)


class RandomCropOrPad(v2.Transform):
    """ Crop and / or pad to size (h, w) w/ random placement within the crop or pad margin. """

    def __init__(self, size, fill=0, padding_mode: str = 'constant'):
        super().__init__()
        self.size = _setup_size(size)
        self.fill = fill
        self.padding_mode = padding_mode

    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        image_height, image_width = F.get_size(_query_image(flat_inputs))
        delta_height = image_height - self.size[0]
        delta_width = image_width - self.size[1]
        top = int(math.copysign(random.randint(0, abs(delta_height)), delta_height))
        left = int(math.copysign(random.randint(0, abs(delta_width)), delta_width))
        return dict(top=top, left=left)

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return _crop_or_pad(
            inpt, params['top'], params['left'], *self.size, fill=self.fill, padding_mode=self.padding_mode)


class RandomResizedCrop(v2.RandomResizedCrop):
    """ v2 RandomResizedCrop w/ the interpolation optionally sampled per call ('random' = bilinear or bicubic). """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation='bilinear'):
        modes = _interp_modes(interpolation)
        super().__init__(size, scale=scale, ratio=ratio, interpolation=modes[0], antialias=True)
        self.interpolations = modes

    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        modes = self.interpolations
        return dict(**super().make_params(flat_inputs), interpolation=random.choice(modes) if len(modes) > 1 else modes[0])

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        params = dict(params)
        interpolation = params.pop('interpolation')
        return F.resized_crop(inpt, **params, size=self.size, interpolation=interpolation, antialias=True)


class ResizeToSeqLen(v2.Transform):
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
        super().__init__()
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len
        self.interpolation = str_to_interp_mode(interpolation)
        self.random_seq_len_prob = random_seq_len_prob
        self.random_seq_len_range = random_seq_len_range
        self.max_ratio = max_ratio

    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        h, w = F.get_size(_query_image(flat_inputs))
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
        return dict(size=[th, tw])

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return F.resize(inpt, params['size'], interpolation=self.interpolation, antialias=True)

    def extra_repr(self) -> str:
        return (f'patch_size={self.patch_size}, max_seq_len={self.max_seq_len}, '
                f'random_seq_len_prob={self.random_seq_len_prob}')


class Patchify(v2.Transform):
    """ Image -> NaFlex patch dict (patches [N, P * P * C], patch_coord [N, 2] (y, x), patch_valid [N]).

    Other inputs (regions) pass through in image pixel coordinates, patch (y, x) covers pixels
    [y * P, (y + 1) * P) x [x * P, (x + 1) * P).
    """

    def __init__(self, patch_size: Union[int, Tuple[int, int]], flatten_patches: bool = True):
        super().__init__()
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        self.flatten_patches = flatten_patches

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if not _is_image(inpt):
            return inpt
        if isinstance(inpt, Image.Image):
            inpt = F.to_dtype(F.to_image(inpt), torch.float32, scale=True)
        patches, coord, valid = patchify_image(
            inpt.as_subclass(torch.Tensor),
            self.patch_size,
            flatten_patches=self.flatten_patches,
            channels_last=True,
        )
        return {'patches': patches, 'patch_coord': coord, 'patch_valid': valid}


class ToPureImage(PixelTransform):
    """ tv_tensors.Image -> plain tensor (end of the pipeline), other inputs keep their tv_tensor type. """

    def transform_image(self, img, params):
        return img.as_subclass(torch.Tensor) if isinstance(img, torch.Tensor) else img


def to_tensor_ops(mean=None, std=None) -> List[v2.Transform]:
    """ PIL -> float [0, 1] tensor (v1 ToTensor equivalent), optional normalize. """
    ops = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
    if mean is not None:
        ops += [v2.Normalize(mean, std)]
    return ops
