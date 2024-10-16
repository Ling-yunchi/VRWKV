import random
from typing import Tuple

import torchvision.transforms.functional as F
from torch import Tensor
from torchvision.transforms import (
    RandomCrop,
    RandomRotation,
    ColorJitter,
    RandomResizedCrop,
)


def resize(img, mask, target: Tuple[int, int]):
    img = F.resize(img, target)
    mask = F.resize(mask, target, interpolation=F.InterpolationMode.NEAREST)
    return img, mask


def random_resize(img, mask, scale: Tuple[int, int], ratio_range: Tuple[float, float]):
    ratio = random.uniform(*ratio_range)
    new_scale = [int(scale[0] * ratio), int(scale[1] * ratio)]
    img = F.resize(img, new_scale)
    mask = F.resize(mask, new_scale, interpolation=F.InterpolationMode.NEAREST)
    return img, mask


def random_resized_crop(
    img,
    mask,
    scale: Tuple[float, float],
    ratio: Tuple[float, float],
    crop_size: Tuple[int, int],
):
    i, j, h, w = RandomResizedCrop.get_params(img, scale, ratio)
    img = F.resized_crop(img, i, j, h, w, crop_size, F.InterpolationMode.BILINEAR)
    mask = F.resized_crop(mask, i, j, h, w, crop_size, F.InterpolationMode.NEAREST)
    return img, mask


def random_crop(img, mask, crop_size: Tuple[int, int]):
    i, j, h, w = RandomCrop.get_params(img, crop_size)
    img = F.crop(img, i, j, h, w)
    mask = F.crop(mask, i, j, h, w)
    return img, mask


def random_hflip(img, mask, prob=0.5):
    if random.random() < prob:
        img = F.hflip(img)
        mask = F.hflip(mask)
    return img, mask


def random_vflip(img, mask, prob=0.5):
    if random.random() < prob:
        img = F.vflip(img)
        mask = F.vflip(mask)
    return img, mask


def random_rotate(img, mask, degrees: list[float], fill=0, mask_fill=255):
    angle = RandomRotation.get_params(degrees)
    img = F.rotate(img, angle, fill=[fill])
    mask = F.rotate(mask, angle, fill=[mask_fill])
    return img, mask


def normalize(img: Tensor, mean: list[float], std: list[float]):
    img = F.normalize(img, mean=mean, std=std)
    return img


def pad(img, mask, crop_size, pad_val=0, seg_pad_val=255):
    img = F.pad(
        img,
        padding=[0, 0, crop_size[1] - img.shape[2], crop_size[0] - img.shape[1]],
        fill=pad_val,
    )
    mask = F.pad(
        mask,
        padding=[0, 0, crop_size[1] - mask.shape[2], crop_size[0] - mask.shape[1]],
        fill=seg_pad_val,
    )
    return img, mask


def color_jitter(img, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
    cj = ColorJitter(
        brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
    )
    img = cj(img)
    return img
