"""Data augmentation. Special care has to be taken with A-Train data, as the interpolation corners and weights also
require transformation during augmentation.
"""

import random
import sys
from typing import Callable

import torch
import torchvision.transforms.functional as TF

if "./src" not in sys.path:
    sys.path.insert(0, "./src")  # TO DO: change this once it's a package
from datasets.normalization import ATRAIN_MEANS, ATRAIN_STDS


def get_norm_transform(img_channel_idx: list, multi_angle_idx: list) -> Callable:
    """Normalization transform. Normalizes sensor input by the A-Train means and standard deviations.

    Args:
        img_channel_idx: Index to the image channels in the multi angle index.
        multi_angle_idx: The multi angle index into the PARASOL fields.

    Returns:
        norm_transform: A function that normalizes a batch.
    """
    multi_angle_img_idx = multi_angle_idx[img_channel_idx]
    means = ATRAIN_MEANS[multi_angle_img_idx]
    stds = ATRAIN_STDS[multi_angle_img_idx]

    def norm_transform(batch: dict) -> dict:
        batch["input"]["sensor_input"][:, img_channel_idx] = TF.normalize(
            batch["input"]["sensor_input"][:, img_channel_idx], means, stds
        )
        return batch

    return norm_transform


def random_hflip(batch: dict) -> dict:
    """Random horizontal flip transform. Also flips the interpolation corners.

    Args:
        batch: A batch to transform.

    Returns:
        batch: The transformed batch.
    """
    if random.random() > 0.5:
        batch["input"]["sensor_input"] = TF.hflip(batch["input"]["sensor_input"])
        w = batch["input"]["sensor_input"].shape[-1]
        corners_x = batch["input"]["interp"]["corners"][..., 0]
        corners_x = w - corners_x - 1
        batch["input"]["interp"]["corners"][..., 0] = corners_x
    return batch


def random_vflip(batch: dict) -> dict:
    """Random vertical flip transform. Also flips the interpolation corners.

    Args:
        batch: A batch to transform.

    Returns:
        batch: The transformed batch.
    """
    if random.random() > 0.5:
        batch["input"]["sensor_input"] = TF.vflip(batch["input"]["sensor_input"])
        h = batch["input"]["sensor_input"].shape[-2]
        corners_y = batch["input"]["interp"]["corners"][..., 1]
        corners_y = h - corners_y - 1
        batch["input"]["interp"]["corners"][..., 1] = corners_y
    return batch


def random_rotate_90(batch: dict) -> dict:
    """Randomly rotate the image by 90 degrees. Also rotate the interpolation corners. For rotations in
    [0, 90, 180, 270], combine with random flips.

    Args:
        batch: A batch to transform.

    Returns:
        batch: The transformed batch.
    """
    if random.random() > 0.5:
        h, w = batch["input"]["sensor_input"].shape[-2], batch["input"]["sensor_input"].shape[-1]
        assert h == w
        batch["input"]["sensor_input"] = torch.rot90(batch["input"]["sensor_input"], 1, [2, 3])
        corners = batch["input"]["interp"]["corners"]
        corners = torch.flip(corners, (2,))  # switch y and x
        corners[:, :, 1] = h - corners[:, :, 1]  # flip vertical
        batch["input"]["interp"]["corners"] = corners
    return batch


def get_transforms(mode: str, img_channel_idx: list, multi_angle_idx: list) -> list:
    """Get the list of transforms for either 'train' or 'val'.

    Args:
        mode: Specifies 'train' or 'val'.
        img_channel_idx: Index to image features.
        multi_angle_idx: Index to multi-angle features.

    Returns:
        transforms: The list of transform functions.
    """
    if mode == "train":
        transforms = [
            get_norm_transform(img_channel_idx, multi_angle_idx),
            random_rotate_90,
            random_hflip,
            random_vflip,
        ]
    elif mode == "val":
        transforms = [get_norm_transform(img_channel_idx, multi_angle_idx)]
    return transforms
