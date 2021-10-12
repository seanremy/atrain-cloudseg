"""The A-Train dataset consists of input/output pairs where input is multi-angle polarimetry from PARASOL/POLDER and
output is cloud scenario labels from the CALTRACK CLDCLASS product.
"""

import json
import os
import pickle
import random
import sys
from collections import defaultdict
from typing import Callable

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

if "./src" not in sys.path:
    sys.path.insert(0, "./src")  # TO DO: change this once it's a package
from datasets.metrics import get_metrics_func
from datasets.normalization import ATRAIN_MEANS, ATRAIN_STDS


class ATrain(Dataset):
    """The A-Train Dataset."""

    def __init__(
        self,
        mode: str,
        task: str,
        angles_to_omit: list = [],
        fields: list = [],
        get_nondir: bool = False,
        get_flags: bool = False,
        split_name: str = "split_default",
    ) -> None:
        """Create an A-Train Dataset.

        Args:
            mode: Which mode to use for training. In the default split, this must be 'train' or 'val'.
            task: Which task to use this dataset for.
            fields: List of fields to get in this dataset.
            get_nondir: Get non-directional fields for each instance. Considerably slows down data loading. Defaults to
                        False.
            get_flags: Get flags for the cloud scenario output. Defaults to False.
            split_name: The name of the split file to use.
        """
        super().__init__()
        assert all([a >= 0 and a < 16 for a in angles_to_omit])
        self.mode = mode
        self.task = task
        self.metrics_func = get_metrics_func(self.task)
        self.angles_to_omit = angles_to_omit
        self.num_angles = 16 - len(angles_to_omit)
        self.fields = [list(f) for f in fields]
        self.get_nondir = get_nondir
        self.get_flags = get_flags
        self.split_name = split_name

        self.dataset_root = os.path.join(os.path.dirname(__file__), "..", "..", "data", "atrain")
        self.datagen_info = json.load(open(os.path.join(self.dataset_root, "dataset_generation_info.json")))
        # read instance info file, make keys into integers
        self.instance_info = json.load(open(os.path.join(self.dataset_root, "instance_info.json")))
        self.instance_info = {int(k): v for k, v in self.instance_info.items()}
        # load our split, get the instance ids
        self.split = json.load(open(os.path.join(self.dataset_root, f"{self.split_name}.json")))
        self.instance_ids = list(self.split[self.mode])

        # pre-compute length so we don't have to later; it won't change
        self.len = len(self.instance_ids)

        # get index of just the multi-angle fields
        self.multi_angle_idx = []
        self.nondir_idx = []
        self.nondir_fields = []
        channel_idx = 0
        for i in range(len(self.datagen_info["par_fields"])):
            field = self.datagen_info["par_fields"][i]
            if field[0] == "Data_Directional_Fields":
                if field in self.fields:
                    self.multi_angle_idx += [
                        channel_idx + angle_idx for angle_idx in range(16) if angle_idx not in self.angles_to_omit
                    ]
                channel_idx += 16
            else:
                if field in self.fields:
                    self.nondir_fields.append(field[1])
                    self.nondir_idx.append(channel_idx)
                channel_idx += 1
        self.multi_angle_idx = np.array(self.multi_angle_idx)
        self.nondir_idx = np.array(self.nondir_idx)

        # get indices to image channels vs geometry channels
        f_ang = [f for f in self.fields for _ in range(self.num_angles)]
        is_img_c = lambda f: f[0] == "Data_Directional_Fields" and f[1][0] in ["I", "Q", "U"] and f[1][-1] == "P"
        self.img_channel_idx = [i for i in range(len(f_ang)) if is_img_c(f_ang[i])]
        self.geom_channel_idx = [i for i in range(len(f_ang)) if not is_img_c(f_ang[i])]
        is_sin_c = lambda f: f[1] in ["thetas", "thetav"]  # solar zenith angle, view zenith angle
        self.sin_channel_idx = [i for i in range(len(f_ang)) if is_sin_c(f_ang[i])]  # channels we want to apply sin to

    def __len__(self) -> int:
        """Get the length of this dataset."""
        return self.len

    def __getitem__(self, idx: int) -> dict:
        """Get the item at the specified index."""
        inst_id = self.instance_ids[idx]
        inst = self.instance_info[inst_id]
        parasol_arr = np.load(os.path.join(self.dataset_root, inst["input_path"]))
        input_arr = parasol_arr[:, :, self.multi_angle_idx]
        # clip the image channels between 0 and 1
        input_arr[:, :, self.img_channel_idx] = np.clip(input_arr[:, :, self.img_channel_idx], 0, 1)
        # get the geometry mask and add it as a feature
        if len(self.geom_channel_idx) > 0:
            geom_mask = input_arr[:, :, self.geom_channel_idx[0]] != -32767
            input_arr = np.concatenate([input_arr, np.expand_dims(geom_mask, axis=2)], axis=2)
        input_arr[input_arr == -32767] = 0
        if len(self.sin_channel_idx) > 0:
            input_arr[:, :, self.sin_channel_idx] = np.sin(np.pi * input_arr[:, :, self.sin_channel_idx] / 180)
        # put channel dimension first
        input_arr = np.transpose(input_arr, (2, 0, 1))
        output_dict = pickle.load(open(os.path.join(self.dataset_root, inst["output_path"]), "rb"))
        interp_corners, interp_weights = output_dict.pop("corner_idx"), output_dict.pop("corner_weights")
        cloud_scenario_flags = output_dict.pop("cloud_scenario")
        cloud_scenario = cloud_scenario_flags.pop("cloud_scenario")  # (p, 125)

        item = {
            "instance_id": inst_id,
            "input": {
                "sensor_input": input_arr,
                "interp": {
                    "corners": interp_corners,
                    "weights": interp_weights,
                },
            },
            "output": {"cloud_scenario": cloud_scenario},
        }

        if self.get_nondir:
            nondir_input = parasol_arr[:, :, self.nondir_idx]
            item["input"]["nondirectional_fields"] = {}
            for i in range(len(self.nondir_fields)):
                nondir_field = self.nondir_fields[i]
                item["input"]["nondirectional_fields"][nondir_field] = nondir_input[:, :, i]
            for f in ["lat", "lon", "height", "time"]:
                item["output"][f] = output_dict.pop(f)

        if self.get_flags:
            item["output"]["cloud_scenario_flags"] = cloud_scenario_flags

        return item

    def evaluate(self, predictions: dict) -> dict:
        """Evaluate a set of predictions w.r.t. a set of metrics on this dataset.

        Args:
            predictions: The predictions to evaluate.

        Returns:
            metrics: The metrics' evaluations on the provided predictions.
        """
        metrics = {"instance_metrics": {}, "overall_metrics": {}}

        # collect per-instance metrics
        for inst_id in self.instance_ids:

            inst = self.instance_info[inst_id]
            pred = predictions[inst_id]

            gt_labels = pickle.load(open(os.path.join(self.dataset_root, inst["output_path"]), "rb"))
            gt_cloud_scenario = gt_labels["cloud_scenario"]["cloud_scenario"]

            inst_metrics = self.metrics_func(gt_cloud_scenario, pred)

            # treat the altitude metrics as their own separate tasks
            tasks = list(inst_metrics.keys())
            for task in tasks:
                if "altitude_metrics" in inst_metrics[task]:
                    alt_metrics = inst_metrics[task].pop("altitude_metrics")
                    for alt_idx in range(len(alt_metrics)):
                        inst_metrics[f"alt_{alt_idx}_{task[:-3]}_2d"] = alt_metrics[alt_idx]

            metrics["instance_metrics"][inst_id] = inst_metrics

            for task in inst_metrics:
                if task not in metrics["overall_metrics"]:
                    metrics["overall_metrics"][task] = {}
                if "2d" in task:
                    arr_size = inst_metrics[task].pop("num_pixels")
                else:
                    arr_size = inst_metrics[task].pop("num_voxels")

                for metric in inst_metrics[task]:
                    if metric not in metrics["overall_metrics"][task]:
                        metrics["overall_metrics"][task][metric] = [0, 0]
                    if metric in ["true_positives", "false_positives", "false_negatives", "true_negatives"]:
                        metrics["overall_metrics"][task][metric][0] += inst_metrics[task][metric]
                        metrics["overall_metrics"][task][metric][1] = 1
                    else:
                        metrics["overall_metrics"][task][metric][0] += inst_metrics[task][metric] * arr_size
                        metrics["overall_metrics"][task][metric][1] += arr_size

        # aggregate overall metrics
        for task in metrics["overall_metrics"]:
            for metric in metrics["overall_metrics"][task]:
                m = metrics["overall_metrics"][task][metric]
                m = m[0] / m[1]
                metrics["overall_metrics"][task][metric] = m

        return metrics


def collate_atrain(batch: list) -> dict:
    """Collate a batch from the A-Train Dataset.

    Args:
        batch: A list of instances, where each instance is a dictionary.

    Returns:
        coll_batch: The collated batch.
    """
    coll_batch = {}
    inst_ids = []
    sensor_input = []
    b_idx = []
    interp_corners = []
    interp_weights = []
    cloud_scenario = []
    for inst_idx in range(len(batch)):
        inst = batch[inst_idx]
        inst_ids.append(inst["instance_id"])
        sensor_input.append(torch.as_tensor(inst["input"]["sensor_input"], dtype=torch.float))
        b_idx.append(torch.as_tensor([inst_idx], dtype=torch.long).repeat(inst["input"]["interp"]["corners"].shape[0]))
        # Pre-compute interpolation corners and weights so that applying the interpolated loss is quick and easy
        interp_corners.append(torch.as_tensor(inst["input"]["interp"]["corners"], dtype=torch.long))
        interp_weights.append(torch.as_tensor(inst["input"]["interp"]["weights"], dtype=torch.float))
        cloud_scenario.append(torch.as_tensor(inst["output"]["cloud_scenario"], dtype=torch.long))
    coll_batch = {
        "instance_id": torch.as_tensor(inst_ids),
        "input": {
            "sensor_input": torch.stack(sensor_input, dim=0),
            "interp": {
                "batch_idx": torch.cat(b_idx, dim=0),
                "corners": torch.cat(interp_corners, dim=0),
                "weights": torch.cat(interp_weights, dim=0),
            },
        },
        "output": {"cloud_scenario": torch.cat(cloud_scenario, dim=0)},
    }

    if "nondirectional_fields" in batch[0]["input"]:
        nondir_fields = {field_name: [] for field_name in batch[0]["input"]["nondirectional_fields"]}
        for instance in batch:
            for field_name in nondir_fields:
                nondir_fields[field_name].append(
                    torch.as_tensor(instance["input"]["nondirectional_fields"][field_name])
                )
        coll_batch["input"]["nondirectional_fields"] = {k: torch.stack(v, dim=0) for k, v in nondir_fields.items()}
        geom_output = {f: [] for f in ["lat", "lon", "height", "time"]}
        for instance in batch:
            for f in geom_output:
                geom_output[f].append(torch.as_tensor(instance["output"][f]))
        coll_batch["output"]["geometry"] = {k: torch.cat(v, dim=0) for k, v in geom_output.items()}

    if "cloud_scenario_flags" in batch[0]["output"]:
        flags = {flag_name: [] for flag_name in batch[0]["output"]["cloud_scenario_flags"]}
        for instance in batch:
            for flag_name in flags:
                flags[flag_name].append(torch.as_tensor(instance["output"]["cloud_scenario_flags"][flag_name]))
        coll_batch["output"]["cloud_scenario_flags"] = {k: torch.cat(v, dim=0) for k, v in flags.items()}

    return coll_batch


def interp_atrain_output(batch: dict, out: torch.Tensor) -> torch.Tensor:
    """Interpolate output from a model to line up with the labels in a batch.

    Args:
        batch: A batch of instances from the A-Train dataset.
        out: The output of a model (same spatial resolution as input) which gets interpolated at labeled locations.

    Returns:
        out_interp: The interpolated output.
    """
    out = out.permute(0, 2, 3, 1)  # (B, H, W, C)

    # repeat the batch index for the 4 interp corners
    batch_idx = batch["input"]["interp"]["batch_idx"].expand(4, -1).T.reshape(-1)

    # height and width
    patch_shape = batch["input"]["sensor_input"].shape[-2:]

    # index to the 4 interp corners
    corner_idx = batch["input"]["interp"]["corners"]
    # keep the corners in bounds
    corner_idx[corner_idx[:, :, 0] < 0] = 0  # too high
    corner_idx[corner_idx[:, :, 0] >= patch_shape[0]] = patch_shape[0] - 1  # too low
    corner_idx[corner_idx[:, :, 1] < 0] = 0  # too far left
    corner_idx[corner_idx[:, :, 1] >= patch_shape[1]] = patch_shape[1] - 1  # too far right
    # get it as a flat index
    corner_idx = corner_idx[:, :, 0] * patch_shape[0] + corner_idx[:, :, 1]
    corner_idx = corner_idx.reshape(-1)

    # add the batch index to get the overall index
    idx = batch_idx * patch_shape[0] * patch_shape[1] + corner_idx

    # get the corner values
    out_corners = out.reshape(-1, out.shape[3])[idx]

    # get the weights of each corner
    weights = batch["input"]["interp"]["weights"].view(-1)

    # get the weighted corner values
    out_corners_weighted = weights.reshape(-1, 1) * out_corners
    out_corners_weighted = out_corners_weighted.view(weights.shape[0] // 4, 4, out_corners.shape[1])

    # sum up the weighted corner values to get final interpolated values
    out_interp = torch.sum(out_corners_weighted, dim=1)
    return out_interp


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
    """Random vertical flip trnasform. Also flips the interpolation corners.

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
        transforms = [get_norm_transform(img_channel_idx, multi_angle_idx), random_hflip, random_vflip]
    elif mode == "val":
        transforms = [get_norm_transform(img_channel_idx, multi_angle_idx)]
    return transforms
