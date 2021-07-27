"""The A-Train dataset consists of input/output pairs where input is multi-angle polarimetry from PARASOL/POLDER and
output is cloud scenario labels from the CALTRACK CLDCLASS product.
"""

import json
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

ALL_METRICS = [
    "cloud_mask_accuracy",
    "cloud_scenario_accuracy",
    "cloudtop_height_bin_accuracy",
    "cloudtop_height_bin_offset_error",
]
MASK_ONLY_METRICS = ["cloud_mask_accuracy", "cloudtop_height_bin_accuracy", "cloudtop_height_bin_offset_error"]


class ATrain(Dataset):
    """The A-Train Dataset."""

    def __init__(
        self, mode: str, split_name: str = "split_default", get_nondir: bool = False, get_flags: bool = False
    ) -> None:
        """Create an A-Train Dataset.

        Args:
            mode: Which mode to use for training. In the default split, this must be 'train' or 'val'.
            split_name: The name of the split file to use.
            get_nondir: Get non-directional fields for each instance. Considerably slows down data loading. Defaults to
                        False.
            get_flags: Get flags for the cloud scenario output. Defaults to False.
        """
        super().__init__()
        self.mode = mode
        self.split_name = split_name
        self.get_nondir = get_nondir
        self.get_flags = get_flags

        self.dataset_root = os.path.join(os.path.dirname(__file__), "..", "..", "data", "atrain")
        self.datagen_info = json.load(open(os.path.join(self.dataset_root, "dataset_generation_info.json")))
        # read instance info file, make keys into integers
        self.instance_info = json.load(open(os.path.join(self.dataset_root, "instance_info.json")))
        self.instance_info = {int(k): v for k, v in self.instance_info.items()}
        # load our split, get the instance ids
        self.split = json.load(open(os.path.join(self.dataset_root, f"{self.split_name}.json")))
        self.instance_ids = list(self.split[self.mode])
        # TO DO: remove
        bad_instances = json.load(open(os.path.join(self.dataset_root, "bad_instances.json")))
        self.instance_ids = [i for i in self.instance_ids if i not in bad_instances]
        # pre-compute length so we don't have to later; it won't change
        self.len = len(self.instance_ids)

        # get index of just the multi-angle fields
        self.multi_angle_idx = []
        self.nondir_fields = []
        channel_idx = 0
        for i in range(len(self.datagen_info["par_fields"])):
            field = self.datagen_info["par_fields"][i]
            if field[0] == "Data_Directional_Fields":
                self.multi_angle_idx += list(range(channel_idx, channel_idx + 16))
                channel_idx += 16  # 16 angles
            else:
                self.nondir_fields.append(field[1])
                channel_idx += 1
        if self.get_nondir:
            self.nondir_idx = np.array([i for i in range(channel_idx) if i not in self.multi_angle_idx])
        self.multi_angle_idx = np.array(self.multi_angle_idx)

    def __len__(self) -> int:
        """Get the length of this dataset."""
        return self.len

    def _patch_idx_to_interp(self, patch_idx: tuple[np.array, np.array]) -> tuple[np.array, np.array]:
        """Convert the patch index to interpolation corners / weights to apply to model output.

        Args:
            patch_idx: A tuple of numpy arrays for the y- and x- coordinates of output values in the input array.

        Returns:
            interp_corners: The locations of corners to use for interpolation
            interp_weights: The weights for these corners, summing to 1 for each box
        """
        idx_y, idx_x = patch_idx

        top_bottom = np.stack([np.floor(idx_y).astype(int), np.ceil(idx_y).astype(int)], axis=1)
        left_right = np.stack([np.floor(idx_x).astype(int), np.ceil(idx_x).astype(int)], axis=1)

        topleft = np.stack([top_bottom[:, 0], left_right[:, 0]], axis=1)
        topright = np.stack([top_bottom[:, 0], left_right[:, 1]], axis=1)
        bottomleft = np.stack([top_bottom[:, 1], left_right[:, 0]], axis=1)
        bottomright = np.stack([top_bottom[:, 1], left_right[:, 1]], axis=1)

        interp_corners = np.stack([topleft, topright, bottomleft, bottomright], axis=1)
        weight_y = 1 - np.abs(interp_corners[:, :, 0] - np.stack([idx_y] * 4, axis=1))
        weight_x = 1 - np.abs(interp_corners[:, :, 1] - np.stack([idx_x] * 4, axis=1))
        interp_weights = np.expand_dims(weight_y * weight_x, axis=2)

        return interp_corners, interp_weights

    def __getitem__(self, idx: int) -> dict:
        """Get the item at the specified index."""
        inst_id = self.instance_ids[idx]
        inst = self.instance_info[inst_id]
        parasol_arr = np.load(os.path.join(self.dataset_root, inst["input_path"]))
        input_arr = parasol_arr[:, :, self.multi_angle_idx]
        input_arr = np.transpose(np.clip(input_arr, 0, 1), (2, 0, 1))
        output_dict = pickle.load(open(os.path.join(self.dataset_root, inst["output_path"]), "rb"))
        assert inst_id == output_dict.pop("instance_id")

        patch_idx = output_dict.pop("patch_idx")
        interp_corners, interp_weights = self._patch_idx_to_interp(patch_idx)  # (p, 4, 3)

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
            item["nondirectional_fields"] = {}
            for i in range(len(self.nondir_fields)):
                nondir_field = self.nondir_fields[i]
                item["input"]["nondirectional_fields"][nondir_field] = nondir_input[:, :, i]

        if self.get_flags:
            item["output"]["cloud_scenario_flags"] = cloud_scenario_flags

        return item

    def evaluate(self, predictions: dict, metrics: list[str] = ALL_METRICS) -> dict:
        """TO DO"""
        metrics = {m: [] for m in metrics}
        for inst_id in self.instance_ids:
            if inst_id not in predictions:
                for m in metrics:
                    metrics[m].append(0)

            inst = self.instance_info[inst_id]
            pred = predictions[inst_id]

            gt_labels = pickle.load(open(os.path.join(self.dataset_root, inst["output_path"]), "rb"))
            gt_cloud_scenario = gt_labels["cloud_scenario"]["cloud_scenario"]

            assert gt_cloud_scenario.shape == pred.shape

            gt_cloud_mask = (gt_cloud_scenario > 0).any(axis=1)
            pred_cloud_mask = (pred > 0).any(axis=1)

            def _min(a):
                if a.shape[0] == 0:
                    return -1
                return np.min(a)

            h_bins_pred = np.array([_min(np.where(pred[i].cpu().detach().numpy())[0]) for i in range(pred.shape[0])])
            h_bins_gt = np.array([_min(np.where(gt_cloud_scenario[i])[0]) for i in range(gt_cloud_scenario.shape[0])])
            import pdb

            pdb.set_trace()

            if "cloud_mask_accuracy" in metrics:
                # cloud mask accuracy := proportion of pixels correctly identified as cloud / not cloud
                metrics["cloud_mask_accuracy"].append(np.mean(gt_cloud_mask == pred_cloud_mask.cpu().detach().numpy()))

            if "cloud_scenario_accuracy" in metrics:
                # cloud scenario accuracy := proportion of pixel + height bin combinations whose cloud scenario is correctly identified
                metrics["cloud_scenario_accuracy"].append(np.mean(gt_cloud_scenario == pred.cpu().detach().numpy()))

            if "cloudtop_height_bin_accuracy" in metrics:
                # cloud-top height bin accuracy := proportion of pixels whose highest cloud is correctly identified
                metrics["cloudtop_height_bin_accuracy"].append(np.mean(h_bins_pred == h_bins_gt))

            if "cloudtop_height_bin_offset_error" in metrics:
                # cloud-top height bin offset := average distance between predicted and GT cloud-top height, only computed for points pixels where both prediction and GT have clouds
                both_clouds = pred_cloud_mask.cpu().detach().numpy() * gt_cloud_mask
                height_bin_offsets = np.abs(h_bins_pred[both_clouds] - h_bins_gt[both_clouds])
                metrics["cloudtop_height_bin_offset_error"].append(np.mean(height_bin_offsets))
        metrics["instance_ids"] = list(self.instance_info.keys())
        metrics = {k: np.array(v) for k, v in metrics.items()}
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
    out = out.permute(0, 2, 3, 1)

    batch_idx = batch["input"]["interp"]["batch_idx"]
    corner_idx = batch["input"]["interp"]["corners"]
    patch_shape = batch["input"]["sensor_input"].shape[2:4]

    # handle out of bounds by simply shifting the offending indices back in bounds...
    # ...this makes it effectively nearest-neighbor on that axis
    corner_idx[corner_idx[:, :, 0] < 0] = 0  # too high
    corner_idx[corner_idx[:, :, 0] >= patch_shape[0]] = patch_shape[0] - 1  # too low
    corner_idx[corner_idx[:, :, 1] < 0] = 0  # too far left
    corner_idx[corner_idx[:, :, 1] >= patch_shape[1]] = patch_shape[1] - 1  # too far right

    corner_idx = corner_idx[:, :, 0] * patch_shape[0] + corner_idx[:, :, 1]
    corner_idx = corner_idx.view(-1)

    idx = batch_idx.repeat(4) * patch_shape[0] * patch_shape[1] + corner_idx
    out_corners = out.reshape(-1, out.shape[3])[idx]

    weights = batch["input"]["interp"]["weights"].view(-1)
    out_corners_weighted = weights.repeat(out_corners.shape[1], 1).T * out_corners
    out_corners_weighted = out_corners.view(weights.shape[0] // 4, 4, out_corners.shape[1])
    out_interp = torch.sum(out_corners_weighted, dim=1)
    return out_interp
