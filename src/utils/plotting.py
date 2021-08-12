"""Plotting utilities."""

import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

if "./src" not in sys.path:
    sys.path.insert(0, "./src")  # TO DO: change this once it's a package
from utils.atrain import CLOUD_SCENARIO_INFO, map_cloud_scenario_colors


def plot_cloud_type(cloud_type: np.array, save_path: str = None) -> None:
    """Plot cloud type colors.

    Args:
        cloud_type: Array of cloud scenario codes, from 0 to 8.
        save_path: Path to output figure file, or None if not saving.
    """
    color_float_to_int = lambda color: tuple([n / 255 for n in color])
    cloud_scenario_colors = map_cloud_scenario_colors(cloud_type)
    legend_elements = [
        Patch(
            facecolor=color_float_to_int(CLOUD_SCENARIO_INFO[scenario_num]["color"]),
            edgecolor=color_float_to_int(CLOUD_SCENARIO_INFO[scenario_num]["color"]),
            label=CLOUD_SCENARIO_INFO[scenario_num]["name"],
        )
        for scenario_num in CLOUD_SCENARIO_INFO
    ]
    plt.figure()
    plt.imshow(cloud_scenario_colors.astype(float) / 255)
    plt.legend(handles=legend_elements, prop={"size": 6})
    if not save_path is None:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)


def plot_cloud_mask(cloud_mask: np.array, save_path: str = None) -> None:
    """Plot the cloud mask.

    Args:
        cloud_mask: Array representing the cloud mask, with shape (# height bins, # entries)
    """
    plt.figure()
    plt.imshow(cloud_mask.astype(float) / 255)
    if not save_path is None:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)


def get_cloud_mask_viz(pred_cloud_mask: np.array, gt_cloud_scenario: np.array) -> np.array:
    """Get a visualization of predicted and ground-truth cloud mask, separated by a gray line.

    Args:
        pred_cloud_mask: The predicted cloud mask.
        gt_cloud_scenario: The ground-truth cloud scenario.

    Returns:
        viz: The cloud mask visualization.
    """
    pred_cloud_mask = pred_cloud_mask.cpu().detach().numpy().T
    gt_cloud_mask = gt_cloud_scenario.view(gt_cloud_scenario.shape[0], -1).cpu().detach().numpy().T > 0
    divider = np.zeros((min(3, gt_cloud_mask.shape[0]), pred_cloud_mask.shape[1])) + 0.5
    viz = np.concatenate([pred_cloud_mask, divider, gt_cloud_mask], axis=0)
    return viz
