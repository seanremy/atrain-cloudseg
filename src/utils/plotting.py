"""Plotting utilities."""

import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

if "./src" not in sys.path:
    sys.path.insert(0, "./src")  # TO DO: change this once it's a package
from utils.atrain import CLOUD_SCENARIO_INFO, map_cloud_scenario_colors


def _color_float_to_int(color):
    """TO DO"""
    return tuple([n / 255 for n in color])


def plot_cloud_type(cloud_type: np.array, save_path: str = None) -> None:
    """TO DO"""
    cloud_scenario_colors = map_cloud_scenario_colors(cloud_type)
    legend_elements = [
        Patch(
            facecolor=_color_float_to_int(CLOUD_SCENARIO_INFO[scenario_num]["color"]),
            edgecolor=_color_float_to_int(CLOUD_SCENARIO_INFO[scenario_num]["color"]),
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
    """TO DO"""
    plt.figure()
    plt.imshow(cloud_mask.astype(float) / 255)
    if not save_path is None:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)


def get_cloud_mask_viz(pred_cloud_mask, gt_cloud_scenario):
    """TO DO"""
    pred_cloud_mask = pred_cloud_mask.cpu().detach().numpy().T
    gt_cloud_mask = gt_cloud_scenario.cpu().detach().numpy().T > 0
    divider = np.zeros((3, pred_cloud_mask.shape[1])) + 0.5
    viz = np.concatenate([pred_cloud_mask, divider, gt_cloud_mask], axis=0)
    return viz
