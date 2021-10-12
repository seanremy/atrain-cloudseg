"""Metrics for the A-Train Cloud Segmentation Dataset."""

import numpy as np


def get_bin_seg_2d_metrics(gt_bin_seg_2d: np.array, pred_bin_seg_2d: np.array) -> dict:
    """Get the binary 2D segmentation metrics for a ground-truth, prediction pair.

    Args:
        gt_bin_seg_2d: Ground-truth binary 2D cloud mask.
        pred_bin_seg_2d: Predicted binary 2D cloud mask.

    Returns:
        metrics: Dictionary of binary 2D segmentation metrics.
    """
    tp = np.multiply(gt_bin_seg_2d, pred_bin_seg_2d).sum()
    fp = np.multiply(~gt_bin_seg_2d, pred_bin_seg_2d).sum()
    fn = np.multiply(gt_bin_seg_2d, ~pred_bin_seg_2d).sum()
    tn = np.multiply(~gt_bin_seg_2d, ~pred_bin_seg_2d).sum()

    metrics = {
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "num_pixels": pred_bin_seg_2d.size,
    }
    dice_denom = 2 * tp + fp + fn
    if dice_denom == 0:
        metrics["dice_score"] = 1
    else:
        metrics["dice_score"] = 2 * tp / dice_denom

    return metrics


def get_bin_seg_3d_metrics(gt_bin_seg_3d: np.array, pred_bin_seg_3d: np.array) -> dict:
    """Get the binary 3D segmentation metrics for a ground-truth, prediction pair.

    Args:
        gt_bin_seg_3d: Ground-truth binary 3D cloud mask.
        pred_bin_seg_3d: Predicted binary 3D cloud mask.

    Returns:
        metrics: Dictionary of binary 3D segmentation metrics.
    """
    altitude_metrics = []
    for alt_idx in range(gt_bin_seg_3d.shape[1]):
        altitude_metrics.append(get_bin_seg_2d_metrics(gt_bin_seg_3d[:, alt_idx], pred_bin_seg_3d[:, alt_idx]))
    summable_metrics = ["true_positives", "false_positives", "false_negatives", "true_negatives", "num_pixels"]
    metrics = {m: np.sum([am[m] for am in altitude_metrics]) for m in summable_metrics}
    metrics["altitude_metrics"] = altitude_metrics
    tp, fp, fn = metrics["true_positives"], metrics["false_positives"], metrics["false_negatives"]
    dice_denom = 2 * tp + fp + fn
    if dice_denom == 0:
        metrics["dice_score"] = 1
    else:
        metrics["dice_score"] = 2 * tp / dice_denom
    metrics["precision"] = tp / max(1, tp + fp)
    metrics["recall"] = tp / max(1, tp + fn)
    metrics["num_voxels"] = metrics.pop("num_pixels")
    return metrics


def get_seg_2d_metrics(gt_seg_2d, pred_seg_2d):
    # TO DO
    raise NotImplementedError


def get_seg_3d_metrics(gt_seg_3d, pred_seg_3d):
    # TO DO
    raise NotImplementedError


def get_metrics_func(task: str) -> function:
    """Get a function to compute metrics for a task.

    Args:
        task: Which task to get the metrics function for.

    Returns:
        get_metrics: A function to get metrics for ground-truth, prediction pairs for this task.
    """

    def get_metrics(gt, pred):
        pred = pred.detach().cpu().numpy()
        metrics = {}
        if task == "seg_3d":
            metrics["seg_3d"] = get_seg_3d_metrics(gt, pred)
        if task in ["bin_seg_3d", "seg_3d"]:
            gt_bin_seg_3d = (gt > 0).astype(bool)
            pred_bin_seg_3d = (pred > 0.5).astype(bool)
            metrics["bin_seg_3d"] = get_bin_seg_3d_metrics(gt_bin_seg_3d, pred_bin_seg_3d)
        if task in ["bin_seg_2d", "bin_seg_3d", "seg_3d"]:
            gt_bin_seg_2d = (gt > 0).any(axis=1).astype(bool)
            pred_bin_seg_2d = (pred > 0.5).any(axis=1).astype(bool)
            metrics["bin_seg_2d"] = get_bin_seg_2d_metrics(gt_bin_seg_2d, pred_bin_seg_2d)
        return metrics

    return get_metrics
