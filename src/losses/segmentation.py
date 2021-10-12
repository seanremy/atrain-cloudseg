"""Loss functions for semantic segmentation of cloud type in each height bin."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseLoss(nn.Module):
    """Abstract base class for segmentation losses."""

    def __init__(self, num_height_bins: int, num_classes: int) -> None:
        """Create a BaseLoss.

        Args:
            num_height_bins: Number of height bins.
            num_classes: Number of classes.
        """
        super(BaseLoss, self).__init__()
        self.num_height_bins = num_height_bins
        self.num_classes = num_classes

    def forward(self, predictions: torch.Tensor, batch: dict) -> torch.Tensor:
        raise NotImplementedError("BaseLoss is abstract, so forward() is unimplemented!")

    def check_shapes(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Check that the shapes of the loss inputs are correct, throwing the appropriate error on failure.

        Args:
            predictions: Predictions, of shape (num_pixels_in_batch, num_height_bins, num_classes).
            targets: Targets, of shape (num_pixels_in_batch, num_height_bins).
        """
        p_shp, t_shp = predictions.shape, targets.shape
        if len(p_shp) != 3:
            raise ValueError(f"Expected 3-dimensional tensor for 'predictions', instead got shape {p_shp}.")
        if len(t_shp) != 2:
            raise ValueError(f"Expected 2-dimensional tensor for 'predictions', instead got shape {t_shp}.")
        if p_shp[1] != self.num_height_bins:
            raise ValueError(
                f"Expected predictions (shape {p_shp}) to have num_height_bins={self.num_height_bins} values in "
                f"dimension 1, instead got: {p_shp[1]}."
            )
        if t_shp[1] != self.num_height_bins:
            raise ValueError(
                f"Expected targets (shape {t_shp}) to have num_height_bins={self.num_height_bins} values in "
                f"dimension 1, instead got: {t_shp[1]}."
            )
        if p_shp[2] != self.num_classes:
            raise ValueError(
                f"Expected predictions (shape {p_shp}) to have num_classes={self.num_classes} values in dimension 2, "
                f"instead got: {p_shp[2]}."
            )
        for i in [0, 1]:
            if p_shp[i] != t_shp[i]:
                raise ValueError(
                    f"Expected predictions (shape {p_shp}) and targets (shape {t_shp}) to have same size in dimension "
                    f"{i}, but got: {p_shp[i]} != {t_shp[i]}"
                )


class CrossEntropyLoss(BaseLoss):
    """This criterion combines LogSoftmax and NLLLoss in one single class. This implementation is specific to cloud
    segmentation in the A-Train dataset.

    For the math, see: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    """

    def __init__(self, num_height_bins: int, num_classes: int, weight: torch.Tensor = None) -> None:
        """Create a Cross Entropy Loss.

        Args:
            num_height_bins: Number of height bins.
            num_classes: Number of classes.
            weight: Class weights.
        """
        super(CrossEntropyLoss, self).__init__(num_height_bins, num_classes)
        self.weight = weight

    def forward(self, predictions: torch.Tensor, batch: dict) -> torch.Tensor:
        """Apply the loss to a set of predictions, with respect to a batch containing ground truth.

        Args:
            predictions: Predictions, of shape (num_pixels_in_batch, num_height_bins, num_classes).
            batch: A batch from the A-Train dataset.

        Returns:
            loss: The cross entropy loss of the predictions with respect to this batch.
        """
        targets = batch["output"]["cloud_scenario"]
        self.check_shapes(predictions, targets)
        loss = F.cross_entropy(torch.swapaxes(predictions, 1, 2), targets.long(), reduction="mean", weight=self.weight)
        return loss


class FocalLoss(BaseLoss):
    """Focal Loss, as defined in the paper "Focal Loss for Dense Object Detection", by Lin et. al.
    This implementation is specific to cloud segmentation in the A-Train dataset.

    For the original paper, see: https://arxiv.org/abs/1708.02002.
    Code Adapted from: https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65938
    """

    def __init__(
        self, num_height_bins: int, num_classes: int, alpha: float = 0.5, gamma: float = 2, weight: torch.Tensor = None
    ) -> None:
        """Create a Focal Loss.

        Args:
            num_height_bins: Number of height bins.
            num_classes: Number of classes.
            alpha: Class balancing parameter.
            gamma: Focusing parameter.
            weight: Class weights.
        """
        super(FocalLoss, self).__init__(num_height_bins, num_classes)
        self.alpha = torch.tensor([alpha] + [(1 - alpha) / (self.num_classes - 1)] * (self.num_classes - 1)).cuda()
        self.gamma = gamma
        self.weight = weight

    def forward(self, predictions: torch.Tensor, batch: dict) -> torch.Tensor:
        """Apply the loss to a set of predictions, with respect to a batch containing ground truth.

        Args:
            predictions: Predictions, of shape (num_pixels_in_batch, num_height_bins, num_classes).
            batch: A batch from the A-Train dataset.

        Returns:
            loss: The focal loss of the predictions with respect to this batch.
        """
        targets = batch["output"]["cloud_scenario"].long()
        self.check_shapes(predictions, targets)
        ce_loss = F.cross_entropy(torch.swapaxes(predictions, 1, 2), targets, reduction="none", weight=self.weight)
        p_t = torch.exp(-ce_loss)
        a_t = self.alpha.gather(0, targets.data.view(-1)).view(*targets.shape)
        loss = a_t * (1 - p_t) ** self.gamma * ce_loss
        loss = loss.mean()
        return loss


class SoftJaccardLoss(BaseLoss):
    """TO DO: something about this isn't working right"""

    def __init__(self, num_height_bins: int, num_classes: int, epsilon: float = 1e-3) -> None:
        """Create a Soft Jacard Loss.

        Args:
            num_height_bins: Number of height bins.
            num_classes: Number of classes.
        """
        super(SoftJaccardLoss, self).__init__(num_height_bins, num_classes)
        self.epsilon = epsilon

    def forward(self, predictions: torch.Tensor, batch: dict) -> torch.Tensor:
        """Apply the loss to a set of predictions, with respect to a batch containing ground truth.

        Args:
            predictions: Predictions, of shape (num_pixels_in_batch, num_height_bins, num_classes).
            batch: A batch from the A-Train dataset.

        Returns:
            loss: The soft Jaccard loss of the predictions with respect to this batch.
        """
        targets = batch["output"]["cloud_scenario"]
        self.check_shapes(predictions, targets)
        predictions = torch.sigmoid(predictions)
        targets_onehot = F.one_hot(targets.long(), num_classes=self.num_classes)
        prod_sum = (targets_onehot * predictions).sum()
        loss = 1 - (prod_sum + self.epsilon) / (targets_onehot.sum() + predictions.sum() - prod_sum + self.epsilon)
        return loss


class SmoothnessPenalty(nn.Module):
    """L1 regularization penalty on the output gradient (note: xy gradient, not gradient in the ML sense) of the
    predictions, modulated by the input image's gradient.

    See (2) from: https://arxiv.org/abs/2003.00752
    """

    def __init__(self) -> None:
        """Create a Smoothness Penalty.

        Args:
            num_height_bins: Number of height bins.
            num_classes: Number of classes.
        """
        super(SmoothnessPenalty, self).__init__()

    def forward(self, out: torch.Tensor, batch: dict, c_idx: list) -> torch.Tensor:
        """Apply the loss to a set of predictions, with respect to a batch containing ground truth.

        Args:
            out: Predictions, before interpolation, of shape (batch_size, num_classes * num_height_bins, height, width).
            batch: A batch from the A-Train dataset.
            c_idx: The index of which channels correspond to images (as opposed to geometry / geography).

        Returns:
            loss: The smoothness penalty of the predictions with respect to this batch.
        """
        img = batch["input"]["sensor_input"][:, c_idx]
        grad_x_filter = torch.Tensor([[1, 0, -1]]).to(img.device).view(1, 1, 1, 3)
        grad_y_filter = torch.Tensor([[1], [0], [-1]]).to(img.device).view(1, 1, 3, 1)

        def get_grad(x, w):
            grad_list = [F.conv2d(x[:, i : i + 1, :, :], w, padding="same") for i in range(x.size(1))]
            return torch.abs(torch.cat(grad_list, dim=1))

        img_grad_x = get_grad(img, grad_x_filter).mean(dim=1, keepdim=True)
        img_grad_y = get_grad(img, grad_y_filter).mean(dim=1, keepdim=True)
        out_grad_x = get_grad(out, grad_x_filter)
        out_grad_y = get_grad(out, grad_y_filter)
        loss = (out_grad_x * torch.exp(-img_grad_x) + out_grad_y * torch.exp(-img_grad_y)).mean()
        return loss
