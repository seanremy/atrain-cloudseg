"""Cloud-Net model definition. See: https://arxiv.org/abs/1901.10077

Code adapted from https://github.com/SorourMo/Cloud-Net-A-semantic-segmentation-CNN-for-cloud-detection
Some parts of their implementation are borrowed from https://www.kaggle.com/cjansen/u-net-in-keras
"""
from typing import List, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class CloudNet(nn.Module):
    """The main module for Cloud-Net. Similar to a U-Net, with two main differences:
    1) The shortcut connections use all previous layer features (stacked channel-wise)
    2) The convolutional blocks are different.
    """

    def __init__(self, num_channels: int = 4, num_classes: int = 1):
        """Create a CloudNet.

        Args:
            num_channels: Number of channels in the input.
            num_classes: Number of classes (channels) in the output.
        """
        super().__init__()
        self.conv0 = nn.Conv2d(num_channels, 16, kernel_size=3)
        self.conv1 = ContractingArm(16, 32)
        self.conv2 = ContractingArm(32, 64)
        self.conv3 = ContractingArm(64, 128)
        self.conv4 = ContractingArm(128, 256)
        self.conv5 = ContractingArm3(256, 512, kernel_size=3)
        self.conv6 = ContractingArm(512, 1024, dropout=True)
        self.conv7 = ExpandingArm3(1024, 512)
        self.conv8 = ExpandingArm(512, 256)
        self.conv9 = ExpandingArm(256, 128)
        self.conv10 = ExpandingArm(128, 64)
        self.conv11 = ExpandingArm(64, 32)
        self.conv12 = nn.Conv2D(32, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.conv0(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x = self.conv7(x6, self.shortcut([x5, x4, x3, x2, x1])) + x5
        x = self.conv8(x, self.shortcut([x4, x3, x2, x1])) + x4
        x = self.conv9(x, self.shortcut([x3, x2, x1])) + x3
        x = self.conv10(x, self.shortcut([x2, x1])) + x2
        x = self.conv11(x, x1) + x1
        x = self.conv12(x)
        return x

    def shortcut(x: List(torch.Tensor)):
        """Add together all of the previous layers' features (repeating to broadcast to same size), then max pool."""
        for i in range(1, len(x)):
            x[i] = F.max_pool2d(x[i].repeat(1, 1, 1, 2 ** i), kernel_size=2 ** i)
        return F.relu(sum(x))


class ContractingArm(nn.Module):
    """Convolutional block.
    Branch A: Concatenates the input with the output from a 1x1 conv
    Branch B: Passes the input through two 3x3 conv layers
    Output: Max pool of the sum of outputs from branches A and B
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: bool = False):
        """Create a ContractingArm.

        Args:
            in_channels: The number of channels in the input.
            out_channels: The number of channels in the output.
            dropout: True to include the dropout layer, defaults to False.
        """
        super().__init__()
        self.dropout = dropout
        # branch A
        self.conv_a = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, padding="same")
        self.bn_a = nn.BatchNorm2d(out_channels // 2)
        # branch B
        self.conv_b1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same")
        self.bn_b1 = nn.BatchNorm2d(out_channels)
        self.conv_b2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same")
        self.bn_b2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # branch A
        x_a = self.conv_a(x)
        x_a = F.relu(self.bn_a(x_a))
        x_a = torch.cat([x, x_a], dim=3)
        # branch B
        x_b = self.conv_b1(x)
        x_b = F.relu(self.bn_b1(x_b))
        x_b = self.conv_b2(x)
        if self.dropout:
            x_b = F.dropout(x_b, p=0.15)
        x_b = F.relu(self.bn_b2(x_b))
        x = F.max_pool2d(x_a + x_b, kernel_size=2)
        return x


class ContractingArm3(nn.Module):
    """Convolutional block.
    Like ContractingArm, except uses three 3x3 conv layers instead of two, and the output of the second 3x3 conv layer
    inputs into a separate 1x1 conv before the addition.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """Create a ContractingArm3.

        Args:
            in_channels: The number of channels in the input.
            out_channels: The number of channels in the output.
        """
        super().__init__()
        # branch A
        self.conv_a = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, padding="same")
        self.bn_a = nn.BatchNorm2d(out_channels // 2)
        # branch B
        self.conv_b1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same")
        self.bn_b3 = nn.BatchNorm2d(out_channels)
        self.conv_b2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same")
        self.bn_b3 = nn.BatchNorm2d(out_channels)
        self.conv_b3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same")
        self.bn_b3 = nn.BatchNorm2d(out_channels)
        # branch C
        self.conv_c = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding="same")
        self.bn_c = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # branch A
        x_a = self.conv_a(x)
        x_a = F.relu(self.bn_a(x_a))
        x_a = torch.cat([x, x_a], dim=3)
        # branches B & C
        x_b = self.conv_b1(x)
        x_b = F.relu(self.bn_b1(x_b))
        x_b = self.conv_b2(x_b)
        x_b = F.relu(self.bn_b2(x_b))
        x_c = self.conv_c(x_b)
        x_c = F.relu(self.bn_c(x_c))
        x_b = self.conv_b3(x_b)
        x_b = F.relu(self.bn_b3(x_b))
        x = F.max_pool2d(x_a + x_b + x_c, kernel_size=2)
        return x


class ExpandingArm(nn.Module):
    """Conv transpose block.
    Concatenates y with the conv transpose features of x, then passes that through two 3x3 conv layers, adding the
    (skip) conv transpose features again at the end.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        Create an ExpandingArm.

        Args:
            in_channels: The number of channels in the input.
            out_channels: The number of channels in the output.
        """
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding="same")
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same")
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same")
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, y):
        x_t = self.conv1(x)
        x = torch.cat([x_t, y], dim=3)
        x = self.conv2(x)
        x = F.relu(self.bn1(x))
        x = self.conv3(x)
        x = F.relu(self.bn2(x))
        return x + x_t


class ExpandingArm3(nn.Module):
    """Convolutional block.
    Like ExpandingArm but has three 3x3 conv layers.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """Create an ExpandingArm3.

        Args:
            in_channels: The number of channels in the input.
            out_channels: The number of channels in the output.
        """
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding="same")
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same")
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same")
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same")
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x, y):
        x_t = self.conv1(x)
        x = torch.cat([x_t, y], dim=3)
        x = self.conv2(x)
        x = F.relu(self.bn1(x))
        x = self.conv3(x)
        x = F.relu(self.bn2(x))
        x = self.conv4(x)
        x = F.relu(self.bn3(x))
        return x + x_t
