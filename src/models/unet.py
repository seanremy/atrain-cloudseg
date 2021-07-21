"""TO DO

Based on: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Down(nn.Module):
    """TO DO"""

    def __init__(self, in_channels, out_channels):
        """TO DO"""
        super().__init__()
        self.mp = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.mp(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        return x


class Up(nn.Module):
    """TO DO"""

    def __init__(self, in_channels, out_channels):
        """TO DO"""
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels // 2)
        self.conv2 = nn.Conv2d(in_channels // 2, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        return x


class UNet(nn.Module):
    """TO DO"""

    def __init__(self, in_channels, out_channels, base_depth, patch_shape):
        """TO DO"""
        super().__init__()
        self.in_channels = in_channels
        self.base_depth = base_depth
        self.patch_shape = patch_shape
        self.num_layers = int(min(np.log(self.patch_shape[0]) / np.log(2), np.log(self.patch_shape[1]) / np.log(2)))
        self.pre_conv_a = nn.Conv2d(self.in_channels, self.base_depth, kernel_size=1)
        self.pre_conv_b = nn.Conv2d(self.in_channels, self.base_depth // 2, kernel_size=1)
        self.down_blocks = []
        for i in range(self.num_layers - 1):
            self.down_blocks.append(Down(self.base_depth * (2 ** i), self.base_depth * (2 ** (i + 1))))
        self.down_blocks.append(
            Down(self.base_depth * (2 ** (self.num_layers - 1)), self.base_depth * (2 ** (self.num_layers - 1)))
        )
        self.down_blocks = nn.Sequential(*self.down_blocks)
        self.up_blocks = []
        for i in range(self.num_layers, 0, -1):
            self.up_blocks.append(Up(int(self.base_depth * (2 ** i)), int(base_depth * (2 ** (i - 2)))))
        self.up_blocks = nn.Sequential(*self.up_blocks)
        self.post_conv = nn.Conv2d(self.base_depth, out_channels, kernel_size=1)

    def forward(self, x):
        x_down = [self.pre_conv_a(x)]
        x_skip = self.pre_conv_b(x)
        for db in self.down_blocks:
            x_down.append(db(x_down[-1]))
        x_up = x_down.pop(-1)
        for ub in self.up_blocks:
            x_up = ub(x_up, x_down.pop(-1))
        x = torch.cat([x_up, x_skip], dim=1)
        x = self.post_conv(x)
        return x
