"""A U-Net-like model, slightly modified to better work for the high number of channels in many satellite datasets.

Based on: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Down(nn.Module):
    """Downsampling block in a U-Net."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Create a Down block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
        """
        super().__init__()
        self.mp = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # # focal loss bias initialization
        # nn.init.normal_(self.conv1.bias, mean=0, std=0.01)
        # nn.init.normal_(self.conv2.bias, mean=0, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mp(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        return x


class Up(nn.Module):
    """Upsampling block in a U-Net."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Create an Up block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
        """
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels // 2)
        self.conv2 = nn.Conv2d(in_channels // 2, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # # focal loss bias initialization
        # nn.init.normal_(self.conv1.bias, mean=0, std=0.01)
        # nn.init.normal_(self.conv2.bias, mean=0, std=0.01)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
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
    """A U-Net model, with a slight modification: there are 1x1 convolutions at the top and bottom of the network to
    deal with the high number of channels in many satellite datasets.
    """

    def __init__(
        self, in_channels: int, out_channels: int, base_depth: int, img_dims: tuple, num_layers: int = None
    ) -> None:
        """Create a U-Net.

        Args:
            in_channels: The number of input channels.
            out_channels: The number of output channels.
            base_depth: The base convolutional filter depth, which increases by 2 at every Down block.
            img_dims: The height and width of images input into this network.
            num_layers: The number of layers in this network. Defaults to the max number of layers possible.
        """
        super().__init__()
        self.in_channels = in_channels
        self.base_depth = base_depth
        self.img_dims = img_dims
        if num_layers is None:
            # as many layers as can be fit with input at least 1x1
            self.num_layers = int(np.log(min(self.img_dims)) / np.log(2))
        else:
            self.num_layers = num_layers

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

        # # focal loss bias initialization
        # nn.init.normal_(self.pre_conv_a.bias, mean=0, std=0.01)
        # nn.init.normal_(self.pre_conv_b.bias, mean=0, std=0.01)
        # nn.init.constant_(self.post_conv.bias, -np.log(99))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_down = [F.relu(self.pre_conv_a(x))]
        x_skip = F.relu(self.pre_conv_b(x))
        for db in self.down_blocks:
            x_down.append(db(x_down[-1]))
        x_up = x_down.pop(-1)
        for ub in self.up_blocks:
            x_up = ub(x_up, x_down.pop(-1))
        x = torch.cat([x_up, x_skip], dim=1)
        x = self.post_conv(x)
        return x
