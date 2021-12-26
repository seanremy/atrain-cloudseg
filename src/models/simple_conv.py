"""TO DO"""

import sys

import torch
import torch.nn as nn

if "./src" not in sys.path:
    sys.path.insert(0, "./src")  # TO DO: change this once it's a package
from datasets.atrain import interp_atrain_output


class SimpleConv(nn.Module):
    """A very simple convolutional MLP."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_depth: int,
        img_dims: tuple,
        num_layers: int,
        max_depth: int = 128,
    ):
        """Create a SimpleConv.

        Args:
            in_channels: The number of input channels.
            out_channels: The number of output channels.
            base_depth: The base convolutional filter depth, which increases by 2 at every Down block.
            img_dims: The height and width of images input into this network.
            num_blocks: The number of blocks in this network. Defaults to the max number of blocks possible.
            net_type: The type of U-Net. Should be "2d", "2dT", "2_1d", or "3d".
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_depth = base_depth
        self.img_dims = img_dims
        self.num_layers = num_layers
        self.max_depth = max_depth

        modules = [
            nn.Conv2d(self.in_channels, self.base_depth, kernel_size=1),
            nn.BatchNorm2d(self.base_depth),
            nn.ReLU(),
        ]
        for i in range(self.num_layers):
            in_depth = min(self.max_depth, self.base_depth * (2 ** i))
            out_depth = min(self.max_depth, self.base_depth * (2 ** (i + 1)))
            modules.append(nn.Conv2d(in_depth, out_depth, kernel_size=3, padding=1))
            modules.append(nn.BatchNorm2d(out_depth))
            modules.append(nn.ReLU())
        last_depth = min(self.max_depth, self.base_depth * (2 ** self.num_layers))
        modules.append(nn.Conv2d(last_depth, self.out_channels, kernel_size=1))
        self.seq = nn.Sequential(*modules)

    def forward(self, batch: dict) -> torch.Tensor:
        x = batch["input"]["sensor_input"]
        out = self.seq(x)
        out_interp = interp_atrain_output(batch, out)
        return out, out_interp


class SimpleConv3d(nn.Module):
    """TO DO"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        """TO DO"""
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        modules = [
            nn.Conv3d(self.in_channels, 64, kernel_size=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=(1, 1, 2), mode="trilinear"),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=(1, 1, 2), mode="trilinear"),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=(1, 1, 2), mode="trilinear"),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=(1, 1, 2), mode="trilinear"),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=(1, 1, 2), mode="trilinear"),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=(1, 1, 2), mode="trilinear"),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=(1, 1, 2), mode="trilinear"),
            nn.Conv3d(64, 1, kernel_size=3, padding=1),
            nn.BatchNorm3d(1),
            nn.ReLU(),
        ]
        self.seq = nn.Sequential(*modules)
        self.post_conv = nn.Conv2d(128, self.out_channels, kernel_size=1)

    def forward(self, batch: dict) -> torch.Tensor:
        x = batch["input"]["sensor_input"]
        x = x.unsqueeze(-1)
        x = self.seq(x)
        x = x.permute(0, 4, 2, 3, 1).squeeze().contiguous()
        out = self.post_conv(x)
        out_interp = interp_atrain_output(batch, out)
        return out, out_interp
