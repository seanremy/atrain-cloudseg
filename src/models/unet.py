"""A U-Net-like model, slightly modified to better work for the high number of channels in many satellite datasets.

Some code based on the following:
- github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
- github.com/proceduralia/pytorch-conv2_1d/blob/master/conv2_1d.py
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

    def __init__(self, in_channels: int, out_channels: int, use_conv_transpose: bool = False) -> None:
        """Create an Up block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
        """
        super().__init__()
        self.use_conv_transpose = use_conv_transpose
        if self.use_conv_transpose:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=3),
                nn.BatchNorm2d(in_channels // 2),
                nn.ReLU(),
            )
        else:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels // 2)
        self.conv2 = nn.Conv2d(in_channels // 2, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

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


class Conv2_1d(nn.Module):
    """ "Separated (2+1)D convolution."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size=3, padding=0) -> None:
        """Create a conv(2+1)D layer.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the kernel, can be an int or a length 3 tuple.
            padding: Padding to apply to the input tensor, can be an int or a length 3 tuple.
        """
        super().__init__()

        if type(kernel_size) == int:
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if type(padding) == int:
            padding = (padding, padding, padding)

        self.conv2d = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size[1:], padding=padding[2:])
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size[0], padding=padding[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, alt, lat, lon = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(b * alt, c, lat, lon)
        x = self.conv2d(x)

        _, c, lat, lon = x.size()
        x = x.view(b, alt, c, lat, lon)
        x = x.permute(0, 3, 4, 2, 1).contiguous()
        x = x.view(b * lat * lon, c, alt)
        x = self.conv1d(x)

        _, c, alt = x.size()
        x = x.view(b, lat, lon, c, alt)
        x = x.permute(0, 3, 4, 1, 2).contiguous()

        return x


class ConvTranspose2_1d(nn.Module):
    """ "Separated (2+1)D transpose convolution."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size=3, padding=0) -> None:
        """Create a transpose conv(2+1)D layer.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the kernel, can be an int or a length 3 tuple.
            padding: Padding to apply to the input tensor, can be an int or a length 3 tuple.
        """
        super().__init__()

        if type(kernel_size) == int:
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if type(padding) == int:
            padding = (padding, padding, padding)

        self.conv2d = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=kernel_size[1:], padding=padding[1:])
        self.conv1d = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=kernel_size[0], padding=padding[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, alt, lat, lon = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(b * alt, c, lat, lon)
        x = self.conv2d(x)

        _, c, lat, lon = x.size()
        x = x.view(b, alt, c, lat, lon)
        x = x.permute(0, 3, 4, 2, 1).contiguous()
        x = x.view(b * lat * lon, c, alt)
        x = self.conv1d(x)

        _, c, alt = x.size()
        x = x.view(b, lat, lon, c, alt)
        x = x.permute(0, 3, 4, 1, 2).contiguous()

        return x


class Up3d(nn.Module):
    """Upsampling block in a 3D U-Net."""

    def __init__(self, in_channels: int, out_channels: int, sep2_1: bool = False) -> None:
        """Create an Up3d block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            sep2_1: Whether or not to use separable 2+1 convolutions.
        """
        super().__init__()
        self.sep2_1 = sep2_1
        if self.sep2_1:
            self.up = ConvTranspose2_1d(in_channels // 2, in_channels // 2, kernel_size=3)
            self.conv1 = Conv2_1d(in_channels, in_channels // 2, kernel_size=3, padding=1)
            self.conv2 = Conv2_1d(in_channels // 2, out_channels, kernel_size=3, padding=1)
        else:
            self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=3)
            self.conv1 = nn.Conv3d(in_channels, in_channels // 2, kernel_size=3, padding=1)
            self.conv2 = nn.Conv3d(in_channels // 2, out_channels, kernel_size=3, padding=1)

        self.bn_up = nn.BatchNorm3d(in_channels // 2)
        self.bn1 = nn.BatchNorm3d(in_channels // 2)
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        x1 = self.bn_up(x1)
        x1 = F.relu(x1)
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])
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
        self,
        in_channels: int,
        out_channels: int,
        base_depth: int,
        img_dims: tuple,
        num_blocks: int = 4,
        net_type: str = "2d",
    ) -> None:
        """Create a U-Net.

        Args:
            in_channels: The number of input channels.
            out_channels: The number of output channels.
            base_depth: The base convolutional filter depth, which increases by 2 at every Down block.
            img_dims: The height and width of images input into this network.
            num_blocks: The number of blocks in this network. Defaults to the max number of blocks possible.
            net_type: The type of U-Net. Should be "2d", "2dT", "2_1d", or "3d".
        """
        super().__init__()
        # 2d: default. 2dT: replace upsample with convtranspose. 2_1d: separable 2+1D convs. 3d: 3D convs.
        assert net_type in ["2d", "2dT", "2_1d", "3d"]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_depth = base_depth
        self.img_dims = img_dims
        # as many layers as can be fit with input at least 1x1
        self.num_blocks = int(np.log(min(self.img_dims)) / np.log(2))
        if not num_blocks is None:
            self.num_blocks = min(num_blocks, self.num_blocks)
        self.net_type = net_type

        self.pre_conv_a = nn.Conv2d(self.in_channels, self.base_depth, kernel_size=1)
        self.pre_conv_b = nn.Conv2d(self.in_channels, self.base_depth // 2, kernel_size=1)

        self.down_blocks = []
        for i in range(self.num_blocks - 1):
            self.down_blocks.append(Down(self.base_depth * (2 ** i), self.base_depth * (2 ** (i + 1))))
        self.down_blocks.append(
            Down(self.base_depth * (2 ** (self.num_blocks - 1)), self.base_depth * (2 ** (self.num_blocks - 1)))
        )
        self.down_blocks = nn.Sequential(*self.down_blocks)
        self.up_blocks = []
        if self.net_type in ["2_1d", "3d"]:
            self.up_alt_scales = [self.out_channels // (2 ** i) for i in range(1, self.num_blocks + 1)][::-1]
        for i in range(self.num_blocks, 0, -1):
            if self.net_type in ["2d", "2dT"]:
                use_conv_transpose = self.net_type == "2dT"
                self.up_blocks.append(
                    Up(int(self.base_depth * (2 ** i)), int(base_depth * (2 ** (i - 2))), use_conv_transpose)
                )
            elif self.net_type in ["2_1d", "3d"]:
                sep2_1 = self.net_type == "2_1d"
                self.up_blocks.append(
                    Up3d(int(self.base_depth * (2 ** i)), int(base_depth * (2 ** (i - 2))), sep2_1=sep2_1)
                )
        self.up_blocks = nn.Sequential(*self.up_blocks)

        if self.net_type in ["2d", "2dT"]:
            self.post_conv = nn.Conv2d(self.base_depth, out_channels, kernel_size=1)
        elif self.net_type in ["2_1d", "3d"]:
            self.post_conv = nn.Conv3d(self.base_depth // 2, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_down = [F.relu(self.pre_conv_a(x))]
        x_skip = F.relu(self.pre_conv_b(x))
        for db in self.down_blocks:
            x_down.append(db(x_down[-1]))
        x_up = x_down.pop(-1)
        if self.net_type in ["2_1d", "3d"]:
            x_up = x_up.unsqueeze(2)

        for i in range(len(self.up_blocks)):
            ub = self.up_blocks[i]
            if self.net_type in ["2d", "2dT"]:
                x_up = ub(x_up, x_down.pop(-1))
            elif self.net_type in ["2_1d", "3d"]:
                x2 = x_down.pop(-1).unsqueeze(2)
                x2 = F.interpolate(
                    x2, scale_factor=(self.up_alt_scales[i], 1, 1), mode="trilinear", align_corners=False
                )
                x_up = ub(x_up, x2)
        if self.net_type in ["2d", "2dT"]:
            x = torch.cat([x_up, x_skip], dim=1)
            x = self.post_conv(x)
        elif self.net_type in ["2_1d", "3d"]:
            x_skip = F.interpolate(
                x_skip.unsqueeze(2),
                scale_factor=(self.out_channels - self.up_alt_scales[-1], 1, 1),
                mode="trilinear",
                align_corners=False,
            )
            x = torch.cat([x_up, x_skip], dim=2)
            x = self.post_conv(x)
            x = x.squeeze(1)
        return x
