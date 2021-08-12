import torch
import torch.nn as nn


class SinglePixel(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_layers: int = 3, mid_layer_depth=None) -> None:
        """Create a Single Pixel model.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            num_layers: Number of layers, defaults to 3.
        """
        super().__init__()
        assert num_layers >= 1
        self.in_channels = in_channels
        self.out_channels = out_channels
        if mid_layer_depth is None:
            self.mid_layer_depth = max(self.in_channels, self.out_channels)
        else:
            self.mid_layer_depth = mid_layer_depth
        self.num_layers = num_layers
        self.layers = [
            nn.Conv2d(self.in_channels, self.mid_layer_depth, kernel_size=1),
            nn.BatchNorm2d(self.mid_layer_depth),
            nn.ReLU(),
        ]
        for _ in range(self.num_layers - 2):
            self.layers.append(nn.Conv2d(self.mid_layer_depth, self.mid_layer_depth, kernel_size=1))
            self.layers.append(nn.BatchNorm2d(self.mid_layer_depth))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Conv2d(self.mid_layer_depth, self.out_channels, kernel_size=1))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
