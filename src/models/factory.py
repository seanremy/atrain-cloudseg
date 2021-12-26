"""Factory for segmentation models."""
import sys

if "./src" not in sys.path:
    sys.path.insert(0, "./src")  # TO DO: change this once it's a package
from models.simple_conv import SimpleConv, SimpleConv3d
from models.single_pixel import SinglePixel
from models.unet import UNet


def get_single_pixel(in_channels, out_channels, patch_size, args):
    return SinglePixel(in_channels, out_channels)


def get_simple_conv(in_channels, out_channels, patch_size, args):
    return SimpleConv(in_channels, out_channels, args.base_depth, patch_size, num_layers=5)


def get_simple_conv3d(in_channels, out_channels, patch_size, args):
    return SimpleConv3d(in_channels, out_channels)


def get_unet_2d(in_channels, out_channels, patch_size, args):
    num_blocks = args.num_blocks if args.num_blocks > 0 else None
    return UNet(in_channels, out_channels, args.base_depth, patch_size, net_type="2d", num_blocks=num_blocks)


def get_unet_2dT(in_channels, out_channels, patch_size, args):
    num_blocks = args.num_blocks if args.num_blocks > 0 else None
    return UNet(in_channels, out_channels, args.base_depth, patch_size, net_type="2dT", num_blocks=num_blocks)


def get_unet_2_1d(in_channels, out_channels, patch_size, args):
    num_blocks = args.num_blocks if args.num_blocks > 0 else None
    return UNet(in_channels, out_channels, args.base_depth, patch_size, net_type="2_1d", num_blocks=num_blocks)


def get_unet_3d(in_channels, out_channels, patch_size, args):
    num_blocks = args.num_blocks if args.num_blocks > 0 else None
    return UNet(in_channels, out_channels, args.base_depth, patch_size, net_type="3d", num_blocks=num_blocks)


model_factory = {
    "single_pixel": get_single_pixel,
    "simple_conv": get_simple_conv,
    "simple_conv3d": get_simple_conv3d,
    "unet_2d": get_unet_2d,
    "unet_2dT": get_unet_2dT,
    "unet_2_1d": get_unet_2_1d,
    "unet_3d": get_unet_3d,
}
