import sys

if "./src" not in sys.path:
    sys.path.insert(0, "./src")  # TO DO: change this once it's a package
from losses.segmentation import *

loss_factory = {
    "cross-entropy": CrossEntropyLoss,
    "focal": FocalLoss,
    "soft-jaccard": SoftJaccardLoss,
}
