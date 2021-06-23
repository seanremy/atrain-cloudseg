"""38-Cloud and its extension (95-Cloud) are datasets of Landsat-8 imagery and pixel-level cloud labels for the purpose
of cloud segmentation. The datasets consist of Landsat-8 scenes split into 384x384 patches. The scenes have 4 spectral
channels: NIR, red, green, and blue. This dataset implementation assumes the user has already run the pre-processing
scripts to combine these channels into numpy arrays, as specified in README.md.

The 38-Cloud dataset: https://github.com/SorourMo/38-Cloud-A-Cloud-Segmentation-Dataset
The 95-Cloud dataset: https://github.com/SorourMo/95-Cloud-An-Extension-to-38-Cloud-Dataset
"""

import os
import warnings

import numpy as np
from torch.utils.data import Dataset


class Cloud38(Dataset):
    """The 38-Cloud dataset has 5587 non-empty training patches, 5155 of which are at least 20% non-empty, and 6205
    testing patches."""

    def __init__(self, subset: str, ignore_mostly_empty: bool = True):
        """Create a Cloud38 dataset.

        Args:
            subset: Either 'train' or 'test'.
            ignore_mostly_empty: Whether to skip patches that are more than 80% empty.
        """
        super().__init__()
        # fixed values
        self.dataset_root = os.path.join(f"{os.path.dirname(__file__)}", "..", "..", "data", "95cloud")
        # params
        self.subset = subset
        self.ignore_mostly_empty = ignore_mostly_empty
        # set all of the paths correctly based on train/test
        if self.subset == "train":
            self.subdir = "38-Cloud_training"
        elif self.subset == "test":
            self.subdir = "38-Cloud_test"
        else:
            raise ValueError(f"Value of subset must be 'train' or 'test', but got: '{self.subset}'")
        # Check the symlink to the dataset has been created
        if not os.path.exists(self.dataset_root):
            raise FileNotFoundError(
                f"Could not find the 95-Cloud dataset at '{self.dataset_root}'. Did you follow the installation "
                "instructions in README.md?"
            )
        self.npy_path = os.path.join(self.dataset_root, self.subdir, f"{self.subset}_npy")
        # Check the dataset has been pre-processed, and the npy folder exists
        if not os.path.exists(self.npy_path):
            raise FileNotFoundError(
                f"Could not find the '{self.subset}_npy' directory in '{os.path.join(self.dataset_root, self.subdir)}'."
                " Did you run the pre-processing script, as instructed in README.md?"
            )
        # read the file of non-empty patches
        if self.ignore_mostly_empty and self.subset == "train":
            self.patches_filename = "../training_patches_38-cloud_nonempty.csv"
        else:
            self.patches_filename = "nonempty_patches.csv"
        self.patches_filepath = os.path.join(self.dataset_root, self.subdir, self.patches_filename)
        self.patches = open(self.patches_filepath).read().split("\n")[1:-1]
        self.num_patches = len(self.patches)
        # warn the user if the number of patches is different than what we expect
        if self.subset == "train":
            expected_num_patches = 5155 if self.ignore_mostly_empty else 5587
        else:
            expected_num_patches = 6205

        if self.num_patches != expected_num_patches:
            warnings.warn(
                f"Expected to find {expected_num_patches} patches in '{self.patches_filepath}', but found "
                f"{self.num_patches} patches."
            )

    def __len__(self):
        """Get the length of this dataset."""
        return self.num_patches

    def __getitem__(self, idx: int):
        """Get the item at the specified index."""
        patch_name = self.patches[idx]
        item = {
            "patch_name": patch_name,
            "input": np.load(os.path.join(self.npy_path, f"{patch_name}.npy")),
            "idx": idx,
        }
        if self.subset == "train":
            item["gt"] = np.load(os.path.join(self.npy_path, f"gt_{self.patches[idx]}.npy"))
        return item


class Cloud95(Cloud38):
    """The 95-Cloud dataset adds 17700 more non-empty training patches to 38-Cloud. Of those, 16347 have at least 20%
    non-empty pixels."""

    def __init__(self, subset: str, ignore_mostly_empty: bool = True):
        """Create a Cloud95 dataset.

        Args:
            subset: Either 'train' or 'test'.
            ignore_mostly_empty: Whether to skip patches that are more than 80% empty.
        """
        super().__init__(subset, ignore_mostly_empty)
        # if this is a test set, no additional steps needed
        if subset == "test":
            return
        # add the additional cloud95 files
        self.subdir_95 = "95-cloud_training_only_additional_to38-cloud"
        self.npy_path_95 = os.path.join(self.dataset_root, self.subdir_95, f"train_npy")
        if self.ignore_mostly_empty:
            self.patches_filename = "training_patches_95-cloud_nonempty.csv"
        self.patches_filepath_95 = os.path.join(self.dataset_root, self.subdir_95, self.patches_filename)
        self.patches_95 = open(self.patches_filepath_95).read().split("\n")[1:-1]
        self.patches_95 = [p for p in self.patches_95 if p not in set(self.patches)]  # remove the 38cloud patches
        self.num_patches_38 = self.num_patches
        self.num_patches += len(self.patches_95)

        # warn the user if the number of patches is different than what we expect
        expected_num_patches_95 = 16347 if self.ignore_mostly_empty else 17700
        if len(self.patches_95) != expected_num_patches_95:
            warnings.warn(
                f"Expected to find {expected_num_patches_95} patches in '{self.patches_filepath_95}', but found "
                f"{len(self.patches_95)} patches."
            )

    def __getitem__(self, idx: int):
        """Get the item at the specified index."""
        if idx < self.num_patches_38:
            return super().__getitem__(idx)
        patch_name = self.patches_95[idx - self.num_patches_38]
        item = {
            "patch_name": patch_name,
            "input": np.load(os.path.join(self.npy_path_95, f"{patch_name}.npy")),
            "idx": idx,
        }
        if self.subset == "train":
            item["gt"] = np.load(os.path.join(self.npy_path_95, f"gt_{patch_name}.npy"))
        return item
