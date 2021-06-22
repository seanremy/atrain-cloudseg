"""Preprocess 95-Cloud

This script preprocesses the 95-Cloud images into a format more readily usable during data loading. Each 95-Cloud image
is saved as 4 separate .TIF files, 1 per wavelength (nir, red, green, blue). The pre-processing involves loading the
images, stacking them into WxHx4 numpy arrays, and saving them as .npy files. For details on why this is important,
please see src/scripts/notebooks/95cloud_dataloader_speed_test.ipynb.

This script assumes that you have downloaded the 38-Cloud dataset and the 95-Cloud dataset, and have created a symbolic
link in the data/ folder, as explained in README.md.
"""

import os
import shutil
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm


def main():
    print("Pre-processing the 95-Cloud dataset! This may take an hour or two, but will save time during training.")

    cloud95_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "95cloud")

    # check if symbolic link has been made, otherwise warn user
    if not os.path.exists(cloud95_path):
        raise FileNotFoundError(
            "Could not find the 95-Cloud dataset. Did you follow the installation instructions in README.md?"
        )

    for subset_dir, patch_filename in [
        ("38-Cloud_test", "test_patches_38-Cloud.csv"),
        ("38-Cloud_training", "training_patches_38-Cloud.csv"),
        ("95-cloud_training_only_additional_to38-cloud", "training_patches_95-cloud_additional_to_38-cloud.csv"),
    ]:
        subset_path = os.path.join(cloud95_path, subset_dir)
        if not os.path.exists(subset_path):
            raise FileNotFoundError(
                f"Could not find the '{subset_dir}' folder in your 95-Cloud data directory. Did you follow the installation"
                " instructions in README.md?"
            )

        split = "train" if "train" in subset_dir else "test"
        if split == "train":
            npy_path = os.path.join(subset_path, "train_npy")
        else:
            npy_path = os.path.join(subset_path, "test_npy")

        # if there's already the npy directory, remove it
        if os.path.exists(npy_path):
            shutil.rmtree(npy_path)
        os.mkdir(npy_path)

        # read the patch list
        patch_names = np.array(open(os.path.join(subset_path, patch_filename)).read().split("\n")[1:-1])
        nonempty_patches = []

        in95 = "additional" in subset_dir  # if this is 95-Cloud or 38-Cloud

        # load each image and save it to .npy
        print(f"{subset_dir} | patch list loaded, pre-processing images.")
        for patch_name in tqdm(patch_names, file=sys.stdout):
            # read and stack the 4 .TIF files into one numpy array
            channel_arrs = []
            for channel in ["nir", "red", "green", "blue"]:
                if in95:
                    channel_path = os.path.join(
                        subset_path, f"{split}_{channel}_additional_to38cloud", f"{channel}_{patch_name}.TIF"
                    )
                else:
                    channel_path = os.path.join(subset_path, f"{split}_{channel}", f"{channel}_{patch_name}.TIF")
                channel_arrs.append(np.array(Image.open(channel_path)))
            patch_arr = np.stack(channel_arrs, axis=2)
            # if the image isn't empty, save it and add it to the non-empty patches
            if not (patch_arr == 0).all():
                np.save(open(os.path.join(npy_path, f"{patch_name}.npy"), "wb"), patch_arr)
                nonempty_patches.append(patch_name)
        # write a file of the non-empty patches
        open(os.path.join(subset_path, "nonempty_patches.csv"), "w").write("\n".join(nonempty_patches))
        # if this is a training set, then also save the GT as numpy arrays
        if split == "train":
            print(f"{subset_dir} | pre-processing ground truth:")
            gt_dir = "train_gt_additional_to38cloud" if in95 else "train_gt"
            gt_path = os.path.join(subset_path, gt_dir)
            for patch_name in tqdm(nonempty_patches, file=sys.stdout):
                gt_arr = np.array(Image.open(os.path.join(gt_path, f"gt_{patch_name}.TIF")))
                np.save(open(os.path.join(npy_path, f"gt_{patch_name}.npy"), "wb"), gt_arr)
        print("\n")

    print("Done pre-processing the 95-Cloud dataset! Enjoy!")


main()
