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

import numpy as np
from PIL import Image
from tqdm import tqdm


def main():
    print("Pre-processing the 95-Cloud dataset! This may take an hour or two, but will save time during training.")

    cloud95_dir = f"{os.path.dirname(__file__)}/../../data/95cloud"

    # check if symbolic link has been made, otherwise warn user
    if not os.path.exists(cloud95_dir):
        raise FileNotFoundError(
            "Could not find the 95-Cloud dataset. Did you follow the installation instructions in README.md?"
        )

    for subset, patch_filename in [
        ("38-Cloud_test", "test_patches_38-Cloud.csv"),
        ("38-Cloud_training", "training_patches_38-Cloud.csv"),
        ("95-cloud_training_only_additional_to38-cloud", "training_patches_95-cloud_additional_to_38-cloud.csv"),
    ]:
        if not os.path.exists(f"{cloud95_dir}/{subset}"):
            raise FileNotFoundError(
                f"Could not find the '{subset}' folder in your 95-Cloud data directory. Did you follow the installation"
                " instructions in README.md?"
            )

        subset_dir = f"{cloud95_dir}/{subset}"
        if "train" in subset:
            npy_dir = f"{subset_dir}/train_npy"
        else:
            npy_dir = f"{subset_dir}/test_npy"

        # if there's already the npy directory, remove it
        if os.path.exists(npy_dir):
            shutil.rmtree(npy_dir)
        os.mkdir(npy_dir)

        # read the patch list
        patch_names = np.array(open(f"{subset_dir}/{patch_filename}").read().split("\n")[1:-1])

        # load each image and save it to .npy
        print(f"{subset} patch list loaded, pre-processing:")
        for patch_name in tqdm(patch_names):
            # read and stack the 4 .TIF files into one numpy array
            channel_arrs = []
            split = "train" if "train" in subset_dir else "test"
            in95 = "additional" in subset_dir
            for channel in ["nir", "red", "green", "blue"]:
                if in95:
                    channel_path = f"{subset_dir}/{split}_{channel}_additional_to38cloud/{channel}_{patch_name}.TIF"
                else:
                    channel_path = f"{subset_dir}/{split}_{channel}/{channel}_{patch_name}.TIF"
                channel_arrs.append(np.array(Image.open(channel_path)))
            patch_arr = np.stack(channel_arrs, axis=2)
            # save it
            np.save(open(f"{npy_dir}/{patch_name}.npy", "wb"), patch_arr)
        print("\n")

    print("Done pre-processing the 95-Cloud dataset! Enjoy!")


main()
