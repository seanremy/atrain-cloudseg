"""Generate a train/val split for the A-Train dataset."""

import argparse
import json
import os
import pickle
from collections import defaultdict

import numpy as np
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments.

    Returns:
        args: Command-line argument namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-name", required=True, type=str, help="Name of this split.")
    parser.add_argument("--split-ratio", type=float, default=0.8, help="Ratio of train/total instances.")
    args = parser.parse_args()
    assert len(args.split_name) > 0
    assert "." not in args.split_name
    assert args.split_ratio > 0 and args.split_ratio < 1
    return args


def main():
    args = parse_args()
    instance_info = json.load(open(os.path.join("data", "atrain", "instance_info.json")))

    # organize instances by orbit file datetimes
    inst_ids_by_datetime = defaultdict(list)
    for inst_id, inst in instance_info.items():
        inst_ids_by_datetime[inst["cal_par_datetime"]].append(inst_id)
    num_train_datetimes = int(args.split_ratio * len(inst_ids_by_datetime))

    # pick a subset of these datetimes, because we don't want overlap between train and val orbit files
    train_datetimes = np.random.choice(list(inst_ids_by_datetime.keys()), num_train_datetimes, replace=False)

    # Note: len(train_ids) / len(val_ids) is only APPROXIMATELY equal to args.split_ratio.
    # This is because we're splitting our dataset by datetime, and not by instances.
    train_ids = [int(inst_id) for dt in train_datetimes for inst_id in inst_ids_by_datetime[dt]]
    val_ids = [int(inst_id) for inst_id in instance_info if int(inst_id) not in set(train_ids)]

    # we also need to get class counts in our training set, in case we want to use them in the loss
    cls_counts = np.zeros(9)  # 9 classes: 8 cloud types, and we use 0 for no cloud
    mask_count, total_pixels = 0, 0
    pbar = tqdm(train_ids)
    pbar.set_description("Counting labels in each class")
    for tid in pbar:
        out = pickle.load(open(f"data/atrain/{instance_info[str(tid)]['output_path']}", "rb"))
        cs_arr = out["cloud_scenario"]["cloud_scenario"]
        mask = cs_arr.any(axis=1)
        cls_counts += np.array([np.sum(cs_arr == i) for i in range(9)])
        mask_count += np.sum(mask)
        total_pixels += mask.size
    cls_counts = [int(c) for c in cls_counts]
    mask_count = int(mask_count)
    total_pixels = int(total_pixels)

    # save it all to file
    train_counts = {"cls_counts": cls_counts, "mask_count": mask_count, "total_pixels": total_pixels}
    split = {"train": sorted(train_ids), "val": sorted(val_ids), "train_counts": train_counts}
    split_file = os.path.join("data", "atrain", f"{args.split_name}.json")
    json.dump(split, open(split_file, "w"))

    # write some output to console
    tqdm.write(f"Successfully saved split to {split_file}.")
    tqdm.write(f"\tTrain instances:\t{len(train_ids)}")
    tqdm.write(f"\tVal instances:\t\t{len(val_ids)}")
    tqdm.write(f"\tClass counts (train):\t{cls_counts}")
    tqdm.write(f"\tCloudy pixels (train):\t{mask_count}")
    tqdm.write(f"\tTotal pixels (train):\t{total_pixels}")


if __name__ == "__main__":
    main()
