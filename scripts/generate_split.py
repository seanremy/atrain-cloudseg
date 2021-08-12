"""Generate a train/val split for the A-Train dataset."""

import argparse
import json
import os
from collections import defaultdict

import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments.

    Returns:
        args: Command-line argument namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-name", required=True, type=str, help="Name of this split.")
    parser.add_argument("--split-ratio", type=float, default=0.8, help="Ratio of train:val instances.")
    args = parser.parse_args()
    assert len(args.split_name) > 0
    assert "." not in args.split_name
    assert args.split_ratio > 0 and args.split_ratio < 1
    return args


def main():
    args = parse_args()
    instance_info = json.load(open(os.path.join("data", "atrain", "instance_info.json")))

    inst_ids_by_datetime = defaultdict(list)
    for inst_id, inst in instance_info.items():
        inst_ids_by_datetime[inst["cal_par_datetime"]].append(inst_id)

    num_train_datetimes = int(args.split_ratio * len(inst_ids_by_datetime))
    train_datetimes = np.random.choice(list(inst_ids_by_datetime.keys()), num_train_datetimes, replace=False)
    train_ids = [int(inst_id) for dt in train_datetimes for inst_id in inst_ids_by_datetime[dt]]
    val_ids = [int(inst_id) for inst_id in instance_info if int(inst_id) not in set(train_ids)]
    split = {"train": sorted(train_ids), "val": sorted(val_ids)}
    split_file = os.path.join("data", "atrain", f"{args.split_name}.json")
    json.dump(split, open(split_file, "w"))
    print(f"Successfully saved split to {split_file}.")
    print(f"Train instances: {len(train_ids)}")
    print(f"Val instances: {len(val_ids)}")


if __name__ == "__main__":
    main()
