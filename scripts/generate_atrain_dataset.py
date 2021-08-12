"""Script for generating the A-Train Cloud Dataset.

This script combines data from PARASOL/POLDER, CALIPSO/CALIOP, and CloudSat/CLDCLASS to make a
machine-learning-friendly dataset of time-synced, location-synced input-output pairs. Input consists of projected
multi-angle polarimetric imagery from PARASOL/POLDER. Output consists of sparse CLDCLASS labels.
"""

import argparse
import json
import os
import pickle
import sys
import warnings
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
from tqdm import tqdm

if "./src" not in sys.path:
    sys.path.insert(0, "./src")  # TO DO: change this once it's a package
from utils.atrain import CALPARScene, CLDCLASSScene, PARASOLScene, tai93_string_to_datetime
from utils.icare import ICARESession, datetime_to_subpath
from utils.parasol_fields import FIELD_DICT


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments.

    Returns:
        args: Command-line argument namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--atrain_dir", type=str, required=True, help="Path to the directory where you want to make the dataset."
    )
    parser.add_argument(
        "--temp_dir", type=str, required=True, help="Path to the temporary directory for the ICARE FTP cache."
    )
    parser.add_argument("--resume", action="store_true", help="Resume from where script left off.")
    parser.add_argument(
        "--time_match_threshold",
        type=int,
        default=600,
        help="Maximum acceptable time offset (in seconds) between POLDER and CALIOP beginning of acquisition.",
    )
    parser.add_argument(
        "--days_per_month",
        type=int,
        default=1000,
        help="Number of days to process per month. Default is arbitrarily high at 1000.",
    )
    parser.add_argument(
        "--files_per_day",
        type=int,
        default=1000,
        help="Number of files to process per day. There are usually 14-15 files. Default is arbitrarily high at 1000.",
    )
    parser.add_argument(
        "--samples_per_file", type=int, default=4, help="Goal number of samples to take per orbit file."
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=100,
        help="Patch size, in pixels, of samples to take. Raising this number has a severe effect on "
        "the size of the dataset on disk.",
    )
    parser.add_argument(
        "--nadir_padding",
        type=int,
        default=10,
        help="Padding, in pixels, to keep between the nadir line and each side of the image patch.",
    )
    parser.add_argument(
        "--min_parasol_views",
        type=int,
        default=13,
        help="Minimum number of angles that need to be available for every pixel in a patch for it to be considered "
        "valid. Raising this number may cause strange sampling artifacts, and lowering it may reduce data quality.",
    )
    parser.add_argument(
        "--field_scheme",
        type=str,
        default="default",
        help="Which scheme of PARASOL fields to use. Fields can be found in 'utils/parasol_fields.py'",
    )
    args = parser.parse_args()
    assert args.time_match_threshold > 0
    if args.time_match_threshold < 300 or args.time_match_threshold > 1200:
        warnings.warn(
            f"Time match threshold of {args.time_match_threshold} falls outside of the typical range "
            "(300-1200). You may notice strange file-matching behavior as a result."
        )
    assert args.files_per_day > 0
    assert args.samples_per_file > 0
    assert args.patch_size > 0
    if args.patch_size > 100:
        warnings.warn(
            f"Patch size of {args.patch_size} is quite large. Remember that 'images' will have 240 channels. You may "
            "notice significant disk usage as a result."
        )
    assert args.nadir_padding >= 0 and args.nadir_padding < args.patch_size // 2
    assert args.min_parasol_views > 0 and args.min_parasol_views <= 16
    assert args.field_scheme.lower() in FIELD_DICT
    return args


def main() -> None:

    args = parse_args()

    icare_ses = ICARESession(args.temp_dir)
    sync_subdir = icare_ses.SUBDIR_LOOKUP["SYNC"]
    cldclass_subdir = icare_ses.SUBDIR_LOOKUP["CLDCLASS"]
    par_subdir = icare_ses.SUBDIR_LOOKUP["PAR"]
    par_fields = FIELD_DICT[args.field_scheme.lower()]

    # set a seed for reproducibility. Note that resuming the script will break reproducibility
    seed = np.random.randint(1e6)
    np.random.seed(seed)

    # the time window where PARASOL, CALIOP, and CloudSat were all operational, before later orbital corrections:
    start_date = datetime(year=2007, month=11, day=27)
    end_date = datetime(year=2009, month=12, day=2)

    # if resuming, pick up where we left off
    if args.resume and not os.path.exists(os.path.join(args.atrain_dir, "instance_info.json")):
        args.resume = False
    if args.resume:
        dataset_generation_info = json.load(open(os.path.join(args.atrain_dir, "dataset_generation_info.json")))
        instance_info = json.load(open(os.path.join(args.atrain_dir, "instance_info.json"), "r"))
        instance_info = {int(k): v for k, v in instance_info.items()}
        inst_id_ctr = max(list(instance_info.keys())) + 1
        last_datetime = datetime.strptime(instance_info[inst_id_ctr - 1]["cal_par_datetime"], "%Y-%m-%d %H:%M:%S")
        start_date = datetime(year=last_datetime.year, month=last_datetime.month, day=last_datetime.day)
        date_list = [datetime.strptime(d, "%Y_%m_%d") for d in dataset_generation_info["date_list"]]
        date_list = [d for d in date_list if d >= last_datetime]
    else:
        instance_info = {}
        inst_id_ctr = 0

        # get which days we're going to use
        day_idxs_per_yearmonth = defaultdict(list)
        for day_idx in range((end_date - start_date).days + 1):
            date = start_date + timedelta(days=day_idx)
            day_idxs_per_yearmonth[f"{date.year}_{date.month}"].append(day_idx)
        date_list = []
        for yearmonth in day_idxs_per_yearmonth:
            num_sample_days = min(len(day_idxs_per_yearmonth[yearmonth]), args.days_per_month)
            day_idxs_from_month = np.random.choice(day_idxs_per_yearmonth[yearmonth], num_sample_days, replace=False)
            date_list += [start_date + timedelta(days=int(day_idx)) for day_idx in day_idxs_from_month]
        dataset_generation_info = {
            "args": vars(args),
            "seed": seed,
            "par_fields": par_fields,
            "date_list": [f"{d.year}_{d.month}_{d.day}" for d in date_list],
        }
        os.makedirs(args.atrain_dir, exist_ok=True)
        json.dump(dataset_generation_info, open(os.path.join(args.atrain_dir, "dataset_generation_info.json"), "w"))

    pbar = tqdm(date_list, position=0, leave=True)
    for date in pbar:
        date_subpath = datetime_to_subpath(date)

        # Loop over files in this day
        date_sync_subdir = os.path.join(sync_subdir, date_subpath)
        try:
            cal_par_filenames = icare_ses.listdir(date_sync_subdir)
        except FileNotFoundError:
            print(f"No CALIPSO/PARASOL sync directory at: {date_sync_subdir}, continuing.")
            continue
        cal_par_datetimes = [tai93_string_to_datetime(fn.split(".")[0].split("_")[3]) for fn in cal_par_filenames]

        # get indices of all files that fall within the valid datetime range
        cal_par_sort_order = []
        for i in range(len(cal_par_datetimes)):
            if not args.resume or cal_par_datetimes[i] > last_datetime:
                cal_par_sort_order.append(i)
        cal_par_sort_order = sorted(cal_par_sort_order, key=lambda idx: cal_par_datetimes[idx])

        # take a subsample of these files
        num_files_today = min(len(cal_par_sort_order), args.files_per_day)
        cal_par_idxs = sorted(np.random.choice(len(cal_par_sort_order), num_files_today, replace=False))

        # loop over the files we picked
        for cp_idx in cal_par_idxs:
            cal_par_filename = cal_par_filenames[cp_idx]
            cal_par_datetime = cal_par_datetimes[cp_idx]
            pbar.set_description(f"Processing {cal_par_filename}")  # progress bar info
            cal_par_filepath = os.path.join(sync_subdir, date_subpath, cal_par_filename)
            local_cal_par_filepath = icare_ses.get_file(cal_par_filepath)

            # we get the matching CLDCLASS file from the merged data filename
            cldclass_filename = cal_par_filename.replace("PAR-RB2", "CS-2B-CLDCLASS")
            cldclass_filepath = os.path.join(cldclass_subdir, date_subpath, cldclass_filename)
            try:
                local_cldclass_filepath = icare_ses.get_file(cldclass_filepath)
            except FileNotFoundError:
                print(f"No CLDCLASS data available at '{cldclass_filepath}', continuing.")
                continue  # if there is no available CLDCLASS data, move on

            # read the CALIPSO/PARASOL sync data scene
            cal_par_scene = CALPARScene(local_cal_par_filepath)

            # we can get all of the PARASOL files within the acceptable time range
            par_time_window = (
                cal_par_scene.acquisition_range[0] - timedelta(seconds=args.time_match_threshold),
                cal_par_scene.acquisition_range[0] + timedelta(seconds=args.time_match_threshold),
            )

            # convert the par time window into directory names
            sameday_par_dirs = set([os.path.join(par_subdir, datetime_to_subpath(dt)) for dt in par_time_window])

            # get all of the parasol files in those directories, and load the best one
            sameday_par_filepaths = []
            for dir in sameday_par_dirs:
                try:
                    dir_contents = icare_ses.listdir(dir)
                except FileNotFoundError:
                    print(f"No PARASOL directory at: '{dir}', continuing.")
                    continue
                for par_file in dir_contents:
                    sameday_par_filepaths.append(os.path.join(dir, par_file))
            sameday_par_filepaths = sorted(set(sameday_par_filepaths))
            par_filepath = cal_par_scene.get_best_parasol_filepath(sameday_par_filepaths, args.time_match_threshold)
            if par_filepath is None:
                print("No PARASOL file found within the time window, continuing.")
                continue  # if the time difference is too large
            local_par_filepath = os.path.join(icare_ses.temp_dir, icare_ses.get_file(par_filepath))

            # read the CLDCLASS scene and make sure there's actually data
            cldclass_scene = CLDCLASSScene(local_cldclass_filepath)
            if not cldclass_scene.cloud_scenario["cloud_scenario"].any():
                print(f"CLDCLASS Scene at {cldclass_filepath} has no cloud scenario data, continuing.")
                continue

            # read the PARASOL and CLDCLASS scenes
            par_scene = PARASOLScene(local_par_filepath, par_fields, min_views=args.min_parasol_views)

            # get the validity masks, multiply them
            view_validity = par_scene.get_view_validity(args.patch_size)
            nadir_validity = cldclass_scene.get_nadir_validity(par_scene, args.patch_size, args.nadir_padding)
            validity = view_validity * nadir_validity
            valid_y, valid_x = np.where(validity)

            # sample square patches from the scene
            num_samples = min(args.samples_per_file, valid_y.shape[0])
            for rand_idx in np.random.choice(valid_y.shape[0], num_samples, replace=False):
                rand_y, rand_x = valid_y[rand_idx], valid_x[rand_idx]

                # get patch array and labels
                patch_arr, patch_box = par_scene.get_patch(rand_x, rand_y, args.patch_size)
                patch_labels = cldclass_scene.get_patch_labels(par_scene, patch_box)
                patch_labels["instance_id"] = inst_id_ctr

                # get the paths
                input_dir = os.path.join("input", date_subpath)
                output_dir = os.path.join("output", date_subpath)
                for d in [input_dir, output_dir]:
                    os.makedirs(os.path.join(args.atrain_dir, d), exist_ok=True)
                input_path = os.path.join(input_dir, f"{inst_id_ctr}.npy")
                output_path = os.path.join(output_dir, f"{inst_id_ctr}.pkl")

                instance_info[inst_id_ctr] = {
                    "cal_par_file": cal_par_scene.filepath,
                    "par_file": par_scene.filepath,
                    "cldclass_file": cldclass_scene.filepath,
                    "instance id": inst_id_ctr,
                    "input_path": input_path,
                    "output_path": output_path,
                    "cal_par_datetime": str(cal_par_datetime),
                }
                inst_id_ctr += 1
                np.save(open(os.path.join(args.atrain_dir, input_path), "wb"), patch_arr)
                pickle.dump(patch_labels, open(os.path.join(args.atrain_dir, output_path), "wb"))
                json.dump(instance_info, open(os.path.join(args.atrain_dir, "instance_info.json"), "w"))

    icare_ses.cleanup()


if __name__ == "__main__":
    main()
