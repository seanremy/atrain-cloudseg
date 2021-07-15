"""Script for generating the A-Train Cloud Dataset.

This script combines data from PARASOL/POLDER, CALIPSO/CALIOP, and CloudSat/CLDCLASS to make a
machine-learning-friendly dataset of time-synced, location-synced input-output pairs. Input consists of projected
multi-angle polarimetric imagery from PARASOL/POLDER. Output consists of sparse CLDCLASS labels.
"""

import argparse
import json
import os
import pdb
import pickle
import sys
import warnings
from datetime import datetime, timedelta

import numpy as np
from tqdm import tqdm

if "./src" not in sys.path:
    sys.path.insert(0, "./src")  # TO DO: change this once it's a package
from utils.atrain import CALPARScene, CLDCLASSScene, PARASOLScene, tai93_string_to_datetime
from utils.icare import ICARESession, datetime_to_subpath
from utils.parasol_fields import FIELD_DICT


def parse_args():
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
    parser.add_argument("--field_scheme", type=str, default="default", help="Which scheme of PARASOL fields to use.")
    args = parser.parse_args()
    assert args.time_match_threshold > 0
    if args.time_match_threshold < 300 or args.time_match_threshold > 1200:
        warnings.warn(
            f"Time match threshold of {args.time_match_threshold} falls outside of the typical range "
            "(300-1200). You may notice strange file-matching behavior as a result."
        )
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


def main():

    args = parse_args()

    icare_ses = ICARESession(args.temp_dir)
    sync_subdir = icare_ses.SUBDIR_LOOKUP["SYNC"]
    cldclass_subdir = icare_ses.SUBDIR_LOOKUP["CLDCLASS"]
    par_subdir = icare_ses.SUBDIR_LOOKUP["PAR"]
    par_fields = FIELD_DICT[args.field_scheme.lower()]

    # set a seed for reproducibility. Note that resuming the script will break reproducibility
    seed = np.random.randint(1e6)
    np.random.seed(seed)

    # if resuming, pick up where we left off
    if args.resume and not os.path.exists(os.path.join(args.atrain_dir, "instance_info.json")):
        args.resume = False
    if args.resume:
        instance_info = json.load(open(os.path.join(args.atrain_dir, "instance_info.json"), "r"))
        instance_info = {int(k): v for k, v in instance_info.items()}
        inst_id_ctr = max(list(instance_info.keys())) + 1
        last_datetime = datetime.strptime(instance_info[inst_id_ctr - 1]["cal_par_datetime"], "%Y-%m-%d %H:%M:%S")
        start_date = datetime(year=last_datetime.year, month=last_datetime.month, day=last_datetime.day)
    else:
        instance_info = {}
        inst_id_ctr = 0
        dataset_generation_info = {
            "args": vars(args),
            "seed": seed,
            "par_fields": par_fields,
        }
        os.makedirs(args.atrain_dir, exist_ok=True)
        json.dump(dataset_generation_info, open(os.path.join(args.atrain_dir, "dataset_generation_info.json"), "w"))
        # the intersection of the operational lifetimes of the relevant A-Train satellites starts on 06/15/2006...
        start_date = datetime(year=2006, month=6, day=15)
    # ...and ends on 10/10/2013
    end_date = datetime(year=2013, month=10, day=10)
    start_datetime = datetime(year=2006, month=6, day=15, hour=12, minute=4, second=34)

    pbar = tqdm(range((end_date - start_date).days + 1), position=0, leave=True)
    for day_idx in pbar:
        date = start_date + timedelta(days=day_idx)
        date_subpath = datetime_to_subpath(date)

        # Loop over files in this day
        date_sync_subdir = os.path.join(sync_subdir, date_subpath)
        try:
            cal_par_filenames = sorted(icare_ses.listdir(date_sync_subdir))
        except FileNotFoundError:
            print(f"No CALIPSO/PARASOL sync directory at: {date_sync_subdir}, continuing.")
            continue
        for cal_par_filename in cal_par_filenames:
            cal_par_datetime = tai93_string_to_datetime(cal_par_filename.split(".")[0].split("_")[3])
            if cal_par_datetime < start_datetime or (args.resume and cal_par_datetime <= last_datetime):
                continue

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

                # TO DO:
                # log the output
                # documentation
                # push
                # make script multi-thread


import pdb
import sys
import traceback

if __name__ == "__main__":
    try:
        main()
    except:
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

# main()
