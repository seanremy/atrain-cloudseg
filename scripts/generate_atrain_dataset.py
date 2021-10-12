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

import geopy.distance
import numpy as np
from scipy.spatial import KDTree
from torch._C import ParameterDict
from tqdm import tqdm

if "./src" not in sys.path:
    sys.path.insert(0, "./src")  # TO DO: change this once it's a package
from utils.atrain import CALPARScene, CLDCLASSScene, PARASOLScene, polder_grid_to_latlon, tai93_string_to_datetime
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
        "--samples_per_file", type=int, default=16, help="Goal number of samples to take per orbit file."
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=100,
        help="Patch size, in pixels, of samples to take. Raising this number has a severe effect on "
        "the size of the dataset on disk.",
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
    # assert args.nadir_padding >= 0 and args.nadir_padding < args.patch_size // 2
    assert args.min_parasol_views > 0 and args.min_parasol_views <= 16
    # assert args.field_scheme.lower() in FIELD_DICT
    return args


def get_patches(par_scene: PARASOLScene, cc_scene: CLDCLASSScene, patch_size: int, patches_per_scene: int) -> list:
    """Get a list of patches for the provided scene pair.

    Args:
        par_scene: The PARASOL scene.
        cc_scene: The 2B-CLDCLASS scene.
        patch_size: The size (height & width) of the square patches to sample from the scenes.
        patches_per_scene: How many patches to get in this pair of scenes.
    """

    # pre-compute POLDER grid stuff for later
    pg_mask = par_scene.get_polder_grid_mask()
    pg_row, pg_col = np.where(pg_mask)
    pg_lat, pg_lon = polder_grid_to_latlon(pg_row, pg_col)
    row_valid, col_valid = np.where(par_scene.view_validity)
    lon_ranges = patch_size * (180 / (3240 * np.cos(pg_lat * np.pi / 180)))

    # use a KD-Tree to find closest 20 PARASOL pixels in lat/lon for every Cloudsat observation
    par_latlon = np.stack([par_scene.lat, par_scene.lon], axis=1)
    cc_latlon = np.stack([cc_scene.lat, cc_scene.lon], axis=1)
    par_ll_kdtree = KDTree(par_latlon)
    _, closest_20_idx = par_ll_kdtree.query(cc_latlon, 20)
    closest_20_latlon = par_latlon[closest_20_idx]

    # get the ground distance between each Cloudsat observation and its closest 20
    dists_to_closest_20 = []
    for i in range(closest_20_idx.shape[0]):
        d = []
        for j in range(20):
            d.append(geopy.distance.distance(cc_latlon[i], closest_20_latlon[i, j], ellipsoid="WGS-84").m)
        dists_to_closest_20.append(np.array(d))
    dists_to_closest_20 = np.array(dists_to_closest_20)
    dists_sort = np.argsort(dists_to_closest_20, axis=1)

    # reorder everything in increasing distance
    closest_20_idx = np.take_along_axis(closest_20_idx, dists_sort, axis=1)
    closest_20_latlon[:, :, 0] = np.take_along_axis(closest_20_latlon[:, :, 0], dists_sort, axis=1)
    closest_20_latlon[:, :, 1] = np.take_along_axis(closest_20_latlon[:, :, 1], dists_sort, axis=1)
    dists_to_closest_20 = np.take_along_axis(dists_to_closest_20, dists_sort, axis=1)

    # get whether the closest 20 points are north and/or west of the cloudsat observations
    cc_lat_stack = np.stack([cc_scene.lat] * 20, axis=1)
    cc_lon_stack = np.stack([cc_scene.lon] * 20, axis=1)
    north_mask = closest_20_latlon[:, :, 0] > cc_lat_stack
    west_mask = closest_20_latlon[:, :, 1] < cc_lon_stack

    # NW, NE, SW, SE corners
    corner_mask = np.stack(
        [north_mask * west_mask, north_mask * ~west_mask, ~north_mask * west_mask, ~north_mask * ~west_mask], axis=2
    )
    enclosed_mask = corner_mask.sum(axis=1).all(axis=1)  # which points can be bounded
    corner_mask = corner_mask[enclosed_mask]

    # get the closest point from each of NW, NE, SW, SE
    closest_corner_idx = np.argmax(corner_mask, axis=1)
    corner_idx = np.take_along_axis(closest_20_idx[enclosed_mask], closest_corner_idx, axis=1)
    corner_dists = np.take_along_axis(dists_to_closest_20[enclosed_mask], closest_corner_idx, axis=1)

    # weight is a factor of distance
    corner_dists_normed = corner_dists / corner_dists.sum(axis=1)[:, np.newaxis]
    corner_weights = (1 / corner_dists_normed) / ((1 / corner_dists_normed).sum(axis=1)[:, np.newaxis])

    # get Cloudsat intervals that are just barely over the latitude range we need to get big enough patches
    cc_lat_intervals = cc_scene.get_lat_intervals(patch_size)

    # get all latitude ranges where all of the Cloudsat observations are enclosed by corners
    enclosed_lat_range_idx = np.array(
        [
            i
            for i in range(cc_lat_intervals.shape[0])
            if enclosed_mask[cc_lat_intervals[i, 0] : cc_lat_intervals[i, 1]].all()
        ]
    )
    enclosed_lat_ranges = cc_lat_intervals[enclosed_lat_range_idx]
    enclosed_center_idx = np.floor(enclosed_lat_ranges.mean(axis=1)).astype(int)
    patch_centers_latlon = np.stack([cc_scene.lat[enclosed_center_idx], cc_scene.lon[enclosed_center_idx]], axis=1)

    # randomly shift patches a little east or west so that labels aren't always in the middle
    lon_ranges_by_patch_center = patch_size * (180 / (3240 * np.cos(patch_centers_latlon[:, 0] * np.pi / 180)))
    lon_offsets = (np.random.random(patch_centers_latlon.shape[0]) - 0.5) * lon_ranges_by_patch_center / 4
    patch_centers_latlon[:, 1] += lon_offsets

    # get patches
    patches = []
    rand_order = np.arange(enclosed_lat_ranges.shape[0])
    np.random.shuffle(rand_order)
    i = 0
    while i < rand_order.shape[0] and len(patches) < patches_per_scene:
        # get the patch mask in the parasol data
        patch_lat_range = enclosed_lat_ranges[rand_order[i]]
        patch_lat_mask_par = (pg_lat > cc_scene.lat[patch_lat_range[0]]) * (pg_lat < cc_scene.lat[patch_lat_range[1]])
        where_patch_lat_mask_par = np.where(patch_lat_mask_par)[0]
        where_not_extra_lat = pg_row[patch_lat_mask_par] - pg_row[patch_lat_mask_par].min() < patch_size
        where_patch_lat_mask_par = where_patch_lat_mask_par[where_not_extra_lat]
        patch_lat_mask_par = np.zeros_like(patch_lat_mask_par).astype(bool)
        patch_lat_mask_par[where_patch_lat_mask_par] = True
        patch_lon_mask_par = (np.abs(pg_lon - patch_centers_latlon[rand_order[i], 1]) % 360) < lon_ranges / 2
        patch_mask_par = patch_lat_mask_par * patch_lon_mask_par

        # index of all of the corners in the parasol data
        lat_range_mask = np.zeros(enclosed_mask.shape).astype(bool)
        lat_range_mask[np.arange(patch_lat_range[0], patch_lat_range[1])] = True
        corners_par_idx = corner_idx[lat_range_mask[enclosed_mask]]

        # get whether or not each cloudclass point has all of its corners within this patch
        corner_idx_full_pg = row_valid[corners_par_idx] * 6480 + col_valid[corners_par_idx]
        patch_idx_full_pg = pg_row[patch_mask_par] * 6480 + pg_col[patch_mask_par]
        corners_in_patch = np.isin(corner_idx_full_pg, patch_idx_full_pg).all(axis=1)

        # re-index stuff to only keep those points with 4 corners
        corners_par_idx = corners_par_idx[corners_in_patch]
        patch_mask_cc = np.zeros(enclosed_mask.shape).astype(bool)
        patch_mask_cc[np.arange(patch_lat_range[0], patch_lat_range[1])[corners_in_patch]] = True

        # only keep this patch if it's the right size and we have enough labeled pixels:
        patch_good = (patch_mask_par.sum() == (patch_size ** 2)) and patch_mask_cc.sum() >= patch_size
        if patch_good:
            patch_pg_row = np.reshape(pg_row[patch_mask_par], (patch_size, patch_size))
            patch_pg_col = np.reshape(pg_col[patch_mask_par], (patch_size, patch_size))
            patch_corner_idx = np.stack([row_valid[corners_par_idx], col_valid[corners_par_idx]], axis=2)
            patch_corner_idx[:, :, 0] -= patch_pg_row.min()
            patch_corner_idx[:, :, 1] -= patch_pg_col.min(axis=1)[patch_corner_idx[:, :, 0]]

            patches.append(
                {
                    "pg_row": patch_pg_row,
                    "pg_col": patch_pg_col,
                    "cc_idx": np.where(patch_mask_cc)[0],
                    "corner_idx": patch_corner_idx,
                    "corner_weights": corner_weights[lat_range_mask[enclosed_mask]][corners_in_patch],
                }
            )
        i += 1
    return patches


def main() -> None:

    args = parse_args()

    # start a session with ICARE's FTP server
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

        # get a PARASOL scene and record the field attributes (metadata)
        ex_par_file = icare_ses.get_file(
            os.path.join(
                "PARASOL", "L1_B-HDF", "2008", "2008_06_03", "POLDER3_L1B-BG1-080176M_2008-06-03T01-34-54_V1-01.h5"
            )
        )
        par_scene = PARASOLScene(ex_par_file, par_fields, min_views=args.min_parasol_views)
        field_attributes = defaultdict(dict)
        for field in par_scene.fields:
            attrs = dict(par_scene.h5[field[0]][field[1]].attrs)
            for k in attrs:
                if type(attrs[k]) in [np.array, np.ndarray]:
                    if attrs[k].dtype == object:
                        attrs[k] = [str(s) for s in attrs[k].tolist()]
                    else:
                        attrs[k] = attrs[k].astype(float).tolist()
                elif np.issubdtype(type(attrs[k]), np.integer):
                    attrs[k] = int(attrs[k])
            field_attributes[field[0]][field[1]] = attrs
        dataset_generation_info["par_field_attributes"] = field_attributes

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
            cc_scene = CLDCLASSScene(local_cldclass_filepath)
            if not cc_scene.cloud_scenario["cloud_scenario"].any():
                print(f"CLDCLASS Scene at {cldclass_filepath} has no cloud scenario data, continuing.")
                continue

            # read the PARASOL scene
            par_scene = PARASOLScene(local_par_filepath, par_fields, min_views=args.min_parasol_views)

            # sample patches from the scene
            patches = get_patches(par_scene, cc_scene, args.patch_size, args.samples_per_file)
            for patch in patches:

                # get patch array and labels
                par_patch = par_scene.get_patch(patch)
                cc_patch = cc_scene.get_patch(patch)
                cc_patch["corner_idx"] = patch["corner_idx"]
                cc_patch["corner_weights"] = patch["corner_weights"]

                # get the paths
                input_dir = os.path.join("input", date_subpath)
                output_dir = os.path.join("output", date_subpath)
                for d in [input_dir, output_dir]:
                    os.makedirs(os.path.join(args.atrain_dir, d), exist_ok=True)
                input_path = os.path.join(input_dir, f"{inst_id_ctr}.npy")
                output_path = os.path.join(output_dir, f"{inst_id_ctr}.pkl")

                # save everything
                instance_info[inst_id_ctr] = {
                    "cal_par_file": cal_par_scene.filepath,
                    "par_file": par_scene.filepath,
                    "cldclass_file": cc_scene.filepath,
                    "instance id": inst_id_ctr,
                    "input_path": input_path,
                    "output_path": output_path,
                    "cal_par_datetime": str(cal_par_datetime),
                }
                inst_id_ctr += 1
                np.save(open(os.path.join(args.atrain_dir, input_path), "wb"), par_patch)
                pickle.dump(cc_patch, open(os.path.join(args.atrain_dir, output_path), "wb"))
                json.dump(instance_info, open(os.path.join(args.atrain_dir, "instance_info.json"), "w"))

    icare_ses.cleanup()
    print("Done generating dataset!!")


if __name__ == "__main__":
    main()
