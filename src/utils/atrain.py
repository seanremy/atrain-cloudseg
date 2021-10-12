"""Utilities for working with data from satellites in the A-train, specifically POLDER, CALIOP, and CloudSat."""

import os
import re
from datetime import datetime
from typing import Tuple

import h5py
import numpy as np
from pyhdf.SD import SD, SDC
from scipy.ndimage.filters import convolve
from scipy.ndimage.interpolation import map_coordinates
from scipy.stats import circmean

CLOUD_SCENARIO_INFO = {
    0: {"name": "No cloud", "color": (0, 0, 0)},
    1: {"name": "Cirrus", "color": (76, 25, 157)},
    2: {"name": "Altostratus", "color": (0, 37, 245)},
    3: {"name": "Altocumulus", "color": (72, 167, 248)},
    4: {"name": "Stratus", "color": (106, 227, 83)},
    5: {"name": "Stratocumulus", "color": (255, 253, 84)},
    6: {"name": "Cumulus", "color": (243, 172, 61)},
    7: {"name": "Nimbostratus", "color": (228, 76, 41)},
    8: {"name": "Deep Convection", "color": (183, 48, 193)},
}


def map_cloud_scenario_colors(cloud_scenario: np.array) -> np.array:
    """Map an array of cloud scenario codes to RGB colors.

    Args:
        cloud_scenario: Array of cloud scenario codes, from 0 to 8.

    Returns:
        colors: Array of RGB colors.
    """
    colors = np.zeros((cloud_scenario.shape[0], cloud_scenario.shape[1], 3))
    for scenario_num in CLOUD_SCENARIO_INFO:
        colors[cloud_scenario == scenario_num] = CLOUD_SCENARIO_INFO[scenario_num]["color"]
    return colors


def polder_grid_to_latlon(lin: np.array, col: np.array, rounding: bool = False) -> tuple[np.array, np.array]:
    """Convert coordinates in the POLDER grid to latitude/longitude. See Appendix B here:
    web-backend.icare.univ-lille.fr//projects_data/parasol/docs/Parasol_Level-1_format_latest.pdf

    Args:
        lin: Row in the POLDER grid, 1 <= lin <= 3240
        col: Column in the POLDER grid, 1 <= col <= 6480
        rounding: True if you want to round, defaults to False

    Returns:
        lat: Latitude
        lon: Longitude
    """
    finite_lin, finite_col = lin[np.isfinite(lin)], col[np.isfinite(col)]
    assert ((0 <= finite_lin) * (finite_lin < 3240)).all()
    assert ((0 <= finite_col) * (finite_col < 6480)).all()
    finite_lin, finite_col = finite_lin + 1, finite_col + 1
    lat = 90 - (lin + 0.5) / 18
    n_i = 3240 * np.cos(lat * np.pi / 180)
    if rounding:
        n_i = np.round(n_i)
    lon = (180 / n_i) * (col - 3239.5)
    lt180, gt180 = lon < -180, lon > 180
    lon[lt180] = lon[lt180] + 360
    lon[gt180] = lon[gt180] - 360
    assert (-90 <= lat).all() and (lat <= 90).all()
    assert (-180 <= lon).all() and (lon <= 180).all()
    return lat, lon


def latlon_to_polder_grid(lat: np.array, lon: np.array, rounding: bool = False) -> tuple[np.array, np.array]:
    """Convert latitude/longitude to coordinates in the POLDER grid. See Appendix B here:
    web-backend.icare.univ-lille.fr//projects_data/parasol/docs/Parasol_Level-1_format_latest.pdf

    Args:
        lat: Latitude
        lon: Longitude
        rounding: True if you want to round, defaults to False

    Returns:
        lin: Row in the POLDER grid, 1 <= lin <= 3240
        col: Column in the POLDER grid, 1 <= col <= 6480
    """

    finite_lat, finite_lon = lat[np.isfinite(lat)], lon[np.isfinite(lon)]
    assert ((-180 <= finite_lat) * (finite_lat <= 180)).all()
    assert ((-180 <= finite_lon) * (finite_lon <= 180)).all()
    lin = 18 * (90 - lat) + 0.5
    if rounding:
        lin = np.round(lin)
    n_i = 3240 * np.sin((np.pi / 180) * (lin - 0.5) / 18)
    if rounding:
        n_i = np.round(n_i)
    col = 3240.5 + (n_i / 180) * lon
    if rounding:
        col = np.round(col)
    # lin, col = lin - 1, col - 1  # start at 0
    assert (0 <= lin).all() and (lin <= 3240).all()
    assert (0 <= col).all() and (col <= 6480).all()
    return lin, col


def _decode_cloud_scenario(cloud_scenario_arr: np.array) -> dict:
    """Decode the cloud scenario data from a 2B-CLDCLASS scene.

    For format specification, see: https://www.ncl.ucar.edu/Applications/cloudsat.shtml

    Args:
        cloud_scenario_arr: An array of cloud scenario codes.

    Returns:
        cloud_scenario: A dictionary containing the decoded cloud scenario information
    """
    assert cloud_scenario_arr.dtype == np.int16
    assert len(cloud_scenario_arr.shape) == 2

    def _get_bits(arr, start, end):
        # get bits in index range [start, end)
        return arr % (2 ** end) // (2 ** start)

    cloud_scenario = {
        "cloud_scenario_flag": _get_bits(cloud_scenario_arr, 0, 1),
        "cloud_scenario": _get_bits(cloud_scenario_arr, 1, 5),
        "land_sea_flag": _get_bits(cloud_scenario_arr, 5, 7),
        "latitude_flag": _get_bits(cloud_scenario_arr, 7, 9),
        "algorithm_flag": _get_bits(cloud_scenario_arr, 9, 11),
        "quality_flag": _get_bits(cloud_scenario_arr, 11, 13),
        "precipitation_flag": _get_bits(cloud_scenario_arr, 13, 15),
    }
    return cloud_scenario


def tai93_string_to_datetime(tai_time: str) -> datetime:
    """Get a datetime from a TAI93 string.

    Args:
        tai_time: A datetime in the TAI 93 format
    Returns:
        dt: The converted datetime
    """
    ymd, hms = tai_time.split("T")
    hms = hms.replace(":", "-")
    year, month, day = ymd.split("-")
    hour, minute, second = hms.split("-")
    second = re.findall("\d+", second)[
        0
    ]  # for some reason there can be a ZD (zone descriptor? has no number though..?)
    dt = datetime(
        year=int(year), month=int(month), day=int(day), hour=int(hour), minute=int(minute), second=int(second)
    )
    return dt


class CALPARScene:
    """A scene containing data from a CALIPSO/CALIOP + PARASOL/POLDER sync half-orbit file."""

    def __init__(self, filepath: str) -> None:
        """Create a CALIPSO/PARASOL Scene.

        Args:
            filepath: Path to the CALIPSO/PARASOL half-orbit file
        """
        self.filepath = filepath
        self.filename = os.path.split(self.filepath)[1]
        self.hdf = SD(self.filepath, SDC.READ)
        self.acquisition_range = self._get_acquisition_range()

    def _get_acquisition_range(self) -> tuple[datetime, datetime]:
        """Get the acquisition range of this scene.

        Returns:
            beginning: The beginning-of-acquisition datetime
            ending: The ending-of-acquisition datetime
        """
        attrs = self.hdf.attributes()
        beginning, ending = (attrs["Beginning_Acquisition_Date"], attrs["Ending_Acquisition_Date"])
        beginning, ending = tuple([tai93_string_to_datetime(d.replace("\x00", "")) for d in (beginning, ending)])
        return beginning, ending

    def get_best_parasol_filepath(self, par_filepaths: list[str], time_match_threshold: int) -> str:
        """Get the closest time-matching PARASOL filepath of those provided, unless none are within the threshold.

        Args:
            par_filepaths: List of PARASOL filepaths
            time_match_threshold: Time threshold for a positive match, in seconds
        Returns:
            best_filepath: Best matching filepath, or None if none match within the threshold
        """
        if par_filepaths == []:
            return None
        par_filenames = [os.path.split(fp)[1] for fp in par_filepaths]
        par_datetimes = [tai93_string_to_datetime(fn.split(".")[0].split("_")[2]) for fn in par_filenames]
        seconds_off_per_file = np.abs([(self.acquisition_range[0] - dt).total_seconds() for dt in par_datetimes])
        min_seconds_off_idx = np.argmin(seconds_off_per_file)
        min_seconds_off = seconds_off_per_file[min_seconds_off_idx]
        if min_seconds_off > time_match_threshold:
            return None
        best_filepath = par_filepaths[min_seconds_off_idx]
        return best_filepath


class PARASOLScene:
    """A scene containing data from a PARASOL/POLDER half-orbit file."""

    def __init__(self, filepath: str, fields: list, min_views: int) -> None:
        """Create a PARASOL Scene.

        Args:
            filepath: Path to the hdf5 file of a PARASOL scene
            fields: List of fields to store
            min_views: Minimum number of available viewing angles for a pixel to be valid
        """
        self.filepath = filepath
        self.filename = os.path.split(self.filepath)[1]
        self.datetime = tai93_string_to_datetime(self.filename.split(".")[0].split("_")[2])
        self.fields = fields
        self.min_views = min_views

        # read the hdf5 file
        self.h5 = h5py.File(self.filepath, "r")

        # where we have enough views
        nviews, nviews_validity = self.get_valid_values_and_mask(["Data_Fields", "Nviews"])
        enough_views = nviews >= self.min_views
        nv_row, nv_col = np.where(nviews_validity)
        self.view_validity = np.zeros((3240, 6480)).astype(bool)
        self.view_validity[nv_row[enough_views], nv_col[enough_views]] = True

        # store latitude and longitude, and the mask of where they're valid
        self.lat, self.geo_validity = self.get_valid_values_and_mask(["Geolocation_Fields", "Latitude"])
        self.lon, _ = self.get_valid_values_and_mask(["Geolocation_Fields", "Longitude"])
        self.lat, self.lon = self.lat[enough_views], self.lon[enough_views]

    def get_valid_values_and_mask(self, keys) -> Tuple[np.array, np.array]:
        """Get the valid values and the mask of a dataset in the scene.

        Args:
            keys: The list of keys, in order, specifying the location of the dataset in the hdf5 file.

        Returns:
            arr: The 1-D array of valid values.
            mask: The 2-D mask at which the valid values are located.
        """
        h5 = self.h5
        for k in keys:
            h5 = h5[k]
        arr = np.array(h5)
        mask = np.ones(arr.shape).astype(bool)
        if h5.attrs["Num_missing_value"] > 0:
            mask = mask * (arr != h5.attrs["missing_value"])
        if h5.attrs["Num_Fill"] > 0:
            mask = mask * (arr != h5.attrs["_FillValue"])
        arr = arr[mask]
        return arr, mask

    def get_polder_grid_mask(self) -> np.array:
        """Get the mask of which pixels in the POLDER grid (sinusoidal projection) are valid.

        Returns:
            pg_mask: The POLDER grid validity mask.
        """
        pg_mask = np.zeros((3240, 6480)).astype(bool)
        row_range = np.arange(3240)
        lat_range = 90 - (row_range + 0.5) / 18
        n_i = np.round(3240 * np.cos(lat_range * np.pi / 180))
        for i in range(3240):
            pg_mask[i, int(np.floor(3240.5 - n_i[i])) : int(np.floor(3240.5 + n_i[i]))] = True
        return pg_mask

    def get_patch(self, patch: dict) -> np.array:
        """Get a patch in this scene.

        Args:
            patch: Geographic info on a patch, with row and column locations in the POLDER grid.

        Returns:
            patch_arr: The patch data as a 3-D array.
        """
        patch_data = []

        for field in self.fields:
            h5_dataset = self.h5[field[0]][field[1]]
            # rows will be the same across columns, which we can use to index this cheaply
            row_crop = np.array(h5_dataset[patch["pg_row"][:, 0]])
            # add extra dimensions to the column index if we need them for take_along_axis
            pg_col = patch["pg_col"]
            if len(row_crop.shape) == 3:
                pg_col = np.expand_dims(pg_col, axis=2)
            # get the patch values
            patch_values = np.take_along_axis(row_crop, pg_col, axis=1)
            if len(patch_values.shape) == 2:
                patch_values = np.expand_dims(patch_values, axis=2)
            # figure out where values are filled or missing
            valid_mask = patch_values != h5_dataset.attrs["_FillValue"]
            if h5_dataset.attrs["Num_missing_value"] > 0:
                valid_mask = valid_mask * (patch_values != h5_dataset.attrs["missing_value"])
            # apply the scaling and addition, where valid
            if h5_dataset.attrs["scale_factor"] != 1 or h5_dataset.attrs["add_offset"] != 1:
                patch_values = patch_values.astype(type(h5_dataset.attrs["scale_factor"]))
                patch_values[valid_mask] = (
                    patch_values[valid_mask] * h5_dataset.attrs["scale_factor"] + h5_dataset.attrs["add_offset"]
                )
            patch_data.append(patch_values)
        patch_arr = np.concatenate(patch_data, axis=2)
        return patch_arr


class CLDCLASSScene:
    """A scene containing data from a CLDCLASS half-orbit file."""

    def __init__(self, filepath: str) -> None:
        """Create a CLDCLASS Scene.

        Args:
            filepath: Path to the CLDCLASS half-orbit file
        """
        self.filepath = filepath
        self.filename = os.path.split(self.filepath)[1]
        self.hdf = SD(self.filepath, SDC.READ)
        self.lat = np.array(self.hdf.select("Latitude"))
        self.lon = np.array(self.hdf.select("Longitude"))
        self.height = np.array(self.hdf.select("Height"))
        self.time = np.array(self.hdf.select("Time"))
        self.cloud_scenario = _decode_cloud_scenario(np.array(self.hdf.select("cloud_scenario")))

    def get_lat_intervals(self, patch_size: int) -> np.array:
        """Get the array of all index pairs in this scene's latitude array that cover a patch.

        Args:
            patch_size: The size (width, height) of square patches in the output dataset.

        Returns:
            cc_lat_intervals: Array of all index pairs in this scene's latitude array that cover a patch.
        """
        patch_lat_range = patch_size / 18  # 18 POLDER Grid rows <=> 1 degree of latitude
        cc_lat_intervals = []
        for i in range(self.lat.shape[0] - 1):
            for j in range(i + 1, self.lat.shape[0]):
                lat_diff = np.abs(self.lat[j] - self.lat[i])
                if lat_diff > patch_lat_range:
                    cc_lat_intervals.append([i, j])
                    break
        cc_lat_intervals = np.array(cc_lat_intervals)
        return cc_lat_intervals

    def get_patch(self, patch: dict) -> dict:
        """Get a patch in this scene.

        Args:
            patch: Geographic info on a patch, with indices to values in this scene.

        Returns:
            patch_dict: Dictionary from fields to their values in this patch.
        """
        patch_dict = {
            "lat": self.lat[patch["cc_idx"]],
            "lon": self.lon[patch["cc_idx"]],
            "height": self.height[patch["cc_idx"]],
            "time": self.time[patch["cc_idx"]],
            "cloud_scenario": {k: self.cloud_scenario[k][patch["cc_idx"]] for k in self.cloud_scenario},
        }
        return patch_dict
