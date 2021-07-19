"""Utilities for working with data from satellites in the A-train, specifically POLDER, CALIOP, and CloudSat."""

import math
import os
import re
from collections import defaultdict
from datetime import datetime

import h5py
import numpy as np
from pyhdf.SD import SD, SDC
from scipy.ndimage.filters import convolve
from scipy.ndimage.interpolation import map_coordinates

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
    lin, col = lin - 1, col - 1  # start at 0
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


def _dataset_to_numpy(h5_dataset: h5py._hl.dataset.Dataset, crop: tuple = None) -> tuple[np.array, np.array]:
    """Convert a dataset from an hdf5 object with geospatial data into a numpy array.

    Args:
        h5_dataset: The dataset to convert. Needs to have standard attributes for ICARE geospatial data
        crop: The index and shape of the crop to use, optional

    Returns:
        arr: The numpy array of physical values of the provided dataset
        valid_mask: The mask of where the returned physical values are valid
    """
    arr = np.array(h5_dataset)
    if len(arr.shape) < 3:
        arr = np.expand_dims(arr, axis=2)
    if not crop is None:
        crop_idx, crop_shape = crop
        arr = np.stack([map_coordinates(arr[:, :, i], crop_idx, order=0) for i in range(arr.shape[2])], axis=1)
        arr = arr.reshape((crop_shape[1], crop_shape[0], arr.shape[1]))
        arr = np.transpose(arr, (1, 0, 2))
    valid_mask = arr != h5_dataset.attrs["_FillValue"]
    if h5_dataset.attrs["Num_missing_value"] > 0:
        valid_mask = valid_mask * (arr != h5_dataset.attrs["missing_value"])
    where_valid = np.where(valid_mask)
    masked_values = arr[where_valid]
    if h5_dataset.attrs["scale_factor"] != 1:
        masked_values = masked_values * h5_dataset.attrs["scale_factor"]
    if h5_dataset.attrs["add_offset"] != 0:
        masked_values = masked_values + h5_dataset.attrs["add_offset"]
    arr = arr.astype(masked_values.dtype)
    arr[where_valid] = masked_values
    return arr, valid_mask


def _crop_to_idx(crop: dict, theta: float = 0) -> tuple[np.array, np.array]:
    """Convert a crop dictionary to an index.

    Args:
        crop: A dictionary with top, bottom, left, and right values of a crop window
        theta: The angle by which the crop's sinusoidal projection is rotated
    Returns:
        crop_grid_y: The y-coordinates of the crop index
        crop_grid_x: The x-coordinates of the crop index
    """
    crop_range_y = np.arange(np.floor(crop["top"]), np.ceil(crop["bottom"]))
    crop_range_x = np.arange(np.floor(crop["left"]), np.ceil(crop["right"]))
    crop_grid_y, crop_grid_x = [g.flatten() for g in np.meshgrid(crop_range_y, crop_range_x)]
    crop_grid_lat, crop_grid_lon = polder_grid_to_latlon(crop_grid_y, crop_grid_x)
    crop_grid_lon = (crop_grid_lon - theta + 180) % 360 - 180
    crop_grid_y, crop_grid_x = latlon_to_polder_grid(crop_grid_lat, crop_grid_lon)
    return crop_grid_y, crop_grid_x


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

        # get the full POLDER grid latitude and longitude
        self.lat_og_arr, self.lat_og_mask = _dataset_to_numpy(self.h5["Geolocation_Fields"]["Latitude"])
        self.lon_og_arr, self.lon_og_mask = _dataset_to_numpy(self.h5["Geolocation_Fields"]["Longitude"])
        self.lat_og = self.lat_og_arr[self.lat_og_mask]
        self.lon_og = self.lon_og_arr[self.lon_og_mask]

        # center and crop the scene
        self._center_and_crop()

        self.field_data = defaultdict(dict)
        for field in self.fields:
            self.field_data[field[0]][field[1]], _ = _dataset_to_numpy(
                self.h5[field[0]][field[1]], (self.crop_idx, self.crop_shape)
            )

    def _center_and_crop(self) -> None:
        """Center and crop this PARASOL Scene."""
        nonpolar = (self.lat_og > -60) * (self.lat_og < 60)
        lat_nonpolar = self.lat_og[nonpolar]
        lon_nonpolar = self.lon_og[nonpolar]
        if np.min(np.abs(self.lon_og - 180)) < 1 or np.min(np.abs(self.lon_og + 180)) < 1:
            # if we are near the 180 meridian, special midpoint computation
            lon_shifted = (self.lon_og % 360 - 180)[nonpolar]
            theta = -(np.max(lon_shifted) + np.min(lon_shifted)) / 2 + 180
        else:
            theta = -(np.max(lon_nonpolar) + np.min(lon_nonpolar)) / 2
        self.theta = (theta + 180) % 360 - 180
        lon_c = (lon_nonpolar + self.theta + 180) % 360 - 180
        row, col = latlon_to_polder_grid(lat_nonpolar, lon_c)
        self.crop = {
            "left": int(np.round(np.min(col))),
            "right": int(np.round(np.max(col))) + 1,
            "top": int(np.round(np.min(row))),
            "bottom": int(np.round(np.max(row))) + 1,
        }
        self.crop_shape = (self.crop["bottom"] - self.crop["top"], self.crop["right"] - self.crop["left"])
        self.crop_idx = _crop_to_idx(self.crop, self.theta)

    def get_view_validity(self, patch_size: int) -> np.array:
        """Get the 'view validity' from this scene. The 'view validity' is the mask of all box centers for
        boxes of size (patch_size, patch_size) where all of the pixels in the box have enough views available.

        Args:
            patch_size: The size of the patches for which to check validity

        Returns:
            view_validity: The view validity mask, in the POLDER grid
        """
        num_views = self.field_data["Data_Fields"]["Nviews"][:, :, 0]
        views_valid = num_views != self.h5["Data_Fields"]["Nviews"].attrs["_FillValue"]
        enough_views = ((num_views >= self.min_views) * views_valid).astype(int)
        filter = np.zeros((1, patch_size)) + 1
        sum_enough_views = convolve(convolve(enough_views, filter, mode="constant"), filter.T, mode="constant")
        view_validity = sum_enough_views == (patch_size * patch_size)
        return view_validity

    def get_patch(self, center_x: int, center_y: int, patch_size: int) -> tuple[np.array, dict]:
        """Get a square patch in this scene, with a provided center and patch size.

        Args:
            center_x: The x-coordinate of the patch center
            center_y: The y-coordinate of the patch center
            patch_size: The side length of the patch square
        Returns:
            patch_arr: The patch, as an array containing all fields
            patch_box: A dictionary containing the row space, column space, and size of this patch
        """
        lat = self.field_data["Geolocation_Fields"]["Latitude"]
        lon = self.field_data["Geolocation_Fields"]["Longitude"]
        top, bottom = center_y - (patch_size - 1) // 2, center_y + math.ceil((patch_size - 1) / 2)
        left, right = center_x - (patch_size - 1) // 2, center_x + math.ceil((patch_size - 1) / 2)

        padding = max(10, patch_size // 4)
        top_pad = min(padding, top)
        bottom_pad = min(padding, self.crop_shape[0] - bottom)
        left_pad = min(padding, left)
        right_pad = min(padding, self.crop_shape[1] - right)
        lat_min = lat[bottom, center_x, 0]
        lat_max = lat[top, center_x, 0]
        lon_min = (lon[center_y, left, 0] + self.theta + 180) % 360 - 180
        lon_max = (lon[center_y, right, 0] + self.theta + 180) % 360 - 180
        lat_space = np.linspace(lat_min, lat_max, patch_size)
        lon_space = np.linspace(lon_min, lon_max, patch_size)
        row_space, col_space = latlon_to_polder_grid(lat_space, lon_space)

        row_space = row_space[::-1] - self.crop["top"]
        col_space = col_space - self.crop["left"]
        patch_box = {
            "row_space": row_space,
            "col_space": col_space,
            "patch_size": patch_size,
        }

        interp_row = row_space - top + top_pad + 1
        interp_col = col_space - left + left_pad + 1
        interp_coords = np.stack(np.meshgrid(interp_row, interp_col), axis=0)

        patch_arrs = []
        for field in self.fields:
            field_arr = self.field_data[field[0]][field[1]]
            interp_input = field_arr[top - top_pad : bottom + bottom_pad, left - left_pad : right + right_pad]
            interp_output = []
            for i in range(interp_input.shape[2]):
                interp_output.append(map_coordinates(interp_input[:, :, i], interp_coords, order=1).T)
            patch_arrs.append(np.stack(interp_output, axis=2))
        patch_arr = np.concatenate(patch_arrs, axis=2)

        return patch_arr, patch_box


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
        self.lat, self.lon = np.array(self.hdf.select("Latitude")), np.array(self.hdf.select("Longitude"))
        self.height = np.array(self.hdf.select("Height"))
        self.time = np.array(self.hdf.select("Time"))
        self.cloud_scenario = _decode_cloud_scenario(np.array(self.hdf.select("cloud_scenario")))

    def get_nadir_mask(self, par_scene: PARASOLScene) -> np.array:
        """Get the nadir mask for this CLDCLASS Scene. The nadir mask is the mask of pixels in the POLDER grid of the
        provided PARASOL Scene that are directly beneath (nadir) CloudSat, and for which we have LiDAR returns.

        Args:
            par_scene: The PARASOL/POLDER scene whose POLDER grid to use
        Returns:
            nadir_mask: The nadir mask for this CLDCLASS scene
        """
        lon_c = (self.lon + par_scene.theta + 180) % 360 - 180
        row, col = latlon_to_polder_grid(self.lat, lon_c)
        valid_row_idx = (row >= par_scene.crop["top"]) * (row < par_scene.crop["bottom"])
        valid_col_idx = (col >= par_scene.crop["left"]) * (col < par_scene.crop["right"])
        valid_idx = valid_row_idx * valid_col_idx
        nadir_mask = np.zeros(par_scene.crop_shape, dtype=bool)
        crop_row = row[valid_idx].astype(int) - par_scene.crop["top"]
        crop_col = col[valid_idx].astype(int) - par_scene.crop["left"]
        nadir_mask[crop_row, crop_col] = 1
        return nadir_mask

    def get_nadir_validity(self, par_scene: PARASOLScene, patch_size: int, padding: int = 0) -> np.array:
        """Get the 'nadir validity' of this CLDCLASS Scene in the provided PARASOL scene. The 'nadir validity' is the
        mask of all box centers of boxes with size (patch_size, patch_size) that satisfy two conditions: 1) the
        CLDCLASS nadir line is within the box, and 2) there are at least 'padding' pixels between the nadir line and the
        left (west) and right (east) edges of the box.

        Args:
            par_scene: The PARASOL/POLDER scene in which to find the validity
            patch_size: The side length of the patch square
            padding: The padding to keep between the nadir line and box edges, defaults to 0.
        Returns:
            nadir_validity: The nadir validity mask, in the POLDER grid
        """
        nadir_mask = self.get_nadir_mask(par_scene).astype(int)
        row_filter = np.zeros((1, patch_size))
        row_filter[:, padding:-padding] = 1
        rows_fit = convolve(nadir_mask, row_filter, mode="constant") >= 1
        col_filter = np.zeros((patch_size, 1)) + 1
        nadir_validity = convolve(rows_fit.astype(int), col_filter, mode="constant") >= patch_size
        return nadir_validity

    def get_patch_labels(self, par_scene: PARASOLScene, patch_box: dict) -> dict:
        """Get the labels for a patch. Labels consist of all the available CLDCLASS data cropped to the patch.

        Arguments:
            par_scene: The PARASOL scene whose grid contains the patch
            patch_box: The patch description, containing the row space and column space of the patch
        Returns:
            labels: The labels for the provided patch
        """
        lon_c = (self.lon + par_scene.theta + 180) % 360 - 180
        row, col = latlon_to_polder_grid(self.lat, lon_c)
        row, col = row - par_scene.crop["top"], col - par_scene.crop["left"]
        top, bottom = patch_box["row_space"][0], patch_box["row_space"][-1]
        left, right = patch_box["col_space"][0], patch_box["col_space"][-1]
        lidar_idx = (row >= top) * (row < bottom) * (col >= left) * (col < right)
        labels = {
            "lat": self.lat[lidar_idx],
            "lon": self.lon[lidar_idx],
            "height": self.height[lidar_idx],
            "time": self.time[lidar_idx],
            "cloud_scenario": {k: self.cloud_scenario[k][lidar_idx] for k in self.cloud_scenario},
            "patch_idx": (row[lidar_idx] - top, col[lidar_idx] - left),
        }
        return labels
