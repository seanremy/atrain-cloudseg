import h5py
import numpy as np
from scipy.ndimage.filters import convolve


def scene_field_to_numpy_arr(h5_dataset: h5py._hl.dataset.Dataset, new_fill_value: float = -np.Inf) -> np.ndarray:
    """Convert a field (dataset) from a PARASOL/POLDER scene to a numpy array.

    Args:
        h5_dataset: A dataset from a PARASOL/POLDER scene
        new_fill_value: The new fill value to use, defaults to -Inf

    Returns:
        arr: The numpy array representation of the dataset
    """
    raw_arr = np.array(h5_dataset).astype(np.float32)
    arr = raw_arr * h5_dataset.attrs["scale_factor"] + h5_dataset.attrs["add_offset"]
    arr[raw_arr == h5_dataset.attrs["_FillValue"]] = new_fill_value
    return arr


def polder_grid_to_latlon(lin: np.array, col: np.array, rounding: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """TO DO: DOCUMENT AND UNIT TEST. See Appendix B here:
    web-backend.icare.univ-lille.fr//projects_data/parasol/docs/Parasol_Level-1_format_latest.pdf

    Args:
        lin: Row in the POLDER grid, 1 <= lin <= 3240
        col: Column in the POLDER grid, 1 <= col <= 6480
        rounding: True if you want to round, defaults to False

    Returns:
        lat: Latitude
        lon: Longitude
    """
    assert ((0 <= lin) * (lin < 3240)).all()
    assert ((0 <= col) * (col < 6480)).all()
    lat = 90 - (lin - 0.5) / 18
    n_i = 3240 * np.cos(lat * np.pi / 180)
    if rounding:
        n_i = np.round(n_i)
    lon = (180 / n_i) * (col - 3240.5)
    lon[np.abs(lon) > 180.0] = -np.Inf  # set invalid longitudes to -infinity
    return lat, lon


def latlon_to_polder_grid(lat: np.array, lon: np.array, rounding: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """TO DO: DOCUMENT AND UNIT TEST. See Appendix B here:
    web-backend.icare.univ-lille.fr//projects_data/parasol/docs/Parasol_Level-1_format_latest.pdf

    Args:
        lat: Latitude
        lon: Longitude
        rounding: True if you want to round, defaults to False

    Returns:
        lin: Row in the POLDER grid, 1 <= lin <= 3240
        col: Column in the POLDER grid, 1 <= col <= 6480
    """
    non_nan_lat, non_nan_lon = lat[~np.isnan(lat)], lon[~np.isnan(lon)]
    assert ((-180 <= non_nan_lat) * (non_nan_lat <= 180)).all()
    assert ((-180 <= non_nan_lon) * (non_nan_lon <= 180)).all()
    lin = 18 * (90 - lat) + 0.5
    if rounding:
        lin = np.round(lin)
    n_i = 3240 * np.sin((np.pi / 180) * (lin - 0.5) / 18)
    if rounding:
        n_i = np.round(n_i)
    col = 3240.5 + (n_i / 180) * lon
    if rounding:
        col = np.round(col)
    return lin, col


def center_polder_grid(lin: np.array, col: np.array, theta: float = None) -> tuple[np.ndarray, np.ndarray, float]:
    """Re-project the Sinusoidal POLDER grid to a new sinusoidal projection. If theta is provided, the central meridian
    of the new sinusoidal projection is the prime meridian + theta. Otherwise, the new central meridian is the midpoint
    between the min and max longitudes of the provided data.

    Args:
        lin: Row in the POLDER grid, 1 <= lin <= 3240
        col: Column in the POLDER grid, 1 <= col <= 6480
        theta: The angle by which to rotate the projection

    Returns:
        lin_: Row in the new projection, 1 <= lin <= 3240
        col_: Column in the new projection, 1 <= col <= 6480
        theta: The angle by which the projection was rotated
    """
    lat, lon = polder_grid_to_latlon(lin, col)
    if theta == None:
        theta = -(np.max(lon) + np.min(lon)) / 2
    lon_ = (lon + theta + 180) % 360 - 180
    lin_, col_ = latlon_to_polder_grid(lat, lon_)
    lin_, col_ = np.round(lin_).astype(int), np.round(col_).astype(int)
    return lin_, col_, theta


def get_fitting_box_centers(mask: np.ndarray, side_length: int) -> np.ndarray:
    """Get the mask of centers of boxes with provided side length that fit in the provided mask. Uses separable
    convolutions for speed. The per-box validity can be thought of as applying a 2D convolution with kernel size
    (patch_size, patch_size) over the mask, and then only keeping centers where the output is equal to patch_size^2.
    This function assumes the input mask uses -Inf instead of False, and only keeps non-infinite outputs. This function
    also leverages the fact that the 2D 'summation' convolution (all 1's) is separable into 2 1D convolutions, which
    are much faster to apply.

    Args:
        mask: A mask of which pixels are keepable, with -Inf instead of False
        side_length: The length of patches to sample

    Returns:
        fitting_centers: The mask of box centers with provided side length that fit in the provided mask
    """
    filter1, filter2 = np.zeros((1, side_length)) + 1, np.zeros((side_length, 1)) + 1
    fitting_centers = convolve(convolve(mask, filter1), filter2)
    fitting_centers = fitting_centers > -np.Inf
    return fitting_centers


def reproject_polder_griddata(
    griddata: np.ndarray,
    old_row: np.ndarray,
    old_col: np.ndarray,
    new_row: np.ndarray,
    new_col: np.ndarray,
    fill_value: float = -np.Inf,
) -> np.ndarray:
    """Re-project data from one POLDER grid into another, using the new indices.

    Args:
        griddata: Data in the original POLDER grid
        old_row: Rows in the original POLDER grid to be projected
        old_col: Columns in the original POLDER grid to be projected
        new_row: Rows in the new POLDER grid
        new_col: Columns in the new POLDER grid
        fill_value: The fill value to use for missing pixels

    Returns:
        reproj: The reprojected data
    """
    reproj = np.zeros_like(griddata) + fill_value
    reproj[new_row, new_col] = griddata[old_row, old_col]
    return reproj


def get_view_validity_from_scene(scene: h5py._hl.files.File, min_views: int, patch_size: int) -> np.ndarray:
    """Get the 'view validity' from a PARASOL/POLDER scene. The 'view validity' is the mask of all box centers for
    boxes of size (patch_size, patch_size) where all of the pixels in the box have at least min_views angles available.

    Args:
        scene: A PARASOL/POLDER scene
        min_views: The minimum number of views
        patch_size: The size of the patches for which to check validity
    """
    num_views = scene_field_to_numpy_arr(scene["Data_Fields"]["Nviews"])
    enough_views = num_views >= min_views
    valid_row, valid_col = np.where(enough_views)
    valid_row_, valid_col_, theta = center_polder_grid(valid_row, valid_col)
    enough_views_centered = reproject_polder_griddata(enough_views, valid_row, valid_col, valid_row_, valid_col_)
    validity_mask = get_fitting_box_centers(enough_views_centered, patch_size)
    return validity_mask
