"""SELREM helper module."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

from sea_level_response.config import (
    MASS_BALANCE_COL_NAME,
    MASS_BALANCE_ERROR_COL_NAME,
    OCEAN_MASK_PATH,
    SELREM_PARAMS_PATH,
)


def load_selrem_parameters() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load SELREM parameters from a JSON file.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ndot_fit, udot_fit, ndot_unc, udot_unc arrays.
    """
    with Path(SELREM_PARAMS_PATH).open("r") as f:
        params = json.load(f)
    ndot_fit = np.array(params["ndot_fit"])
    udot_fit = np.array(params["udot_fit"])
    ndot_unc = np.array(params["ndot_unc"])
    udot_unc = np.array(params["udot_unc"])

    return ndot_fit, udot_fit, ndot_unc, udot_unc


def load_ocean_mask() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the ocean mask from a NetCDF file.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing the ocean mask array, longitude array, and latitude array.
    """
    ds = xr.open_dataset(OCEAN_MASK_PATH)
    ocean_mask = ds["ocean_mask"].to_numpy()
    lons = ds["lon"].to_numpy()
    lats = ds["lat"].to_numpy()

    return ocean_mask, lons, lats


def reproject_ocean_mask(
    ocean_mask: np.ndarray,
    mask_lons: np.ndarray,
    mask_lats: np.ndarray,
    target_lons: np.ndarray,
    target_lats: np.ndarray,
) -> np.ndarray:
    """
    Reproject an ocean mask onto a new longitude/latitude grid using interpolation.

    Parameters
    ----------
    ocean_mask : np.ndarray
        The original ocean mask array.
    mask_lons : np.ndarray
        Longitudes of the original mask grid.
    mask_lats : np.ndarray
        Latitudes of the original mask grid.
    target_lons : np.ndarray
        Target longitudes for the new grid.
    target_lats : np.ndarray
        Target latitudes for the new grid.

    Returns
    -------
    np.ndarray
        The reprojected ocean mask on the target grid.
    """
    lon2d, lat2d = np.meshgrid(target_lons, target_lats, indexing="ij")
    points = np.stack([lon2d.flatten(), lat2d.flatten()], axis=-1)
    interp = RegularGridInterpolator((mask_lons, mask_lats), ocean_mask, bounds_error=False, fill_value=0)
    ocean_mask_reprojected = interp(points).reshape(lon2d.shape)
    return (ocean_mask_reprojected > 0.5).astype(int)


def slice_and_prepare_dataset(
    vmb_dataset_path: Path | str,
    start_time: datetime,
    end_time: datetime,
) -> xr.Dataset:
    """
    Load, slice, and prepare the VMB dataset for SELREM computation.

    Parameters
    ----------
    vmb_dataset_path : Path | str
        Path to the VMB GeoZarr dataset.
    start_time : datetime
        Start time for slicing.
    end_time : datetime
        End time for slicing.

    Returns
    -------
    xr.Dataset
        Sliced and prepared VMB dataset.

    Raises
    ------
    ValueError
        If the dataset does not contain a 'time' dimension or if the time range is invalid
    """
    if "time" not in xr.open_zarr(vmb_dataset_path).dims:
        raise ValueError("The VMB dataset does not contain a 'time' dimension.")
    if start_time >= end_time:
        raise ValueError("start_time must be earlier than end_time.")
    ds = xr.open_zarr(vmb_dataset_path)
    ds = ds.sel(time=slice(start_time, end_time))
    if ds.time.size == 0:
        raise ValueError("The sliced dataset has no time points. Please check the provided time range.")
    return ds


def compute_mean_mass_balance_over_time_window(
    ds: xr.Dataset, start_time: datetime | None = None, end_time: datetime | None = None
) -> xr.Dataset:
    """
    Compute the mean mass balance and associated error over the selected time range, collapsing the time dimension.

    Parameters
    ----------
    ds : xr.Dataset
        Input mass balance dataset with a time dimension.
    start_time : datetime | None, optional
        Start time for slicing, by default None. If both start_time and end_time are provided, the dataset is sliced
        accordingly before computing the mean. Otherwise, the entire time range is used.
    end_time : datetime | None, optional
        End time for slicing, by default None. If both start_time and end_time are provided, the dataset is sliced
        accordingly before computing the mean. Otherwise, the entire time range is used.

    Returns
    -------
    xr.Dataset
        Dataset with the same spatial dimensions ('point', 'x', 'y'), where the 'time' dimension has been removed and
        each variable represents the mean value across the specified years.
    """
    if start_time and end_time:
        ds = ds.sel(time=slice(start_time, end_time))
    return xr.Dataset(
        {
            MASS_BALANCE_COL_NAME: (["point"], ds[MASS_BALANCE_COL_NAME].mean(dim="time", skipna=True).data),
            MASS_BALANCE_ERROR_COL_NAME: (
                ["point"],
                ds[MASS_BALANCE_ERROR_COL_NAME].mean(dim="time", skipna=True).data,
            ),
        },
        coords={
            "y": ds["y"].data,
            "x": ds["x"].data,
            "point": ds["point"].data,
        },
    )


def preprocess_location_timeseries(
    annual_slr_ds: xr.Dataset,
    lat0: float,
    lon0: float,
) -> dict:
    """
    Preprocess annual SLR data for a specific location, returning plot-ready time series.

    This function extracts time series data at the nearest grid point to the specified coordinates,
    computes cumulative sums, calculates linear trends, and returns data ready for plotting.

    Parameters
    ----------
    annual_slr_ds : xr.Dataset
        Annual SLR dataset containing 'ndot', 'udot', 'sdot' and their uncertainties.
    lat0 : float
        Target latitude in decimal degrees (-90 to 90).
    lon0 : float
        Target longitude in decimal degrees (-180 to 180).

    Returns
    -------
    dict
        Dictionary containing:
        - 'years': List of year labels (integers)
        - 'sdot_cumulative': Cumulative relative SLR (mm)
        - 'ndot_cumulative': Cumulative absolute SLR (mm)
        - 'udot_cumulative': Cumulative vertical deformation (mm)
        - 'sig_sdot_cumulative': Cumulative uncertainty for sdot (mm)
        - 'sig_ndot_cumulative': Cumulative uncertainty for ndot (mm)
        - 'sig_udot_cumulative': Cumulative uncertainty for udot (mm)
        - 'slope_sdot': Linear trend for relative SLR (mm/yr)
        - 'slope_ndot': Linear trend for absolute SLR (mm/yr)
        - 'slope_udot': Linear trend for vertical deformation (mm/yr)
        - 'actual_lat': Actual latitude of nearest grid point
        - 'actual_lon': Actual longitude of nearest grid point
    """
    # Extract coordinates and time
    x_coords = annual_slr_ds["x"].to_numpy()  # Longitude
    y_coords = annual_slr_ds["y"].to_numpy()  # Latitude
    time_coords = annual_slr_ds["time"].to_numpy()
    years = pd.to_datetime(time_coords).year
    years_numeric = years.astype(float)

    # Convert longitude if needed
    if x_coords.min() >= 0 and x_coords.max() > 180 and lon0 < 0:
        lon0_adjusted = lon0 + 360
    elif x_coords.min() < 0 and lon0 > 180:
        lon0_adjusted = lon0 - 360
    else:
        lon0_adjusted = lon0

    # Find nearest grid point
    lon2d, lat2d = np.meshgrid(x_coords, y_coords, indexing="ij")
    dist2 = (lat2d - lat0) ** 2 + (lon2d - lon0_adjusted) ** 2
    ilon, ilat = np.unravel_index(np.argmin(dist2), dist2.shape)

    actual_lon = float(lon2d[ilon, ilat])
    actual_lat = float(lat2d[ilon, ilat])

    # Extract time series data at this location
    ndot_series = np.array([annual_slr_ds["ndot"].isel(time=i).to_numpy()[ilon, ilat] for i in range(len(time_coords))])
    udot_series = np.array([annual_slr_ds["udot"].isel(time=i).to_numpy()[ilon, ilat] for i in range(len(time_coords))])
    sdot_series = np.array([annual_slr_ds["sdot"].isel(time=i).to_numpy()[ilon, ilat] for i in range(len(time_coords))])

    # Extract uncertainty series
    sig_ndot_series = np.array(
        [annual_slr_ds["sig_ndot"].isel(time=i).to_numpy()[ilon, ilat] for i in range(len(time_coords))]
    )
    sig_udot_series = np.array(
        [annual_slr_ds["sig_udot"].isel(time=i).to_numpy()[ilon, ilat] for i in range(len(time_coords))]
    )
    sig_sdot_series = np.array(
        [annual_slr_ds["sig_sdot"].isel(time=i).to_numpy()[ilon, ilat] for i in range(len(time_coords))]
    )

    # Accumulate over time (cumulative sum)
    ndot_cum = np.cumsum(ndot_series)
    udot_cum = np.cumsum(udot_series)
    sdot_cum = np.cumsum(sdot_series)

    # Cumulative uncertainties
    sig_ndot_cum = np.sqrt(np.cumsum(sig_ndot_series**2))
    sig_udot_cum = np.sqrt(np.cumsum(sig_udot_series**2))
    sig_sdot_cum = np.sqrt(np.cumsum(sig_sdot_series**2))

    # Filter to whole years only
    whole_years_mask = years_numeric % 1 == 0
    year_labels = years_numeric[whole_years_mask].astype(int).tolist()

    # Calculate linear trends (mm/yr)
    slope_ndot, _ = np.polyfit(years_numeric[whole_years_mask], ndot_cum[whole_years_mask], 1)
    slope_udot, _ = np.polyfit(years_numeric[whole_years_mask], udot_cum[whole_years_mask], 1)
    slope_sdot, _ = np.polyfit(years_numeric[whole_years_mask], sdot_cum[whole_years_mask], 1)

    return {
        "years": year_labels,
        "sdot_cumulative": sdot_cum[whole_years_mask].tolist(),
        "ndot_cumulative": ndot_cum[whole_years_mask].tolist(),
        "udot_cumulative": udot_cum[whole_years_mask].tolist(),
        "sig_sdot_cumulative": sig_sdot_cum[whole_years_mask].tolist(),
        "sig_ndot_cumulative": sig_ndot_cum[whole_years_mask].tolist(),
        "sig_udot_cumulative": sig_udot_cum[whole_years_mask].tolist(),
        "slope_sdot": float(slope_sdot),
        "slope_ndot": float(slope_ndot),
        "slope_udot": float(slope_udot),
        "actual_lat": actual_lat,
        "actual_lon": actual_lon,
    }
