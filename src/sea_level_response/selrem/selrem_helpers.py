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


def wrap180(lon_deg: np.ndarray | float) -> np.ndarray | float:
    """
    Wrap any longitude (e.g. 0-360) to [-180, 180).

    Parameters
    ----------
    lon_deg : np.ndarray | float
        Longitude in decimal degrees

    Returns
    -------
    np.ndarray | float
        Longitude in decimal degrees wrapped to [-180, 180)
    """
    return ((lon_deg + 180.0) % 360.0) - 180.0


def build_lonlat(
    res_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a regular global grid: lon = -180 to 180 (periodic), lat = 90 to -90.

    Parameters
    ----------
    res_deg : float
        Grid resolution in decimal degrees

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        (Longitude, Latitude, 2D Longitude, 2D Latitude)
    """
    nlon = int(360 / res_deg)
    nlat = int(180 / res_deg)
    lons = np.linspace(-180, 180, nlon, endpoint=False)
    lats = np.linspace(90, -90, nlat)
    lon2d, lat2d = np.meshgrid(lons, lats, indexing="ij")
    return lons, lats, lon2d, lat2d


def interp_ocean_mask_to(
    target_lons: np.ndarray,
    target_lats: np.ndarray,
    ocean_mask: np.ndarray,
    mask_lons: np.ndarray,
    mask_lats: np.ndarray,
) -> np.ndarray:
    """
    Reproject an ocean mask onto a new longitude/latitude grid using interpolation.

    Parameters
    ----------
    target_lons : np.ndarray
        Target longitudes for the new grid
    target_lats : np.ndarray
        Target latitudes for the new grid
    ocean_mask : np.ndarray
        Spatial grid masking ocean cells, shape (nlon, nlat)
    mask_lons : np.ndarray
        Longitudes of the ocean mask grid in decimal degrees
    mask_lats : np.ndarray
        Latitudes of the ocean mask grid in decimal degrees

    Returns
    -------
    np.ndarray
        The reprojected ocean mask on the target grid.
    """
    itp = RegularGridInterpolator((mask_lons, mask_lats), ocean_mask, bounds_error=False, fill_value=0)
    lon2d_t, lat2d_t = np.meshgrid(target_lons, target_lats, indexing="ij")
    pts = np.stack([lon2d_t.ravel(), lat2d_t.ravel()], axis=-1)
    m = itp(pts).reshape(lon2d_t.shape)
    return (m > 0.5).astype(np.int8)


def interp_to_grid(
    field: np.ndarray,
    lons_src: np.ndarray,
    lats_src: np.ndarray,
    lon2d_t: np.ndarray,
    lat2d_t: np.ndarray,
) -> np.ndarray:
    """
    Linearly interpolate a (lon, lat) field to the target grid.

    Points outside the source grid become NaN.

    Parameters
    ----------
    field : np.ndarray
        2D field on the source grid, shape (nlon, nlat)
    lons_src : np.ndarray
        Longitudes of the source grid in decimal degrees
    lats_src : np.ndarray
        Latitudes of the source grid in decimal degrees
    lon2d_t : np.ndarray
        2D longitudes of the target grid in decimal degrees
    lat2d_t : np.ndarray
        2D latitudes of the target grid in decimal degrees

    Returns
    -------
    np.ndarray
        The field interpolated onto the target grid, same shape as ``lon2d_t``.
    """
    itp = RegularGridInterpolator((lons_src, lats_src), field, bounds_error=False, fill_value=np.nan)
    pts = np.stack([lon2d_t.ravel(), lat2d_t.ravel()], axis=-1)
    return itp(pts).reshape(lon2d_t.shape)


def load_selrem_parameters() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load SELREM parameters from a JSON file.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        gdot_fit, udot_fit, gdot_unc, udot_unc arrays.
    """
    with Path(SELREM_PARAMS_PATH).open("r") as f:
        params = json.load(f)
    gdot_fit = np.array(params["gdot_fit"], dtype=np.float64)
    udot_fit = np.array(params["udot_fit"], dtype=np.float64)
    gdot_unc = np.array(params["gdot_unc"], dtype=np.float64)
    udot_unc = np.array(params["udot_unc"], dtype=np.float64)

    return gdot_fit, udot_fit, gdot_unc, udot_unc


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


def bin_loads_to_points(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    mass_gt: np.ndarray,
    masserr_gt: np.ndarray,
    bin_res_deg: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Bin loads into coarse lat/lon cells and return one point load per occupied cell (at the cell centre).

    Masses are summed. Errors are combined as sqrt(sum(err^2)), i.e. assuming independent errors.

    Parameters
    ----------
    lat_deg : np.ndarray
        Latitude of the loads in degrees (any shape; they are flattened).
    lon_deg : np.ndarray
        Longitude of the loads in degrees (any shape; they are flattened).
    mass_gt : np.ndarray
        Mass in Gt, same shape as lat/lon.
    masserr_gt : np.ndarray
        Mass error in Gt, same shape as lat/lon.
    bin_res_deg : float
        Resolution at which mass-balance loads are binned to point loads.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        lat_centres, lon_centres (degrees), mass (Gt), mass error (Gt) of the occupied bins.
    """
    lat = np.asarray(lat_deg, dtype=np.float64).ravel()
    lon = wrap180(np.asarray(lon_deg, dtype=np.float64).ravel())
    m = np.asarray(mass_gt, dtype=np.float64).ravel()
    me = np.asarray(masserr_gt, dtype=np.float64).ravel()

    ok = np.isfinite(m) & np.isfinite(lat) & np.isfinite(lon) & (np.abs(m) > 0.0)
    lat, lon, m, me = lat[ok], lon[ok], m[ok], me[ok]

    nlon_bins = int(np.ceil(360.0 / bin_res_deg))
    nlat_bins = int(np.ceil(180.0 / bin_res_deg))

    lon_bin = np.clip(np.floor((lon + 180.0) / bin_res_deg).astype(np.int64), 0, nlon_bins - 1)
    lat_bin = np.clip(np.floor((lat + 90.0) / bin_res_deg).astype(np.int64), 0, nlat_bins - 1)
    flat = lon_bin * nlat_bins + lat_bin

    unique_bins, inverse = np.unique(flat, return_inverse=True)
    m_sum = np.bincount(inverse, weights=m, minlength=unique_bins.size)
    var_sum = np.bincount(inverse, weights=me**2, minlength=unique_bins.size)

    keep = np.abs(m_sum) > 0.0
    unique_bins, m_sum, var_sum = unique_bins[keep], m_sum[keep], var_sum[keep]

    jlat = unique_bins % nlat_bins
    jlon = unique_bins // nlat_bins
    lon_cent = -180.0 + (jlon + 0.5) * bin_res_deg
    lat_cent = -90.0 + (jlat + 0.5) * bin_res_deg

    return lat_cent, lon_cent, m_sum, np.sqrt(var_sum)


def compute_mean_mass_balance_over_time_window(
    ds: xr.Dataset, start_time: datetime | None = None, end_time: datetime | None = None
) -> xr.Dataset:
    """
    Compute the mean mass balance and, if present, its error over a time range, collapsing the time dimension.

    Parameters
    ----------
    ds : xr.Dataset
        Input mass balance dataset with a time dimension. The mass-balance error variable is optional.
    start_time : datetime | None
        Start time for slicing, by default None. If both start_time and end_time are provided, the dataset is sliced
        accordingly before computing the mean. Otherwise, the entire time range is used.
    end_time : datetime | None
        End time for slicing, by default None. If both start_time and end_time are provided, the dataset is sliced
        accordingly before computing the mean. Otherwise, the entire time range is used.

    Returns
    -------
    xr.Dataset
        Dataset with the same spatial dimensions ('point', 'x', 'y'), where the 'time' dimension has been removed and
        each variable represents the mean value across the specified years. The error variable is only included if it
        was present in the input.
    """
    if start_time and end_time:
        ds = ds.sel(time=slice(start_time, end_time))

    data_vars = {
        MASS_BALANCE_COL_NAME: (["point"], ds[MASS_BALANCE_COL_NAME].mean(dim="time", skipna=True).data),
    }
    if MASS_BALANCE_ERROR_COL_NAME in ds:
        data_vars[MASS_BALANCE_ERROR_COL_NAME] = (
            ["point"],
            ds[MASS_BALANCE_ERROR_COL_NAME].mean(dim="time", skipna=True).data,
        )

    return xr.Dataset(
        data_vars,
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
