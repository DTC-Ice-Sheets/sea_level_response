"""SLR emulator module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from scipy.special import sph_harm_y

from sea_level_response.config import EARTH_RADIUS_M, MASS_BALANCE_COL_NAME, MASS_BALANCE_ERROR_COL_NAME
from sea_level_response.selrem.numba_functions import accumulate_fields
from sea_level_response.selrem.selrem_helpers import (
    compute_mean_mass_balance_over_time_window,
    load_ocean_mask,
    load_selrem_parameters,
    reproject_ocean_mask,
)


def run_emulator(
    loading_points: pd.DataFrame | dict,
    ndot_fit: np.ndarray,
    udot_fit: np.ndarray,
    ocean_mask: np.ndarray,
    ndot_unc: np.ndarray,
    udot_unc: np.ndarray,
    target_res: float = 0.5,
    compute_res: float = 5.0,
) -> tuple:
    """
    Compute the necessary parameters for the emulator.

    Parameters
    ----------
    loading_points : pd.DataFrame | dict
        DataFrame or dictionary containing loading point data with keys/columns:
        'lat' (latitude in degrees), 'lon' (longitude in degrees),
        'Mass(Gt)' (mass in gigatonnes), and optionally 'Mass_err(Gt)' (mass error in gigatonnes).
    ndot_fit : np.ndarray
        SELREM fit parameters for ndot.
    udot_fit : np.ndarray
        SELREM fit parameters for udot.
    ocean_mask : np.ndarray
        2D array representing the ocean mask (1 for ocean, 0 for land).
    ndot_unc : np.ndarray
        SELREM uncertainty parameters for ndot.
    udot_unc : np.ndarray
        SELREM uncertainty parameters for udot.
    target_res : float
        Target resolution in degrees for the final output grid, by default 0.5.
    compute_res : float
        Computation resolution in degrees for intermediate calculations, by default 5.0.

    Returns
    -------
    tuple
        Tuple containing the following elements:
        - lat_arr : np.ndarray
            1D array of latitude (in radians) for the loading points.
        - lon_arr : np.ndarray
            1D array of longitude (in radians) for the loading points.
        - mass_arr : np.ndarray
            1D array of mass (in Gt) for the loading points.
        - mass_err_arr : np.ndarray
            1D array of mass error (in Gt) for the loading points.
        - sin_lat : np.ndarray
            2D array of sine of latitude for the target grid.
        - cos_lat : np.ndarray
            2D array of cosine of latitude for the target grid.
        - lon_rad : np.ndarray
            2D array of longitude (in radians) for the target grid.
        - y21_real : np.ndarray
            2D array of real part of spherical harmonic Y21 for the target grid.
        - ndot_fit : np.ndarray
            SELREM fit parameters for ndot.
        - udot_fit : np.ndarray
            SELREM fit parameters for udot.
        - ndot_unc : np.ndarray
            SELREM uncertainty parameters for ndot.
        - udot_unc : np.ndarray
            SELREM uncertainty parameters for udot.
    """
    nlon_c = int(360 / compute_res)
    nlat_c = int(180 / compute_res)
    lons_c = np.linspace(-180, 180, nlon_c)
    lats_c = np.linspace(90, -90, nlat_c)
    lon2d_c, lat2d_c = np.meshgrid(lons_c, lats_c, indexing="ij")

    nlon_mask, nlat_mask = ocean_mask.shape
    lons_mask = np.linspace(-180, 180, nlon_mask)
    lats_mask = np.linspace(90, -90, nlat_mask)

    interp_mask = RegularGridInterpolator((lons_mask, lats_mask), ocean_mask, bounds_error=False, fill_value=0)
    pts_c = np.stack([lon2d_c.flatten(), lat2d_c.flatten()], axis=-1)
    ocean_mask_c = interp_mask(pts_c).reshape(lon2d_c.shape)
    ocean_mask_c = (ocean_mask_c > 0.5).astype(int)

    lon_rad = np.radians(lon2d_c)
    lat_rad = np.radians(lat2d_c)
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)

    theta = np.radians(90 - lat2d_c)
    phi = lon_rad
    y21 = sph_harm_y(2, 1, theta, phi)
    y21_real = np.real(y21)
    y21_real /= np.max(np.abs(y21_real))

    # Accept dict or DataFrame
    if isinstance(loading_points, dict):
        lat_arr = np.radians(np.array(loading_points["y"], dtype=float))
        lon_arr = np.radians(np.array(loading_points["x"], dtype=float))
        mass_arr = np.array(loading_points["Mass(Gt)"], dtype=float)
        mass_err_arr = np.array(loading_points.get("Mass_err(Gt)", np.zeros_like(mass_arr)), dtype=float)
    else:  # DataFrame
        lat_arr = np.radians(loading_points["y"].to_numpy().astype(float))
        lon_arr = np.radians(loading_points["x"].to_numpy().astype(float))
        mass_arr = loading_points["Mass(Gt)"].to_numpy().astype(float)
        mass_err_arr = (
            loading_points["Mass_err(Gt)"].to_numpy().astype(float)
            if "Mass_err(Gt)" in loading_points
            else np.zeros_like(mass_arr)
        )

    # Filter out NaNs
    mask = ~np.isnan(lat_arr) & ~np.isnan(lon_arr) & ~np.isnan(mass_arr)
    lat_arr = lat_arr[mask]
    lon_arr = lon_arr[mask]
    mass_arr = mass_arr[mask]
    mass_err_arr = mass_err_arr[mask]

    (
        ndot_total,
        udot_total,
        rot_total,
        ndot_radial_unc_total,
        udot_radial_unc_total,
        ndot_vmb_unc_total,
        udot_vmb_unc_total,
    ) = accumulate_fields(
        lat_arr,
        lon_arr,
        mass_arr,
        mass_err_arr,
        sin_lat,
        cos_lat,
        lon_rad,
        y21_real,
        ndot_fit,
        udot_fit,
        ndot_unc,
        udot_unc,
    )

    sdot_total = ndot_total - udot_total

    ndot_vmb_unc_total = np.sqrt(ndot_vmb_unc_total)
    udot_vmb_unc_total = np.sqrt(udot_vmb_unc_total)
    sdot_vmb_unc_total = np.sqrt(ndot_vmb_unc_total**2 + udot_vmb_unc_total**2)

    sdot_radial_unc_total = np.sqrt(ndot_radial_unc_total**2 + udot_vmb_unc_total**2)

    ndot_unc_total = np.sqrt(ndot_radial_unc_total**2 + ndot_vmb_unc_total**2)
    udot_unc_total = np.sqrt(udot_radial_unc_total**2 + udot_vmb_unc_total**2)
    sdot_unc_total = np.sqrt(sdot_vmb_unc_total**2 + sdot_radial_unc_total**2)

    dlon = np.radians(lons_c[1] - lons_c[0])
    dlat = np.radians(lats_c[1] - lats_c[0])

    weights = np.cos(np.radians(lats_c))
    area_weights = np.repeat(weights[np.newaxis, :], len(lons_c), axis=0)

    a_ocean = np.sum(ocean_mask_c * area_weights) * (EARTH_RADIUS_M**2) * dlon * dlat
    total_mass = mass_arr.sum()
    delta_h = (total_mass * 1e12) / (1000.00 * a_ocean) * 1000

    mean_sdot = np.sum(sdot_total * ocean_mask_c * area_weights) / np.sum(ocean_mask_c * area_weights)

    factor = delta_h - mean_sdot
    ndot_corr = ndot_total.copy()
    sdot_corr = sdot_total.copy()
    udot_corr = udot_total.copy()

    ndot_corr[ocean_mask_c == 1] += factor
    sdot_corr[ocean_mask_c == 1] += factor
    udot_corr = -(sdot_corr - ndot_corr)

    nlon_t = int(360 / target_res)
    nlat_t = int(180 / target_res)
    lons_t = np.linspace(-180, 180, nlon_t)
    lats_t = np.linspace(90, -90, nlat_t)
    lon2d_t, lat2d_t = np.meshgrid(lons_t, lats_t, indexing="ij")

    def _interp_field(field_c: np.ndarray) -> np.ndarray:
        interp = RegularGridInterpolator((lons_c, lats_c), field_c, bounds_error=False, fill_value=np.nan)
        pts = np.stack([lon2d_t.flatten(), lat2d_t.flatten()], axis=-1)
        return interp(pts).reshape(lon2d_t.shape)

    ndot_corr_t = _interp_field(ndot_corr)
    udot_corr_t = _interp_field(udot_corr)
    sdot_corr_t = _interp_field(sdot_corr)
    rot_total_t = _interp_field(rot_total)

    sig_ndot_t = _interp_field(ndot_unc_total)
    sig_udot_t = _interp_field(udot_unc_total)
    sig_sdot_t = _interp_field(sdot_unc_total)

    return (
        ndot_corr_t,
        udot_corr_t,
        sdot_corr_t,
        rot_total_t,
        sig_ndot_t,
        sig_udot_t,
        sig_sdot_t,
        delta_h,
        factor,
        lons_t,
        lats_t,
    )


def _prepare_vmb_points(
    ds: xr.Dataset, scale_factor: float = 1.0
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    Prepare mass change points from a CSV-based xarray.Dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing mass balance data. Should contain mass balance in kilograms.
    scale_factor : float
        Factor to scale the mass balance values, by default 1.0

    Returns
    -------
    tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]
        Dictionary with keys "y", "x", "Mass(Gt)", latitude array, longitude array.
    """
    if "time" in ds.dims:
        ds = compute_mean_mass_balance_over_time_window(ds)
    lat = ds["y"].to_numpy()
    lon = ds["x"].to_numpy()
    mass = ds[MASS_BALANCE_COL_NAME].to_numpy() * scale_factor  # Already mean over selected years
    # Convert kg to Gt if needed (assuming VMB is in kg)
    mass_gt = mass / 1e12
    points = {"y": lat, "x": lon, "Mass(Gt)": mass_gt}
    return points, lat, lon


def compute_annual_slr(ds: xr.Dataset, scale_factor: float = 1.0) -> xr.Dataset:
    """
    Compute annual SELREM SLR results and return as an xarray.Dataset.

    Parameters
    ----------
    ds : xr.Dataset
        xarray.Dataset with dims ("point", "year"), variables "VMB", "VMBer", "y", "x"
    scale_factor : float
        Scaling factor to apply to mass balance values (default: 1.0).

    Returns
    -------
    xr.Dataset
        xarray.Dataset with dims ("year", "y", "x") containing:
        - ndot, udot, sdot
        - sig_ndot, sig_udot, sig_sdot
        - rot_total
    """
    # Load parameters and mask
    ndot_fit, udot_fit, ndot_unc, udot_unc = load_selrem_parameters()
    ocean_mask, _, _ = load_ocean_mask()

    years = pd.to_datetime(ds["time"].to_numpy()).year
    output_data = []
    lons_t, lats_t = None, None

    for i, _ in enumerate(years):
        # Extract VMB for this year
        vmb_year = ds[MASS_BALANCE_COL_NAME].isel(time=i).to_numpy()
        vmb_er_year = (
            ds[MASS_BALANCE_ERROR_COL_NAME].isel(time=i).to_numpy()
            if MASS_BALANCE_ERROR_COL_NAME in ds
            else np.zeros_like(vmb_year)
        )
        lat = ds["y"].to_numpy()
        lon = ds["x"].to_numpy()

        # Build a temporary dataset for the emulator
        ds_year = xr.Dataset(
            {
                MASS_BALANCE_COL_NAME: (("point",), vmb_year),
                MASS_BALANCE_ERROR_COL_NAME: (("point",), vmb_er_year),
                "y": (("point",), lat),
                "x": (("point",), lon),
            },
            coords={"point": ds["point"].to_numpy()},
        )

        # Prepare points
        points, _, _ = _prepare_vmb_points(ds_year, scale_factor=scale_factor)

        # Run emulator
        ndot_corr, udot_corr, sdot_corr, rot_total, sig_ndot, sig_udot, sig_sdot, *_unused, lons_t, lats_t = (
            run_emulator(points, ndot_fit, udot_fit, ocean_mask, ndot_unc, udot_unc, target_res=0.5, compute_res=5.0)
        )

        output_data.append(
            {
                "ndot": ndot_corr,
                "udot": udot_corr,
                "sdot": sdot_corr,
                "sig_ndot": sig_ndot,
                "sig_udot": sig_udot,
                "sig_sdot": sig_sdot,
                "rot_total": rot_total,
            }
        )

    # Stack results into xarray
    data_vars = {}
    for key in output_data[0]:
        # shape: (year, lon, lat)
        stacked = np.stack([d[key] for d in output_data], axis=0)
        data_vars[key] = (("time", "x", "y"), stacked)

    return xr.Dataset(
        data_vars=data_vars,
        coords={"time": np.array([np.datetime64(f"{y}-01-01") for y in years]), "x": lons_t, "y": lats_t},
    )


def compute_global_slr(ds: xr.Dataset, scale_factor: float = 1.0) -> xr.Dataset:
    """
    Compute global sea-level response from a VMB dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Single-snapshot VMB dataset (no 'year' dimension required).
    scale_factor : float
        Scaling factor for mass balance, default 1.0.

    Returns
    -------
    xr.Dataset
        {
            "sdot": (lat, lon) gridded relative SLR,
            "ndot": (lat, lon) absolute SLR,
            "udot": (lat, lon) vertical deformation,
            "sig_sdot": (lat, lon) uncertainty,
            "global_mean_sdot": scalar,
            "global_mean_ndot": scalar,
            "global_mean_udot": scalar
        }
    """
    # Load SELREM parameters and ocean mask
    ndot_fit, udot_fit, ndot_unc, udot_unc = load_selrem_parameters()
    ocean_mask, mask_lons, mask_lats = load_ocean_mask()

    # Prepare VMB points
    points, _, _ = _prepare_vmb_points(ds, scale_factor)

    # Run emulator
    ndot_corr, udot_corr, sdot_corr, rot_total, sig_ndot, sig_udot, sig_sdot, _, _, lons_t, lats_t = run_emulator(
        points, ndot_fit, udot_fit, ocean_mask, ndot_unc, udot_unc, target_res=0.5, compute_res=5.0
    )

    # Reproject ocean mask to emulator grid
    ocean_mask_on_emulator_grid = reproject_ocean_mask(ocean_mask, mask_lons, mask_lats, lons_t, lats_t)

    # Area weights
    weights = np.cos(np.radians(lats_t))
    area_weights = np.repeat(weights[np.newaxis, :], len(lons_t), axis=0)

    # Global mean SLR
    gmsl_sdot = np.nansum(sdot_corr * ocean_mask_on_emulator_grid * area_weights) / np.nansum(
        ocean_mask_on_emulator_grid * area_weights
    )
    gmsl_ndot = np.nansum(ndot_corr * ocean_mask_on_emulator_grid * area_weights) / np.nansum(
        ocean_mask_on_emulator_grid * area_weights
    )
    gmsl_udot = np.nansum(udot_corr * ocean_mask_on_emulator_grid * area_weights) / np.nansum(
        ocean_mask_on_emulator_grid * area_weights
    )

    # Pack results into xarray.Dataset
    data_vars = {
        "sig_sdot": (("x", "y"), sig_sdot),
        "sig_ndot": (("x", "y"), sig_ndot),
        "sig_udot": (("x", "y"), sig_udot),
        "rot_total": (("x", "y"), rot_total),
        "sdot": (("x", "y"), sdot_corr),
        "ndot": (("x", "y"), ndot_corr),
        "udot": (("x", "y"), udot_corr),
    }

    ds_out = xr.Dataset(data_vars=data_vars, coords={"y": lats_t, "x": lons_t})
    ds_out.attrs["gmsl_ndot"] = gmsl_ndot
    ds_out.attrs["gmsl_udot"] = gmsl_udot
    ds_out.attrs["gmsl_sdot"] = gmsl_sdot

    return ds_out
