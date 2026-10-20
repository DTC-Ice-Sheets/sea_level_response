"""
SLR emulator module.

This is a re-write of the module so that it follows the reference notebook (selrem_run.ipynb) step by step:

1. Bin the gridded VMB loads to 1 degree point loads.
2. For each point load, compute gdot / udot / rotation on a coarse "compute" grid with the SELREM fits.
3. Apply the self-attraction factor to gdot, then add the uniform mass-conservation term ``c`` to get ndot / sdot.
4. Mask land, interpolate to the fine target grid.

All emulator maths is done in **metres** (as in the notebook). The public ``compute_*`` functions convert the result
to millimetres by default (``output_units="mm"``), because the rest of the code base talks about mm.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from scipy.special import sph_harm_y

from sea_level_response.config import MASS_BALANCE_COL_NAME, MASS_BALANCE_ERROR_COL_NAME
from sea_level_response.selrem.numba_functions import accumulate_fields
from sea_level_response.selrem.selrem_helpers import load_ocean_mask, load_selrem_parameters

# ---------------------------------------------------------------------------
# Physics constants and settings (same values as the notebook)
# ---------------------------------------------------------------------------
R_EARTH_KM = 6371.0
GMSL_PER_GT_M = -1.0 / 362.0 / 1000.0  # metres of global-mean sea level per Gt (mass loss -> sea-level rise)
ROT_COEF = 0.20  # rotational feedback coefficient
BETA_SAL = 0.20  # self-attraction and loading coefficient
Y21_LON_SHIFT_DEG = 20.0  # longitude shift of the Y21 rotational pattern

TARGET_RES_DEG = 0.5  # output resolution
COMPUTE_RES_DEG = 2.0  # internal compute grid resolution
BIN_RES_DEG = 1.0  # resolution at which VMB loads are binned to point loads

_UNIT_FACTORS = {"m": 1.0, "mm": 1000.0}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def wrap180(lon_deg: np.ndarray | float) -> np.ndarray | float:
    """Wrap any longitude (e.g. 0-360) to [-180, 180)."""
    return ((lon_deg + 180.0) % 360.0) - 180.0


def _build_lonlat(res_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a regular global grid: lon = -180..180 (periodic, no duplicated end point), lat = 90..-90."""
    nlon = int(360 / res_deg)
    nlat = int(180 / res_deg)
    lons = np.linspace(-180, 180, nlon, endpoint=False)
    lats = np.linspace(90, -90, nlat)
    lon2d, lat2d = np.meshgrid(lons, lats, indexing="ij")
    return lons, lats, lon2d, lat2d


def _interp_ocean_mask_to(lons_t: np.ndarray, lats_t: np.ndarray, ocean_mask: np.ndarray) -> np.ndarray:
    """Interpolate the global ocean mask (assumed lon -180..180 without end point, lat 90..-90) to a target grid."""
    nlon_m, nlat_m = ocean_mask.shape
    lons_m = np.linspace(-180, 180, nlon_m, endpoint=False)
    lats_m = np.linspace(90, -90, nlat_m)

    itp = RegularGridInterpolator((lons_m, lats_m), ocean_mask, bounds_error=False, fill_value=0)
    lon2d_t, lat2d_t = np.meshgrid(lons_t, lats_t, indexing="ij")
    pts = np.stack([lon2d_t.ravel(), lat2d_t.ravel()], axis=-1)
    m = itp(pts).reshape(lon2d_t.shape)
    return (m > 0.5).astype(np.int8)


def _interp_to_grid(
    field: np.ndarray, lons_src: np.ndarray, lats_src: np.ndarray, lon2d_t: np.ndarray, lat2d_t: np.ndarray
) -> np.ndarray:
    """Linearly interpolate a (lon, lat) field to the target grid. Points outside the source grid become NaN."""
    itp = RegularGridInterpolator((lons_src, lats_src), field, bounds_error=False, fill_value=np.nan)
    pts = np.stack([lon2d_t.ravel(), lat2d_t.ravel()], axis=-1)
    return itp(pts).reshape(lon2d_t.shape)


# ---------------------------------------------------------------------------
# Step 1: binning of the loads
# ---------------------------------------------------------------------------
def bin_loads_to_points(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    mass_gt: np.ndarray,
    masserr_gt: np.ndarray,
    bin_res_deg: float = BIN_RES_DEG,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Bin loads into coarse lat/lon cells and return one point load per occupied cell (at the cell centre).

    Masses are summed. Errors are combined as sqrt(sum(err^2)), i.e. assuming independent errors.

    Parameters
    ----------
    lat_deg, lon_deg : np.ndarray
        Latitude / longitude of the loads in degrees (any shape; they are flattened).
    mass_gt, masserr_gt : np.ndarray
        Mass and mass error in Gt, same shape as lat/lon.
    bin_res_deg : float
        Bin size in degrees.

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


# ---------------------------------------------------------------------------
# Steps 2-4: the emulator
# ---------------------------------------------------------------------------
def run_emulator(
    lat_p_deg: np.ndarray,
    lon_p_deg: np.ndarray,
    mass_gt: np.ndarray,
    masserr_gt: np.ndarray,
    gdot_fit: np.ndarray,
    udot_fit: np.ndarray,
    gdot_unc: np.ndarray,
    udot_unc: np.ndarray,
    ocean_mask: np.ndarray,
    target_res_deg: float = TARGET_RES_DEG,
    compute_res_deg: float = COMPUTE_RES_DEG,
) -> dict[str, np.ndarray]:
    """
    Run the SELREM emulator for a set of point loads.

    Parameters
    ----------
    lat_p_deg, lon_p_deg : np.ndarray
        Load locations in degrees.
    mass_gt, masserr_gt : np.ndarray
        Loads and their uncertainties in Gt (positive = mass gain).
    gdot_fit, udot_fit, gdot_unc, udot_unc : np.ndarray
        SELREM fit and uncertainty parameters.
    ocean_mask : np.ndarray
        Global ocean mask, shape (nlon, nlat), 1 = ocean. Assumed lon -180..180 (no end point), lat 90..-90.
    target_res_deg : float
        Resolution of the output grid in degrees.
    compute_res_deg : float
        Resolution of the internal computation grid in degrees.

    Returns
    -------
    dict[str, np.ndarray]
        Keys: lons, lats (1D), ndot, udot, sdot, rot, sig_ndot, sig_udot, sig_sdot (2D, shape (nlon, nlat), in
        metres), ocean_mask (2D, target grid), gmsl (m, global-mean sea level of the input loads) and
        c (m, the uniform mass-conservation term).
    """
    # Compute and target grids
    lons_c, lats_c, lon2d_c, lat2d_c = _build_lonlat(compute_res_deg)
    lons_t, lats_t, lon2d_t, lat2d_t = _build_lonlat(target_res_deg)

    lat_c_rad = np.radians(lat2d_c)
    lon_c_rad = np.radians(lon2d_c)
    sin_lat_c = np.sin(lat_c_rad)
    cos_lat_c = np.cos(lat_c_rad)

    # Real part of Y21 (rotational pattern), shifted by 20 degrees in longitude and normalised to +-1
    theta = np.radians(90 - lat2d_c)  # colatitude
    phi = lon_c_rad + np.radians(Y21_LON_SHIFT_DEG)  # azimuth
    y21_real = np.real(sph_harm_y(2, 1, theta, phi))
    y21_real /= np.max(np.abs(y21_real))

    # Loads
    lat_p_rad = np.radians(np.asarray(lat_p_deg, dtype=np.float64))
    lon_p_rad = np.radians(wrap180(np.asarray(lon_p_deg, dtype=np.float64)))
    mass_gt = np.asarray(mass_gt, dtype=np.float64)
    masserr_gt = np.asarray(masserr_gt, dtype=np.float64)

    # Point loads -> compute grid
    gdot_c, udot_c, rot_c, gd_unc_c, ud_unc_c, gd_vmb_var_c, ud_vmb_var_c = accumulate_fields(
        lat_p_rad,
        lon_p_rad,
        mass_gt,
        masserr_gt,
        sin_lat_c,
        cos_lat_c,
        lon_c_rad,
        y21_real,
        np.asarray(gdot_fit, dtype=np.float64),
        np.asarray(udot_fit, dtype=np.float64),
        np.asarray(gdot_unc, dtype=np.float64),
        np.asarray(udot_unc, dtype=np.float64),
        R_EARTH_KM,
        GMSL_PER_GT_M,
        ROT_COEF,
    )

    # Uncertainties (computed *before* the self-attraction factor, as in the notebook)
    sig_gdot_c = np.sqrt(gd_unc_c**2 + gd_vmb_var_c)
    sig_udot_c = np.sqrt(ud_unc_c**2 + ud_vmb_var_c)
    sig_sdot_c = np.sqrt(sig_gdot_c**2 + sig_udot_c**2)

    # Self-attraction and loading factor
    gdot_c = gdot_c * (1.0 + BETA_SAL)

    # Global-mean sea level of the loads
    sle = GMSL_PER_GT_M * np.sum(mass_gt)

    # Area weights and ocean mask on compute grid
    area_w = np.cos(np.radians(lats_c))[None, :].repeat(len(lons_c), axis=0)
    ocean_mask_c = _interp_ocean_mask_to(lons_c, lats_c, ocean_mask)
    ocean = ocean_mask_c > 0

    # Uniform mass-conservation term: ocean-mean sdot must equal sle
    diff = gdot_c - udot_c
    mean_diff_ocean = np.sum(diff[ocean] * area_w[ocean]) / np.sum(area_w[ocean])
    c = sle - mean_diff_ocean

    ndot_c = gdot_c + c
    sdot_c = ndot_c - udot_c

    # Enforce ocean-mean sdot = sle (numerically this is ~0 after the step above, kept as in the notebook)
    mean_sdot_ocean = np.sum(sdot_c[ocean] * area_w[ocean]) / np.sum(area_w[ocean])
    sdot_c[ocean] += sle - mean_sdot_ocean
    ndot_c = sdot_c + udot_c

    sig_ndot_c = sig_gdot_c.copy()

    # Mask land on compute grid
    def _mask(field: np.ndarray) -> np.ndarray:
        return np.where(ocean, field, np.nan)

    fields_c = {
        "ndot": _mask(ndot_c),
        "udot": _mask(udot_c),
        "sdot": _mask(sdot_c),
        "rot": _mask(rot_c),
        "sig_ndot": _mask(sig_ndot_c),
        "sig_udot": _mask(sig_udot_c),
        "sig_sdot": _mask(sig_sdot_c),
    }

    # Compute grid -> target grid
    out: dict[str, np.ndarray] = {
        name: _interp_to_grid(field, lons_c, lats_c, lon2d_t, lat2d_t) for name, field in fields_c.items()
    }
    out["lons"] = lons_t
    out["lats"] = lats_t
    out["ocean_mask"] = _interp_ocean_mask_to(lons_t, lats_t, ocean_mask)
    out["gmsl"] = np.asarray(sle)
    out["c"] = np.asarray(c)
    return out


# ---------------------------------------------------------------------------
# Dataset-level functions
# ---------------------------------------------------------------------------
def _points_from_vmb(
    vmb: np.ndarray,
    vmb_err: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    scale_factor: float = 1.0,
    bin_res_deg: float = BIN_RES_DEG,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert VMB (kg) and its error (kg) at (lat, lon) to binned point loads in Gt."""
    mass_gt = np.asarray(vmb, dtype=np.float64) * scale_factor / 1e12
    masserr_gt = np.asarray(vmb_err, dtype=np.float64) * scale_factor / 1e12
    return bin_loads_to_points(lat, lon, mass_gt, masserr_gt, bin_res_deg)


def _error_or_zeros(da_source: xr.Dataset, like: xr.DataArray) -> xr.DataArray:
    """Return the mass-balance error variable if present, otherwise zeros shaped like ``like``."""
    if MASS_BALANCE_ERROR_COL_NAME in da_source:
        return da_source[MASS_BALANCE_ERROR_COL_NAME]
    return xr.zeros_like(like)


def compute_annual_slr(ds: xr.Dataset, scale_factor: float = 1.0, output_units: str = "mm") -> xr.Dataset:
    """
    Compute annual SELREM SLR results and return them as an xarray.Dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with dims ("time", "point"), variables MASS_BALANCE_COL_NAME (kg), optionally
        MASS_BALANCE_ERROR_COL_NAME (kg), and coordinates "y" (latitude, deg) and "x" (longitude, deg) along "point".
    scale_factor : float
        Scaling factor applied to mass balance and its error (default: 1.0).
    output_units : str
        "mm" (default) or "m".

    Returns
    -------
    xr.Dataset
        Dataset with dims ("time", "x", "y") containing ndot, udot, sdot, sig_ndot, sig_udot, sig_sdot, rot_total.
    """
    unit_factor = _UNIT_FACTORS[output_units]
    gdot_fit, udot_fit, gdot_unc, udot_unc = load_selrem_parameters()
    ocean_mask, _, _ = load_ocean_mask()

    time_vals = ds["time"].to_numpy()
    years = np.asarray(pd.to_datetime(time_vals).year, dtype=int)

    lat = ds["y"].to_numpy()
    lon = ds["x"].to_numpy()
    vmb_all = ds[MASS_BALANCE_COL_NAME]
    err_all = _error_or_zeros(ds, vmb_all)

    steps: list[dict[str, np.ndarray]] = []
    lons_t = lats_t = None
    for i in range(len(years)):
        lat_p, lon_p, m_p, me_p = _points_from_vmb(
            vmb_all.isel(time=i).to_numpy(), err_all.isel(time=i).to_numpy(), lat, lon, scale_factor
        )
        out = run_emulator(lat_p, lon_p, m_p, me_p, gdot_fit, udot_fit, gdot_unc, udot_unc, ocean_mask)
        lons_t, lats_t = out["lons"], out["lats"]
        steps.append(
            {
                "ndot": out["ndot"],
                "udot": out["udot"],
                "sdot": out["sdot"],
                "sig_ndot": out["sig_ndot"],
                "sig_udot": out["sig_udot"],
                "sig_sdot": out["sig_sdot"],
                "rot_total": out["rot"],
            }
        )

    data_vars = {
        key: (("time", "x", "y"), np.stack([s[key] for s in steps], axis=0) * unit_factor) for key in steps[0]
    }
    ds_out = xr.Dataset(
        data_vars=data_vars,
        coords={"time": np.array([np.datetime64(f"{y}-01-01") for y in years]), "x": lons_t, "y": lats_t},
    )
    ds_out.attrs["units"] = output_units
    return ds_out


def compute_global_slr(ds: xr.Dataset, scale_factor: float = 1.0, output_units: str = "mm") -> xr.Dataset:
    """
    Compute the global sea-level response from a VMB dataset.

    If the dataset has a "time" dimension, the mass balance (and its error) is first averaged over time and the
    emulator is run once on the mean. Otherwise the dataset is used as a single snapshot.

    Parameters
    ----------
    ds : xr.Dataset
        VMB dataset with dim "point" (and optionally "time"); see ``compute_annual_slr``.
    scale_factor : float
        Scaling factor for mass balance and its error (default: 1.0).
    output_units : str
        "mm" (default) or "m".

    Returns
    -------
    xr.Dataset
        Dataset with dims ("x", "y") containing sdot, ndot, udot, rot_total, sig_sdot, sig_ndot, sig_udot.
        The ocean-mean values are stored in ``attrs`` as gmsl_sdot, gmsl_ndot, gmsl_udot.
    """
    unit_factor = _UNIT_FACTORS[output_units]
    gdot_fit, udot_fit, gdot_unc, udot_unc = load_selrem_parameters()
    ocean_mask, _, _ = load_ocean_mask()

    vmb = ds[MASS_BALANCE_COL_NAME]
    err = _error_or_zeros(ds, vmb)
    if "time" in vmb.dims:
        vmb = vmb.mean(dim="time", skipna=True)
        err = err.mean(dim="time", skipna=True)

    lat_p, lon_p, m_p, me_p = _points_from_vmb(
        vmb.to_numpy(), err.to_numpy(), ds["y"].to_numpy(), ds["x"].to_numpy(), scale_factor
    )
    out = run_emulator(lat_p, lon_p, m_p, me_p, gdot_fit, udot_fit, gdot_unc, udot_unc, ocean_mask)

    lons_t, lats_t = out["lons"], out["lats"]
    ocean_t = out["ocean_mask"].astype(np.float64)
    area_w = np.repeat(np.cos(np.radians(lats_t))[np.newaxis, :], len(lons_t), axis=0)

    def ocean_mean(field: np.ndarray) -> float:
        # Same definition as the notebook: NaNs are dropped from the numerator only.
        return float(np.nansum(field * ocean_t * area_w) / np.nansum(ocean_t * area_w))

    data_vars = {
        "sig_sdot": (("x", "y"), out["sig_sdot"] * unit_factor),
        "sig_ndot": (("x", "y"), out["sig_ndot"] * unit_factor),
        "sig_udot": (("x", "y"), out["sig_udot"] * unit_factor),
        "rot_total": (("x", "y"), out["rot"] * unit_factor),
        "sdot": (("x", "y"), out["sdot"] * unit_factor),
        "ndot": (("x", "y"), out["ndot"] * unit_factor),
        "udot": (("x", "y"), out["udot"] * unit_factor),
    }
    ds_out = xr.Dataset(data_vars=data_vars, coords={"y": lats_t, "x": lons_t})
    ds_out.attrs["gmsl_ndot"] = ocean_mean(out["ndot"]) * unit_factor
    ds_out.attrs["gmsl_udot"] = ocean_mean(out["udot"]) * unit_factor
    ds_out.attrs["gmsl_sdot"] = ocean_mean(out["sdot"]) * unit_factor
    ds_out.attrs["units"] = output_units
    return ds_out
