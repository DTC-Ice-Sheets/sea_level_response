"""SLR emulator module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from scipy.special import sph_harm_y

from sea_level_response.config import (
    ASSUMED_MASS_BALANCE_UNCERTAINTY_FRACTION,
    BETA_SAL,
    GMSL_PER_GT_M,
    MASS_BALANCE_COL_NAME,
    MASS_BALANCE_ERROR_COL_NAME,
    UNIT_FACTORS,
    Y21_LON_SHIFT_DEG,
)
from sea_level_response.selrem.numba_functions import accumulate_fields
from sea_level_response.selrem.selrem_helpers import (
    bin_loads_to_points,
    build_lonlat,
    compute_mean_mass_balance_over_time_window,
    interp_ocean_mask_to,
    interp_to_grid,
    load_ocean_mask,
    load_selrem_parameters,
    wrap180,
)


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
    target_res_deg: float = 0.5,
    compute_res_deg: float = 2.0,
) -> dict[str, np.ndarray]:
    """
    Run the SELREM emulator for a set of point loads.

    Parameters
    ----------
    lat_p_deg : np.ndarray
        Load location latitudes in degrees.
    lon_p_deg : np.ndarray
        Load location longitudes in degrees.
    mass_gt : np.ndarray
        Load in Gt (positive = mass gain).
    masserr_gt : np.ndarray
        Load uncertainties in Gt
    gdot_fit : np.ndarray
        SELREM fit for rate of change of the geoid.
    udot_fit : np.ndarray
        SELREM fit for rate of vertical land motion.
    gdot_unc : np.ndarray
        SELREM fit uncertainty for rate of change of the geoid.
    udot_unc : np.ndarray
        SELREM fit uncertainty for rate of vertical land motion.
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
    lons_c, lats_c, lon2d_c, lat2d_c = build_lonlat(compute_res_deg)
    lons_t, lats_t, lon2d_t, lat2d_t = build_lonlat(target_res_deg)

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
    ocean_mask_c = interp_ocean_mask_to(lons_c, lats_c, ocean_mask)
    ocean = ocean_mask_c > 0

    # Uniform mass-conservation term: ocean-mean sdot must equal sle
    diff = gdot_c - udot_c
    mean_diff_ocean = np.sum(diff[ocean] * area_w[ocean]) / np.sum(area_w[ocean])
    c = sle - mean_diff_ocean

    ndot_c = gdot_c + c
    sdot_c = ndot_c - udot_c

    # Enforce ocean-mean sdot = sle
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
        name: interp_to_grid(field, lons_c, lats_c, lon2d_t, lat2d_t) for name, field in fields_c.items()
    }
    out["lons"] = lons_t
    out["lats"] = lats_t
    out["ocean_mask"] = interp_ocean_mask_to(lons_t, lats_t, ocean_mask)
    out["gmsl"] = np.asarray(sle)
    out["c"] = np.asarray(c)
    return out


def _prepare_vmb_points(
    ds: xr.Dataset, scale_factor: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """
    Convert a VMB dataset (kg) into binned point loads (Gt) for the emulator.

    If ds has no mass-balance error variable, a pre-defined uncertainty
    (in percent of the mass change) of each grid cell is assumed.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with dim "point", variable MASS_BALANCE_COL_NAME (kg), optionally MASS_BALANCE_ERROR_COL_NAME (kg),
        and coordinates "y" (latitude, deg) and "x" (longitude, deg) along "point".
    scale_factor : float
        Factor to scale the mass balance values (and their error), by default 1.0.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]
        Latitude (deg), longitude (deg), mass (Gt) and mass error (Gt) of the binned loads, and a flag that is True if
        the mass-balance uncertainty was assumed because the dataset did not contain one.
    """
    if "time" in ds.dims:
        ds = compute_mean_mass_balance_over_time_window(ds)

    mass_gt = ds[MASS_BALANCE_COL_NAME].to_numpy() * scale_factor / 1e12
    if MASS_BALANCE_ERROR_COL_NAME in ds:
        masserr_gt = ds[MASS_BALANCE_ERROR_COL_NAME].to_numpy() * scale_factor / 1e12
        uncertainty_assumed = False
    else:
        masserr_gt = ASSUMED_MASS_BALANCE_UNCERTAINTY_FRACTION * np.abs(mass_gt)
        uncertainty_assumed = True

    lat_p, lon_p, m_p, me_p = bin_loads_to_points(ds["y"].to_numpy(), ds["x"].to_numpy(), mass_gt, masserr_gt)
    return lat_p, lon_p, m_p, me_p, uncertainty_assumed


def _uncertainty_attrs(uncertainty_assumed: bool) -> dict:
    """
    Build dataset attributes describing whether the mass-balance uncertainty was assumed.

    Parameters
    ----------
    uncertainty_assumed : bool
        True if the input data contained no mass-balance uncertainty and a fixed fraction of the
        mass change (``ASSUMED_MASS_BALANCE_UNCERTAINTY_FRACTION``) was used instead.

    Returns
    -------
    dict
        Always contains ``mass_balance_uncertainty_assumed`` (int): 1 if the uncertainty was assumed, else 0.
        If the uncertainty was assumed, it additionally contains:

        - ``mass_balance_uncertainty_assumed_fraction`` (float): the assumed fraction of the mass change.
        - ``mass_balance_uncertainty_note`` (str): a human-readable explanation of the assumption.
    """
    attrs: dict = {"mass_balance_uncertainty_assumed": int(uncertainty_assumed)}
    if uncertainty_assumed:
        fraction = ASSUMED_MASS_BALANCE_UNCERTAINTY_FRACTION
        attrs["mass_balance_uncertainty_assumed_fraction"] = fraction
        attrs["mass_balance_uncertainty_note"] = (
            "The input data contains no mass-balance uncertainty. An uncertainty of "
            f"+/-{fraction:.0%} of the mass change of each grid cell was assumed."
        )
    return attrs


def compute_annual_slr(ds: xr.Dataset, scale_factor: float = 1.0, output_units: str = "mm") -> xr.Dataset:
    """
    Compute annual SELREM SLR results and return them as an xarray.Dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with dims ("time", "point"), variable MASS_BALANCE_COL_NAME (kg), optionally
        MASS_BALANCE_ERROR_COL_NAME (kg), and coordinates "y" (latitude, deg) and "x" (longitude, deg) along "point".
        If the error variable is missing, an uncertainty of 15 % of the mass change is assumed (see attrs below).
    scale_factor : float
        Scaling factor applied to mass balance and its error (default: 1.0).
    output_units : str
        "mm" (default) or "m".

    Returns
    -------
    xr.Dataset
        Dataset with dims ("time", "x", "y") containing ndot, udot, sdot, sig_ndot, sig_udot, sig_sdot, rot_total.
        ``attrs`` contains ``units`` and ``mass_balance_uncertainty_assumed`` (1 if the uncertainty was assumed, else 0;
        if 1, also ``mass_balance_uncertainty_assumed_fraction`` and ``mass_balance_uncertainty_note``).
    """
    unit_factor = UNIT_FACTORS[output_units]
    gdot_fit, udot_fit, gdot_unc, udot_unc = load_selrem_parameters()
    ocean_mask, _, _ = load_ocean_mask()

    years = np.asarray(pd.to_datetime(ds["time"].to_numpy()).year, dtype=int)

    steps: list[dict[str, np.ndarray]] = []
    lons_t = lats_t = None
    uncertainty_assumed = False
    for i in range(len(years)):
        lat_p, lon_p, m_p, me_p, uncertainty_assumed = _prepare_vmb_points(ds.isel(time=i), scale_factor)
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

    data_vars = {key: (("time", "x", "y"), np.stack([s[key] for s in steps], axis=0) * unit_factor) for key in steps[0]}
    ds_out = xr.Dataset(
        data_vars=data_vars,
        coords={"time": np.array([np.datetime64(f"{y}-01-01") for y in years]), "x": lons_t, "y": lats_t},
    )
    ds_out.attrs["units"] = output_units
    ds_out.attrs.update(_uncertainty_attrs(uncertainty_assumed))
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
        If the error variable is missing, an uncertainty of 15 % of the mass change is assumed (see attrs below).
    scale_factor : float
        Scaling factor for mass balance and its error (default: 1.0).
    output_units : str
        "mm" (default) or "m".

    Returns
    -------
    xr.Dataset
        Dataset with dims ("x", "y") containing sdot, ndot, udot, rot_total, sig_sdot, sig_ndot, sig_udot.
        The ocean-mean values are stored in ``attrs`` as gmsl_sdot, gmsl_ndot, gmsl_udot, together with ``units`` and
        ``mass_balance_uncertainty_assumed`` (1 if the uncertainty was assumed, else 0; if 1, also
        ``mass_balance_uncertainty_assumed_fraction`` and ``mass_balance_uncertainty_note``).
    """
    unit_factor = UNIT_FACTORS[output_units]
    gdot_fit, udot_fit, gdot_unc, udot_unc = load_selrem_parameters()
    ocean_mask, _, _ = load_ocean_mask()

    lat_p, lon_p, m_p, me_p, uncertainty_assumed = _prepare_vmb_points(ds, scale_factor)
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
    ds_out.attrs.update(_uncertainty_attrs(uncertainty_assumed))
    return ds_out
