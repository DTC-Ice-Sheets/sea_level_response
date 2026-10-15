"""
Functions accelerated with Numba for SELREM computation.

These have been separated because Numba JIT-compiled functions can have issues with type hints. Keeping them here allows
us to keep type hints in other parts of the  SELREM logic.
"""

import numba
import numpy as np
from numba import njit, prange

from sea_level_response.config import EARTH_RADIUS_M, SELREM_ROT_COEFF


# Keep this Numba kernel unannotated: pytest-typeguard instruments annotations such as np.ndarray with runtime
# TypeCheckMemo checks, which Numba cannot type in nopython mode after @njit compilation.
@njit(parallel=True, fastmath=True)
def accumulate_fields(
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
):
    """
    Accumulate sea-level and deformation fields from mass change points using SELREM parameterizations.

    Parameters
    ----------
    lat_arr : np.ndarray
        Array of latitudes (in radians) for mass change points.
    lon_arr : np.ndarray
        Array of longitudes (in radians) for mass change points.
    mass_arr : np.ndarray
        Array of mass changes (in Gt) for each point.
    mass_err_arr : np.ndarray
        Array of mass change uncertainties (in Gt) for each point.
    sin_lat : np.ndarray
        2D array of sine of latitude for the target grid.
    cos_lat : np.ndarray
        2D array of cosine of latitude for the target grid.
    lon_rad : np.ndarray
        2D array of longitude (in radians) for the target grid.
    y21_real : np.ndarray
        2D array of real part of spherical harmonic Y21 for the target grid.
    ndot_fit : np.ndarray
        SELREM fit parameters for ndot.
    udot_fit : np.ndarray
        SELREM fit parameters for udot.
    ndot_unc : np.ndarray
        SELREM uncertainty parameters for ndot.
    udot_unc : np.ndarray
        SELREM uncertainty parameters for udot.

    Returns
    -------
    tuple of np.ndarray
        Tuple containing ndot_total, udot_total, rot_total, ndot_radial_unc_total,
        udot_radial_unc_total, ndot_vmb_unc_total, udot_vmb_unc_total arrays.
    """
    nlon, nlat = sin_lat.shape

    # Number of threads
    n_threads = numba.config.NUMBA_DEFAULT_NUM_THREADS

    # Allocate thread-local arrays
    ndot_loc = np.zeros((n_threads, nlon, nlat))
    udot_loc = np.zeros((n_threads, nlon, nlat))
    rot_loc = np.zeros((n_threads, nlon, nlat))
    ndot_radial_unc_loc = np.zeros((n_threads, nlon, nlat))
    udot_radial_unc_loc = np.zeros((n_threads, nlon, nlat))
    ndot_vmb_unc_loc = np.zeros((n_threads, nlon, nlat))
    udot_vmb_unc_loc = np.zeros((n_threads, nlon, nlat))

    for p in prange(lat_arr.shape[0]):
        thread_id = numba.get_thread_id()
        lat0 = lat_arr[p]
        lon0 = lon_arr[p]
        m = mass_arr[p]
        sigm = mass_err_arr[p]

        sin_lat0 = np.sin(lat0)
        cos_lat0 = np.cos(lat0)

        for i in range(nlon):
            for j in range(nlat):
                cos_d = sin_lat0 * sin_lat[i, j] + cos_lat0 * cos_lat[i, j] * np.cos(lon_rad[i, j] - lon0)
                cos_d = min(1.0, max(-1.0, cos_d))
                dist_km = (EARTH_RADIUS_M / 1000.0) * np.arccos(cos_d)

                r_nd = (
                    ndot_fit[0]
                    + ndot_fit[1] * dist_km
                    + ndot_fit[2] * dist_km**2
                    + ndot_fit[3] * np.exp(-dist_km / ndot_fit[4])
                )
                r_ud = (
                    udot_fit[0]
                    + udot_fit[1] * dist_km
                    + udot_fit[2] * dist_km**2
                    + udot_fit[3] * np.exp(-dist_km / udot_fit[4])
                    + udot_fit[5] * np.exp(-dist_km / udot_fit[6])
                )

                nd = r_nd * m
                ud = r_ud * m
                rot = SELREM_ROT_COEFF * m * y21_real[i, j]

                ndot_loc[thread_id, i, j] += nd + rot
                udot_loc[thread_id, i, j] += ud
                rot_loc[thread_id, i, j] += rot

                sigr_nd = (
                    ndot_unc[0] * np.exp(-dist_km / ndot_unc[1])
                    + ndot_unc[2]
                    + ndot_unc[3] * (1 - np.exp(-dist_km / ndot_unc[4]))
                )
                sigr_ud = (
                    udot_unc[0] * np.exp(-dist_km / udot_unc[1])
                    + udot_unc[2]
                    + udot_unc[3] * (1 - np.exp(-dist_km / udot_unc[4]))
                )

                ndot_radial_unc_loc[thread_id, i, j] += sigr_nd * abs(m)
                udot_radial_unc_loc[thread_id, i, j] += sigr_ud * abs(m)
                ndot_vmb_unc_loc[thread_id, i, j] += (r_nd * sigm) ** 2
                udot_vmb_unc_loc[thread_id, i, j] += (r_ud * sigm) ** 2

    # Sum over threads
    ndot_total = np.sum(ndot_loc, axis=0)
    udot_total = np.sum(udot_loc, axis=0)
    rot_total = np.sum(rot_loc, axis=0)
    ndot_radial_unc_total = np.sum(ndot_radial_unc_loc, axis=0)
    udot_radial_unc_total = np.sum(udot_radial_unc_loc, axis=0)
    ndot_vmb_unc_total = np.sum(ndot_vmb_unc_loc, axis=0)
    udot_vmb_unc_total = np.sum(udot_vmb_unc_loc, axis=0)

    return (
        ndot_total,
        udot_total,
        rot_total,
        ndot_radial_unc_total,
        udot_radial_unc_total,
        ndot_vmb_unc_total,
        udot_vmb_unc_total,
    )
