"""
Functions accelerated with Numba for SELREM computation.

These have been separated because Numba JIT-compiled functions can have issues with type hints. Keeping them here allows
us to keep type hints in other parts of the SELREM logic.

The kernel below is a direct port of ``_emulator_kernel_gridwise`` from the reference notebook (selrem_run.ipynb).
"""

import numpy as np
from numba import njit, prange


# Keep this Numba kernel unannotated: pytest-typeguard instruments annotations such as np.ndarray with runtime
# TypeCheckMemo checks, which Numba cannot type in nopython mode after @njit compilation.
@njit(parallel=False, fastmath=True)
def accumulate_fields(
    lat_p_rad,
    lon_p_rad,
    mass_gt,
    masserr_gt,
    sin_lat_g,
    cos_lat_g,
    lon_g_rad,
    y21_real_g,
    gdot_fit,
    udot_fit,
    gdot_unc,
    udot_unc,
    r_earth_km,
    gmsl_per_gt_m,
    rot_coef,
):
    """
    Accumulate the response of every grid cell to every point load.

    Parameters
    ----------
    lat_p_rad, lon_p_rad : np.ndarray
        1D arrays, load locations in radians.
    mass_gt, masserr_gt : np.ndarray
        1D arrays, loads and their uncertainties in Gt.
    sin_lat_g, cos_lat_g, lon_g_rad, y21_real_g : np.ndarray
        2D (nlon, nlat) arrays describing the computation grid.
    gdot_fit, udot_fit, gdot_unc, udot_unc : np.ndarray
        SELREM fit / uncertainty parameters (from ``selrem_udot_gdot_fit.json``).
    r_earth_km : float
        Earth radius in km.
    gmsl_per_gt_m : float
        Global-mean sea level (m) per Gt of mass (negative: mass loss raises sea level).
    rot_coef : float
        Rotational feedback coefficient.

    Returns
    -------
    tuple of np.ndarray
        gdot, udot, rot, gdot_radial_unc, udot_radial_unc, gdot_vmb_var, udot_vmb_var, all in metres
        (the last two are *variances*, i.e. m^2). ``gdot`` already includes the rotational term.
    """
    nlon, nlat = sin_lat_g.shape
    npts = mass_gt.size

    gdot = np.zeros((nlon, nlat))
    udot = np.zeros((nlon, nlat))
    rot = np.zeros((nlon, nlat))
    gdot_radial_unc = np.zeros((nlon, nlat))
    udot_radial_unc = np.zeros((nlon, nlat))
    gdot_vmb_var = np.zeros((nlon, nlat))
    udot_vmb_var = np.zeros((nlon, nlat))

    # Pre-compute trigonometry of the loads once
    sin_lat_p = np.sin(lat_p_rad)
    cos_lat_p = np.cos(lat_p_rad)

    # Every (i, j) cell is independent, so parallelising over i is safe.
    for i in prange(nlon):
        for j in range(nlat):
            sin_lat = sin_lat_g[i, j]
            cos_lat = cos_lat_g[i, j]
            lonr = lon_g_rad[i, j]

            gd_sum = 0.0
            ud_sum = 0.0
            rot_sum = 0.0
            gd_unc_sum = 0.0
            ud_unc_sum = 0.0
            gd_vmb_var_sum = 0.0
            ud_vmb_var_sum = 0.0

            for k in range(npts):
                # Great-circle distance between cell and load
                c = sin_lat_p[k] * sin_lat + cos_lat_p[k] * cos_lat * np.cos(lon_p_rad[k] - lonr)
                if c > 1.0:
                    c = 1.0
                elif c < -1.0:
                    c = -1.0
                dist_km = r_earth_km * np.arccos(c)

                # Radial Green's functions
                r_gd = (
                    gdot_fit[0]
                    + gdot_fit[1] * dist_km
                    + gdot_fit[2] * dist_km**2
                    + gdot_fit[3] * np.exp(-dist_km / gdot_fit[4])
                )
                r_ud = (
                    udot_fit[0]
                    + udot_fit[1] * dist_km
                    + udot_fit[2] * dist_km**2
                    + udot_fit[3] * np.exp(-dist_km / udot_fit[4])
                    + udot_fit[5] * np.exp(-dist_km / udot_fit[6])
                )

                # Mass (Gt) -> global-mean sea level (m)
                sle_m = gmsl_per_gt_m * mass_gt[k]
                gd = r_gd * sle_m
                ud = r_ud * sle_m
                ro = rot_coef * sle_m * y21_real_g[i, j]

                gd_sum += gd + ro
                ud_sum += ud
                rot_sum += ro

                # Radial (fit) uncertainty
                sigr_gd = (
                    gdot_unc[0] * np.exp(-dist_km / gdot_unc[1])
                    + gdot_unc[2]
                    + gdot_unc[3] * (1.0 - np.exp(-dist_km / gdot_unc[4]))
                )
                sigr_ud = (
                    udot_unc[0] * np.exp(-dist_km / udot_unc[1])
                    + udot_unc[2]
                    + udot_unc[3] * (1.0 - np.exp(-dist_km / udot_unc[4]))
                )
                gd_unc_sum += sigr_gd * abs(sle_m)
                ud_unc_sum += sigr_ud * abs(sle_m)

                # Mass-balance (VMB) uncertainty, as variance
                sle_err = gmsl_per_gt_m * masserr_gt[k]
                gd_vmb_var_sum += (r_gd * sle_err) ** 2
                ud_vmb_var_sum += (r_ud * sle_err) ** 2

            gdot[i, j] = gd_sum
            udot[i, j] = ud_sum
            rot[i, j] = rot_sum
            gdot_radial_unc[i, j] = gd_unc_sum
            udot_radial_unc[i, j] = ud_unc_sum
            gdot_vmb_var[i, j] = gd_vmb_var_sum
            udot_vmb_var[i, j] = ud_vmb_var_sum

    return gdot, udot, rot, gdot_radial_unc, udot_radial_unc, gdot_vmb_var, udot_vmb_var
