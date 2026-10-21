"""Configuration constants for sea_level_response."""

from pathlib import Path

OCEAN_MASK_PATH = Path(__file__).resolve().parent / "selrem/data/ocean_mask.nc"
SELREM_PARAMS_PATH = Path(__file__).resolve().parent / "selrem/data/selrem_udot_gdot_fit.json"

MASS_BALANCE_COL_NAME = "land_ice_surface_specific_mass_balance_flux"
MASS_BALANCE_ERROR_COL_NAME = "land_ice_surface_specific_mass_balance_flux_uncertainty"

R_EARTH_KM = 6371.0  # in km
GMSL_PER_GT_M = -1.0 / 362.0 / 1000.0  # metres of global-mean sea level per Gt (mass loss -> sea-level rise)
ROT_COEF = 0.20  # rotational feedback coefficient
BETA_SAL = 0.20  # self-attraction and loading coefficient
Y21_LON_SHIFT_DEG = 20.0  # longitude shift of the Y21 rotational pattern

TARGET_RES_DEG = 0.5  # output resolution
COMPUTE_RES_DEG = 2.0  # internal compute grid resolution
BIN_RES_DEG = 1.0  # resolution at which VMB loads are binned to point loads

ASSUMED_MASS_BALANCE_UNCERTAINTY_FRACTION = 0.15  # if input mass-balance dataset doesn't have uncertainties, assume 15%

_UNIT_FACTORS = {"m": 1.0, "mm": 1000.0}
