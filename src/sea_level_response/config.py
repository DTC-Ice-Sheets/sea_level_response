"""Configuration constants for sea_level_response."""

from pathlib import Path

EARTH_RADIUS_M = 6_371_000  # in meters
SELREM_ROT_COEFF = -0.00030

OCEAN_MASK_PATH = Path(__file__).resolve().parent / "selrem/data/ocean_mask.nc"
SELREM_PARAMS_PATH = Path(__file__).resolve().parent / "selrem/data/selrem_udot_gdot_fit.json"

MASS_BALANCE_COL_NAME = "land_ice_surface_specific_mass_balance_flux"
MASS_BALANCE_ERROR_COL_NAME = "land_ice_surface_specific_mass_balance_flux_uncertainty"
