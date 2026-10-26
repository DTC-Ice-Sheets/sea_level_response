"""Configuration constants for sea_level_response."""

from pathlib import Path

#: Path to the bundled ocean mask dataset.
#:
#: :meta hide-value:
OCEAN_MASK_PATH = Path(__file__).resolve().parent / "selrem/data/ocean_mask.nc"

#: Path to the bundled SELREM fit parameters.
#:
#: :meta hide-value:
SELREM_PARAMS_PATH = Path(__file__).resolve().parent / "selrem/data/selrem_udot_gdot_fit.json"

#: Column name for land-ice surface specific mass balance flux.
MASS_BALANCE_COL_NAME = "land_ice_surface_specific_mass_balance_flux"

#: Column name for uncertainty in land-ice surface specific mass balance flux.
MASS_BALANCE_ERROR_COL_NAME = "land_ice_surface_specific_mass_balance_flux_uncertainty"

#: Earth radius in kilometers.
EARTH_RADIUS_KM = 6371.0

#: Metres of global-mean sea level per Gt (mass loss -> sea-level rise).
GMSL_PER_GT_M = -1.0 / 362.0 / 1000.0

#: Rotational correction coefficient used by SELREM.
ROT_COEF = 0.20

#: Self-attraction and loading coefficient used by SELREM.
BETA_SAL = 0.20

#: Longitude shift of the Y21 rotational pattern
Y21_LON_SHIFT_DEG = 20.0

#: If input mass-balance dataset doesn't have uncertainties, assume 15%
ASSUMED_MASS_BALANCE_UNCERTAINTY_FRACTION = 0.15

#: Dictionary for unit conversions
UNIT_FACTORS = {"m": 1.0, "mm": 1000.0}
