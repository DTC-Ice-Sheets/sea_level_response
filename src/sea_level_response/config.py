"""Configuration constants for sea_level_response."""

from pathlib import Path

#: Earth radius in metres.
EARTH_RADIUS_M = 6_371_000

#: Rotational correction coefficient used by SELREM.
SELREM_ROT_COEFF = -0.00030

#: Path to the bundled ocean mask dataset.
#:
#: :meta hide-value:
OCEAN_MASK_PATH = Path(__file__).resolve().parent / "selrem/data/ocean_mask.nc"

#: Path to the bundled SELREM fit parameters.
#:
#: :meta hide-value:
SELREM_PARAMS_PATH = Path(__file__).resolve().parent / "selrem/data/selrem_udot_ndot_fit.json"

#: Column name for land-ice surface specific mass balance flux.
MASS_BALANCE_COL_NAME = "land_ice_surface_specific_mass_balance_flux"

#: Column name for uncertainty in land-ice surface specific mass balance flux.
MASS_BALANCE_ERROR_COL_NAME = "land_ice_surface_specific_mass_balance_flux_uncertainty"
