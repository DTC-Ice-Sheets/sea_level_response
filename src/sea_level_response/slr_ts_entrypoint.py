"""Command line interface entry point for the SELREM module.

For location-specific time-series computation only.
"""

import json
import logging
from pathlib import Path

import cyclopts
import xarray as xr

from sea_level_response.monitoring import setup_logging
from sea_level_response.selrem.selrem_helpers import preprocess_location_timeseries

setup_logging()

logger = logging.getLogger(__name__)

app = cyclopts.App()


@app.default
def main(
    annual_slr_gz_path: Path,
    output_path: Path,
    latitude: float | None = None,
    longitude: float | None = None,
) -> None:
    """Command line entry point for location specific sea-level time-series computation.

    Parameters
    ----------
    annual_slr_gz_path : Path
        Path to the local GeoZarr dataset containing global, annual sea-level rise data.
    output_path : Path
        Where to save the output time series data as JSON.
    latitude : float | None
        Target latitude for time series extraction.
    longitude : float | None
        Target longitude for time series extraction.
    output_path : Path
        Where to save the output time series data as JSON.
    """
    logger.info("Loading annual SLR dataset from: %s", annual_slr_gz_path)
    annual_slr_ds = xr.open_dataset(annual_slr_gz_path, engine="zarr")

    # Validate inputs
    if latitude is None or longitude is None:
        raise ValueError("Both latitude and longitude must be provided for location time series extraction")

    logger.info("Preprocessing location time series for coordinates: (%.2f, %.2f)", latitude, longitude)
    timeseries_data = preprocess_location_timeseries(
        annual_slr_ds=annual_slr_ds,
        lat0=latitude,
        lon0=longitude,
    )
    logger.info("Location time series preprocessing complete")

    # Save as JSON (better format for small time series data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Path(output_path).open("w") as f:
        json.dump(timeseries_data, f, indent=2)

    logger.info("Time series data saved to: %s", output_path)
    logger.info("Output contains %d time steps", len(timeseries_data.get("years", [])))


if __name__ == "__main__":
    app()
