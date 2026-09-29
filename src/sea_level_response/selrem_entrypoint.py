"""Command line interface entry point for the SELREM module --> for global computation only."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Literal

import cyclopts

from sea_level_response.helpers import ZarrType
from sea_level_response.monitoring import setup_logging
from sea_level_response.selrem.selrem import compute_annual_slr, compute_global_slr
from sea_level_response.selrem.selrem_helpers import slice_and_prepare_dataset
from sea_level_response.selrem.selrem_input_prep import prepare_gridded_ds_for_selrem

setup_logging()

logger = logging.getLogger(__name__)

app = cyclopts.App()


@app.default
def main(
    mass_balance_geozarr: ZarrType,
    start_time: datetime,
    end_time: datetime,
    analysis_mode: Literal["global", "annual"],
    output_path: Path,
    scaling_factor: float = 1.0,
    coarsen_factor: int = 1,
) -> None:
    """Command line entry point for the SELREM module (global data).

    Parameters
    ----------
    mass_balance_geozarr : ZarrType
        Location of a GeoZarr dataset containing VMB data in kg. Can be a local path or a web URL.
    start_time : datetime
        Start time for the time window.
    end_time : datetime
        End time for the time window.
    analysis_mode : Literal["global", "annual"]
        Whether to compute global mean SLR or annual gridded SLR.
    output_path : Path
        Where to save the output sea level rise zarr.
    scaling_factor : float
        Scaling factor to apply to mass balance values. Default is 1.0.
    coarsen_factor : int
        Spatial coarsening factor applied before flattening (gridded input only). Mass is summed (mass-conserving);
        coordinates are averaged; uncertainties are aggregated using RSS/N. Default 1 (no coarsening).
    """
    start_time = start_time.replace(tzinfo=None)
    end_time = end_time.replace(tzinfo=None)
    ds = slice_and_prepare_dataset(
        vmb_dataset_path=mass_balance_geozarr,
        start_time=start_time,
        end_time=end_time,
    )
    if "point" not in ds.dims:
        ds = prepare_gridded_ds_for_selrem(ds, coarsen_factor=coarsen_factor)

    # Run the computation
    if analysis_mode == "global":
        result = compute_global_slr(ds, scale_factor=scaling_factor)
        logger.info("Global SLR computation complete. Output: %s", result.info())
    elif analysis_mode == "annual":
        result = compute_annual_slr(ds, scale_factor=scaling_factor)
        logger.info("Annual SLR computation complete. Output: %s", result.info())
    else:
        raise ValueError(f"Unknown analysis mode: {analysis_mode}")

    result.to_zarr(output_path, zarr_format=2)


if __name__ == "__main__":
    app()
