"""Command line interface entry point for the SELREM module --> for global computation only."""

import logging
from pathlib import Path

import cyclopts
import xarray as xr

from sea_level_response.helpers import ZarrType
from sea_level_response.monitoring import setup_logging
from sea_level_response.selrem.selrem_input_prep import prepare_gridded_ds_for_selrem

setup_logging()

logger = logging.getLogger(__name__)

app = cyclopts.App()


@app.default
def main(
    gridded_dataset_geozarr: ZarrType,
    output_path: Path,
    coarsen_factor: int = 1,
) -> None:
    """Command line entry point for the prepare gridded data for SELREM.

    This allows external workflows to prepare gridded datasets for SELREM processing. This is included to prevent the
    need for other modules to import the sea_level_response module directly.

    Parameters
    ----------
    gridded_dataset_geozarr : ZarrType
        Location of a GeoZarr dataset containing VMB data in kg. Can be a local path or a web URL.
    output_path : Path
        Where to save the generated point dataset zarr.
    coarsen_factor : int
        Spatial coarsening factor applied before flattening (gridded input only). Mass is summed (mass-conserving);
        coordinates are averaged; uncertainties are aggregated using RSS/N. Default 1 (no coarsening).

    Raises
    ------
    ValueError
        If the dataset is already in point format.
    """
    ds = xr.open_zarr(gridded_dataset_geozarr)
    if "point" not in ds.dims:
        ds = prepare_gridded_ds_for_selrem(ds, coarsen_factor=coarsen_factor)
        logger.info("Prepared dataset for SELREM and saving to output path %s.", output_path)
    else:
        raise ValueError("Dataset is already in point format! No actions to perform.")

    ds.to_zarr(output_path, zarr_format=2)


if __name__ == "__main__":
    app()
