"""Command line interface entry point for the SELREM module --> for global computation only."""

import logging
from pathlib import Path

import cyclopts
import xarray as xr

from sea_level_response.helpers import ZarrType
from sea_level_response.monitoring import setup_logging
from sea_level_response.selrem.selrem_input_prep import get_dataset_crs, prepare_gridded_ds_for_selrem

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
        If the coarsen factor is less than 1.
    """
    if coarsen_factor < 1:
        raise ValueError("Coarsen factor must be >= 1.")

    ds = xr.open_zarr(gridded_dataset_geozarr)
    # convert MB output to a dataset with the appropriate CRS metadata - needed for writing to GeoZarr
    crs = get_dataset_crs(ds)
    ds = ds.rio.write_crs(crs)
    # print(f"Dataset CRS: {crs}")

    if "point" in ds.dims:
        logger.warning("Dataset is already in point format!! Not modifying it.")
    else:
        ds = prepare_gridded_ds_for_selrem(ds, coarsen_factor=coarsen_factor)
        logger.info("Prepared dataset for SELREM and saving to output path %s.", output_path)

    # Save the dataset to the output path (whether or not any processing was done)
    ds.to_zarr(output_path, zarr_format=2)


if __name__ == "__main__":
    app()
