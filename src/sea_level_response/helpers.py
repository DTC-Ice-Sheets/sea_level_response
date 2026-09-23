"""Utility functions for use across sea_level_response modules."""

import logging
from pathlib import Path
from typing import Annotated

import numcodecs
import pyproj
import xarray as xr
from cyclopts import Parameter
from data_converter.geo_zarr import GeoZarr, ZarrStorage

logger = logging.getLogger(__name__)


def _determine_spatial_coord_names(dataset: xr.Dataset) -> tuple[str, str] | None:
    """
    Determine the names of the spatial coordinates in the dataset.

    This function has been taken from the Data Download Module. It checks for the presence of common spatial coordinate
    names in the dataset and returns them as a tuple. If no valid spatial coordinates are found, it returns None.

    Parameters
    ----------
    dataset : xr.Dataset
        The dataset from which to determine the spatial coordinate names.

    Returns
    -------
    tuple[str, str] | None
        A tuple containing the names of the x and y coordinates if found, otherwise None.
    """
    coord_names = list(dataset.coords)
    valid_coord_name_pairs = [
        ("x", "y"),
        ("projection_x_coordinate", "projection_y_coordinate"),
        ("longitude", "latitude"),
    ]
    for x, y in valid_coord_name_pairs:
        if x in coord_names and y in coord_names:
            return x, y
    return None


def get_dataset_crs(dataset: xr.Dataset) -> pyproj.CRS | None:
    """
    Retrieve the CRS details from the dataset.

    This function has been taken from the Data Download Module. It first checks for CRS written with rioxarray. Next,
    it checks the dataset for a grid mapping attribute If the CRS is not found in the grid mapping, it checks the
    dataset attributes for a `_CRS` attribute containing the WKT. If the CRS is still not found, it checks one of the
    data variable's attributes for a `_CRS` attribute with a `wkt` key (to match the current format of S1 ice velocity
    data on EODS).

    Parameters
    ----------
    dataset : xr.Dataset
        The dataset from which to retrieve the CRS.

    Returns
    -------
    pyproj.CRS | None
        The CRS of the dataset.
    """
    import rioxarray  # noqa: F401

    # check for CRS written with rioxarray first
    try:
        rio_crs = dataset.rio.crs
        if rio_crs is not None:
            return pyproj.CRS(rio_crs)
    except AttributeError:
        pass

    crs_wkt = None
    spatial_coord_names = _determine_spatial_coord_names(dataset)

    # check for grid mapping attribute in the spatial coordinates
    if (
        spatial_coord_names is not None
        and (grid_mapping_name := dataset[spatial_coord_names[0]].attrs.get("grid_mapping")) is not None
    ):
        crs_wkt = dataset[grid_mapping_name].attrs.get("crs_wkt", None)

    # check for CRS in the dataset attributes (e.g., as written by rioxarray)
    if crs_wkt is None and "_CRS" in dataset.attrs and "crs_wkt" in dataset.attrs["_CRS"]:
        crs_wkt = dataset.attrs["_CRS"]["crs_wkt"]

    # for the last try, handle current format of S1 ice velocity data on EODS
    example_data_var = next(iter(dataset.data_vars), None)
    if crs_wkt is None and example_data_var is None:
        return None
    if crs_wkt is None and (crs := dataset[example_data_var].attrs.get("_CRS")) is not None:
        crs_wkt = crs.get("wkt", None)
    return pyproj.CRS(crs_wkt) if crs_wkt else None


def write_to_geozarr_file(data: xr.Dataset, output_path: Path) -> None:
    """
    Write a GeoZarr-compliant dataset to a file.

    Parameters
    ----------
    data : xr.Dataset
        Geo-Zarr-compliant dataset to be written to file.
    output_path : Path
        Path where the GeoZarr file will be saved.
    """
    logger.info("Writing GeoZarr to directory at %s...", output_path)
    dataset_crs = get_dataset_crs(data)
    if dataset_crs is None:
        # TODO: Remove once CRS is not mandatory in GeoZarr converter class
        logger.warning("Dataset does not contain CRS information, saving to file without GeoZarr converter.")
        if "time" in data:
            for k in ["units", "calendar"]:
                if k in data["time"].attrs:
                    data["time"].attrs.pop(k)
        encoding = {var: {"compressor": numcodecs.Blosc(cname="zstd", clevel=3, shuffle=2)} for var in data.data_vars}
        data.to_zarr(output_path, mode="w", encoding=encoding)
    else:
        geozarr_writer = GeoZarr(storage_type=ZarrStorage.local_store, storage_directory=output_path, crs=dataset_crs)
        geozarr_writer.convert_xarray_dataset_to_geo_zarr(dataset=data)

    logger.info("GeoZarr written successfully.")


def path_or_web_url_validator(type_: object, path_or_url: Path | str) -> str:
    """Validate that the provided path is either a valid local path or a web URL.

    NOTE: Path("") resolves to the current directory (".") - which is considered a valid path.

    Parameters
    ----------
    type_ : object
        The type of the argument being validated - unused, purely required for cyclopts validator compatibility.
    path_or_url : Path | str
        The candidate path or URL to validate.

    Returns
    -------
    str
        The validated path or URL as a string.
    """
    if not Path(path_or_url).exists() and not str(path_or_url).startswith(("http://", "https://", "s3://")):
        raise ValueError(f"The provided path '{path_or_url}' is neither a valid local path nor a web URL.")
    return str(path_or_url)


# alias for use in cyclopts Parameter definitions
type ZarrType = Annotated[str, Parameter(validator=path_or_web_url_validator)]
