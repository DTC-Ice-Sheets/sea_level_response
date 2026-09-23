"""
Prepare mass balance data for the SELREM entrypoint.

SELREM expects a (time, point) dataset with lat/lon coordinates and annual timesteps. This module converts from a
gridded (time, y, x) dataset with CF metadata and a CRS (e.g., as written by rioxarray) to the SELREM schema, handling
coordinate reprojection and annual resampling as needed. If the input is already in (time, point) format, it will be
passed through after ensuring annual resolution and appropriate coordinate variables are present.
"""

from __future__ import annotations

import logging

import numpy as np
import xarray as xr
from pyproj import CRS, Transformer

from sea_level_response.config import MASS_BALANCE_COL_NAME, MASS_BALANCE_ERROR_COL_NAME
from sea_level_response.helpers import get_dataset_crs

logger = logging.getLogger(__name__)


def _assign_latlon_coords(da: xr.DataArray, crs: CRS) -> xr.DataArray:
    """
    Attach 2D lat/lon coordinate arrays to a (y, x) DataArray.

    If the data is already lat/lon, we broadcast the existing coordinates to 2D and assign them as lat/lon.
    Otherwise, we reproject to EPSG:4326 from the input CRS.

    Parameters
    ----------
    da : xr.DataArray
        DataArray with (y, x) dimensions and a known CRS.
    crs : CRS
        Coordinate reference system of the input DataArray.

    Returns
    -------
    xr.DataArray
        DataArray with attached 2D lat/lon coordinates.
    """
    if crs.is_geographic:
        lat, lon = xr.broadcast(da.y, da.x)
        return da.assign_coords(lat=lat, lon=lon)

    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    x_2d, y_2d = np.meshgrid(da.x.values, da.y.values)
    lon_2d, lat_2d = transformer.transform(x_2d, y_2d)
    return da.assign_coords(
        lat=(("y", "x"), lat_2d),
        lon=(("y", "x"), lon_2d),
    )


def to_annual(da: xr.DataArray, is_uncertainty: bool = False) -> xr.DataArray:
    """
    Resample to annual resolution if sub-annual timesteps are present.

    Parameters
    ----------
    da : xr.DataArray
        DataArray with a time dimension.
    is_uncertainty : bool
        If True, resample using RSS/N, use mean otherwise. Default False.

    Returns
    -------
    xr.DataArray
        Annually resampled DataArray; same as input if already at annual resolution.
    """
    time_years = da.time.dt.year.values
    if len(time_years) > len(np.unique(time_years)):
        if is_uncertainty:
            resampled = (da**2).resample(time="1YE")
            return np.sqrt(resampled.sum()) / resampled.count()
        return da.resample(time="1YE").mean()
    return da


def prepare_gridded_ds_for_selrem(
    ds: xr.Dataset,
    coarsen_factor: int = 1,
) -> xr.Dataset:
    """
    Convert gridded data to the (time, point) schema SELREM expects.

    - Resamples to annual resolution if sub-annual timesteps are detected.
    - Converts to geographic coordinates (if the input CRS is not already geographic), stacks and drops all-NaN points.
    - Coarsens spatially by the given factor if coarsen_factor > 1, using mass-conserving sum for the mass balance
    variable and averaging for coordinates.

    Parameters
    ----------
    ds : xr.Dataset
        Gridded dataset with x, y, time coords and mass balance and (optionally) mass balance error variables.
    coarsen_factor : int
        Spatial coarsening factor applied before flattening. Mass is summed (mass-conserving); coordinates are averaged.
        Default 1 (no coarsening).

    Returns
    -------
    xr.Dataset
        Dataset with (time, point) dimensions and y = latitude, x = longitude coordinate variables.
    """
    crs = get_dataset_crs(ds)
    if crs is None:
        raise ValueError(
            "No CRS found in dataset. Ensure it has been written with rioxarray "
            "or contains a CF-conventions 'spatial_ref' coordinate with 'crs_wkt' metadata."
        )
    if any(dim not in ds.dims for dim in ("x", "y", "time")):
        raise ValueError("Input dataset must have 'x', 'y', and 'time' dimensions.")
    if MASS_BALANCE_COL_NAME not in ds.data_vars:
        raise ValueError(f"Input dataset must contain a mass balance variable named '{MASS_BALANCE_COL_NAME}'.")

    # ensure annual resolution, as expected by SELREM; if already annual, this is a no-op
    mb_da = to_annual(ds[MASS_BALANCE_COL_NAME])

    if coarsen_factor > 1:
        _valid = mb_da.notnull().coarsen(y=coarsen_factor, x=coarsen_factor, boundary="trim").sum()
        mb_da = mb_da.coarsen(y=coarsen_factor, x=coarsen_factor, boundary="trim").sum().where(_valid > 0)
    elif coarsen_factor < 1:
        logger.warning("coarsen_factor < 1 is not supported; ignoring and using coarsen_factor=1.")
        coarsen_factor = 1

    mb_da = _assign_latlon_coords(mb_da, crs)
    mb_da_flat = mb_da.stack(point=("y", "x")).dropna("point", how="all").compute()

    if MASS_BALANCE_ERROR_COL_NAME in ds.data_vars:
        err_da = to_annual(ds[MASS_BALANCE_ERROR_COL_NAME], is_uncertainty=True)
        if coarsen_factor > 1:
            # use root sum of squares (RSS) for aggregation of uncertainties - is this appropriate for mass balance err?
            _err_sq = err_da**2
            _err_valid = _err_sq.notnull().coarsen(y=coarsen_factor, x=coarsen_factor, boundary="trim").sum()
            err_da = np.sqrt(
                _err_sq.coarsen(y=coarsen_factor, x=coarsen_factor, boundary="trim").sum().where(_err_valid > 0)
            )
        err_da = _assign_latlon_coords(err_da, crs)
        err_da_flat = err_da.stack(point=("y", "x")).sel(point=mb_da_flat.point).compute()
        err_values = err_da_flat.values
    else:
        err_values = np.full_like(mb_da_flat.values, np.nan)

    ds = xr.Dataset(
        {
            MASS_BALANCE_COL_NAME: (("time", "point"), mb_da_flat.values),
            MASS_BALANCE_ERROR_COL_NAME: (("time", "point"), err_values),
            "y": ("point", mb_da_flat.lat.values),
            "x": ("point", mb_da_flat.lon.values),
        },
        coords={
            "time": mb_da_flat.time.values,
            "point": np.arange(mb_da_flat.sizes["point"]),
        },
    )
    ds = ds.rio.write_crs("EPSG:4326")
    return ds
