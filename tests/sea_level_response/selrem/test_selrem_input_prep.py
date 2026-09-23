"""Unit tests for selrem_input_prep module."""

import logging

import numpy as np
import pandas as pd
import pytest
import rioxarray  # noqa: F401
import xarray as xr
from pyproj import CRS

from sea_level_response.config import MASS_BALANCE_COL_NAME, MASS_BALANCE_ERROR_COL_NAME
from sea_level_response.selrem.selrem_input_prep import (
    _assign_latlon_coords,
    prepare_gridded_ds_for_selrem,
    to_annual,
)


@pytest.fixture
def example_mass_balance_ds() -> xr.Dataset:
    """Minimal (time=3, y=4, x=4) dataset with geographic CRS (EPSG:4326) and random mass balance data."""
    times = pd.date_range("2010-01-01", periods=3, freq="YE")
    x_vals = np.linspace(-50.0, -47.0, 4)
    y_vals = np.linspace(68.0, 70.0, 4)
    rng = np.random.default_rng(0)
    data = rng.standard_normal((len(times), 4, 4)).astype(np.float32)
    ds = xr.Dataset(
        {MASS_BALANCE_COL_NAME: (("time", "y", "x"), data)},
        coords={"time": times, "y": y_vals, "x": x_vals},
    )
    ds = ds.rio.write_crs(CRS.from_epsg(4326))
    return ds


def test_assign_latlon_geographic_crs():
    """For a geographic CRS, lat/lon should equal y/x (broadcast to 2D)."""
    crs = CRS.from_epsg(4326)
    x_vals = np.array([-50.0, -49.0])
    y_vals = np.array([68.0, 69.0])
    da = xr.DataArray(
        np.ones((2, 2)),
        dims=("y", "x"),
        coords={"y": y_vals, "x": x_vals},
    )
    result = _assign_latlon_coords(da, crs)
    np.testing.assert_equal(result.coords["lat"].values, np.array([[68.0, 68.0], [69.0, 69.0]]))
    np.testing.assert_equal(result.coords["lon"].values, np.array([[-50.0, -49.0], [-50.0, -49.0]]))


def test_assign_latlon_projected_crs_shape():
    """For a projected CRS the attached lat/lon arrays should be 2D (y, x)."""
    crs = CRS.from_epsg(3413)  # Polar Stereographic North
    x_vals = np.array([-200_000.0, 0.0, 200_000.0])
    y_vals = np.array([-2_000_000.0, -1_800_000.0])
    da = xr.DataArray(
        np.ones((2, 3)),
        dims=("y", "x"),
        coords={"y": y_vals, "x": x_vals},
    )
    result = _assign_latlon_coords(da, crs)
    assert result.coords["lat"].dims == ("y", "x")
    assert result.coords["lon"].dims == ("y", "x")
    assert result.coords["lat"].shape == (2, 3)


def test_to_annual_already_annual():
    """Annual data should pass through unchanged."""
    times = pd.date_range("2010-01-01", periods=5, freq="YE")
    da = xr.DataArray(np.arange(5.0), dims=("time",), coords={"time": times})
    result = to_annual(da)
    assert result.sizes["time"] == 5
    np.testing.assert_array_equal(result.values, da.values)


def test_to_annual_monthly_data():
    """Monthly data should be resampled to annual means."""
    times = pd.date_range("2010-01-01", periods=24, freq="ME")
    values = np.ones(24)
    da = xr.DataArray(values, dims=("time",), coords={"time": times})
    result = to_annual(da)
    assert result.sizes["time"] == 2
    np.testing.assert_array_equal(result.values, [1.0, 1.0])


def test_to_annual_monthly_varying_values():
    """Annual mean should average over all months in the year."""
    times = pd.date_range("2010-01-01", periods=12, freq="ME")
    values = np.arange(12.0)
    da = xr.DataArray(values, dims=("time",), coords={"time": times})
    result = to_annual(da)
    assert result.sizes["time"] == 1
    np.testing.assert_array_equal(result.values[0], np.mean(values))


def test_to_annual_uncertainty_uses_rss_over_n():
    """With is_uncertainty=True, monthly values should be combined as RSS/N, not mean."""
    times = pd.date_range("2010-01-01", periods=12, freq="ME")
    values = np.arange(1.0, 13.0)
    da = xr.DataArray(values, dims=("time",), coords={"time": times})
    result = to_annual(da, is_uncertainty=True)
    assert result.sizes["time"] == 1
    expected = np.sqrt((values**2).sum()) / len(values)
    np.testing.assert_allclose(result.values[0], expected, rtol=1e-6)


def test_prepare_gridded_ds_for_selrem_raises_no_crs(example_mass_balance_ds: xr.Dataset):
    ds = example_mass_balance_ds.drop_vars("spatial_ref")
    with pytest.raises(ValueError, match="No CRS"):
        prepare_gridded_ds_for_selrem(ds)


def test_prepare_gridded_ds_for_selrem_raises_missing_dim(example_mass_balance_ds: xr.Dataset):
    ds = example_mass_balance_ds.isel(time=0).drop_vars("time")
    with pytest.raises(ValueError, match="'x', 'y', and 'time'"):
        prepare_gridded_ds_for_selrem(ds)


def test_prepare_gridded_ds_for_selrem_raises_missing_mb_var(example_mass_balance_ds: xr.Dataset):
    ds = example_mass_balance_ds.rename({MASS_BALANCE_COL_NAME: "some_other_var"})
    with pytest.raises(ValueError, match=MASS_BALANCE_COL_NAME):
        prepare_gridded_ds_for_selrem(ds)


def test_prepare_gridded_ds_for_selrem_output_schema_no_error_col(example_mass_balance_ds: xr.Dataset):
    """Without an error column the output should have NaN-filled error values."""
    out = prepare_gridded_ds_for_selrem(example_mass_balance_ds)

    assert set(out.dims) == {"time", "point"}
    assert MASS_BALANCE_COL_NAME in out.data_vars
    assert MASS_BALANCE_ERROR_COL_NAME in out.data_vars
    assert "y" in out.variables
    assert "x" in out.variables
    assert out.sizes["time"] == example_mass_balance_ds.sizes["time"]
    assert np.all(np.isnan(out[MASS_BALANCE_ERROR_COL_NAME].values))


def test_prepare_gridded_ds_for_selrem_output_schema_with_error_col(example_mass_balance_ds: xr.Dataset):
    """With an error column present it should be processed and populated."""
    err_data = np.ones_like(example_mass_balance_ds[MASS_BALANCE_COL_NAME].values)
    ds = example_mass_balance_ds.assign({MASS_BALANCE_ERROR_COL_NAME: (("time", "y", "x"), err_data)})
    out = prepare_gridded_ds_for_selrem(ds)

    assert MASS_BALANCE_ERROR_COL_NAME in out.data_vars
    assert "y" in out.variables
    assert "x" in out.variables
    # values should not all be NaN (the error data was random non-NaN values)
    assert not np.all(np.isnan(out[MASS_BALANCE_ERROR_COL_NAME].values))


def test_prepare_error_col_values_match_input(example_mass_balance_ds: xr.Dataset):
    """The processed error column should reflect the input error data."""
    mb_data = np.ones_like(example_mass_balance_ds[MASS_BALANCE_COL_NAME].values)
    err_data = np.full_like(example_mass_balance_ds[MASS_BALANCE_COL_NAME].values, 2.0)
    ds = example_mass_balance_ds.assign(
        {
            MASS_BALANCE_COL_NAME: (("time", "y", "x"), mb_data),
            MASS_BALANCE_ERROR_COL_NAME: (("time", "y", "x"), err_data),
        }
    )
    out = prepare_gridded_ds_for_selrem(ds)

    np.testing.assert_allclose(out[MASS_BALANCE_ERROR_COL_NAME].values, 2.0)
    np.testing.assert_allclose(out[MASS_BALANCE_COL_NAME].values, 1.0)


def test_prepare_point_dim_drops_all_nan_points(example_mass_balance_ds: xr.Dataset):
    """Columns where every timestep is NaN should be excluded from points."""
    mb_data = example_mass_balance_ds[MASS_BALANCE_COL_NAME].values.copy()
    mb_data[:, -1, :] = np.nan  # make the last row all-NaN
    ds = example_mass_balance_ds.assign({MASS_BALANCE_COL_NAME: (("time", "y", "x"), mb_data)})
    out = prepare_gridded_ds_for_selrem(ds)
    # Last row is all-NaN -> those 4 points should be dropped
    assert (
        out.sizes["point"]
        == example_mass_balance_ds.sizes["y"] * example_mass_balance_ds.sizes["x"] - example_mass_balance_ds.sizes["x"]
    )


def test_prepare_latlon_coords_are_geographic(example_mass_balance_ds: xr.Dataset):
    """Output x/y coords (lon/lat) should be within geographic bounds."""
    out = prepare_gridded_ds_for_selrem(example_mass_balance_ds)

    lats = out["y"].values
    lons = out["x"].values
    assert np.all(lats >= -90) and np.all(lats <= 90)
    assert np.all(lons >= -180) and np.all(lons <= 180)


def test_prepare_point_index_is_sequential(example_mass_balance_ds: xr.Dataset):
    """The point coordinate should be a zero-based integer index."""
    out = prepare_gridded_ds_for_selrem(example_mass_balance_ds)
    np.testing.assert_array_equal(out["point"].values, np.arange(out.sizes["point"]))


def test_prepare_coarsen_reduces_points(example_mass_balance_ds: xr.Dataset):
    """Coarsening by 2 should reduce the spatial extent."""
    out_full = prepare_gridded_ds_for_selrem(example_mass_balance_ds, coarsen_factor=1)
    out_coarse = prepare_gridded_ds_for_selrem(example_mass_balance_ds, coarsen_factor=2)
    ny, nx = example_mass_balance_ds.sizes["y"], example_mass_balance_ds.sizes["x"]
    assert out_full.sizes["point"] == ny * nx
    assert out_coarse.sizes["point"] == (ny // 2) * (nx // 2)


def test_prepare_coarsen_factor_below_one_warns_and_does_not_coarsen(
    example_mass_balance_ds: xr.Dataset,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A coarsen_factor < 1 should emit a warning and leave the data un-coarsened."""
    out_default = prepare_gridded_ds_for_selrem(example_mass_balance_ds, coarsen_factor=1)

    with caplog.at_level(logging.WARNING, logger="sea_level_response.selrem.selrem_input_prep"):
        out_invalid = prepare_gridded_ds_for_selrem(example_mass_balance_ds, coarsen_factor=0)

    assert any("coarsen_factor < 1 is not supported" in r.message for r in caplog.records)
    assert out_invalid.sizes["point"] == out_default.sizes["point"]


def test_prepare_coarsen_sums_mass(example_mass_balance_ds: xr.Dataset):
    """With uniform data, coarsening (sum) should produce 4 x the per-cell value for factor=2 for both MB and error."""
    mb_data = np.ones_like(example_mass_balance_ds[MASS_BALANCE_COL_NAME].values)
    err_data = np.full_like(example_mass_balance_ds[MASS_BALANCE_COL_NAME].values, 2.0)
    ds = example_mass_balance_ds.assign(
        {
            MASS_BALANCE_COL_NAME: (("time", "y", "x"), mb_data),
            MASS_BALANCE_ERROR_COL_NAME: (("time", "y", "x"), err_data),
        }
    )
    out = prepare_gridded_ds_for_selrem(ds, coarsen_factor=2)
    # MB sums 4 cells of 1.0 -> 4.0; error combines in quadrature: sqrt(4 × 2.0^2) = 4.0
    np.testing.assert_allclose(out[MASS_BALANCE_COL_NAME].values, 4.0)
    np.testing.assert_allclose(out[MASS_BALANCE_ERROR_COL_NAME].values, 4.0)


def test_prepare_sub_annual_input_resampled(example_mass_balance_ds: xr.Dataset):
    """Sub-annual input should be resampled to annual resolution in the output."""
    monthly_times = pd.date_range("2010-01-01", periods=24, freq="ME")
    ny, nx = example_mass_balance_ds.sizes["y"], example_mass_balance_ds.sizes["x"]
    rng = np.random.default_rng(0)
    data = rng.standard_normal((len(monthly_times), ny, nx)).astype(np.float32)
    err_data = np.abs(rng.standard_normal((len(monthly_times), ny, nx))).astype(np.float32)
    ds = xr.Dataset(
        {
            MASS_BALANCE_COL_NAME: (("time", "y", "x"), data),
            MASS_BALANCE_ERROR_COL_NAME: (("time", "y", "x"), err_data),
            "spatial_ref": example_mass_balance_ds["spatial_ref"],
        },
        coords={
            "time": monthly_times,
            "y": example_mass_balance_ds.coords["y"],
            "x": example_mass_balance_ds.coords["x"],
        },
    )
    out = prepare_gridded_ds_for_selrem(ds)

    # 24 monthly steps over 2 years → 2 annual steps
    assert out.sizes["time"] == 2
    # spatial extent is unchanged by temporal resampling
    assert out.sizes["point"] == ny * nx
    # time labels should cover both years
    assert out.coords["time"].dt.year.values.tolist() == [2010, 2011]
    # values should match the per-year mean of the monthly input (stack order is row-major)
    np.testing.assert_allclose(
        out[MASS_BALANCE_COL_NAME].isel(time=0).values, data[:12].mean(axis=0).ravel(), rtol=1e-5
    )
    np.testing.assert_allclose(
        out[MASS_BALANCE_COL_NAME].isel(time=1).values, data[12:].mean(axis=0).ravel(), rtol=1e-5
    )
    # error is resampled using RSS/N
    np.testing.assert_allclose(
        out[MASS_BALANCE_ERROR_COL_NAME].isel(time=0).values,
        (np.sqrt((err_data[:12] ** 2).sum(axis=0)) / 12).ravel(),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        out[MASS_BALANCE_ERROR_COL_NAME].isel(time=1).values,
        (np.sqrt((err_data[12:] ** 2).sum(axis=0)) / 12).ravel(),
        rtol=1e-5,
    )
