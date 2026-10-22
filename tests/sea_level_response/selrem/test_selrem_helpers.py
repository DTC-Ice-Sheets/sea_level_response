"""Testing the SELREM helper functions."""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import mock_open, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sea_level_response.config import MASS_BALANCE_COL_NAME, MASS_BALANCE_ERROR_COL_NAME
from sea_level_response.selrem.selrem_helpers import (
    bin_loads_to_points,
    build_lonlat,
    compute_mean_mass_balance_over_time_window,
    interp_ocean_mask_to,
    interp_to_grid,
    load_ocean_mask,
    load_selrem_parameters,
    preprocess_location_timeseries,
    slice_and_prepare_dataset,
    wrap180,
)


def test_load_selrem_parameters() -> None:
    """Test loading SELREM parameters from JSON file."""
    fake_json_data = {
        "gdot_fit": [[1.0, 2.0]],
        "udot_fit": [[0.5, 1.0]],
        "gdot_unc": [[0.1, 0.2]],
        "udot_unc": [[0.05, 0.1]],
    }

    with patch("sea_level_response.selrem.selrem_helpers.Path.open", mock_open(read_data=json.dumps(fake_json_data))):
        gdot_fit, _, _, _ = load_selrem_parameters()

    assert isinstance(gdot_fit, np.ndarray)
    assert gdot_fit.shape == (1, 2)


def test_load_ocean_mask() -> None:
    """Test loading ocean mask from NetCDF file."""
    fake_ds = xr.Dataset(
        {"ocean_mask": (["lat", "lon"], np.array([[1, 0], [0, 1]]))},
        coords={"lon": np.array([0.0, 10.0]), "lat": np.array([50.0, 60.0])},
    )

    with patch("xarray.open_dataset", return_value=fake_ds):
        ocean_mask, _, _ = load_ocean_mask()

    assert ocean_mask.shape == (2, 2)


def test_wrap180() -> None:
    """Test longitude wrapping to [-180, 180)."""
    result = wrap180(np.array([0.0, 190.0, 360.0, 180.0, -190.0]))
    np.testing.assert_allclose(result, [0.0, -170.0, 0.0, -180.0, 170.0])
    assert wrap180(270.0) == pytest.approx(-90.0)


def test_build_lonlat() -> None:
    """Test building a regular global grid."""
    lons, lats, lon2d, lat2d = build_lonlat(90.0)
    np.testing.assert_allclose(lons, [-180.0, -90.0, 0.0, 90.0])
    np.testing.assert_allclose(lats, [90.0, -90.0])
    assert lon2d.shape == (4, 2)
    assert lat2d.shape == (4, 2)
    np.testing.assert_allclose(lon2d[:, 0], lons)
    np.testing.assert_allclose(lat2d[0, :], lats)


def test_interp_to_grid() -> None:
    """Test interpolation to a target grid, with NaN outside the source grid."""
    lons_src = np.array([0.0, 10.0])
    lats_src = np.array([0.0, 10.0])
    field = np.array([[0.0, 0.0], [10.0, 10.0]])

    lon2d_t, lat2d_t = np.meshgrid(np.array([5.0, 20.0]), np.array([5.0]), indexing="ij")
    result = interp_to_grid(field, lons_src, lats_src, lon2d_t, lat2d_t)

    assert result.shape == (2, 1)
    assert result[0, 0] == pytest.approx(5.0)
    assert np.isnan(result[1, 0])


def test_slice_and_prepare_dataset_with_realistic_data(tmp_path: Path, small_vmb_dataset: xr.Dataset) -> None:
    """Test slice_and_prepare_dataset with realistic VMB data."""
    # Save to temporary zarr
    zarr_path = tmp_path / "realistic_vmb.zarr"
    small_vmb_dataset.to_zarr(zarr_path)

    # Test slicing
    start_time = datetime(2021, 1, 1)
    end_time = datetime(2024, 12, 31)

    result = slice_and_prepare_dataset(zarr_path, start_time, end_time)

    # Verify structure
    assert "time" in result.dims
    assert "point" in result.dims
    assert len(result.time) <= 10
    assert len(result.point) == 50


def test_compute_mean_with_realistic_vmb_data(large_vmb_dataset: xr.Dataset) -> None:
    """Test compute_mean_mass_balance_over_time_window with realistic VMB data."""
    # Compute mean without time slicing
    result = compute_mean_mass_balance_over_time_window(large_vmb_dataset)

    # Verify time dimension removed
    assert "time" not in result.dims
    assert "point" in result.dims
    assert len(result.point) == 100
    assert np.all(np.isfinite(result[MASS_BALANCE_COL_NAME].to_numpy()))


def test_preprocess_location_with_realistic_annual_slr(realistic_annual_slr_dataset) -> None:
    """Test preprocess_location_timeseries with realistic annual SLR data."""
    # Test at a specific location (mid-Atlantic)
    lat0 = 40.0
    lon0 = -30.0

    output = preprocess_location_timeseries(realistic_annual_slr_dataset, lat0, lon0)

    # Verify output structure
    assert isinstance(output, dict)
    assert len(output["years"]) == 5
    assert output["years"] == [2020, 2021, 2022, 2023, 2024]

    # Verify realistic values
    assert all(isinstance(v, float) for v in output["sdot_cumulative"])
    assert all(isinstance(v, float) for v in output["ndot_cumulative"])

    # Slopes should be reasonable (not NaN or inf)
    assert np.isfinite(output["slope_sdot"])
    assert np.isfinite(output["slope_ndot"])
    assert np.isfinite(output["slope_udot"])


def test_interp_ocean_mask_to_realistic(realistic_ocean_mask: tuple) -> None:
    """Test interp_ocean_mask_to with realistic ocean mask."""
    ocean_mask, _mask_lons, _mask_lats = realistic_ocean_mask

    # Reproject to finer grid
    target_lons = np.linspace(-180, 179, 72)
    target_lats = np.linspace(-90, 89, 36)

    reprojected = interp_ocean_mask_to(target_lons, target_lats, ocean_mask)

    # Verify shape (should be nx, ny to match lon, lat)
    assert reprojected.shape == (72, 36)

    # Verify still binary
    assert np.all((reprojected == 0) | (reprojected == 1))

    # Check ocean fraction is reasonable: Real Earth is ~71% ocean
    ocean_fraction = reprojected.sum() / reprojected.size
    assert 0.4 <= ocean_fraction <= 0.9


def test_bin_loads_to_points() -> None:
    """Loads in the same 1 degree bin are summed."""
    lat = np.array([10.2, 10.4])
    lon = np.array([20.1, 20.3])
    mass = np.array([1.0, 2.0])
    err = np.array([3.0, 4.0])

    lat_c, lon_c, m, me = bin_loads_to_points(lat, lon, mass, err)

    np.testing.assert_allclose(lat_c, [10.5])
    np.testing.assert_allclose(lon_c, [20.5])
    np.testing.assert_allclose(m, [3.0])
    np.testing.assert_allclose(me, [5.0])


def test_compute_mean_mass_balance_over_time_window() -> None:
    """Test computing mean mass balance."""
    times = pd.date_range("2020-01-01", periods=5, freq="YS")
    ds = xr.Dataset(
        {
            MASS_BALANCE_COL_NAME: (["point", "time"], np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])),
            MASS_BALANCE_ERROR_COL_NAME: (["point", "time"], np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])),
        },
        coords={"point": [0], "time": times, "x": ("point", [0.0]), "y": ("point", [50.0])},
    )

    result = compute_mean_mass_balance_over_time_window(ds)
    assert "time" not in result.dims
    np.testing.assert_allclose(result[MASS_BALANCE_COL_NAME].to_numpy()[0], 3.0)
