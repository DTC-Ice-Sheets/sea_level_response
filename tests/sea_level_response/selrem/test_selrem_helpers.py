"""Testing the SELREM helper functions."""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import mock_open, patch

import numpy as np
import pandas as pd
import xarray as xr

from sea_level_response.selrem.selrem_helpers import (
    compute_mean_mass_balance_over_time_window,
    load_ocean_mask,
    load_selrem_parameters,
    preprocess_location_timeseries,
    reproject_ocean_mask,
    slice_and_prepare_dataset,
)


def test_load_selrem_parameters() -> None:
    """Test loading SELREM parameters from JSON file."""
    fake_json_data = {
        "ndot_fit": [[1.0, 2.0]],
        "udot_fit": [[0.5, 1.0]],
        "ndot_unc": [[0.1, 0.2]],
        "udot_unc": [[0.05, 0.1]],
    }
    print("lashdah")

    with patch("sea_level_response.selrem.selrem_helpers.Path.open", mock_open(read_data=json.dumps(fake_json_data))):
        ndot_fit, udot_fit, ndot_unc, udot_unc = load_selrem_parameters()

    assert isinstance(ndot_fit, np.ndarray)
    assert ndot_fit.shape == (1, 2)


def test_load_ocean_mask() -> None:
    """Test loading ocean mask from NetCDF file."""
    fake_ds = xr.Dataset(
        {"ocean_mask": (["lat", "lon"], np.array([[1, 0], [0, 1]]))},
        coords={"lon": np.array([0.0, 10.0]), "lat": np.array([50.0, 60.0])},
    )

    with patch("xarray.open_dataset", return_value=fake_ds):
        ocean_mask, lons, lats = load_ocean_mask()

    assert ocean_mask.shape == (2, 2)


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

    # Verify all values are finite
    from sea_level_response.config import MASS_BALANCE_COL_NAME

    assert np.all(np.isfinite(result[MASS_BALANCE_COL_NAME].values))


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


def test_reproject_ocean_mask_realistic(realistic_ocean_mask: tuple) -> None:
    """Test reproject_ocean_mask with realistic ocean mask."""
    ocean_mask, mask_lons, mask_lats = realistic_ocean_mask

    # Reproject to finer grid
    target_lons = np.linspace(-180, 179, 72)
    target_lats = np.linspace(-90, 89, 36)

    reprojected = reproject_ocean_mask(ocean_mask, mask_lons, mask_lats, target_lons, target_lats)

    # Verify shape (should be nx, ny to match lon, lat)
    assert reprojected.shape == (72, 36)

    # Verify still binary
    assert np.all((reprojected == 0) | (reprojected == 1))

    # Check ocean fraction is reasonable (roughly 40-90% due to randomness)
    # Real Earth is ~71% ocean, but our simple model has variability
    ocean_fraction = reprojected.sum() / reprojected.size
    assert 0.4 <= ocean_fraction <= 0.9


def test_compute_mean_mass_balance_over_time_window() -> None:
    """Test computing mean mass balance."""
    from sea_level_response.config import MASS_BALANCE_COL_NAME, MASS_BALANCE_ERROR_COL_NAME

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
    np.testing.assert_allclose(result[MASS_BALANCE_COL_NAME].values[0], 3.0)
