"""Testing the SLR time series entrypoint."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from sea_level_response import slr_ts_entrypoint


def test_main_with_valid_inputs(caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
    """Test main with valid latitude and longitude inputs.

    Parameters
    ----------
    caplog : pytest.LogCaptureFixture
        The caplog fixture for capturing log output.
    tmp_path : Path
        Temporary directory for output.
    """
    from sea_level_response.slr_ts_entrypoint import app

    # Create a fake annual SLR dataset to mock
    fake_annual_slr_ds = xr.Dataset(
        {
            "ndot": (["x", "y", "time"], [[[1.0, 2.0]]]),
            "udot": (["x", "y", "time"], [[[0.5, 1.0]]]),
            "sdot": (["x", "y", "time"], [[[0.5, 1.0]]]),
            "sig_ndot": (["x", "y", "time"], [[[0.1, 0.2]]]),
            "sig_udot": (["x", "y", "time"], [[[0.05, 0.1]]]),
            "sig_sdot": (["x", "y", "time"], [[[0.05, 0.1]]]),
        },
        coords={
            "x": [0.0],
            "y": [55.0],
            "time": ["2020-01-01", "2021-01-01"],
        },
    )

    output_file = tmp_path / "output_timeseries.json"

    with (
        patch.object(
            sys,
            "argv",
            [
                "sea_level_response.slr_ts_entrypoint",
                "--annual_slr_gz_path",
                "path/to/annual_slr.zarr",
                "--output_path",
                str(output_file),
                "--latitude",
                "55.0",
                "--longitude",
                "0.0",
            ],
        ),
        patch("xarray.open_dataset", return_value=fake_annual_slr_ds),
        patch.object(
            slr_ts_entrypoint,
            "preprocess_location_timeseries",
            return_value={
                "years": [2020, 2021],
                "sdot_cumulative": [1.0, 2.0],
                "ndot_cumulative": [2.0, 4.0],
                "udot_cumulative": [0.5, 1.0],
                "sig_sdot_cumulative": [0.1, 0.2],
                "sig_ndot_cumulative": [0.2, 0.4],
                "sig_udot_cumulative": [0.05, 0.1],
                "slope_sdot": 1.0,
                "slope_ndot": 2.0,
                "slope_udot": 0.5,
                "actual_lat": 55.0,
                "actual_lon": 0.0,
            },
        ) as mock_preprocess,
    ):
        app(result_action="return_value")

        # Verify preprocessing was called with correct parameters
        mock_preprocess.assert_called_once()
        call_kwargs = mock_preprocess.call_args.kwargs
        assert np.isclose(call_kwargs["lat0"], 55.0)
        assert np.isclose(call_kwargs["lon0"], 0.0)
        assert isinstance(call_kwargs["annual_slr_ds"], xr.Dataset)

    # Verify output file was created
    assert output_file.exists()

    # Verify JSON content
    with Path(output_file).open() as f:
        data = json.load(f)
    assert np.isclose(data["years"], [2020, 2021]).all()
    assert np.isclose(data["slope_sdot"], 1.0)
    assert len(data["sdot_cumulative"]) == 2

    # Check log messages
    assert "Location time series preprocessing complete" in caplog.text
    assert "Output contains 2 time steps" in caplog.text


def test_main_missing_coordinates(tmp_path: Path) -> None:
    """Test that main raises ValueError when latitude or longitude is missing.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory for output.
    """
    from sea_level_response.slr_ts_entrypoint import app

    output_file = tmp_path / "output_timeseries.json"

    # Test with missing longitude
    with (
        patch.object(
            sys,
            "argv",
            [
                "sea_level_response.slr_ts_entrypoint",
                "--annual_slr_gz_path",
                "path/to/annual_slr.zarr",
                "--output_path",
                str(output_file),
                "--latitude",
                "55.0",
                # Missing --longitude
            ],
        ),
        patch("xarray.open_dataset", return_value=xr.Dataset()),
        pytest.raises(ValueError, match="Both latitude and longitude must be provided"),
    ):
        app(result_action="return_value")


def test_main_creates_output_directory(tmp_path: Path) -> None:
    """Test that main creates output directory if it doesn't exist.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory for output.
    """
    from sea_level_response.slr_ts_entrypoint import app

    # Create nested directory path that doesn't exist yet
    output_file = tmp_path / "nested" / "dir" / "output.json"
    assert not output_file.parent.exists()

    fake_annual_slr_ds = xr.Dataset(
        {
            "ndot": (["x", "y", "time"], [[[1.0]]]),
            "udot": (["x", "y", "time"], [[[0.5]]]),
            "sdot": (["x", "y", "time"], [[[0.5]]]),
            "sig_ndot": (["x", "y", "time"], [[[0.1]]]),
            "sig_udot": (["x", "y", "time"], [[[0.05]]]),
            "sig_sdot": (["x", "y", "time"], [[[0.05]]]),
        },
        coords={"x": [0.0], "y": [55.0], "time": ["2020-01-01"]},
    )

    with (
        patch.object(
            sys,
            "argv",
            [
                "sea_level_response.slr_ts_entrypoint",
                "--annual_slr_gz_path",
                "path/to/annual_slr.zarr",
                "--output_path",
                str(output_file),
                "--latitude",
                "55.0",
                "--longitude",
                "0.0",
            ],
        ),
        patch("xarray.open_dataset", return_value=fake_annual_slr_ds),
        patch.object(
            slr_ts_entrypoint,
            "preprocess_location_timeseries",
            return_value={
                "years": [2020],
                "sdot_cumulative": [1.0],
                "ndot_cumulative": [2.0],
                "udot_cumulative": [0.5],
                "sig_sdot_cumulative": [0.1],
                "sig_ndot_cumulative": [0.2],
                "sig_udot_cumulative": [0.05],
                "slope_sdot": 1.0,
                "slope_ndot": 2.0,
                "slope_udot": 0.5,
                "actual_lat": 55.0,
                "actual_lon": 0.0,
            },
        ),
    ):
        app(result_action="return_value")

    # Verify directory was created
    assert output_file.parent.exists()
    assert output_file.exists()
