"""Testing the entrypoint."""

import re
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import ANY, patch

import cyclopts
import pytest
import xarray as xr

from sea_level_response import prepare_gridded_ds_for_selrem_entrypoint
from sea_level_response.prepare_gridded_ds_for_selrem_entrypoint import app


@pytest.mark.parametrize(
    "input_data_source",
    [
        "valid_geozarr_file",
        "valid_url",
    ],
)
def test_main_valid(
    tmp_path: Path,
    input_data_source: str,
    simple_zarr_dataset: xr.Dataset,
    request,
) -> None:
    """
    Test main.

    Parameters
    ----------
    caplog : pytest.LogCaptureFixture
        The caplog fixture
    tmp_path : Path
        Temporary directory for output
    """
    input_data = str(request.getfixturevalue(input_data_source))
    from sea_level_response.prepare_gridded_ds_for_selrem_entrypoint import app

    start_time = datetime(2020, 1, 1, 0, 0, 0)
    end_time = datetime(2020, 12, 31, 23, 59, 59)

    coarsen_factor = 2.0

    with (
        patch.object(
            sys,
            "argv",
            [
                "sea_level_response.prepare_gridded_ds_for_selrem_entrypoint",
                "--start_time",
                start_time.isoformat(),
                "--end_time",
                end_time.isoformat(),
                "--mass_balance_geozarr",
                input_data,
                "--coarsen_factor",
                str(coarsen_factor),
                "--output_path",
                str(tmp_path / "output.zarr"),
            ],
        ),
        patch.object(
            prepare_gridded_ds_for_selrem_entrypoint, "prepare_gridded_ds_for_selrem", return_value=simple_zarr_dataset
        ) as mock_prepare_gridded,
        patch.object(
            prepare_gridded_ds_for_selrem_entrypoint, "slice_and_prepare_dataset", return_value=simple_zarr_dataset
        ) as mock_slice_and_prepare,
    ):
        app(result_action="return_value")
        mock_slice_and_prepare.assert_called_once_with(
            vmb_dataset_path=str(input_data),
            start_time=start_time,
            end_time=end_time,
        )

        mock_prepare_gridded.assert_called_once()
        assert mock_prepare_gridded.call_args[0][0] is simple_zarr_dataset
        mock_prepare_gridded.assert_called_once_with(ANY, coarsen_factor=coarsen_factor)

    # verify the output file was created
    assert (tmp_path / "output.zarr").exists()


@pytest.mark.parametrize(
    "test_source",
    ["invalid_geozarr_path", "invalid_url"],
)
def test_invalid_path_raises_validation_error(tmp_path: Path, test_source: str, request) -> None:
    error_string = test_data = str(request.getfixturevalue(test_source))

    with (
        patch.object(
            sys,
            "argv",
            [
                "sea_level_response.prepare_gridded_ds_for_selrem_entrypoint",
                "--start_time",
                "2020-01-01T00:00:00",
                "--end_time",
                "2020-12-31T23:59:59",
                "--mass_balance_geozarr",
                test_data,
                "--coarsen_factor",
                "2.0",
                "--output_path",
                str(tmp_path / "output.zarr"),
            ],
        ),
        pytest.raises(
            cyclopts.ValidationError,
            match=re.escape(f"The provided path '{error_string}' is neither a valid local path nor a web URL."),
        ),
    ):
        app(result_action="return_value", exit_on_error=False)


def test_main_raises_value_error_for_point_dataset(tmp_path: Path) -> None:
    point_dataset = xr.Dataset(coords={"point": [0]})

    with (
        patch.object(
            prepare_gridded_ds_for_selrem_entrypoint,
            "slice_and_prepare_dataset",
            return_value=point_dataset,
        ),
        pytest.raises(ValueError, match="Dataset is already in point format"),
    ):
        prepare_gridded_ds_for_selrem_entrypoint.main(
            mass_balance_geozarr="input.zarr",
            start_time=datetime(2020, 1, 1),
            end_time=datetime(2020, 12, 31),
            output_path=tmp_path / "output.zarr",
        )
