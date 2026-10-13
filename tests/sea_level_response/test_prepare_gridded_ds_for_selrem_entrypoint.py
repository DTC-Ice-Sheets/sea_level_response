"""Testing the entrypoint."""

import re
import sys
from pathlib import Path
from unittest.mock import patch

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
    input_data = str(request.getfixturevalue(input_data_source))
    coarsen_factor = 2

    with (
        patch.object(
            sys,
            "argv",
            [
                "sea_level_response.prepare_gridded_ds_for_selrem_entrypoint",
                "--gridded_dataset_geozarr",
                input_data,
                "--coarsen_factor",
                str(coarsen_factor),
                "--output_path",
                str(tmp_path / "output.zarr"),
            ],
        ),
        patch.object(
            prepare_gridded_ds_for_selrem_entrypoint.xr,
            "open_zarr",
            return_value=simple_zarr_dataset,
        ) as mock_open_zarr,
        patch.object(
            prepare_gridded_ds_for_selrem_entrypoint, "prepare_gridded_ds_for_selrem", return_value=simple_zarr_dataset
        ) as mock_prepare_gridded,
    ):
        app(result_action="return_value")
        mock_open_zarr.assert_called_once_with(input_data)
        mock_prepare_gridded.assert_called_once_with(simple_zarr_dataset, coarsen_factor=coarsen_factor)

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
                "--gridded_dataset_geozarr",
                test_data,
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


def test_main_processes_saved_gridded_dataset(
    tmp_path: Path,
    gridded_dataset_geozarr: Path,
) -> None:
    output_path = tmp_path / "output.zarr"

    prepare_gridded_ds_for_selrem_entrypoint.main(
        gridded_dataset_geozarr=gridded_dataset_geozarr,
        output_path=output_path,
    )

    assert output_path.exists()
