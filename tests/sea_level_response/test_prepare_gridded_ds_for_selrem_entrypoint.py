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


def test_main_valid(
    tmp_path: Path,
    gridded_dataset_geozarr: Path,
    expected_dataset_geozarr: Path,
) -> None:
    coarsen_factor = 1.9
    output_path = tmp_path / "output.zarr"

    app(
        [
            "--gridded_dataset_geozarr",
            str(gridded_dataset_geozarr),
            "--coarsen_factor",
            str(coarsen_factor),
            "--output_path",
            str(output_path),
        ],
        result_action="return_value",
    )

    with xr.open_zarr(output_path) as output_dataset, xr.open_zarr(expected_dataset_geozarr) as expected_dataset:
        xr.testing.assert_identical(output_dataset, expected_dataset)


def test_main_ignores_processing_step_when_point_data_supplied(
    tmp_path: Path,
    point_zarr_dataset: xr.Dataset,
    caplog: pytest.LogCaptureFixture,
) -> None:
    input_path = tmp_path / "point_input.zarr"
    output_path = tmp_path / "point_output.zarr"
    point_zarr_dataset.to_zarr(input_path, zarr_format=2)

    prepare_gridded_ds_for_selrem_entrypoint.main(
        gridded_dataset_geozarr=input_path,
        output_path=output_path,
        coarsen_factor=2,
    )

    output_dataset = xr.open_zarr(output_path)
    xr.testing.assert_equal(output_dataset, point_zarr_dataset)
    assert "Dataset is already in point format!! Not modifying it." in caplog.messages


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
