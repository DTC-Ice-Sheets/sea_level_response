"""Testing the entrypoint."""

import re
import sys
from pathlib import Path
from unittest.mock import patch

import cyclopts
import pytest
import rioxarray  # noqa: F401
import xarray as xr

from sea_level_response import prepare_gridded_ds_for_selrem_entrypoint
from sea_level_response.prepare_gridded_ds_for_selrem_entrypoint import app


def test_main_valid(
    tmp_path: Path,
    gridded_and_expected_datasets: tuple[xr.Dataset, xr.Dataset],
) -> None:
    input_data = tmp_path / "input.zarr"
    output_path = tmp_path / "output.zarr"
    input_dataset, expected_dataset = gridded_and_expected_datasets
    input_dataset.to_zarr(input_data, zarr_format=2)

    with (
        patch.object(
            sys,
            "argv",
            [
                "sea_level_response.prepare_gridded_ds_for_selrem_entrypoint",
                "--gridded_dataset_geozarr",
                str(input_data),
                "--coarsen_factor",
                "2.0",
                "--output_path",
                str(output_path),
            ],
        ),
    ):
        app(result_action="return_value")
    with xr.open_zarr(output_path) as prepared_dataset:
        xr.testing.assert_equal(prepared_dataset.drop_vars("spatial_ref", errors="ignore"), expected_dataset)

    assert output_path.exists()


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


def test_main_raises_value_error_for_point_dataset(tmp_path: Path) -> None:
    point_dataset = xr.Dataset(coords={"point": [0]})

    with (
        patch.object(
            prepare_gridded_ds_for_selrem_entrypoint.xr,
            "open_zarr",
            return_value=point_dataset,
        ),
        pytest.raises(ValueError, match="Dataset is already in point format"),
    ):
        prepare_gridded_ds_for_selrem_entrypoint.main(
            gridded_dataset_geozarr="input.zarr",
            output_path=tmp_path / "output.zarr",
        )
