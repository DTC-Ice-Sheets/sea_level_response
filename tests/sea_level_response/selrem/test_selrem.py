"""Integration tests for SELREM computation."""

from pathlib import Path

import pytest
import xarray as xr

from sea_level_response.selrem.selrem import compute_annual_slr, compute_global_slr


@pytest.mark.parametrize(
    ("input_filename", "expected_output_filename"),
    [
        ("jakobshavn_mass_balance.zarr", "expected_annual_slr_for_jakobshavn_mb_sampled.zarr"),
        ("GrIS_VMB_subset.zarr", "expected_annual_slr_for_GrIS_VMB_subset_sampled.zarr"),
    ],
)
def test_integrated_annual_slr(input_filename: str, expected_output_filename: str, test_inputs_dir: Path):
    example_mass_balance_dataset = xr.open_dataset(test_inputs_dir / input_filename, engine="zarr")
    output_annual_slr = compute_annual_slr(example_mass_balance_dataset, 3)
    assert isinstance(output_annual_slr, xr.Dataset)
    assert output_annual_slr.dims["time"] == example_mass_balance_dataset.dims["time"]
    assert list(output_annual_slr.data_vars) == [
        "ndot",
        "udot",
        "sdot",
        "sig_ndot",
        "sig_udot",
        "sig_sdot",
        "rot_total",
    ]
    output_subset_ds = output_annual_slr.isel(
        x=slice(0, None, 50),
        y=slice(0, None, 25),
    )
    expected_output = xr.open_dataset(test_inputs_dir / expected_output_filename, engine="zarr")
    xr.testing.assert_allclose(output_subset_ds, expected_output, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize(
    ("input_filename", "expected_output_filename"),
    [
        ("jakobshavn_mass_balance.zarr", "expected_global_slr_for_jakobshavn_mb_sampled.zarr"),
        ("GrIS_VMB_subset.zarr", "expected_global_slr_for_GrIS_VMB_subset_sampled.zarr"),
    ],
)
def test_integrated_global_slr(input_filename: str, expected_output_filename: str, test_inputs_dir: Path):
    example_mass_balance_dataset = xr.open_dataset(test_inputs_dir / input_filename, engine="zarr")
    output_global_slr = compute_global_slr(example_mass_balance_dataset, 3)
    assert isinstance(output_global_slr, xr.Dataset)
    assert "time" not in output_global_slr.dims
    assert sorted(output_global_slr.data_vars) == [
        "ndot",
        "rot_total",
        "sdot",
        "sig_ndot",
        "sig_sdot",
        "sig_udot",
        "udot",
    ]
    output_subset_ds = output_global_slr.isel(
        x=slice(0, None, 50),
        y=slice(0, None, 25),
    )
    expected_output = xr.open_dataset(test_inputs_dir / expected_output_filename, engine="zarr")
    xr.testing.assert_allclose(output_subset_ds, expected_output, rtol=1e-5, atol=1e-8)
