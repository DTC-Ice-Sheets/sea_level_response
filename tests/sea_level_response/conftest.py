"""Shared test fixtures for sea_level_response tests."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr


@pytest.fixture
def test_inputs_dir() -> Path:
    return Path(__file__).resolve().parent / "test_inputs"


@pytest.fixture
def example_mass_balance_dataset(test_inputs_dir: Path) -> xr.Dataset:
    path = test_inputs_dir / "jakobshavn_mass_balance.zarr"
    return xr.open_dataset(path, engine="zarr")


@pytest.fixture
def valid_geozarr_file(tmp_path: Path) -> Path:
    """Create a dummy GeoZarr store directory for testing.

    Parameters
    ----------
    tmp_path : Path
        Temporary path provided by pytest.

    Returns
    -------
    Path
        Path to the dummy GeoZarr store directory.
    """
    dummy_file = tmp_path / "dummy.zarr"
    dummy_file.mkdir()  # Create the directory to ensure it exists
    return dummy_file


@pytest.fixture
def invalid_geozarr_path(tmp_path: Path) -> Path:
    """Provide a path to a non-existent GeoZarr store directory for testing.

    Parameters
    ----------
    tmp_path : Path
        Temporary path provided by pytest.

    Returns
    -------
    Path
        Path to the non-existent GeoZarr store directory.
    """
    return tmp_path / "non_existent_file.zarr"


@pytest.fixture
def valid_url() -> str:
    """Provide a valid URL for testing."""
    return "https://example.com/valid.zarr"


@pytest.fixture
def invalid_url() -> str:
    """Provide an invalid URL for testing."""
    return "invalid_url/doesnt/start/with/http/or/https"


@pytest.fixture
def invalid_path() -> str:
    """Provide an invalid path for testing."""
    return "invalid_path/doesnt/exist"


@pytest.fixture
def simple_zarr_dataset() -> xr.Dataset:
    """Provide a dummy xarray Dataset for testing."""
    return xr.Dataset({"dummy_var": (("x", "y"), [[1, 2], [3, 4]])})


@pytest.fixture
def empty_str() -> str:
    """Provide an empty string for testing."""
    return ""


@pytest.fixture
def empty_path() -> Path:
    """Provide an empty Path for testing."""
    return Path()


@pytest.fixture
def mass_balance_results_dict() -> dict:
    """Provide a dummy mass balance dictionary for testing."""
    return {
        "total_mass_balance": 123.45,
        "control_mass_balance": 120.0,
        "x_min": -1000.0,
        "x_max": 1000.0,
        "x_res": 3,
        "y_min": -1000.0,
        "y_max": 1000.0,
        "y_res": 3,
        "years": [2000, 2001, 2002],
        "crs": "EPSG:3031",
        "ice_velocity": [1.0, 2.0, 3.0],
        "land_surface_temperature": [-10.0, -9.5, -9.0],
        "radar_elevation": [100.0, 101.0, 102.0],
        "prediction": [0.1, 0.2, 0.3],
    }


def _create_realistic_vmb_dataset(n_points: int = 100, n_times: int = 5) -> xr.Dataset:
    """Create a realistic VMB (Vertically-integrated Mass Balance) dataset.

    This mimics the structure of real precomputed mass balance datasets.

    Parameters
    ----------
    n_points : int
        Number of spatial points (default: 100).
    n_times : int
        Number of time steps (default: 5).

    Returns
    -------
    xr.Dataset
        Realistic VMB dataset with proper dimensions and coordinates.
    """
    from sea_level_response.config import MASS_BALANCE_COL_NAME, MASS_BALANCE_ERROR_COL_NAME

    # Create time coordinates (yearly data)
    times = pd.date_range("2020-01-01", periods=n_times, freq="YS")

    # Create spatial coordinates (lat/lon for ice sheet regions)
    # Greenland-like coordinates: ~60-80°N, ~-60 to -20°W
    lats = np.random.uniform(60, 80, n_points)
    lons = np.random.uniform(-60, -20, n_points)

    # Mass balance values (kg/yr per grid cell)
    # Realistic range: -1e11 to 1e11 kg/yr (melting to accumulation)
    mass_balance = np.random.uniform(-5e10, 5e10, (n_points, n_times))

    # Uncertainty (typically 10-20% of absolute value)
    mass_balance_error = np.abs(mass_balance) * np.random.uniform(0.1, 0.2, (n_points, n_times))

    return xr.Dataset(
        {
            MASS_BALANCE_COL_NAME: (["point", "time"], mass_balance),
            MASS_BALANCE_ERROR_COL_NAME: (["point", "time"], mass_balance_error),
        },
        coords={
            "point": np.arange(n_points),
            "time": times,
            "x": ("point", lons),
            "y": ("point", lats),
        },
    )


@pytest.fixture
def small_vmb_dataset() -> xr.Dataset:
    return _create_realistic_vmb_dataset(n_points=50, n_times=10)


@pytest.fixture
def large_vmb_dataset() -> xr.Dataset:
    return _create_realistic_vmb_dataset(n_points=100, n_times=5)


@pytest.fixture
def realistic_annual_slr_dataset() -> xr.Dataset:
    """Create a realistic annual SLR dataset.

    This mimics the output from SELREM computations with proper structure.

    Returns
    -------
    xr.Dataset
        Realistic annual SLR dataset with ndot, udot, sdot and uncertainties.
    """
    nx = 36
    ny = 18
    n_times = 5
    # Global grid: 1° resolution
    lons = np.linspace(-180, 179, nx)
    lats = np.linspace(-90, 89, ny)
    times = pd.date_range("2020-01-01", periods=n_times, freq="YS")

    # Create 3D grids
    _, lat_grid, time_idx = np.meshgrid(lons, lats, np.arange(n_times), indexing="ij")

    # Realistic SLR patterns (mm/yr)
    # ndot (absolute SLR): varies with distance from ice sheets, ~0-5 mm/yr
    ndot = np.abs(lat_grid / 90) * (time_idx + 1) * np.random.uniform(0.5, 2.0, (nx, ny, n_times))

    # udot (vertical deformation): small, ~-1 to 1 mm/yr, mostly near ice sheets
    udot = np.sin(lat_grid * np.pi / 180) * np.random.uniform(-0.5, 0.5, (nx, ny, n_times))

    # sdot (relative SLR) = ndot - udot
    sdot = ndot - udot

    # Uncertainties (10-15% of values)
    sig_ndot = np.abs(ndot) * 0.12
    sig_udot = np.abs(udot) * 0.15
    sig_sdot = np.abs(sdot) * 0.12

    return xr.Dataset(
        {
            "ndot": (["x", "y", "time"], ndot),
            "udot": (["x", "y", "time"], udot),
            "sdot": (["x", "y", "time"], sdot),
            "sig_ndot": (["x", "y", "time"], sig_ndot),
            "sig_udot": (["x", "y", "time"], sig_udot),
            "sig_sdot": (["x", "y", "time"], sig_sdot),
        },
        coords={"x": lons, "y": lats, "time": times},
    )


@pytest.fixture
def realistic_ocean_mask() -> tuple:
    """
    Create a realistic ocean mask dataset.

    Returns
    -------
    tuple
        (ocean_mask array, lons array, lats array)
        ocean_mask shape is (nx, ny) to match (lons, lats) for RegularGridInterpolator
    """
    nx = 36
    ny = 18
    lons = np.linspace(-180, 179, nx)
    lats = np.linspace(-90, 89, ny)

    # Simple ocean mask: 1 for ocean, 0 for land
    # Shape must be (nx, ny) to match coordinate order (lons, lats)
    ocean_mask = np.zeros((nx, ny))
    for j, lat in enumerate(lats):
        ocean_fraction = 0.7 - 0.3 * abs(lat) / 90  # More ocean at equator
        ocean_mask[:, j] = np.random.random(nx) < ocean_fraction

    return ocean_mask.astype(int), lons, lats
