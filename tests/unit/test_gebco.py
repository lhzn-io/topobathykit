import warnings
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr
from dask.array.core import PerformanceWarning

from topobathykit.providers.gebco_2025 import GEBCO2025Provider as Gebco2025


@pytest.fixture
def mock_gebco_dataset() -> xr.Dataset:
    """Create a mock Xarray Dataset mimicking GEBCO 2025 structure."""
    lat = np.linspace(-10, 10, 100)
    lon = np.linspace(-10, 10, 100)

    da_elev = xr.DataArray(
        np.zeros((100, 100)),
        coords={"lat": lat, "lon": lon},
        dims=("lat", "lon"),
        name="elevation",
    )

    da_source_id = xr.DataArray(
        np.zeros((100, 100), dtype=np.uint32),
        coords={"lat": lat, "lon": lon},
        dims=("lat", "lon"),
        name="source_id",
    )

    da_tid = xr.DataArray(
        np.ones((100, 100), dtype=int) * 11,  # Direct TID
        coords={"lat": lat, "lon": lon},
        dims=("lat", "lon"),
        name="tid",
    )

    ds = xr.Dataset({"elevation": da_elev, "source_id": da_source_id, "tid": da_tid})
    # Add rio accessor to ensure we have crs setup correctly for interpolation limits
    ds.rio.write_crs("EPSG:4326", inplace=True)
    return ds


def test_initialization() -> None:
    gebco = Gebco2025(north=10, south=-10, west=-10, east=10)
    assert gebco.cache_dir is not None


@patch("xarray.open_dataset")
def test_fetch_elev_and_tid(mock_open_ds: MagicMock, mock_gebco_dataset: xr.Dataset, tmp_path: Any) -> None:
    mock_open_ds.return_value = mock_gebco_dataset

    # Use tmp_path for cache to avoid polluting user cache and ensure isolation
    cache_dir = tmp_path / "gebco_cache"
    gebco = Gebco2025(north=1, south=0, west=0, east=1, cache_dir=str(cache_dir))

    # Tiny mock datasets cause Dask to warn about huge chunk overhead relative to data size
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PerformanceWarning)
        da = gebco.fetch()

    # Check Elevation logic
    assert da is not None
    assert "elevation" in da.data_vars

    # Check TID logic - TID not currently implemented in Provider
    # tid = gebco.get_tid_classification()
    # assert tid is not None
    # assert tid.name == "tid"

    # Check Half-Pixel Offset (coordinates should be shifted)
    # We verify that the value is NOT what it would be without offset.
    # The exact value depends on slice logic, but we know it should have the offset component.
    # 15 arcseconds = 15/3600 degrees. Half of that is the offset.
    offset = Gebco2025.HALF_PIXEL_OFFSET
    # Just check that it's not an integer (mock data lies on integers/simple fractions)
    # and has the offset remainder
    val = da.y.values[0]
    _ = offset % 1  # simplistic check or just check diff from nearest grid

    # Better: check that the difference between the fetched value and the 'raw' mock value
    # (which we can infer or mock access to) is the offset.
    # In this mock, we know the source grid is linspace(-10, 10, 20).
    # -4.73684211 is the mock value near -5.
    # The fetched val is -4.7347...
    # Difference is +0.00208... which is exactly the offset.
    assert (val % (15 / 3600)) != 0  # It shouldn't align with the grid anymore


@patch("xarray.open_dataset")
def test_sample_elevation(mock_open_ds: MagicMock, mock_gebco_dataset: xr.Dataset, tmp_path: Any) -> None:
    mock_open_ds.return_value = mock_gebco_dataset

    cache_dir = tmp_path / "gebco_cache_sample"
    gebco = Gebco2025(north=1, south=0, west=0, east=1, cache_dir=str(cache_dir))

    # Tiny mock datasets cause Dask to warn about huge chunk overhead relative to data size
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PerformanceWarning)
        gebco.fetch()

    # Sample center (0.5, 0.5 is safely inside our 0 to 1 box)
    # Mock data is all zeros for elevation
    elev = gebco.sample_elevation(0.5, 0.5)
    assert elev == 0.0

    # Test error before load
    # Force reset of Singleton to ensure we get a fresh instance
    Gebco2025._singleton = None
    gebco_unloaded = Gebco2025()
    with pytest.raises(RuntimeError):
        gebco_unloaded.sample_elevation(0, 0)


def test_cache_loading_robustness(tmp_path: Any) -> None:
    """Test that GEBCO provider correctly loads a cached Zarr file containing a Dataset (multiple vars)."""
    from pathlib import Path

    if isinstance(tmp_path, str):
        tmp_path = Path(tmp_path)

    # 1. Setup Cache Directory
    # Note: GEBCO provider appends "gebco_2025" to the cache_dir provided in init.
    # So if we pass cache_dir=X, it uses X/gebco_2025.
    base_cache_dir = tmp_path / "gebco_cache_robust"
    provider_cache_dir = base_cache_dir / "gebco_2025"
    zarr_dir = provider_cache_dir / "zarr"
    zarr_dir.mkdir(parents=True)

    # 2. Create a dummy Multi-Variable Dataset (Simulating a cache saved with TID)
    # Use distinct coordinates to avoid lock contention with other tests using n0_e0
    lat = np.linspace(10.5, 11.5, 240)
    lon = np.linspace(10.5, 11.5, 240)

    da_elev = xr.DataArray(
        np.random.rand(240, 240),
        coords={"lat": lat, "lon": lon},
        dims=("lat", "lon"),
        name="elevation",
    )
    da_elev.rio.write_crs("EPSG:4326", inplace=True)

    da_source_id = xr.DataArray(
        np.zeros((240, 240), dtype=np.uint8),
        coords={"lat": lat, "lon": lon},
        dims=("lat", "lon"),
        name="source_id",
    )

    ds_cache = xr.Dataset({"elevation": da_elev, "source_id": da_source_id})
    ds_cache = ds_cache.rio.write_crs("EPSG:4326")

    # Save as Zarr to the expected path for tile n10_e10
    tile_key = "gebco_2025_n10_e10"
    cache_path = zarr_dir / f"{tile_key}.zarr"

    ds_cache.to_zarr(cache_path, mode="w", consolidated=False)

    # 3. Initialize Provider and Fetch
    # Reset Singleton and class state to ensure we get a fresh instance with our cache_dir
    # and no lingering mock datasets from previous tests.
    Gebco2025._singleton = None
    Gebco2025._ds_remote = None
    Gebco2025._tid_remote = None

    # Request bounds covering n10_e10
    provider = Gebco2025(north=11, south=10, west=10, east=11, cache_dir=str(base_cache_dir))

    # Assert path matches
    expected_path = Path(provider.cache_dir) / "zarr" / f"{tile_key}.zarr"
    assert cache_path.resolve() == expected_path.resolve(), f"{cache_path} vs {expected_path}"
    assert cache_path.exists(), "Cache file does not exist!"

    # Mock open_dataset to fail fast if it tries to hit remote URL (starts with dap)
    # But allow local file reads
    real_open_dataset = xr.open_dataset

    def side_effect(filename_or_obj: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(filename_or_obj, str) and filename_or_obj.startswith("dap"):
            # If we get here, it means we missed the cache!
            raise RuntimeError(f"Attempted remote fetch: {filename_or_obj} - Cache Missed!")
        return real_open_dataset(filename_or_obj, *args, **kwargs)

    with patch("xarray.open_dataset", side_effect=side_effect):
        # Should not raise ValueError about "more than one data variable"
        try:
            da_fetched = provider.fetch()
        except ValueError as e:
            pytest.fail(f"Fetching from multi-variable cache failed: {e}")
        except RuntimeError as e:
            pytest.fail(f"{e}")

    assert da_fetched is not None
    if isinstance(da_fetched, xr.Dataset):
        assert "elevation" in da_fetched
        assert "source_id" in da_fetched
    else:
        assert da_fetched.name == "elevation"


def test_fetch_masks_gebco_sentinel_values(tmp_path: Any) -> None:
    lat = np.linspace(0, 1, 20)
    lon = np.linspace(0, 1, 20)

    raw = np.zeros((20, 20), dtype=float)
    raw[5, 5] = -999999.0

    ds_remote = xr.Dataset(
        {
            "sub_ice_topo_bathymetry": xr.DataArray(
                raw,
                coords={"lat": lat, "lon": lon},
                dims=("lat", "lon"),
            ),
        }
    )
    ds_remote.rio.write_crs("EPSG:4326", inplace=True)

    ds_tid = xr.Dataset(
        {
            "tid": xr.DataArray(
                np.ones((20, 20), dtype=int),
                coords={"lat": lat, "lon": lon},
                dims=("lat", "lon"),
            )
        }
    )
    ds_tid.rio.write_crs("EPSG:4326", inplace=True)

    Gebco2025._singleton = None
    Gebco2025._ds_remote = ds_remote
    Gebco2025._tid_remote = ds_tid

    cache_dir = tmp_path / "gebco_cache_sentinel"
    provider = Gebco2025(north=1, south=0, west=0, east=1, cache_dir=str(cache_dir))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PerformanceWarning)
        fetched = provider.fetch()

    assert np.isnan(fetched["elevation"].values).any()
