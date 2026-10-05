from unittest.mock import patch

import numpy as np
import pytest
import rasterio
import rioxarray
import xarray as xr
from rasterio.transform import from_origin

from topobathykit.providers.noaa_topobathy import NoaaTopobathyProvider


@pytest.fixture
def mock_tif(tmp_path: object) -> object:
    """Creates a dummy GeoTIFF for testing (in UTM to test reprojection)."""
    tif_path = tmp_path / "test_tile_utm.tif"  # type: ignore
    # Create 100x100 array
    arr = np.ones((100, 100), dtype=np.float32)

    # Use EPSG:26918 (UTM 18N)
    # Approx coords for Oyster Bay NY: X=625000, Y=4530000
    # Resolution 10m (10.0, 10.0) -> 100x100 = 1000m x 1000m
    transform = from_origin(625000, 4531000, 10.0, 10.0)

    with rasterio.open(
        tif_path,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype=arr.dtype,
        crs="EPSG:26918",
        transform=transform,
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dst:
        dst.write(arr, 1)

    return tif_path


@pytest.mark.integration
def test_noaa_clip_logic(mock_tif: object) -> None:
    """
    Verifies the "Clip-then-Cache" logic using a local mock file in UTM.
    Ensures that fetch_layer correctly reprojects the requested bbox (EPSG:4326)
    to the source CRS (EPSG:26918) before clipping.
    """
    prov = NoaaTopobathyProvider()

    # Mock project metadata
    pid = "MOCK_PROJECT"
    prov._projects = {pid: "mock_folder_name"}

    # Use patch.object to ensure mocks attach correctly to the instance methods
    # AND patch fcntl.flock to avoid hangs
    orig_open = rioxarray.open_rasterio
    with (
        patch.object(prov, "find_projects_by_box", return_value=[pid]),
        patch.object(prov, "resolve_tiles_in_bbox", return_value=["test_tile_utm.tif"]),
        patch(
            "topobathykit.providers.noaa_topobathy.rioxarray.open_rasterio",
            side_effect=lambda *a, **kw: orig_open(mock_tif, masked=True),
        ),
        patch("fcntl.flock", return_value=None),
    ):
        # Request a sub-set in EPSG:4326 that overlaps with the UTM bbox
        # UTM Extent: 625000, 4530000 to 626000, 4531000
        # Approx Center: -73.509, 40.916
        # Let's request a small box (approx 500m) around the center.
        bbox = (-73.515, 40.910, -73.500, 40.920)

        print("Fetching layer with clip (Cross-CRS)...")
        # Resolution 0.0001 (~10m)
        try:
            da = prov.fetch_layer(bbox, resolution=0.0001)
        except Exception as e:
            pytest.fail(f"fetch_layer failed with exception: {e}")

        # Verify
        assert isinstance(da, xr.Dataset)
        da_elev = da["elevation"]
        print(f"Result Shape: {da_elev.shape}")

        # Trigger compute to test reading
        val = da_elev.compute()
        valid_count = val.count().item()
        print(f"Valid Pixels: {valid_count}")

        # Expectation:
        # If reprojection failed/ignored, clip would be using -73/40 on a grid of 625000/4530000 -> Empty
        # If reprojection worked, clip uses UTM coords -> Data found.

        assert valid_count > 8000, "Should have valid data after clip with reprojection"
