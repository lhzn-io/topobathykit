from pathlib import Path

import pytest
import xarray as xr
from pyproj import Transformer

from topobathykit.providers.usgs_lidar import UsgsLidarProvider as LidarProvider

EPT_URL = (
    "https://s3-us-west-2.amazonaws.com/usgs-lidar-public/USGS_LPC_NY_LongIsland_Z18_2014_LAS_2015/ept.json"
)

# Central Long Island (Islip area) - Sure land
# Must project to Web Mercator (EPSG:3857) matching the EPT resource.
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
minx, miny = transformer.transform(-73.10, 40.80)
maxx, maxy = transformer.transform(-73.08, 40.82)
BOUNDS = (minx, miny, maxx, maxy)


@pytest.mark.integration
@pytest.mark.network
def test_fetch_lidar_from_ept(persistent_cache_dir: Path) -> None:
    provider = LidarProvider(cache_dir=str(persistent_cache_dir))
    print(f"\nFetching from {EPT_URL}...")
    print(f"Bounds (EPSG:3857): {BOUNDS}")

    # 20m resolution (Meters)
    da = provider.fetch_lidar_from_ept(EPT_URL, BOUNDS, resolution=20.0, target_crs="EPSG:3857")

    assert da is not None
    assert isinstance(da, xr.DataArray)
    # Result should be returned in 3857
    assert da.rio.crs.to_epsg() == 3857

    # Check data content
    print(f"Shape: {da.shape}")
    print(f"Min/Max: {da.min().item()}, {da.max().item()}")

    assert da.shape[0] > 5, "Should have decent Y dimension"
    assert da.shape[1] > 5, "Should have decent X dimension"
    assert da.count() > 0, "Should have valid pixels"
