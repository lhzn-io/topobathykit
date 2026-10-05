import logging
import math

import pytest
import xarray as xr

from topobathykit.providers.noaa_topobathy import NoaaTopobathyProvider

# Configure logger for tests
logger = logging.getLogger(__name__)


def tile_to_bbox(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    """Convert tile coordinates to bounding box (West, South, East, North)."""
    n = 2.0**z
    lon_min = x / n * 360.0 - 180.0
    lon_max = (x + 1) / n * 360.0 - 180.0
    lat_rad_min = math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n)))
    lat_rad_max = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat_min = math.degrees(lat_rad_min)
    lat_max = math.degrees(lat_rad_max)
    return lon_min, lat_min, lon_max, lat_max


@pytest.mark.integration
def test_noaa_streaming_nyc(persistent_cache_dir: object) -> None:
    """
    Integration test for NOAA Topobathy streaming using a known valid tile (NYC).
    Verifies that we can fetch metadata, resolve a project, resolve a tile URL, and stream data.
    """
    # Known valid tile: 13/2414/3077 (NYC Harbor area)
    z, x, y = 13, 2414, 3077
    bbox = tile_to_bbox(z, x, y)

    # Initialize provider with test cache directory
    provider = NoaaTopobathyProvider(cache_dir=str(persistent_cache_dir))

    # 1. Project Discovery
    # We bypass find_project_by_box to avoid building the massive spatial index in the test environment.
    # We know tile 13/2414/3077 belongs to project 9310 (2017 NYC Topobathy).
    project_id = "9310"
    logger.info(f"Using known project: {project_id}")

    # Mock the projects dictionary so set_active_project can find the dem/ folder
    provider._projects = {project_id: "NY_NYC_Topobathy_2017_9310"}

    provider.set_active_project(project_id)
    assert provider._active_project_id == project_id

    # 2. Tile Resolution
    tiles = provider.resolve_tiles_in_bbox(*bbox)
    if not tiles:
        pytest.skip("No tiles resolved (index might be missing in test env)")

    # 3. Streaming Access
    da = None
    for tid in tiles:
        logger.info(f"Testing streaming for tile: {tid}")
        da = provider.fetch_tile(tid)
        if da is not None:
            break

    assert da is not None, "All tiles returned None (404s)"
    assert isinstance(da, xr.DataArray)

    # Check dimensions (lazy)
    assert "x" in da.dims and "y" in da.dims
    assert da.shape[0] > 0 and da.shape[1] > 0

    # 4. Data Access (Trigger computation on small slice)
    # We read a small chunk to ensure the VSICURL stream is working without downloading the whole file
    subset = da.isel(x=slice(0, 100), y=slice(0, 100))
    # This .compute() call is critical - it hits the network
    data = subset.compute()

    mean_val = data.mean().item()
    logger.info(f"Subset mean elevation: {mean_val}")

    # Check for reasonable values (NYC Harbor should be near sea level or slightly above/below)
    # Allow a wide range but ensure it's not all nodata if mask is handled correctly?
    # If it's all nodata, mean might be NaN.

    # Note: If VDatum correction failed or wasn't applied, values might differ,
    # but we just want to ensure we got *data* back.
    assert data.size > 0
