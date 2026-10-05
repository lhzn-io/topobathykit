import logging
import os
import sys
import warnings
from pathlib import Path

import pytest
from dotenv import load_dotenv

from topobathykit.providers.gebco_2025 import GEBCO2025Provider as Gebco2025
from topobathykit.providers.noaa_bluetopo import NoaaBlueTopoProvider as BlueTopoProvider

# Coordinates for Execution Rocks, Long Island Sound (near Rye, NY)
# This should fall into one of the user's identified tiles (e.g. BH4XH5FN or similar)
EXECUTION_ROCKS_LAT = 40.8835
EXECUTION_ROCKS_LON = -73.7369


@pytest.mark.integration
@pytest.mark.network
def test_bluetopo_real_integration(tmp_path: Path) -> None:
    """Run an integration test against real NOAA BlueTopo APIs."""

    # 1. Configure Logging to see what's happening
    # Configure root logger to print to stderr so pytest captures it with -s
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
        force=True,  # Override any existing config
    )
    # Ensure our package logger is chatty
    logging.getLogger("topobathykit").setLevel(logging.DEBUG)

    # Load env vars
    load_dotenv()
    # Map OPEN_TOPOGRAPHY_API_KEY (from .env) to OPENTOPOGRAPHY_API_KEY (expected by lib)
    if "OPEN_TOPOGRAPHY_API_KEY" in os.environ:
        os.environ["OPENTOPOGRAPHY_API_KEY"] = os.environ["OPEN_TOPOGRAPHY_API_KEY"]

    # Suppress spurious warnings from third-party libs
    warnings.filterwarnings("ignore", category=UserWarning, module="pydap")

    # Use a temp directory for cache to verify download occurs
    cache_dir = tmp_path / "bathy_cache"
    provider = BlueTopoProvider(cache_dir=str(cache_dir))

    print(f"\n[Integration] Resolving tile for Lat: {EXECUTION_ROCKS_LAT}, Lon: {EXECUTION_ROCKS_LON}...")

    # 2. Test Tile Resolution (Requires network for GPKG download)
    tile_id = provider.resolve_tile_id(EXECUTION_ROCKS_LAT, EXECUTION_ROCKS_LON)
    print(f"[Integration] Resolved Tile ID: {tile_id}")

    # Assert we found *something*
    assert tile_id is not None, "Failed to resolve any tile for Execution Rocks location."

    # 3. Test Fetch Elevation (Requires network for COG download)
    print("[Integration] Fetching elevation (downloading COG)...")
    depth = provider.fetch_elevation(EXECUTION_ROCKS_LAT, EXECUTION_ROCKS_LON)
    print(f"[Integration] Elevation: {depth} meters")

    assert depth is not None, "Failed to fetch elevation value."
    assert isinstance(depth, float)
    # Depth in Long Island Sound is roughly 20-30m deep at max, Execution rocks is shallow.
    # Value should be reasonable (e.g. -50 to +10).
    assert -100.0 < depth < 100.0

    # 4. Verify Cache
    # We expect the Tile Scheme GPKG to be cached (we lazily stream TIFFs from S3)
    bluetopo_cache = cache_dir / "noaa_bluetopo"
    # Recursive glob in case of structure changes, or direct glob in subdir
    downloaded_files = list(bluetopo_cache.rglob("*.gpkg"))
    print(f"[Integration] Cached files: {[f.name for f in downloaded_files]}")
    assert len(downloaded_files) > 0, f"No GPKG scheme files found in cache {bluetopo_cache} after fetch."

    # 5. Compare with GEBCO
    print("\n[Integration] Fetching GEBCO 2025 value for comparison...")

    # CRITICAL FIX: Limit GEBCO scope to a tiny area to avoid downloading the whole world
    # Previous implementation used default bounds (globally!) which triggered massive downloads.
    # We just need a 1x1 degree tile around our point.
    s, n = EXECUTION_ROCKS_LAT - 0.05, EXECUTION_ROCKS_LAT + 0.05
    w, e = EXECUTION_ROCKS_LON - 0.05, EXECUTION_ROCKS_LON + 0.05

    gebco = Gebco2025(
        dem_type="GEBCO_2025",
        north=n,
        south=s,
        west=w,
        east=e,
        cache_dir=str(cache_dir),  # Reuse our temp cache
    )
    gebco.fetch()
    gebco_depth = gebco.sample_elevation(EXECUTION_ROCKS_LAT, EXECUTION_ROCKS_LON)

    print(f"[Integration] GEBCO Elevation: {gebco_depth} meters")

    delta = abs(depth - gebco_depth)
    print(f"[Integration] Difference (BlueTopo - GEBCO): {delta:.4f} meters")
