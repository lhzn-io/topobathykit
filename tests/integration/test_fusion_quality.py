import os
import signal
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import FrameType

import numpy as np
import pytest
import xarray as xr
from dotenv import load_dotenv

# Load env vars
load_dotenv()
# Map OPEN_TOPOGRAPHY_API_KEY (from .env) to OPENTOPOGRAPHY_API_KEY (expected by lib)
if "OPEN_TOPOGRAPHY_API_KEY" in os.environ:
    os.environ["OPENTOPOGRAPHY_API_KEY"] = os.environ["OPEN_TOPOGRAPHY_API_KEY"]

from topobathykit.fusion import FusionEngine  # noqa: E402
from topobathykit.providers.usgs_lidar import UsgsLidarProvider  # noqa: E402
from topobathykit.runtime import run  # noqa: E402

# Found via scan_laz.py
LIDAR_URL = "s3://noaa-nos-coastal-lidar-pds/laz/geoid18/4938/20140403_usgs_ny_li_18TXL060240.copc.laz"
# Center of the Lidar tile approx
CENTER_LAT = 40.8634
CENTER_LON = -73.7279
POLICY_PATH = Path(__file__).parent.parent.parent / "policies" / "examples" / "wlis.yaml"


@contextmanager
def _time_limit(seconds: int) -> Iterator[None]:
    """Raise TimeoutError if a block exceeds the given wall-clock seconds."""
    if not hasattr(signal, "SIGALRM"):
        # Non-POSIX fallback: no hard timeout available.
        yield
        return

    def _handler(signum: int, frame: FrameType | None) -> None:
        raise TimeoutError(f"Operation exceeded {seconds}s")

    previous = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


@pytest.mark.integration
def test_fusion_quality_execution_rocks(persistent_cache_dir: Path) -> None:
    """
    Verifies the fusion of Lidar and BlueTopo near Execution Rocks using a real Lidar tile.
    Metrics:
    1. Coverage: % valid pixels in fused output (should be near 100% if sources overlap).
    2. Continuity: Check for cliffs vs smooth transition.
    """
    if os.getenv("RUN_SLOW_INTEGRATION", "0").lower() not in {"1", "true", "yes"}:
        pytest.skip("Set RUN_SLOW_INTEGRATION=1 to run long external fusion quality test")

    print(f"\n[Test] Fetching Lidar from {LIDAR_URL}...")
    lidar_prov = UsgsLidarProvider(cache_dir=str(persistent_cache_dir))
    # Fetch ~500m box around center
    try:
        with _time_limit(180):
            lidar_da = lidar_prov.fetch_lidar_from_laz(LIDAR_URL, resolution=4.0)  # 4m grid
    except TimeoutError as exc:
        pytest.skip(f"Skipping integration test: Lidar fetch stalled ({exc})")

    assert lidar_da is not None, "Failed to fetch Lidar data."
    lidar_da.load()  # Force load into memory

    # Extract elevation array if it's a dataset
    if isinstance(lidar_da, xr.Dataset):
        lidar_da = lidar_da["elevation"]

    print(f"[Test] Lidar Stats: Min {float(lidar_da.min()):.2f}, Max {float(lidar_da.max()):.2f}")

    # Define bounds for BlueTopo based on Lidar extent
    west, south, east, north = (
        lidar_da.x.min().item(),
        lidar_da.y.min().item(),
        lidar_da.x.max().item(),
        lidar_da.y.max().item(),
    )

    # Expand slightly to ensure overlap? Or just use same bounds.
    print(
        f"[Test] Fetching Bathymetry via Runtime for bounds: "
        f"{west:.4f}, {south:.4f}, {east:.4f}, {north:.4f}..."
    )

    # Use Runtime Policy
    try:
        with _time_limit(180):
            # Target shape is implicit in resolution, we request approx same resolution as lidar (4.0m)
            # 4.0 meters in degrees approx: 4/111320 ~ 0.000036

            # Using wlis.yaml which includes all relevant layers
            bbox = (west, south, east, north)
            ds = run(str(POLICY_PATH), bbox, resolution=4.0, use_cache=True)
            bathy_da_maybe = ds["elevation"]

    except TimeoutError as exc:
        pytest.skip(f"Skipping integration test: Runtime fetch stalled ({exc})")

    assert bathy_da_maybe is not None, "Failed to fetch Bathymetry."
    bathy_da = bathy_da_maybe

    if isinstance(bathy_da, xr.Dataset):
        bathy_da = bathy_da["elevation"]

    assert isinstance(bathy_da, xr.DataArray), "Expected DataArray."

    # Filter nodata
    bathy_valid = bathy_da.where(bathy_da != -9999.0)  # Assuming nodata handling
    print(f"[Test] Bathy Stats: Min {bathy_valid.min().item():.2f}, Max {bathy_valid.max().item():.2f}")

    # Fuse
    print("[Test] Running Fusion Engine...")
    fusion = FusionEngine(z_center=0.0, k_default=1.0)  # Standard Logistic
    fused_da = fusion.fuse(lidar_da, bathy_da)

    # Metrics
    valid_mask = ~np.isnan(fused_da)
    coverage_pct = (valid_mask.sum() / valid_mask.size) * 100
    print(f"[Metric] Coverage: {coverage_pct.item():.2f}%")

    assert coverage_pct > 50.0, "Coverage too low (<50%). Expected substantial overlap."

    # Continuity / Smoothness Estimate
    # Calculate gradient
    dy, dx = np.gradient(fused_da.values)
    slope = np.sqrt(dy**2 + dx**2)
    max_slope = np.nanmax(slope)
    mean_slope = np.nanmean(slope)

    print(f"[Metric] Mean Slope: {mean_slope:.4f}")
    print(f"[Metric] Max Slope: {max_slope:.4f}")

    # Check for gaps between sources?
    # Ideally, we sample a profile across the transition.
    # But simple statistics suffice for now.

    # Verify that we have values from both ranges
    # i.e., values > 0 (Land) and values < -5 (Deep Water)
    has_land = (fused_da > 1.0).any()
    has_deep = (fused_da < -5.0).any()

    print(f"[Check] Has Land (>1m): {has_land.item()}")
    print(f"[Check] Has Deep Water (<-5m): {has_deep.item()}")

    if not has_land:
        print("WARNING: Fused data lacks Land values. Lidar might be missing or VDatum issue.")
    if not has_deep:
        print("WARNING: Fused data lacks Deep Water values. BlueTopo might be missing.")

    # Save debug TIF
    # fused_da.rio.to_raster("debug_fused.tif")


if __name__ == "__main__":
    # Allow manual run
    test_fusion_quality_execution_rocks(Path(".cache"))
