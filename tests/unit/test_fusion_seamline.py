import numpy as np
import xarray as xr

from topobathykit.fusion import FusionEngine


def test_fuse_seamline_gap_fill() -> None:
    """
    Test that fuse_seamline fills a gap between Lidar and Bathy
    and creates a smooth transition.
    """
    # 1. Setup Synthetic Data (1D-like strip)
    # Lidar: Valid on left (x=0 to 4), NaN on right
    # Bathy: Valid on right (x=6 to 10), NaN on left
    # Gap: x=5 is NaN in both initially (or we treat Bathy as background)

    # Actually, USGS logic assumes we fuse Lidar (Priority) onto Bathy (Background).
    # IF Bathy is also NaN in the gap, we just extrapolate Lidar proxy.
    # IF Bathy has data in the gap, we blend.

    # Case A: Bathy covers the whole domain (Background), Lidar covers left half.
    x = np.linspace(0, 100, 101)  # 0 to 100 meters
    y = np.linspace(0, 10, 11)

    lidar_vals = np.full((11, 101), np.nan)
    lidar_vals[:, :50] = 10.0  # Lidar is flat 10m until x=50

    bathy_vals = np.full((11, 101), -10.0)  # Bathy is flat -10m everywhere

    coords = {"y": y, "x": x}
    lidar_da = xr.DataArray(lidar_vals, coords=coords, dims=("y", "x"))
    bathy_da = xr.DataArray(bathy_vals, coords=coords, dims=("y", "x"))

    # Add rio accessor mock info if needed for resolution check
    # But code defaults to using resolution from coords if Rio not present?
    # The code calculates dist_m = dist_px * res.
    # If Rio missing, code checks: if hasattr(bathy_da, "rio")... else res=1.0.
    # Our dx=1.0 here.

    engine = FusionEngine()

    # 2. Run Fusion with blend_dist = 20m
    # Seam is at x=49 (last valid pixel). Dist starts counting at x=50.
    # At x=50, dist=1. w = 1 - 1/20 = 0.95. Fused = 0.95*10 + 0.05*-10 = 9.5 - 0.5 = 9.0
    # At x=69, dist=20. w = 0. Fused = -10.

    fused_da = engine.fuse_seamline(lidar_da, bathy_da, blend_dist=20.0)

    # 3. Assertions
    res = fused_da.isel(y=5)  # take middle profile

    # Check Lidar area (x < 50) - should be exactly 10.0
    assert np.allclose(res.sel(x=slice(0, 49)), 10.0), "Lidar area modified!"

    # Check Transition Area (x=50 to ~70)
    # Value at x=50 should be blended (~9.0)
    val_50 = res.sel(x=50).item()
    assert 8.0 < val_50 < 10.0, f"Expected blend near 9.0, got {val_50}"

    # Value at x=60 (mid blend, dist=11). w=0.45. Fused = 0.45*10 + 0.55*-10 = -1.0.
    val_60 = res.sel(x=60).item()
    assert -1.1 <= val_60 <= -0.9, f"Expected blend near -1.0, got {val_60}"

    # Check Bathy area (x > 75) - should be exactly -10.0
    assert np.allclose(res.sel(x=slice(75, 100)), -10.0), "Far field bathy modified!"


def test_fuse_seamline_extrapolation() -> None:
    """
    Test case where Bathy is MISSING in the transition zone.
    The logic should pure-extrapolate Lidar Proxy values.
    """
    x = np.arange(20)
    y = np.arange(5)

    lidar_vals = np.full((5, 20), np.nan)
    lidar_vals[:, :5] = 10.0  # Lidar x=0..4

    bathy_vals = np.full((5, 20), np.nan)
    bathy_vals[:, 15:] = -10.0  # Bathy x=15..19
    # Gap x=5..14 is NaN in BOTH.

    coords = {"y": y, "x": x}
    lidar_da = xr.DataArray(lidar_vals, coords=coords, dims=("y", "x"))
    bathy_da = xr.DataArray(bathy_vals, coords=coords, dims=("y", "x"))

    engine = FusionEngine()

    # Blend dist 10px. Lidar edge at x=4.
    # Weights > 0 until x=14.
    fused_da = engine.fuse_seamline(lidar_da, bathy_da, blend_dist=10.0)

    # Check gap (x=5..13)
    # Since Bathy is NaN, it should extrapolate Lidar (10.0)
    gap_vals = fused_da.isel(y=2).sel(x=slice(5, 13))
    assert np.allclose(gap_vals, 10.0), "Gap should be filled with Lidar Proxy (10.0) when Bathy is NaN"

    # At x=15, Bathy starts. w=0 (dist=11 > 10). Should be -10.0
    assert fused_da.isel(y=2, x=15).item() == -10.0
