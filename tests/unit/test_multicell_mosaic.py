"""
Unit tests for the multi-cell mosaic path in runtime.run().

Background
----------
run() splits arbitrary bboxes into 0.05° grid cells, processes each cell
independently via _run_cell(), then mosaics them with combine_by_coords.
Because each cell is computed on its own pixel grid, adjacent cells can have
y-coordinates that are slightly offset from each other (~0.1 m at z=17).
combine_by_coords unions both coordinate arrays rather than aligning them,
producing interleaved valid/NaN rows.  Without the interpolate_na fix, a
downstream linear interp in the tile renderer propagates those NaN values
to 100% of the output pixels (blank tile).

These tests exercise that path directly by mocking _run_cell to return
synthetic cell datasets with controlled pixel-grid offsets, then verifying
that run() returns elevation data with no unexpected NaN.
"""

from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr
import yaml

from topobathykit.runtime import run

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cell_ds(
    x_start: float,
    x_end: float,
    y_vals: np.ndarray,
    elev_val: float = -5.0,
    crs: str = "EPSG:4326",
) -> xr.Dataset:
    """Build a minimal Dataset mimicking _run_cell() output.

    Args:
        x_start: West edge of the x coordinate range.
        x_end: East edge of the x coordinate range.
        y_vals: Explicit y coordinate array (allows injecting grid offsets).
        elev_val: Scalar elevation to fill the array with.
        crs: CRS string written via rioxarray.
    """
    xs = np.linspace(x_start, x_end, 20)
    data = np.full((len(y_vals), len(xs)), elev_val, dtype=np.float64)
    elev = xr.DataArray(data, coords={"y": y_vals, "x": xs}, dims=("y", "x"))
    elev.rio.write_crs(crs, inplace=True)
    src = xr.zeros_like(elev).astype("int32")
    ds = xr.Dataset({"elevation": elev, "source_elevation": src})
    ds.attrs = {
        "provenance_dict": {},
        "grid_cell_size": 0.05,
        "policy_legend": "{}",
        "crs": crs,
    }
    return ds


@pytest.fixture
def simple_policy_file(tmp_path: Any) -> str:
    """Minimal policy YAML with no real providers (cells are mocked)."""
    content = {
        "name": "TestMosaicPolicy",
        "crs": "EPSG:4326",
        "variables": [
            {
                "name": "elevation",
                "steps": [],
            }
        ],
    }
    p = tmp_path / "mosaic_policy.yaml"
    p.write_text(yaml.dump(content))
    return str(p)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_offset_y_grids_produce_no_nan_after_mosaic(simple_policy_file: str) -> None:
    """
    Regression test for the blank-tile bug at grid cell boundaries.

    Two adjacent cells are returned with y-pixel grids that are offset by
    half a step (0.001°), simulating the real-world misalignment between
    independently-computed zarr caches.  After combine_by_coords the merged
    dataset has ~50% NaN in an alternating row pattern.  The fix
    (interpolate_na) must eliminate those stripes so the final elevation
    array has no NaN.

    Bbox (-0.01, 0.0, 0.04, 0.05) spans the 0.0° cell boundary → two cells.
    """
    step = 0.002  # 0.002° spacing = 25 rows per cell

    # Cell A y-grid: [0.000, 0.002, 0.004, ..., 0.048]
    y_a = np.arange(0.0, 0.05, step)
    # Cell B y-grid: same range but offset by half a step
    y_b = y_a + step / 2

    cell_a = _cell_ds(-0.05, 0.0, y_a, elev_val=-3.0)
    cell_b = _cell_ds(0.0, 0.05, y_b, elev_val=-3.0)

    call_count = {"n": 0}
    cells_to_return = [cell_a, cell_b]

    def mock_run_cell(**kwargs: Any) -> xr.Dataset:
        ds = cells_to_return[call_count["n"]]
        call_count["n"] += 1
        return ds

    # Bbox spans the 0.0° x-boundary but stays well inside one y-band (0.0-0.05°)
    # to avoid also crossing the y-boundary and generating 4 cells instead of 2.
    with patch("topobathykit.runtime._run_cell", side_effect=mock_run_cell):
        ds = run(
            simple_policy_file,
            bbox=(-0.01, 0.001, 0.04, 0.049),
            resolution=1000.0,
            use_cache=False,
        )

    elev = ds["elevation"]
    nan_count = int(np.isnan(elev).sum())
    nan_pct = nan_count / elev.size

    # A handful of edge pixels (outermost rows of the combined y-range) may remain
    # NaN because interpolate_na requires a valid neighbor on both sides.  Those
    # are stripped by tile pad_px trimming downstream.  The critical check is that
    # the ~50% NaN stripe pattern produced by combine_by_coords is eliminated.
    assert nan_pct < 0.02, (
        f"Expected <2% NaN after mosaic NaN-fill, got {nan_pct * 100:.1f}% "
        f"({nan_count}/{elev.size} pixels). "
        "The ~50% NaN stripe pattern from offset y-grids was not eliminated."
    )
    # Non-NaN values should be close to the fill value (not extrapolated wildly)
    valid = elev.values[~np.isnan(elev.values)]
    assert np.mean(valid) == pytest.approx(-3.0, abs=0.1)


def test_single_cell_bbox_bypasses_nan_fill(simple_policy_file: str) -> None:
    """
    A bbox that falls entirely within one 0.05° cell never enters the mosaic
    branch.  The interpolate_na fix must not alter single-cell output.
    """
    y_vals = np.linspace(0.001, 0.049, 25)
    cell = _cell_ds(0.001, 0.049, y_vals, elev_val=7.5)

    with patch("topobathykit.runtime._run_cell", return_value=cell):
        ds = run(
            simple_policy_file,
            bbox=(0.001, 0.001, 0.049, 0.049),
            resolution=1000.0,
            use_cache=False,
        )

    elev = ds["elevation"]
    valid = elev.values[~np.isnan(elev.values)]
    assert len(valid) > 0, "Single-cell result should contain valid pixels"
    assert np.allclose(valid, 7.5), "Single-cell values should be unchanged"


def test_genuine_edge_nan_not_filled_by_mosaic(simple_policy_file: str) -> None:
    """
    If one cell has genuinely no elevation data (all NaN), those pixels must
    remain NaN after mosaicking.  interpolate_na only fills interior NaN
    (values with valid neighbors on both sides); edge NaN with no valid
    neighbor in that dimension must be preserved.

    Cell A (x < 0): valid data at -3.0 m.
    Cell B (x > 0): all NaN — no provider covered this area.
    """
    step = 0.002
    y_shared = np.arange(0.0, 0.05, step)

    cell_a = _cell_ds(-0.05, 0.0, y_shared, elev_val=-3.0)

    # Cell B: elevation is all NaN
    xs_b = np.linspace(0.0, 0.05, 20)
    nan_data = np.full((len(y_shared), len(xs_b)), np.nan)
    elev_nan = xr.DataArray(nan_data, coords={"y": y_shared, "x": xs_b}, dims=("y", "x"))
    elev_nan.rio.write_crs("EPSG:4326", inplace=True)
    cell_b = xr.Dataset({"elevation": elev_nan, "source_elevation": xr.zeros_like(elev_nan).astype("int32")})
    cell_b.attrs = {"provenance_dict": {}, "grid_cell_size": 0.05, "policy_legend": "{}", "crs": "EPSG:4326"}

    call_count = {"n": 0}
    cells_to_return = [cell_a, cell_b]

    def mock_run_cell(**kwargs: Any) -> xr.Dataset:
        ds = cells_to_return[call_count["n"]]
        call_count["n"] += 1
        return ds

    with patch("topobathykit.runtime._run_cell", side_effect=mock_run_cell):
        ds = run(
            simple_policy_file,
            bbox=(-0.01, 0.001, 0.04, 0.049),
            resolution=1000.0,
            use_cache=False,
        )

    elev = ds["elevation"]
    # Pixels from cell A (x < 0) should be valid
    valid_side = elev.sel(x=elev.x[elev.x < -0.001])
    assert not bool(valid_side.isnull().all()), "Cell A pixels should remain valid"

    # Pixels from cell B (x > 0.001) should still be NaN — edge NaN not filled
    empty_side = elev.sel(x=elev.x[elev.x > 0.001])
    if empty_side.size > 0:
        assert bool(empty_side.isnull().all()), (
            "Genuine all-NaN cell (no data coverage) should not be filled by interpolate_na. "
            "Edge NaN values with no valid neighbor must be preserved."
        )
