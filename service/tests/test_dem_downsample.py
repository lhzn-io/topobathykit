"""DEM viewer downsampling (#59): elevation by area average, source ids by nearest neighbor."""

import sys
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from topobathyserve.routers import dem_viewer


def _centers(n: int, n_out: int) -> np.ndarray:
    """Native-index position of each output cell's center."""
    f = n / n_out
    return (np.arange(n_out) + 0.5) * f - 0.5


def _footprint_means(n: int, n_out: int, k: int = 1000) -> np.ndarray:
    """Mean native index over each output footprint, supersampling each cell k times (independent check)."""
    fine = np.repeat(np.arange(n, dtype=np.float64), k)
    edges = np.round(np.linspace(0, n * k, n_out + 1)).astype(int)
    return np.array([fine[a:b].mean() for a, b in pairwise(edges)])


@pytest.mark.parametrize(
    ("shape", "out"), [((600, 600), (177, 177)), ((965, 1041), (285, 307)), ((40, 30), (13, 9))]
)
def test_each_cell_is_the_mean_of_its_footprint(shape: tuple[int, int], out: tuple[int, int]) -> None:
    """Each output cell is the mean of the native cells under its footprint, partial cells pro rata."""
    rows, cols = np.mgrid[0 : shape[0], 0 : shape[1]].astype(np.float64)
    plane = (-2000.0 + 3.0 * rows - 7.0 * cols).astype(np.float32)
    got = dem_viewer._downsample_mean(plane, *out)
    r, c = np.meshgrid(_footprint_means(shape[0], out[0]), _footprint_means(shape[1], out[1]), indexing="ij")
    np.testing.assert_allclose(got, -2000.0 + 3.0 * r - 7.0 * c, atol=0.02)
    assert got.dtype == np.float32


def test_a_non_integer_reduction_leaves_no_periodic_residual() -> None:
    """Nearest decimation of a smooth slope by 3.39 has a periodic error; the area mean has none."""
    n, n_out = 600, 177
    x = np.arange(n, dtype=np.float64)
    ramp = np.tile(np.tanh((x - 300.0) / 60.0) * 300.0, (8, 1)).astype(np.float32)
    exact = np.tanh((_centers(n, n_out) - 300.0) / 60.0) * 300.0
    near = dem_viewer._downsample_nearest(ramp, 8, n_out)[0]
    mean = dem_viewer._downsample_mean(ramp, 8, n_out)[0]
    assert np.abs(near - exact).max() > 1.0  # up to half a native cell of slope
    assert np.abs(mean - exact).max() < 0.2  # only the curvature within a footprint


def test_nan_cells_are_ignored_and_all_nan_footprints_stay_nan() -> None:
    a = np.full((12, 12), 5.0, dtype=np.float32)
    a[:, :3] = np.nan  # the first output column's footprint (cells 0..2) has no data
    a[6, 6] = np.nan
    got = dem_viewer._downsample_mean(a, 4, 4)
    assert np.isnan(got[:, 0]).all()
    np.testing.assert_allclose(got[:, 1:], 5.0)


def test_source_ids_are_never_averaged() -> None:
    ids = np.zeros((90, 90), dtype=np.uint32)
    ids[:, 45:] = 7
    got = dem_viewer._downsample(ids, 26, 26)
    assert got.dtype == np.uint32
    assert set(np.unique(got)) <= {0, 7}


def test_slice_and_downsample_uses_the_area_mean_for_elevation() -> None:
    rows, cols = np.mgrid[0:100, 0:100].astype(np.float32)
    plane = rows * 10.0 + cols
    meta = {"full_width": 100, "full_height": 100}
    grid = {"window": (0, 100, 0, 100), "width": 29, "height": 29}
    got = np.frombuffer(dem_viewer._slice_and_downsample(plane.tobytes(), np.float32, meta, grid), np.float32)
    r, c = np.meshgrid(_footprint_means(100, 29), _footprint_means(100, 29), indexing="ij")
    np.testing.assert_allclose(got.reshape(29, 29), r * 10.0 + c, atol=0.01)
