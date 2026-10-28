"""
Tests that a multi-tile, multi-project NOAA topobathy fusion is deterministic.

Background: `resolve_tiles_in_bbox` returned `list(set(...))`, whose order follows
the per-process string hash seed, and tiles were merged with `merge_arrays`, which
takes its output grid from the first input. The same request could therefore give
different pixels from one process to the next. Tiles are now resolved in sorted
order, resampled onto one explicit grid and merged by priority.
"""

import os
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
import yaml
from affine import Affine
from shapely.geometry import box

BBOX = (-73.52, 40.88, -73.43, 40.97)
RES_M = 30.0

# Project 111: three tiles side by side, each on its own grid (different pixel size
# and origin), overlapping by a few pixels at their seams. Project 222: one coarser
# tile under the whole bbox, which fills a gap left in project 111.
TILES_111: dict[str, tuple[tuple[float, float, float, float], float]] = {
    "a.tif": ((-73.525, 40.875, -73.488, 40.975), 0.000090),
    "b.tif": ((-73.490, 40.875, -73.458, 40.975), 0.000110),
    "c.tif": ((-73.460, 40.875, -73.425, 40.975), 0.000100),
}
TILE_222 = ((-73.53, 40.87, -73.42, 40.98), 0.00030)
GAP = (-73.47, 40.90, -73.45, 40.92)  # project 111 has no data here


def _field(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    out: np.ndarray = 100.0 * (lon + 73.5) - 80.0 * (lat - 40.9) + np.sin(300.0 * lon) * np.cos(200.0 * lat)
    return out


def _tile(bounds: tuple[float, float, float, float], res: float, offset: float, gap: bool) -> xr.DataArray:
    west, south, east, north = bounds
    nx = int(round((east - west) / res))
    ny = int(round((north - south) / res))
    xs = west + (np.arange(nx) + 0.5) * res
    ys = north - (np.arange(ny) + 0.5) * res
    lon, lat = np.meshgrid(xs, ys)
    z = (_field(lon, lat) + offset).astype(np.float32)
    if gap:
        inside = (lon > GAP[0]) & (lon < GAP[2]) & (lat > GAP[1]) & (lat < GAP[3])
        z[inside] = np.nan
    da = xr.DataArray(z, coords={"y": ys, "x": xs}, dims=("y", "x"))
    da.rio.write_crs("EPSG:4326", inplace=True)
    da.rio.write_transform(Affine.translation(west, north) @ Affine.scale(res, -res), inplace=True)
    return da


def _tile_index(names: list[str], bounds: dict[str, tuple[float, float, float, float]]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"name": names}, geometry=[box(*bounds[n]) for n in names], crs="EPSG:4326")


def _fresh_provider(cache_dir: Path, order_111: list[str]) -> Any:
    from topobathykit.providers.noaa_topobathy import NoaaTopobathyProvider

    NoaaTopobathyProvider._singleton = None
    NoaaTopobathyProvider._cls_spatial_index = None
    NoaaTopobathyProvider._cls_projects.clear()
    NoaaTopobathyProvider._cls_projects_metadata_urls.clear()
    NoaaTopobathyProvider._cls_tile_indices.clear()
    provider = NoaaTopobathyProvider(cache_dir=str(cache_dir))
    provider._projects.update({"111": "Project_111", "222": "Project_222"})
    bounds_111 = {name: b for name, (b, _) in TILES_111.items()}
    NoaaTopobathyProvider._cls_tile_indices["111"] = _tile_index(order_111, bounds_111)
    NoaaTopobathyProvider._cls_tile_indices["222"] = _tile_index(["z.tif"], {"z.tif": TILE_222[0]})
    return provider


def _fetch_tile(name: str, bbox: Any = None, project_id: Any = None, **_kw: Any) -> xr.DataArray:
    if project_id == "222":
        return _tile(TILE_222[0], TILE_222[1], offset=-5.0, gap=False)
    bounds, res = TILES_111[name]
    return _tile(bounds, res, offset=0.0, gap=True)


def _patched(provider: Any) -> ExitStack:
    stack = ExitStack()
    stack.enter_context(patch.object(provider, "find_projects_by_box", return_value=["111", "222"]))
    stack.enter_context(patch.object(provider, "fetch_tile", side_effect=_fetch_tile))
    return stack


def fuse_once(cache_dir: Path, order_111: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Fuse BBOX through the provider and return (elevation, source_id) arrays."""
    provider = _fresh_provider(cache_dir, order_111)
    with _patched(provider):
        ds = provider.fetch_layer(BBOX, resolution=RES_M)
    return np.asarray(ds["elevation"].values), np.asarray(ds["source_id"].values)


def fuse_cell_once(tmp_path: Path, order_111: list[str]) -> xr.Dataset:
    """Run one runtime cell with the patched provider as its only step."""
    import topobathykit.runtime as runtime

    provider = _fresh_provider(tmp_path / "tb", order_111)
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(
        yaml.dump(
            {
                "name": "DeterminismTest",
                "crs": "EPSG:3857",
                "variables": [
                    {"name": "elevation", "steps": [{"provider": "noaa_topobathy", "operation": "overwrite"}]}
                ],
            }
        )
    )
    policy = runtime._resolve_policy(str(policy_path))
    cell = runtime._get_grid_cells((-73.49, 40.91, -73.46, 40.94), "EPSG:3857")[0][0]
    with _patched(provider):
        return runtime._run_cell(policy, cell, RES_M, None, "EPSG:3857")


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOPOBATHYKIT_CACHE_DIR", str(tmp_path / "cache"))


def test_resolve_tiles_in_bbox_is_sorted(tmp_path: Path) -> None:
    provider = _fresh_provider(tmp_path, ["c.tif", "a.tif", "b.tif"])
    assert provider.resolve_tiles_in_bbox(*BBOX, project_id="111") == ["a.tif", "b.tif", "c.tif"]


def test_project_priority_and_gap_fill(tmp_path: Path) -> None:
    elev, src = fuse_once(tmp_path, list(TILES_111))
    uids = set(np.unique(src)) - {0}
    assert uids == {1_000_111, 1_000_222}
    assert np.isfinite(elev).mean() > 0.99
    # Outside the gap project 111 wins; inside it, project 222 (offset -5 m) fills.
    assert (src == 1_000_111).mean() > 0.9
    assert (src == 1_000_222).sum() > 0
    # source_id follows the elevation it sits beside.
    assert np.isfinite(elev[src > 0]).all()


def test_shuffled_tile_order_gives_identical_fusion(tmp_path: Path) -> None:
    ref_elev, ref_src = fuse_once(tmp_path / "1", ["a.tif", "b.tif", "c.tif"])
    for order in (["c.tif", "a.tif", "b.tif"], ["b.tif", "c.tif", "a.tif"]):
        elev, src = fuse_once(tmp_path / "".join(order), order)
        np.testing.assert_array_equal(elev, ref_elev)
        np.testing.assert_array_equal(src, ref_src)


def test_shuffled_tile_order_gives_identical_runtime_cell(tmp_path: Path) -> None:
    ref = fuse_cell_once(tmp_path / "1", ["a.tif", "b.tif", "c.tif"])
    other = fuse_cell_once(tmp_path / "2", ["c.tif", "b.tif", "a.tif"])
    assert np.isfinite(ref["elevation"].values).mean() > 0.99
    np.testing.assert_array_equal(other["elevation"].values, ref["elevation"].values)
    np.testing.assert_array_equal(other["source_elevation"].values, ref["source_elevation"].values)


_SUBPROCESS_SCRIPT = """
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, {test_dir!r})
import test_merge_determinism as t

out = Path(sys.argv[1])
elev, src = t.fuse_once(out.parent / "cache", ["a.tif", "b.tif", "c.tif"])
np.savez(out, elev=elev, src=src)
"""


def test_fusion_is_identical_across_hash_seeds(tmp_path: Path) -> None:
    script = tmp_path / "fuse.py"
    script.write_text(_SUBPROCESS_SCRIPT.format(test_dir=str(Path(__file__).parent)))
    repo = Path(__file__).resolve().parents[2]
    results = []
    for seed in ("1", "2", "3"):
        run_dir = tmp_path / f"seed{seed}"
        run_dir.mkdir()
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = seed
        env["TOPOBATHYKIT_CACHE_DIR"] = str(run_dir / "cache")
        env["PYTHONPATH"] = os.pathsep.join(
            [str(repo / "src"), str(repo / "service"), env.get("PYTHONPATH", "")]
        )
        out = run_dir / "out.npz"
        proc = subprocess.run(
            [sys.executable, str(script), str(out)], env=env, capture_output=True, text=True, timeout=300
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
        with np.load(out) as data:
            results.append((data["elev"].copy(), data["src"].copy()))
    for elev, src in results[1:]:
        np.testing.assert_array_equal(elev, results[0][0])
        np.testing.assert_array_equal(src, results[0][1])
    # Sanity: the arrays are not trivially empty.
    assert np.isfinite(results[0][0]).sum() > 0
