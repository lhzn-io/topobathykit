"""
Tests that fused cells computed by another fusion algorithm are never reused.

The fusion algorithm version is folded into the fused cache key, so a changed
algorithm writes to a new directory, and each cell records `fusion_algo_version`,
so a cell without it (or with another value) is rebuilt rather than served.
"""

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr
import yaml
from affine import Affine

import topobathykit.runtime as runtime
from topobathykit.policy.loader import hash_policy
from topobathykit.providers.base import Provider
from topobathykit.providers.registry import registry

HYDRATE_BBOX = (-73.49, 40.91, -73.46, 40.94)
CELL_BBOX = runtime._get_grid_cells(HYDRATE_BBOX, "EPSG:4326")[0][0]
RES_M = 500.0


class _Counting(Provider):
    calls = 0

    def fetch_layer(self, bbox: Any, resolution: Any = None, crs: str = "EPSG:4326", **kw: Any) -> Any:
        _Counting.calls += 1
        min_lon, min_lat, max_lon, max_lat = bbox
        n = 20
        res_x = (max_lon - min_lon) / n
        res_y = (max_lat - min_lat) / n
        da = xr.DataArray(
            np.full((n, n), 3.0, dtype=np.float32),
            coords={
                "y": np.linspace(max_lat - res_y / 2, min_lat + res_y / 2, n),
                "x": np.linspace(min_lon + res_x / 2, max_lon - res_x / 2, n),
            },
            dims=("y", "x"),
        )
        da.rio.write_crs(crs, inplace=True)
        da.rio.write_transform(
            Affine.translation(min_lon, max_lat) @ Affine.scale(res_x, -res_y), inplace=True
        )
        return da

    def get_metadata(self) -> dict[str, Any]:
        return {"name": "counting"}


class _DaemonProcess:
    """Reports daemon=True so hydrate() uses threads (see test_hydration_integrity)."""

    daemon = True

    def __init__(self, real: Any) -> None:
        self._real = real

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOPOBATHYKIT_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("TOPOBATHY_MAX_CELL_NAN_FRACTION", raising=False)
    registry.register("it_counting", _Counting)
    _Counting.calls = 0
    real = runtime.multiprocessing.current_process()
    monkeypatch.setattr(runtime.multiprocessing, "current_process", lambda: _DaemonProcess(real))


def _policy(tmp_path: Path) -> str:
    content = {
        "name": "CacheVersionTest",
        "crs": "EPSG:4326",
        "variables": [
            {"name": "elevation", "steps": [{"provider": "it_counting", "operation": "overwrite"}]}
        ],
    }
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.dump(content))
    return str(path)


def _rewrite_algo_attr(cache_path: Path, value: int | None) -> None:
    """Rewrite a cached cell as an older release would have left it."""
    ds = xr.open_dataset(cache_path, engine="zarr", decode_coords="all").load()
    if value is None:
        ds.attrs.pop("fusion_algo_version", None)
    else:
        ds.attrs["fusion_algo_version"] = value
    ds.to_zarr(cache_path, mode="w", consolidated=True)


def _algo_attr(cache_path: Path) -> Any:
    ds = xr.open_dataset(cache_path, engine="zarr")
    try:
        return ds.attrs.get("fusion_algo_version")
    finally:
        ds.close()


def test_cache_directory_and_key_change_with_the_algorithm_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy = _policy(tmp_path)
    path_now, key_now = runtime.get_fused_cache_info(policy, CELL_BBOX, RES_M)

    # The directory is not the bare policy hash that releases before the version used.
    content_hash = hash_policy(runtime._resolve_policy(policy).model_dump())
    assert path_now.parent.name != content_hash[:8]

    monkeypatch.setattr(runtime, "FUSION_ALGO_VERSION", runtime.FUSION_ALGO_VERSION + 1)
    path_next, key_next = runtime.get_fused_cache_info(policy, CELL_BBOX, RES_M)
    assert path_next.parent != path_now.parent
    assert key_next != key_now

    # The policy file is named after the directory, so cache tooling pairs them.
    policies = tmp_path / "cache" / "policies"
    assert list(policies.glob(f"*.{path_now.parent.name}.yaml"))
    assert list(policies.glob(f"*.{path_next.parent.name}.yaml"))


@pytest.mark.parametrize("stale", [None, 1])
def test_run_rebuilds_a_cell_from_another_algorithm(tmp_path: Path, stale: int | None) -> None:
    policy = _policy(tmp_path)
    cache_path, _ = runtime.get_fused_cache_info(policy, CELL_BBOX, RES_M)

    # HYDRATE_BBOX lies inside one standard cell, so each run touches only CELL_BBOX.
    runtime.run(policy, HYDRATE_BBOX, resolution=RES_M, use_cache=True)
    assert _Counting.calls == 1
    assert _algo_attr(cache_path) == runtime.FUSION_ALGO_VERSION

    runtime.run(policy, HYDRATE_BBOX, resolution=RES_M, use_cache=True)
    assert _Counting.calls == 1, "a current cell is reused"

    _rewrite_algo_attr(cache_path, stale)
    runtime.run(policy, HYDRATE_BBOX, resolution=RES_M, use_cache=True)
    assert _Counting.calls == 2, "a cell without the current fusion_algo_version must be rebuilt"
    assert _algo_attr(cache_path) == runtime.FUSION_ALGO_VERSION


def test_run_fast_path_rebuilds_a_cell_from_another_algorithm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The fast path serves a bbox that is exactly one standard cell. _get_grid_cells
    # also returns the neighbours sharing its far edges, so pin it to the one cell.
    real = runtime._get_grid_cells
    monkeypatch.setattr(runtime, "_get_grid_cells", lambda bbox, crs: ([CELL_BBOX], real(bbox, crs)[1]))
    policy = _policy(tmp_path)
    cache_path, _ = runtime.get_fused_cache_info(policy, CELL_BBOX, RES_M)

    runtime.run(policy, CELL_BBOX, resolution=RES_M, use_cache=True)
    runtime.run(policy, CELL_BBOX, resolution=RES_M, use_cache=True)
    assert _Counting.calls == 1

    _rewrite_algo_attr(cache_path, None)
    ds = runtime.run(policy, CELL_BBOX, resolution=RES_M, use_cache=True)
    assert _Counting.calls == 2
    assert ds.attrs["fusion_algo_version"] == runtime.FUSION_ALGO_VERSION


def test_hydrate_rebuilds_a_cell_from_another_algorithm(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    cache_path, _ = runtime.get_fused_cache_info(policy, CELL_BBOX, RES_M)

    assert runtime.hydrate(policy, HYDRATE_BBOX, resolution=RES_M, max_workers=1)["processed"] == 1
    assert runtime.hydrate(policy, HYDRATE_BBOX, resolution=RES_M, max_workers=1)["cached"] == 1

    _rewrite_algo_attr(cache_path, None)
    assert runtime.hydrate(policy, HYDRATE_BBOX, resolution=RES_M, max_workers=1)["processed"] == 1
    assert _algo_attr(cache_path) == runtime.FUSION_ALGO_VERSION


def test_hydrate_and_run_write_the_same_policy_hash(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    cache_path, key_hash = runtime.get_fused_cache_info(policy, CELL_BBOX, RES_M)
    content_hash = hash_policy(runtime._resolve_policy(policy).model_dump())

    runtime.hydrate(policy, HYDRATE_BBOX, resolution=RES_M, max_workers=1)
    with xr.open_dataset(cache_path, engine="zarr") as ds:
        hydrated = dict(ds.attrs)
    shutil.rmtree(cache_path)
    runtime.run(policy, HYDRATE_BBOX, resolution=RES_M, use_cache=True)
    with xr.open_dataset(cache_path, engine="zarr") as ds:
        written_by_run = dict(ds.attrs)

    for attrs in (hydrated, written_by_run):
        assert attrs["policy_hash"] == content_hash
        assert attrs["cache_key"] == key_hash
        assert attrs["fusion_algo_version"] == runtime.FUSION_ALGO_VERSION
