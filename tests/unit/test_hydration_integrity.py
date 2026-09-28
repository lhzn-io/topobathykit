"""
Tests for the per-step completeness gate that keeps incomplete cells out of the cache.

Background: in May 2026, five WLIS cells were cached with the noaa_topobathy step
silently missing (a transient failure swallowed as "no data"), and every later
hydrate reused them. A cell is now cached only when no step failed and its NaN
fraction is within the configured ceiling.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr
import yaml
from affine import Affine

import topobathysim.runtime as runtime
from topobathysim.providers.base import Provider, ProviderFetchError, ProviderNoDataError
from topobathysim.providers.registry import registry

# A bbox strictly inside one standard 0.05 degree grid cell, and that cell. At 500 m
# the cell canvas is about 13 x 13 pixels.
HYDRATE_BBOX = (-73.49, 40.91, -73.46, 40.94)
CELL_BBOX = runtime._get_grid_cells(HYDRATE_BBOX, "EPSG:4326")[0][0]
RES_M = 500.0


def _grid(bbox: tuple[float, float, float, float], value: float, crs: str) -> xr.DataArray:
    min_lon, min_lat, max_lon, max_lat = bbox
    n = 20
    res_x = (max_lon - min_lon) / n
    res_y = (max_lat - min_lat) / n
    da = xr.DataArray(
        np.full((n, n), value, dtype=np.float32),
        coords={
            "y": np.linspace(max_lat - res_y / 2, min_lat + res_y / 2, n),
            "x": np.linspace(min_lon + res_x / 2, max_lon - res_x / 2, n),
        },
        dims=("y", "x"),
    )
    da.rio.write_crs(crs, inplace=True)
    da.rio.write_transform(Affine.translation(min_lon, max_lat) * Affine.scale(res_x, -res_y), inplace=True)
    return da


class _FullCover(Provider):
    """Covers the whole request with a constant elevation."""

    def fetch_layer(self, bbox: Any, resolution: Any = None, crs: str = "EPSG:4326", **kw: Any) -> Any:
        return _grid(bbox, 5.0, crs)

    def get_metadata(self) -> dict[str, Any]:
        return {"name": "full"}


class _WestHalf(Provider):
    """Covers only the western half of the request, leaving the rest NaN."""

    def fetch_layer(self, bbox: Any, resolution: Any = None, crs: str = "EPSG:4326", **kw: Any) -> Any:
        da = _grid(bbox, 5.0, crs)
        mid = (bbox[0] + bbox[2]) / 2
        return da.where(da.x < mid)

    def get_metadata(self) -> dict[str, Any]:
        return {"name": "west"}


class _NoCoverage(Provider):
    def fetch_layer(self, *a: Any, **kw: Any) -> Any:
        raise ProviderNoDataError("no coverage here")

    def get_metadata(self) -> dict[str, Any]:
        return {"name": "nodata"}


class _Flaky(Provider):
    """Raises a transient error until `healthy` is set, then covers the request."""

    healthy = False

    def fetch_layer(self, bbox: Any, resolution: Any = None, crs: str = "EPSG:4326", **kw: Any) -> Any:
        if not _Flaky.healthy:
            raise ProviderFetchError("simulated VDatum read timeout")
        return _grid(bbox, 1.0, crs)

    def get_metadata(self) -> dict[str, Any]:
        return {"name": "flaky"}


class _OutOfMemory(Provider):
    def fetch_layer(self, *a: Any, **kw: Any) -> Any:
        raise MemoryError("simulated RLIMIT_AS exhaustion")

    def get_metadata(self) -> dict[str, Any]:
        return {"name": "oom"}


class _Partial(Provider):
    """Returns data but reports that some tiles were lost to errors."""

    def fetch_layer(self, bbox: Any, resolution: Any = None, crs: str = "EPSG:4326", **kw: Any) -> Any:
        ds = xr.Dataset({"elevation": _grid(bbox, 2.0, crs)})
        ds.rio.write_crs(crs, inplace=True)
        ds.attrs["fetch_errors"] = 2
        return ds

    def get_metadata(self) -> dict[str, Any]:
        return {"name": "partial"}


_PROVIDERS: dict[str, type[Provider]] = {
    "it_full": _FullCover,
    "it_west": _WestHalf,
    "it_nodata": _NoCoverage,
    "it_flaky": _Flaky,
    "it_oom": _OutOfMemory,
    "it_partial": _Partial,
}


class _DaemonProcess:
    """A view of the current process that reports daemon=True and delegates the rest."""

    daemon = True

    def __init__(self, real: Any) -> None:
        self._real = real

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOPOBATHYSIM_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("TOPOBATHY_MAX_CELL_NAN_FRACTION", raising=False)
    for name, cls in _PROVIDERS.items():
        registry.register(name, cls)
    _Flaky.healthy = False
    # hydrate() takes the thread-pool path inside a daemon process. Outside one it
    # uses ProcessPoolExecutor(fork), whose nested worker cannot be pickled.
    # runtime.multiprocessing is the shared module, so the stand-in must keep every
    # other attribute of the real process: numcodecs' Blosc (zarr 2) reads .pid.
    real = runtime.multiprocessing.current_process()
    monkeypatch.setattr(runtime.multiprocessing, "current_process", lambda: _DaemonProcess(real))


def _policy(tmp_path: Path, steps: list[str]) -> str:
    content = {
        "name": "IntegrityTest",
        "crs": "EPSG:4326",
        "variables": [
            {"name": "elevation", "steps": [{"provider": s, "operation": "overwrite"} for s in steps]}
        ],
    }
    path = tmp_path / f"policy_{'_'.join(steps)}.yaml"
    path.write_text(yaml.dump(content))
    return str(path)


def _outcomes(ds: xr.Dataset) -> dict[str, dict[str, Any]]:
    return {o["provider"]: o for o in json.loads(ds.attrs["step_outcomes_json"])}


def _cache_path(policy: str) -> Path:
    path, _ = runtime.get_fused_cache_info(policy, CELL_BBOX, resolution=RES_M)
    return path


def test_run_cell_records_step_outcomes(tmp_path: Path) -> None:
    policy = runtime._resolve_policy(_policy(tmp_path, ["it_full", "it_nodata", "it_flaky", "it_partial"]))
    ds = runtime._run_cell(policy, CELL_BBOX, RES_M, None, "EPSG:4326")

    outcomes = _outcomes(ds)
    assert outcomes["it_full"]["status"] == "ok"
    assert outcomes["it_full"]["pixels"] > 0
    assert outcomes["it_nodata"]["status"] == "nodata"
    assert outcomes["it_flaky"]["status"] == "error"
    assert "ProviderFetchError" in outcomes["it_flaky"]["error"]
    assert outcomes["it_partial"]["status"] == "partial"
    assert outcomes["it_partial"]["pixels"] > 0


def test_hydrate_does_not_cache_cell_with_failed_step_and_retries_it(tmp_path: Path) -> None:
    policy = _policy(tmp_path, ["it_full", "it_flaky"])
    progress: list[dict[str, int]] = []

    stats = runtime.hydrate(
        policy, HYDRATE_BBOX, resolution=RES_M, max_workers=1, on_progress=progress.append
    )

    assert stats == {"total": 1, "cached": 0, "processed": 0, "failed": 1}
    assert progress[-1]["failed"] == 1, "failed cells must reach the job ledger via on_progress"
    assert not _cache_path(policy).exists(), "a cell with a failed step must not be cached"

    # The transient failure clears; the next hydrate rebuilds the cell instead of reusing it.
    _Flaky.healthy = True
    stats = runtime.hydrate(policy, HYDRATE_BBOX, resolution=RES_M, max_workers=1)

    assert stats == {"total": 1, "cached": 0, "processed": 1, "failed": 0}
    cached = xr.open_dataset(_cache_path(policy), engine="zarr")
    try:
        assert {o["status"] for o in _outcomes(cached).values()} == {"ok"}
        assert cached.attrs["nan_fraction"] == 0.0
    finally:
        cached.close()


def test_hydrate_caches_cell_when_a_step_has_no_coverage(tmp_path: Path) -> None:
    policy = _policy(tmp_path, ["it_full", "it_nodata"])

    stats = runtime.hydrate(policy, HYDRATE_BBOX, resolution=RES_M, max_workers=1)

    assert stats["processed"] == 1
    assert stats["failed"] == 0
    assert _cache_path(policy).exists()


@pytest.mark.parametrize("failing", ["it_oom", "it_partial"])
def test_hydrate_does_not_cache_memory_error_or_partial_step(tmp_path: Path, failing: str) -> None:
    policy = _policy(tmp_path, ["it_full", failing])

    stats = runtime.hydrate(policy, HYDRATE_BBOX, resolution=RES_M, max_workers=1)

    assert stats["failed"] == 1
    assert not _cache_path(policy).exists()


def test_hydrate_nan_ceiling_is_configurable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    policy = _policy(tmp_path, ["it_west"])

    stats = runtime.hydrate(policy, HYDRATE_BBOX, resolution=RES_M, max_workers=1)
    assert stats["failed"] == 1, "a half-NaN cell exceeds the default 1% ceiling"
    assert not _cache_path(policy).exists()

    monkeypatch.setenv("TOPOBATHY_MAX_CELL_NAN_FRACTION", "0.9")
    stats = runtime.hydrate(policy, HYDRATE_BBOX, resolution=RES_M, max_workers=1)
    assert stats["processed"] == 1
    assert _cache_path(policy).exists()


def test_run_serves_but_does_not_cache_cell_with_failed_step(tmp_path: Path) -> None:
    policy = _policy(tmp_path, ["it_full", "it_flaky"])

    ds = runtime.run(policy, CELL_BBOX, resolution=RES_M, use_cache=True)

    assert int(ds["elevation"].notnull().sum()) > 0
    assert not _cache_path(policy).exists()

    # A partially covered cell with no failed step is still cached by run(); the NaN
    # ceiling applies to hydrate only.
    policy_west = _policy(tmp_path, ["it_west"])
    runtime.run(policy_west, CELL_BBOX, resolution=RES_M, use_cache=True)
    assert _cache_path(policy_west).exists()


# ---------------------------------------------------------------------------
# noaa_topobathy: failed tiles and projects must not be reported as "no data"
# ---------------------------------------------------------------------------

TB_BBOX = (-73.52, 40.88, -73.43, 40.97)


@pytest.fixture
def topobathy(tmp_path: Path) -> Any:
    from topobathysim.providers.noaa_topobathy import NoaaTopobathyProvider

    NoaaTopobathyProvider._singleton = None
    NoaaTopobathyProvider._cls_spatial_index = None
    NoaaTopobathyProvider._cls_projects.clear()
    NoaaTopobathyProvider._cls_projects_metadata_urls.clear()
    NoaaTopobathyProvider._cls_tile_indices.clear()
    provider = NoaaTopobathyProvider(cache_dir=str(tmp_path / "tb"))
    provider._projects.update({"111": "Project_111", "222": "Project_222"})
    return provider


def _patched_topobathy(provider: Any, fetch_tile: Any) -> Any:
    from contextlib import ExitStack
    from unittest.mock import patch

    stack = ExitStack()
    stack.enter_context(patch.object(provider, "find_projects_by_box", return_value=["111", "222"]))
    stack.enter_context(patch.object(provider, "_load_tile_index", return_value=None))
    stack.enter_context(patch.object(provider, "resolve_tiles_in_bbox", return_value=["t.tif"]))
    stack.enter_context(patch.object(provider, "fetch_tile", side_effect=fetch_tile))
    return stack


def test_topobathy_all_projects_failing_raises_fetch_error(topobathy: Any) -> None:
    def failing(*_a: Any, **_kw: Any) -> None:
        raise ProviderFetchError("simulated MemoryError while merging")

    with _patched_topobathy(topobathy, failing), pytest.raises(ProviderFetchError, match="2 error"):
        topobathy.fetch_layer(TB_BBOX)


def test_topobathy_tiles_without_data_is_no_data(topobathy: Any) -> None:
    with _patched_topobathy(topobathy, lambda *_a, **_kw: None), pytest.raises(ProviderNoDataError):
        topobathy.fetch_layer(TB_BBOX)


def test_topobathy_one_failing_project_is_partial(topobathy: Any) -> None:
    def per_project(_tile: str, bbox: Any = None, project_id: Any = None, **_kw: Any) -> Any:
        if project_id == "222":
            raise ProviderFetchError("simulated S3 read timeout")
        return _grid(TB_BBOX, -1.0, "EPSG:4326")

    with _patched_topobathy(topobathy, per_project):
        ds = topobathy.fetch_layer(TB_BBOX)

    assert ds.attrs["fetch_errors"] == 1
    assert int(ds["elevation"].notnull().sum()) > 0


def test_topobathy_fetch_tile_raise_errors(topobathy: Any) -> None:
    """fetch_tile keeps returning None by default; with raise_errors it raises on transient errors."""
    from unittest.mock import patch

    import rioxarray

    boom = RuntimeError("Connection reset by peer")
    with patch.object(rioxarray, "open_rasterio", side_effect=boom):
        assert topobathy.fetch_tile("a.tif", bbox=TB_BBOX, project_id="111") is None
        with pytest.raises(ProviderFetchError, match="Connection reset"):
            topobathy.fetch_tile("b.tif", bbox=TB_BBOX, project_id="111", raise_errors=True)

    # A 404 is missing data, not a transient error, even with raise_errors.
    with patch.object(rioxarray, "open_rasterio", side_effect=RuntimeError("HTTP response code: 404")):
        assert topobathy.fetch_tile("c.tif", bbox=TB_BBOX, project_id="111", raise_errors=True) is None
