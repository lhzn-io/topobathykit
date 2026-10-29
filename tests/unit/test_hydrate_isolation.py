"""Per-cell worker isolation in hydrate() (issue #45)."""

import logging
import multiprocessing
import os
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import xarray as xr
import yaml

from topobathykit import runtime
from topobathykit.providers.base import Provider
from topobathykit.providers.registry import registry


def _worker_job(job: str) -> str:
    """Stand-in for _hydrate_cell, importable by spawned workers."""
    if job == "ok":
        return "processed"
    if job == "raise":
        raise RuntimeError("boom")
    if job == "exit":
        os._exit(3)
    if job == "hang":
        time.sleep(600)
    if job == "log":
        logging.getLogger("topobathykit.test_worker").warning("hello from worker")
        return "processed"
    raise AssertionError(job)


def _run(jobs: list[str], timeout_s: float = 60.0, max_workers: int = 2) -> dict[str, list[str]]:
    results: dict[str, list[str]] = {}
    runtime._run_isolated(
        _worker_job,
        jobs,
        max_workers,
        timeout_s,
        lambda job, status: results.setdefault(job, []).append(status),
    )
    return results


def test_each_job_reports_once_and_failures_are_contained() -> None:
    results = _run(["ok", "raise", "exit", "ok"])

    assert results == {"ok": ["processed", "processed"], "raise": ["failed"], "exit": ["failed"]}
    assert not multiprocessing.active_children()


def test_hung_worker_is_killed_after_timeout_and_others_finish() -> None:
    started = time.monotonic()
    results = _run(["hang", "ok"], timeout_s=3.0)

    assert results == {"hang": ["failed"], "ok": ["processed"]}
    assert time.monotonic() - started < 60
    assert not multiprocessing.active_children()


def test_worker_logging_reaches_parent_loggers(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        results = _run(["log"])

    assert results == {"log": ["processed"]}
    assert any(
        r.name == "topobathykit.test_worker" and "hello from worker" in r.getMessage() for r in caplog.records
    )


def test_running_workers_are_killed_when_the_parent_unwinds() -> None:
    def _on_result(job: Any, status: str) -> None:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        runtime._run_isolated(_worker_job, ["ok", "hang"], 2, 600.0, _on_result)

    assert not multiprocessing.active_children()


def test_cell_timeout_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TOPOBATHY_CELL_TIMEOUT_S", raising=False)
    assert runtime._cell_timeout_s() == runtime.DEFAULT_CELL_TIMEOUT_S
    monkeypatch.setenv("TOPOBATHY_CELL_TIMEOUT_S", "900")
    assert runtime._cell_timeout_s() == 900.0
    monkeypatch.setenv("TOPOBATHY_CELL_TIMEOUT_S", "-1")
    assert runtime._cell_timeout_s() == runtime.DEFAULT_CELL_TIMEOUT_S


class _Constant(Provider):
    def fetch_layer(
        self,
        bbox: tuple[float, float, float, float],
        resolution: float | None = None,
        crs: str = "EPSG:4326",
        **kwargs: Any,
    ) -> xr.DataArray:
        west, south, east, north = bbox
        n = 16
        dx, dy = (east - west) / n, (north - south) / n
        da = xr.DataArray(
            np.full((n, n), -5.0, dtype=np.float32),
            coords={
                "y": np.linspace(north - dy / 2, south + dy / 2, n),
                "x": np.linspace(west + dx / 2, east - dx / 2, n),
            },
            dims=("y", "x"),
        )
        return cast(xr.DataArray, da.rio.write_crs(crs))

    def get_metadata(self) -> dict[str, Any]:
        return {"name": "constant"}


def test_hydrate_sends_only_uncached_cells_to_isolated_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TOPOBATHYKIT_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("TOPOBATHY_CELL_TIMEOUT_S", "123")
    registry.register("iso_constant", _Constant)
    policy = tmp_path / "policy.yaml"
    policy.write_text(
        yaml.dump(
            {
                "name": "IsolationTest",
                "crs": "EPSG:4326",
                "variables": [
                    {"name": "elevation", "steps": [{"provider": "iso_constant", "operation": "overwrite"}]}
                ],
            }
        )
    )
    calls: list[tuple[int, float]] = []

    def _inline(fn: Any, jobs: Any, max_workers: int, timeout_s: float, on_result: Any) -> None:
        jobs = list(jobs)
        calls.append((len(jobs), timeout_s))
        for job in jobs:
            on_result(job, fn(job))

    monkeypatch.setattr(runtime, "_run_isolated", _inline)
    bbox = (-72.02, 41.22, -71.98, 41.26)

    first = runtime.hydrate(str(policy), bbox, resolution=100.0, max_workers=1, max_nan_fraction=1.0)
    second = runtime.hydrate(str(policy), bbox, resolution=100.0, max_workers=1, max_nan_fraction=1.0)

    assert first["processed"] == first["total"] > 0
    assert second["cached"] == second["total"]
    assert calls == [(first["total"], 123.0)], "cached cells must not start workers"
