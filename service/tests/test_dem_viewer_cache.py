"""Tests for DEM viewer mosaic sidecar invalidation."""

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent.parent))

from topobathyserve.routers import dem_viewer

DATASET_ID = "abcd1234_1m"


def _write_cell(policy_dir: Path, name: str, x0: float, value: float) -> Path:
    """Write a 4 x 4 cell zarr on a 1-unit grid whose west edge is at x0."""
    xs = x0 + np.arange(4, dtype=np.float64)
    ys = np.arange(3, -1, -1, dtype=np.float64)
    ds = xr.Dataset(
        {
            "elevation": (("y", "x"), np.full((4, 4), value, dtype=np.float32)),
            "source_elevation": (("y", "x"), np.full((4, 4), 1, dtype=np.uint32)),
        },
        coords={"x": xs, "y": ys},
        attrs={"crs": "EPSG:4326", "cell_bbox": [x0, 0.0, x0 + 3.0, 3.0], "policy_legend": ""},
    )
    path = policy_dir / name
    ds.to_zarr(path, mode="w", consolidated=True)
    return path


def _set_mtime(path: Path, when: float) -> None:
    for p in (path, path / ".zmetadata", path / ".zattrs", path / "zarr.json"):
        if p.exists():
            os.utime(p, (when, when))


@pytest.fixture
def policy_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "fused_zarr"
    pdir = root / "abcd1234"
    pdir.mkdir(parents=True)
    monkeypatch.setattr(dem_viewer, "_fused_zarr_root", lambda: root)
    monkeypatch.setattr(dem_viewer, "_parse_dataset_id", lambda _root, _id: ("abcd1234", 1.0, 1.0, None))
    monkeypatch.setattr(dem_viewer, "_resolve_policy_dir", lambda _root, _prefix: pdir)
    return pdir


def _elevation(dataset_id: str) -> np.ndarray:
    resp: Any = asyncio.run(dem_viewer.get_elevation_binary(dataset_id, max_dim=None))
    return np.frombuffer(resp.body, dtype=np.float32)


def _provenance(dataset_id: str) -> np.ndarray:
    resp: Any = asyncio.run(dem_viewer.get_provenance_binary(dataset_id, max_dim=None))
    return np.frombuffer(resp.body, dtype=np.uint32)


def test_mosaic_rebuilt_when_cell_rewritten_in_place(policy_dir: Path) -> None:
    """Re-hydrating a cell in place (same count and canvas) must invalidate the sidecars."""
    _write_cell(policy_dir, "a.zarr", 0.0, 1.0)
    cell_b = _write_cell(policy_dir, "b.zarr", 4.0, np.nan)

    first = _elevation(DATASET_ID)
    assert np.isnan(first).any(), "cell b starts as a hole"
    assert (_provenance(DATASET_ID) == 1).all()
    assert (policy_dir / f"_mosaic_{DATASET_ID}.bin").exists()

    # Rewrite cell b with data, as a re-hydrate would, with a later mtime than the sidecars.
    _write_cell(policy_dir, "b.zarr", 4.0, 7.0)
    later = (policy_dir / f"_mosaic_{DATASET_ID}.bin").stat().st_mtime + 60
    _set_mtime(cell_b, later)

    second = _elevation(DATASET_ID)
    assert not np.isnan(second).any(), "the rebuilt mosaic must include the re-hydrated cell"
    assert (second == 7.0).any()


def test_mosaic_cache_hit_when_cells_unchanged(policy_dir: Path) -> None:
    _write_cell(policy_dir, "a.zarr", 0.0, 1.0)
    _elevation(DATASET_ID)
    cache_bin = policy_dir / f"_mosaic_{DATASET_ID}.bin"
    # Mark the cached binary so a rebuild would be visible.
    marker = np.full(cache_bin.stat().st_size // 4, 42.0, dtype=np.float32).tobytes()
    cache_bin.write_bytes(marker)

    assert (_elevation(DATASET_ID) == 42.0).all(), "unchanged cells must be served from the sidecar"


def test_clear_cache_removes_provenance_sidecar(policy_dir: Path) -> None:
    for suffix in (".bin", ".json", "_provenance.bin"):
        (policy_dir / f"_mosaic_{DATASET_ID}{suffix}").write_bytes(b"x")

    result = asyncio.run(dem_viewer.clear_dataset_cache(DATASET_ID))

    assert sorted(result["deleted"]) == sorted(
        [f"_mosaic_{DATASET_ID}.bin", f"_mosaic_{DATASET_ID}.json", f"_mosaic_{DATASET_ID}_provenance.bin"]
    )
    assert not list(policy_dir.glob("_mosaic_*"))
