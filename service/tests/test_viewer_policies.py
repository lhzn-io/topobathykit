"""Policy naming in the viewers: dataset labels, the policy list and the per-request policy."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from fastapi.testclient import TestClient
from topobathyserve import main
from topobathyserve.main import app, get_policy_path
from topobathyserve.policy_catalog import cache_dir_of, resolve_policy_file, saved_policies
from topobathyserve.routers import dem_viewer

client = TestClient(app, follow_redirects=False)

_POLICY = """
name: {name}
version: "{version}"
description: >
  A test policy
  over two lines.
crs: EPSG:4326
variables:
  - name: elevation
    steps:
      - provider: gebco_2025
"""


def _write_cell(policy_dir: Path, name: str, x0: float) -> None:
    """A 4 x 4 cell on a 0.01 degree grid, close enough to its neighbour to share a cluster."""
    xs = x0 + 0.01 * np.arange(4, dtype=np.float64)
    ys = 0.01 * np.arange(3, -1, -1, dtype=np.float64)
    ds = xr.Dataset(
        {"elevation": (("y", "x"), np.ones((4, 4), dtype=np.float32))},
        coords={"x": xs, "y": ys},
        attrs={"crs": "EPSG:4326", "cell_bbox": [x0, 0.0, x0 + 0.03, 0.03], "policy_legend": ""},
    )
    ds.to_zarr(policy_dir / name, mode="w", consolidated=True)


@pytest.fixture
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A cache with two revisions of one policy name, one fused, and a served policy dir."""
    root = tmp_path / "cache"
    policies = root / "policies"
    policies.mkdir(parents=True)
    (policies / "OuterShelf.aaaa1111.yaml").write_text(_POLICY.format(name="Outer Shelf", version="1.2.0"))
    (policies / "OuterShelf.bbbb2222.yaml").write_text(_POLICY.format(name="Outer Shelf", version="1.1.0"))
    fused = root / "fused_zarr" / "aaaa1111"
    fused.mkdir(parents=True)
    _write_cell(fused, "c0.zarr", 0.0)
    _write_cell(fused, "c1.zarr", 0.04)

    served = tmp_path / "served"
    served.mkdir()
    default = served / "coast.yaml"
    default.write_text(_POLICY.format(name="Coast", version="1.0.0"))
    (served / "notes.txt").write_text("not a policy")

    monkeypatch.setenv("TOPOBATHYKIT_CACHE_DIR", str(root))
    monkeypatch.setattr(dem_viewer, "_fused_zarr_root", lambda: root / "fused_zarr")
    monkeypatch.setattr(main, "get_cache_root", lambda: root)
    app.dependency_overrides[get_policy_path] = lambda: default
    yield root
    app.dependency_overrides.pop(get_policy_path, None)


def test_cache_dir_of() -> None:
    assert cache_dir_of("OuterShelf.aaaa1111.yaml") == "aaaa1111"
    assert cache_dir_of("aaaa1111.yaml") == "aaaa1111"


def test_resolve_policy_file_takes_bare_names_only(tmp_path: Path) -> None:
    (tmp_path / "p.yaml").write_text("name: p\n")
    assert resolve_policy_file("p.yaml", [tmp_path]) == tmp_path / "p.yaml"
    for bad in ("../p.yaml", "sub/p.yaml", str(tmp_path / "p.yaml"), "p.txt", "", "missing.yaml"):
        assert resolve_policy_file(bad, [tmp_path]) is None


def test_saved_policies_pairs_files_with_cache_dirs(cache: Path) -> None:
    found = saved_policies(cache / "policies")
    assert {k: v.name for k, v in found.items()} == {
        "aaaa1111": "OuterShelf.aaaa1111.yaml",
        "bbbb2222": "OuterShelf.bbbb2222.yaml",
    }


def test_datasets_carry_policy_name(cache: Path) -> None:
    datasets = client.get("/api/v1/dem/datasets").json()
    assert len(datasets) == 1
    ds = datasets[0]
    assert ds["policy_hash"] == "aaaa1111"
    assert ds["policy_name"] == "Outer Shelf"
    assert ds["policy_version"] == "1.2.0"
    assert ds["policy_file"] == "OuterShelf.aaaa1111.yaml"
    assert ds["policy_description"] == "A test policy over two lines."
    assert ds["updated"] is not None


def test_dataset_without_saved_policy_has_no_name(cache: Path) -> None:
    (cache / "policies" / "OuterShelf.aaaa1111.yaml").unlink()
    ds = client.get("/api/v1/dem/datasets").json()[0]
    assert ds["policy_name"] is None
    assert ds["policy_file"] is None


def test_policies_lists_served_files_and_fused_cache_policies(cache: Path) -> None:
    body = client.get("/policies").json()
    assert body["default"] == "coast.yaml"
    assert [(p["policy_file"], p["default"]) for p in body["files"]] == [("coast.yaml", True)]
    # Only the revision with fused cells is offered from the cache.
    assert [(p["policy_file"], p["cache_dir"], p["cell_count"]) for p in body["cached"]] == [
        ("OuterShelf.aaaa1111.yaml", "aaaa1111", 2)
    ]


def test_view_policy_resolves_named_policy(cache: Path) -> None:
    assert main.get_view_policy_path(cache / "unused.yaml", None) == cache / "unused.yaml"
    served = app.dependency_overrides[get_policy_path]()
    assert main.get_view_policy_path(served, "coast.yaml") == served
    cached = main.get_view_policy_path(served, "OuterShelf.bbbb2222.yaml")
    assert cached == cache / "policies" / "OuterShelf.bbbb2222.yaml"


def test_unknown_or_path_policy_is_rejected(cache: Path) -> None:
    for bad in ("nope.yaml", "../cache/policies/OuterShelf.aaaa1111.yaml"):
        resp = client.get("/legend", params={"policy": bad})
        assert resp.status_code == 404, bad


def test_landing_is_cache_inventory() -> None:
    for path in ("/", "/viewer/"):
        resp = client.get(path)
        assert resp.status_code in (302, 307)
        assert resp.headers["location"] == "/viewer/cache-viewer.html"
    # The README link has no trailing slash; it must land on the inventory too.
    resp = TestClient(app).get("/viewer")
    assert resp.status_code == 200
    assert str(resp.url).endswith("/viewer/cache-viewer.html")


def test_old_tile_viewer_path_keeps_query() -> None:
    resp = client.get("/viewer/index.html?lat=41&lon=-72&zoom=12")
    assert resp.status_code in (302, 307)
    assert resp.headers["location"] == "/viewer/tile-viewer.html?lat=41&lon=-72&zoom=12"
    assert client.get("/viewer/tile-viewer.html").status_code == 200
