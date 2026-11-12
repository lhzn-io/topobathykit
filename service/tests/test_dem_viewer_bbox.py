"""DEM viewer bbox crop: /meta and the binaries agree on the cropped grid."""

import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from topobathyserve.main import app
from topobathyserve.routers import dem_viewer

DATASET_ID = "abcd1234_1m"
BASE = f"/api/v1/dem/datasets/{DATASET_ID}"

client = TestClient(app)


def _write_cell(policy_dir: Path, name: str, x0: float, n: int, source_id: int) -> None:
    """An n x n cell on a 1-unit grid with west edge x0 and south edge 0.

    Each pixel holds ``1000 * row + column`` on the dataset canvas (row 0 is north), so a
    crop can be checked against the canvas positions it claims to cover.
    """
    xs = x0 + np.arange(n, dtype=np.float64)
    ys = np.arange(n - 1, -1, -1, dtype=np.float64)
    rows, cols = np.meshgrid(np.arange(n), np.arange(n) + int(x0), indexing="ij")
    elev = (1000 * rows + cols).astype(np.float32)
    ds = xr.Dataset(
        {
            "elevation": (("y", "x"), elev),
            "source_elevation": (("y", "x"), np.full((n, n), source_id, dtype=np.uint32)),
        },
        coords={"x": xs, "y": ys},
        attrs={"crs": "EPSG:4326", "cell_bbox": [x0, 0.0, x0 + n - 1, n - 1.0], "policy_legend": ""},
    )
    ds.to_zarr(policy_dir / name, mode="w", consolidated=True)


@pytest.fixture
def policy_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "fused_zarr"
    pdir = root / "abcd1234"
    pdir.mkdir(parents=True)
    monkeypatch.setattr(dem_viewer, "_fused_zarr_root", lambda: root)
    monkeypatch.setattr(dem_viewer, "_parse_dataset_id", lambda _root, _id: ("abcd1234", 1.0, 1.0, None))
    monkeypatch.setattr(dem_viewer, "_resolve_policy_dir", lambda _root, _prefix: pdir)
    return pdir


@pytest.fixture
def small(policy_dir: Path) -> Path:
    """Two 4 x 4 cells side by side: an 8 x 4 canvas over bounds [0, 0, 7, 3]."""
    _write_cell(policy_dir, "a.zarr", 0.0, 4, 1)
    _write_cell(policy_dir, "b.zarr", 4.0, 4, 2)
    return policy_dir


@pytest.fixture
def large(policy_dir: Path) -> Path:
    """Two 100 x 100 cells side by side: a 200 x 100 canvas over bounds [0, 0, 199, 99]."""
    _write_cell(policy_dir, "a.zarr", 0.0, 100, 1)
    _write_cell(policy_dir, "b.zarr", 100.0, 100, 2)
    return policy_dir


def _get(**params: object) -> tuple[dict, np.ndarray, np.ndarray]:
    meta = client.get(f"{BASE}/meta", params=params)
    elev = client.get(f"{BASE}/elevation.bin", params=params)
    prov = client.get(f"{BASE}/provenance.bin", params=params)
    for r in (meta, elev, prov):
        assert r.status_code == 200, r.text
    return (
        meta.json(),
        np.frombuffer(elev.content, dtype=np.float32),
        np.frombuffer(prov.content, dtype=np.uint32),
    )


def test_meta_without_bbox_reports_whole_dataset_in_metres(small: Path) -> None:
    meta, elev, prov = _get()
    assert meta["units"] == "m"
    assert (meta["width"], meta["height"]) == (8, 4)
    assert meta["bounds"] == [0.0, 0.0, 7.0, 3.0]
    assert elev.size == prov.size == 32


def test_bbox_crops_to_enclosing_pixels(small: Path) -> None:
    meta, elev, prov = _get(bbox="2.5,0.5,5.2,1.5")

    # Columns 2..6 enclose x 2.5..5.2; rows 1..3 (north-first) enclose y 1.5..0.5.
    assert (meta["width"], meta["height"]) == (5, 3)
    assert meta["bounds"] == [2.0, 0.0, 6.0, 2.0]
    assert (meta["full_width"], meta["full_height"]) == (8, 4)
    assert meta["dx"] == pytest.approx(1.0)
    assert meta["dy"] == pytest.approx(1.0)
    assert meta["units"] == "m"

    assert elev.size == meta["width"] * meta["height"]
    assert prov.size == meta["width"] * meta["height"]
    grid = elev.reshape(3, 5)
    rows, cols = np.meshgrid(np.arange(1, 4), np.arange(2, 7), indexing="ij")
    np.testing.assert_array_equal(grid, 1000 * rows + cols)
    np.testing.assert_array_equal(prov.reshape(3, 5), np.where(cols < 4, 1, 2))


def test_bbox_overhanging_the_dataset_is_clipped_to_it(small: Path) -> None:
    meta, elev, _ = _get(bbox="-10,-10,1.0,1.0")
    assert meta["bounds"] == [0.0, 0.0, 1.0, 1.0]
    assert (meta["width"], meta["height"]) == (2, 2)
    np.testing.assert_array_equal(elev.reshape(2, 2), [[2000, 2001], [3000, 3001]])


@pytest.mark.parametrize("max_dim", [None, 64, 100])
def test_elevation_length_matches_meta_for_bbox_and_max_dim(large: Path, max_dim: int | None) -> None:
    params: dict[str, object] = {"bbox": "20.4,10,170.2,90"}
    if max_dim is not None:
        params["max_dim"] = max_dim
    meta, elev, prov = _get(**params)

    assert elev.size == meta["width"] * meta["height"]
    assert prov.size == meta["width"] * meta["height"]
    # The crop is columns 20..171 and rows 9..89: 152 x 81 before max_dim.
    assert meta["bounds"] == [20.0, 10.0, 171.0, 90.0]
    if max_dim is None:
        assert (meta["width"], meta["height"]) == (152, 81)
    else:
        assert max(meta["width"], meta["height"]) == max_dim
        assert meta["width"] < 152


def test_max_dim_applies_to_the_crop_not_the_dataset(large: Path) -> None:
    """A crop already under max_dim is served at full resolution."""
    meta, elev, _ = _get(bbox="10,10,60,40", max_dim=64)
    assert (meta["width"], meta["height"]) == (51, 31)
    assert meta["scale"] == 1.0
    assert elev.size == 51 * 31


def test_headers_match_meta(large: Path) -> None:
    params = {"bbox": "20.4,10,170.2,90", "max_dim": 64}
    meta = client.get(f"{BASE}/meta", params=params).json()
    elev = client.get(f"{BASE}/elevation.bin", params=params)
    assert elev.headers["X-Grid-Width"] == str(meta["width"])
    assert elev.headers["X-Grid-Height"] == str(meta["height"])


@pytest.mark.parametrize("endpoint", ["meta", "elevation.bin", "provenance.bin"])
def test_bbox_outside_dataset_is_rejected_without_a_mosaic_build(small: Path, endpoint: str) -> None:
    r = client.get(f"{BASE}/{endpoint}", params={"bbox": "20,20,30,30"})
    assert r.status_code in (400, 422)
    assert not list(small.glob("_mosaic_*")), "a rejected bbox must not build the mosaic"


@pytest.mark.parametrize("bbox", ["1,2,3", "a,b,c,d", "5,0,1,3", "0,3,7,1", "0,0,nan,3"])
@pytest.mark.parametrize("endpoint", ["meta", "elevation.bin", "provenance.bin"])
def test_malformed_bbox_is_422(small: Path, endpoint: str, bbox: str) -> None:
    r = client.get(f"{BASE}/{endpoint}", params={"bbox": bbox})
    assert r.status_code == 422
