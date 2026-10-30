"""Unit tests for the GMRT provider, with the GridServer replaced by a local fake."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
import requests
from rasterio.transform import from_bounds

from topobathykit.providers import gmrt as gmrt_mod
from topobathykit.providers.base import (
    SOURCE_ID_BLOCK,
    SOURCE_ID_BLOCKS,
    ProviderFetchError,
    ProviderNoDataError,
)
from topobathykit.providers.gmrt import GMRTProvider
from topobathykit.providers.registry import registry

BBOX = (-69.6, 39.8, -69.4, 40.0)


def _metadata_xml(version: str, mpn: float) -> bytes:
    return f"""<?xml version="1.0" encoding="utf-8"?>
<doc>
  <meters_per_node>{mpn:.2f}</meters_per_node>
  <gmrt_version>{version}</gmrt_version>
  <gmrt_release_date>June 2026</gmrt_release_date>
</doc>""".encode()


def _geotiff(path: Path, data: np.ndarray) -> bytes:
    h, w = data.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_bounds(*BBOX, w, h),
    ) as dst:
        dst.write(data.astype(np.float32), 1)
    return path.read_bytes()


class _Resp:
    def __init__(self, content: bytes, status: int = 200) -> None:
        self.content = content
        self.text = content.decode(errors="replace")
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class FakeGridServer:
    """Answers metadata and grid requests; records each call."""

    def __init__(self, tif: bytes, version: str = "4.5.0", mpn: float = 61.15) -> None:
        self.tif = tif
        self.version = version
        self.mpn = mpn
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.down = False

    def get(self, url: str, params: dict[str, Any] | None = None, timeout: float | None = None) -> _Resp:
        self.calls.append((url, dict(params or {})))
        if self.down:
            raise requests.ConnectionError("offline")
        if url == gmrt_mod.METADATA_URL:
            return _Resp(_metadata_xml(self.version, self.mpn))
        return _Resp(self.tif)

    def grid_calls(self) -> list[dict[str, Any]]:
        return [p for u, p in self.calls if u == gmrt_mod.GRIDSERVER_URL]


@pytest.fixture(autouse=True)
def _clear_memo() -> None:
    GMRTProvider._metadata_memo.clear()


@pytest.fixture
def grid(tmp_path: Path) -> np.ndarray:
    data = np.full((20, 16), -500.0)
    data[:5, :] = np.nan  # no multibeam in the northern rows
    return data


@pytest.fixture
def server(tmp_path: Path, grid: np.ndarray, monkeypatch: pytest.MonkeyPatch) -> FakeGridServer:
    fake = FakeGridServer(_geotiff(tmp_path / "src.tif", grid))
    monkeypatch.setattr(gmrt_mod.requests, "get", fake.get)
    return fake


def test_registered() -> None:
    assert registry.get_provider_class("gmrt") is GMRTProvider
    assert "gmrt" in SOURCE_ID_BLOCKS


def test_fetch_defaults_to_topo_mask(server: FakeGridServer, tmp_path: Path) -> None:
    ds = GMRTProvider(cache_dir=tmp_path).fetch_layer(BBOX, resolution=100.0)

    (params,) = server.grid_calls()
    assert params["layer"] == "topo-mask"
    assert params["format"] == "geotiff"
    assert params["mresolution"] == 100.0
    assert (params["minlongitude"], params["minlatitude"]) == (-69.6, 39.8)

    elev = ds["elevation"]
    assert elev.dtype == np.float32
    assert ds.rio.crs.to_epsg() == 4326
    assert int(np.isnan(elev.values).sum()) == 5 * 16

    sid = ds["source_id"].values
    assert set(np.unique(sid)) == {0, int(sid.max())}
    assert int(sid.max()) // SOURCE_ID_BLOCK == SOURCE_ID_BLOCKS["gmrt"]
    assert np.array_equal(sid == 0, np.isnan(elev.values))

    entry = ds.attrs["provenance_dict"][int(sid.max())]
    assert entry["provider"] == "gmrt"
    assert entry["gmrt_version"] == "4.5.0"
    assert entry["product"] == "topo-mask"
    assert entry["license"] == "CC BY 4.0"
    assert "doi:10.1029/2008GC002332" in entry["citation"] and "4.5.0" in entry["citation"]
    assert ds.attrs["vertical_datum"] == "MSL"


def test_topo_layer_option(server: FakeGridServer, tmp_path: Path) -> None:
    GMRTProvider(cache_dir=tmp_path).fetch_layer(BBOX, filter={"layer": "topo"})
    (params,) = server.grid_calls()
    assert params["layer"] == "topo"
    assert params["resolution"] == "max"


def test_rejects_unknown_layer(server: FakeGridServer, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="layer"):
        GMRTProvider(cache_dir=tmp_path).fetch_layer(BBOX, filter={"layer": "bathy"})


def test_cache_hit_skips_download(server: FakeGridServer, tmp_path: Path) -> None:
    provider = GMRTProvider(cache_dir=tmp_path)
    provider.fetch_layer(BBOX, resolution=100.0)
    GMRTProvider._metadata_memo.clear()  # as in a new process
    provider.fetch_layer(BBOX, resolution=100.0)
    assert len(server.grid_calls()) == 1


def test_new_release_invalidates_cache(server: FakeGridServer, tmp_path: Path) -> None:
    provider = GMRTProvider(cache_dir=tmp_path)
    provider.fetch_layer(BBOX, resolution=100.0)

    GMRTProvider._metadata_memo.clear()
    server.version = "4.6.0"
    ds = provider.fetch_layer(BBOX, resolution=100.0)

    assert len(server.grid_calls()) == 2
    assert ds.attrs["gmrt_version"] == "4.6.0"
    assert {p.name for p in (tmp_path / "gmrt").iterdir()} == {"v4_5_0", "v4_6_0"}


def test_offline_uses_cached_grid(server: FakeGridServer, tmp_path: Path) -> None:
    provider = GMRTProvider(cache_dir=tmp_path)
    provider.fetch_layer(BBOX, resolution=100.0)

    GMRTProvider._metadata_memo.clear()
    server.down = True
    ds = provider.fetch_layer(BBOX, resolution=100.0)
    assert ds.attrs["gmrt_version"] == "4.5.0"


def test_offline_without_cache_is_a_fetch_error(server: FakeGridServer, tmp_path: Path) -> None:
    server.down = True
    with pytest.raises(ProviderFetchError):
        GMRTProvider(cache_dir=tmp_path).fetch_layer(BBOX)


def test_error_text_is_a_fetch_error(server: FakeGridServer, tmp_path: Path) -> None:
    server.tif = b"Error: bounds out of range"
    with pytest.raises(ProviderFetchError, match="GeoTIFF"):
        GMRTProvider(cache_dir=tmp_path).fetch_layer(BBOX)
    assert not list((tmp_path / "gmrt").rglob("*.tif"))


def test_no_multibeam_is_nodata(server: FakeGridServer, tmp_path: Path) -> None:
    server.tif = _geotiff(tmp_path / "empty.tif", np.full((8, 8), np.nan))
    with pytest.raises(ProviderNoDataError):
        GMRTProvider(cache_dir=tmp_path).fetch_layer(BBOX)
