"""
Tests for the order in which NCEI BAG surveys are merged.

The BAG merge is first-valid-wins, so the order decides which survey supplies an
overlapping pixel. Surveys are ordered newest first (by DATE_SURVEY_END, or
SURVEY_YEAR), then finest. Discovery used to cache only URLs, so the date was lost on
every cache hit; entries now carry {url, res, date}, and URL-only entries from older
caches still load, with an unknown date.
"""

import contextlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from topobathykit.providers.ncei_bag import BAGDiscovery, BAGProvider, order_bags_newest_first

BASE = "https://example.com/bags"
MS_PER_YEAR = 31536000000


def _ms(year: int) -> int:
    return (year - 1970) * MS_PER_YEAR


OLD_FINE = f"{BASE}/H12001_MB_50cm_MLLW_1of1.bag"  # 2009, 0.5 m
NEW_COARSE = f"{BASE}/H13500_MB_4m_MLLW_1of1.bag"  # 2021, 4 m
NEW_FINE = f"{BASE}/H13500_MB_1m_MLLW_1of1.bag"  # 2021, 1 m (same survey, finer band)
UNDATED_FINE = f"{BASE}/H11000_MB_50cm_MLLW_1of1.bag"
UNDATED_W = f"{BASE}/W00410_MB_4m_MLLW_combined.bag"

DATES = {OLD_FINE: _ms(2009), NEW_COARSE: _ms(2021), NEW_FINE: _ms(2021)}

BBOX = (-73.8, 40.8, -73.7, 40.9)


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("TOPOBATHYKIT_CACHE_DIR", str(tmp_path / "cache"))
    cache_path = tmp_path / "ncei_bag" / "discovery_cache.json"
    monkeypatch.setattr(BAGDiscovery, "DISCOVERY_CACHE_PATH", cache_path)
    monkeypatch.setattr(BAGDiscovery, "_bag_meta", {})
    return cache_path


def _entry(url: str) -> dict[str, Any]:
    return {"url": url, "res": None, "date": DATES.get(url)}


def _bbox_key() -> str:
    # The key find_bags_by_bbox derives for BBOX (snapped to a 0.01 degree grid).
    g = 0.01
    w, s, e, n = BBOX
    parts = [np.floor(w / g) * g, np.floor(s / g) * g, np.ceil(e / g) * g, np.ceil(n / g) * g]
    return "_".join(str(round(float(p), 4)) for p in parts) + "_EPSG_4326"


def _features(urls: list[str]) -> dict[str, Any]:
    feats = []
    for u in urls:
        attrs: dict[str, Any] = {"SURVEY_ID": u.split("/")[-1][:6], "DOWNLOAD_URL": u}
        if u in DATES:
            attrs["DATE_SURVEY_END"] = DATES[u]
        feats.append({"attributes": attrs})
    return {"features": feats}


def _no_network() -> Any:
    return patch("topobathykit.providers.ncei_bag.requests.get", side_effect=AssertionError("network used"))


def test_order_is_newest_first_then_finest() -> None:
    from topobathykit.providers.ncei_bag import _bag_entry

    entries = [_bag_entry(e) for e in map(_entry, [UNDATED_W, OLD_FINE, NEW_COARSE, UNDATED_FINE, NEW_FINE])]
    ordered = [e["url"] for e in order_bags_newest_first([e for e in entries if e])]
    # Newest survey first (its finer band ahead of its coarser one), then the older
    # survey, then entries with no date: finest first, as before dates were recorded.
    assert ordered == [NEW_FINE, NEW_COARSE, OLD_FINE, UNDATED_FINE, UNDATED_W]


def test_query_caches_dates_and_returns_newest_first(_isolated: Path) -> None:
    response = MagicMock()
    response.json.return_value = _features([OLD_FINE, NEW_COARSE, NEW_FINE])
    response.raise_for_status.return_value = None
    with patch("topobathykit.providers.ncei_bag.requests.get", return_value=response) as get:
        urls = BAGDiscovery.find_bags_by_bbox(*BBOX)
    assert get.call_count == 1
    assert urls == [NEW_FINE, NEW_COARSE, OLD_FINE]

    data = json.loads(_isolated.read_text())
    entries = data[_bbox_key() + BAGDiscovery.DISCOVERY_KEY_SUFFIX]
    assert {e["url"]: e["date"] for e in entries} == DATES
    assert all(isinstance(e["res"], float) for e in entries)

    # A cache hit keeps the dates: same order, no network.
    BAGDiscovery._bag_meta.clear()
    with _no_network():
        assert BAGDiscovery.find_bags_by_bbox(*BBOX) == [NEW_FINE, NEW_COARSE, OLD_FINE]
    assert BAGDiscovery.entry_for(OLD_FINE)["date"] == DATES[OLD_FINE]


def test_url_only_legacy_cache_still_loads(_isolated: Path) -> None:
    # A cache file from before dates were stored: bare URL lists under the bbox key.
    _isolated.parent.mkdir(parents=True, exist_ok=True)
    _isolated.write_text(json.dumps({_bbox_key(): [UNDATED_W, NEW_COARSE, UNDATED_FINE]}))

    # The bbox is re-queried once for dates; when that fails the legacy list is used,
    # with unknown dates (finest first, then survey id descending).
    with patch("topobathykit.providers.ncei_bag.requests.get", side_effect=OSError("offline")):
        urls = BAGDiscovery.find_bags_by_bbox(*BBOX)
    assert urls == [UNDATED_FINE, UNDATED_W, NEW_COARSE]
    assert BAGDiscovery.entry_for(NEW_COARSE)["date"] is None

    # The legacy entry is left as it was, for releases that still read it.
    assert json.loads(_isolated.read_text()) == {_bbox_key(): [UNDATED_W, NEW_COARSE, UNDATED_FINE]}


def test_legacy_cache_is_upgraded_when_the_query_succeeds(_isolated: Path) -> None:
    _isolated.parent.mkdir(parents=True, exist_ok=True)
    _isolated.write_text(json.dumps({_bbox_key(): [OLD_FINE, NEW_COARSE]}))
    response = MagicMock()
    response.json.return_value = _features([OLD_FINE, NEW_COARSE])
    response.raise_for_status.return_value = None
    with patch("topobathykit.providers.ncei_bag.requests.get", return_value=response):
        urls = BAGDiscovery.find_bags_by_bbox(*BBOX)
    # With dates, the 2021 4 m survey supersedes the 2009 50 cm one.
    assert urls == [NEW_COARSE, OLD_FINE]
    data = json.loads(_isolated.read_text())
    assert data[_bbox_key()] == [OLD_FINE, NEW_COARSE]
    assert _bbox_key() + BAGDiscovery.DISCOVERY_KEY_SUFFIX in data


def test_fetch_layer_merges_newest_survey_first(tmp_path: Path) -> None:
    provider = BAGProvider(cache_dir=str(tmp_path))
    BAGDiscovery._remember([{"url": u, "res": 1.0, "date": d} for u, d in DATES.items()])
    fetch_order: list[str] = []

    def spy_fetch_bag(survey_id: str, download_url: Any = None, **_kw: Any) -> None:
        fetch_order.append(download_url)
        return None

    with (
        patch.object(BAGDiscovery, "find_bags_by_bbox", return_value=[OLD_FINE, UNDATED_FINE, NEW_COARSE]),
        patch.object(provider, "fetch_bag", side_effect=spy_fetch_bag),
        contextlib.suppress(Exception),  # nothing loads; only the order matters here
    ):
        provider.fetch_layer(bbox=BBOX, resolution=1.0, crs="EPSG:4326")

    assert fetch_order == [NEW_COARSE, OLD_FINE, UNDATED_FINE]
