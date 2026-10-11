"""
Unit tests for the USGS 3DEP provider's filter patterns and error handling.

STAC and raster I/O are mocked, so these run offline.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

from topobathykit.providers.base import ProviderFetchError, ProviderNoDataError
from topobathykit.providers.usgs_3dep import Usgs3DepProvider

BBOX = (-74.01, 40.70, -74.00, 40.71)
BASE_URL = "https://landcoverdata.blob.core.windows.net/usgs-3dep/USGS_13_{}.tif"
ITEMS = [
    {"href": BASE_URL.format("n41w074") + "?sig=x", "bbox": [-74.0, 40.0, -73.0, 41.0]},
    {"href": BASE_URL.format("n41w075"), "bbox": [-75.0, 40.0, -74.0, 41.0]},
]


@pytest.fixture
def provider(tmp_path: Path) -> Iterator[Usgs3DepProvider]:
    Usgs3DepProvider._instance = None
    Usgs3DepProvider._initialized = False
    Usgs3DepProvider._catalog = None
    p = Usgs3DepProvider(cache_dir=str(tmp_path))
    yield p
    Usgs3DepProvider._instance = None
    Usgs3DepProvider._initialized = False
    Usgs3DepProvider._catalog = None


def _raster(*args: Any, **kwargs: Any) -> xr.DataArray:
    x = np.linspace(-74.012, -73.998, 20)
    y = np.linspace(40.712, 40.698, 20)
    da = xr.DataArray(np.full((20, 20), 5.0, dtype=np.float32), coords={"y": y, "x": x}, dims=("y", "x"))
    da.rio.write_crs("EPSG:4326", inplace=True)
    return da


def _opened_names(mock_open: MagicMock) -> list[str]:
    return [Path(c.args[0].split("?")[0]).name for c in mock_open.call_args_list]


@pytest.mark.parametrize(
    ("filter_opts", "expected"),
    [
        ({}, ["USGS_13_n41w074.tif", "USGS_13_n41w075.tif"]),
        ({"include_patterns": ["*n41w074*"]}, ["USGS_13_n41w074.tif"]),
        ({"exclude_patterns": ["*n41w074*"]}, ["USGS_13_n41w075.tif"]),
    ],
)
def test_patterns_select_assets_by_file_name(
    provider: Usgs3DepProvider, filter_opts: dict[str, Any], expected: list[str]
) -> None:
    items = [dict(i) for i in ITEMS]
    with (
        patch.object(provider, "_query_stac_api", return_value=items),
        patch("topobathykit.providers.usgs_3dep.rioxarray.open_rasterio", side_effect=_raster) as mock_open,
    ):
        ds = provider.fetch_layer(BBOX, filter=filter_opts)

    assert _opened_names(mock_open) == expected
    names = sorted(v["name"] for v in ds.attrs["provenance_dict"].values())
    assert names == [f"3DEP: {n}" for n in expected]


def test_excluding_every_asset_is_no_data_but_not_negative_cached(provider: Usgs3DepProvider) -> None:
    items = [dict(i) for i in ITEMS]
    with (
        patch.object(provider, "_query_stac_api", return_value=items),
        patch("topobathykit.providers.usgs_3dep.rioxarray.open_rasterio") as mock_open,
        pytest.raises(ProviderNoDataError),
    ):
        provider.fetch_layer(BBOX, filter={"exclude_patterns": ["USGS_13_*"]})

    mock_open.assert_not_called()
    assert not provider.manifest.has_no_coverage("3dep-seamless", BBOX)


def test_stac_outage_is_a_fetch_error_not_no_coverage(provider: Usgs3DepProvider) -> None:
    catalog = MagicMock()
    catalog.search.side_effect = ConnectionError("ConnectionError: gateway timeout")
    provider._catalog = catalog

    with patch("topobathykit.providers.usgs_3dep.time.sleep"), pytest.raises(ProviderFetchError):
        provider.fetch_layer(BBOX)

    assert catalog.search.call_count == 5  # first attempt plus four retries
    assert not provider.manifest.has_no_coverage("3dep-seamless", BBOX)


def test_every_asset_failing_to_stream_is_a_fetch_error(provider: Usgs3DepProvider) -> None:
    items = [dict(i) for i in ITEMS]
    with (
        patch.object(provider, "_query_stac_api", return_value=items),
        patch(
            "topobathykit.providers.usgs_3dep.rioxarray.open_rasterio",
            side_effect=RuntimeError("HTTP 500 from blob storage"),
        ),
        pytest.raises(ProviderFetchError, match="2 3dep-seamless asset"),
    ):
        provider.fetch_layer(BBOX)

    assert not provider.manifest.has_no_coverage("3dep-seamless", BBOX)


def test_partial_stream_failure_is_flagged(provider: Usgs3DepProvider) -> None:
    items = [dict(i) for i in ITEMS]

    def flaky(href: str, **kwargs: Any) -> xr.DataArray:
        if "n41w075" in href:
            raise RuntimeError("HTTP 500 from blob storage")
        return _raster()

    with (
        patch.object(provider, "_query_stac_api", return_value=items),
        patch("topobathykit.providers.usgs_3dep.rioxarray.open_rasterio", side_effect=flaky),
    ):
        ds = provider.fetch_layer(BBOX)

    assert ds.attrs["fetch_errors"] == 1
    assert int(ds["elevation"].notnull().sum()) > 0
