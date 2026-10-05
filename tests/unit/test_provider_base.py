from typing import Any

import numpy as np
import pytest
import xarray as xr

from topobathykit.providers.base import Provider, sanitize_elevation_nodata


class DummyProvider(Provider):
    def fetch_layer(
        self,
        bbox: tuple[float, float, float, float],
        resolution: float | None = None,
        crs: str = "EPSG:4326",
        **kwargs: Any,
    ) -> Any:
        """Use our new helper."""
        return self._normalize_bbox(bbox, crs)

    def get_metadata(self) -> dict[str, Any]:
        """Return dummy metadata."""
        return {"name": "dummy"}


def test_normalize_bbox_4326() -> None:
    """Test normalization with EPSG:4326."""
    provider = DummyProvider()
    bbox = (-74.0, 40.0, -73.0, 41.0)
    # Should return unchanged
    result = provider.fetch_layer(bbox, crs="EPSG:4326")
    assert result == bbox


def test_normalize_bbox_3857() -> None:
    """Test normalization from EPSG:3857."""
    provider = DummyProvider()
    # NYC-ish in 3857 meters
    bbox_3857 = (-8238000.0, 4970000.0, -8226000.0, 4983000.0)
    result = provider.fetch_layer(bbox_3857, crs="EPSG:3857")

    # Verify result is in degrees (-180 to 180, -90 to 90)
    for coord in result:
        assert -180 <= coord <= 180

    # Specific check: -8238000 is approx -74 lon
    assert -74.5 < result[0] < -73.5
    assert 40.0 < result[1] < 41.0


def test_sanitize_elevation_nodata_masks_metadata_nodata() -> None:
    da = xr.DataArray(np.array([[3.0, -999999.0], [1.0, -2.0]]), dims=("y", "x"))
    da.rio.write_nodata(-999999.0, inplace=True)

    cleaned = sanitize_elevation_nodata(da)

    assert np.isnan(cleaned.values[0, 1])
    assert cleaned.values[1, 1] == pytest.approx(-2.0)


def test_sanitize_elevation_nodata_applies_custom_bounds() -> None:
    da = xr.DataArray(np.array([[4.0, -999500.0], [2.0, 7.0]]), dims=("y", "x"))

    cleaned = sanitize_elevation_nodata(da, min_valid=-999000, max_valid=6.0)

    assert np.isnan(cleaned.values[0, 1])
    assert np.isnan(cleaned.values[1, 1])
    assert cleaned.values[0, 0] == pytest.approx(4.0)


def test_sanitize_elevation_nodata_masks_extra_sentinels() -> None:
    da = xr.DataArray(np.array([[3.2, -9999.0], [0.0, 1.0]]), dims=("y", "x"))

    cleaned = sanitize_elevation_nodata(da, extra_sentinels=(-9999.0,))

    assert np.isnan(cleaned.values[0, 1])
    assert cleaned.values[0, 0] == pytest.approx(3.2)
