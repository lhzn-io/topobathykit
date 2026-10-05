from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import xarray as xr

from topobathykit.providers.noaa_bluetopo import NoaaBlueTopoProvider


def _make_tile_dataset(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    crs: str,
    value: float = -5.0,
    src_id: int = 1,
) -> xr.Dataset:
    x = np.linspace(x_min, x_max, 32)
    y = np.linspace(y_max, y_min, 32)

    elev = xr.DataArray(
        np.full((32, 32), value, dtype=np.float32),
        coords={"y": y, "x": x},
        dims=("y", "x"),
        name="elevation",
    )
    src = xr.DataArray(
        np.full((32, 32), src_id, dtype=np.uint32),
        coords={"y": y, "x": x},
        dims=("y", "x"),
        name="source_id",
    )

    elev.rio.write_crs(crs, inplace=True)
    src.rio.write_crs(crs, inplace=True)

    ds = xr.Dataset({"elevation": elev, "source_id": src})
    ds.rio.write_crs(crs, inplace=True)
    ds.rio.write_transform(elev.rio.transform(), inplace=True)
    return ds


def _new_provider(tmp_path: Path) -> NoaaBlueTopoProvider:
    # Provider is a singleton; reset for test isolation.
    NoaaBlueTopoProvider._singleton = None
    provider = NoaaBlueTopoProvider(cache_dir=str(tmp_path / "cache"))
    return provider


def test_validate_geotransform_success_and_failure(tmp_path: Path) -> None:
    provider = _new_provider(tmp_path)

    valid_ds = _make_tile_dataset(0.0, 10.0, 0.0, 10.0, "EPSG:4326")
    ok, _ = provider._validate_geotransform(valid_ds, label="valid")
    assert ok is True

    invalid_ds = xr.Dataset(
        {
            "elevation": xr.DataArray(
                np.ones((4, 4), dtype=np.float32),
                coords={"y": np.arange(4), "x": np.arange(4)},
                dims=("y", "x"),
            )
        }
    )
    ok, msg = provider._validate_geotransform(invalid_ds, label="invalid")
    assert ok is False
    assert "missing CRS" in msg


def test_check_tile_alignment_detects_mixed_crs(tmp_path: Path) -> None:
    provider = _new_provider(tmp_path)

    ds_4326 = _make_tile_dataset(-72.0, -71.0, 41.0, 42.0, "EPSG:4326")
    ds_3857 = _make_tile_dataset(-8100000.0, -8099000.0, 5000000.0, 5001000.0, "EPSG:3857")

    aligned = provider._check_tile_alignment([ds_4326, ds_3857], label="mixed")
    assert aligned is False


def test_fetch_layer_non_4326_bbox_keeps_bluetopo_contribution(tmp_path: Path) -> None:
    provider = _new_provider(tmp_path)

    tile_ds = _make_tile_dataset(
        x_min=1_000_000.0,
        x_max=1_001_000.0,
        y_min=1_000_000.0,
        y_max=1_001_000.0,
        crs="EPSG:3857",
        value=-8.0,
        src_id=1,
    )

    request_bbox = (1_000_050.0, 1_000_050.0, 1_000_950.0, 1_000_950.0)

    with (
        patch.object(provider, "resolve_tiles_in_bbox", return_value=["TILE_1"]),
        patch.object(provider, "load_tile_as_da", return_value=tile_ds),
        patch.object(provider, "_resolve_from_sidecar_rat", return_value=None),
        patch.object(provider, "get_source_survey_id", return_value="H_TEST"),
    ):
        result = provider.fetch_layer(bbox=request_bbox, crs="EPSG:3857")

    assert isinstance(result, xr.Dataset)
    assert "source_id" in result

    source_vals = result["source_id"].values
    assert np.nanmax(source_vals) > 0
    assert np.isfinite(result["elevation"].values).any()

    bounds = result.rio.bounds()
    # Allow one-cell tolerance because clip bounds snap to raster grid edges.
    tol = 5.0
    assert bounds[0] >= request_bbox[0] - tol
    assert bounds[1] >= request_bbox[1] - tol
    assert bounds[2] <= request_bbox[2] + tol
    assert bounds[3] <= request_bbox[3] + tol


def test_fetch_layer_reprojects_mixed_crs_tiles(tmp_path: Path) -> None:
    provider = _new_provider(tmp_path)

    ds_4326 = _make_tile_dataset(-72.0, -71.9, 41.0, 41.1, "EPSG:4326", value=-4.0, src_id=1)
    ds_3857 = _make_tile_dataset(
        -8_100_000.0,
        -8_099_000.0,
        5_000_000.0,
        5_001_000.0,
        "EPSG:3857",
        value=-6.0,
        src_id=1,
    )

    # Return different CRSs for the two tile loads.
    call_order: dict[str, Any] = {"i": 0}

    def _load_side_effect(_: str, __: tuple[float, float, float, float]) -> xr.Dataset:
        call_order["i"] += 1
        return ds_4326 if call_order["i"] == 1 else ds_3857

    with (
        patch.object(provider, "resolve_tiles_in_bbox", return_value=["TILE_A", "TILE_B"]),
        patch.object(provider, "load_tile_as_da", side_effect=_load_side_effect),
        patch.object(provider, "_resolve_from_sidecar_rat", return_value=None),
        patch.object(provider, "get_source_survey_id", return_value="H_TEST"),
    ):
        result = provider.fetch_layer(bbox=(-72.1, 40.9, -71.8, 41.2), crs="EPSG:4326")

    assert isinstance(result, xr.Dataset)
    assert result.rio.crs is not None
    assert result.rio.crs.to_string() == "EPSG:4326"
    assert np.nanmax(result["source_id"].values) > 0


def test_fetch_layer_single_projected_tile_reprojects_to_requested_crs(tmp_path: Path) -> None:
    provider = _new_provider(tmp_path)

    # Single projected tile (no mixed CRS merge path).
    ds_3857 = _make_tile_dataset(
        x_min=732_700.0,
        x_max=739_400.0,
        y_min=4_553_500.0,
        y_max=4_562_200.0,
        crs="EPSG:3857",
        value=-7.0,
        src_id=1,
    )

    with (
        patch.object(provider, "resolve_tiles_in_bbox", return_value=["TILE_ONLY"]),
        patch.object(provider, "load_tile_as_da", return_value=ds_3857),
        patch.object(provider, "_resolve_from_sidecar_rat", return_value=None),
        patch.object(provider, "get_source_survey_id", return_value="H_TEST"),
    ):
        result = provider.fetch_layer(
            bbox=(-72.158203125, 41.11246878918086, -72.1142578125, 41.14556973100949),
            crs="EPSG:4326",
        )

    assert isinstance(result, xr.Dataset)
    assert result.rio.crs is not None
    assert result.rio.crs.to_string() == "EPSG:4326"
    assert np.isfinite(result["elevation"].values).any()
    assert np.nanmax(result["source_id"].values) > 0


# ---------------------------------------------------------------------------
# Fetch-error signalling (transient failures must not look like "no data")
# ---------------------------------------------------------------------------


def test_fetch_layer_all_tiles_failing_raises_fetch_error(tmp_path: Path) -> None:
    import pytest

    from topobathykit.providers.base import ProviderFetchError

    provider = _new_provider(tmp_path)

    with (
        patch.object(provider, "resolve_tiles_in_bbox", return_value=["TILE_A", "TILE_B"]),
        patch.object(provider, "load_tile_as_da", side_effect=ProviderFetchError("VDatum read timeout")),
        pytest.raises(ProviderFetchError, match="2 error"),
    ):
        provider.fetch_layer(bbox=(-72.1, 40.9, -71.8, 41.2), crs="EPSG:4326")


def test_fetch_layer_reports_partial_result(tmp_path: Path) -> None:
    from topobathykit.providers.base import ProviderFetchError

    provider = _new_provider(tmp_path)
    ds_ok = _make_tile_dataset(-72.0, -71.9, 41.0, 41.1, "EPSG:4326", value=-4.0, src_id=1)

    def _load(tile_id: str, _bbox: tuple[float, float, float, float]) -> xr.Dataset:
        if tile_id == "TILE_BAD":
            raise ProviderFetchError("VDatum read timeout")
        return ds_ok

    with (
        patch.object(provider, "resolve_tiles_in_bbox", return_value=["TILE_OK", "TILE_BAD"]),
        patch.object(provider, "load_tile_as_da", side_effect=_load),
        patch.object(provider, "_resolve_from_sidecar_rat", return_value=None),
        patch.object(provider, "get_source_survey_id", return_value="H_TEST"),
    ):
        result = provider.fetch_layer(bbox=(-72.1, 40.9, -71.8, 41.2), crs="EPSG:4326")

    assert result.attrs["fetch_errors"] == 1
    assert np.isfinite(result["elevation"].values).any()


def _three_band_tile() -> xr.DataArray:
    """A small BlueTopo-like GeoTIFF read: band 1 elevation, band 2 uncertainty, band 3 contributor."""
    x = np.linspace(-72.0, -71.99, 8)
    y = np.linspace(41.01, 41.0, 8)
    data = np.stack([np.full((8, 8), -5.0), np.ones((8, 8)), np.ones((8, 8))]).astype(np.float32)
    da = xr.DataArray(data, coords={"band": [1, 2, 3], "y": y, "x": x}, dims=("band", "y", "x"))
    da.rio.write_crs("EPSG:4326", inplace=True)
    return da


def test_load_tile_vdatum_outage_raises_fetch_error(tmp_path: Path) -> None:
    """A VDatum timeout must raise, not return an empty Dataset.

    The empty Dataset used to surface in fetch_layer as the unrelated error
    "No variable named 'elevation'" and drop the whole BlueTopo step.
    """
    import pytest
    import rioxarray

    from topobathykit.providers.base import ProviderFetchError
    from topobathykit.vdatum import VDatumUnavailableError

    provider = _new_provider(tmp_path)

    with (
        patch.object(provider, "_resolve_tile_url", return_value="https://example.invalid/tile.tif"),
        patch.object(rioxarray, "open_rasterio", return_value=_three_band_tile()),
        patch.object(
            provider.vdatum,
            "get_robust_mllw_to_navd88_offset",
            side_effect=VDatumUnavailableError("read timeout"),
        ),
        pytest.raises(ProviderFetchError, match="VDatum"),
    ):
        provider.load_tile_as_da("TILE_T", (-72.0, 41.0, -71.99, 41.01))

    assert not (provider.cache_dir / "zarr" / "TILE_T_navd88.zarr").exists()


def test_load_tile_without_vdatum_coverage_drops_tile(tmp_path: Path) -> None:
    import rioxarray

    from topobathykit.vdatum import VDatumNoDataError

    provider = _new_provider(tmp_path)

    with (
        patch.object(provider, "_resolve_tile_url", return_value="https://example.invalid/tile.tif"),
        patch.object(rioxarray, "open_rasterio", return_value=_three_band_tile()),
        patch.object(
            provider.vdatum,
            "get_robust_mllw_to_navd88_offset",
            side_effect=VDatumNoDataError("outside model"),
        ),
    ):
        assert provider.load_tile_as_da("TILE_T", (-72.0, 41.0, -71.99, 41.01)) is None
