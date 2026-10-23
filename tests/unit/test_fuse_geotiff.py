from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr
from affine import Affine
from fastapi.testclient import TestClient
from rasterio.crs import CRS
from rasterio.io import MemoryFile
from topobathyserve import main

from topobathykit import runtime
from topobathykit.policy.schema import CompositionStep, FusionPolicy, VariableStrategy
from topobathykit.providers.base import Provider
from topobathykit.providers.registry import registry


class _ConstantProvider(Provider):
    def fetch_layer(
        self,
        bbox: tuple[float, float, float, float],
        resolution: float | None = None,
        crs: str = "EPSG:4326",
        **kwargs: Any,
    ) -> xr.DataArray:
        west, south, east, north = bbox
        dx, dy = (east - west) / 20, (north - south) / 20
        return xr.DataArray(
            np.full((20, 20), 5.0, dtype=np.float32),
            coords={
                "x": np.linspace(west + dx / 2, east - dx / 2, 20),
                "y": np.linspace(north - dy / 2, south + dy / 2, 20),
            },
            dims=("y", "x"),
        ).rio.write_crs(crs)

    def get_metadata(self) -> dict[str, Any]:
        return {"name": "constant"}


@pytest.mark.parametrize("method", ["get", "post", "post_yaml"])
@pytest.mark.parametrize("policy_crs", ["EPSG:3857", "EPSG:32619"])
def test_fuse_geotiff_fresh_and_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str, policy_crs: str
) -> None:
    monkeypatch.setitem(registry._providers, "geotiff_test", _ConstantProvider)
    policy = FusionPolicy(
        name="GeoTIFF metadata",
        crs=policy_crs,
        variables=[
            VariableStrategy(
                name="elevation",
                steps=[CompositionStep(provider="geotiff_test", operation="overwrite")],
            )
        ],
    )
    policy_path = tmp_path / "policy.yaml"
    default_policy = policy.model_copy(update={"crs": "EPSG:4326"}) if method == "post_yaml" else policy
    policy_path.write_text(default_policy.model_dump_json())
    monkeypatch.setitem(main.app.dependency_overrides, main.get_policy_path, lambda: policy_path)
    monkeypatch.setenv("TOPOBATHYKIT_CACHE_DIR", str(tmp_path / "cache"))
    bbox = [-70.735, 43.049, -70.681, 43.0885]
    transform = None
    cell_count = 0

    with patch.object(runtime, "_run_cell", wraps=runtime._run_cell) as run_cell:
        client = TestClient(main.app)
        for cached in (False, True):
            if method == "get":
                response = client.get(
                    "/fuse",
                    params={"bbox": ",".join(map(str, bbox)), "resolution": 500, "format": "geotiff"},
                )
            else:
                body = {"bbox": bbox, "resolution": 500, "format": "geotiff"}
                if method == "post_yaml":
                    body["policy_yaml"] = policy.model_dump_json()
                response = client.post("/fuse", json=body)

            assert response.status_code == 200, response.text
            assert response.headers["content-type"] == "image/tiff"
            with MemoryFile(response.content) as mem, mem.open() as src:
                assert src.crs == CRS.from_string(policy_crs)
                assert src.nodata is not None and np.isnan(src.nodata)
                assert src.transform.a == pytest.approx(500)
                assert src.transform.e == pytest.approx(-500)
                np.testing.assert_allclose(src.read(1), 5.0)
                if cached:
                    assert src.transform == transform
                transform = src.transform

            if cached:
                assert run_cell.call_count == cell_count
            else:
                cell_count = run_cell.call_count
                assert cell_count > 1
                assert len(list((tmp_path / "cache" / "fused_zarr").rglob("*.zarr"))) == cell_count


@pytest.mark.parametrize("method", ["get", "post"])
@pytest.mark.parametrize("array_crs", [None, "EPSG:32619"])
def test_fuse_geotiff_array_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str, array_crs: str | None
) -> None:
    policy = FusionPolicy(
        name="GeoTIFF metadata",
        crs="EPSG:3857",
        variables=[VariableStrategy(name="elevation", steps=[])],
    )
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(policy.model_dump_json())
    monkeypatch.setitem(main.app.dependency_overrides, main.get_policy_path, lambda: policy_path)
    values = np.array([[1.0, np.nan], [-2.0, 3.0]], dtype=np.float32)
    ds = xr.Dataset(
        {"elevation": (("y", "x"), values)},
        coords={"x": [-7874182.5, -7874179.5], "y": [5325451.5, 5325448.5]},
    )
    if array_crs:
        ds = ds.rio.write_crs(array_crs)
    monkeypatch.setattr(main, "run", lambda *args, **kwargs: ds)
    client = TestClient(main.app)
    bbox = [-70.735, 43.049, -70.681, 43.0885]
    if method == "get":
        response = client.get("/fuse", params={"bbox": ",".join(map(str, bbox)), "resolution": 3})
    else:
        response = client.post("/fuse", json={"bbox": bbox, "resolution": 3, "format": "geotiff"})

    assert response.status_code == 200, response.text
    with MemoryFile(response.content) as mem, mem.open() as src:
        assert src.crs == CRS.from_string(array_crs or policy.crs)
        assert src.transform == Affine(3, 0, -7874184, 0, -3, 5325453)
        assert src.nodata is not None and np.isnan(src.nodata)
        np.testing.assert_array_equal(src.read(1), values)
        np.testing.assert_array_equal(src.read(1, masked=True).mask, np.isnan(values))

    assert ds["elevation"].rio.crs == (CRS.from_string(array_crs) if array_crs else None)
    assert ds["elevation"].rio.nodata is None
