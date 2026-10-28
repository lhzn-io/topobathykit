import math
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from fastapi.testclient import TestClient
from rasterio.crs import CRS
from rasterio.io import MemoryFile
from topobathyserve import main

from topobathykit.policy.schema import FusionPolicy, VariableStrategy

Z, X, Y = 12, 1243, 1495  # covers -70.7, 43.07


def _tile_bounds_3857(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    r = 6378137.0
    n = 2.0**z
    west = x / n * 2 * math.pi * r - math.pi * r
    east = (x + 1) / n * 2 * math.pi * r - math.pi * r
    lat_s = math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n)))
    lat_n = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    south = r * math.log(math.tan(math.pi / 4 + lat_s / 2))
    north = r * math.log(math.tan(math.pi / 4 + lat_n / 2))
    return west, south, east, north


@pytest.mark.parametrize("array_crs", ["EPSG:3857", None])
def test_tile_geotiff_tags_crs_and_nodata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, array_crs: str | None
) -> None:
    policy = FusionPolicy(
        name="Tile GeoTIFF metadata",
        crs="EPSG:3857",
        variables=[VariableStrategy(name="elevation", steps=[])],
    )
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(policy.model_dump_json())
    monkeypatch.setitem(main.app.dependency_overrides, main.get_policy_path, lambda: policy_path)
    monkeypatch.setenv("TOPOBATHYKIT_CACHE_DIR", str(tmp_path / "cache"))

    west, south, east, north = _tile_bounds_3857(Z, X, Y)
    size = 16
    dx, dy = (east - west) / size, (north - south) / size
    # One cell of margin, as the fetch halo gives, so the resample stays inside the data
    n = size + 2
    values = np.full((n, n), -4.0, dtype=np.float32)
    values[:, : n // 2] = np.nan  # land half of the tile has no data
    ds = xr.Dataset(
        {"elevation": (("y", "x"), values)},
        coords={
            "x": np.linspace(west - dx / 2, east + dx / 2, n),
            "y": np.linspace(north + dy / 2, south - dy / 2, n),
        },
    )
    if array_crs:
        ds = ds.rio.write_crs(array_crs)
    monkeypatch.setattr(main, "run", lambda *args, **kwargs: ds)

    # A tile cached before tiles carried these tags must not be served
    _, param_hash = main._tile_cache_key(policy_path, None, None, size)
    legacy = tmp_path / "cache" / "tiles" / "raw" / str(Z) / str(X) / f"{Y}_{param_hash}.tif"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"untagged")

    client = TestClient(main.app)
    for _cached in (False, True):
        response = client.get(f"/tiles/{Z}/{X}/{Y}.tif", params={"tile_size": size})
        assert response.status_code == 200, response.text
        assert response.headers["content-type"] == "image/tiff"
        with MemoryFile(response.content) as mem, mem.open() as src:
            assert src.crs == CRS.from_string("EPSG:3857")
            assert src.nodata is not None and np.isnan(src.nodata)
            band = src.read(1, masked=True)
            assert band.mask[:, 0].all()
            assert not band.mask[:, -1].any()
            np.testing.assert_allclose(band[:, -1], -4.0)

    assert ds["elevation"].rio.nodata is None
