"""
Tests for resampling provider grids onto the runtime canvas.

Background: `_run_cell` used to align every provider with `reproject_match` and its
default nearest resampling, and asked providers for `resolution` ground metres while
an EPSG:3857 canvas cell of `resolution` nominal metres spans only about 0.75 of that
on the ground at 41N. The source was then about 1.33x coarser than the canvas, and
nearest duplicated roughly one row and column in four.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr
import yaml
from affine import Affine
from pyproj import Transformer
from rasterio.enums import Resampling

import topobathykit.runtime as runtime
from topobathykit.providers.base import (
    METERS_PER_DEG_LAT,
    METERS_PER_DEG_LON_EQUATOR,
    Provider,
    align_to_grid,
    choose_resampling,
    ground_resolution_m,
)
from topobathykit.providers.registry import registry

# One standard 0.05 degree cell at 41N.
CELL_BBOX = runtime._get_grid_cells((-73.49, 40.91, -73.46, 40.94), "EPSG:3857")[0][0]
RES_M = 30.0
# Western limit of the "west half" provider's coverage, inside the cell.
COVERAGE_EDGE_LON = -73.475


def _field(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """A smooth surface whose value changes between any two distinct pixels."""
    return 2000.0 * (lon + 73.5) + 3000.0 * (lat - 40.9) + 0.5 * np.sin(400.0 * lon) * np.cos(300.0 * lat)


def _geographic_grid(
    bbox: tuple[float, float, float, float], ground_m: float, mid_lat: float
) -> xr.DataArray:
    """EPSG:4326 grid over `bbox` whose cells are `ground_m` metres on the ground."""
    west, south, east, north = bbox
    res_x = ground_m / (METERS_PER_DEG_LON_EQUATOR * np.cos(np.radians(mid_lat)))
    res_y = ground_m / METERS_PER_DEG_LAT
    nx = int(np.ceil((east - west) / res_x))
    ny = int(np.ceil((north - south) / res_y))
    xs = west + (np.arange(nx) + 0.5) * res_x
    ys = north - (np.arange(ny) + 0.5) * res_y
    lon, lat = np.meshgrid(xs, ys)
    da = xr.DataArray(_field(lon, lat).astype(np.float32), coords={"y": ys, "x": xs}, dims=("y", "x"))
    da.rio.write_crs("EPSG:4326", inplace=True)
    da.rio.write_transform(Affine.translation(west, north) @ Affine.scale(res_x, -res_y), inplace=True)
    return da


class _GroundResolutionSurface(Provider):
    """Returns the smooth field on an EPSG:4326 grid at the requested ground resolution."""

    def fetch_layer(self, bbox: Any, resolution: Any = None, crs: str = "EPSG:4326", **kw: Any) -> Any:
        mid_lat = (bbox[1] + bbox[3]) / 2.0
        return _geographic_grid(bbox, float(resolution), mid_lat)

    def get_metadata(self) -> dict[str, Any]:
        return {"name": "surface"}


class _WestHalfSurface(_GroundResolutionSurface):
    """The same field, NaN east of COVERAGE_EDGE_LON."""

    def fetch_layer(self, bbox: Any, resolution: Any = None, crs: str = "EPSG:4326", **kw: Any) -> Any:
        da = super().fetch_layer(bbox, resolution, crs, **kw)
        return da.where(da.x < COVERAGE_EDGE_LON)


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOPOBATHYKIT_CACHE_DIR", str(tmp_path / "cache"))
    registry.register("it_surface", _GroundResolutionSurface)
    registry.register("it_west_surface", _WestHalfSurface)


def _policy(tmp_path: Path, provider: str) -> Any:
    content = {
        "name": "AlignmentTest",
        "crs": "EPSG:3857",
        "variables": [{"name": "elevation", "steps": [{"provider": provider, "operation": "overwrite"}]}],
    }
    path = tmp_path / f"{provider}.yaml"
    path.write_text(yaml.dump(content))
    return runtime._resolve_policy(str(path))


def _duplicate_fraction(values: np.ndarray, axis: int) -> float:
    """Fraction of adjacent row (axis 0) or column (axis 1) pairs that are exactly equal."""
    a = np.moveaxis(values, axis, 0)
    pairs = 0
    dupes = 0
    for i in range(a.shape[0] - 1):
        x, y = a[i], a[i + 1]
        both = np.isfinite(x) & np.isfinite(y)
        if both.sum() < 3:
            continue
        pairs += 1
        dupes += bool(np.array_equal(x[both], y[both]))
    return dupes / pairs if pairs else 0.0


def _canvas_lonlat(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    xx, yy = np.meshgrid(ds.x.values, ds.y.values)
    to_geo = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    return to_geo.transform(xx, yy)


def test_ground_resolution_of_mercator_and_geographic_cells() -> None:
    lat = 41.0
    coslat = np.cos(np.radians(lat))
    merc = ground_resolution_m(Affine.scale(30.0, -30.0), "EPSG:3857", lat)
    assert merc == pytest.approx(30.0 * coslat)
    geo = ground_resolution_m(
        Affine.scale(30.0 / (METERS_PER_DEG_LON_EQUATOR * coslat), -30.0 / METERS_PER_DEG_LAT),
        "EPSG:4326",
        lat,
    )
    assert geo == pytest.approx(30.0)


def test_choose_resampling_thresholds() -> None:
    assert choose_resampling(30.0, 30.0) == Resampling.bilinear
    assert choose_resampling(60.0, 30.0) == Resampling.bilinear  # source coarser
    assert choose_resampling(25.0, 30.0) == Resampling.bilinear  # 1.2x finer
    assert choose_resampling(20.0, 30.0) == Resampling.average  # 1.5x finer
    assert choose_resampling(1.0, 30.0) == Resampling.average


def test_mercator_canvas_has_no_duplicated_rows_or_columns(tmp_path: Path) -> None:
    ds = runtime._run_cell(_policy(tmp_path, "it_surface"), CELL_BBOX, RES_M, None, "EPSG:3857")
    values = ds["elevation"].values
    assert np.isfinite(values).all()
    assert _duplicate_fraction(values, axis=0) < 0.01
    assert _duplicate_fraction(values, axis=1) < 0.01

    # The canvas follows the field: no stair-steps, values within a pixel's change.
    lon, lat = _canvas_lonlat(ds)
    expected = _field(lon, lat)
    step = 3000.0 * RES_M / METERS_PER_DEG_LAT  # field change across one pixel, north-south
    assert np.nanmax(np.abs(values - expected)) < step


def test_nearest_at_nominal_resolution_duplicates_rows() -> None:
    """The metric above detects the old behaviour: nearest from a 1.33x coarser source."""
    mid_lat = (CELL_BBOX[1] + CELL_BBOX[3]) / 2.0
    source = _geographic_grid(CELL_BBOX, RES_M, mid_lat)  # RES_M ground metres
    canvas = source.rio.reproject("EPSG:3857", resolution=RES_M, resampling=Resampling.nearest)
    assert _duplicate_fraction(canvas.values, axis=0) > 0.1


def test_no_data_does_not_bleed_at_coverage_edges(tmp_path: Path) -> None:
    ds = runtime._run_cell(_policy(tmp_path, "it_west_surface"), CELL_BBOX, RES_M, None, "EPSG:3857")
    values = ds["elevation"].values
    lon, lat = _canvas_lonlat(ds)
    pixel_deg = RES_M / (METERS_PER_DEG_LON_EQUATOR)  # canvas cell width in degrees of longitude

    # East of the edge (beyond half a canvas cell) everything stays no-data.
    assert np.isnan(values[lon > COVERAGE_EDGE_LON + pixel_deg / 2]).all()
    # West of it (beyond one cell) everything is covered.
    assert np.isfinite(values[lon < COVERAGE_EDGE_LON - pixel_deg]).all()
    # Valid pixels next to the edge are not pulled towards a fill value.
    expected = _field(lon, lat)
    step = 3000.0 * RES_M / METERS_PER_DEG_LAT
    valid = np.isfinite(values)
    assert np.abs(values[valid] - expected[valid]).max() < step


def test_average_resampling_masks_mostly_empty_cells() -> None:
    """A source 4x finer than the target is averaged; cells under half covered are no-data."""
    mid_lat = 40.95
    bbox = (-73.50, 40.90, -73.45, 40.95)
    source = _geographic_grid(bbox, 7.5, mid_lat)
    src_res_x = float(source.rio.transform().a)
    # Each 30 m target column holds exactly four 7.5 m source columns. Put the edge
    # so that target columns are fully, half (2 of 4) and a quarter (1 of 4) covered.
    edges = {"half": bbox[0] + (40 * 4 + 2) * src_res_x, "quarter": bbox[0] + (40 * 4 + 1) * src_res_x}

    target_res_x = 4 * src_res_x
    target_res_y = 4 * float(-source.rio.transform().e)
    transform = Affine.translation(bbox[0], bbox[3]) @ Affine.scale(target_res_x, -target_res_y)
    shape = (60, 60)

    for kind, edge in edges.items():
        masked = source.where(source.x < edge)
        vals = align_to_grid(masked, "EPSG:4326", transform, shape, Resampling.average).values
        assert np.isfinite(vals[:, :40]).all(), kind
        assert np.isnan(vals[:, 41:]).all(), kind
        # Column 40 is half covered (kept, at the 0.5 threshold) or a quarter covered (no-data).
        if kind == "half":
            assert np.isfinite(vals[:, 40]).all()
            # Its value is the mean of the two valid source columns, not diluted by no-data.
            src_cols = masked.values[: shape[0] * 4, 160:162]
            expected = src_cols.reshape(shape[0], 4, 2).mean(axis=(1, 2))
            np.testing.assert_allclose(vals[:, 40], expected, rtol=0, atol=1e-3)
        else:
            assert np.isnan(vals[:, 40]).all()
