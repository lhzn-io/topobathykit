"""
Tests that a NOAA topobathy tile index's whole-coverage file gives way to its tiles.

Background (#45): the New England CoNED tile index lists one whole-coverage COG beside
the tiles cut from it. The whole file sorted first, so every fetch over the project
read from it, and a failed ranged read deep into it left GDAL's curl state broken.
"""

from pathlib import Path
from typing import Any

import geopandas as gpd
import pytest
from shapely.geometry import box

# A 4 x 4 grid of 0.1 degree tiles over (-72.4, 41.0, -72.0, 41.4), plus one
# whole-coverage row extending well beyond them.
TILES = {
    f"tile_{i}_{j}.tif": (-72.4 + 0.1 * i, 41.0 + 0.1 * j, -72.3 + 0.1 * i, 41.1 + 0.1 * j)
    for i in range(4)
    for j in range(4)
}
WHOLE = ("Whole_Coverage_DEM.tif", (-74.0, 40.0, -70.0, 43.0))


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOPOBATHYKIT_CACHE_DIR", str(tmp_path / "cache"))


def _provider(tmp_path: Path, rows: dict[str, tuple[float, float, float, float]]) -> Any:
    from topobathykit.providers.noaa_topobathy import NoaaTopobathyProvider

    NoaaTopobathyProvider._singleton = None
    NoaaTopobathyProvider._cls_tile_indices.clear()
    provider = NoaaTopobathyProvider(cache_dir=str(tmp_path / "cache"))
    names = list(rows)
    NoaaTopobathyProvider._cls_tile_indices["111"] = gpd.GeoDataFrame(
        {"name": names}, geometry=[box(*rows[n]) for n in names], crs="EPSG:4326"
    )
    return provider


def test_tiles_replace_whole_coverage_file_inside_tiled_area(tmp_path: Path) -> None:
    provider = _provider(tmp_path, {WHOLE[0]: WHOLE[1], **TILES})
    tiles = provider.resolve_tiles_in_bbox(-72.25, 41.05, -72.05, 41.25, project_id="111")
    assert WHOLE[0] not in tiles
    assert tiles == sorted(
        n for n, b in TILES.items() if box(*b).intersects(box(-72.25, 41.05, -72.05, 41.25))
    )


def test_whole_coverage_file_kept_where_tiles_leave_a_gap(tmp_path: Path) -> None:
    provider = _provider(tmp_path, {WHOLE[0]: WHOLE[1], **TILES})
    # Extends 0.05 degrees east of the tiled area.
    tiles = provider.resolve_tiles_in_bbox(-72.15, 41.15, -71.95, 41.25, project_id="111")
    assert WHOLE[0] in tiles


def test_whole_coverage_file_kept_with_missing_tile(tmp_path: Path) -> None:
    rows = {WHOLE[0]: WHOLE[1], **TILES}
    del rows["tile_2_2.tif"]
    provider = _provider(tmp_path, rows)
    tiles = provider.resolve_tiles_in_bbox(-72.25, 41.05, -72.05, 41.25, project_id="111")
    assert WHOLE[0] in tiles


def test_index_of_similar_large_files_is_unchanged(tmp_path: Path) -> None:
    # Statewide files of comparable size (as in the SLR viewer index) are all tiles.
    rows = {
        "DE.tif": (-75.8, 38.4, -75.0, 39.9),
        "NJ.tif": (-75.6, 38.9, -73.9, 41.4),
        "NY.tif": (-74.3, 40.5, -71.8, 41.3),
    }
    provider = _provider(tmp_path, rows)
    tiles = provider.resolve_tiles_in_bbox(-74.2, 40.6, -74.0, 40.8, project_id="111")
    assert tiles == ["NJ.tif", "NY.tif"]
