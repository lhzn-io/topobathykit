import logging
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import geopandas as gpd
import pytest
from shapely.geometry import box

from topobathykit.providers.noaa_topobathy import NoaaTopobathyProvider

# Configure logger
logger = logging.getLogger(__name__)


@pytest.fixture
def temp_cache_dir(tmp_path: Path) -> Path:
    """Provide a temporary cache directory for isolated tests."""
    cache_dir = tmp_path / "topobathykit_cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture(autouse=True)
def reset_singleton() -> Iterator[None]:
    """Ensure each test gets a fresh NoaaTopobathyProvider instance."""
    NoaaTopobathyProvider._singleton = None
    yield
    NoaaTopobathyProvider._singleton = None


@pytest.mark.integration
def test_topobathy_index_cache_hit_prevents_network(
    temp_cache_dir: Path, caplog: Any, monkeypatch: Any
) -> None:
    """
    Verify that if a local tile index exists, NoaaTopobathyProvider:
    1. Loads it via geopandas.
    2. Does NOT attempt to list S3 buckets.
    3. Does NOT attempt to download the index.
    """
    # Scope to the package logger, not just root: importing the FastAPI app
    # (service/topobathyserve/main.py) pins the "topobathykit" logger to INFO at
    # import time, which drops the provider's DEBUG cache-hit records before
    # they reach caplog when this test runs after test_api_caching.
    caplog.set_level(logging.DEBUG, logger="topobathykit")

    project_id = "TEST_PROJECT_1234"
    folder_name = "test_folder_1234"

    # 1. Initialize Provider (fresh singleton via fixture)
    # Ensure we use the temp_cache_dir
    provider = NoaaTopobathyProvider(cache_dir=str(temp_cache_dir))

    # 2. Inject fake project into the class-level shared dictionary
    # This prevents _ensure_project_list from trying to fetch the HTML index from NOAA
    NoaaTopobathyProvider._cls_projects[project_id] = folder_name

    # 3. Create a dummy cached ZIP index file
    # This simulates a previous run where the index was already downloaded
    # Correct path: provider.metadata_dir / "tile_index" / project_id.zip
    # provider.metadata_dir = base_cache_dir / "metadata"
    # IMPORTANT: The provider looks for files named by PROJECT ID, not folder name!
    metadata_dir = temp_cache_dir / "metadata" / "tile_index"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    zip_path = metadata_dir / f"{project_id}.zip"

    # Create a dummy valid zip file
    import zipfile

    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(f"{folder_name}.shp", "DUMMY SHAPEFILE CONTENT")
        zf.writestr(f"{folder_name}.shx", "DUMMY SHX CONTENT")
        zf.writestr(f"{folder_name}.dbf", "DUMMY DBF CONTENT")

    # Debug logs
    logger = logging.getLogger("test_debug")
    logger.info(f"Test Zip Path: {zip_path}")
    logger.info(f"Zip exists: {zip_path.exists()}")
    logger.info(f"Provider Metadata Dir: {provider.metadata_dir}")
    logger.info(f"Provider project list: {provider._projects}")

    # 4. Mock geopandas.read_file to return a dummy GDF without reading the bad files
    def mock_read_file(filepath: Any, **kwargs: Any) -> Any:
        # Verify we are reading the ZIP we just created
        assert str(filepath) == str(zip_path) or f"zip://{zip_path}" in str(filepath)
        import geopandas as gpd
        from shapely.geometry import box

        return gpd.GeoDataFrame({"tile_url": ["http://fake/tile1.laz"]}, geometry=[box(0, 0, 1, 1)])

    monkeypatch.setattr("geopandas.read_file", mock_read_file)

    # 5. Execute: Set active project
    # This should trigger the logic that checks for ZIP existence
    provider.set_active_project(project_id)

    # 6. Verify Logs & Behavior
    # We expect:
    # - the disk cache-hit record for this project
    # We DO NOT expect:
    # - "Fetching NOAA Coastal Lidar PDS Index" (prevented by _cls_projects injection)
    # - "S3 List Objects" or filesystem calls (prevented by ZIP check logic)

    log_text = caplog.text

    assert f"Topobathy Tile Index Cache Hit (Disk): {project_id}" in log_text
    assert "Fetching NOAA Coastal Lidar PDS Index" not in log_text

    # Ensure no network calls were made (caplog is good, but we can also assert on specific mocked methods)
    # Since we didn't mock requests/s3fs, strictly speaking they *could* handle it, but if they did,
    # the test inevitably fails or hangs or logs errors if credentials/network are real.
    # The absence of "Listing S3" logs is our key indicator if we added logging there,
    # but the provider currently uses fsspec.

    # --- Second scenario: a real GPKG index, read by the real geopandas. ---
    # Undo the read_file mock (it asserts the path is the zip) and remove the
    # zip so the provider's glob finds only the GPKG; clear the log buffer so
    # the assertions below reflect this scenario alone.
    monkeypatch.undo()
    zip_path.unlink()
    caplog.clear()

    # Use a distinct project id. The first scenario left TEST_PROJECT_1234 as the
    # provider's active project with a loaded index, and set_active_project
    # returns early for an already-active project, which would skip the disk
    # lookup this scenario is meant to exercise.
    gpkg_project_id = "TEST_PROJECT_GPKG_1234"

    # 1. Setup Cache Structure
    metadata_dir = temp_cache_dir / "metadata" / "tile_index"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # 2. Create a dummy Index File (GPKG)
    # Use a simple box as the tile coverage
    geometry = [box(-74.0, 40.0, -73.0, 41.0)]
    df = gpd.GeoDataFrame({"filename": ["test_tile_1.laz"]}, geometry=geometry, crs="EPSG:4326")

    index_path = metadata_dir / f"{gpkg_project_id}.gpkg"
    df.to_file(index_path, driver="GPKG")

    # 3. Instantiate Provider
    provider = NoaaTopobathyProvider(cache_dir=str(temp_cache_dir))

    # Pre-seed the project list so it doesn't try to fetch index.html or fail validation
    provider._projects = {gpkg_project_id: "Test_Project_Folder"}

    # 4. Trigger set_active_project
    # This should find the local gpkg and return early
    provider.set_active_project(gpkg_project_id)

    # 5. Verify Behavior via Logs

    # Positive Assertions
    assert f"Topobathy Tile Index Cache Hit (Disk): {gpkg_project_id}" in caplog.text

    # Negative Assertions (Proof of Fix)
    # The provider should NOT attempt S3 operations if cache is hit
    assert "Downloading Tile Index" not in caplog.text
    assert "Calling endpoint provider" not in caplog.text

    # Verify the index was actually loaded
    assert provider._tile_index is not None
    assert len(provider._tile_index) == 1
    assert provider._active_project_id == gpkg_project_id


@pytest.mark.integration
def test_topobathy_index_cache_hit_zip(temp_cache_dir: Path, caplog: Any) -> None:
    """
    Verify that NoaaTopobathyProvider correctly handles .zip files in the cache.
    (This was the specific failure mode where geopandas.read_file() was needed)
    """
    # See note in test_topobathy_index_cache_hit_prevents_network: scope to the
    # package logger so an earlier app import cannot suppress DEBUG records.
    caplog.set_level(logging.DEBUG, logger="topobathykit")

    project_id = "TEST_ZIP_PROJECT_5678"

    # 1. Setup Cache Structure
    metadata_dir = temp_cache_dir / "metadata" / "tile_index"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # 2. Create a dummy Shapefile and Zip it
    geometry = [box(-74.0, 40.0, -73.0, 41.0)]
    df = gpd.GeoDataFrame({"filename": ["test_tile_1.laz"]}, geometry=geometry, crs="EPSG:4326")

    # Save as shapefile in a temp dir first
    shp_temp = temp_cache_dir / "temp_shp"
    shp_temp.mkdir()
    shp_path = shp_temp / f"{project_id}.shp"
    df.to_file(shp_path)

    # Zip it up to the target location
    # shutil.make_archive creates a zip file without the extension input, so we pass base_name
    target_base = metadata_dir / project_id
    shutil.make_archive(str(target_base), "zip", shp_temp)

    # Verify zip exists
    zip_path = metadata_dir / f"{project_id}.zip"
    assert zip_path.exists()

    # 3. Instantiate Provider
    provider = NoaaTopobathyProvider(cache_dir=str(temp_cache_dir))

    # Pre-seed for test
    provider._projects = {project_id: "Test_Zip_Project_Folder"}

    # 4. Trigger set_active_project
    try:
        provider.set_active_project(project_id)
    except Exception as e:
        pytest.fail(f"set_active_project raised exception on zip file: {e}")

    # 5. Verify Behavior
    if "Downloading Tile Index" in caplog.text:
        pytest.fail("Provider attempted to download index despite local .zip existing!")

    assert f"Topobathy Tile Index Cache Hit (Disk): {project_id}" in caplog.text
    assert provider._tile_index is not None
    assert len(provider._tile_index) == 1
    assert provider._active_project_id == project_id
    assert len(provider._tile_index) == 1
