"""
NOAA Index Builder Script.

This script scans NOAA's data repositories (S3 and InPort) to construct a localized
GeoJSON index of available Topobathymetric surveys, including their bounding boxes,
CRS, and resolution metadata for faster runtime querying.
"""

import concurrent.futures
import logging
from pathlib import Path
from typing import Any

import fsspec
import geopandas as gpd
from shapely.geometry import box

from ..config import get_cache_root
from ..providers.noaa_topobathy import NoaaTopobathyProvider

# Setup logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("s3fs").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

CACHE_DIR = get_cache_root()
METADATA_DIR = CACHE_DIR / "metadata"
INDEX_PATH = METADATA_DIR / "noaa_project_extents.geojson"

BUCKET_BASE = "noaa-nos-coastal-lidar-pds"
# S3 FileSystem
try:
    FS = fsspec.filesystem("s3", anon=True)
except Exception:
    FS = None  # Handle offline/error gracefully


def get_project_tile_index_url(project_id: str, project_name: str) -> str | None:
    """
    Locates the tile index file for a project on S3.
    Checking laz/geoid18 first (preferred), then laz/geoid12b, then dem/{project_name}.
    """
    candidates = [f"laz/geoid18/{project_id}/", f"laz/geoid12b/{project_id}/", f"dem/{project_name}/"]

    if not FS:
        return None

    for prefix in candidates:
        try:
            # List files in the directory
            path_to_list = f"{BUCKET_BASE}/{prefix}"
            # logger.debug(f"Checking {path_to_list}")
            try:
                files = FS.ls(path_to_list)
            except FileNotFoundError:
                # logger.debug(f"[{project_id}] Path not found: {path_to_list}")
                continue

            if not files:
                logger.debug(f"[{project_id}] Empty directory: {path_to_list}")
                continue

            for f in files:
                name = Path(f).name.lower()
                if "tileindex" in name and (
                    name.endswith(".gpkg") or name.endswith(".zip") or name.endswith(".shp")
                ):
                    logger.debug(f"[{project_id}] Found index in {prefix}: {f}")
                    return str(f)

            # logger.debug(f"[{project_id}] No tileindex found in {path_to_list}, Files: {len(files)}")

        except Exception as e:
            # Too noisy for 491 projects if most fail, but valuable for debug now
            logger.debug(f"[{project_id}] Error listing {path_to_list}: {e}")
            continue

    return None


def process_project(
    project_id: str, project_name: str, provider: NoaaTopobathyProvider
) -> dict[str, Any] | None:
    """
    Worker function to process a single project:
    1. Find tile index
    2. Download/Cache it
    3. Load & Extract Bounds/CRS
    4. Fetch InPort Metadata
    """
    try:
        remote_path = get_project_tile_index_url(project_id, project_name)
        if not remote_path:
            # logger.debug(f"[{project_id}] No tile index found.")
            return None

        # Download to local cache
        local_filename = Path(remote_path).name
        # Differentiate by ID to avoid collisions if names are generic
        local_path = provider.cache_dir / f"{project_id}_{local_filename}"

        if not local_path.exists():
            logger.info(f"[{project_id}] Downloading index {remote_path}...")
            FS.get(remote_path, str(local_path))

        # Load Geometry
        try:
            gdf = gpd.read_file(local_path)
            if gdf.empty:
                logger.warning(f"[{project_id}] Empty tile index.")
                return None

            # Extract total bounds (minx, miny, maxx, maxy)
            bounds = gdf.total_bounds

            crs_str = "EPSG:4326"
            if gdf.crs:
                # If not 4326, reproject bbox to 4326 for global index
                # We need the geometry to be in 4326
                # But to get bounds, we can reproject the whole GDF or just the box?
                # Better to reproject the bbox if we only store the box.
                pass

            # Create box in original CRS
            bbox_geom = box(*bounds)

            # Reproject to 4326 if needed
            if gdf.crs and gdf.crs.to_string() != "EPSG:4326":
                # Create a temporary GDF with just the bounding box
                bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_geom], crs=gdf.crs)
                bbox_gdf_4326 = bbox_gdf.to_crs("EPSG:4326")
                bbox_geom_4326 = bbox_gdf_4326.geometry[0]
                crs_str = gdf.crs.to_string()
            else:
                bbox_geom_4326 = bbox_geom

            # Metadata
            meta = provider.fetch_inport_metadata(project_id) or {}

            # Extract Resolution from Meta
            # resolution_meters = None # Unused

            # Note: The `meta` dict handles parsing the XML tags such as <spatial-resolution>
            # The resolution extraction logic is implemented in `NoaaTopobathyProvider._parse_inport_xml`.

            return {
                "project_id": str(project_id),
                "project_name": project_name,
                "geometry": bbox_geom_4326,
                "original_crs": crs_str,
                "s3_index_path": remote_path,
                "start_date": meta.get("start_date"),
                "end_date": meta.get("end_date"),
                "vertical_datum": meta.get("vertical_datum"),
                "sensor_name": meta.get("sensor_name"),
                "resolution_meters": meta.get("resolution_meters"),
                "is_topobathy": meta.get("is_topobathy", False)
                or "topobathy" in project_name.lower()
                or "bathymetr" in project_name.lower(),
                "metadata_url": provider._projects_metadata_urls.get(project_id),
            }

        except Exception as e:
            logger.error(f"[{project_id}] Error reading index file {local_path}: {e}")
            return None

    except Exception as e:
        logger.error(f"[{project_id}] Unexpected error: {e}")
        return None


def main() -> None:
    """
    Main entry point for building the NOAA spatial index.
    Orchestrates the discovery and parallel processing of projects.
    """
    logger.info("Starting NOAA Spatial Index Builder...")

    provider = NoaaTopobathyProvider()
    # Populate existing project list from HTML/JSON
    provider._ensure_project_list()

    projects = provider._projects  # Dict[ID, Name]

    # DEBUG: Filter for select IDs to debug
    # debug_ids = {"10", "10274", "31393"}
    # filtered = {k: v for k, v in projects.items() if k in debug_ids}
    # if not filtered:
    # If not in list, manually inject "10" to test
    #     filtered = {"10": "TestProject10", "10274": "LIS_Test"}

    # projects = filtered
    # logger.info(f"Processing debug subset: {list(projects.keys())}")
    logger.info(f"Processing all {len(projects)} projects...")

    results = []

    # Use ThreadPoolExecutor
    # S3 rate limits might be an issue, stick to 4-8 workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_to_pid = {
            executor.submit(process_project, pid, pname, provider): pid for pid, pname in projects.items()
        }

        total = len(projects)

        for completed_count, future in enumerate(concurrent.futures.as_completed(future_to_pid), start=1):
            pid = future_to_pid[future]
            if completed_count % 10 == 0:
                logger.info(f"Progress: {completed_count}/{total}")

            try:
                res = future.result()
                if res:
                    results.append(res)
            except Exception as exc:
                logger.error(f"[{pid}] Generated exception: {exc}")

    logger.info(f"Successfully processed {len(results)} valid project indices.")

    if not results:
        logger.error("No valid project indices found. Aborting save.")
        return

    # Create Main GeoDataFrame
    gdf_main = gpd.GeoDataFrame(results, crs="EPSG:4326")

    # Save
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving index to {INDEX_PATH}...")
    try:
        gdf_main.to_file(INDEX_PATH, driver="GeoJSON")
        logger.info("Done.")
    except Exception as e:
        logger.error(f"Failed to save GeoJSON: {e}")


if __name__ == "__main__":
    main()
