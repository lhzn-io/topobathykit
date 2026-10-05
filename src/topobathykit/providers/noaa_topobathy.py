import contextlib
import logging
import re
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar, cast

import fsspec
import geopandas as gpd
import requests  # type: ignore
import rioxarray
import xarray as xr
from rioxarray.exceptions import NoDataInBounds, OneDimensionalRaster
from rioxarray.merge import merge_arrays
from shapely.geometry import box

from ..config import get_cache_root
from ..runtime import should_consolidate
from ..vdatum import VDatumNoDataError, VDatumResolver
from .base import Provider, ProviderFetchError, ProviderNoDataError, sanitize_elevation_nodata
from .registry import registry

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _fetch_index_cached(url: str) -> str:
    """Cached fetch of the index page (Standalone)."""
    logger.info("Fetching NOAA Coastal Lidar PDS Index...")
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return str(response.text)


class NoaaTopobathyProvider(Provider):
    """
    Provider for NOAA Topobathymetric LiDAR DEMs (Tier 0).
    Distinguished from Terrestrial Lidar (USGS 3DEP) by its use of green-wavelength
    lasers that penetrate the water column to capture submerged topography.

    Bucket: s3://noaa-nos-coastal-lidar-pds/
    """

    INDEX_URL = "https://noaa-nos-coastal-lidar-pds.s3.amazonaws.com/laz/index.html"
    BUCKET_BASE = "noaa-nos-coastal-lidar-pds"

    # Singleton instance to avoid re-initializing S3 filesystem and discovery caches
    _singleton: ClassVar["NoaaTopobathyProvider | None"] = None

    # Class-level in-process cache.
    #
    # run() creates one NoaaTopobathyProvider instance per grid cell (up to 9x
    # per tile request).  Using class-level variables instead of instance-level ones
    # means the project list and spatial index are loaded from disk only once per
    # process instead of nine times.
    #
    # Disk backing files (under ~/.cache/topobathykit/metadata/):
    #   noaa_coastal_lidar.json   - project ID -> folder name mapping (fetched from S3 HTML)
    #   noaa_project_extents.geojson - spatial bbox index (built from project list)
    #
    # To discover newly published topo-bathy lidar projects, delete those files:
    #   manage_discovery_cache.py --invalidate noaa_topobathy
    _cls_projects: ClassVar[dict[str, str]] = {}
    _cls_projects_metadata_urls: ClassVar[dict[str, str]] = {}
    _cls_spatial_index: ClassVar["gpd.GeoDataFrame | None"] = None
    _cls_tile_indices: ClassVar[dict[str, "gpd.GeoDataFrame"]] = {}
    _cls_lock: ClassVar[threading.Lock] = threading.Lock()
    # Serialises tile index loads so concurrent cells load each project's index once.
    _cls_index_lock: ClassVar[threading.Lock] = threading.Lock()
    _initialized: bool

    def __new__(cls, *args: Any, **kwargs: Any) -> "NoaaTopobathyProvider":
        if cls._singleton is None:
            cls._singleton = super().__new__(cls)
        return cast(NoaaTopobathyProvider, cls._singleton)

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        """
        Initializes the Topobathy provider, creating necessary cache directories.
        """
        if getattr(self, "_initialized", False):
            return

        if cache_dir is None:
            cache_dir = get_cache_root()

        self.base_cache_dir = Path(cache_dir).expanduser()
        self.cache_dir = self.base_cache_dir / "noaa_topobathy"
        self.metadata_dir = self.base_cache_dir / "metadata"
        self.inport_cache_dir = self.metadata_dir / "inport"

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "zarr").mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.inport_cache_dir.mkdir(parents=True, exist_ok=True)

        self.vdatum = VDatumResolver()

        # Internal state — point to class-level shared dicts so all instances within
        # the same process share a single loaded copy (avoids 9x redundant disk reads
        # when run() creates one provider instance per cell).
        self._projects: dict[str, str] = NoaaTopobathyProvider._cls_projects
        self._projects_metadata_urls: dict[str, str] = NoaaTopobathyProvider._cls_projects_metadata_urls

        self._active_project_id: str | None = None
        self.fs = fsspec.filesystem("s3", anon=True)
        self._initialized = True

    @staticmethod
    def _sanitize_nodata_values(da: xr.DataArray) -> xr.DataArray:
        """Mask known NOAA nodata sentinels so they cannot leak into fusion outputs."""
        return cast(
            xr.DataArray,
            sanitize_elevation_nodata(
                da,
                extra_sentinels=(-999999.0,),
                min_valid=-999000,
                abs_valid_limit=1000000.0,
            ),
        )

    @property
    def _tile_index(self) -> "gpd.GeoDataFrame | None":
        """Transparently reads from the class-level shared tile index cache."""
        if self._active_project_id is None:
            return None
        return NoaaTopobathyProvider._cls_tile_indices.get(self._active_project_id)

    @_tile_index.setter
    def _tile_index(self, value: "gpd.GeoDataFrame | None") -> None:
        """Writes to the class-level shared tile index cache."""
        if self._active_project_id is not None:
            if value is None:
                NoaaTopobathyProvider._cls_tile_indices.pop(self._active_project_id, None)
            else:
                NoaaTopobathyProvider._cls_tile_indices[self._active_project_id] = value

    @property
    def _spatial_index(self) -> "gpd.GeoDataFrame | None":
        """Transparently reads from the class-level shared spatial index."""
        return NoaaTopobathyProvider._cls_spatial_index

    @_spatial_index.setter
    def _spatial_index(self, value: "gpd.GeoDataFrame | None") -> None:
        """Writes to the class-level shared spatial index."""
        NoaaTopobathyProvider._cls_spatial_index = value

    def fetch_layer(
        self,
        bbox: tuple[float, float, float, float],
        resolution: float | None = None,
        crs: str = "EPSG:4326",
        **kwargs: Any,
    ) -> xr.DataArray | xr.Dataset:
        """
        Fetches and merges Topobathy data from ALL overlapping projects.
        Prioritizes better datasets (Name/Datum/Recency) by layering them on top.
        """
        import numpy as np

        # Ensure the bbox is in EPSG:4326 for project discovery
        west, south, east, north = self._normalize_bbox(bbox, crs)

        # Extract filter from kwargs
        filter_criteria = kwargs.get("filter")

        # 1. Identify Projects
        # Returns sorted list: [Best, ..., Worst]
        pids = self.find_projects_by_box(west, south, east, north, filter_criteria=filter_criteria)
        if not pids:
            # Try to report what happened
            raise ProviderNoDataError(f"No NOAA Topobathy projects found for bbox {bbox}")

        logger.info(f"Topobathy fetch: Found {len(pids)} overlapping projects. Processing...")

        project_layers = []
        provenance_dict: dict[int, dict[str, str]] = {}
        # Tiles or projects lost to errors (as opposed to no coverage). The runtime
        # refuses to cache a cell when this is non-zero, so the loss is retried.
        fetch_errors: list[str] = []

        # 2. Process Each Project
        for pid in pids:
            try:
                # OPTIMIZATION: Check if all expected tiles for this project are already in Zarr cache.
                # If they are, we can skip the S3 directory listing in set_active_project().
                # However, since tiles are dynamic per bbox, we still need the tile index.
                # BUT, we can cache the tile index in memory (done in _load_tile_index).

                # Load and query this project's index by explicit pid. The shared
                # _active_project_id is not used here: concurrent cells on the
                # singleton would otherwise resolve tiles against each other's project.
                self._load_tile_index(pid)

                # Resolve Tiles
                tiles = self.resolve_tiles_in_bbox(west, south, east, north, project_id=pid)
                if not tiles:
                    continue

                # Fetch & Merge Tiles for this Project
                project_das = []
                for t in tiles:
                    try:
                        # Optimization: Pass bbox to fetch_tile to enable "Clip-then-Cache"
                        da_fetched = self.fetch_tile(t, bbox=bbox, project_id=pid, raise_errors=True)
                        if da_fetched is None:
                            continue

                        da_fetched = self._sanitize_nodata_values(da_fetched)

                        # Optimization: Clip Early
                        try:
                            # Use 4326 clip since index is 4326
                            da = da_fetched.rio.clip_box(
                                minx=west,
                                miny=south,
                                maxx=east,
                                maxy=north,
                                crs="EPSG:4326",
                                allow_one_dimensional_raster=True,
                            )
                        except (NoDataInBounds, OneDimensionalRaster):
                            # Empty after clip
                            continue

                        if da.size == 0:
                            continue

                        # Reproject if needed
                        if crs and da.rio.crs and da.rio.crs != crs:
                            try:
                                reproj_knn = {}
                                if resolution and "EPSG:4326" not in crs:
                                    reproj_knn["resolution"] = resolution
                                da = da.rio.reproject(crs, **reproj_knn)
                            except Exception as e:
                                logger.warning(f"Reprojection failed for tile {t}: {e}")
                                fetch_errors.append(f"{pid}/{t}: reproject {type(e).__name__}: {e}")
                                continue

                        project_das.append(da)
                    except Exception as e:
                        logger.warning(f"Failed to fetch tile {t}: {type(e).__name__}: {e}")
                        fetch_errors.append(f"{pid}/{t}: {type(e).__name__}: {e}")

                if project_das:
                    # Merge tiles for this project
                    # Default is last-on-top, but for a single project tiles shouldn't overlap much
                    try:
                        p_merged = merge_arrays(project_das)
                        p_merged = self._sanitize_nodata_values(p_merged)

                        # Extract Provenance info
                        try:
                            project_uid = int(pid) + 10000
                        except ValueError:
                            import hashlib

                            project_uid = int(hashlib.md5(pid.encode()).hexdigest(), 16) % 100000 + 10000

                        project_name = self._projects.get(pid, f"Project {pid}")

                        # Try to find resolution in spatial index
                        res_str = "Unknown"
                        if (
                            self._spatial_index is not None
                            and "resolution_meters" in self._spatial_index.columns
                        ):
                            try:
                                project_ids = self._spatial_index["project_id"].astype(str)
                                row = self._spatial_index[project_ids == str(pid)]
                                if not row.empty:
                                    val = row.iloc[0]["resolution_meters"]
                                    if val is not None:
                                        import numpy as np

                                        if not np.isnan(val):
                                            res_str = f"{float(val):.2f}m"
                            except Exception:
                                pass

                        provenance_dict[project_uid] = {
                            "name": project_name,
                            "provider": "noaa_topobathy",
                            "resolution": res_str,
                        }

                        p_merged.name = "elevation"
                        p_source = xr.where(p_merged.notnull(), project_uid, 0).astype(np.uint32)
                        p_source.name = "source_id"
                        p_source = p_source.rio.write_nodata(0)
                        p_source.attrs["_FillValue"] = 0

                        p_ds = xr.Dataset({"elevation": p_merged, "source_id": p_source})
                        # carry over spatial attributes
                        p_ds = p_ds.rio.write_crs(p_merged.rio.crs)
                        p_ds = p_ds.rio.write_transform(p_merged.rio.transform())

                        project_layers.append(p_ds)
                    except Exception as e:
                        logger.warning(f"Failed to merge tiles for project {pid}: {type(e).__name__}: {e}")
                        fetch_errors.append(f"{pid}: merge {type(e).__name__}: {e}")

            except Exception as e:
                logger.warning(f"Failed to process project {pid}: {type(e).__name__}: {e}")
                fetch_errors.append(f"{pid}: {type(e).__name__}: {e}")

        if not project_layers:
            if fetch_errors:
                # Coverage exists but every contributing tile or project failed; this
                # is not a genuine no-data result and must not be cached as one.
                raise ProviderFetchError(
                    f"NOAA Topobathy fetch failed for bbox {bbox} with {len(fetch_errors)} error(s); "
                    f"first: {fetch_errors[0]}"
                )
            raise ProviderNoDataError(f"No valid data returned from {len(pids)} projects for bbox {bbox}")
        if fetch_errors:
            logger.warning(
                f"NOAA Topobathy returned partial data for bbox {bbox}: {len(fetch_errors)} error(s); "
                f"first: {fetch_errors[0]}"
            )

        # 3. Merge Projects
        # project_layers is ordered [Best, ..., Worst]
        # rioxarray's merge_arrays defaults to method='first', so the FIRST element takes precedence.
        # Thus, passing the list directly [Best, ..., Worst] ensures Best is on top.
        elevs = [ds["elevation"] for ds in project_layers]
        sources = [ds["source_id"] for ds in project_layers]

        final_elev = merge_arrays(elevs)
        final_elev = self._sanitize_nodata_values(final_elev)
        final_src = merge_arrays(sources)

        # 4. Final Clip
        with contextlib.suppress(Exception):
            final_elev = final_elev.rio.clip_box(
                minx=west,
                miny=south,
                maxx=east,
                maxy=north,
                crs="EPSG:4326",
                allow_one_dimensional_raster=True,
            )
            final_src = final_src.rio.clip_box(
                minx=west,
                miny=south,
                maxx=east,
                maxy=north,
                crs="EPSG:4326",
                allow_one_dimensional_raster=True,
            )

        final_ds = xr.Dataset({"elevation": final_elev, "source_id": final_src})
        # Ensure CRS is maintained after merge and clip
        target_crs = final_elev.rio.crs or "EPSG:4326"
        final_ds.rio.write_crs(target_crs, inplace=True)
        final_ds["elevation"].rio.write_crs(target_crs, inplace=True)
        final_ds["source_id"].rio.write_crs(target_crs, inplace=True)

        final_ds.attrs["provenance_dict"] = provenance_dict
        final_ds.attrs["fetch_errors"] = len(fetch_errors)
        logger.debug(f"NOAA Topobathy Return -> CRS: {final_ds.rio.crs}")
        return cast(xr.Dataset, final_ds)

    def get_metadata(self) -> dict[str, Any]:
        """
        Return metadata for the NOAA Topobathy provider.

        Returns:
            Dictionary containing provider name, citation, resolution, and URL.
        """
        return {
            "name": "NOAA Coastal Topobathy LiDAR",
            "citation": "NOAA National Geodetic Survey (NGS) / OCM.",
            "resolution": "High (Variable, typically 1m - 3m)",
            "url": "https://coast.noaa.gov/digitalcoast/data/coastallidar.html",
        }

    def _fetch_index(self) -> str:
        """
        Fetches the HTML index page from NOAA.
        """
        logger.info("Fetching NOAA Coastal Lidar PDS Index (HTML)...")
        return _fetch_index_cached(self.INDEX_URL)

    def _ensure_project_list(self) -> None:
        """
        Parses the map of ID -> Project Folder Name.
        Checks local JSON or fetches remote HTML index.
        Uses a FileLock so the disk/network hit happens only once per machine.
        """
        if self._projects:
            logger.debug(f"Topobathy Project Index Cache Hit (in-process): {len(self._projects)} projects")
            return

        with NoaaTopobathyProvider._cls_lock:
            # Re-check after acquiring lock (another thread may have loaded it)
            if self._projects:
                logger.debug(f"Topobathy Project Index Cache Hit (post-lock): {len(self._projects)} projects")
                return

            json_path = self.metadata_dir / "noaa_coastal_lidar.json"
            lock_path = self.metadata_dir / "noaa_coastal_lidar.json.lock"
            import json
            import time

            from filelock import FileLock

            with FileLock(str(lock_path)):
                # Inside the lock, check if we accidentally loaded it
                # (unlikely in this structure but good practice)
                # Note: self._projects is cleared/reloaded below.

                # 1. Try Loading JSON (if fresh)
                refresh_needed = True
                if json_path.exists():
                    age = time.time() - json_path.stat().st_mtime
                    if age < 604800:  # 7 days
                        try:
                            with open(json_path) as f:
                                data = json.load(f)
                                NoaaTopobathyProvider._cls_projects.clear()
                                NoaaTopobathyProvider._cls_projects_metadata_urls.clear()

                                for k, v in data.items():
                                    if isinstance(v, dict):
                                        NoaaTopobathyProvider._cls_projects[k] = v.get("name", "")
                                        if "info" in v:
                                            NoaaTopobathyProvider._cls_projects_metadata_urls[k] = v["info"]
                                    else:
                                        NoaaTopobathyProvider._cls_projects[k] = str(v)

                                logger.debug(f"Loaded {len(self._projects)} projects from JSON metadata.")
                                refresh_needed = False
                        except Exception as e:
                            logger.warning(f"Corrupt or stale metadata JSON {json_path}: {e}")
                    else:
                        logger.info(f"NOAA Index cache expired ({age / 86400:.1f} days).")

                if not refresh_needed:
                    return

                # 2. Fetch and Parse HTML (if missing or stale)
                try:
                    logger.info("Refreshing NOAA Coastal Lidar PDS Index...")
                    text = self._fetch_index()
                    new_projects = {}
                    new_metadata_urls = {}

                    lines = text.split("\n")
                    current_folder = None

                    for line in lines:
                        # 1. Find Folder / Bulk Link
                        folder_match = re.search(r'href=".*dem/([^/]+)/index\.html"', line)
                        if folder_match:
                            folder_name = folder_match.group(1)
                            parts = folder_name.split("_")
                            if parts and parts[-1].isdigit():
                                pid = parts[-1]
                                current_folder = (pid, folder_name)
                                new_projects[pid] = folder_name

                        # 2. Find InPort Link — search any line within the current project's
                        # block, not just lines containing "metadata" (case-sensitive miss).
                        if current_folder:
                            pid = current_folder[0]
                            if pid not in new_metadata_urls:
                                inport_match = re.search(
                                    r'href="(https://www\.fisheries\.noaa\.gov/inport/item/\d+)"',
                                    line,
                                )
                                if inport_match:
                                    new_metadata_urls[pid] = inport_match.group(1)

                    # Update shared class-level dicts
                    NoaaTopobathyProvider._cls_projects.update(new_projects)
                    NoaaTopobathyProvider._cls_projects_metadata_urls.update(new_metadata_urls)

                    p_count = len(self._projects)
                    m_count = len(self._projects_metadata_urls)
                    logger.info(f"Discovered {p_count} Projects & {m_count} Metadata Links.")

                    # 3. Save to JSON
                    save_data = {}
                    for pid, folder in self._projects.items():
                        entry = {"name": folder}
                        if pid in self._projects_metadata_urls:
                            entry["info"] = self._projects_metadata_urls[pid]
                        save_data[pid] = entry

                    try:
                        with open(json_path, "w") as f:
                            json.dump(save_data, f, indent=2)
                    except Exception as e:
                        logger.warning(f"Failed to write metadata JSON: {e}")

                except Exception as e:
                    logger.error(f"Failed to load project index: {e}")
                    # Fallback: Try to load stale JSON if it exists
                    if json_path.exists():
                        logger.warning("Using stale metadata JSON as fallback.")
                        try:
                            with open(json_path) as f:
                                data = json.load(f)
                                for k, v in data.items():
                                    if isinstance(v, dict):
                                        NoaaTopobathyProvider._cls_projects[k] = v.get("name", "")
                                        if "info" in v:
                                            NoaaTopobathyProvider._cls_projects_metadata_urls[k] = v["info"]
                                    else:
                                        NoaaTopobathyProvider._cls_projects[k] = str(v)
                        except Exception as ex:
                            logger.error(f"Fallback failed: {ex}")

    def fetch_inport_metadata(self, project_id: str) -> dict | None:
        """
        Fetches and parses the InPort XML metadata for a project.
        """
        self._ensure_project_list()

        info_url = self._projects_metadata_urls.get(project_id)
        if not info_url:
            logger.debug(f"No InPort URL known for Project {project_id}")
            return None

        # Ensure it's the XML endpoint
        # We prefer the 'inport-xml' format (full metadata) over 'xml' (catalog item)
        if not info_url.endswith("/inport-xml"):
            # If it currently ends in /xml, strip it
            if info_url.endswith("/xml"):
                info_url = info_url[:-4]

            info_url = f"{info_url}/inport-xml"

        # Extract InPort ID for cache filename
        try:
            parts = info_url.split("/")
            # Handle /xml and /inport-xml
            if "inport-xml" in parts:
                inport_id = parts[parts.index("inport-xml") - 1]
            elif "xml" in parts:
                inport_id = parts[parts.index("xml") - 1]
            else:
                inport_id = parts[-1]
        except Exception:
            inport_id = f"pid_{project_id}"

        xml_path = self.inport_cache_dir / f"{inport_id}.xml"

        # 1. Fetch
        if not xml_path.exists():
            try:
                logger.info(f"Fetching InPort Metadata: {info_url}")
                headers = {"User-Agent": "Mozilla/5.0 topobathykit/1.0"}
                r = requests.get(info_url, headers=headers, timeout=10)
                if r.status_code == 200 and len(r.content) > 100:
                    with open(xml_path, "wb") as f:
                        f.write(r.content)
                else:
                    logger.warning(f"InPort fetch failed {r.status_code} for {project_id}")
                    return None
            except Exception as e:
                logger.warning(f"InPort fetch error: {e}")
                return None

        # 2. Parse
        return self._parse_inport_xml(xml_path)

    def _parse_inport_xml(self, xml_path: Path) -> dict:
        """
        Parses FGDC/InPort XML to extract Vertical Datum and Temporal range.
        """
        try:
            import xml.etree.ElementTree

            tree = xml.etree.ElementTree.parse(xml_path)
            root = tree.getroot()

            meta: dict[str, Any] = {
                "vertical_datum": "Unknown",
                "start_date": None,
                "end_date": None,
                "sensor_name": None,
                "is_topobathy": False,
            }

            # 1. Vertical Datum
            # Strategy A: Structured <reference-system>
            for elem in root.iter("reference-system"):
                for crs in elem.iter("coordinate-reference-system"):
                    # Check if it's Vertical
                    is_vertical = False
                    for child in crs:
                        if "crs-type" in child.tag and child.text == "Vertical":
                            is_vertical = True
                            break

                    if is_vertical:
                        datum_name = None
                        epsg_name = None

                        for child in crs:
                            if "datum-name" in child.tag:
                                datum_name = child.text
                            if "epsg-name" in child.tag:
                                epsg_name = child.text

                        if datum_name:
                            meta["vertical_datum"] = datum_name
                        elif epsg_name:
                            meta["vertical_datum"] = epsg_name

                        # Normalize common names
                        if meta["vertical_datum"]:
                            v = meta["vertical_datum"].lower()
                            if "navd88" in v or "north american vertical datum 1988" in v:
                                meta["vertical_datum"] = "NAVD88"
                            elif "geoid18" in v and "navd88" in v:
                                meta["vertical_datum"] = "NAVD88 (Geoid18)"

            # Strategy B: Fallback to text search if still Unknown
            if meta["vertical_datum"] == "Unknown":
                for elem in root.iter():
                    if ("altdatum" in elem.tag or "vdatum" in elem.tag) and elem.text:
                        txt = elem.text.lower()
                        if "navd88" in txt or "88" in txt:
                            meta["vertical_datum"] = "NAVD88"
                        elif "geoid18" in txt:
                            meta["vertical_datum"] = "NAVD88 (Geoid18)"
                        elif "ellipsoid" in txt:
                            meta["vertical_datum"] = "Ellipsoid"

            # 2. Time Period
            # Parsing strategy: Check <extents> for detailed time frames,
            # check top-level <time-period>, then fallback to name analysis.

            starts = []
            ends = []

            # A) Check <extents>
            # (Loop removed as it was empty/placeholder)

            # Robust Iteration for Extents > TimeFrames
            for extent in root.iter("extent"):
                for tf in extent.iter("time-frame"):
                    # find child regardless of NS
                    for child in tf:
                        if "start-date-time" in child.tag and child.text:
                            starts.append(child.text[:10])
                        if "end-date-time" in child.tag and child.text:
                            ends.append(child.text[:10])

            # B) Check top-level <time-period> if extents empty
            if not starts:
                for elem in root.iter():
                    if "begdate" in elem.tag and elem.text:
                        starts.append(elem.text)
                    if "enddate" in elem.tag and elem.text:
                        ends.append(elem.text)

            if starts:
                meta["start_date"] = min(starts)
            if ends:
                meta["end_date"] = max(ends)

            # 3. Resolution (New)
            # Strategy A: Structured Tag
            res_val = None
            res_unit = None
            for elem in root.iter():
                if "spatial-resolution" in elem.tag:
                    for child in elem:
                        if "horizontal-distance" in child.tag and child.text:
                            with contextlib.suppress(ValueError):
                                res_val = float(child.text)
                        if "horizontal-distance-units" in child.tag and child.text:
                            res_unit = child.text.lower()

            if res_val is not None:
                # Normalize to meters
                if res_unit and "meter" in res_unit:
                    meta["resolution_meters"] = res_val
                elif res_unit and "foot" in res_unit:
                    meta["resolution_meters"] = res_val * 0.3048
                elif res_unit and "degree" in res_unit:
                    # Rough conversion? 1 deg ~ 111km.
                    # Usually these are small fractions like 0.00001
                    meta["resolution_meters"] = res_val * 111320.0
                else:
                    # Assume meters if unit missing but value exists? check value range?
                    # Safer to just assume meters for lidar DEMs.
                    meta["resolution_meters"] = res_val

            # Strategy B: Abstract Regex Fallback
            if meta.get("resolution_meters") is None:
                # Concatenate all text content to search (abstract, supplemental-info, lineage, etc.)
                # Limiting to fields likely to contain resolution to avoid false positives (e.g. "500m tiles")
                search_text = ""
                for elem in root.iter():
                    if (
                        elem.tag
                        in [
                            "abstract",
                            "supplemental-information",
                            "lineage",
                            "purpose",
                            "description",
                        ]
                        and elem.text
                    ):
                        search_text += elem.text + " "

                if search_text:
                    import re

                    # Patterns:
                    # 1. "1m pixel resolution" / "0.5 meter resolution"
                    # 2. "1m bare earth"
                    # 3. "1m resolution"
                    pattern = (
                        r"(\d+(?:\.\d+)?)\s*(?:m|meter|meters)\s+(?:Pixel\s+)?"
                        r"(?:Bare\s+Earth\s+)?(?:Grid\s+)?(?:Resolution)"
                    )
                    match = re.search(pattern, search_text, re.IGNORECASE)

                    if not match:
                        # Fallback for "1m bare earth grids"
                        match = re.search(
                            r"(\d+(?:\.\d+)?)\s*(?:m|meter|meters)\s+bare\s+earth", search_text, re.IGNORECASE
                        )

                    if match:
                        with contextlib.suppress(ValueError):
                            meta["resolution_meters"] = float(match.group(1))

            # 3. Topobathy Classification
            # Default to False unless we find positive evidence
            # Keywords: "topobathy", "bathymetr", "submerged"
            # Negative Keywords: "hydro flattened" (common in topo), but "hydro" might be too broad.
            is_topobathy = False

            # Check Title
            title = root.find(".//title")
            if title is not None and title.text:
                t = title.text.lower()
                if "topobathy" in t or "bathymetr" in t:
                    is_topobathy = True

            # If not in title, check abstract/keywords
            if not is_topobathy:
                for elem in root.iter():
                    if elem.tag in ("abstract", "purpose", "keyword") and elem.text:
                        txt = elem.text.lower()
                        if "topobathy" in txt or "bathymetr" in txt or "submerged" in txt:
                            is_topobathy = True
                            break

            meta["is_topobathy"] = is_topobathy

            return meta

        except Exception as e:
            logger.warning(f"Error parsing XML {xml_path}: {e}")
            return {}

    def _ensure_spatial_index(self) -> None:
        """
        Loads the spatial index of project extents (GeoJSON).
        If missing, attempts to build it (this may take time).
        Uses a class-level lock so the file is read only once per process.
        """
        if self._spatial_index is not None:
            logger.debug(
                f"Topobathy Spatial Index Cache Hit (in-process): {len(self._spatial_index)} projects"
            )
            return

        with NoaaTopobathyProvider._cls_lock:
            # Re-check after acquiring lock (another thread may have loaded it)
            if self._spatial_index is not None:
                logger.debug(
                    f"Topobathy Spatial Index Cache Hit (post-lock): {len(self._spatial_index)} projects"
                )
                return

            index_path = self.metadata_dir / "noaa_project_extents.geojson"

            if not index_path.exists():
                logger.warning(
                    "Spatial index missing. Building NOAA Project Index (this may take several minutes)..."
                )
                try:
                    # scripts must be a package for relative import to resolve
                    from ..scripts.build_noaa_index import main as build_index

                    # Run the builder script logic
                    build_index()
                except ImportError:
                    # Fallback if scripts isn't importable as package
                    logger.error(
                        "Could not import build_noaa_index script. Please run "
                        "'python -m topobathykit.scripts.build_noaa_index' manually."
                    )
                    return
                except Exception as e:
                    logger.error(f"Failed to build spatial index: {e}")
                    return

            if index_path.exists():
                try:
                    self._spatial_index = gpd.read_file(index_path)
                    # Parse dates for sorting
                    import pandas as pd

                    if "end_date" in self._spatial_index.columns:
                        self._spatial_index["end_date"] = pd.to_datetime(
                            self._spatial_index["end_date"], errors="coerce"
                        )

                    logger.info(f"Loaded NOAA Spatial Index with {len(self._spatial_index)} projects.")
                except Exception as e:
                    logger.error(f"Failed to load spatial index {index_path}: {e}")
                    self._spatial_index = None

    def find_projects_by_box(
        self,
        west: float,
        south: float,
        east: float,
        north: float,
        filter_criteria: dict[str, Any] | None = None,
    ) -> list[str]:
        """
        Identifies ALL project IDs intersecting the bounding box using the spatial index.
        Returns list sorted by priority (Highest Priority First).

        Applies optional filters:
        - min_year: Exclude projects older than this year.
        - max_resolution: Exclude projects with resolution > this value (in meters).
        """
        self._ensure_project_list()

        # Try loading spatial index
        self._ensure_spatial_index()

        if self._spatial_index is not None and not self._spatial_index.empty:
            query_box = box(west, south, east, north)

            # Drop rows with missing geometries (corrupt index entries) before spatial query
            valid_mask = self._spatial_index.geometry.notnull()
            valid_index = self._spatial_index[valid_mask]

            # Filter by intersection
            candidates = valid_index[valid_index.intersects(query_box)].copy()

            if not candidates.empty:
                # Helper: Extract Year
                def get_year(row: Any) -> int:
                    """
                    Extracts the survey year from a project row.
                    Prefers explicit end_date, then regex on project_name, then fallback.
                    """
                    # Try explicit date first
                    if hasattr(row, "end_date") and row.end_date:
                        try:
                            val = str(row.end_date)
                            # Handle YYYY-MM-DD or simple YYYY
                            if "-" in val:
                                return int(val.split("-")[0])
                            return int(val[:4])
                        except Exception:
                            pass

                    # Parse from name (e.g. ..._2023_...)
                    match = re.search(r"_(\d{4})_", str(row.project_name))
                    if match:
                        return int(match.group(1))

                    # Fallback to last 4 digits
                    matches = re.findall(r"(\d{4})", str(row.project_name))
                    if matches:
                        for y in reversed(matches):
                            iy = int(y)
                            if 1950 <= iy <= 2030:
                                return iy
                    return 0

                candidates["year_computed"] = candidates.apply(get_year, axis=1)

                # --- APPLY FILTERS ---
                if filter_criteria:
                    # 1. Year Filter
                    if "min_year" in filter_criteria:
                        min_year = int(filter_criteria["min_year"])
                        candidates = candidates[candidates["year_computed"] >= min_year]

                    # 1.5 Project ID Filter (Explicit inclusion)
                    if "project_id" in filter_criteria:
                        pid_val = filter_criteria["project_id"]
                        if isinstance(pid_val, list):
                            allowed_pids = [str(p) for p in pid_val]
                        else:
                            allowed_pids = [str(pid_val)]
                        candidates = candidates[candidates["project_id"].astype(str).isin(allowed_pids)]
                        if not candidates.empty:
                            logger.info(f"Explicitly filtering to requested project_ids: {allowed_pids}")

                    # 2. Resolution Filter
                    if "max_resolution" in filter_criteria and "resolution_meters" in candidates.columns:
                        max_res = float(filter_criteria["max_resolution"])
                        # Keep items where resolution is less than max_res or unknown
                        candidates = candidates[
                            (candidates["resolution_meters"] <= max_res)
                            | (candidates["resolution_meters"].isna())
                        ]

                if candidates.empty:
                    logger.info(
                        f"No projects matched filters {filter_criteria} in bbox {west},{south},{east},{north}"
                    )
                # Filter by Project Type (Topobathy vs Topographic)
                # Default: Return Topobathy projects only.
                # If filter_criteria has 'project_type'='topographic', invert.
                target_type = "topobathy"
                if filter_criteria and filter_criteria.get("project_type") == "topographic":
                    target_type = "topographic"

                is_explicit = filter_criteria and "project_id" in filter_criteria
                # If the index has .is_topobathy. column and not explicitly requested
                if not is_explicit and "is_topobathy" in self._spatial_index.columns:
                    if target_type == "topobathy":
                        # Candidates must be True OR have "topobathy"/"bathymetry" in the project name
                        # This fallback is crucial for datasets where XML metadata parsing missed the
                        # 'is_topobathy' flag but the project name clearly indicates it.
                        is_flagged = candidates["is_topobathy"].fillna(False).astype(bool)

                        # Case-insensitive name check
                        name_match = (
                            candidates["project_name"]
                            .astype(str)
                            .str.contains("topobathy|bathymetr", case=False, regex=True)
                        )

                        candidates = candidates[is_flagged | name_match]
                    else:
                        # Candidates must be False or NaN (using ~ for inversion of True,
                        # but handling NaNs carefully)

                        # We treat NaN as False (Topographic)
                        mask = ~candidates["is_topobathy"].fillna(False).astype(bool)
                        candidates = candidates[mask]

                if candidates.empty:
                    # Resolve logging variables safely
                    max_res_log: float | None = (
                        float(filter_criteria["max_resolution"])
                        if filter_criteria and "max_resolution" in filter_criteria
                        else None
                    )
                    min_year_log: int | None = (
                        int(filter_criteria["min_year"])
                        if filter_criteria and "min_year" in filter_criteria
                        else None
                    )

                    res_val = str(max_res_log) if max_res_log else "N/A"
                    year_val = str(min_year_log) if min_year_log else "N/A"
                    bbox = (west, south, east, north)

                    logger.warning(
                        f"No NOAA projects found in bbox {bbox} matching criteria "
                        f"(Type={target_type}"
                        f"{', Year>=' + year_val if min_year_log else ''}"
                        f"{', Res<=' + res_val if max_res_log else ''})"
                    )
                    return []

                # --- SCORING & SORTING ---
                # Check for explicit list ordering
                allowed_pids = []
                if filter_criteria and isinstance(filter_criteria.get("project_id"), list):
                    allowed_pids = [str(p) for p in filter_criteria["project_id"]]

                def explicit_score(pid: object) -> int:
                    pid_str = str(pid)
                    if pid_str in allowed_pids:
                        # Give higher scores to indices earlier in the YAML list
                        return len(allowed_pids) - allowed_pids.index(pid_str)
                    return 0

                candidates["explicit_score"] = candidates["project_id"].apply(explicit_score)

                # 1. Datum Priority (Geoid18 > NAVD88 > Ellipsoid)
                def datum_score(d: object) -> int:
                    """
                    Scores the vertical datum for prioritization.
                    NAVD88 > NAVD88 (Geoid18) > Ellipsoid > Unknown.
                    """
                    d_str = str(d).lower()
                    if "geoid18" in d_str:
                        return 3
                    if "navd88" in d_str:
                        return 2
                    if "ellipsoid" in d_str:
                        return 0
                    return 1  # Unknown/Other -> Neutral

                candidates["datum_score"] = candidates["vertical_datum"].apply(datum_score)

                # 2. Prioritize "Topobathy" in name (vs generic DEM)
                def name_score(n: object) -> int:
                    """
                    Scores the project name for topobathymetric relevance.
                    Prioritizes explicit topobathy/bathymetry keywords.
                    """
                    n_str = str(n).lower()
                    if "topobathy" in n_str:
                        return 1
                    return 0

                candidates["name_score"] = candidates["project_name"].apply(name_score)

                # 3. Recency (Year)
                # candidates["year_computed"] is already there

                # 4. Sort by Explicit List > Name > Year > Datum
                candidates = candidates.sort_values(
                    by=["explicit_score", "name_score", "year_computed", "datum_score"],
                    ascending=[False, False, False, False],
                )

                # Return list of IDs
                projects = candidates["project_id"].astype(str).tolist()
                if projects:
                    best = candidates.iloc[0]
                    filtered_msg = " (filtered from others)" if filter_criteria else ""
                    logger.info(
                        f"Selected Best{filtered_msg}: {best['project_id']} ({best['project_name']}) "
                        f"[Year={best['year_computed']}, Res={best.get('resolution_meters', 'N/A')}]"
                    )
                    return cast(list[str], projects)

        # If spatial index is missing or empty, we fail loudly as requested
        if self._spatial_index is None or self._spatial_index.empty:
            raise RuntimeError(
                "NOAA Spatial Index is unavailable. "
                "Please run 'python -m topobathykit.scripts.build_noaa_index' to generate it."
            )

        return []

    def find_project_by_box(self, west: float, south: float, east: float, north: float) -> str | None:
        """
        Backward compatibility wrapper. Returns the best project ID.
        """
        projects = self.find_projects_by_box(west, south, east, north)
        return projects[0] if projects else None

    def set_active_project(self, project_id: str) -> None:
        """
        Sets the active project and loads its tile index.

        The active project is shared state on the process-wide singleton, so it is
        only a convenience for single-threaded callers. Concurrent code must pass
        the project ID explicitly to `resolve_tiles_in_bbox` and `fetch_tile`.
        """
        self._ensure_project_list()
        if project_id not in self._projects:
            logger.error(f"Project ID {project_id} not found in index.")
            # Clear rather than keep the previous project active, so a later
            # resolve_tiles_in_bbox() cannot silently use another project's index.
            self._active_project_id = None
            return

        self._active_project_id = project_id
        self._load_tile_index(project_id)

    def _load_tile_index(self, project_id: str) -> "gpd.GeoDataFrame | None":
        """Return the tile index for `project_id`, loading it into the class cache if needed.

        Reads and writes `_cls_tile_indices[project_id]` directly and never touches
        `_active_project_id`, so it is safe to call from concurrent threads.
        """
        cached = NoaaTopobathyProvider._cls_tile_indices.get(project_id)
        if cached is not None:
            return cached

        self._ensure_project_list()
        if project_id not in self._projects:
            logger.error(f"Project ID {project_id} not found in index.")
            return None

        with NoaaTopobathyProvider._cls_index_lock:
            # Re-check after acquiring the lock (another thread may have loaded it)
            cached = NoaaTopobathyProvider._cls_tile_indices.get(project_id)
            if cached is not None:
                return cached
            index = self._read_tile_index(project_id)
            if index is not None:
                NoaaTopobathyProvider._cls_tile_indices[project_id] = index
            return index

    def _read_tile_index(self, project_id: str) -> "gpd.GeoDataFrame | None":
        """Read the tile index for `project_id` from the disk cache, or download it from S3."""
        # Use disk-backed caching for the tile index to eliminate S3 directory listings
        # and redundant downloads for the same project across grid cells.
        index_cache_dir = self.metadata_dir / "tile_index"
        index_cache_dir.mkdir(parents=True, exist_ok=True)

        # Look for local GPKG or other cached index files for this project ID
        # (We use project_id as filename prefix)
        cached_indices = list(index_cache_dir.glob(f"{project_id}.*"))
        if cached_indices:
            try:
                # Load the first one found (usually .gpkg or .shp).
                # Geopandas can read .zip files directly if they contain a shapefile.
                index = gpd.read_file(cached_indices[0])
                logger.debug(f"Topobathy Tile Index Cache Hit (Disk): {project_id}")
                return index
            except Exception as e:
                logger.warning(f"Failed to load cached tile index {cached_indices[0]}: {e}")

        # Cache Miss - Identify index file on S3
        # Find tile index in laz/geoid18/{ID}, laz/geoid12b/{ID}, or dem/{FOLDER}/
        # We try geoid18 first, then geoid12b, then fallback to dem/ folder
        candidates = [f"laz/geoid18/{project_id}/", f"laz/geoid12b/{project_id}/"]

        folder_name = self._projects.get(project_id)
        if folder_name:
            candidates.append(f"dem/{folder_name}/")

        index_file_key = None

        for prefix in candidates:
            try:
                # s3fs ls returns full paths
                # e.g. ["bucket/key", ...]
                path_to_list = f"{self.BUCKET_BASE}/{prefix}"
                try:
                    files = self.fs.ls(path_to_list)
                except FileNotFoundError:
                    continue

                if not files:
                    continue

                for f in files:
                    # f is full path e.g. noaa-nos.../laz/...
                    name = Path(f).name.lower()  # ensure case-insensitive check
                    if "tileindex" in name and (
                        name.endswith(".gpkg") or name.endswith(".zip") or name.endswith(".shp")
                    ):
                        index_file_key = f
                        break
                if index_file_key:
                    break
            except Exception:
                continue

        if not index_file_key:
            # Fallback for datasets with non-standard tile index locations
            pass

        if not index_file_key:
            logger.warning(f"No tile index found for Project {project_id} in standard locations.")
            return None

        # Download Index to persistent metadata cache
        local_index_path = index_cache_dir / f"{project_id}{Path(index_file_key).suffix}"
        if not local_index_path.exists():
            logger.info(f"Downloading Tile Index: {index_file_key}")
            try:
                self.fs.get(index_file_key, str(local_index_path))
            except Exception as e:
                logger.error(f"Failed to download tile index: {e}")
                return None

        try:
            return gpd.read_file(local_index_path)
        except Exception as e:
            logger.error(f"Failed to load tile index {local_index_path}: {e}")
            return None

    def resolve_tiles_in_bbox(
        self,
        west: float,
        south: float,
        east: float,
        north: float,
        project_id: str | None = None,
    ) -> list[str]:
        """
        Returns list of tile filenames (or download URLs) for the bbox.

        Pass `project_id` explicitly from concurrent code; it falls back to the
        active project only for single-threaded callers of `set_active_project`.
        """
        pid = project_id if project_id is not None else self._active_project_id
        if pid is None:
            return []
        tile_index = NoaaTopobathyProvider._cls_tile_indices.get(pid)
        if tile_index is None:
            return []

        search_box = box(west, south, east, north)

        # Reproject search box to index CRS if different
        query_geom = search_box
        if tile_index.crs and tile_index.crs.to_string() != "EPSG:4326":
            gdf_box = gpd.GeoSeries([search_box], crs="EPSG:4326")
            with contextlib.suppress(Exception):
                query_geom = gdf_box.to_crs(tile_index.crs)[0]

        # Drop rows with missing geometries before spatial query
        valid_mask = tile_index.geometry.notnull()
        valid_index = tile_index[valid_mask]
        matches = valid_index[valid_index.intersects(query_geom)]

        results = []
        for _, row in matches.iterrows():
            fname = None
            # Extended column search order — prefer URL/url first so full S3 URLs are used
            # directly (avoids subdir path pollution from internal NAS `location` paths).
            for col in ["URL", "url", "location", "Location", "Name", "name", "id", "TileName"]:
                if col in row:
                    val = str(row[col])
                    if val and val.lower() != "nan":
                        fname = val
                        break

            if fname:
                # Full URL: pass directly; fetch_tile extracts the local filename.
                if fname.startswith("http"):
                    if not fname.endswith(".tif"):
                        fname += ".tif"
                    results.append(fname)
                    continue

                # Cleanup internal absolute paths like /san1/raster/.../Project_Folder/block/tile.tif
                if pid:
                    folder_name = self._projects.get(pid)
                    if folder_name and folder_name in fname:
                        fname = fname.split(folder_name + "/")[-1]
                    else:
                        # Just in case it's an absolute path without the folder name
                        if "/" in fname and not fname.startswith(folder_name or "dummy"):
                            fname = Path(fname).name
                else:
                    if "/" in fname:
                        fname = Path(fname).name

                if not fname.endswith(".tif"):
                    fname += ".tif"  # Append extension if missing
                results.append(fname)

        return list(set(results))

    def fetch_tile(
        self,
        tile_filename: str,
        bbox: tuple[float, float, float, float] | None = None,
        project_id: str | None = None,
        raise_errors: bool = False,
    ) -> xr.DataArray | None:
        """
        Fetches the specific COG, applies VDatum corrections, and caches as Zarr.

        Optimization:
        If 'bbox' is provided, we use "Clip-then-Cache":
        1. Generate cache key based on Tile + BBox.
        2. If miss, stream lazy COG -> Clip to BBox -> Download only that subset -> Cache Zarr.

        This avoids downloading 2GB+ files for a small request.

        Returns None when the tile has no data for the request (404, empty clip, no
        VDatum coverage). With ``raise_errors=True``, failures that a retry may fix
        (network, VDatum outage, MemoryError) raise `ProviderFetchError` instead of
        also returning None, so `fetch_layer` can tell them apart from no coverage.
        """
        # Use caller-supplied project_id so concurrent threads on the singleton don't
        # clobber each other's _active_project_id state mid-fetch.
        pid = project_id if project_id is not None else self._active_project_id
        if not pid:
            return None

        # Resolve Project Folder Name
        if pid not in self._projects:
            self._ensure_project_list()

        if pid not in self._projects:
            logger.error(f"Cannot resolve folder for Project ID {pid}")
            return None

        # Construct Remote and Local paths
        if "https://" in tile_filename or "http://" in tile_filename:
            # If tile_filename comes in as a full URL (from 'URL' column in index)
            # Use it directly as the download source
            http_url = tile_filename

            # Preserve path structure for unique cache keys instead of just the basename
            import urllib.parse

            parsed = urllib.parse.urlparse(http_url)
            path = parsed.path.lstrip("/")
            if path.startswith("dem/"):
                path = path[4:]
            tile_filename = path
        else:
            # Construct standard URL
            # Clean up leading slashes that cause double slashes in S3 URL string
            clean_tile_filename = tile_filename.lstrip("/")

            folder_name = self._projects.get(pid)
            if folder_name and not clean_tile_filename.startswith(folder_name):
                clean_tile_filename = f"{folder_name}/{clean_tile_filename}"

            http_url = f"https://{self.BUCKET_BASE}.s3.amazonaws.com/dem/{clean_tile_filename}"

        local_filename = f"{pid}_{tile_filename}"

        # --- CACHE KEY GENERATION ---
        # Legacy/Full file cache name (without bbox slice)
        full_zarr_name = local_filename.replace(".tif", "").replace(".tiff", "") + "_navd88.zarr"
        full_zarr_path = self.cache_dir / "zarr" / full_zarr_name

        # If bbox provided, include hash in filename
        if bbox:
            import hashlib

            # Precision 4 decimals (~10m) to allow some fuzzy hit if slightly different?
            # No, requests should be stable. Use raw bbox string.
            # Rounding to avoids float drift issues.
            rb = tuple(round(x, 6) for x in bbox)
            bbox_hash = hashlib.md5(str(rb).encode()).hexdigest()[:8]
            zarr_name = local_filename.replace(".tif", "").replace(".tiff", "") + f"_navd88_{bbox_hash}.zarr"
        else:
            zarr_name = full_zarr_name

        zarr_path = self.cache_dir / "zarr" / zarr_name
        missing_marker_path = self.cache_dir / "zarr" / (zarr_name + ".404")

        # 0. Check Negative Cache
        if missing_marker_path.exists():
            return None

        # 0. Check if FULL Tile is already cached (Avoids streaming from S3 if we already have the parent)
        if bbox and full_zarr_path.exists():
            try:
                da = xr.open_dataarray(full_zarr_path, engine="zarr", chunks="auto", decode_coords="all")
                logger.debug(f"Topobathy Zarr Cache Hit (Full Tile Proxy for BBox): {full_zarr_name}")
                # Clip it in memory and return immediately (no need to write a micro-zarr to disk)
                da = da.rio.clip_box(*bbox, allow_one_dimensional_raster=True)

                return cast(xr.DataArray, da)
            except Exception as e:
                logger.debug(f"Full Zarr proxy cache load failed or clip failed: {e}")
                pass  # Fall back to normal flow

        # 1. Zarr Cache Hit (Fast Path)
        if zarr_path.exists():
            try:
                da = xr.open_dataarray(zarr_path, engine="zarr", chunks="auto", decode_coords="all")
                logger.debug(f"Topobathy Zarr Cache Hit: {zarr_name}")
                return da
            except Exception as e:
                logger.warning(f"Corrupt Zarr {zarr_path}: {e}")
                import shutil

                shutil.rmtree(zarr_path, ignore_errors=True)

        # 2. Cache Miss - Acquire Lock
        # filelock.FileLock is used instead of fcntl.flock for two reasons:
        # (a) flock + manual lock-file deletion creates a TOCTOU race: a thread
        #     blocked on flock wakes up on the old inode while a new thread creates
        #     a fresh inode, so both acquire "exclusive" locks simultaneously.
        # (b) FileLock is cross-platform and correctly handles the within-process
        #     thread case without relying on POSIX advisory lock semantics.
        from filelock import FileLock

        lock_path = self.cache_dir / "zarr" / (zarr_name + ".lock")

        # Ensure dir exists
        zarr_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with FileLock(str(lock_path)):
                # Double Check inside lock
                if zarr_path.exists():
                    try:
                        da = xr.open_dataarray(zarr_path, engine="zarr", chunks="auto", decode_coords="all")
                        logger.info(f"Topobathy Zarr Cache Hit (Post-Lock): {zarr_name}")
                        return da
                    except Exception:
                        pass

                logger.info(f"Topobathy Zarr Cache Miss: Creating {zarr_name} from {http_url}")

                # 3. Stream from Remote (VSI)
                open_source = http_url
                da_raw = None

                # 4. Open COG (Remote) and Apply Adjustments
                logger.info(f"Streaming NOAA Asset: {open_source}")

                # Load lazily with chunks
                # Use a context manager where possible, or explicit close
                # Since open_rasterio returns a DataArray, we manage it manually
                da_raw = cast(
                    xr.DataArray,
                    rioxarray.open_rasterio(open_source, chunks={"x": 2048, "y": 2048}, masked=True),
                )
                da_raw = self._sanitize_nodata_values(da_raw)

                try:
                    # --- OPTIMIZATION: CLIP BEFORE METADATA/COMPUTE ---
                    if bbox:
                        try:
                            from rasterio.warp import transform_bounds

                            source_crs = da_raw.rio.crs
                            clip_bbox = bbox
                            clip_crs = "EPSG:4326"

                            # If source has a CRS and it's not 4326/NAD83-LatLon, reproject explicitly
                            # This avoids rioxarray clip_box potential failures with on-the-fly reprojection
                            if source_crs and source_crs.to_string() not in ["EPSG:4326", "EPSG:4269"]:
                                try:
                                    clip_bbox = transform_bounds("EPSG:4326", source_crs, *bbox)
                                    clip_crs = source_crs
                                except Exception as e:
                                    logger.warning(
                                        f"Failed to reproject bbox to {source_crs}: {e}. Retrying with 4326."
                                    )

                            da_raw = da_raw.rio.clip_box(
                                minx=clip_bbox[0],
                                miny=clip_bbox[1],
                                maxx=clip_bbox[2],
                                maxy=clip_bbox[3],
                                crs=clip_crs,
                                allow_one_dimensional_raster=True,
                            )
                        except Exception as e:
                            logger.warning(f"Clip failed for {local_filename}: {e}. Skipping tile.")
                            return None

                    if da_raw is None or da_raw.size == 0:
                        logger.debug(f"Empty tile after clip: {local_filename}")
                        return None

                    # Metadata & VDatum Logic
                    meta = self.fetch_inport_metadata(pid) or {}

                    # --- PROVENANCE METADATA ---
                    import datetime

                    # Core identity
                    da_raw.attrs["survey_source"] = pid
                    da_raw.attrs["source_url"] = http_url

                    # Time
                    da_raw.attrs["date_created"] = datetime.datetime.now(datetime.UTC).isoformat()
                    da_raw.attrs["start_date"] = meta.get("start_date", "Unknown")
                    da_raw.attrs["end_date"] = meta.get("end_date", "Unknown")

                    # Vertical Reference
                    vdatum = meta.get("vertical_datum", "Unknown").lower()
                    da_raw.attrs["vertical_datum_original"] = vdatum
                    da_raw.attrs["vertical_datum"] = vdatum  # Current state

                    # Links
                    info_url = self._projects_metadata_urls.get(pid)
                    if info_url:
                        da_raw.attrs["metadata_url"] = info_url

                    # Apply Correction if Ellipsoid
                    if "ellipsoid" in vdatum:
                        try:
                            from pyproj import Transformer

                            bounds = da_raw.rio.bounds()
                            cx, cy = (bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2

                            crs = da_raw.rio.crs
                            if crs:
                                t = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
                                lon, lat = t.transform(cx, cy)

                                offset = self.vdatum.get_robust_ellipsoid_to_navd88_offset(lat, lon)
                                logger.info(f"Applying VDatum Offset {offset:.3f}m to {local_filename}")
                                da_raw = da_raw + offset

                                # Update Attributes post-correction
                                da_raw.attrs["vertical_datum"] = "NAVD88"
                                da_raw.attrs["correction_method"] = "VDatum Geoid18"
                                da_raw.attrs["vdatum_offset"] = offset
                        except VDatumNoDataError as e:
                            logger.warning(
                                f"VDatum has no coverage for {local_filename}: {e}. "
                                "Dropping dataset to prevent artificial cliffs."
                            )
                            return None
                        except Exception as e:
                            logger.warning(
                                f"VDatum correction failed: {e}. "
                                "Dropping dataset to prevent artificial cliffs."
                            )
                            if raise_errors:
                                raise ProviderFetchError(
                                    f"VDatum correction failed for {local_filename}: {type(e).__name__}: {e}"
                                ) from e
                            return None

                    # 5. Write to Zarr
                    # Ensure good chunks for writing
                    if "x" in da_raw.dims and "y" in da_raw.dims:
                        da_raw = da_raw.chunk({"y": 1024, "x": 1024})

                    da_raw.to_zarr(zarr_path, mode="w", consolidated=should_consolidate())
                    logger.info(f"Created Zarr Cache: {zarr_path.name}")
                finally:
                    # Explicitly close the file handle for the remote COG
                    # This prevents accumulation of open HTTP connections/file descriptors
                    if da_raw is not None:
                        with contextlib.suppress(Exception):
                            da_raw.close()

                # Return re-opened Zarr
                return xr.open_dataarray(zarr_path, engine="zarr", chunks="auto", decode_coords="all")

        except ProviderFetchError:
            raise
        except Exception as e:
            if "404" in str(e):
                logger.warning(
                    f"File not found remotely (404), marking as missing: {missing_marker_path.name}"
                )
                missing_marker_path.parent.mkdir(parents=True, exist_ok=True)
                with open(missing_marker_path, "w") as f:
                    f.write("404 Not Found")
            else:
                logger.error(f"Zarr lock/process failed: {type(e).__name__}: {e}")
                if raise_errors:
                    raise ProviderFetchError(
                        f"Topobathy tile {local_filename} failed: {type(e).__name__}: {e}"
                    ) from e
            return None
        # Note: lock file (.lock) is intentionally NOT deleted here. Deleting it creates a
        # TOCTOU race: a thread blocked on FileLock wakes up holding the old inode while a
        # new thread creates a fresh inode, letting both proceed simultaneously. The FileLock
        # releases the OS lock on context-manager exit; the empty file is harmless on disk.

    def get_grid(
        self, west: float, south: float, east: float, north: float, project_id: str | None = None
    ) -> xr.DataArray | None:
        """
        High level interface to get merged grid from NOAA Topobathy.
        Attempts to auto-detect project if not provided.
        """
        project_ids_to_try = []
        if project_id:
            project_ids_to_try = [project_id]
        else:
            project_ids_to_try = self.find_projects_by_box(west, south, east, north)

        for pid in project_ids_to_try:
            self._load_tile_index(pid)

            tiles = self.resolve_tiles_in_bbox(west, south, east, north, project_id=pid)
            if not tiles:
                continue

            bbox = (west, south, east, north)
            das = []
            for t in tiles:
                da = self.fetch_tile(t, bbox=bbox, project_id=pid)
                if da is not None:
                    das.append(da)

            if not das:
                continue

            if len(das) == 1:
                return das[0]

            from rioxarray.merge import merge_arrays

            # Regarding Fusion order:
            # For NOAA DEMs within a single project, tiles are typically spatially disjoint (mosaic).
            # Overlap is minimal. If overlap exists, we trust rioxarray default (last layer wins).
            try:
                merged = merge_arrays(das)
                return cast(xr.DataArray, merged)
            except Exception as e:
                logger.error(f"Merge error: {e}")
                continue

        # If we exhausted all projects and found nothing:
        logger.warning(
            f"No valid data returned from {len(project_ids_to_try)} intersecting projects "
            f"for bbox {(west, south, east, north)}"
        )
        return None


# Register
registry.register(Path(__file__).stem, NoaaTopobathyProvider)
