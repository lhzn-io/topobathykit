"""
NOAA NCEI BAG Provider module.

This module implements the provider for accessing Bathymetric Attributed Grid (BAG) files
from NOAA's National Centers for Environmental Information (NCEI). It handles discovery,
downloading, caching, and reading of BAG files.
"""

import contextlib
import json
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ClassVar, cast

import numpy as np  # Added numpy import
import requests  # type: ignore
import rioxarray as rxr
import xarray as xr
from filelock import FileLock

from ..config import get_cache_root
from ..runtime import should_consolidate
from ..utils.cache import concurrent_lru_cache
from ..vdatum import VDatumNoDataError, VDatumResolver
from .base import Provider, ProviderNoDataError, interpolate_small_gaps, sanitize_elevation_nodata
from .registry import registry

logger = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (HTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# Discovery requests are expensive immediately after cache purges.
# Allow a longer read timeout and make it tunable via env.
BAG_DISCOVERY_TIMEOUT_SECONDS = int(os.getenv("TOPOBATHY_BAG_DISCOVERY_TIMEOUT", "120"))

# Lower LRU Size to prevent "Too Many Open Files" with Zarr chunks
# and limit peak memory from cached DataArrays held open in memory.
BAG_CACHE_SIZE = int(os.getenv("TOPOBATHY_BAG_LRU_SIZE", "8"))


def _parse_bag_resolution(filename: str) -> float:
    """Parse resolution in meters from a BAG filename. Returns 999.0 if unknown."""
    import re

    fname = filename.lower()
    m_cm = re.search(r"[._-](\d+(?:\.\d+)?)cm[._-]", fname)
    m_m = re.search(r"[._-](\d+(?:\.\d+)?)m[._-]", fname)
    if m_cm:
        return float(m_cm.group(1)) / 100.0
    elif m_m:
        return float(m_m.group(1))
    return 999.0  # VR or unknown — keep as fallback


def _parse_survey_id(filename: str) -> str:
    """Extract the survey ID prefix from a BAG filename.

    Examples:
        H12696_MB_50cm_MLLW_1of3.bag -> H12696
        W00181_MB_4m_MLLW_combined.bag -> W00181
        E01096_MB_7m_MLLW_6of7.bag -> E01096
    """
    import re

    m = re.match(r"^([a-zA-Z]\d{4,5})", filename, flags=re.IGNORECASE)
    return m.group(1).upper() if m else filename


def _dedup_survey_resolutions(urls: list[str], target_resolution: float | None = None) -> list[str]:
    """NOAA often splits surveys spatially based on depth, generating different resolutions
    for each depth bin (e.g. 50cm for 0-20m depth, 2m for >40m depth).

    Attempting to 'deduplicate' them and only picking one resolution will lead to massive
    spatial holes (NaN gaps). We MUST return all available files.
    """
    return urls


class _EmptyWindowError(Exception):
    """Raised when a windowed read produces no pixels."""


def _windowed_read(
    da: xr.DataArray,
    west: float,
    south: float,
    east: float,
    north: float,
) -> xr.DataArray:
    """Read a spatial subset from a DataArray without Dask.

    Transforms the EPSG:4326 clip bounds to the array's native CRS,
    computes index slices from the coordinate arrays, and reads only
    those pixels into a new in-memory DataArray. No Dask task graph
    is created or materialized.

    Raises _EmptyWindowError if the window has zero pixels.
    """
    import numpy as np
    from pyproj import Transformer

    native_crs = da.rio.crs
    if native_crs and str(native_crs) != "EPSG:4326":
        transformer = Transformer.from_crs("EPSG:4326", native_crs, always_xy=True)
        bx_west, bx_south = transformer.transform(west, south)
        bx_east, bx_north = transformer.transform(east, north)
        # Handle axis flips from reprojection
        bx_west, bx_east = min(bx_west, bx_east), max(bx_west, bx_east)
        bx_south, bx_north = min(bx_south, bx_north), max(bx_south, bx_north)
    else:
        bx_west, bx_south, bx_east, bx_north = west, south, east, north

    # Resolve coordinate arrays (works for both x/y and lon/lat naming)
    x_dim = da.rio.x_dim
    y_dim = da.rio.y_dim
    xs = da[x_dim].values
    ys = da[y_dim].values

    # Find index range via searchsorted on monotonic coord arrays
    if xs[0] < xs[-1]:
        x0 = int(np.searchsorted(xs, bx_west, side="left"))
        x1 = int(np.searchsorted(xs, bx_east, side="right"))
    else:
        x0 = int(len(xs) - np.searchsorted(xs[::-1], bx_east, side="right"))
        x1 = int(len(xs) - np.searchsorted(xs[::-1], bx_west, side="left"))

    if ys[0] < ys[-1]:
        y0 = int(np.searchsorted(ys, bx_south, side="left"))
        y1 = int(np.searchsorted(ys, bx_north, side="right"))
    else:
        y0 = int(len(ys) - np.searchsorted(ys[::-1], bx_north, side="right"))
        y1 = int(len(ys) - np.searchsorted(ys[::-1], bx_south, side="left"))

    # Clamp to valid range
    x0, x1 = max(0, x0), min(len(xs), x1)
    y0, y1 = max(0, y0), min(len(ys), y1)

    if x1 <= x0 or y1 <= y0:
        raise _EmptyWindowError("No overlap")

    # Direct numpy slice — bypasses Dask entirely
    sliced = da.isel({x_dim: slice(x0, x1), y_dim: slice(y0, y1)})
    result = cast(xr.DataArray, sliced.load())
    return result


@concurrent_lru_cache(maxsize=BAG_CACHE_SIZE)
def _read_bag_cached(local_path: Path) -> xr.DataArray | None:
    """
    Cached, standalone function to read a single BAG file.
    This avoids keeping large dataset references inside class definitions.
    Cached to avoid re-opening/parsing headers for every tile.
    """
    # Define a pathway for the cached Zarr version (directory)
    # Zarr is more parallel-friendly and I/O efficient than monolithic HDF5/NetCDF
    # Store Zarr in "zarr" subdirectory
    zarr_dir = local_path.parent / "zarr"
    zarr_path = zarr_dir / local_path.name.replace(".bag", "_navd88.zarr")

    # If we have a pre-computed NAVD88 Zarr store, load strictly that.
    if zarr_path.exists():
        try:
            # chunks="auto" enables Dask lazy loading efficiently
            # consolidated=False is safer for local directory stores unless explicitly consolidated
            da_cached = xr.open_dataarray(zarr_path, engine="zarr", chunks="auto", decode_coords="all")
            logger.info(f"BAG Zarr Cache Hit: {zarr_path.name}")
            return da_cached
        except Exception as e:
            logger.warning(f"Failed to load cached Zarr {zarr_path}, falling back to raw BAG: {e}")
            # If load fails, delete and regenerate
            try:
                import shutil

                if zarr_path.is_dir():
                    shutil.rmtree(zarr_path)
            except Exception:
                pass

    # Cache Miss - Acquire Lock
    from filelock import FileLock

    lock_path = zarr_path.with_suffix(".zarr.lock")
    try:
        with FileLock(lock_path):
            # Double check
            if zarr_path.exists():
                logger.info(f"BAG Zarr Cache Hit (after lock): {zarr_path.name}")
                return xr.open_dataarray(zarr_path, engine="zarr", chunks="auto", decode_coords="all")

            logger.info(f"BAG Zarr Cache Miss: {zarr_path.name}")

            # Use chunks to initiate Delayed/Lazy loading via Dask.
            # Prevents OOM crashes when reading large BAG files.
            with ignore_specific_gdal_warnings("cornerPoints not consistent with resolution"):
                da_raw: Any = rxr.open_rasterio(local_path, chunks={"x": 2048, "y": 2048}, masked=True)

                # VR BAG Detection & Handling:
                # 1. Filename Heuristic: Often contains "_VR_" or "_Variable_Resolution_"
                # 2. Metadata Heuristic: Contains MIN_RESOLUTION_X tag
                # 3. Dimension Heuristic: Tiny 256x256 grid for a large >1MB file (GDAL default supergrid)

                is_vr_bag = False
                fname_lower = local_path.name.lower()
                if "_vr_" in fname_lower or "_variable_resolution_" in fname_lower:
                    is_vr_bag = True
                    logger.info(f"Detected VR BAG by filename pattern: {local_path.name}")

                if not is_vr_bag:
                    if isinstance(da_raw, list):
                        pass
                    else:
                        # Check attributes (rioxarray loads GDAL metadata into attrs)
                        attrs = da_raw.attrs
                        if "MIN_RESOLUTION_X" in attrs or "MAX_RESOLUTION_X" in attrs:
                            is_vr_bag = True
                            logger.info(f"Detected VR BAG metadata in {local_path.name}")

                        # Heuristic: Tiny grid implies supergrid if file is large (e.g. >1MB but 256x256)
                        if not is_vr_bag and hasattr(da_raw, "shape"):
                            shape = da_raw.shape
                            is_tiny = len(shape) >= 2 and shape[-1] <= 512 and shape[-2] <= 512
                            is_large = local_path.stat().st_size > 1_000_000  # > 1MB

                            if is_tiny and is_large:
                                is_vr_bag = True
                                logger.info(
                                    f"Detected potential VR BAG via dimensions {shape} "
                                    f"for large file {local_path.name}"
                                )

                if is_vr_bag:
                    # Re-open VR BAGs with interpolation to merge supergrids (GDAL >= 3.8 feature).
                    # Prevents sparse "screen door" artifacts from raw supergrid sampling.
                    logger.info(f"Re-opening VR BAG {local_path.name} with MODE='INTERPOLATED'...")
                    da_raw = rxr.open_rasterio(
                        local_path, chunks={"x": 2048, "y": 2048}, masked=True, MODE="INTERPOLATED"
                    )

                da: xr.DataArray
                if isinstance(da_raw, list):
                    da = cast(xr.DataArray, da_raw[0])
                elif isinstance(da_raw, xr.Dataset):
                    da = cast(xr.DataArray, da_raw.to_array().isel(variable=0))
                else:
                    da = cast(xr.DataArray, da_raw)

            # BAGs usually have 'elevation' and 'uncertainty'.
            elev = da.isel(band=0).drop_vars("band")

            # MASK SUSPICIOUS VALUES
            # Mask standard BAG NoData values (1,000,000 or -1,000,000).
            # These can leak through if GDAL doesn't parse the BAG XML correctly.
            elev = sanitize_elevation_nodata(
                elev,
                extra_sentinels=(1_000_000.0, -1_000_000.0),
                abs_valid_limit=100000.0,
            )

            # Note: Additional dynamic filtering (e.g., configurable spike removal or NoData masking)
            # is deferred to `fetch_layer` where policy configuration (`kwargs`) is available.

            # Check for Ellipsoid vs MLLW
            filename = local_path.name
            is_ellipsoid = "_Ellipsoid_" in filename or "Ellipsoid" in filename

            # Vertical Datum Correction (to NAVD88)
            crs = elev.rio.crs
            bounds = elev.rio.bounds()  # (minx, miny, maxx, maxy)

            # Candidate points: Center, then corners, then mid-edges
            candidate_points = [
                ((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2),  # Center
                (bounds[0], bounds[1]),  # MinX, MinY
                (bounds[2], bounds[3]),  # MaxX, MaxY
                (bounds[0], bounds[3]),  # MinX, MaxY
                (bounds[2], bounds[1]),  # MaxX, MinY
            ]

            offset = 0.0
            try:
                from pyproj import Transformer

                if crs:
                    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

                    found_offset = False
                    last_error: Exception | None = None

                    for cx, cy in candidate_points:
                        try:
                            lon, lat = transformer.transform(cx, cy)

                            if is_ellipsoid:
                                # Ellipsoid -> NAVD88
                                offset = VDatumResolver.get_robust_ellipsoid_to_navd88_offset(  # type: ignore
                                    lat, lon
                                )
                                logger.info(
                                    f"Applying Ellipsoid->NAVD88 offset of {offset:.3f}m for {filename} "
                                    f"(at {lat:.4f}, {lon:.4f})"
                                )
                            else:
                                # MLLW -> NAVD88
                                offset = VDatumResolver.get_robust_mllw_to_navd88_offset(  # type: ignore
                                    lat, lon
                                )
                                logger.info(
                                    f"Applying MLLW->NAVD88 offset of {offset:.3f}m for {filename} "
                                    f"(at {lat:.4f}, {lon:.4f})"
                                )

                            found_offset = True
                            break  # Success!

                        except VDatumNoDataError as e:
                            # This point failed, try next
                            last_error = e
                            continue
                        except Exception as e:
                            # Other errors should probably abort or warn
                            logger.warning(f"VDatum check failed at ({lat:.4f}, {lon:.4f}): {e}")
                            last_error = e

                    if not found_offset:
                        msg = (
                            f"Could not determine VDatum offset for {filename} "
                            f"(tried {len(candidate_points)} points). "
                            "Dropping dataset to prevent artificial cliffs."
                        )
                        if last_error:
                            msg += f" Last error: {last_error}"
                        logger.warning(msg)
                        return None

                    # Applying offset to Dask array adds a task to the graph (Lazy).
                    if found_offset:
                        elev = elev + offset

                # --- OPTIMIZATION: Persist the adjusted data to Zarr ---
                # Zarr allows chunked parallel writes and reads, ideal for this use case.
                try:
                    logger.info(f"Caching NAVD88 adjusted BAG to {zarr_path}...")

                    # Ensure we have good chunks for Zarr (e.g. 1024x1024)
                    # This balances number of files vs I/O size
                    if "x" in elev.dims and "y" in elev.dims:
                        elev = elev.chunk({"y": 1024, "x": 1024})

                    # compute() happens during the write.
                    # mode='w' overwrites if exists (safe here as we checked existence above)
                    elev.to_zarr(zarr_path, mode="w", consolidated=should_consolidate())
                    logger.info(f"BAG Zarr Cache Created: {zarr_path.name}")

                    # Re-open from the fresh Zarr to return a consistent Dask-backed object
                    elev = xr.open_dataarray(zarr_path, engine="zarr", chunks="auto", decode_coords="all")
                except Exception as e:
                    logger.error(f"Failed to write cached Zarr: {e}")
                # ---------------------------------------------------------

            except Exception as e:
                logger.warning(f"Failed to apply VDatum correction: {e}")

            elev.attrs["survey_source"] = filename
            return cast(xr.DataArray, elev)

    except Exception as e:
        logger.error(f"Error reading BAG {local_path}: {e}")
        return None


def _filter_block(block: Any, threshold: float = 50.0) -> Any:
    """
    Apply median filter to a block.
    """
    if block.size == 0:
        return block

    from scipy.ndimage import median_filter

    # Apply standard median filter (fast, but NaN propagates)
    med = median_filter(block, size=3)
    diff = np.abs(block - med)

    if not np.issubdtype(block.dtype, np.floating):
        block = block.astype(np.float32)

    # Mask spikes exceeding threshold with NaN
    return np.where(diff > threshold, np.nan, block)


def clean_data_deviation(da: xr.DataArray, threshold: float = 50.0) -> xr.DataArray:
    """
    Apply a 'Deviation from Median' filter to remove spikes.
    Calculates median of 3x3 neighborhood. If abs(pixel - median) > threshold, mask as NaN.

    Args:
        da: Input DataArray (elevation)
        threshold: Max allowed deviation from local median (meters)

    Returns:
        xr.DataArray: Filtered data
    """
    try:
        from functools import partial

        logger.debug(f"clean_data_deviation: shape={da.shape}, chunks={da.chunks}, threshold={threshold}")

        filter_func = partial(_filter_block, threshold=threshold)

        if da.chunks is not None:
            # Apply via dask map_overlap for chunked processing
            # Provide meta to avoid auto-computation of the first chunk
            meta = np.array([], dtype=da.dtype)

            cleaned_dask = da.data.map_overlap(
                filter_func,
                depth=1,
                boundary="reflect",
                dtype=da.dtype,
                meta=meta,
            )

            # Return new DataArray with same coords
            return xr.DataArray(cleaned_dask, coords=da.coords, dims=da.dims, attrs=da.attrs)
        else:
            # Numpy array - Direct application
            # If array is massive, this will block!
            logger.warning("clean_data_deviation running on eager (numpy) array. This may hang.")
            cleaned_np = _filter_block(da.values, threshold=threshold)
            return xr.DataArray(cleaned_np, coords=da.coords, dims=da.dims, attrs=da.attrs)

    except Exception as e:
        logger.warning(f"Failed to apply cleaning filter: {e}", exc_info=True)
        return da


def _depth_range_within_threshold(da: xr.DataArray, threshold: float) -> bool:
    """
    Cheap guard for spike filtering.

    If the total finite depth range is already within the max deviation threshold,
    the median-deviation filter cannot remove any values and can be skipped.
    """
    if threshold <= 0:
        return False

    try:
        min_raw = da.min(skipna=True).values
        max_raw = da.max(skipna=True).values

        min_val = float(np.asarray(min_raw).squeeze())
        max_val = float(np.asarray(max_raw).squeeze())

        if not np.isfinite(min_val) or not np.isfinite(max_val):
            return True

        depth_range = max_val - min_val
        return depth_range <= threshold
    except Exception as e:
        logger.debug(f"Depth range pre-check failed; applying filter conservatively: {e}")
        return False


@contextmanager
def ignore_specific_gdal_warnings(message_substring: str) -> Iterator[None]:
    """
    Context manager to suppress specific GDAL/rasterio warnings that are known to be benign.
    e.g. "CPLE_AppDefined in cornerPoints not consistent with resolution"
    """
    gdal_logger = logging.getLogger("rasterio._env")

    class Filter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return message_substring not in record.getMessage()

    f = Filter()
    gdal_logger.addFilter(f)
    try:
        yield
    finally:
        gdal_logger.removeFilter(f)


class BAGDiscovery:
    """
    Discovers BAG (Bathymetric Attributed Grid) files via NOAA NCEI API so we can
    bypass generalized BlueTopo tiles and fetch raw 50cm sonar data.
    """

    # NCEI Hydrodynamic Survey Service (NOAA GeoPlatform)
    QUERY_URL = "https://services2.arcgis.com/C8EMgrsFcRFL6LrL/arcgis/rest/services/NOS_Hydro_Surveys/FeatureServer/0/query"

    # Persistent Cache for Redirects (HTML Landing Page -> .bag URL)
    # Stored in <cache root>/metadata/ncei_bag_redirects.json
    REDIRECT_CACHE_PATH = get_cache_root() / "metadata" / "ncei_bag_redirects.json"

    # Persistent Cache for BBox -> BAG URL list discovery results
    # Stored in <cache root>/ncei_bag/discovery_cache.json
    # Format: {"west_south_east_north": ["https://...bag", ...], ...}
    # Invalidate to discover newly published multibeam surveys from NCEI.
    # See: manage_discovery_cache.py --invalidate ncei_bag
    DISCOVERY_CACHE_PATH = get_cache_root() / "ncei_bag" / "discovery_cache.json"

    @classmethod
    def _read_bag_cached(cls, local_path: Path) -> xr.DataArray | None:
        """
        Reads BAG using h5py (or rasterio) and converts to xarray with NAVD88 correction.
        Cached to avoid re-opening/parsing headers for every tile.
        """
        return cast("xr.DataArray | None", _read_bag_cached(local_path))

    @classmethod
    def _get_redirect_from_cache(cls, download_url: str) -> list[str] | None:
        """Read from the JSON cache if available."""
        if not cls.REDIRECT_CACHE_PATH.exists():
            return None

        try:
            # Simple read without locking (locking on write is critical)
            with open(cls.REDIRECT_CACHE_PATH) as f:
                data = json.load(f)
                result = data.get(download_url)
                if isinstance(result, str):
                    return [result]
                return cast(list[str] | None, result)
        except Exception:
            return None

    @classmethod
    def _update_redirect_cache(cls, download_url: str, bag_urls: list[str]) -> None:
        """Update the JSON cache with a new mapping."""
        try:
            cls.REDIRECT_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            # Use lock file alongside the json
            lock_path = cls.REDIRECT_CACHE_PATH.parent / (cls.REDIRECT_CACHE_PATH.name + ".lock")

            with FileLock(lock_path):
                data = {}
                if cls.REDIRECT_CACHE_PATH.exists():
                    try:
                        with open(cls.REDIRECT_CACHE_PATH) as f:
                            data = json.load(f)
                    except json.JSONDecodeError:
                        data = {}

                # Check if already added
                if download_url in data and data[download_url] == bag_urls:
                    return

                data[download_url] = bag_urls

                # Write back sorted
                with open(cls.REDIRECT_CACHE_PATH, "w") as f:
                    json.dump(data, f, indent=2, sort_keys=True)
                    f.write("\n")  # EOF newline

        except Exception as e:
            logger.warning(f"Failed to update redirect cache: {e}")

    @classmethod
    def _get_from_discovery_cache(cls, bbox_key: str) -> list[str] | None:
        """
        Read bbox→BAG URL list from the disk discovery cache.

        Lock-free on read; the write side uses ``FileLock`` for safety.
        Returns ``None`` on a cache miss so the caller can fall back to the
        network query and then populate the cache via
        :meth:`_update_discovery_cache`.

        To force a fresh lookup (e.g. after NCEI publishes new surveys), delete
        ``DISCOVERY_CACHE_PATH`` or run
        ``manage_discovery_cache.py --invalidate ncei_bag``.
        """
        if not cls.DISCOVERY_CACHE_PATH.exists():
            return None
        try:
            with open(cls.DISCOVERY_CACHE_PATH) as f:
                data = json.load(f)
            result = data.get(bbox_key)
            return list(result) if isinstance(result, list) else None
        except Exception:
            return None

    @classmethod
    def _update_discovery_cache(cls, bbox_key: str, urls: list[str]) -> None:
        """
        Persist a bbox→BAG URL list to the disk discovery cache atomically.

        Uses ``FileLock`` to prevent concurrent writes from multiple worker
        processes from corrupting the JSON file.  Empty result lists are also
        stored so that known-empty bounding boxes are not re-queried.
        """
        try:
            cls.DISCOVERY_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            lock_path = cls.DISCOVERY_CACHE_PATH.parent / (cls.DISCOVERY_CACHE_PATH.name + ".lock")
            with FileLock(lock_path):
                data: dict[str, list[str]] = {}
                if cls.DISCOVERY_CACHE_PATH.exists():
                    try:
                        with open(cls.DISCOVERY_CACHE_PATH) as f:
                            data = json.load(f)
                    except json.JSONDecodeError:
                        data = {}
                if bbox_key in data and data[bbox_key] == urls:
                    return
                data[bbox_key] = urls
                with open(cls.DISCOVERY_CACHE_PATH, "w") as f:
                    json.dump(data, f, indent=2, sort_keys=True)
                    f.write("\n")
        except Exception as e:
            logger.warning(f"Failed to update BAG discovery cache: {e}")

    @classmethod
    def _scrape_landing_page(cls, download_url: str) -> list[str]:
        """Helper to scrape .bag URL zfrom HTML landing page."""
        # 1. Check Cache
        cached_urls = cls._get_redirect_from_cache(download_url)
        if cached_urls:
            logger.debug(f"Redirect Cache Hit: {download_url} -> {cached_urls}")
            return cached_urls

        try:
            logger.debug(f"Scraping landing page: {download_url}")
            headers = {"User-Agent": USER_AGENT}
            r_scrape = requests.get(download_url, headers=headers, timeout=10)
            r_scrape.raise_for_status()
            import re

            # Simple regex for .bag file, allowing for potential whitespace or newlines
            # Find ALL .bag links (Ellipsoid + MLLW)
            # href="..." or href='...'
            matches = re.findall(r'href=["\']?([^"\'>\s]+\.bag)["\']?', r_scrape.text, re.IGNORECASE)

            bag_links: list[str] = []
            for m in matches:
                # Cleanup logic
                link = m
                if link.startswith("//"):
                    link = "https:" + link
                elif not link.startswith("http"):
                    from urllib.parse import urljoin

                    link = urljoin(download_url, link)
                bag_links.append(link)

            final_urls: list[str] = []

            # Prioritize MLLW (Tidal Datum usually matches user expectation)
            # Deprioritize Ellipsoid because we lack robust Geoid separation grids.
            mllw_links = [L for L in bag_links if "_MLLW_" in L or "MLLW" in L]
            ellip_links = [L for L in bag_links if "_Ellipsoid_" in L or "Ellipsoid" in L]

            if mllw_links:
                logger.info(f"Preferred MLLW BAGs found: {len(mllw_links)} files")
                final_urls = sorted(list(set(mllw_links)))
            elif ellip_links:
                logger.info(
                    "Selecting Ellipsoid BAGs (Fallback, may require large Geoid offset): "
                    f"{len(ellip_links)} files"
                )
                final_urls = sorted(list(set(ellip_links)))
            elif bag_links:
                final_urls = sorted(list(set(bag_links)))

            if final_urls:
                # 2. Update Cache
                cls._update_redirect_cache(download_url, final_urls)
                return final_urls

        except Exception as e:
            logger.warning(f"Scraping failed for {download_url}: {e}")
        return []

    @classmethod
    @concurrent_lru_cache()
    def find_bag_by_survey_id(cls, survey_id: str) -> list[str]:
        """
        Query NCEI API for a specific Survey ID (e.g., 'H13385') to get the BAG download URL(s).
        """
        headers = {"User-Agent": USER_AGENT}
        # Try to clean Survey ID (e.g. H13385_MB_... -> H13385)
        clean_id = survey_id
        import re

        # Look for standard pattern: 1 letter + 5 digits
        match = re.search(r"([A-Z]\d{5})", survey_id)
        if match:
            clean_id = match.group(1)
            if clean_id != survey_id:
                logger.info(f"Cleaned Survey ID: {survey_id} -> {clean_id}")

        params = {
            "where": f"SURVEY_ID='{clean_id}' AND BAGS_EXIST='Y'",
            "outFields": "SURVEY_ID,DOWNLOAD_URL",
            "returnGeometry": "false",
            "f": "json",
        }

        try:
            logger.info(f"Querying NCEI for BAG Survey: {clean_id}")
            resp = requests.get(cls.QUERY_URL, params=params, headers=headers, timeout=10)
            logger.info(f"NCEI API Request URL: {resp.url}")
            resp.raise_for_status()
            data = resp.json()

            if "features" in data and len(data["features"]) > 0:
                attr = data["features"][0]["attributes"]
                download_url = attr.get("DOWNLOAD_URL")

                if download_url and download_url.lower().endswith(".bag"):
                    return [str(download_url)]

                if download_url and download_url.lower().endswith(".html"):
                    return cls._scrape_landing_page(download_url)

                # Default return
                return [str(download_url)] if download_url else []
            else:
                logger.warning(f"No features in response: {data}")

            logger.warning(f"No NCEI record found for Survey ID: {survey_id}")
            return []

        except Exception as e:
            logger.error(f"BAG Discovery Field: {e}")
            return []

    @classmethod
    @concurrent_lru_cache()
    def find_bag_by_location(cls, lat: float, lon: float) -> list[str]:
        """
        Query NCEI API by location (spatial intersection).
        Useful when Survey ID resolving fails or matches a dataset not in the NCEI ID index.
        """
        try:
            from pyproj import Transformer

            # Project to Web Mercator (EPSG:3857) as the service uses it
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
            x, y = transformer.transform(lon, lat)

            headers = {"User-Agent": USER_AGENT}

            # Query Layer 0 (Surveys with BAGs) using geometry intersection
            # Use JSON geometry for robustness
            geo_json = f'{{"x":{x},"y":{y},"spatialReference":{{"wkid":3857}}}}'

            params = {
                "geometry": geo_json,
                "geometryType": "esriGeometryPoint",
                "spatialRel": "esriSpatialRelIntersects",
                "where": "BAGS_EXIST='Y'",
                "outFields": "SURVEY_ID,DOWNLOAD_URL",
                "returnGeometry": "false",
                "f": "json",
            }

            logger.info(f"Querying NCEI by Location: {lat}, {lon} -> {x:.1f}, {y:.1f}")
            resp = requests.get(cls.QUERY_URL, params=params, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            if "features" in data and len(data["features"]) > 0:
                attr = data["features"][0]["attributes"]
                survey_id = attr.get("SURVEY_ID")
                download_url = attr.get("DOWNLOAD_URL")

                logger.info(f"Spatial Query found Survey: {survey_id}")

                if download_url:
                    if download_url.lower().endswith(".bag"):
                        return [str(download_url)]
                    if download_url.lower().endswith(".html"):
                        return cls._scrape_landing_page(download_url)
            else:
                logger.info(f"Spatial Query - No features found (Standard Fallback). Response: {data}")
        except Exception as e:
            logger.warning(f"Spatial query failed: {e}")

        return []

    @classmethod
    def find_bags_by_bbox(
        cls,
        west: float,
        south: float,
        east: float,
        north: float,
        crs: str = "EPSG:4326",
        target_resolution: float | None = None,
    ) -> list[str]:
        """
        Queries NCEI for BAGs intersecting the bounding box.
        Returns list of download URLs.
        Results are persisted to disk so subsequent calls (including across server restarts)
        return immediately without a network round-trip.

        If target_resolution is provided, deduplicates multi-resolution exports
        of the same survey at the discovery layer, before any files are fetched.
        """
        # OPTIMIZATION: Snap to a 0.01 degree grid to ensure high cache hit rate
        # regardless of small floating point or halo variations.
        grid = 0.01
        swest = np.floor(west / grid) * grid
        ssouth = np.floor(south / grid) * grid
        seast = np.ceil(east / grid) * grid
        snorth = np.ceil(north / grid) * grid

        crs_tag = crs.replace(":", "_")
        bbox_key = f"{round(swest, 4)}_{round(ssouth, 4)}_{round(seast, 4)}_{round(snorth, 4)}_{crs_tag}"

        # 1. Check disk cache (survives server restarts and process recycling) (Fast Path)
        cached = cls._get_from_discovery_cache(bbox_key)
        if cached is not None:
            logger.debug(f"BAG Discovery Cache Hit: {bbox_key} ({len(cached)} urls)")
            if target_resolution and cached:
                return _dedup_survey_resolutions(cached, target_resolution)
            return cached

        # 2. Cache Miss - Synchronization Barrier
        # Use a granular file lock to prevent cache stampedes query the same bbox simultaneously.
        lock_dir = cls.DISCOVERY_CACHE_PATH.parent / "locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_path = lock_dir / f"{bbox_key}.lock"

        found_urls: list[str] = []
        query_succeeded = False

        try:
            with FileLock(lock_path):
                # Double-check cache inside lock (in case another process just finished)
                cached = cls._get_from_discovery_cache(bbox_key)
                if cached is not None:
                    logger.debug(f"BAG Discovery Cache Hit (Post-Lock): {bbox_key}")
                    if target_resolution and cached:
                        return _dedup_survey_resolutions(cached, target_resolution)
                    return cached

                logger.debug(f"BAG Discovery Cache Miss: {bbox_key} — querying NCEI ArcGIS API")

                # ... Network Query Logic ...
                from pyproj import Transformer

            # MapServer expects EPSG:3857 (Web Mercator)
            transformer = Transformer.from_crs(crs, "EPSG:3857", always_xy=True)
            minx, miny = transformer.transform(swest, ssouth)
            maxx, maxy = transformer.transform(seast, snorth)

            # CRITICAL FIX: Buffer the query envelope to ensure intersections with large,
            # complex polygons (like W00387) even if the tile is fully interior.
            # Without this buffer, the ArcGIS API spatial index often fails to return
            # containment results for small query windows inside search features.
            # Buffer by ~5km (in EPSG:3857 units).
            buffer_meters = 5000.0
            minx -= buffer_meters
            miny -= buffer_meters
            maxx += buffer_meters
            maxy += buffer_meters

            headers = {"User-Agent": USER_AGENT}

            # Envelope Query
            geo_json = (
                f'{{"xmin":{minx},"ymin":{miny},"xmax":{maxx},'
                f'"ymax":{maxy},"spatialReference":{{"wkid":3857}}}}'
            )

            params = {
                "geometry": geo_json,
                "geometryType": "esriGeometryEnvelope",
                "spatialRel": "esriSpatialRelIntersects",
                "where": "BAGS_EXIST='Y'",
                "outFields": "SURVEY_ID,DOWNLOAD_URL,SURVEY_YEAR,DATE_SURVEY_END",
                "returnGeometry": "false",
                "f": "json",
            }

            logger.info(f"Querying NCEI by BBox: {swest},{ssouth},{seast},{snorth}")
            resp = requests.get(
                cls.QUERY_URL,
                params=params,
                headers=headers,
                timeout=(10, BAG_DISCOVERY_TIMEOUT_SECONDS),
            )
            resp.raise_for_status()
            data = resp.json()
            query_succeeded = True

            if "features" in data:
                import re

                # Collect valid BAGs with their sort key (Resolution, Date)
                # Primary Sort: Resolution (Ascending, so 0.5 < 1.0)
                # Secondary Sort: Date (Descending, so Newest first)
                found_bags: list[tuple[str, float, int]] = []  # (URL, Resolution, DateTimestamp)

                for feature in data["features"]:
                    attr = feature.get("attributes", {})
                    download_url = attr.get("DOWNLOAD_URL")
                    # Prefer precise end date (timestamp), fallback to year (int), fallback to 0
                    d_end = attr.get("DATE_SURVEY_END")
                    d_year = attr.get("SURVEY_YEAR")
                    sort_date = 0

                    try:
                        if d_end is not None:
                            sort_date = int(d_end)
                        elif d_year is not None:
                            sort_date = (int(d_year) - 1970) * 31536000000
                    except Exception:
                        pass

                    # Extract Resolution from URL/Filename
                    # e.g. H13384_MB_50cm_MLLW_1of1.bag
                    resolution_val = 100.0  # Default low priority
                    if download_url:
                        fname = download_url.split("/")[-1].lower()
                        # Match _50cm_ or _0.5m_
                        m_cm = re.search(r"[._-](\d+(?:\.\d+)?)cm[._-]", fname)
                        m_m = re.search(r"[._-](\d+(?:\.\d+)?)m[._-]", fname)

                        if m_cm:
                            resolution_val = float(m_cm.group(1)) / 100.0
                        elif m_m:
                            resolution_val = float(m_m.group(1))

                    # Tuple: (URL, Res, Date)
                    if download_url:
                        final_urls = []
                        if download_url.lower().endswith(".bag"):
                            final_urls = [str(download_url)]
                        elif download_url.lower().endswith(".html"):
                            scraped = cls._scrape_landing_page(download_url)
                            if scraped:
                                final_urls = scraped

                        for url in final_urls:
                            # Re-check resolution on final URLs (in case scraping found variations)
                            fname = url.split("/")[-1].lower()
                            res_final = resolution_val
                            # If scraped, re-calculate resolution from actual filename
                            m_cm = re.search(r"[._-](\d+(?:\.\d+)?)cm[._-]", fname)
                            m_m = re.search(r"[._-](\d+(?:\.\d+)?)m[._-]", fname)
                            if m_cm:
                                res_final = float(m_cm.group(1)) / 100.0
                            elif m_m:
                                res_final = float(m_m.group(1))

                            found_bags.append((url, res_final, sort_date))
                            logger.debug(f"Discovered BAG: {fname} | Res: {res_final}m | Date: {sort_date}")

                # Sort:
                # 1. Resolution ASC (Smallest = Best)
                # 2. Date DESC (Largest = Newest = Best) -> Use negative
                found_bags.sort(key=lambda x: (x[1], -x[2]))

                # Extract just the URLs for compatibility
                # Deduplicate while preserving order
                seen = set()
                found_urls = []
                for url, _, _ in found_bags:
                    if url not in seen:
                        found_urls.append(url)
                        seen.add(url)

            logger.info(f"Found {len(found_urls)} BAGs in BBox (Sorted by Date).")

        except Exception as e:
            logger.warning(f"BBox query failed: {e}")

        # 2. Persist to disk only when query succeeded.
        # Avoid caching transient network failures as permanent empty results.
        if query_succeeded:
            # Re-acquire lock for writing? No, _update_discovery_cache handles its own locking.
            cls._update_discovery_cache(bbox_key, found_urls)
            logger.debug(f"BAG Discovery Cache Created: {bbox_key} ({len(found_urls)} urls persisted)")
        else:
            logger.warning(f"BAG Discovery Cache Skipped: {bbox_key} (query failed)")

        # Apply survey dedup at discovery layer so pruned tiles are never
        # fetched, file-locked, or loaded into memory.
        if target_resolution and found_urls:
            found_urls = _dedup_survey_resolutions(found_urls, target_resolution)

        return found_urls


class BAGProvider(Provider):
    """
    Manages downloading, caching, and reading of NOAA BAG files.
    """

    # Singleton instance to avoid re-initializing discovery caches
    _singleton: ClassVar["BAGProvider | None"] = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "BAGProvider":
        if cls._singleton is None:
            cls._singleton = super().__new__(cls)
        return cast(BAGProvider, cls._singleton)

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        """
        Initialize the BAG provider.

        Args:
            cache_dir: Directory to store cached data files.
        """
        if getattr(self, "_initialized", False):
            return

        if cache_dir is None:
            cache_dir = get_cache_root()

        self.cache_dir = Path(cache_dir).expanduser() / "ncei_bag"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "zarr").mkdir(exist_ok=True)  # Create Zarr subdir
        self.vdatum = VDatumResolver()
        self._initialized = True

    def fetch_layer(
        self,
        bbox: tuple[float, float, float, float],
        resolution: float | None = None,
        crs: str = "EPSG:4326",
        **kwargs: Any,
    ) -> xr.DataArray | xr.Dataset:
        """
        Fetches and merges BAG files intersecting the bounding box.
        """
        from .base import matches_pattern

        west, south, east, north = bbox

        filter_opts = kwargs.get("filter", {})
        include_patterns = filter_opts.get("include_patterns")
        exclude_patterns = filter_opts.get("exclude_patterns")
        max_interpolation_gap_m = filter_opts.get("max_interpolation_gap_m")

        # 1. Discover BAGs (dedup applied at discovery layer to skip
        # network/filelock for pruned multi-resolution exports)
        urls = BAGDiscovery.find_bags_by_bbox(west, south, east, north, crs=crs, target_resolution=resolution)

        # 1.5 Apply declarative pattern filtering
        if urls and include_patterns:
            urls = [u for u in urls if matches_pattern(u.split("/")[-1], include_patterns)]
        if urls and exclude_patterns:
            urls = [u for u in urls if not matches_pattern(u.split("/")[-1], exclude_patterns)]

        if not urls:
            raise ProviderNoDataError(f"No BAG files found for bbox {bbox}")

        # 1.6 Re-sort by (resolution ASC, survey_id DESC) before processing.
        # The discovery cache may store URLs in NCEI API order (arbitrary) if
        # populated before sorting was introduced, or on a cache hit which bypasses
        # the sort at write time. Since the merge is first-in-list-wins (fill-NaN-only),
        # sort order here determines which survey wins overlapping pixels — finest
        # resolution must come first. Survey filename DESC is a reliable recency proxy
        # within the same resolution (higher NOAA survey numbers are more recent).
        import re as _re

        def _resolution_from_url(u: str) -> float:
            fname = u.split("/")[-1].lower()
            m = _re.search(r"[._-](\d+(?:\.\d+)?)cm[._-]", fname)
            if m:
                return float(m.group(1)) / 100.0
            m = _re.search(r"[._-](\d+(?:\.\d+)?)m[._-]", fname)
            if m:
                return float(m.group(1))
            return 100.0  # unknown resolution → lowest priority

        # Two-pass stable sort: secondary key first, then primary (Python stable sort preserves order).
        urls = sorted(urls, key=lambda u: u.split("/")[-1].lower(), reverse=True)  # survey id DESC
        # resolution ASC (stable: preserves secondary within ties)
        urls = sorted(urls, key=_resolution_from_url)

        logger.info(
            f"BAG fetch: {len(urls)} files (sorted finest-first): "
            + ", ".join(u.split("/")[-1] for u in urls[:5])
            + ("..." if len(urls) > 5 else "")
        )

        def _log_mem(label: str) -> None:
            """Log current process RSS for memory debugging."""
            try:
                import resource

                rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                logger.debug(f"[MEM] {label}: peak RSS {rss_kb / 1024:.0f} MB")
            except Exception:
                pass

        # 2. Fetch, clip, reproject, and merge each tile incrementally.
        # Previous approach accumulated all clipped tiles in a list before merging,
        # which OOM'd for dense survey areas (30+ tiles x ~800MB each = 24GB+).
        # Now each tile is merged into an accumulator immediately and freed.
        #
        # The accumulator is pre-initialized to cover the FULL cell bbox at the
        # target resolution. Without this, the first tile's smaller footprint would
        # become the accumulator grid and all subsequent tiles would be clipped to it
        # via reproject_match, silently dropping coverage outside the first tile.
        merged_ds: xr.Dataset | None = None
        if crs and resolution:
            try:
                import math

                if "EPSG:4326" in crs:
                    _x0, _y0 = west, south
                    _x1, _y1 = east, north

                    _center_lat_rad = math.radians((south + north) / 2.0)
                    _res_deg_y = resolution / 111111.0
                    _res_deg_x = resolution / (111111.0 * math.cos(_center_lat_rad))
                else:
                    from pyproj import Transformer

                    _t = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
                    _x0, _y0 = _t.transform(west, south)
                    _x1, _y1 = _t.transform(east, north)
                    _res_deg_x = resolution
                    _res_deg_y = resolution

                _x0, _x1 = min(_x0, _x1), max(_x0, _x1)
                _y0, _y1 = min(_y0, _y1), max(_y0, _y1)

                _nx = max(1, round((_x1 - _x0) / _res_deg_x))
                _ny = max(1, round((_y1 - _y0) / _res_deg_y))
                _xs = np.linspace(_x0 + _res_deg_x / 2, _x1 - _res_deg_x / 2, _nx)
                _ys = np.linspace(_y1 - _res_deg_y / 2, _y0 + _res_deg_y / 2, _ny)  # N→S

                _empty_elev = xr.DataArray(
                    np.full((1, _ny, _nx), np.nan, dtype=np.float32),
                    dims=["band", "y", "x"],
                    coords={"band": [1], "y": _ys, "x": _xs},
                )
                _empty_src = xr.DataArray(
                    np.zeros((1, _ny, _nx), dtype=np.uint32),
                    dims=["band", "y", "x"],
                    coords={"band": [1], "y": _ys, "x": _xs},
                )
                _empty_elev.rio.write_crs(crs, inplace=True)
                _empty_src.rio.write_crs(crs, inplace=True)
                merged_ds = xr.Dataset({"elevation": _empty_elev, "source_id": _empty_src})
                logger.debug(f"BAG accumulator pre-initialized: {_nx}x{_ny} @ {resolution}m ({crs})")
            except Exception as _e:
                logger.debug(f"BAG accumulator pre-init skipped: {_e}")

        provenance_dict: dict[int, dict[str, str]] = {}
        tiles_merged = 0
        _log_mem(f"BAG fetch start ({len(urls)} urls)")

        for url in urls:
            try:
                # Load (Lazy/Zarr if available)
                da = self.fetch_bag(survey_id="unknown_bbox_fetch", download_url=url)
                if da is None:
                    logger.warning(f"BAG fetch: Failed to load data from {url}")
                    continue

                # OPTIMIZATION: Windowed read via rasterio instead of Dask clip.
                # Computes the pixel window from bounds + transform, reads only
                # those pixels from the zarr-backed array. No Dask task graph,
                # deterministic memory = output size.
                try:
                    da_work = _windowed_read(da, west, south, east, north)
                except _EmptyWindowError:
                    logger.debug(f"BAG tile empty after clip (url={url}): no overlap")
                    continue
                except Exception as e:
                    logger.debug(f"BAG tile empty after clip (url={url}): {e}")
                    continue
                finally:
                    # Release the Dask-backed DataArray reference immediately
                    del da

                if da_work is None or da_work.size == 0:
                    continue

                # Reproject individual chunk to target CRS
                if crs and da_work.rio.crs and str(da_work.rio.crs) != crs:
                    try:
                        reproj_knn: dict[str, Any] = {}
                        if resolution and "EPSG:4326" not in crs:
                            reproj_knn["resolution"] = resolution

                        da_work = cast(
                            xr.DataArray,
                            da_work.rio.reproject(crs, num_threads=4, **reproj_knn),
                        )
                    except Exception as e:
                        logger.warning(f"Reprojection failed for BAG segment: {e}")
                        continue

                # --- CLEANING / FILTERING ---
                filter_cfg = kwargs.get("filter", {})
                max_dev = filter_cfg.get("max_depth_change") or filter_cfg.get("max_deviation")
                if max_dev:
                    threshold = float(max_dev)
                    if _depth_range_within_threshold(da_work, threshold):
                        logger.info(
                            f"Skipping BAG Deviation Filter (Threshold={threshold}m): "
                            f"depth range within threshold for {url}"
                        )
                    else:
                        logger.info(f"Applying BAG Deviation Filter (Threshold={threshold}m) to {url}")
                        da_work = clean_data_deviation(da_work, threshold=threshold)

                # --- Provenance ---
                import hashlib
                import re

                filename = url.split("/")[-1].replace(".bag", "")
                project_uid = int(hashlib.md5(filename.encode()).hexdigest(), 16) % 100000 + 40000

                res_str = "Unknown"
                m_cm = re.search(r"[._-](\d+(?:\.\d+)?)cm[._-]", filename, re.IGNORECASE)
                m_m = re.search(r"[._-](\d+(?:\.\d+)?)m[._-]", filename, re.IGNORECASE)
                if m_cm:
                    res_str = f"{m_cm.group(1)}cm"
                elif m_m:
                    res_str = f"{m_m.group(1)}m"

                provenance_dict[project_uid] = {
                    "name": filename,
                    "provider": "ncei_bag",
                    "resolution": res_str,
                }

                da_work.name = "elevation"
                p_source = xr.where(da_work.notnull(), project_uid, 0).astype(np.uint32)
                p_source.name = "source_id"
                p_source.rio.write_nodata(0, inplace=True)
                p_source.attrs["_FillValue"] = 0

                p_ds = xr.Dataset({"elevation": da_work, "source_id": p_source})
                if da_work.rio.crs:
                    p_ds.rio.write_crs(da_work.rio.crs, inplace=True)
                p_ds.rio.write_transform(da_work.rio.transform(), inplace=True)
                del da_work

                # --- Incremental merge: fold this tile into the accumulator ---
                # Strategy: reproject each incoming tile onto the accumulator's fixed
                # coordinate grid, then fill NaN holes via in-place numpy masking.
                # This avoids merge_arrays(), which allocates a full output copy of
                # the union extent alongside both inputs — 3x peak memory per merge step.
                tiles_merged += 1
                if merged_ds is None:
                    merged_ds = p_ds
                else:
                    try:
                        import gc

                        from rasterio.enums import Resampling

                        # Align incoming tile to the accumulator grid. reproject_match
                        # uses the accumulator's exact pixel grid so no union expansion occurs.
                        p_elev_aligned = p_ds["elevation"].rio.reproject_match(
                            merged_ds["elevation"],
                            resampling=Resampling.bilinear,
                        )
                        # Pixels where accumulator is NaN and tile has valid data
                        fill_mask = (
                            np.isnan(merged_ds["elevation"].values) & np.isfinite(p_elev_aligned.values)
                        ).squeeze()
                        merged_ds["elevation"].values.squeeze()[fill_mask] = p_elev_aligned.values.squeeze()[
                            fill_mask
                        ]
                        del p_elev_aligned
                        gc.collect()

                        # Same for source_id — nearest resampling preserves integer IDs
                        try:
                            p_src_aligned = p_ds["source_id"].rio.reproject_match(
                                merged_ds["source_id"],
                                resampling=Resampling.nearest,
                            )
                            merged_ds["source_id"].values.squeeze()[fill_mask] = (
                                p_src_aligned.values.squeeze()[fill_mask].astype(np.uint32)
                            )
                            del p_src_aligned
                        except Exception as e_src:
                            logger.warning(
                                f"Source ID merge failed for {filename} (elevation merged OK): {e_src}"
                            )
                        del fill_mask
                    except Exception as e:
                        logger.error(f"Incremental merge failed for {filename}, tile skipped: {e}")
                del p_ds

                if tiles_merged % 10 == 0:
                    _log_mem(f"BAG progress: {tiles_merged} tiles merged")

            except Exception as e:
                logger.warning(f"Failed to process BAG {url}: {e}")

        if merged_ds is None:
            raise ProviderNoDataError(f"No BAG data intersects bbox {bbox} after clipping")

        _log_mem(f"BAG done: {tiles_merged} tiles merged")

        # Defensive: Drop duplicate indices from merge
        for var in ["elevation", "source_id"]:
            if var not in merged_ds:
                continue
            da_var = merged_ds[var]
            for dim in ["x", "y"]:
                if dim in da_var.indexes and not da_var.indexes[dim].is_unique:
                    merged_ds[var] = da_var.drop_duplicates(dim=dim)
                    da_var = merged_ds[var]

        # 4. Final Clip (Cleanup)
        # Ensure exact bounds (reprojection might have introduced slight over-run)
        with contextlib.suppress(Exception):
            final_elev = merged_ds["elevation"].rio.clip_box(
                minx=west,
                miny=south,
                maxx=east,
                maxy=north,
                crs="EPSG:4326",
                allow_one_dimensional_raster=True,
            )
            final_src = merged_ds["source_id"].rio.clip_box(
                minx=west,
                miny=south,
                maxx=east,
                maxy=north,
                crs="EPSG:4326",
                allow_one_dimensional_raster=True,
            )
            merged_ds = xr.Dataset({"elevation": final_elev, "source_id": final_src})

        if max_interpolation_gap_m and resolution:
            merged_ds["elevation"] = interpolate_small_gaps(
                merged_ds["elevation"], max_interpolation_gap_m, resolution
            )

        merged_ds.attrs["provenance_dict"] = provenance_dict
        logger.debug(f"Found NCEI BAG Coverage with provenance {provenance_dict}")
        return cast(xr.Dataset, merged_ds)

    def get_metadata(self) -> dict[str, Any]:
        """
        Return metadata for the BAG provider.

        Returns:
            Dictionary containing provider name, citation, resolution, and URL.
        """
        return {
            "name": "NOAA NCEI BAG (Bathymetric Attributed Grid)",
            "citation": "NOAA National Centers for Environmental Information.",
            "resolution": "High (Variable, typically 0.5m - 4m)",
            "url": "https://www.ncei.noaa.gov/products/bathymetry",
        }

    def fetch_dataset(self, url: str, bbox: tuple[float, float, float, float]) -> xr.DataArray | None:
        raw_da = self.fetch_bag(url.split("/")[-1].replace(".bag", ""), url)
        if raw_da is None:
            return None
        try:
            clipped = raw_da.rio.clip_box(*bbox, crs="EPSG:4326")
            return cast(xr.DataArray, clipped)
        except Exception:
            return None

    def fetch_bag(self, survey_id: str, download_url: str | list[str] | None = None) -> xr.DataArray | None:
        """
        Fetches and reads a BAG file for a given Survey ID.
        Auto-discovers URL if not provided. Supports multiple BAG files for a single survey.
        """
        if not download_url:
            download_url = BAGDiscovery.find_bag_by_survey_id(survey_id)

        if not download_url:
            return None

        # Normalize to list
        urls = [download_url] if isinstance(download_url, str) else download_url

        loaded_arrays = []

        for url in urls:
            filename = url.split("/")[-1]
            local_path = self.cache_dir / filename

            # Fast-path cache check: if Zarr exists, we don't need the massive .bag file anymore
            zarr_dir = local_path.parent / "zarr"
            zarr_path = zarr_dir / local_path.name.replace(".bag", "_navd88.zarr")

            if zarr_path.exists():
                da = self._read_bag(local_path)
            else:
                downloaded_path = self._ensure_downloaded(url)
                if downloaded_path is not None:
                    da = self._read_bag(downloaded_path)
                else:
                    continue

            if da is not None:
                loaded_arrays.append(da)

        if not loaded_arrays:
            return None

        if len(loaded_arrays) == 1:
            return loaded_arrays[0]

        # Merge if multiple
        try:
            from rioxarray.merge import merge_arrays

            # Sort by resolution (finest first) to preserve detail
            # abs(res[0]) is pixel width. Smallest pixel width = highest resolution.
            loaded_arrays.sort(key=lambda da: abs(da.rio.resolution()[0]))

            logger.info(f"Merging {len(loaded_arrays)} BAG segments for {survey_id}")
            # merge_arrays returns a DataArray by default if inputs are DataArrays
            merged = merge_arrays(loaded_arrays)
            merged.attrs["survey_source"] = f"{survey_id} (Merged {len(loaded_arrays)})"
            return cast(xr.DataArray, merged)
        except Exception as e:
            logger.error(f"Failed to merge BAGs for {survey_id}: {e}")
            # Return the first one as best effort
            return loaded_arrays[0]

    def _ensure_downloaded(self, url: str) -> Path | None:
        """Helper to download a single BAG file if missing."""
        filename = url.split("/")[-1]
        local_path = self.cache_dir / filename

        # 1. Download if missing
        if not local_path.exists():
            import fcntl

            lock_path = self.cache_dir / f"{filename}.lock"
            temp_path = self.cache_dir / f".tmp_{filename}"

            try:
                with open(lock_path, "w") as lock_file:
                    fcntl.flock(lock_file, fcntl.LOCK_EX)
                    if local_path.exists():
                        pass  # Double check inside lock
                    else:
                        logger.info(f"Downloading BAG: {url}")
                        # Use requests for progress? Or urllib/shutil for simplicity
                        # Since BAGs are large, streaming via requests is better
                        with requests.get(url, stream=True) as r:
                            r.raise_for_status()
                            total_size = int(r.headers.get("content-length", 0))
                            with open(temp_path, "wb") as f:
                                for chunk in r.iter_content(chunk_size=32768):
                                    f.write(chunk)

                            # Verify size if content-length was provided
                            if total_size > 0 and temp_path.stat().st_size != total_size:
                                raise OSError(
                                    f"Incomplete download: {temp_path.stat().st_size}/{total_size} bytes"
                                )

                        Path(temp_path).rename(local_path)

                    # Cleanup Lock if successful
                    lock_path.unlink(missing_ok=True)
            except Exception as e:
                logger.error(f"Failed to download BAG {url}: {e}")
                if Path(temp_path).exists():
                    Path(temp_path).unlink()
                return None

        return local_path

    def _read_bag(self, local_path: Path) -> xr.DataArray | None:
        """wrapper to call standalone cached function."""
        # Use standalone function to leverage python LRU cache safely outside class methods.
        return cast("xr.DataArray | None", _read_bag_cached(local_path))


# Register the provider
registry.register(Path(__file__).stem, BAGProvider)
