"""
NASADEM Provider module.

This module implements the provider for NASADEM global elevation data.
It queries the Microsoft Planetary Computer STAC API for the 'nasadem' collection.
"""

import logging
import random
import time
from pathlib import Path
from typing import Any, cast

import planetary_computer
import rioxarray
import xarray as xr
from pystac_client import Client
from rioxarray.merge import merge_arrays

from ..config import get_cache_root
from ..manifest import OfflineManifest
from .base import Provider, ProviderNoDataError, sanitize_elevation_nodata
from .registry import registry

logger = logging.getLogger(__name__)


class NasademProvider(Provider):
    """
    Provider for NASADEM Topography (30m)
    """

    _instance = None
    _initialized = False

    def __new__(cls, *args: Any, **kwargs: Any) -> "NasademProvider":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cast(NasademProvider, cls._instance)

    def __init__(self, cache_dir: str | Path | None = None, offline_mode: bool = False) -> None:
        """
        Initialize the NASADEM provider.

        Args:
            cache_dir: Directory to store cached data files.
            offline_mode: If True, only use locally cached data/manifests.
        """
        if self._initialized:
            return

        if cache_dir is None:
            cache_dir = get_cache_root()

        base_cache = Path(cache_dir).expanduser()
        self.cache_dir = base_cache / "nasadem"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Metadata Cache Directory
        self.metadata_dir = base_cache / "metadata"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        self.offline_mode = offline_mode

        # Manifest for Offline Lookup
        self.manifest = OfflineManifest(self.metadata_dir, filename="stac_manifest.json")
        self._initialized = True

    def fetch_layer(
        self,
        bbox: tuple[float, float, float, float],
        resolution: float | None = None,
        crs: str = "EPSG:4326",
        **kwargs: Any,
    ) -> xr.Dataset:
        """
        Fetch NASADEM data for the given bounding box.
        """
        # Ensure the bbox is in EPSG:4326 for STAC/Manifest queries
        bounds = self._normalize_bbox(bbox, crs)

        da = self._fetch_collection(bounds, "nasadem")
        if da is not None:
            logger.debug("Found NASADEM Coverage")
            return da

        raise ProviderNoDataError(f"No NASADEM coverage found for bbox {bbox}")

    def get_grid(
        self,
        west: float,
        south: float,
        east: float,
        north: float,
        target_shape: tuple[int, int] | None = None,
    ) -> xr.DataArray | xr.Dataset | None:
        """
        Unified access method for Manager compatibility.
        """
        # Maintain get_grid compatibility (it shouldn't throw KeyError anymore but handled natively)
        return self.fetch_layer(bbox=(west, south, east, north))

    def get_metadata(self) -> dict[str, Any]:
        """
        Return metadata for the NASADEM provider.

        Returns:
            Dictionary containing provider name, citation, resolution, and URL.
        """
        return {
            "name": "NASADEM",
            "citation": "NASA JPL.",
            "resolution": "30m (1 arc-second)",
            "url": "https://planetarycomputer.microsoft.com/dataset/nasadem",
        }

    def _query_stac_api(
        self,
        bbox: tuple[float, float, float, float],
        collection_id: str,
    ) -> list[dict] | None:
        """
        Query STAC API for Items. Handles Negative Caching and Locking.
        """
        if self.offline_mode:
            return None

        # 1. Check Negative Cache (No Coverage)
        if self.manifest.has_no_coverage(collection_id, bbox):
            logger.debug(f"Negative Cache Hit: {collection_id} is known empty for {bbox}")
            return None

        # Query-Specific Concurrency Lock
        import hashlib

        from filelock import FileLock

        query_key = f"{collection_id}_{bbox}"
        query_hash = hashlib.md5(query_key.encode()).hexdigest()
        lock_file_path = self.cache_dir / f"stac_query_{query_hash}.lock"

        stac_url = "https://planetarycomputer.microsoft.com/api/stac/v1"
        max_retries = 2

        try:
            with FileLock(lock_file_path):
                # Double-Check Negative Cache (in case another worker updated it while we waited)
                if self.manifest.has_no_coverage(collection_id, bbox):
                    return None

                # Also Double-Check Positive Cache/Manifest?
                # Ideally yes, but the Manifest is loaded in memory mainly.
                # If 'manifest.record_search' updates persistent storage, we should reload it.
                # Assuming manifest handles its own persistence or we rely on the negative cache check.

                for attempt in range(max_retries + 1):
                    try:
                        logger.debug(
                            f"Usgs3DepProvider querying {collection_id} for {bbox} (Attempt {attempt + 1})"
                        )

                        catalog = Client.open(stac_url, modifier=planetary_computer.sign_inplace)
                        search = catalog.search(collections=[collection_id], bbox=bbox, limit=10)
                        items = list(search.items())

                        # Record Search Result (Positive or Negative)
                        self.manifest.record_search(collection_id, bbox, len(items))

                        if not items:
                            return None

                        results = []
                        for item in items:
                            # Extract asset href
                            asset_key = "data"
                            if asset_key not in item.assets:
                                if "elevation" in item.assets:
                                    asset_key = "elevation"
                                elif "merged" in item.assets:
                                    asset_key = "merged"

                            if asset_key in item.assets:
                                results.append(
                                    {
                                        "href": item.assets[asset_key].href,
                                        "bbox": item.bbox,
                                        "properties": item.properties,
                                    }
                                )
                        return results

                    except Exception as e:
                        # Retry logic
                        is_last_attempt = attempt == max_retries

                        # Reduce log noise for common network issues
                        err_str = str(e)
                        is_network = any(
                            x in err_str
                            for x in ["NameResolutionError", "ConnectionError", "Max retries exceeded"]
                        )

                        if is_network:
                            msg = (
                                f"Network Error fetching {collection_id} (Attempt {attempt + 1}): "
                                f"{err_str.split('Caused by')[-1].strip()}"
                            )
                        else:
                            msg = f"Error fetching {collection_id} (Attempt {attempt + 1}): {e}"

                        log_level = logging.WARNING if not is_last_attempt else logging.ERROR
                        logger.log(log_level, msg)

                        if is_last_attempt:
                            return None

                        # If network error, maybe wait longer?
                        sleep_time = (2**attempt) + random.uniform(0.1, 1.0)
                        if is_network:
                            sleep_time += 1.0

                        time.sleep(sleep_time)

        except Exception as e:
            logger.error(f"STAC Query Lock/Execution Failed: {e}")
            return None

        return None

    def _fetch_collection(self, bounds: tuple, collection_id: str) -> xr.Dataset | None:
        try:
            items = None

            # 1. Check Manifest (Positive Cache)
            # 1. Check Manifest (Positive Cache) - Logic Update:
            # We trust the Zarr cache over the manifest for DATA.
            # But we need to know WHICH items to look for.
            # The issue is: If we have Zarrs, we don't need to query API.
            # But we don't know the Zarr filenames without the HREFs.
            # So:
            # A) Query API/Manifest to get Items/HREFs.
            # B) For each Item, Check Zarr.
            # C) If Zarr exists, load it. If not, stream & save.

            logger.debug(f"Checking Local Manifest for {collection_id} in {bounds}")
            manifest_items = self.manifest.find_items(collection_id, bounds)

            if manifest_items:
                # Planetary Computer SAS tokens (sig=, se=, sv=) expire in ~1 hour.
                # If any cached href contains a token signature, skip the manifest and
                # re-query STAC to get fresh signed URLs rather than hitting a guaranteed 403.
                def _has_signed_token(href: str) -> bool:
                    return any(p in href for p in ("sig=", "%3D&se=", "se=", "&sv="))

                stale = [m for m in manifest_items if _has_signed_token(m["href"])]
                if stale:
                    logger.info(
                        f"Manifest contains {len(stale)} SAS-signed URL(s) for {collection_id} "
                        f"— skipping manifest and re-querying STAC for fresh tokens."
                    )
                    for s in stale:
                        self.manifest.remove_item_by_href(s["href"])
                    manifest_items = [m for m in manifest_items if not _has_signed_token(m["href"])]

            if manifest_items:
                items = [
                    {"href": m["href"], "bbox": m["bbox"], "properties": m.get("properties")}
                    for m in manifest_items
                ]
                logger.info(
                    f"Manifest Cache Hit: Found {len(items)} items for {collection_id} (Skipping API)."
                )

            # 2. Query API (if Miss)
            if not items and not self.offline_mode:
                items = self._query_stac_api(bounds, collection_id)

            # Logic continues... Checks Zarr inside the loop now.

            if not items:
                logger.debug(f"Usgs3DepProvider found 0 items for {collection_id}")
                return None

            items_found: list[dict[str, Any]] = []
            das: list[xr.DataArray] = []

            for item in items:
                href = item["href"]
                max_retries = 3

                for retry in range(max_retries + 1):
                    try:
                        logger.debug(f"Streaming Land Asset: {href}")
                        # Open streaming (Lazy)
                        da_raw = rioxarray.open_rasterio(href, chunks={"x": 2048, "y": 2048})

                        if isinstance(da_raw, list):
                            da = cast(xr.DataArray, da_raw[0])
                        elif isinstance(da_raw, xr.Dataset):
                            da = da_raw.to_array().isel(variable=0)
                        else:
                            da = da_raw

                        da = cast(xr.DataArray, da)
                        if "band" in da.dims:
                            da = da.isel(band=0).drop_vars("band")

                        da = sanitize_elevation_nodata(
                            da,
                            extra_sentinels=(-9999.0,),
                            abs_valid_limit=100000.0,
                        )

                        # CRITICAL: Clip to requested bounds before collecting.
                        # Otherwise we would hold the entire 1x1 degree COG in memory.
                        try:
                            from rasterio.warp import transform_bounds

                            source_crs = da.rio.crs
                            clip_bbox = bounds
                            clip_crs = "EPSG:4326"

                            if source_crs and source_crs.to_string() not in ["EPSG:4326", "EPSG:4269"]:
                                try:
                                    clip_bbox = transform_bounds("EPSG:4326", source_crs, *bounds)
                                    clip_crs = source_crs
                                except Exception as e:
                                    logger.warning(
                                        f"Failed to reproject bbox to {source_crs}: {e}. "
                                        f"Retrying with EPSG:4326."
                                    )

                            da = da.rio.clip_box(*clip_bbox, crs=clip_crs, allow_one_dimensional_raster=True)
                        except Exception as e:
                            # No overlap — skip this tile.
                            logger.debug(f"Clip FAILED for {href} with bounds {bounds}: {e}. Skipping item.")
                            continue

                        if da.size > 100_000_000:
                            logger.warning(
                                f"3DEP Asset {href} is unexpectedly large ({da.size} pixels) after clip. "
                                "Skipping to avoid stall."
                            )
                            continue

                        items_found.append(item)
                        das.append(da)
                        break  # success — exit retry loop, move to next item

                    except Exception as e:
                        err_msg = str(e)
                        is_403 = "403" in err_msg or "Forbidden" in err_msg or "Access Denied" in err_msg
                        is_network = any(
                            x in err_msg
                            for x in ["NameResolutionError", "ConnectionError", "Max retries exceeded"]
                        )

                        if is_network:
                            logger.warning(
                                f"Network Error streaming {href}: {err_msg.split('Caused by')[-1].strip()}"
                            )
                            if retry < max_retries:
                                time.sleep(2**retry + random.uniform(0.1, 1.0))
                                continue
                            else:
                                logger.error(f"Failed to stream {href} after ({max_retries}) retries.")

                        elif is_403 and retry < max_retries:
                            logger.warning(f"403 Forbidden for {href}. Refreshing token and retrying...")

                            # 1. Provide Feedback: Remove stale item
                            self.manifest.remove_item_by_href(href)

                            # 2. Refresh: Force Query API
                            # We query the specific item's bbox to get a fresh token for it
                            item_bbox = tuple(item.get("bbox", bounds))
                            fresh_items = self._query_stac_api(item_bbox, collection_id)

                            if fresh_items:
                                # Find the matching item in fresh results (by spatial overlap)
                                # Simple heuristic: replace 'href' with the first fresh item's href
                                # roughly matching location.
                                # Actually, just updating the manifest via _query_stac_api is enough?
                                # No, we need the *new URL* to try opening again.

                                # Let's try to match by bounds check
                                best_match = None
                                from shapely.geometry import box

                                target_box = box(*item_bbox)

                                for fresh in fresh_items:
                                    fresh_box = box(*fresh["bbox"])
                                    if fresh_box.intersects(target_box):
                                        best_match = fresh["href"]
                                        break

                                if best_match:
                                    logger.info(f"Refreshed URL: {best_match}")
                                    href = best_match  # Update loop variable for next try
                                    # Update 'item' dict too so we save valid one to manifest later
                                    item["href"] = best_match
                                    continue  # Retry loop

                            logger.error("Failed to refresh SAS token.")

                        logger.warning(f"Failed to stream land asset {href}: {e}")
                        break

            # Update Manifest only with successfully opened items
            if not self.offline_mode and items_found:
                for item in items_found:
                    self.manifest.add_item(
                        collection_id=collection_id,
                        bbox=item.get("bbox", bounds),
                        asset_href=item["href"],
                        properties=item.get("properties"),
                    )

            if not das:
                logger.debug(f"3DEP/Land items found but resulted in no valid data for {collection_id}.")
                # Negative Cache: Mark this bbox as having no coverage to prevent re-querying
                self.manifest.record_search(collection_id, bounds, 0)
                return None

            # 3. Create Provenance and merge as Dataset
            provenance_dict = {}
            import hashlib

            import numpy as np

            p_das = []
            for item_idx, da in enumerate(das):
                href = items_found[item_idx].get("href", f"unknown_stac_asset_{item_idx}")
                project_uid = int(hashlib.md5(href.encode()).hexdigest(), 16) % 100000 + 60000

                # Try to get a clean name
                asset_name = Path(href.split("?")[0]).name
                if not asset_name or len(asset_name) > 50:
                    asset_name = f"Asset {project_uid}"

                provenance_dict[project_uid] = {
                    "name": f"3DEP: {asset_name}",
                    "provider": "usgs_3dep",
                }

                da.name = "elevation"
                da = sanitize_elevation_nodata(
                    da,
                    extra_sentinels=(-9999.0,),
                    abs_valid_limit=100000.0,
                )
                p_source = xr.where(da.notnull(), project_uid, 0).astype(np.uint32)
                p_source.name = "source_id"
                p_source.rio.write_nodata(0, inplace=True)
                p_source.attrs["_FillValue"] = 0

                p_ds = xr.Dataset({"elevation": da, "source_id": p_source})
                if da.rio.crs:
                    p_ds.rio.write_crs(da.rio.crs, inplace=True)
                p_ds.rio.write_transform(da.rio.transform(), inplace=True)

                p_das.append(p_ds)

            # Merge
            try:
                if len(p_das) == 1:
                    merged = p_das[0]
                else:
                    elevs = [ds["elevation"] for ds in p_das]
                    sources = [ds["source_id"] for ds in p_das]

                    final_elev = merge_arrays(elevs)
                    final_elev = sanitize_elevation_nodata(
                        final_elev,
                        extra_sentinels=(-9999.0,),
                        abs_valid_limit=100000.0,
                    )
                    final_src = merge_arrays(sources)
                    merged = xr.Dataset({"elevation": final_elev, "source_id": final_src})
            except Exception as e:
                logger.error(f"Failed to merge 3DEP STAC items: {e}")
                merged = p_das[0]

            # Ensure CRS
            if not merged["elevation"].rio.crs:
                # STAC Land collections (3DEP/COP-30) are EPSG:4326
                merged["elevation"].rio.write_crs("EPSG:4326", inplace=True)
                merged["source_id"].rio.write_crs("EPSG:4326", inplace=True)

            # Check if effective data remains
            if merged["elevation"].isnull().all():
                logger.debug(f"3DEP/Land data masked out (All Water?) for {collection_id}.")
                self.manifest.record_search(collection_id, bounds, 0)
                return None

            merged.attrs["provenance_dict"] = provenance_dict
            return cast(xr.Dataset, merged)

        except Exception as e:
            logger.error(f"Error fetching {collection_id}: {e}", exc_info=True)
            return None


# Register
registry.register(Path(__file__).stem, NasademProvider)
