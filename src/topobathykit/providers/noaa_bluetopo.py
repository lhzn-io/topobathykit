"""
NOAA BlueTopo Provider module.

This module implements the provider for NOAA's BlueTopo data set, providing high-resolution
bathymetry from modern surveys. It handles S3 access, tile resolution via RAT, and Zarr caching.
"""

import contextlib
import json
import logging
import math
from pathlib import Path
from typing import Any, ClassVar, cast

import fsspec
import geopandas as gpd
import requests  # type: ignore
import rioxarray
import xarray as xr
from affine import Affine
from filelock import FileLock
from rioxarray.merge import merge_arrays
from shapely.geometry import Point, box

from ..config import get_cache_root
from ..quality import QualityClass
from ..runtime import should_consolidate
from ..vdatum import VDatumNoDataError, VDatumResolver
from .base import (
    Provider,
    ProviderFetchError,
    ProviderNoDataError,
    interpolate_small_gaps,
    sanitize_elevation_nodata,
)
from .registry import registry

logger = logging.getLogger(__name__)


class NoaaBlueTopoProvider(Provider):
    """
    Provider for NOAA BlueTopo High-Resolution Bathymetry.
    Accesses Cloud Optimized GeoTIFFs (COGs) from AWS S3.
    Resolves tile IDs using the official BlueTopo Tile Scheme GPKG.
    Implements 'Download-and-Cache' strategy for full tiles.
    """

    BUCKET_BASE = "noaa-ocs-nationalbathymetry-pds/BlueTopo"  # Bucket path for s3fs
    S3_URI_BASE = "s3://noaa-ocs-nationalbathymetry-pds/BlueTopo"

    # Official Tile Scheme URL (Fallback)
    TILE_SCHEME_URL = (
        "https://noaa-ocs-nationalbathymetry-pds.s3.amazonaws.com/"
        "BlueTopo/_BlueTopo_Tile_Scheme/BlueTopo_Tile_Scheme_20260206_112953.gpkg"
    )

    # Persistent cache mapping tile_id -> resolved HTTPS URL.
    # Format: {"BA04XA": "https://noaa-ocs-nationalbathymetry-pds.s3.../BA04XA_20240101.tiff", ...}
    # Stored in: <cache root>/noaa_bluetopo/tile_url_cache.json
    #
    # BlueTopo tile filenames include a date stamp so their S3 keys change when the NOAA
    # team republishes a tile with updated bathymetry.  Invalidate this cache to pick up
    # renamed/republished tile assets:
    #   manage_discovery_cache.py --invalidate noaa_bluetopo
    TILE_URL_CACHE_PATH = get_cache_root() / "noaa_bluetopo" / "tile_url_cache.json"

    _singleton: ClassVar["NoaaBlueTopoProvider | None"] = None
    _initialized: bool
    _CACHE_CRS_ATTR: ClassVar[str] = "_tbs_cache_crs"
    _TARGET_GRID_PAD_PX: ClassVar[int] = 4
    _SEAM_FILL_MAX_PX: ClassVar[int] = 4

    def __new__(cls, *args: Any, **kwargs: Any) -> "NoaaBlueTopoProvider":
        if cls._singleton is None:
            cls._singleton = super().__new__(cls)
        return cast(NoaaBlueTopoProvider, cls._singleton)

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        """
        Initialize the BlueTopo provider.

        Args:
            cache_dir: Directory to store cached data files.
        """
        if getattr(self, "_initialized", False):
            return

        if cache_dir is None:
            cache_dir = get_cache_root()

        self.vdatum = VDatumResolver()
        base_cache = Path(cache_dir).expanduser()
        self.cache_dir = base_cache / "noaa_bluetopo"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "zarr").mkdir(exist_ok=True)  # Create Zarr subdir
        self.scheme_path = self.cache_dir / "BlueTopo_Tile_Scheme.gpkg"
        self._gdf = None
        self._initialized = True

    def _validate_geotransform(self, ds: xr.Dataset | xr.DataArray, label: str = "") -> tuple[bool, str]:
        """
        Validates that a dataset has a valid geotransform.
        Returns (is_valid, diagnostic_message).
        """
        try:
            if not hasattr(ds, "rio"):
                return False, f"{label} missing rio accessor"

            crs = self._dataset_crs(ds)
            if not crs:
                return False, f"{label} missing CRS"

            transform = None
            with contextlib.suppress(Exception):
                transform = ds.rio.transform()

            # Some Zarr reads preserve CRS/transform at variable level but not dataset
            # level. Fall back to elevation transform for validation.
            if (
                (transform is None or transform.a == 0 or transform.e == 0)
                and isinstance(ds, xr.Dataset)
                and "elevation" in ds
            ):
                with contextlib.suppress(Exception):
                    transform = ds["elevation"].rio.transform()

            if transform is None:
                return False, f"{label} missing geotransform"

            # Check for NaN or zero scale
            if (
                transform.a == 0
                or transform.e == 0
                or (transform.a != transform.a)
                or (transform.e != transform.e)
            ):  # NaN check
                return False, f"{label} geotransform scale is zero/NaN: {transform}"

            # Log successful validation
            logger.debug(
                f"{label} geotransform valid: CRS={crs}, "
                f"origin=({transform.c:.6f}, {transform.f:.6f}), "
                f"scale=({transform.a:.6f}, {transform.e:.6f})"
            )
            return True, f"{label} geotransform OK"
        except Exception as e:
            return False, f"{label} geotransform validation error: {e}"

    def _dataset_crs(self, ds: xr.Dataset | xr.DataArray) -> Any | None:
        """Return a reliable CRS for Dataset/DataArray.

        Dataset.rio.crs can be None after merges; in that case we fall back to the
        elevation variable's CRS when present.
        """
        try:
            if hasattr(ds, "rio") and ds.rio.crs:
                return ds.rio.crs
        except Exception:
            pass

        if isinstance(ds, xr.Dataset) and "elevation" in ds:
            with contextlib.suppress(Exception):
                if ds["elevation"].rio.crs:
                    return ds["elevation"].rio.crs
        return None

    def _ensure_dataset_spatial_ref(self, ds: xr.Dataset) -> xr.Dataset:
        """Ensure dataset-level CRS/transform exist, deriving from elevation if needed."""
        try:
            if ds.rio.crs:
                return ds
        except Exception:
            pass

        if "elevation" in ds:
            elev = ds["elevation"]
            with contextlib.suppress(Exception):
                if elev.rio.crs:
                    ds.rio.write_crs(elev.rio.crs, inplace=True)
            with contextlib.suppress(Exception):
                ds.rio.write_transform(elev.rio.transform(), inplace=True)
        return ds

    def _stash_cache_crs(self, ds: xr.Dataset) -> xr.Dataset:
        """Persist CRS in attrs as a fallback for Zarr round-trips."""
        crs = self._dataset_crs(ds)
        if not crs:
            return ds

        crs_text = crs.to_string() if hasattr(crs, "to_string") else str(crs)
        ds.attrs[self._CACHE_CRS_ATTR] = crs_text
        if "elevation" in ds:
            ds["elevation"].attrs[self._CACHE_CRS_ATTR] = crs_text
        if "source_id" in ds:
            ds["source_id"].attrs[self._CACHE_CRS_ATTR] = crs_text
        return ds

    def _restore_cache_crs(self, ds: xr.Dataset) -> xr.Dataset:
        """Restore CRS from fallback attrs when rio metadata is absent after read."""
        if self._dataset_crs(ds):
            return ds

        crs_text = ds.attrs.get(self._CACHE_CRS_ATTR) or ds.get("elevation", xr.DataArray()).attrs.get(
            self._CACHE_CRS_ATTR
        )
        if not crs_text:
            return ds

        with contextlib.suppress(Exception):
            ds.rio.write_crs(crs_text, inplace=True)
        if "elevation" in ds:
            with contextlib.suppress(Exception):
                ds["elevation"].rio.write_crs(crs_text, inplace=True)
        if "source_id" in ds:
            with contextlib.suppress(Exception):
                ds["source_id"].rio.write_crs(crs_text, inplace=True)
        return ds

    def _check_tile_alignment(self, tiles: list[xr.Dataset], label: str = "") -> bool:
        """
        Validates that all tiles in a list have compatible coordinate systems.
        Returns True if all compatible, False if misaligned.
        """
        if not tiles or len(tiles) < 2:
            return True

        try:
            ref_ds = tiles[0]
            ref_crs = ref_ds.rio.crs
            ref_transform = ref_ds.rio.transform()

            for i, ds in enumerate(tiles[1:], start=1):
                # Check CRS match
                if ds.rio.crs != ref_crs:
                    logger.warning(
                        f"{label} Tile {i} CRS mismatch: {ds.rio.crs} != {ref_crs}. Will reproject."
                    )
                    return False

                # Check geotransform
                ds_transform = ds.rio.transform()
                if ds_transform != ref_transform:
                    logger.warning(
                        f"{label} Tile {i} geotransform mismatch: {ds_transform} != {ref_transform}"
                    )
                    # Small pixel-size differences OK, but warn on major differences
                    if (
                        abs(ds_transform.a - ref_transform.a) > 0.001
                        or abs(ds_transform.e - ref_transform.e) > 0.001
                    ):
                        return False

            logger.info(f"{label} All tiles aligned (CRS + geotransform OK)")
            return True
        except Exception as e:
            logger.warning(f"{label} Tile alignment check failed: {e}")
            return False

    def _align_tiles_to_reference_grid(self, tiles: list[xr.Dataset], label: str = "") -> list[xr.Dataset]:
        """Align all tiles to the first tile's grid to avoid merge seamline gaps."""
        if not tiles:
            return tiles

        try:
            from rasterio.enums import Resampling
        except Exception:
            logger.warning(f"{label} Could not import rasterio Resampling. Skipping grid alignment.")
            return tiles

        ref = self._ensure_dataset_spatial_ref(tiles[0])
        aligned: list[xr.Dataset] = [ref]

        for i, ds in enumerate(tiles[1:], start=1):
            try:
                ds = self._ensure_dataset_spatial_ref(ds)

                ref_crs = self._dataset_crs(ref)
                ds_crs = self._dataset_crs(ds)
                if not ref_crs or not ds_crs or ds_crs.to_string() != ref_crs.to_string():
                    logger.warning(
                        f"{label} tile {i} CRS mismatch during reference alignment. Keeping original tile."
                    )
                    aligned.append(ds)
                    continue

                aligned_elev = ds["elevation"].rio.reproject_match(
                    ref["elevation"], resampling=Resampling.bilinear
                )
                aligned_src = ds["source_id"].rio.reproject_match(
                    ref["source_id"], resampling=Resampling.nearest
                )
                aligned_src = aligned_src.fillna(0).astype("uint32")

                aligned_ds = xr.Dataset({"elevation": aligned_elev, "source_id": aligned_src})
                aligned_ds.rio.write_crs(ref_crs, inplace=True)
                aligned_ds.rio.write_transform(ref["elevation"].rio.transform(), inplace=True)
                aligned_ds = self._ensure_dataset_spatial_ref(aligned_ds)

                is_valid, msg = self._validate_geotransform(aligned_ds, label=f"{label}:tile{i}:aligned")
                if not is_valid:
                    logger.warning(f"{label} alignment validation failed for tile {i}: {msg}")
                    aligned.append(ds)
                else:
                    aligned.append(aligned_ds)
            except Exception as e:
                logger.warning(f"{label} failed to align tile {i} to reference grid: {e}")
                aligned.append(ds)

        return aligned

    def _align_tiles_to_target_grid(
        self,
        tiles: list[xr.Dataset],
        target_crs: str,
        bounds: tuple[float, float, float, float],
        resolution: float,
        label: str = "",
    ) -> list[xr.Dataset]:
        """Reproject all tiles onto an explicit shared grid (most robust)."""
        if not tiles:
            return tiles

        try:
            from rasterio.enums import Resampling
        except Exception:
            logger.warning(f"{label} Could not import rasterio Resampling. Skipping grid alignment.")
            return tiles

        west, south, east, north = bounds

        if target_crs.upper() == "EPSG:4326":
            mid_lat = (north + south) / 2.0
            meters_per_deg_lat = 110540.0
            meters_per_deg_lon = 111320.0 * max(math.cos(math.radians(mid_lat)), 1.0e-6)
            res_x = resolution / meters_per_deg_lon
            res_y = resolution / meters_per_deg_lat
        else:
            res_x = resolution
            res_y = resolution

        # Add a padding halo so neighboring reprojected tiles overlap and avoid seam gaps.
        # Measured seamline gaps in WLIS zoom-13 tiles are typically 4 pixels.
        pad_px = self._TARGET_GRID_PAD_PX
        west -= res_x * pad_px
        east += res_x * pad_px
        south -= res_y * pad_px
        north += res_y * pad_px

        width = max(1, math.ceil((east - west) / res_x))
        height = max(1, math.ceil((north - south) / res_y))

        transform = Affine.translation(west, north) * Affine.scale(res_x, -res_y)

        aligned: list[xr.Dataset] = []
        for i, ds in enumerate(tiles):
            try:
                ds = self._ensure_dataset_spatial_ref(ds)

                aligned_elev = ds["elevation"].rio.reproject(
                    target_crs,
                    transform=transform,
                    shape=(height, width),
                    resampling=Resampling.bilinear,
                )
                aligned_src = ds["source_id"].rio.reproject(
                    target_crs,
                    transform=transform,
                    shape=(height, width),
                    resampling=Resampling.nearest,
                )
                aligned_src = aligned_src.fillna(0).astype("uint32")

                aligned_ds = xr.Dataset({"elevation": aligned_elev, "source_id": aligned_src})
                aligned_ds.rio.write_crs(target_crs, inplace=True)
                aligned_ds.rio.write_transform(transform, inplace=True)
                aligned_ds = self._ensure_dataset_spatial_ref(aligned_ds)

                is_valid, msg = self._validate_geotransform(aligned_ds, label=f"{label}:tile{i}:target-grid")
                if not is_valid:
                    logger.warning(f"{label} target-grid validation failed for tile {i}: {msg}")
                    aligned.append(ds)
                else:
                    aligned.append(aligned_ds)
            except Exception as e:
                logger.warning(f"{label} failed to reproject tile {i} to target grid: {e}")
                aligned.append(ds)

        return aligned

    def _fill_small_seams(self, ds: xr.Dataset, label: str = "") -> xr.Dataset:
        """Fill small seam gaps inside BlueTopo coverage using nearest valid pixels."""
        if self._SEAM_FILL_MAX_PX <= 0:
            return ds

        if "elevation" not in ds or "source_id" not in ds:
            return ds

        try:
            import numpy as np
            from scipy.ndimage import distance_transform_edt
        except Exception as e:
            logger.warning(f"{label} seam fill unavailable: {e}")
            return ds

        elev = ds["elevation"].values
        src = ds["source_id"].values

        valid = np.isfinite(elev) & (src > 0)
        if not np.any(valid):
            return ds

        missing = ~valid
        if not np.any(missing):
            return ds

        distance, indices = distance_transform_edt(missing, return_indices=True)  # type: ignore[assignment]
        fill_mask = distance <= self._SEAM_FILL_MAX_PX
        if not np.any(fill_mask):
            return ds

        nearest_elev = elev[tuple(indices)]
        nearest_src = src[tuple(indices)]

        elev_filled = elev.copy()
        src_filled = src.copy()

        elev_filled[fill_mask] = nearest_elev[fill_mask]
        src_filled[fill_mask] = nearest_src[fill_mask]

        ds_out = xr.Dataset(
            {
                "elevation": (ds["elevation"].dims, elev_filled),
                "source_id": (ds["source_id"].dims, src_filled.astype("uint32")),
            },
            coords=ds.coords,
            attrs=ds.attrs,
        )
        ds_out = self._ensure_dataset_spatial_ref(ds_out)
        logger.info(f"{label} filled seam pixels: {int(fill_mask.sum())}")
        return ds_out

    def fetch_layer(
        self,
        bbox: tuple[float, float, float, float],
        resolution: float | None = None,
        crs: str = "EPSG:4326",
        **kwargs: Any,
    ) -> xr.DataArray | xr.Dataset:
        """
        Fetch BlueTopo layer for the given bounding box.
        Resolves, fetches, and merges all intersecting tiles.
        """
        from .base import matches_pattern

        # Ensure the bbox is in EPSG:4326 for tile resolution and metadata checks
        west, south, east, north = self._normalize_bbox(bbox, crs)

        filter_opts = kwargs.get("filter", {})
        include_patterns = filter_opts.get("include_patterns")
        exclude_patterns = filter_opts.get("exclude_patterns")
        max_interpolation_gap_m = filter_opts.get("max_interpolation_gap_m")

        # Keep request-CRS bounds for final clipping when the caller asks for
        # a non-4326 output (e.g., web mercator tile requests).
        req_west, req_south, req_east, req_north = west, south, east, north
        if crs and str(crs).upper() != "EPSG:4326":
            req_west, req_south, req_east, req_north = bbox

        # 1. Resolve Tiles
        tile_ids = self.resolve_tiles_in_bbox(west, south, east, north)
        if not tile_ids:
            raise ProviderNoDataError(f"No BlueTopo tiles found for bbox {bbox}")

        logger.info(f"BlueTopo fetch: Resolved {len(tile_ids)} tiles for bbox {bbox}")

        # 2. Fetch/Load Tiles
        das = []
        provenance_dict = {}
        # Tiles lost to errors (as opposed to no coverage). The runtime refuses to
        # cache a cell when this is non-zero, so the loss is retried.
        fetch_errors: list[str] = []
        for tid in tile_ids:
            # Pass the query bbox to maximize efficiency if underlying method supports it
            # defaulting to full tile load via existing method
            try:
                ds = self.load_tile_as_da(tid, bbox)
            except ProviderFetchError as e:
                logger.warning(f"BlueTopo tile {tid} failed: {e}")
                fetch_errors.append(f"{tid}: {e}")
                continue
            if ds is not None and "elevation" not in ds:
                logger.warning(
                    f"BlueTopo tile {tid} loaded without an elevation variable; treating as failed"
                )
                fetch_errors.append(f"{tid}: no elevation variable")
                continue
            if ds is not None:
                import hashlib

                import numpy as np

                da_elev = sanitize_elevation_nodata(ds["elevation"], abs_valid_limit=100000.0)

                if "source_id" in ds:
                    da_src = ds["source_id"]
                    unique_vals = np.unique(da_src.values)

                    translated_src = xr.zeros_like(da_src, dtype=np.uint32)
                    for pval in unique_vals:
                        if pval == 0 or np.isnan(pval):
                            continue

                        pval_int = int(pval)
                        survey_id = self._resolve_from_sidecar_rat(tid, pval_int)
                        # fallback
                        if not survey_id:
                            mid_lat = (south + north) / 2
                            mid_lon = (west + east) / 2
                            survey_id = self.get_source_survey_id(mid_lat, mid_lon)

                        if survey_id:
                            # Keep BlueTopo data available as a resilient fallback.
                            # Runtime fusion order resolves final precedence.
                            survey_id = survey_id.strip()

                        # Apply include/exclude pattern masking
                        if survey_id:
                            skip = False
                            if include_patterns and not matches_pattern(survey_id, include_patterns):
                                skip = True
                            if exclude_patterns and matches_pattern(survey_id, exclude_patterns):
                                skip = True

                            if skip:
                                logger.debug(f"Skipping filtered BlueTopo source: {survey_id}")
                                continue

                        name = f"BlueTopo: {survey_id}" if survey_id else f"BlueTopo: {tid}_{pval_int}"
                        digest = hashlib.md5(name.encode()).hexdigest()
                        project_uid = int(digest, 16) % 100000 + 50000

                        provenance_dict[project_uid] = {
                            "name": name,
                            "provider": "noaa_bluetopo",
                        }

                        translated_src = xr.where(da_src == pval, project_uid, translated_src)

                    p_source = translated_src.astype(np.uint32)
                else:
                    digest = hashlib.md5(tid.encode()).hexdigest()
                    project_uid = int(digest, 16) % 100000 + 50000
                    mid_lat = (south + north) / 2
                    mid_lon = (west + east) / 2
                    survey_id = self.get_source_survey_id(mid_lat, mid_lon)

                    if survey_id:
                        # Keep BlueTopo data available as a resilient fallback.
                        # Runtime fusion order resolves final precedence.
                        survey_id = survey_id.strip()

                    # Apply include/exclude pattern masking
                    if survey_id:
                        skip = False
                        if include_patterns and not matches_pattern(survey_id, include_patterns):
                            skip = True
                        if exclude_patterns and matches_pattern(survey_id, exclude_patterns):
                            skip = True

                        if skip:
                            logger.debug(f"Skipping filtered BlueTopo source: {survey_id}")
                            continue

                    name = f"BlueTopo: {survey_id}" if survey_id else f"BlueTopo: {tid}"

                    # Parse local resolution from survey_id (e.g. H13385_MB_50cm_MLLW -> 50cm)
                    res_str = "4m (Default)"
                    if survey_id:
                        import re

                        # Common patterns: _50cm_, _1m_, _2m_, _4m_
                        match = re.search(r"_(\d+(?:\.\d+)?(?:m|cm))_", survey_id, re.IGNORECASE)
                        if match:
                            res_str = match.group(1)

                    provenance_dict[project_uid] = {
                        "name": name,
                        "provider": "noaa_bluetopo",
                        "resolution": res_str,
                    }
                    p_source = xr.where(da_elev.notnull(), project_uid, 0).astype(np.uint32)

                p_source.name = "source_id"
                p_source.rio.write_nodata(0, inplace=True)
                p_source.attrs["_FillValue"] = 0

                # Mask elevation where source was filtered out (0)
                da_elev = da_elev.where(p_source != 0)

                p_ds = xr.Dataset({"elevation": da_elev, "source_id": p_source})
                if da_elev.rio.crs:
                    p_ds.rio.write_crs(da_elev.rio.crs, inplace=True)
                p_ds.rio.write_transform(da_elev.rio.transform(), inplace=True)
                p_ds = self._ensure_dataset_spatial_ref(p_ds)

                # Validate geotransform immediately after loading
                is_valid, msg = self._validate_geotransform(p_ds, label=f"Tile {tid}")
                if not is_valid:
                    logger.error(f"Geotransform validation failed: {msg}")
                else:
                    logger.info(msg)

                das.append(p_ds)

        if not das:
            if fetch_errors:
                raise ProviderFetchError(
                    f"BlueTopo fetch failed for bbox {bbox} with {len(fetch_errors)} error(s); "
                    f"first: {fetch_errors[0]}"
                )
            raise ProviderNoDataError(f"Failed to load any BlueTopo data for bbox {bbox}")
        if fetch_errors:
            logger.warning(
                f"BlueTopo returned partial data for bbox {bbox}: {len(fetch_errors)} error(s); "
                f"first: {fetch_errors[0]}"
            )

        # 3. Handle Mixed CRSs (e.g. crossing UTM zones)
        # If tiles are in different CRSs, merge_arrays will fail or produce garbage.
        # We must reproject to a common CRS (the requested one, or 4326) before merging.
        unique_crss = set()
        for ds in das:
            if ds.rio.crs:
                unique_crss.add(ds.rio.crs.to_string())

        # Validate pre-reproject alignment
        alignment_ok = self._check_tile_alignment(das, label="BlueTopo:pre-reproject")

        if len(unique_crss) > 1:
            logger.info(f"BlueTopo tiles have mixed CRSs {unique_crss}. Reprojecting to {crs} before merge.")
            reprojected_das = []
            for i, ds in enumerate(das):
                try:
                    # Log pre-reproject state
                    pre_transform = ds.rio.transform()
                    pre_crs = ds.rio.crs
                    logger.debug(f"BlueTopo tile {i}: pre-reproject CRS={pre_crs}, transform={pre_transform}")

                    # Reproject to target
                    reprojected_ds = ds.rio.reproject(crs)

                    # Validate post-reproject
                    is_valid, msg = self._validate_geotransform(
                        reprojected_ds, label=f"BlueTopo:tile{i}:post-reproject"
                    )
                    if not is_valid:
                        logger.error(f"Post-reproject validation failed: {msg}. Keeping original.")
                        reprojected_das.append(ds)
                    else:
                        # Log post-reproject state
                        post_transform = reprojected_ds.rio.transform()
                        logger.debug(
                            f"BlueTopo tile {i}: post-reproject CRS={reprojected_ds.rio.crs}, "
                            f"transform={post_transform}"
                        )
                        reprojected_das.append(reprojected_ds)

                except Exception as e:
                    logger.warning(f"Failed to reproject BlueTopo tile {i} during merge: {e}")
                    reprojected_das.append(ds)
            das = reprojected_das
            alignment_ok = self._check_tile_alignment(das, label="BlueTopo:post-reproject")

        # Mixed UTM tiles can reproject to slightly different transforms/resolutions
        # (e.g., 0.000040 vs 0.000041 deg), creating seamline nodata slivers that
        # later get filled by lower-priority providers. Normalize to one grid.
        if len(das) > 1:
            if resolution:
                grid_bounds = (west, south, east, north)
                if crs and str(crs).upper() != "EPSG:4326":
                    grid_bounds = (req_west, req_south, req_east, req_north)
                logger.info("BlueTopo: reprojecting tiles to explicit shared grid")
                das = self._align_tiles_to_target_grid(
                    das,
                    target_crs=crs,
                    bounds=grid_bounds,
                    resolution=resolution,
                    label="BlueTopo:target-grid",
                )
                alignment_ok = self._check_tile_alignment(das, label="BlueTopo:post-target-grid")
            elif not alignment_ok:
                logger.info("BlueTopo: normalizing tiles to a shared reference grid before merge")
                das = self._align_tiles_to_reference_grid(das, label="BlueTopo:align-ref")
                alignment_ok = self._check_tile_alignment(das, label="BlueTopo:post-align-ref")

        # 4. Merge
        if len(das) == 1:
            merged = self._ensure_dataset_spatial_ref(das[0])
        else:
            try:
                logger.info(
                    f"BlueTopo: Merging {len(das)} tiles. "
                    f"Alignment check: {'PASS' if alignment_ok else 'WARN'}"
                )

                # Merge arrays natively puts last element on top
                elevs = [ds["elevation"] for ds in das]
                sources = [ds["source_id"] for ds in das]

                final_elev = merge_arrays(elevs)
                final_elev = sanitize_elevation_nodata(final_elev, abs_valid_limit=100000.0)
                final_src = merge_arrays(sources)

                # Validate merged result
                merged_ds = xr.Dataset({"elevation": final_elev, "source_id": final_src})
                merged_ds = self._ensure_dataset_spatial_ref(merged_ds)
                is_valid, msg = self._validate_geotransform(merged_ds, label="BlueTopo:merged")
                if not is_valid:
                    logger.error(f"Post-merge validation failed: {msg}")
                else:
                    logger.info(msg)

                merged = merged_ds
            except Exception as e:
                logger.error(f"Failed to merge BlueTopo tiles: {e}")
                merged = das[0]  # Fallback
                # The fallback keeps only the first tile, so the result is partial.
                fetch_errors.append(f"merge {type(e).__name__}: {e}")

        # Fill small seam gaps before clipping/reprojection.
        merged = self._fill_small_seams(merged, label="BlueTopo:seam-fill")

        # OPTIMIZATION: Clip in Source CRS first (using 4326 bounds + 10% buffer)
        # We add a 10% buffer to the clip bounds. This prevents "rotated swatch" gaps
        # where the source-CRS clip (axis-aligned in source) becomes a rotated rectangle
        # in the target CRS that doesn't fully cover the corners of the target bbox.
        try:
            pad_x = (east - west) * 0.1
            pad_y = (north - south) * 0.1
            merged = self._ensure_dataset_spatial_ref(merged)

            # Check current CRS of merged dataset
            current_crs = self._dataset_crs(merged)
            clip_crs = "EPSG:4326"  # Define clip bounds in 4326

            # If merged is in a different CRS, we need to reproject bounds or clip in native CRS
            if current_crs and current_crs.to_string() != clip_crs:
                logger.debug(
                    f"BlueTopo: Merged dataset in {current_crs}, clip bounds in {clip_crs}. "
                    f"Will clip in native CRS."
                )
                # Clip uses the current CRS, so we must transform bbox to current CRS
                from pyproj import Transformer

                transformer = Transformer.from_crs(clip_crs, current_crs, always_xy=True)
                clip_west, clip_south = transformer.transform(west - pad_x, south - pad_y)
                clip_east, clip_north = transformer.transform(east + pad_x, north + pad_y)

                merged = merged.rio.clip_box(
                    minx=min(clip_west, clip_east),
                    miny=min(clip_south, clip_north),
                    maxx=max(clip_west, clip_east),
                    maxy=max(clip_south, clip_north),
                    crs=current_crs,
                    allow_one_dimensional_raster=True,
                )
            else:
                merged = merged.rio.clip_box(
                    minx=west - pad_x,
                    miny=south - pad_y,
                    maxx=east + pad_x,
                    maxy=north + pad_y,
                    crs=clip_crs,
                    allow_one_dimensional_raster=True,
                )
        except Exception as e:
            logger.warning(f"BlueTopo Clip failed (no overlap?): {e}")
            pass

        # Reproject to Requested CRS if needed
        merged_current_crs = self._dataset_crs(merged)
        if crs and merged_current_crs and merged_current_crs.to_string() != crs:
            try:
                logger.info(f"BlueTopo: Reprojecting merged result to {crs}")
                pre_final_transform = merged.rio.transform()
                logger.debug(f"BlueTopo: pre-final-reproject transform={pre_final_transform}")

                merged = merged.rio.reproject(crs)
                merged = self._ensure_dataset_spatial_ref(merged)

                is_valid, msg = self._validate_geotransform(merged, label="BlueTopo:final-reproject")
                if not is_valid:
                    logger.error(f"Final reprojection validation failed: {msg}")
                else:
                    logger.info(msg)

                post_final_transform = merged.rio.transform()
                logger.debug(f"BlueTopo: post-final-reproject transform={post_final_transform}")
            except Exception as e:
                logger.warning(f"Final reprojection failed: {e}")

        # Final Exact Clip to exact bbox (cleanup, in request CRS)
        try:
            merged = self._ensure_dataset_spatial_ref(merged)
            current_crs = self._dataset_crs(merged)
            final_clip_crs = crs if crs else "EPSG:4326"

            final_west, final_south, final_east, final_north = west, south, east, north
            if final_clip_crs and str(final_clip_crs).upper() != "EPSG:4326":
                final_west, final_south, final_east, final_north = (
                    req_west,
                    req_south,
                    req_east,
                    req_north,
                )

            if current_crs and current_crs.to_string() != str(final_clip_crs):
                # Need to transform bbox to current CRS for final clip
                from pyproj import Transformer

                transformer = Transformer.from_crs(final_clip_crs, current_crs, always_xy=True)
                c_west, c_south = transformer.transform(final_west, final_south)
                c_east, c_north = transformer.transform(final_east, final_north)

                merged_elev = merged["elevation"].rio.clip_box(
                    minx=min(c_west, c_east),
                    miny=min(c_south, c_north),
                    maxx=max(c_west, c_east),
                    maxy=max(c_south, c_north),
                    crs=current_crs,
                    allow_one_dimensional_raster=True,
                )
                merged_src = merged["source_id"].rio.clip_box(
                    minx=min(c_west, c_east),
                    miny=min(c_south, c_north),
                    maxx=max(c_west, c_east),
                    maxy=max(c_south, c_north),
                    crs=current_crs,
                    allow_one_dimensional_raster=True,
                )
            else:
                merged_elev = merged["elevation"].rio.clip_box(
                    minx=final_west,
                    miny=final_south,
                    maxx=final_east,
                    maxy=final_north,
                    crs=final_clip_crs,
                    allow_one_dimensional_raster=True,
                )
                merged_src = merged["source_id"].rio.clip_box(
                    minx=final_west,
                    miny=final_south,
                    maxx=final_east,
                    maxy=final_north,
                    crs=final_clip_crs,
                    allow_one_dimensional_raster=True,
                )

            merged = xr.Dataset({"elevation": merged_elev, "source_id": merged_src})
            merged = self._ensure_dataset_spatial_ref(merged)
        except Exception as e:
            logger.warning(f"BlueTopo final clip failed: {e}")

        if max_interpolation_gap_m and resolution:
            merged["elevation"] = interpolate_small_gaps(
                merged["elevation"], max_interpolation_gap_m, resolution
            )

        merged["elevation"].name = "elevation"
        merged.attrs["provenance_dict"] = provenance_dict
        merged.attrs["fetch_errors"] = len(fetch_errors)
        logger.debug("Found BlueTopo Coverage")

        from typing import cast

        return cast(xr.Dataset, merged)

    def get_metadata(self) -> dict[str, Any]:
        """
        Return metadata for the BlueTopo provider.

        Returns:
            Dictionary containing provider name, citation, resolution, and URL.
        """
        return {
            "name": "NOAA BlueTopo",
            "citation": "NOAA Office of Coast Survey (2025). BlueTopo™.",
            "resolution": "Variable (approx 4-8m)",
            "url": "https://nauticalcharts.noaa.gov/data/bluetopo.html",
        }

    def _resolve_scheme_url(self) -> str:
        """
        Resolves the latest BlueTopo Tile Scheme GPKG URL from S3.
        """
        try:
            fs = fsspec.filesystem("s3", anon=True)
            pattern = f"{self.BUCKET_BASE}/_BlueTopo_Tile_Scheme/*.gpkg"
            files = fs.glob(pattern)

            if not files:
                logger.warning("No BlueTopo scheme files found via S3 glob.")
                return self.TILE_SCHEME_URL

            # Sort by name (timestamps are in filename)
            latest = sorted(files)[-1]
            filename = Path(latest).name

            logger.info(f"Resolved latest BlueTopo Scheme: {filename}")

            # Construct HTTPS URL
            return (
                "https://noaa-ocs-nationalbathymetry-pds.s3.amazonaws.com/"
                f"BlueTopo/_BlueTopo_Tile_Scheme/{filename}"
            )
        except Exception as e:
            logger.warning(f"Failed to resolve dynamic scheme URL: {e}")
            return self.TILE_SCHEME_URL

    def _ensure_scheme_loaded(self) -> None:
        """
        Downloads and loads the Tile Scheme GPKG if not already loaded.
        """
        if self._gdf is not None:
            return

        from filelock import FileLock

        lock_path = self.scheme_path.with_suffix(".lock")

        with FileLock(lock_path):
            # Re-check inside lock: another thread may have loaded _gdf while we waited.
            if self._gdf is not None:
                return

            if not self.scheme_path.exists():
                try:
                    # Ensure directory exists (it might have been cleared runtime)
                    self.cache_dir.mkdir(parents=True, exist_ok=True)

                    # Dynamic Resolution
                    download_url = self._resolve_scheme_url()

                    logger.info(f"Downloading BlueTopo Tile Scheme to {self.scheme_path}...")
                    response = requests.get(download_url, stream=True)
                    response.raise_for_status()
                    with open(self.scheme_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                except Exception as e:
                    logger.warning(f"Failed to download BlueTopo Tile Scheme: {e}")
                    return

            try:
                self._gdf = gpd.read_file(self.scheme_path)
                if self._gdf is not None and hasattr(self._gdf, "sindex"):
                    _ = self._gdf.sindex
            except Exception as e:
                logger.error(f"Failed to load Tile Scheme GPKG: {e}")
                self._gdf = None

    def resolve_tile_id(self, lat: float, lon: float) -> str | None:
        """
        Queries the Tile Scheme to find the tile covering the coordinate.
        Returns the tile identifier (e.g. 'BlueTopo_Tile_X_Y_...') or None.
        """
        self._ensure_scheme_loaded()
        if self._gdf is None:
            return None

        point = Point(lon, lat)

        # Proj check (assuming GDF might not be 4326)
        query_point = point
        if self._gdf.crs and self._gdf.crs.to_string() != "EPSG:4326":
            pt_gdf = gpd.GeoSeries([point], crs="EPSG:4326")
            try:
                pt_gdf_proj = pt_gdf.to_crs(self._gdf.crs)
                query_point = pt_gdf_proj[0]
            except Exception:
                pass

        try:
            matches = self._gdf[self._gdf.contains(query_point)]
        except Exception:
            return None

        if matches.empty:
            return None

        row = matches.iloc[0]
        # 'tile' is the primary ID column in BlueTopo Scheme 2025+
        for col in ["tile", "file_name", "tile_name", "name", "tile_id", "standard_name"]:
            if col in row:
                return str(row[col])
        return None

    def resolve_tiles_in_bbox(self, west: float, south: float, east: float, north: float) -> list[str]:
        """
        Finds all tiles intersecting the bounding box.
        """
        self._ensure_scheme_loaded()
        if self._gdf is None:
            return []

        search_box = box(west, south, east, north)

        # CRS check
        query_geom = search_box
        if self._gdf.crs and self._gdf.crs.to_string() != "EPSG:4326":
            gdf_box = gpd.GeoSeries([search_box], crs="EPSG:4326")
            try:
                gdf_proj = gdf_box.to_crs(self._gdf.crs)
                query_geom = gdf_proj[0]
            except Exception:
                pass

        try:
            matches = self._gdf[self._gdf.intersects(query_geom)]
        except Exception:
            return []

        if matches.empty:
            return []

        results = []
        for _, row in matches.iterrows():
            for col in ["tile", "file_name", "tile_name", "name", "tile_id", "standard_name"]:
                if col in row:
                    results.append(str(row[col]))
                    break

        return list(set(results))

    def _get_tile_url_from_cache(self, tile_id: str) -> str | None:
        """Read tile_id→url from disk cache. No lock needed on read."""
        if not self.TILE_URL_CACHE_PATH.exists():
            return None
        try:
            with open(self.TILE_URL_CACHE_PATH) as f:
                data = json.load(f)
            result = data.get(tile_id)
            return str(result) if isinstance(result, str) else None
        except Exception:
            return None

    def _update_tile_url_cache(self, tile_id: str, url: str) -> None:
        """Write tile_id→url to disk cache atomically with FileLock."""
        try:
            self.TILE_URL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            lock_path = self.TILE_URL_CACHE_PATH.with_suffix(".json.lock")
            with FileLock(lock_path):
                data: dict[str, str] = {}
                if self.TILE_URL_CACHE_PATH.exists():
                    try:
                        with open(self.TILE_URL_CACHE_PATH) as f:
                            data = json.load(f)
                    except json.JSONDecodeError:
                        data = {}
                if data.get(tile_id) == url:
                    return
                data[tile_id] = url
                with open(self.TILE_URL_CACHE_PATH, "w") as f:
                    json.dump(data, f, indent=2, sort_keys=True)
                    f.write("\n")
        except Exception as e:
            logger.warning(f"Failed to update BlueTopo tile URL cache: {e}")

    def _invalidate_tile_url_cache(self, tile_id: str) -> None:
        """Removes a corrupt or stale URL from the cache."""
        try:
            import json

            from filelock import FileLock

            if not self.TILE_URL_CACHE_PATH.exists():
                return
            lock_path = self.TILE_URL_CACHE_PATH.with_suffix(".json.lock")
            with FileLock(str(lock_path)):
                with open(self.TILE_URL_CACHE_PATH) as f:
                    data = json.load(f)
                if tile_id in data:
                    del data[tile_id]
                    with open(self.TILE_URL_CACHE_PATH, "w") as f:
                        json.dump(data, f)
        except Exception:
            pass

    def _resolve_tile_url(self, tile_id: str) -> str | None:
        """
        Resolves the HTTPS URL for a given tile ID by checking S3.
        Results are persisted to disk so subsequent calls avoid S3 glob lookups.
        """
        # 1. Check disk cache (tile filenames include a date stamp, so S3 glob is needed
        #    on first access; thereafter the URL is stable until the tile scheme is refreshed)
        cached_url = self._get_tile_url_from_cache(tile_id)
        if cached_url:
            logger.debug(f"BlueTopo tile URL cache hit: {tile_id}")
            return cached_url

        from filelock import FileLock

        lock_path = self.cache_dir / f"url_resolve_{tile_id}.lock"

        with FileLock(str(lock_path)):
            # Double-check cache inside lock
            cached_url = self._get_tile_url_from_cache(tile_id)
            if cached_url:
                logger.debug(f"BlueTopo tile URL cache hit (post-lock): {tile_id}")
                return cached_url

            logger.debug(f"BlueTopo tile URL cache miss: {tile_id} — resolving via S3 glob")
            try:
                fs = fsspec.filesystem("s3", anon=True)
                search_pattern = f"{self.BUCKET_BASE}/{tile_id}/*.tiff"
                files = fs.glob(search_pattern)

                if not files:
                    logger.warning(f"BlueTopo: No files found in S3 for pattern: {search_pattern}")
                    return None

                source_path = files[0]
                # source_path example:
                # "noaa-ocs-nationalbathymetry-pds/BlueTopo/TileX/BlueTopo_TileX_2024.tiff"

                # Construct HTTPS URL
                # BUCKET_BASE = "noaa-ocs-nationalbathymetry-pds/BlueTopo"
                # We want: https://noaa-ocs-nationalbathymetry-pds.s3.amazonaws.com/BlueTopo/TileX/BlueTopo_TileX_2024.tiff

                # Split bucket from key
                parts = source_path.split("/", 1)
                if len(parts) < 2:
                    return None
                key = parts[1]

                url = f"https://noaa-ocs-nationalbathymetry-pds.s3.amazonaws.com/{key}"

                # 2. Persist to disk for future calls
                self._update_tile_url_cache(tile_id, url)
                logger.debug(f"BlueTopo tile URL cache created: {tile_id} -> {url}")
                return url

            except Exception as e:
                logger.error(f"BlueTopo URL Resolve Error: {e}")
                return None

    def fetch_elevation(self, lat: float, lon: float) -> float | None:
        """
        Fetches elevation from BlueTopo.
        """
        tile_id = self.resolve_tile_id(lat, lon)
        if not tile_id:
            return None

        try:
            url = self._resolve_tile_url(tile_id)
            if not url:
                return None

            # Stream with rioxarray
            # Use chunks=True (or dict) to enable dask streaming, but for single point we want
            # direct window read if possible.
            # rioxarray.open_rasterio loads lazily.
            # .sel() on a lazy array might trigger a download of the chunk.

            # Note: For single point access, standard requests/vsicurl is best.
            # rioxarray might read metadata footer first.

            with rioxarray.open_rasterio(url) as da_raw:  # type: ignore
                from typing import cast

                if isinstance(da_raw, list):
                    da = cast(xr.DataArray, da_raw[0])
                elif isinstance(da_raw, xr.Dataset):
                    da = cast(xr.DataArray, da_raw.to_array().isel(variable=0))
                else:
                    da = cast(xr.DataArray, da_raw)

                # Reproject point to raster CRS
                sample_x, sample_y = lon, lat
                if da.rio.crs and da.rio.crs != "EPSG:4326":
                    from pyproj import Transformer

                    transformer = Transformer.from_crs("EPSG:4326", da.rio.crs, always_xy=True)
                    sample_x, sample_y = transformer.transform(lon, lat)

                # Sample nearest
                val = da.sel(x=sample_x, y=sample_y, method="nearest")
                if "band" in val.dims or val.size > 1:
                    val = val.isel(band=0)

                val_item = val.values.item()

                if val_item == da.rio.nodata:
                    return None

                # VDatum: Convert Raw MLLW to NAVD88
                # Consistent with tile loading logic
                offset = self.vdatum.get_robust_mllw_to_navd88_offset(lat, lon)
                return float(val_item) + offset

        except Exception as e:
            logger.error(f"BlueTopo Fetch Error: {e}", exc_info=True)
            return None

    def is_covered(self, lat: float, lon: float) -> bool:
        """
        Check if the coordinate is covered by a BlueTopo tile.
        """
        return self.resolve_tile_id(lat, lon) is not None

    def get_quality_tier(self, lat: float, lon: float) -> QualityClass:
        """
        Return the quality tier for the given coordinate.
        Returns QualityClass.DIRECT if covered, else UNKNOWN.
        """
        if self.is_covered(lat, lon):
            return QualityClass.DIRECT
        return QualityClass.UNKNOWN

    def load_tile_as_da(self, tile_id: str, bbox: tuple[float, float, float, float]) -> "xr.Dataset | None":
        """
        Loads the cached tile and clips to bbox (west, south, east, north).
        Uses Streaming Access (VSICURL) to avoid full download.
        Applies caching to Zarr format for faster subsequent reads.
        Validates geotransform preservation through the caching pipeline.
        """
        # 1. Check Zarr Cache First (Fastest)
        # Append _navd88 to distinguish from raw caches
        zarr_name = f"{tile_id}_navd88.zarr"
        zarr_dir = self.cache_dir / "zarr"
        zarr_path = zarr_dir / zarr_name

        if zarr_path.exists():
            try:
                ds = xr.open_dataset(zarr_path, engine="zarr", chunks="auto", decode_coords="all")
                ds = self._restore_cache_crs(ds)
                ds = self._ensure_dataset_spatial_ref(ds)
                logger.debug(f"BlueTopo Zarr Cache Hit: {zarr_name}")

                # Validate cached data has proper spatial info
                is_valid_elev, _ = self._validate_geotransform(
                    ds["elevation"], label=f"BlueTopo:zarr_cache:{tile_id}:elevation"
                )
                # source_id can legitimately omit CRS attrs while still sharing
                # georeferenced x/y coordinates with elevation.
                is_valid_src = "source_id" not in ds or (
                    ds["source_id"].dims == ds["elevation"].dims
                    and "x" in ds["source_id"].coords
                    and "y" in ds["source_id"].coords
                )

                if not (is_valid_elev and is_valid_src):
                    logger.warning(
                        f"Cached BlueTopo tile {tile_id} failed geotransform validation. "
                        f"Elevation valid={is_valid_elev}, SourceID valid={is_valid_src}. "
                        f"Will invalidate and re-fetch."
                    )
                    import shutil

                    shutil.rmtree(zarr_path)
                    # Fall through to re-fetch
                else:
                    if "elevation" in ds:
                        return ds
                    return ds

            except Exception as e:
                logger.warning(f"Corrupt BlueTopo Zarr cache {zarr_path}: {e}")
                import shutil

                if zarr_path.exists():
                    shutil.rmtree(zarr_path)

        # 2. Resolve Remote URL (Cache Miss)
        http_url = self._resolve_tile_url(tile_id)
        if not http_url:
            return None

        logger.info(f"Streaming BlueTopo Asset: {http_url}")

        # 3. Stream & Cache to Zarr
        da_raw = None
        try:
            # Open Streaming
            try:
                da_raw = cast(xr.DataArray, rioxarray.open_rasterio(http_url, chunks={"x": 2048, "y": 2048}))
            except Exception as e:
                if "404" in str(e):
                    logger.warning(
                        f"BlueTopo 404 for {http_url}. Invalidating cache and retrying S3 resolution."
                    )
                    self._invalidate_tile_url_cache(tile_id)
                    http_url = self._resolve_tile_url(tile_id)
                    if not http_url:
                        raise e
                    da_raw = cast(
                        xr.DataArray, rioxarray.open_rasterio(http_url, chunks={"x": 2048, "y": 2048})
                    )
                else:
                    raise e

            # Log original source geotransform
            orig_crs = da_raw.rio.crs
            orig_transform = da_raw.rio.transform()
            logger.debug(f"BlueTopo tile {tile_id} raw: CRS={orig_crs}, transform={orig_transform}")

            # BlueTopo: Band 1=Elevation, Band 2=Uncertainty, Band 3=Contributor
            if "band" in da_raw.dims and da_raw.sizes["band"] >= 3:
                da_elev = da_raw.isel(band=0).drop_vars("band")
                da_elev = sanitize_elevation_nodata(
                    cast(xr.DataArray, da_elev),
                    extra_sentinels=(-999999.0,),
                    abs_valid_limit=100000.0,
                )
                da_elev.name = "elevation"
                da_src = da_raw.isel(band=2).drop_vars("band")
                da_src.name = "source_id"
                ds_to_cache = xr.Dataset({"elevation": da_elev, "source_id": da_src})
            else:
                if isinstance(da_raw, list):
                    da = cast(xr.DataArray, da_raw[0])
                elif isinstance(da_raw, xr.Dataset):
                    da = da_raw.to_array().isel(variable=0)
                else:
                    da = cast(xr.DataArray, da_raw)

                if "band" in da.dims:
                    da = da.isel(band=0).drop_vars("band")
                da = cast(xr.DataArray, da)
                da = sanitize_elevation_nodata(
                    da,
                    extra_sentinels=(-999999.0,),
                    abs_valid_limit=100000.0,
                )
                da.name = "elevation"
                ds_to_cache = xr.Dataset({"elevation": da})

            # CRITICAL: Ensure CRS and geotransform are properly set on dataset before caching
            # rioxarray's .to_zarr() may not preserve spatial metadata, so we add it explicitly
            if orig_crs:
                ds_to_cache.rio.write_crs(orig_crs, inplace=True)
            if orig_transform:
                ds_to_cache.rio.write_transform(orig_transform, inplace=True)
            if "source_id" in ds_to_cache:
                with contextlib.suppress(Exception):
                    ds_to_cache["source_id"].rio.write_crs(orig_crs, inplace=True)
                with contextlib.suppress(Exception):
                    ds_to_cache["source_id"].rio.write_transform(orig_transform, inplace=True)

            # --- VDatum: MLLW -> NAVD88 Conversion ---
            # BlueTopo (Chart Datum / MLLW) needs correction to NAVD88 for fusion consistency.
            try:
                # Use tile center to look up offset
                bounds = ds_to_cache.rio.bounds()  # (minx, miny, maxx, maxy)
                cx = (bounds[0] + bounds[2]) / 2.0
                cy = (bounds[1] + bounds[3]) / 2.0

                lat, lon = cy, cx
                # Check CRS against EPSG:4326 object or string.
                # rioxarray/rasterio CRS comparison can be tricky.
                # Simplest check is .to_epsg() or string comparison if it's not None.
                crs_obj = ds_to_cache.rio.crs
                is_4326 = False
                if crs_obj:
                    try:
                        if crs_obj.to_epsg() == 4326:
                            is_4326 = True
                    except Exception:
                        pass
                    if str(crs_obj).upper() == "EPSG:4326":
                        is_4326 = True

                if not is_4326 and crs_obj:
                    from pyproj import Transformer

                    transformer = Transformer.from_crs(crs_obj, "EPSG:4326", always_xy=True)
                    lon, lat = transformer.transform(cx, cy)

                # Retrieve offset: NAVD88 = MLLW + Offset
                # Note: get_mllw_to_navd88_offset is cached and static
                offset = self.vdatum.get_robust_mllw_to_navd88_offset(lat, lon)

                # Validate offset is reasonable (e.g. within -100 to 100m)
                if -100 < offset < 100:
                    logger.info(
                        f"Applying MLLW->NAVD88 offset of {offset:.3f}m for BlueTopo tile {tile_id} "
                        f"(at {lat:.4f}, {lon:.4f})"
                    )
                    ds_to_cache["elevation"] = ds_to_cache["elevation"] + offset
                    ds_to_cache.attrs["vdatum_offset"] = offset
                    ds_to_cache.attrs["vertical_datum"] = "NAVD88"
                else:
                    logger.warning(f"Suspicious VDatum offset {offset} for {tile_id}, dropping tile.")
                    return None

            except VDatumNoDataError as e:
                # No VDatum model near this tile: a property of the location, not an
                # outage, so dropping the tile is the stable outcome.
                logger.warning(
                    f"VDatum has no MLLW->NAVD88 coverage for BlueTopo tile {tile_id}: {e}. "
                    "Dropping tile to prevent artificial cliffs."
                )
                return None
            except Exception as e:
                # Timeouts or other API failures. Previously this returned an empty
                # Dataset, which fetch_layer then indexed for "elevation" and failed
                # the whole step with an unrelated KeyError. Raise the transient
                # signal instead so the cell is not cached and is retried.
                logger.warning(
                    f"VDatum conversion failed for BlueTopo tile {tile_id}: {e}. "
                    "Dropping dataset to prevent artificial cliffs."
                )
                raise ProviderFetchError(
                    f"VDatum MLLW->NAVD88 conversion failed for BlueTopo tile {tile_id}: "
                    f"{type(e).__name__}: {e}"
                ) from e

            ds_to_cache = self._stash_cache_crs(ds_to_cache)
            ds_to_cache = self._ensure_dataset_spatial_ref(ds_to_cache)

            # Validate before caching
            is_valid, msg = self._validate_geotransform(ds_to_cache, label=f"BlueTopo:{tile_id}:pre-zarr")
            if not is_valid:
                logger.error(f"Pre-Zarr validation failed: {msg}. Will not cache.")
                return ds_to_cache  # Return anyway, but don't cache

            # Cache to Zarr with explicit attributes
            from filelock import FileLock

            lock_path = zarr_path.with_suffix(".zarr.lock")

            with FileLock(lock_path):
                if zarr_path.exists():
                    cached_ds = xr.open_dataset(zarr_path, engine="zarr", chunks="auto", decode_coords="all")
                    cached_ds = self._restore_cache_crs(cached_ds)
                    cached_ds = self._ensure_dataset_spatial_ref(cached_ds)
                    is_valid, msg = self._validate_geotransform(
                        cached_ds, label=f"BlueTopo:{tile_id}:zarr_cache_exists"
                    )
                    if is_valid:
                        return cached_ds
                    else:
                        logger.warning(f"Existing Zarr cache invalid: {msg}. Overwriting.")
                        import shutil

                        shutil.rmtree(zarr_path)

                if ds_to_cache["elevation"].size > 0:
                    if "y" in ds_to_cache.dims and "x" in ds_to_cache.dims:
                        ds_to_cache = ds_to_cache.chunk({"y": 1024, "x": 1024})

                    logger.info(f"Caching BlueTopo tile to Zarr: {zarr_name}")
                    # Write with explicit encoding to preserve spatial metadata
                    ds_to_cache.to_zarr(
                        zarr_path,
                        mode="w",
                        consolidated=should_consolidate(),
                        encoding={"elevation": {}, "source_id": {}},
                    )

                    # Verify cache can be read back
                    cached_ds = xr.open_dataset(zarr_path, engine="zarr", chunks="auto", decode_coords="all")
                    cached_ds = self._restore_cache_crs(cached_ds)
                    cached_ds = self._ensure_dataset_spatial_ref(cached_ds)
                    is_valid, msg = self._validate_geotransform(
                        cached_ds, label=f"BlueTopo:{tile_id}:post-zarr-write"
                    )
                    if not is_valid:
                        logger.error(
                            "Post-Zarr validation failed: %s. Invalidating cache and using memory copy.",
                            msg,
                        )
                        with contextlib.suppress(Exception):
                            cached_ds.close()
                        import shutil

                        if zarr_path.exists():
                            shutil.rmtree(zarr_path)
                        return ds_to_cache

                    logger.info(msg)
                    return cached_ds

        except ProviderFetchError:
            raise
        except Exception as e:
            logger.error(f"Failed to stream/cache BlueTopo tile {tile_id}: {type(e).__name__}: {e}")
            if "404" in str(e):
                return None
            raise ProviderFetchError(
                f"Failed to stream/cache BlueTopo tile {tile_id}: {type(e).__name__}: {e}"
            ) from e
        finally:
            if da_raw is not None:
                with contextlib.suppress(Exception):
                    da_raw.close()

        return None

    def get_tile_id(self, lat: float, lon: float) -> str | None:
        """
        Returns the BlueTopo tile ID covering the given coordinate.
        """
        self._ensure_scheme_loaded()
        if self._gdf is None:
            return None

        from shapely.geometry import Point

        p = Point(lon, lat)
        # Assuming _gdf is in 4326
        matches = self._gdf[self._gdf.contains(p)]

        if matches.empty:
            return None

        row = matches.iloc[0]
        return str(row.get("tile_id", row.get("tile")))

    def get_source_survey_id(self, lat: float, lon: float) -> str | None:
        """
        Identifies the Source Survey ID (e.g., 'H13385') at the given coordinate.
        Strategy Cascade: Embedded RAT -> Sidecar RAT -> HSMDB API.
        """
        # 1. Tile Resolution
        tile_id = self.get_tile_id(lat, lon)
        if not tile_id:
            logger.debug("No BlueTopo tile found. Fallback to API.")
            return self._resolve_from_hsmdb_api(lat, lon)

        # 2. Local File Inspection (Embedded RAT + Pixel Value)
        url = self._resolve_tile_url(tile_id)
        pixel_val: int | None = None

        if url:
            try:
                # Use rioxarray/rasterio as osgeo.gdal is unreliable in this env
                with rioxarray.open_rasterio(url, masked=True) as da:  # type: ignore
                    # Reproject point to tile CRS
                    import numpy as np
                    from pyproj import Transformer

                    # da.rio.crs must be present
                    if da.rio.crs:
                        logger.debug(f"CRS: {da.rio.crs}")

                        # Handle potential Compound CRS (Horiz + Vert) which causes failure without geoids
                        from pyproj import CRS

                        target_crs = da.rio.crs
                        try:
                            crs_obj = CRS.from_user_input(da.rio.crs)
                            if crs_obj.is_compound:
                                logger.debug("Compound CRS detected, extracting horizontal component.")
                                # Assuming first sub-crs is horizontal (standard)
                                target_crs = crs_obj.sub_crs_list[0]
                                if target_crs.to_epsg():
                                    target_crs = f"EPSG:{target_crs.to_epsg()}"
                                    logger.debug(f"Using EPSG Code: {target_crs}")
                        except Exception as e:
                            logger.warning(f"CRS parsing warning: {e}")

                        transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)

                        xx, yy = transformer.transform(lon, lat)
                        if np.isinf(xx) or np.isinf(yy):
                            logger.warning("Transformation returned INF! Retrying with hardcoded EPSG:26918")
                            t2 = Transformer.from_crs("EPSG:4326", "EPSG:26918", always_xy=True)
                            xx, yy = t2.transform(lon, lat)

                        logger.debug(
                            f"Rio Pixel Check: {lat},{lon} -> {xx:.2f}, {yy:.2f} in bounds {da.rio.bounds()}"
                        )

                        # Select from Band 3 (Contributor)
                        # Use nearest neighbor lookup
                        try:
                            # band=3 (1-based index in rioxarray usually)
                            val = da.sel(x=xx, y=yy, method="nearest").sel(band=3).item()
                            # Check masking
                            if val is not None and not np.isnan(val):
                                pixel_val = int(val)
                                logger.debug(f"Resolved Pixel Value: {pixel_val}")
                        except Exception:
                            # Point might be out of bounds
                            pass
            except Exception as e:
                logger.warning(f"Error inspecting local tile (rioxarray) {tile_id}: {e}")

        # 3. Sidecar Fallback
        if pixel_val is not None:
            survey_id = self._resolve_from_sidecar_rat(tile_id, pixel_val)
            if survey_id:
                logger.debug(f"Resolved from Sidecar RAT: {survey_id}")
                return survey_id

        # 4. API Fallback
        logger.info(f"Fallback to HSMDB API for {lat}, {lon}")
        return self._resolve_from_hsmdb_api(lat, lon)

    def _geo_to_pixel(self, ds: Any, lat: float, lon: float) -> tuple[int | None, int | None]:
        """
        Helper to transform lat/lon to pixel coordinates.
        """
        try:
            from osgeo import gdal, osr

            gt = ds.GetGeoTransform()
            logger.debug(f"GeoTransform: {gt}")
            proj = ds.GetProjection()
            x_geo, y_geo = lon, lat

            if proj:
                src_srs = osr.SpatialReference()
                src_srs.ImportFromEPSG(4326)
                if hasattr(osr, "OAMS_TRADITIONAL_GIS_ORDER"):
                    src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
                dst_srs = osr.SpatialReference()
                dst_srs.ImportFromWkt(proj)
                transform = osr.CoordinateTransformation(src_srs, dst_srs)
                # Traditional: Lon, Lat
                point = transform.TransformPoint(x_geo, y_geo)
                x_geo, y_geo = point[0], point[1]

            inv_gt = gdal.InvGeoTransform(gt)
            if inv_gt:
                x_pix = int(inv_gt[0] + x_geo * inv_gt[1] + y_geo * inv_gt[2])
                y_pix = int(inv_gt[3] + x_geo * inv_gt[4] + y_geo * inv_gt[5])

                logger.debug(f"GeoToPixel: {lat},{lon} -> {x_geo},{y_geo} (proj) -> {x_pix},{y_pix} (pix)")

                if 0 <= x_pix < ds.RasterXSize and 0 <= y_pix < ds.RasterYSize:
                    return x_pix, y_pix
        except Exception as e:
            logger.warning(f"GeoToPixel Error: {e}")
            pass
        return None, None

    def _lookup_rat(self, rat: Any, pixel_val: int) -> str | None:
        """
        Helper to query a GDAL RAT.
        """
        for i in range(rat.GetColumnCount()):
            col_name = rat.GetNameOfCol(i)
            if (
                col_name.lower() in ["survey_id", "source_survey_id", "source_id"]
                and 0 <= pixel_val < rat.GetRowCount()
            ):
                val = rat.GetValueAsString(pixel_val, i)
                if val:
                    survey_id = str(val)
                    return survey_id
        return None

    def _resolve_from_sidecar_rat(self, tile_id: str, pixel_val: int) -> str | None:
        """Look up the survey ID for a pixel value via the tile's sidecar RAT file.

        Downloads the `.aux.xml` or `.dbf` RAT linked in the BlueTopo Tile Scheme GPKG and
        caches it locally under ``sidecars/``.  A ``.failed`` sentinel is written on HTTP
        errors (typically 404 when the tile scheme's RAT_Link is stale) so the download is
        not retried on subsequent requests for the same tile.
        """
        self._ensure_scheme_loaded()
        if self._gdf is None:
            return None

        try:
            # Use 'tile' column (common in geopackage)
            # Check columns first
            tile_col = "tile" if "tile" in self._gdf.columns else "Tile_Name"
            matches = self._gdf[self._gdf[tile_col] == tile_id]
            if matches.empty:
                logger.warning(f"Tile {tile_id} not found in scheme index.")
                return None

            tile_row = matches.iloc[0]
            rat_link = tile_row.get("RAT_Link") or tile_row.get("rat_link")
            if not rat_link:
                return None
            # Download Sidecar
            filename = Path(rat_link).name
            sidecar_dir = self.cache_dir / "sidecars"
            sidecar_dir.mkdir(parents=True, exist_ok=True)
            sidecar_file = sidecar_dir / filename
            failed_sentinel = sidecar_file.with_suffix(sidecar_file.suffix + ".failed")

            if failed_sentinel.exists():
                logger.debug(f"Skipping previously failed sidecar: {filename}")
                return None

            if not sidecar_file.exists():
                logger.info(f"Downloading Sidecar RAT: {rat_link}")
                r = requests.get(rat_link, timeout=30)
                if r.status_code == 200:
                    with open(sidecar_file, "wb") as f:
                        f.write(r.content)
                else:
                    logger.warning(
                        f"Failed to download sidecar {filename}: HTTP {r.status_code} — "
                        f"tile scheme RAT_Link may be stale. Will not retry."
                    )
                    failed_sentinel.touch()
                    return None

            # Parse based on extension
            if sidecar_file.suffix.lower() == ".xml":
                return self._parse_aux_xml_rat(sidecar_file, pixel_val)

            # Default .dbf parser (geopandas)
            df = gpd.read_file(sidecar_file)
            # Normalize columns
            df.columns = [c.lower() for c in df.columns]

            val_col = next((c for c in df.columns if c in ["value", "oid", "id"]), None)
            survey_col = next(
                (c for c in df.columns if c in ["survey_id", "source_survey_id", "source_id"]), None
            )

            if val_col and survey_col:
                match = df[df[val_col] == pixel_val]
                if not match.empty:
                    return str(match.iloc[0][survey_col])

        except Exception as e:
            logger.warning(f"Sidecar parsing failed: {e}")

    def _parse_aux_xml_rat(self, xml_path: Path, pixel_val: int) -> str | None:
        """
        Parses GDAL PAM XML to find Survey ID.
        """
        import xml.etree.ElementTree as Et

        try:
            tree = Et.parse(xml_path)
            root = tree.getroot()

            # Look for Band 3 RAT
            rat_node = None
            for band in root.findall("PAMRasterBand"):
                if band.get("band") == "3":
                    rat_node = band.find("GDALRasterAttributeTable")
                    break

            # Fallback
            if rat_node is None:
                rat_node = root.find("GDALRasterAttributeTable")

            if rat_node is None:
                return None

            # Map Columns
            field_map = {}
            for fd in rat_node.findall("FieldDefn"):
                idx_str = fd.get("index")
                if not idx_str:
                    continue
                idx = int(idx_str)
                name_tag = fd.find("Name")
                if name_tag is not None and name_tag.text:
                    field_map[idx] = name_tag.text.lower()

            val_idx = next((i for i, n in field_map.items() if n in ["value", "id"]), None)
            survey_idx = next(
                (i for i, n in field_map.items() if n in ["survey_id", "source_survey_id", "source_id"]), None
            )

            if val_idx is None or survey_idx is None:
                return None

            # Search Rows
            for row in rat_node.findall("Row"):
                fs = row.findall("F")
                if len(fs) <= max(val_idx, survey_idx):
                    continue

                v_txt = fs[val_idx].text
                if v_txt:
                    try:
                        v = int(float(str(v_txt)))
                        if v == pixel_val:
                            s_txt = fs[survey_idx].text
                            return str(s_txt) if s_txt else None
                    except (ValueError, TypeError):
                        pass
        except Exception as e:
            logger.warning(f"XML Parsing Error: {e}")

        return None

    def _resolve_from_hsmdb_api(self, lat: float, lon: float) -> str | None:
        """
        Tertiary fallback: Query NCEI HSMDB API.
        """
        try:
            from pyproj import Transformer

            # Project to EPSG:3857
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
            x, y = transformer.transform(lon, lat)

            url = "https://gis.ngdc.noaa.gov/arcgis/rest/services/web_mercator/nos_hydro_dynamic/MapServer/0/query"
            # Use strictly formatted JSON geometry with SR
            geo_json = f'{{"x":{x},"y":{y},"spatialReference":{{"wkid":3857}}}}'

            params = {
                "geometry": geo_json,
                "geometryType": "esriGeometryPoint",
                "spatialRel": "esriSpatialRelIntersects",
                "outFields": "SURVEY_ID",
                "returnGeometry": "false",
                "f": "json",
            }

            r = requests.get(url, params=params, timeout=10)
            if r.status_code == 200:
                data = r.json()
                features = data.get("features", [])
                if features:
                    # Return first match
                    val = features[0]["attributes"].get("SURVEY_ID")
                    return str(val) if val else None
        except Exception as e:
            logger.warning(f"HSMDB API Query failed: {e}")
        return None

    def get_all_survey_ids(self, tile_id: str) -> list[str]:
        """
        Extracts all unique survey IDs present within a given tile's sidecar RAT file.
        Downloads the sidecar if missing.
        """
        self._ensure_scheme_loaded()
        if self._gdf is None:
            return []

        try:
            tile_col = "tile" if "tile" in self._gdf.columns else "Tile_Name"
            matches = self._gdf[self._gdf[tile_col] == tile_id]
            if matches.empty:
                return []

            tile_row = matches.iloc[0]
            rat_link = tile_row.get("RAT_Link") or tile_row.get("rat_link")
            if not rat_link:
                return []

            filename = Path(rat_link).name
            sidecar_dir = self.cache_dir / "sidecars"
            sidecar_dir.mkdir(parents=True, exist_ok=True)
            sidecar_file = sidecar_dir / filename
            failed_sentinel = sidecar_file.with_suffix(sidecar_file.suffix + ".failed")

            if failed_sentinel.exists():
                return []

            if not sidecar_file.exists():
                r = requests.get(rat_link, timeout=30)
                if r.status_code == 200:
                    with open(sidecar_file, "wb") as f:
                        f.write(r.content)
                else:
                    failed_sentinel.touch()
                    return []

            if sidecar_file.suffix.lower() == ".xml":
                import xml.etree.ElementTree as Et

                tree = Et.parse(sidecar_file)
                root = tree.getroot()

                rat_node = None
                for band in root.findall("PAMRasterBand"):
                    if band.get("band") == "3":
                        rat_node = band.find("GDALRasterAttributeTable")
                        break
                if rat_node is None:
                    rat_node = root.find("GDALRasterAttributeTable")
                if rat_node is None:
                    return []

                field_map = {}
                for fd in rat_node.findall("FieldDefn"):
                    idx_str = fd.get("index")
                    name_tag = fd.find("Name")
                    if idx_str and name_tag is not None and name_tag.text:
                        field_map[int(idx_str)] = name_tag.text.lower()

                survey_idx = next(
                    (i for i, n in field_map.items() if n in ["survey_id", "source_survey_id", "source_id"]),
                    None,
                )
                if survey_idx is None:
                    return []

                surveys = set()
                for row in rat_node.findall("Row"):
                    fs = row.findall("F")
                    if len(fs) > survey_idx:
                        s_txt = fs[survey_idx].text
                        if s_txt:
                            surveys.add(str(s_txt).strip())
                return sorted(list(surveys))

            # For DBF parsing
            import geopandas as gpd

            df = gpd.read_file(sidecar_file)
            df.columns = [c.lower() for c in df.columns]
            survey_col = next(
                (c for c in df.columns if c in ["survey_id", "source_survey_id", "source_id"]), None
            )
            if survey_col:
                return sorted(list(df[survey_col].dropna().unique()))

        except Exception as e:
            logger.warning(f"Failed to get_all_survey_ids for {tile_id}: {e}")
        return []


# Register the provider
registry.register(Path(__file__).stem, NoaaBlueTopoProvider)
