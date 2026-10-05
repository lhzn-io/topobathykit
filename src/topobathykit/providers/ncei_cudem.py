"""
NOAA NCEI CUDEM Provider module.

This module implements the provider for the Continuously Updated Digital Elevation Model (CUDEM).
It handles tile resolution, downloading, caching, and processing of CUDEM data.
"""

import contextlib
import logging
import zipfile
from pathlib import Path
from typing import Any, ClassVar, cast

import geopandas as gpd
import numpy as np
import requests  # type: ignore
import rioxarray
import xarray as xr
from rioxarray.merge import merge_arrays
from shapely.geometry import box

from ..config import get_cache_root
from ..quality import QualityClass
from ..runtime import should_consolidate
from ..vdatum import VDatumResolver
from .base import Provider, ProviderNoDataError, sanitize_elevation_nodata
from .registry import registry

logger = logging.getLogger(__name__)


class CUDEMProvider(Provider):
    """
    Provider for NOAA/NCEI Continuously Updated Digital Elevation Model (CUDEM).
    Ninth-Arc-Second (~3m) Resolution.
    Acts as Tier 1 Data (Gap Filler for Intertidal/Coastal).
    """

    # Singleton instance to avoid re-initializing S3 filesystem and tile index
    _singleton: ClassVar["CUDEMProvider | None"] = None
    _initialized: bool

    # Bucket: noaa-nos-coastal-lidar-pds
    # Prefix: dem/NCEI_ninth_Topobathy_2014_8483/
    BASE_S3_URL = "https://noaa-nos-coastal-lidar-pds.s3.amazonaws.com/dem/NCEI_ninth_Topobathy_2014_8483"
    TILE_INDEX_ZIP = "tileindex_NCEI_ninth_Topobathy_2014.zip"

    def __new__(cls, *args: Any, **kwargs: Any) -> "CUDEMProvider":
        if cls._singleton is None:
            cls._singleton = super().__new__(cls)
        return cast(CUDEMProvider, cls._singleton)

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        """
        Initialize the CUDEM provider.

        Args:
            cache_dir: Directory to store cached data files.
        """
        if getattr(self, "_initialized", False):
            return

        if cache_dir is None:
            cache_dir = get_cache_root()

        self.cache_dir = Path(cache_dir).expanduser() / "ncei_cudem"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "zarr").mkdir(exist_ok=True)  # Create Zarr subdir

        self.index_path = self.cache_dir / self.TILE_INDEX_ZIP
        self.index_shp_dir = self.cache_dir / "index_shp"
        self._gdf: gpd.GeoDataFrame | None = None
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
        Fetch CUDEM data for the given bounding box.
        """
        # Ensure the bbox is in EPSG:4326 for spatial index lookup
        west, south, east, north = self._normalize_bbox(bbox, crs)
        matches = self.resolve_tiles(west, south, east, north)

        if matches.empty:
            raise ProviderNoDataError(f"No CUDEM tiles found for bbox {bbox}")

        logger.info(f"Found {len(matches)} CUDEM tiles for bbox.")

        project_layers = []
        provenance_dict: dict[int, dict[str, str]] = {}

        for _, row in matches.iterrows():
            url = self._get_tile_url(row)
            if url:
                tile_result = self.load_tile(row, (west, south, east, north))

                if isinstance(tile_result, xr.DataArray):
                    da = tile_result
                elif tile_result is not None:
                    try:
                        da = cast(xr.DataArray, tile_result)
                    except Exception:
                        continue
                else:
                    continue

                import hashlib

                filename = url.split("/")[-1].replace(".tif", "").replace(".tiff", "")
                project_uid = int(hashlib.md5(filename.encode()).hexdigest(), 16) % 100000 + 30000
                provenance_dict[project_uid] = {
                    "name": filename,
                    "provider": "ncei_cudem",
                }

                da.name = "elevation"
                da = sanitize_elevation_nodata(da, abs_valid_limit=100000.0)
                p_source = xr.where(da.notnull(), project_uid, 0).astype(np.uint32)
                p_source.name = "source_id"
                p_source.rio.write_nodata(0, inplace=True)
                p_source.attrs["_FillValue"] = 0

                p_ds = xr.Dataset({"elevation": da, "source_id": p_source})
                if da.rio.crs:
                    p_ds.rio.write_crs(da.rio.crs, inplace=True)
                p_ds.rio.write_transform(da.rio.transform(), inplace=True)

                project_layers.append(p_ds)

        if not project_layers:
            raise ProviderNoDataError(f"Failed to load valid CUDEM data for bbox {bbox}")

        if len(project_layers) == 1:
            merged_ds = project_layers[0]
        else:
            try:
                elevs = [ds["elevation"] for ds in project_layers]
                sources = [ds["source_id"] for ds in project_layers]

                final_elev = merge_arrays(elevs)
                final_elev = sanitize_elevation_nodata(final_elev, abs_valid_limit=100000.0)
                final_src = merge_arrays(sources)
                merged_ds = xr.Dataset({"elevation": final_elev, "source_id": final_src})
            except Exception as e:
                logger.error(f"Failed to merge CUDEM tiles: {e}")
                merged_ds = project_layers[0]

        # OPTIMIZATION: Clip in Source CRS first (using 4326 bounds) to avoid reprojecting full tile
        try:
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
        except Exception as e:
            logger.warning(f"Clip failed (no overlap?): {e}")
            if len(project_layers) == 1:  # If this was a single tile flow
                raise ProviderNoDataError(f"CUDEM tile clipped out: {bbox}") from e

        # Reproject if needed
        if crs and merged_ds["elevation"].rio.crs and merged_ds["elevation"].rio.crs != crs:
            try:
                final_elev = merged_ds["elevation"].rio.reproject(crs)
                final_src = merged_ds["source_id"].rio.reproject(crs)
                merged_ds = xr.Dataset({"elevation": final_elev, "source_id": final_src})
            except Exception as e:
                logger.warning(f"Reprojection failed: {e}")

        # Final Exact Clip (to target bbox in target CRS)
        # This ensures pixel alignment with the requested window
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

        merged_ds.attrs["provenance_dict"] = provenance_dict
        logger.debug(f"Found CUDEM Coverage with provenance {provenance_dict}")
        return cast(xr.Dataset, merged_ds)

    def get_metadata(self) -> dict[str, Any]:
        """
        Return metadata for the CUDEM provider.

        Returns:
            Dictionary containing provider name, citation, resolution, and URL.
        """
        return {
            "name": "NOAA CUDEM (Continuously Updated DEM)",
            "citation": "NOAA National Centers for Environmental Information.",
            "resolution": "~3m (1/9th arc-second)",
            "url": "https://coast.noaa.gov/digitalcoast/data/cudem.html",
        }

    def _ensure_index_loaded(self) -> None:
        """Download and load the spatial tile index."""
        if self._gdf is not None:
            return

        from filelock import FileLock

        lock_path = self.index_path.with_suffix(".lock")

        with FileLock(lock_path):
            # 1. Download Index Zip (Atomic)
            if not self.index_path.exists():
                url = f"{self.BASE_S3_URL}/{self.TILE_INDEX_ZIP}"
                logger.info(f"Downloading CUDEM Tile Index from {url}...")
                try:
                    r = requests.get(url, stream=True, timeout=60)
                    r.raise_for_status()
                    temp_zip = self.index_path.with_suffix(".tmp")
                    with open(temp_zip, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                    temp_zip.rename(self.index_path)
                except Exception as e:
                    logger.error(f"Failed to download CUDEM Index: {e}")
                    if self.index_path.exists():  # Cleanup if we somehow left a mess
                        self.index_path.unlink()
                    return

            # 2. Extract Shapefile (Atomic & Robust)
            # Check for valid content, not just directory existence
            has_shapes = False
            if self.index_shp_dir.exists():
                has_shapes = any(self.index_shp_dir.rglob("*.shp"))

            if not has_shapes:
                if self.index_shp_dir.exists():
                    import shutil

                    shutil.rmtree(self.index_shp_dir)

                try:
                    temp_extract_dir = self.index_shp_dir.with_name(self.index_shp_dir.name + "_tmp")
                    if temp_extract_dir.exists():
                        import shutil

                        shutil.rmtree(temp_extract_dir)

                    with zipfile.ZipFile(self.index_path, "r") as z:
                        z.extractall(temp_extract_dir)

                    temp_extract_dir.rename(self.index_shp_dir)
                except Exception as e:
                    logger.error(f"Failed to unzip CUDEM Index: {e}")
                    return

        # 3. Load GDF

        try:
            shps = list(self.index_shp_dir.glob("*.shp"))
            if not shps:
                shps = list(self.index_shp_dir.rglob("*.shp"))

            if not shps:
                logger.warning("No shapefile found in CUDEM index zip.")
                return

            self._gdf = gpd.read_file(shps[0])

            if self._gdf is None:
                logger.warning("CUDEM Index Shapefile read returned None.")
                return

            # Normalize CRS to 4326
            if self._gdf.crs is None:
                self._gdf.set_crs("EPSG:4326", inplace=True)
            elif self._gdf.crs.to_string() != "EPSG:4326":
                self._gdf = self._gdf.to_crs("EPSG:4326")

            # Normalize Columns to lowercase
            if self._gdf is not None:
                self._gdf.columns = [c.lower() for c in self._gdf.columns]

        except Exception as e:
            logger.error(f"Failed to load CUDEM Index GDF: {e}")

    def resolve_tiles(self, west: float, south: float, east: float, north: float) -> gpd.GeoDataFrame:
        """Find tiles intersecting the bbox."""
        self._ensure_index_loaded()
        if self._gdf is None:
            return gpd.GeoDataFrame()

        bbox_geom = box(west, south, east, north)
        matches = self._gdf[self._gdf.intersects(bbox_geom)]
        return matches

    def _get_tile_url(self, row: Any) -> str | None:
        """Determine URL from row metadata."""
        for col in ["url", "path", "location", "fileurl"]:
            if row.get(col):
                val = str(row[col])
                if val.startswith("http"):
                    return val
                return f"{self.BASE_S3_URL}/{val}"
        return None

    def _ensure_tile_cached(self, url: str) -> Path | None:
        """Download remote tile to cache."""
        filename = Path(url).name
        local_path = self.cache_dir / filename
        lock_path = self.cache_dir / f"{filename}.lock"

        from filelock import FileLock

        if local_path.exists():
            return local_path

        try:
            with FileLock(lock_path):
                # Double-check inside lock
                if local_path.exists():
                    return local_path

                logger.info(f"Downloading CUDEM Tile: {filename}")
                r = requests.get(url, stream=True, timeout=120)
                if r.status_code != 200:
                    logger.warning(f"CUDEM Tile 404/Error: {url}")
                    return None

                # Write to temp file first for atomicity
                temp_path = local_path.with_suffix(".tmp")

                # Ensure clean state
                if temp_path.exists():
                    temp_path.unlink()

                try:
                    with open(temp_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)

                    temp_path.rename(local_path)
                    logger.info(f"CUDEM download complete: {filename}")
                    return local_path
                except Exception as ex:
                    logger.error(f"Download write failed: {ex}")
                    if temp_path.exists():
                        temp_path.unlink()
                    return None

        except Exception as e:
            logger.error(f"Failed to acquire lock or download CUDEM tile: {e}")
            return None

    def load_tile(self, row: Any, bbox: tuple[float, float, float, float]) -> xr.DataArray | None:
        """Load single tile and harmonize datum."""
        url = self._get_tile_url(row)
        if not url:
            return None

        # Ensure raw file is present
        path = self._ensure_tile_cached(url)
        if not path:
            return None

        # --- Zarr Cache Layer ---
        # If we have already converted this tile to Zarr (optimized & VDatum adjusted), use it.
        zarr_dir = path.parent / "zarr"
        zarr_path = zarr_dir / path.with_suffix(".zarr").name  # e.g. tile.tif -> tile.zarr

        if zarr_path.exists():
            try:
                # Use chunks="auto" for Dask lazy loading
                da_cached = xr.open_dataarray(zarr_path, engine="zarr", chunks="auto", decode_coords="all")
                logger.info(f"CUDEM Zarr Cache Hit: {zarr_path.name}")
                return da_cached
            except Exception as e:
                logger.warning(f"Corrupt CUDEM Zarr cache {zarr_path}: {e}")
                import shutil

                if zarr_path.exists():
                    shutil.rmtree(zarr_path)
        # Load Raw
        try:
            da = rioxarray.open_rasterio(path, chunks={"x": 2048, "y": 2048})
            if isinstance(da, list):
                da = da[0]
            elif isinstance(da, xr.Dataset):
                da = da.to_array().isel(variable=0)

            # Explicit cast to a new variable to force mypy narrowing
            da_final: xr.DataArray = cast(xr.DataArray, da)

            if "band" in da_final.dims:
                da_final = da_final.isel(band=0).drop_vars("band")
            da_final = sanitize_elevation_nodata(da_final, abs_valid_limit=100000.0)

            # Robust CRS
            if da_final.rio.crs is None:
                da_final.rio.write_crs("EPSG:4269", inplace=True)

            # Cache to Zarr
            try:
                if da_final.size > 0:
                    da_final = da_final.chunk({"y": 1024, "x": 1024})
                    da_final.name = "elevation"

                    # Atomic Zarr Write
                    zarr_temp = zarr_path.with_suffix(".tmp.zarr")
                    if zarr_temp.exists():
                        import shutil

                        shutil.rmtree(zarr_temp)

                    da_final.to_zarr(zarr_temp, mode="w", consolidated=should_consolidate())
                    zarr_temp.rename(zarr_path)

                    return xr.open_dataarray(zarr_path, engine="zarr", chunks="auto", decode_coords="all")
            except Exception as e:
                logger.warning(f"Zarr cache write failed: {e}")
                if zarr_temp.exists():
                    import shutil

                    shutil.rmtree(zarr_temp)

            return da_final

        except Exception as e:
            logger.error(f"Error reading CUDEM tile {path}: {e}")
            return None

    def get_quality_tier(self, lat: float, lon: float) -> QualityClass:
        """
        Return quality tier for the given coordinate.
        """
        matches = self.resolve_tiles(lon, lat, lon, lat)
        if not matches.empty:
            return QualityClass.INDIRECT
        return QualityClass.UNKNOWN


# Register
registry.register(Path(__file__).stem, CUDEMProvider)
