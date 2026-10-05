"""
GEBCO 2025 Provider module.

This module implements the provider for the GEBCO 2025 Grid, a global terrain model for ocean and land.
It uses 'bmi-topography' or OPeNDAP access to fetch data, managing caching locally.
"""

import logging
import threading
from collections import namedtuple
from pathlib import Path
from typing import Any, ClassVar, cast

import xarray as xr
from rioxarray.merge import merge_arrays

from ..config import get_cache_root
from ..runtime import should_consolidate
from .base import Provider, sanitize_elevation_nodata
from .registry import registry

logger = logging.getLogger(__name__)

BBox = namedtuple("BBox", ["west", "south", "east", "north"])


class GEBCO2025Provider(Provider):
    """
    Interface for GEBCO 2025 data.
    Fetches from OPeNDAP with specific corrections.
    """

    # Official GEBCO 2025 OpenDAP URL
    # Note: 'sub_ice_topo_bathymetry' is the elevation variable
    OPENDAP_URL = "dap2://dap.ceda.ac.uk/thredds/dodsC/bodc/gebco/global/gebco_2025/sub_ice_topography_bathymetry/netcdf/gebco_2025_sub_ice.nc"
    TID_URL = "dap2://dap.ceda.ac.uk/thredds/dodsC/bodc/gebco/global/gebco_2025/type_identifier_grid/netcdf/gebco_2025_tid.nc"

    # Static instance to avoid re-initializing OPeNDAP metadata
    _singleton: ClassVar["GEBCO2025Provider | None"] = None

    RESOLUTION_ARCSEC = 15
    HALF_PIXEL_OFFSET = (15 / 3600) * 0.5  # Degrees

    _locks: ClassVar[dict[str, threading.Lock]] = {}
    _locks_lock: ClassVar[threading.Lock] = threading.Lock()  # eagerly initialized — never None
    _ds_remote: ClassVar[xr.Dataset | None] = None
    _tid_remote: ClassVar[xr.Dataset | None] = None
    _initialized: bool

    def __new__(cls, *args: Any, **kwargs: Any) -> "GEBCO2025Provider":
        if cls._singleton is None:
            cls._singleton = super().__new__(cls)
        return cast(GEBCO2025Provider, cls._singleton)

    def __init__(
        self,
        dem_type: str = "GEBCO_2025",
        south: float = -90,
        north: float = 90,
        west: float = -180,
        east: float = 180,
        output_format: str = "GTiff",
        cache_dir: str | Path | None = None,
    ):
        """
        Initialize the GEBCO provider.

        Args:
            dem_type: Type of DEM (default: GEBCO_2025).
            south, north, west, east: Global bounds.
            output_format: Output format for the grid.
            cache_dir: Directory to store cached data files.
        """
        if cache_dir is None:
            cache_dir = get_cache_root()

        if hasattr(self, "_initialized") and self._initialized:
            return

        p = Path(cache_dir).expanduser() / "gebco_2025"
        p.mkdir(parents=True, exist_ok=True)
        (p / "zarr").mkdir(exist_ok=True)  # Zarr subdir
        super().__init__()
        self.bbox = BBox(west, south, east, north)
        self.cache_dir = str(p)
        self.tid_data: xr.DataArray | None = None
        self._da: xr.DataArray | None = None
        self._initialized = True

    def fetch_layer(
        self,
        bbox: tuple[float, float, float, float],
        resolution: float | None = None,
        crs: str = "EPSG:4326",
        **kwargs: Any,
    ) -> xr.DataArray | xr.Dataset:
        """
        Implements Provider.fetch_layer
        """
        # Ensure the bbox is in EPSG:4326 for the fetch logic
        lon_min, lat_min, lon_max, lat_max = self._normalize_bbox(bbox, crs)
        bbox_override = BBox(lon_min, lat_min, lon_max, lat_max)
        return self.fetch(bbox_override=bbox_override)

    def get_metadata(self) -> dict[str, Any]:
        """
        Return metadata for the GEBCO provider.

        Returns:
            Dictionary containing provider name, citation, resolution, and URL.
        """
        return {
            "name": "GEBCO 2025",
            "citation": "GEBCO Compilation Group (2025) GEBCO 2025 Grid.",
            "resolution": "15 arc-second",
            "url": self.OPENDAP_URL,
        }

    def fetch(self, bbox_override: BBox | None = None) -> xr.Dataset:
        """
        Fetch data from GEBCO 2025 OPeNDAP server, utilizing a local Zarr cache
        tiled by 1x1 degree chunks to minimize repeat OPeNDAP hits.
        """
        # 1. Determine which 1x1 degree tiles we need
        import math
        import warnings

        current_bbox = bbox_override if bbox_override else self.bbox

        # Round bounds to nearest degree to find covering tiles
        s, w, n, e = current_bbox.south, current_bbox.west, current_bbox.north, current_bbox.east

        min_lat = math.floor(s)
        max_lat = math.ceil(n)  # Use ceil for north/east to ensure coverage
        min_lon = math.floor(w)
        max_lon = math.ceil(e)

        das_to_merge = []

        ds_remote = None  # Lazy init

        for lat_idx in range(min_lat, max_lat):
            for lon_idx in range(min_lon, max_lon):
                # Define 1x1 degree tile key and path
                # e.g. gebco_2025_n40_w74
                lat_str = f"n{lat_idx}" if lat_idx >= 0 else f"s{abs(lat_idx)}"
                lon_str = f"e{lon_idx}" if lon_idx >= 0 else f"w{abs(lon_idx)}"
                tile_key = f"gebco_2025_{lat_str}_{lon_str}"

                # Store in zarr subdirectory
                zarr_dir = Path(self.cache_dir) / "zarr"
                cache_path = zarr_dir / f"{tile_key}.zarr"
                da_tile: xr.DataArray | xr.Dataset | None = None
                provenance_dict = {}

                # A. Try loading from Zarr Cache
                if cache_path.exists():
                    try:
                        # Use chunks="auto" for Dask
                        # Suppress "grid_mapping not found" warning for existing caches derived from netCDF

                        with warnings.catch_warnings():
                            warnings.filterwarnings(
                                "ignore", message="Variable.s. referenced in grid_mapping not in variables"
                            )
                            # Try to open as Dataset first (new format with TID)
                            try:
                                da_tile = xr.open_dataset(
                                    cache_path, engine="zarr", chunks="auto", decode_coords="all"
                                )
                                if "elevation" not in da_tile:
                                    # Fallback to DataArray
                                    da_tile = xr.open_dataarray(
                                        cache_path, engine="zarr", chunks="auto", decode_coords="all"
                                    )
                            except Exception:
                                da_tile = xr.open_dataarray(
                                    cache_path, engine="zarr", chunks="auto", decode_coords="all"
                                )

                        # Ensure CRS
                        assert da_tile is not None
                        has_crs = False
                        if isinstance(da_tile, xr.Dataset):
                            has_crs = da_tile["elevation"].rio.crs is not None
                        else:
                            has_crs = da_tile.rio.crs is not None

                        if not has_crs:
                            logger.warning(f"GEBCO Zarr Cache {tile_key} missing CRS. Setting EPSG:4326")
                            da_tile = da_tile.rio.write_crs("EPSG:4326")

                        # Basic check
                        if da_tile is not None and (
                            (isinstance(da_tile, xr.Dataset) and da_tile["elevation"].size == 0)
                            or (not isinstance(da_tile, xr.Dataset) and da_tile.size == 0)
                        ):
                            da_tile = None
                        elif da_tile is not None:
                            logger.info(f"GEBCO Zarr Cache Hit: {tile_key}")
                    except Exception as err:
                        logger.warning(f"Corrupt GEBCO Zarr cache {cache_path}: {err}")
                        import shutil

                        if cache_path.exists():
                            shutil.rmtree(cache_path)
                        da_tile = None

                # B. Fetch from OPeNDAP if missing
                if da_tile is None:
                    from filelock import FileLock

                    with GEBCO2025Provider._locks_lock:
                        if tile_key not in GEBCO2025Provider._locks:
                            GEBCO2025Provider._locks[tile_key] = threading.Lock()
                        thread_lock = GEBCO2025Provider._locks[tile_key]

                    lock_path = cache_path.with_suffix(".zarr.lock")
                    with thread_lock, FileLock(lock_path):
                        # Double check
                        if cache_path.exists():
                            logger.info(f"GEBCO Zarr Cache Hit (after lock): {tile_key}")
                            with warnings.catch_warnings():
                                warnings.filterwarnings(
                                    "ignore",
                                    message="Variable.s. referenced in grid_mapping not in variables",
                                )
                                try:
                                    da_tile = xr.open_dataset(
                                        cache_path, engine="zarr", chunks="auto", decode_coords="all"
                                    )
                                    if "elevation" not in da_tile:
                                        # Fallback to DataArray
                                        da_tile = xr.open_dataarray(
                                            cache_path, engine="zarr", chunks="auto", decode_coords="all"
                                        )
                                except Exception:
                                    da_tile = xr.open_dataarray(
                                        cache_path, engine="zarr", chunks="auto", decode_coords="all"
                                    )
                        else:
                            logger.info(f"GEBCO Zarr Cache Miss: {tile_key}")
                            try:
                                if GEBCO2025Provider._ds_remote is None:
                                    # Protect OPeNDAP connection init with _locks_lock so concurrent
                                    # fetches of *different* tiles (different thread_lock) don't each
                                    # race to open a redundant pydap connection.
                                    with GEBCO2025Provider._locks_lock:
                                        if GEBCO2025Provider._ds_remote is None:
                                            try:
                                                logger.info("Initializing global GEBCO OPeNDAP Connection...")
                                                GEBCO2025Provider._ds_remote = xr.open_dataset(
                                                    self.OPENDAP_URL, engine="pydap"
                                                )
                                                GEBCO2025Provider._tid_remote = xr.open_dataset(
                                                    self.TID_URL, engine="pydap"
                                                )
                                            except Exception as connection_err:
                                                logger.error(
                                                    f"Failed to connect to GEBCO OPeNDAP: {connection_err}"
                                                )
                                                raise

                                ds_remote = GEBCO2025Provider._ds_remote
                                tid_remote = GEBCO2025Provider._tid_remote

                                logger.info(f"Downloading GEBCO 1x1 Tile + TID to Cache: {tile_key}")

                                # Slice exactly this 1x1 degree chunk
                                subset = ds_remote.sel(
                                    lat=slice(lat_idx, lat_idx + 1.0),
                                    lon=slice(lon_idx, lon_idx + 1.0),
                                )
                                tid_subset = None
                                if tid_remote is not None:
                                    tid_subset = tid_remote.sel(
                                        lat=slice(lat_idx, lat_idx + 1.0),
                                        lon=slice(lon_idx, lon_idx + 1.0),
                                    )

                                if "sub_ice_topo_bathymetry" in subset:
                                    da_source = subset["sub_ice_topo_bathymetry"]
                                elif "elevation" in subset:
                                    da_source = subset["elevation"]
                                else:
                                    logger.warning(f"Variable missing in GEBCO tile {tile_key}")
                                    continue

                                da_source = da_source.load()
                                da_source = sanitize_elevation_nodata(da_source, abs_valid_limit=100000.0)
                                if da_source.rio.crs is None:
                                    da_source.rio.write_crs("EPSG:4326", inplace=True)

                                if "grid_mapping" in da_source.attrs:
                                    del da_source.attrs["grid_mapping"]
                                if "grid_mapping" in da_source.encoding:
                                    del da_source.encoding["grid_mapping"]

                                # Process TID
                                da_tid = None
                                if tid_subset is not None and "tid" in tid_subset:
                                    da_tid = tid_subset["tid"].load()
                                    if "grid_mapping" in da_tid.attrs:
                                        del da_tid.attrs["grid_mapping"]
                                    if "grid_mapping" in da_tid.encoding:
                                        del da_tid.encoding["grid_mapping"]

                                if da_source.size > 0:
                                    da_source = da_source.chunk({"lat": 240, "lon": 240})
                                    da_source.name = "elevation"

                                    logger.info(f"GEBCO Zarr Cache Created: {tile_key}")

                                    if da_tid is not None:
                                        da_tid = da_tid.chunk({"lat": 240, "lon": 240})
                                        da_tid.name = "source_id"
                                        # Merge into a Dataset and write CRS before saving so it
                                        # survives the zarr round-trip without needing a patch on load.
                                        ds_to_save = xr.Dataset({"elevation": da_source, "source_id": da_tid})
                                        ds_to_save = ds_to_save.rio.write_crs("EPSG:4326")
                                        ds_to_save.to_zarr(
                                            cache_path, mode="w", consolidated=should_consolidate()
                                        )
                                    else:
                                        da_source.to_zarr(
                                            cache_path, mode="w", consolidated=should_consolidate()
                                        )

                                    with warnings.catch_warnings():
                                        warnings.filterwarnings(
                                            "ignore",
                                            message="Variable.s. referenced in grid_mapping not in variables",
                                        )
                                        # Reload exactly as saved
                                        if da_tid is None:
                                            da_tile = xr.open_dataarray(
                                                cache_path, engine="zarr", chunks="auto", decode_coords="all"
                                            )
                                        else:
                                            da_tile = xr.open_dataset(
                                                cache_path, engine="zarr", chunks="auto", decode_coords="all"
                                            )

                                        # Re-apply CRS if it was dropped during save
                                        assert da_tile is not None
                                        has_crs = False
                                        if isinstance(da_tile, xr.Dataset):
                                            has_crs = da_tile["elevation"].rio.crs is not None
                                        else:
                                            has_crs = da_tile.rio.crs is not None

                                        if not has_crs:
                                            da_tile = da_tile.rio.write_crs("EPSG:4326")

                                else:
                                    logger.warning(f"GEBCO Tile {tile_key} returned empty data.")
                            except Exception as tile_err:
                                logger.error(f"Failed to fetch GEBCO tile {tile_key}: {tile_err}")
                                continue

                    # Attempt to clean up lock file if cache was successfully created
                    try:
                        if cache_path.exists() and lock_path.exists():
                            lock_path.unlink()
                    except Exception:
                        pass

                if da_tile is not None:
                    # Extract unique TIDs for this chunk to build the provenance_dict
                    if isinstance(da_tile, xr.Dataset) and "source_id" in da_tile:
                        # Find unique values in the chunk
                        import numpy as np

                        unique_tids = np.unique(da_tile["source_id"].values)
                        for tid in unique_tids:
                            tid = int(tid)
                            if tid == 0 or np.isnan(tid):
                                continue
                            if tid not in provenance_dict:
                                provenance_dict[tid] = {
                                    "name": f"GEBCO 2025 (TID {tid})",
                                    "provider": "gebco_2025",
                                }

                    das_to_merge.append(da_tile)

        if not das_to_merge:
            raise KeyError(f"Failed to load any GEBCO data for bounds {current_bbox}")

        # 2. Merge tiles if multiple
        try:
            if len(das_to_merge) == 1:
                merged_ds = das_to_merge[0]
            else:
                normalized: list[xr.DataArray | xr.Dataset] = []
                for d in das_to_merge:
                    # GEBCO dimensions are lat/lon until renamed later.
                    # rioxarray.merge_arrays requires descending Y and ascending X.
                    if "lat" in d.dims:
                        d = d.sortby("lat", ascending=False)
                    if "lon" in d.dims:
                        d = d.sortby("lon", ascending=True)

                    if isinstance(d, xr.Dataset):
                        d["elevation"].name = "elevation"
                        if "source_id" in d:
                            d["source_id"].name = "source_id"
                            # Ensure _FillValue is appropriately set for merge_arrays
                            d["source_id"].rio.write_nodata(0, inplace=True)
                            d["source_id"].attrs["_FillValue"] = 0
                        normalized.append(d)
                    else:
                        d.name = "elevation"
                        normalized.append(d)

                # Separate elevation and source_id for merging
                elevs = [d["elevation"] if isinstance(d, xr.Dataset) else d for d in normalized]
                sources = [
                    d["source_id"] for d in normalized if isinstance(d, xr.Dataset) and "source_id" in d
                ]

                # rioxarray.merge_arrays requires exactly 'y' and 'x' mappings to safely handle
                # descending Y coordinates (which look "upside down" structurally if they are just
                # named lat/lon and missing CRS).
                for i in range(len(elevs)):
                    if "lat" in elevs[i].dims and "lon" in elevs[i].dims:
                        elevs[i] = elevs[i].rename({"lat": "y", "lon": "x"})
                        elevs[i] = elevs[i].rio.write_crs("EPSG:4326")
                        elevs[i] = elevs[i].rio.set_spatial_dims(x_dim="x", y_dim="y")

                for i in range(len(sources)):
                    if "lat" in sources[i].dims and "lon" in sources[i].dims:
                        sources[i] = sources[i].rename({"lat": "y", "lon": "x"})
                        sources[i] = sources[i].rio.write_crs("EPSG:4326")
                        sources[i] = sources[i].rio.set_spatial_dims(x_dim="x", y_dim="y")

                final_elev = merge_arrays(elevs)
                final_elev = final_elev.rename({"y": "lat", "x": "lon"})  # rename back for downstream slices
                final_elev = sanitize_elevation_nodata(final_elev, abs_valid_limit=100000.0)

                if sources:
                    final_src = merge_arrays(sources)
                    final_src = final_src.rename({"y": "lat", "x": "lon"})
                    merged_ds = xr.Dataset({"elevation": final_elev, "source_id": final_src})
                else:
                    merged_ds = xr.Dataset({"elevation": final_elev})

        except Exception as merge_err:
            logger.error(f"Merge failed: {merge_err}")
            merged_ds = das_to_merge[0]  # Fallback to first

        # 3. Slice to request bounds with padding
        # Pad bounds by 0.01 degrees to ensure data is returned for high zoom levels
        # where the requested extent is smaller than the native (15 arcsec) resolution.
        if merged_ds is not None:
            # Normalize lat to ascending order before slicing.
            # The multi-tile merge path sorts lat descending for rioxarray.merge_arrays,
            # so slice(south, north) with start < stop would return empty on a
            # descending coordinate.  Sorting ascending here is safe because the
            # rename to y/x and rio.write_crs happen immediately after.
            if "lat" in merged_ds.dims:
                import numpy as np

                lat_vals = merged_ds["lat"].values
                if len(lat_vals) > 1 and float(lat_vals[0]) > float(lat_vals[-1]):
                    merged_ds = merged_ds.sortby("lat", ascending=True)

            pad = 0.01
            merged_ds = merged_ds.sel(
                lat=slice(max(s - pad, -90), min(n + pad, 90)),
                lon=slice(max(w - pad, -180), min(e + pad, 180)),
            )

        if merged_ds is None or merged_ds["elevation"].size == 0:
            raise KeyError("Failed to fetch GEBCO data for GEBCO 2025")

        logger.debug("Found GEBCO 2025 Coverage")
        # Ensure consistent naming
        merged_ds["elevation"].name = "elevation"
        merged_ds["elevation"] = sanitize_elevation_nodata(merged_ds["elevation"], abs_valid_limit=100000.0)
        merged_ds.attrs["provenance_dict"] = provenance_dict

        # Rename lat/lon to y/x so rioxarray can identify spatial dimensions
        if "lat" in merged_ds.dims and "lon" in merged_ds.dims:
            merged_ds = merged_ds.rename({"lat": "y", "lon": "x"})

        # Ensure CRS is maintained
        merged_ds = merged_ds.rio.write_crs("EPSG:4326")

        logger.debug(f"GEBCO Final Output CRS: {merged_ds.rio.crs}")
        logger.debug(f"GEBCO Array CRS: {merged_ds['elevation'].rio.crs}")

        self._da = merged_ds["elevation"]  # Keep reference for sample_elevation
        return cast(xr.Dataset, merged_ds)

    def get_tid_classification(self) -> xr.DataArray | None:
        """Returns the cached TID DataArray"""
        return self.tid_data

    def sample_elevation(self, lat: float, lon: float) -> float:
        """
        Returns bilinearly interpolated elevation at the given coordinate.

        Args:
            lat: Latitude in decimal degrees.
            lon: Longitude in decimal degrees.

        Returns:
            Depth/Elevation in meters (negative for depth).
        """
        if self._da is None:
            raise RuntimeError("Data not loaded. Call fetch() or load() first.")

        # xarray's interp handles bilinear interpolation by default (linear)
        val = self._da.interp(y=lat, x=lon, method="linear")
        return float(val.values)


# Register the provider
registry.register(Path(__file__).stem, GEBCO2025Provider)
