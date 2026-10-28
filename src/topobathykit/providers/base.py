import fnmatch
import hashlib
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, cast

import numpy as np
import rioxarray  # noqa: F401
import xarray as xr
from pyproj import Transformer

logger = logging.getLogger(__name__)


def sanitize_elevation_nodata(
    da: xr.DataArray,
    *,
    extra_sentinels: tuple[float, ...] = (),
    min_valid: float | None = None,
    max_valid: float | None = None,
    abs_valid_limit: float | None = 100000.0,
) -> xr.DataArray:
    """Mask nodata sentinels and impossible elevation values as NaN.

    This helper handles common metadata-driven nodata values and optional
    provider-specific sentinels/ranges.
    """
    nodata_candidates: set[float] = set(extra_sentinels)
    for value in (
        da.rio.nodata,
        da.rio.encoded_nodata,
        da.attrs.get("_FillValue"),
        da.attrs.get("missing_value"),
    ):
        if value is None:
            continue
        try:
            nodata_candidates.add(float(value))
        except (TypeError, ValueError):
            continue

    cleaned = da
    cleaned = cleaned.where(np.isfinite(cleaned))
    for nodata in nodata_candidates:
        cleaned = cleaned.where(cleaned != nodata)

    if abs_valid_limit is not None:
        cleaned = cleaned.where(np.abs(cleaned) < abs_valid_limit)
    if min_valid is not None:
        cleaned = cleaned.where(cleaned > min_valid)
    if max_valid is not None:
        cleaned = cleaned.where(cleaned < max_valid)

    return cast(xr.DataArray, cleaned)


def interpolate_small_gaps(da: xr.DataArray, max_gap_m: float, resolution_m: float) -> xr.DataArray:
    """
    Interpolate localized NaNs (data holidays) up to the specified size constraint.

    Uses rasterio's fillnodata (GDAL) to efficiently fill small gaps
    without massive memory overhead, avoiding SciPy ArrayMemoryError on huge grids.
    """
    import scipy.ndimage as ndi
    from rasterio.fill import fillnodata

    vals = da.values.copy()
    valid = ~np.isnan(vals)

    # Ensure there's actually missing data to fill and valid data to interpolate from
    if np.all(valid) or not np.any(valid):
        return da

    # Find internal holes using binary_fill_holes
    filled_mask = ndi.binary_fill_holes(valid)
    holes_mask = filled_mask & ~valid

    if not np.any(holes_mask):
        return da

    # Distance threshold in pixels (gap width = 2 * distance_to_edge)
    max_px = (max_gap_m / 2.0) / resolution_m

    if max_px <= 0:
        return da

    # Use GDAL's highly optimized fillnodata to perform Inverse Distance Weighting
    # instead of scipy.ndimage.distance_transform_edt which causes 19+ GB RAM spikes.
    try:
        from dask.array.core import Array as DaskArray

        if isinstance(vals, DaskArray):
            # Compute to numpy array before passing to rasterio
            vals = vals.compute()
            valid = ~np.isnan(vals)
            filled_mask = ndi.binary_fill_holes(valid)
            holes_mask = filled_mask & ~valid
    except ImportError:
        pass

    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", module="rasterio")
        filled_vals = fillnodata(vals, mask=valid, max_search_distance=max_px, smoothing_iterations=0)

    # We only want to fill pixels that are within internal holes
    # and were previously NaN. Revert anything outside the holes_mask.
    out_of_bounds = (~holes_mask) & (~valid)
    filled_vals[out_of_bounds] = np.nan

    return cast(xr.DataArray, da.copy(data=filled_vals))


def despike_median_deviation(
    da: xr.DataArray, threshold: float, min_valid: int = 5
) -> tuple[xr.DataArray, int]:
    """Mask pixels that depart from their 3x3 neighbourhood median by more than `threshold`.

    Returns the filtered array and the number of pixels masked to NaN.

    The median ignores NaN, so a pixel beside a coverage edge is judged against the data
    that is actually there. A window with fewer than `min_valid` finite values (the
    centre included) is left alone rather than judged on too few neighbours.

    Scale matters. The filter sees features about one pixel wide at the resolution it is
    run at, so it belongs after alignment to the output canvas. The Mount Sinai Harbor
    defect in New_England_Coned_Topobathy_DEM_2016_6194 is a smooth bowl about 40 m across
    reaching -290 m in the source COG: invisible to a 3x3 window at the native 1 m, and a
    one or two pixel spike at 30 m.
    """
    if threshold <= 0:
        return da, 0
    vals = np.asarray(da.values, dtype=np.float64)
    if vals.ndim != 2 or vals.size == 0:
        return da, 0

    import warnings

    from numpy.lib.stride_tricks import sliding_window_view

    padded = np.pad(vals, 1, mode="constant", constant_values=np.nan)
    windows = sliding_window_view(padded, (3, 3))
    with warnings.catch_warnings():
        # All-NaN windows (outside coverage) are expected and handled by the count below.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.nanmedian(windows, axis=(-2, -1))
    count = np.isfinite(windows).sum(axis=(-2, -1))

    spikes = np.isfinite(vals) & (count >= min_valid) & (np.abs(vals - med) > threshold)
    n = int(spikes.sum())
    if not n:
        return da, 0
    cleaned = np.where(spikes, np.nan, da.values)
    return cast(xr.DataArray, da.copy(data=cleaned)), n


# --- Grid alignment -----------------------------------------------------------------

# Metres per degree used to convert between ground and geographic resolution. The
# same constants size the BlueTopo target grid and the runtime canvas.
METERS_PER_DEG_LON_EQUATOR = 111320.0
METERS_PER_DEG_LAT = 110540.0

# A source at least this many times finer than the target cell is block-averaged
# onto it; anything coarser or similar is interpolated bilinearly. Below 1.5x a
# target cell spans at most two source pixels per axis, which bilinear already
# weighs; above it, bilinear would sample a few source pixels and alias the rest.
AVERAGE_RESAMPLING_RATIO = 1.5

# Fraction of a target cell that valid source pixels must cover, under average
# resampling, for the cell to be valid. 0.5 matches the footprint that nearest
# resampling gives, so coverage edges neither grow nor shrink by more than half a cell.
MIN_VALID_FRACTION = 0.5


def _is_mercator(crs_obj: Any) -> bool:
    """True when a projected CRS is a Mercator variant (scale grows as 1 / cos(lat))."""
    try:
        if crs_obj.to_epsg() in (3857, 3395, 900913):
            return True
        op = crs_obj.coordinate_operation
        return op is not None and "mercator" in op.method_name.lower()
    except Exception:
        return False


def ground_resolution_m(transform: Any, crs: Any, lat: float) -> float:
    """Approximate ground size in metres of one pixel of a grid, at latitude `lat`.

    Returns the geometric mean of the x and y pixel sizes. Geographic grids are
    converted with the METERS_PER_DEG constants; Mercator grids are scaled by
    cos(lat), since their nominal metres exceed ground metres away from the equator.
    """
    import math

    from pyproj import CRS

    crs_obj = CRS.from_user_input(crs)
    if crs_obj.is_compound and crs_obj.sub_crs_list:
        crs_obj = crs_obj.sub_crs_list[0]
    dx = abs(float(transform.a))
    dy = abs(float(transform.e))
    coslat = max(math.cos(math.radians(lat)), 1.0e-6)
    if crs_obj.is_geographic:
        gx = dx * METERS_PER_DEG_LON_EQUATOR * coslat
        gy = dy * METERS_PER_DEG_LAT
    else:
        try:
            unit = float(crs_obj.axis_info[0].unit_conversion_factor)
        except (IndexError, AttributeError, TypeError):
            unit = 1.0
        gx = dx * unit
        gy = dy * unit
        if _is_mercator(crs_obj):
            gx *= coslat
            gy *= coslat
    return math.sqrt(gx * gy)


def choose_resampling(source_m: float, target_m: float) -> Any:
    """Pick the resampling for elevation going from `source_m` to `target_m` pixels.

    Average when the source is at least AVERAGE_RESAMPLING_RATIO times finer than the
    target, bilinear otherwise. Integer ids (source_id) always use nearest instead.
    """
    from rasterio.enums import Resampling

    if source_m > 0 and target_m / source_m >= AVERAGE_RESAMPLING_RATIO:
        return Resampling.average
    return Resampling.bilinear


def grid_for_bounds(
    bounds: tuple[float, float, float, float],
    crs: str,
    resolution_m: float,
    pad_px: int = 0,
) -> tuple[Any, tuple[int, int]]:
    """Return (transform, (height, width)) of a north-up grid covering `bounds`.

    `bounds` (west, south, east, north) and the grid are in `crs`. `resolution_m` is
    a ground resolution in metres; for a geographic CRS it is converted to degrees
    at the mid latitude of the bounds. `pad_px` adds that many cells on every side.
    """
    import math

    from affine import Affine
    from pyproj import CRS

    west, south, east, north = bounds
    if CRS.from_user_input(crs).is_geographic:
        mid_lat = (north + south) / 2.0
        res_x = resolution_m / (METERS_PER_DEG_LON_EQUATOR * max(math.cos(math.radians(mid_lat)), 1.0e-6))
        res_y = resolution_m / METERS_PER_DEG_LAT
    else:
        res_x = resolution_m
        res_y = resolution_m

    west -= res_x * pad_px
    east += res_x * pad_px
    south -= res_y * pad_px
    north += res_y * pad_px

    width = max(1, math.ceil((east - west) / res_x))
    height = max(1, math.ceil((north - south) / res_y))
    transform = Affine.translation(west, north) @ Affine.scale(res_x, -res_y)
    return transform, (height, width)


def _strip_nodata(da: xr.DataArray) -> xr.DataArray:
    """Copy of `da` with every nodata marker removed, so GDAL treats all values as data."""
    out = da.copy()
    out.attrs.pop("_FillValue", None)
    out.attrs.pop("missing_value", None)
    out.encoding.pop("_FillValue", None)
    out.encoding.pop("missing_value", None)
    return cast(xr.DataArray, out.rio.write_nodata(None, encoded=False))


def align_to_grid(
    da: xr.DataArray,
    crs: Any,
    transform: Any,
    shape: tuple[int, int],
    resampling: Any = None,
) -> xr.DataArray:
    """Resample `da` onto the grid (crs, transform, shape) without no-data bleeding.

    Nearest resampling is a plain reproject, used for integer ids. For elevation
    (bilinear, cubic, average, ...) NaN is declared as the source nodata, so GDAL
    weighs only valid pixels, and the result is then masked by the source validity
    carried onto the same grid:

    - interpolating kernels keep a cell only where nearest-resampled validity is set,
      so the valid footprint is exactly the one nearest resampling would give;
    - average keeps a cell only where valid source pixels cover at least
      MIN_VALID_FRACTION of it.

    Either way coverage edges do not grow into no-data, and no-data never pulls
    values towards a fill value.
    """
    from rasterio.enums import Resampling

    if resampling is None:
        resampling = Resampling.bilinear
    height, width = shape
    grid_kw: dict[str, Any] = {"transform": transform, "shape": (height, width)}

    if resampling == Resampling.nearest:
        return cast(xr.DataArray, da.rio.reproject(crs, resampling=Resampling.nearest, **grid_kw))

    src = da if np.issubdtype(da.dtype, np.floating) else da.astype(np.float32)
    valid = src.notnull()
    nodata = da.rio.nodata
    if nodata is not None and not np.isnan(nodata):
        valid = valid & (src != nodata)
    src = _strip_nodata(src.where(valid)).rio.write_nodata(np.nan, encoded=False)

    out = src.rio.reproject(crs, resampling=resampling, nodata=np.nan, **grid_kw)

    mask_src = _strip_nodata(src.copy(data=valid.values.astype(np.float32)))
    if resampling == Resampling.average:
        frac = mask_src.rio.reproject(crs, resampling=Resampling.average, nodata=0.0, **grid_kw)
        keep = frac.fillna(0.0) >= MIN_VALID_FRACTION
    else:
        nn = mask_src.rio.reproject(crs, resampling=Resampling.nearest, nodata=0.0, **grid_kw)
        keep = nn.fillna(0.0) > 0.5

    return cast(xr.DataArray, out.copy(data=np.where(keep.values, out.values, np.nan).astype(out.dtype)))


def merge_by_priority(layers: list[xr.Dataset], var: str = "elevation") -> xr.Dataset:
    """Merge datasets that share one grid; the first layer with a valid `var` wins each pixel.

    Every other variable follows the winner of `var`, so a source id always
    describes the elevation beside it. The result does not depend on anything but
    the order of `layers`, unlike a merge over differing grids.
    """
    if not layers:
        raise ValueError("merge_by_priority needs at least one layer")
    first = layers[0]
    if len(layers) == 1:
        return first.copy(deep=True)
    # Plain numpy copies: assigning into .values of a dask-backed array is lost.
    arrays = {str(name): np.array(first[name].values, copy=True) for name in first.data_vars}
    shape = arrays[var].shape
    for layer in layers[1:]:
        layer_var = np.asarray(layer[var].values)
        if layer_var.shape != shape:
            raise ValueError(f"merge_by_priority: layer shape {layer_var.shape} != {shape}")
        fill = np.isnan(arrays[var]) & np.isfinite(layer_var)
        if not fill.any():
            continue
        for name, arr in arrays.items():
            if name in layer.data_vars:
                arr[fill] = np.asarray(layer[name].values)[fill]
    return first.copy(data=arrays)


# Per-pixel source ids. Each provider owns a disjoint block of SOURCE_ID_BLOCK
# ids, so an id identifies its provider even before provenance_dict is
# consulted. Ids below SOURCE_ID_BLOCK are reserved for policy legend ids and
# GEBCO type ids. The largest id stays below 2**24, so it survives a round
# trip through float32.
SOURCE_ID_BLOCK = 1_000_000
SOURCE_ID_BLOCKS: dict[str, int] = {
    "noaa_topobathy": 1,
    "ncei_cudem": 3,
    "ncei_bag": 4,
    "noaa_bluetopo": 5,
    "usgs_3dep": 6,
    "usgs_lidar": 7,
    "copernicus_dsm": 8,
    "nasadem": 9,
}


def source_uid(provider: str, key: str | int) -> int:
    """Return the per-pixel source id for one survey, project or asset of a provider.

    An integer ``key`` below SOURCE_ID_BLOCK (e.g. a NOAA project id) is used
    as the offset directly, so the id stays readable; any other key is hashed
    into the block.
    """
    base = SOURCE_ID_BLOCKS[provider] * SOURCE_ID_BLOCK
    if isinstance(key, int) and 0 <= key < SOURCE_ID_BLOCK:
        return base + key
    digest = hashlib.md5(str(key).encode()).hexdigest()
    return base + int(digest, 16) % SOURCE_ID_BLOCK


def matches_pattern(name: str, patterns: list[str]) -> bool:
    """Return True if name matches any pattern in the list.

    Supports fnmatch globs (e.g. ``H*``, ``W*``) and inline regex for cases
    that need case-insensitive or complex matching (prefix with ``(?i)`` or
    ``^``).  Both are stdlib — no extra dependency needed.
    """
    for p in patterns:
        if p.startswith("(?i)") or p.startswith("^"):
            if re.match(p, name):
                return True
        elif fnmatch.fnmatch(name, p):
            return True
    return False


class ProviderNoDataError(LookupError):
    """Raised when a provider has no data for the requested bounding box.

    This is a normal operating condition — it means the provider simply does not
    cover the requested area.  The runtime handles it by skipping the provider and
    continuing with the next step in the fusion policy.  It is intentionally *not*
    logged as an error.
    """


class ProviderFetchError(RuntimeError):
    """Raised when a provider covers the bbox but failed to deliver its data.

    Distinct from `ProviderNoDataError`: the failure came from an error while
    fetching (network timeout, VDatum outage, MemoryError, merge failure), not
    from a lack of coverage, so a retry may succeed. The runtime records the
    step as failed and the cell is not cached.

    Providers that return data despite dropping some tiles or projects to
    errors signal the partial result instead through an integer
    ``attrs["fetch_errors"]`` on the returned Dataset.
    """


class Provider(ABC):
    """
    Abstract Base Class for topobathykit data providers.

    All data sources (GEBCO, BlueTopo, BAG, etc.) must implement this interface
    to ensure a unified data access layer for the fusion runtime.
    """

    def _normalize_bbox(
        self,
        bbox: tuple[float, float, float, float],
        crs: str = "EPSG:4326",
    ) -> tuple[float, float, float, float]:
        """
        Ensures the bounding box is in WGS84 (EPSG:4326) degrees.
        If input is already in 4326, returns it unchanged.
        If in any other CRS (e.g. 3857 meters), transforms it.

        Args:
            bbox: (west, south, east, north)
            crs: The CRS of the input bbox.

        Returns:
            Tuple[float, float, float, float]: (west, south, east, north) in WGS84.
        """
        if crs.upper() == "EPSG:4326":
            return bbox

        try:
            west, south, east, north = bbox
            # We use always_xy=True to ensure (lon, lat) ordering
            transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            w, s = transformer.transform(west, south)
            e, n = transformer.transform(east, north)
            logger.debug(f"Normalized bbox from {crs} to EPSG:4326: {bbox} -> ({w}, {s}, {e}, {n})")
            return (w, s, e, n)
        except Exception as exc:
            logger.warning(f"Failed to normalize bbox from {crs} to EPSG:4326: {exc}")
            return bbox

    @abstractmethod
    def fetch_layer(
        self,
        bbox: tuple[float, float, float, float],
        resolution: float | None = None,
        crs: str = "EPSG:4326",
        **kwargs: Any,
    ) -> xr.DataArray | xr.Dataset:
        """
        Fetch a data layer for the given bounding box.

        Args:
            bbox: Tuple of (min_lon, min_lat, max_lon, max_lat).
            resolution: Desired resolution in meters (approximate).
            crs: Coordinate Reference System string (default: EPSG:4326).
            **kwargs: Additional provider-specific arguments (e.g., filters).

        Returns:
            xr.DataArray or xr.Dataset:
                - Legacy: Returns an `xr.DataArray` of elevation.
                - Detailed Provenance: Returns an `xr.Dataset` containing 'elevation' (float32) and
                  'source_id' (uint32) DataArrays, with a dictionary in `attrs['provenance_dict']`
                  mapping `source_id` integers to source asset names.
        """
        pass

    @abstractmethod
    def get_metadata(self) -> dict[str, Any]:
        """
        Return metadata about the provider.

        Returns:
            Dict[str, Any]: Dictionary containing provider name, version,
                            citation, and any other relevant metadata.
        """
        pass
