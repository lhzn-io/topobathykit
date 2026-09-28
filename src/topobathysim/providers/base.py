import fnmatch
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, cast

import numpy as np
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
    Abstract Base Class for TopoBathySim data providers.

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
