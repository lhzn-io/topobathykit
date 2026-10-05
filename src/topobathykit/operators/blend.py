"""
Geospatial blend operators for topobathykit.

This module contains functions for mixing and merging elevation datasets,
specifically handling CRS transformations and metric distance feathering.
"""

from typing import cast

import numpy as np
import xarray as xr
from scipy.ndimage import distance_transform_edt


def overwrite(base: xr.DataArray, overlay: xr.DataArray, mask: xr.DataArray | None = None) -> xr.DataArray:
    """
    Overwrites base with overlay where mask is True (or where overlay is valid if mask is None).
    """
    if mask is None:
        mask = overlay.notnull()

    # Valid overlay pixels replace base pixels.
    # If overlay is NaN where mask is True, the result will be NaN (desired hard cut behavior).

    return cast(xr.DataArray, base.where(~mask, overlay))


def metric_feather(base: xr.DataArray, overlay: xr.DataArray, distance_m: float) -> xr.DataArray:
    """
    Blends an overlay onto a base using a distance-based feathering in meters.

    This function handles CRS differences by internally reprojecting the overlay's
    valid mask to a metric CRS (WebMercator EPSG:3857) to calculate the distance
    transform. This ensures that the feathering width is consistent in meters
    regardless of the input CRS (e.g., WGS84) or latitude distortion.

    The blending algorithm is:
    1. Align overlay to base grid.
    2. Compute valid data mask.
    3. Compute Euclidean Distance Transform (EDT) on the mask in metric space.
    4. Normalize distances to weights (0.0 at edge, 1.0 at distance_m).
    5. Linear blend: result = base * (1 - weight) + overlay * weight.

    Args:
        base (xr.DataArray): The base elevation layer (e.g., GEBCO).
        overlay (xr.DataArray): The overlay elevation layer (e.g., BlueTopo).
                                Must have a valid CRS and transform.
        distance_m (float): The blending distance in meters.
                            Pixels within this distance from the edge of the
                            overlay's valid data will be blended.

    Returns:
        xr.DataArray: The blended elevation array in the same CRS and coordinates as the base.
    """
    # 0. Handle Zero/Negative Distance (Hard Cut)
    if distance_m <= 0:
        return overwrite(base, overlay)

    # 1. Align overlay to base grid
    # 1. Align overlay to base grid
    if not overlay.rio.crs:
        raise ValueError("Overlay must have a CRS defined.")

    aligned_overlay = overlay.rio.reproject_match(base)

    # 2. Identify valid data region
    valid_mask = ~np.isnan(aligned_overlay)

    if not valid_mask.any():
        return base

    # 3. Compute Weights in Metric Space (EPSG:3857)
    # We use nearest neighbor for mask reprojection to maintain crisp edges for the distance transform.
    mask_source = valid_mask.astype(np.uint8)
    mask_source.rio.write_nodata(0, inplace=True)

    mask_metric = mask_source.rio.reproject(
        "EPSG:3857",
        resolution=None,
        resampling=0,  # Nearest
    )

    # Compute Euclidean Distance Transform
    # Result is distance in pixels, so we multiply by resolution to get meters.
    res_meters = np.mean(np.abs(mask_metric.rio.resolution()))
    dist_pixels = distance_transform_edt(mask_metric.values)
    dist_meters = dist_pixels * res_meters

    # Normalize to 0-1 weight
    weights_metric_values = np.clip(dist_meters / distance_m, 0, 1)

    weights_metric = xr.DataArray(weights_metric_values, coords=mask_metric.coords, dims=mask_metric.dims)
    weights_metric.rio.write_crs("EPSG:3857", inplace=True)

    # 4. Reproject weights back to Base CRS
    # Bilinear resampling ensures smooth gradients in the weights
    weights_native = weights_metric.rio.reproject_match(base, resampling=1)
    weights_native = weights_native.fillna(0)

    # 5. Apply Blending
    # We only blend where BOTH base and overlay have valid data.
    # If base is missing, we use overlay directly.
    # If overlay is missing, we use base directly.
    valid_base = base.notnull()

    result = base.copy()

    overlap = valid_base & valid_mask
    blended = (base * (1 - weights_native)) + (aligned_overlay * weights_native)

    result = xr.where(overlap, blended, result)
    result = xr.where(~valid_base & valid_mask, aligned_overlay, result)

    return cast(xr.DataArray, result)
