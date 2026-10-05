"""
Core Runtime for topobathykit.

This module executes fusion policies to generate topobathymetric datasets.
"""

import contextlib
import gc
import hashlib
import json
import logging
import multiprocessing
import os
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import rioxarray  # noqa: F401
import xarray as xr
from affine import Affine
from pyproj import CRS, Transformer

from topobathykit.policy.loader import (
    generate_provider_legend,
    hash_policy,
    load_policy,
    load_policy_from_str,
)
from topobathykit.policy.schema import OperatorType
from topobathykit.providers.base import ProviderNoDataError, despike_median_deviation
from topobathykit.providers.registry import registry

from .config import get_cache_root
from .operators.blend import metric_feather, overwrite

# Suppress rasterio boto3/dummysession spam
logging.getLogger("rasterio.session").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def should_consolidate() -> bool:
    """
    Check if Zarr consolidation should be enabled based on environment variables.
    Defaults to True for performance, but can be disabled via TOPOBATHY_ZARR_CONSOLIDATED=0/false
    to support non-Python Zarr readers like Zarr.jl.
    """
    val = os.environ.get("TOPOBATHY_ZARR_CONSOLIDATED", "true").lower()
    return val in ("true", "1", "yes", "on")


def _get_grid_cells(
    bbox: tuple[float, float, float, float], target_crs: str
) -> tuple[list[tuple[float, float, float, float]], float]:
    """Calculate the discrete standard grid cells covering the requested bounding box."""
    start_lon, start_lat, end_lon, end_lat = bbox

    # Either EPSG:4326 or default to 0.05 anyways for standard cells
    grid_size = 0.05

    # Snap to grid
    min_x_idx = np.floor(start_lon / grid_size)
    max_x_idx = np.floor(end_lon / grid_size)
    min_y_idx = np.floor(start_lat / grid_size)
    max_y_idx = np.floor(end_lat / grid_size)

    cells = []
    # +1 to include the max index
    for x in range(int(min_x_idx), int(max_x_idx) + 1):
        for y in range(int(min_y_idx), int(max_y_idx) + 1):
            cell_min_lon = x * grid_size
            cell_min_lat = y * grid_size
            cell_max_lon = (x + 1) * grid_size
            cell_max_lat = (y + 1) * grid_size
            cells.append((cell_min_lon, cell_min_lat, cell_max_lon, cell_max_lat))

    return cells, grid_size


def _run_cell(
    policy: Any,
    cell_bbox: tuple[float, float, float, float],
    resolution: float | None,
    time: datetime | None,
    target_crs: str,
    halo_pct: float = 0.10,
) -> xr.Dataset:
    """Execute fusion on a standard grid cell with a small context halo."""
    # 1. Expand bbox by halo percentage to fetch context for seamless blending
    min_lon, min_lat, max_lon, max_lat = cell_bbox
    lon_halo = (max_lon - min_lon) * halo_pct
    lat_halo = (max_lat - min_lat) * halo_pct

    fetch_bbox = (
        min_lon - lon_halo,
        min_lat - lat_halo,
        max_lon + lon_halo,
        max_lat + lat_halo,
    )

    # We calculate explicit Reprojection bounds for Target CRS canvases.
    # The Halo padding ensures seamless pixel transitions across edges.

    # Reproject BBox if Target CRS is not EPSG:4326
    if target_crs != "EPSG:4326":
        transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
        # Fetch canvas bounds using the HALO bbox
        f_min_x, f_min_y = transformer.transform(fetch_bbox[0], fetch_bbox[1])
        f_max_x, f_max_y = transformer.transform(fetch_bbox[2], fetch_bbox[3])
        fetch_min_x, fetch_max_x = min(f_min_x, f_max_x), max(f_min_x, f_max_x)
        fetch_min_y, fetch_max_y = min(f_min_y, f_max_y), max(f_min_y, f_max_y)

        # We also need the EXACT cell bounds in the target CRS to clip the halo away later
        c_min_x, c_min_y = transformer.transform(min_lon, min_lat)
        c_max_x, c_max_y = transformer.transform(max_lon, max_lat)
        clip_min_x, clip_max_x = min(c_min_x, c_max_x), max(c_min_x, c_max_x)
        clip_min_y, clip_max_y = min(c_min_y, c_max_y), max(c_min_y, c_max_y)

    else:
        fetch_min_x, fetch_min_y, fetch_max_x, fetch_max_y = fetch_bbox
        clip_min_x, clip_min_y, clip_max_x, clip_max_y = cell_bbox

    # Resolve Resolution
    # Default to 30.0 meters if not provided
    input_resolution_meters = resolution if resolution is not None else 30.0

    # Smart Resolution Logic
    if target_crs == "EPSG:4326":
        # Global isotropic alignment to guarantee flawless xarray mosaicing
        # 1 degree at equator is approx 111320.0 meters
        res_x = input_resolution_meters / 111320.0
        res_y = input_resolution_meters / 111320.0
    else:
        res_x = input_resolution_meters
        res_y = input_resolution_meters

    # 3. Snap Canvas to Global Resolution Grid
    # This guarantees that regardless of the requested cell or halo,
    # the pixel centers align perfectly globally, avoiding
    # interpolation/interleaving artifacts when mosaicing.
    x_idx_start = int(np.floor(fetch_min_x / res_x))
    x_idx_end = int(np.ceil(fetch_max_x / res_x))
    width = max(1, x_idx_end - x_idx_start)

    y_idx_start = int(np.floor(fetch_min_y / res_y))
    y_idx_end = int(np.ceil(fetch_max_y / res_y))
    height = max(1, y_idx_end - y_idx_start)

    # Actual snapped bounds
    snapped_min_x = x_idx_start * res_x
    snapped_max_y = y_idx_end * res_y

    transform = Affine.translation(snapped_min_x, snapped_max_y) * Affine.scale(res_x, -res_y)

    # Calculate pixel centers globally
    xs = (np.arange(x_idx_start, x_idx_start + width) + 0.5) * res_x
    ys = (np.arange(y_idx_end - 1, y_idx_end - 1 - height, -1) + 0.5) * res_y

    # Optional: round coordinates to 8 decimal places for floating-point stability
    xs = np.round(xs, 8)
    ys = np.round(ys, 8)

    elevation = xr.DataArray(
        data=np.full((height, width), np.nan, dtype=np.float32),
        coords={"y": ys, "x": xs},
        dims=("y", "x"),
        name="elevation",
    )
    elevation.rio.write_crs(target_crs, inplace=True)
    elevation.rio.write_transform(transform, inplace=True)

    source_elevation = xr.DataArray(
        data=np.full((height, width), 0, dtype=np.uint32),
        coords={"y": ys, "x": xs},
        dims=("y", "x"),
        name="source_elevation",
    )
    source_elevation.rio.write_crs(target_crs, inplace=True)
    source_elevation.rio.write_transform(transform, inplace=True)
    source_elevation.attrs["_FillValue"] = 0
    source_elevation.encoding["_FillValue"] = 0

    base_legend = generate_provider_legend(policy)
    cell_provenance_dict: dict[int, dict[str, str]] = {
        k: {"name": f"{v} (Base Terrain)", "provider": v} for k, v in base_legend.items()
    }

    # Per-step outcome record. hydrate() reads it to refuse caching a cell in
    # which any step failed, so a transient error is retried on the next run
    # instead of being locked into the cache. `pixels` counts the halo-canvas
    # pixels the step supplied valid data for.
    step_outcomes: list[dict[str, Any]] = []

    def _record(
        provider: str, status: str, pixels: int = 0, error: str | None = None, despiked: int = 0
    ) -> None:
        outcome: dict[str, Any] = {"provider": provider, "status": status, "pixels": int(pixels)}
        if error:
            outcome["error"] = error[:500]
        if despiked:
            outcome["despiked"] = int(despiked)
        step_outcomes.append(outcome)

    # 5. Execution Loop
    for variable in policy.variables:
        if variable.name != "elevation":
            continue

        if variable.background is not None:
            elevation = elevation.fillna(variable.background)

        logger.info(f"Execution order: {[s.provider for s in variable.steps]}")

        for step in variable.steps:
            provider_cls = registry.get_provider_class(step.provider)
            provider_instance = provider_cls()

            try:
                # Fetch using the expanded halo bbox to ensure we grab edge data
                fetched_data = provider_instance.fetch_layer(
                    fetch_bbox,
                    resolution=input_resolution_meters,
                    crs="EPSG:4326",
                    filter=step.filter.model_dump() if step.filter else {},
                )
            except ProviderNoDataError as e:
                logger.info(f"{step.provider}: NoData - {e}")
                _record(step.provider, "nodata")
                continue
            except MemoryError as e:
                # Logged apart from other failures: under the hydration RLIMIT_AS
                # ceiling this is the most likely cause of a dropped step.
                logger.error(f"{step.provider}: MemoryError during fetch - {e}", exc_info=True)
                _record(step.provider, "error", error=f"MemoryError: {e}")
                continue
            except Exception as e:
                logger.error(f"{step.provider}: Failed to fetch - {e}", exc_info=True)
                _record(step.provider, "error", error=f"{type(e).__name__}: {e}")
                continue

            if fetched_data is None:
                logger.info(f"[PROBE-RUNTIME] Provider {step.provider} returned None.")
                _record(step.provider, "nodata")
                continue

            # A provider that dropped tiles or projects to errors still returns
            # what it has, flagged with a count of the errors.
            fetch_errors = int(getattr(fetched_data, "attrs", {}).get("fetch_errors", 0) or 0)

            # Load the data into memory IMMEDIATELY so we never trigger
            # complex Dask blockwise warnings when filtering or reprojecting!
            with contextlib.suppress(AttributeError):
                fetched_data = fetched_data.compute()

            # Extract per-provider source provenance from Dataset
            provider_source_id_da = None
            if isinstance(fetched_data, xr.Dataset):
                if "provenance_dict" in fetched_data.attrs:
                    cell_provenance_dict.update(fetched_data.attrs["provenance_dict"])
                provider_source_id_da = fetched_data.get("source_id")
                fetched_data = fetched_data["elevation"]

            # --- Global Filter Constraints ---
            if step.filter:
                if step.filter.min_elevation is not None:
                    fetched_data = fetched_data.where(fetched_data >= step.filter.min_elevation)
                if step.filter.max_elevation is not None:
                    fetched_data = fetched_data.where(fetched_data <= step.filter.max_elevation)

            # --- Enforce 2D Array Shape ---
            # Providers might return (Band, Y, X) or (Y, X, Band)
            if fetched_data.ndim == 3:
                if fetched_data.shape[0] == 1:
                    # (1, y, x) -> (y, x)
                    fetched_data = fetched_data.squeeze(axis=0)
                elif fetched_data.shape[-1] == 1:
                    # (y, x, 1) -> (y, x)
                    fetched_data = fetched_data.squeeze(axis=-1)
                else:
                    # Multi-band (C, y, x) -> Take first band
                    logger.warning(
                        f"Provider {step.provider} returned multi-band data {fetched_data.shape}. "
                        "Using first band only."
                    )
                    fetched_data = fetched_data[0, :, :]

            # Final check to ensure we are 2D
            if fetched_data.ndim != 2:
                logger.warning(
                    f"Provider {step.provider} returned unexpected shape {fetched_data.shape}. Skipping."
                )
                _record(step.provider, "error", error=f"unexpected shape {fetched_data.shape}")
                continue
            # ------------------------------

            # Align to Canvas
            try:
                # Defensive check on duplicates before reproject
                # Handle both projected (x,y) and geographic (lon,lat) dimensions
                for dim in ["x", "y", "lon", "lat"]:
                    if (
                        dim in fetched_data.dims
                        and dim in fetched_data.indexes
                        and not fetched_data.indexes[dim].is_unique
                    ):
                        fetched_data = fetched_data.drop_duplicates(dim=dim)

                aligned_data = fetched_data.rio.reproject_match(elevation).compute()

            except Exception as e:
                logger.error(f"Reprojection/Index Alignment failed for {step.provider}: {e}")
                # Try dropping duplicates forcefully again? Or continue
                _record(step.provider, "error", error=f"alignment: {type(e).__name__}: {e}")
                continue

            # --- Despike at canvas resolution ---
            # Runs after alignment because the filter only sees features about one pixel
            # wide at the scale it runs at (see despike_median_deviation). Masked pixels
            # become NaN, so lower-priority steps show through them. ncei_bag already
            # applies the same thresholds at its native resolution, so it is not filtered
            # twice.
            despiked = 0
            if step.filter and step.provider != "ncei_bag":
                max_dev = step.filter.max_deviation or step.filter.max_depth_change
                if max_dev:
                    aligned_data, despiked = despike_median_deviation(aligned_data, float(max_dev))
                    if despiked:
                        logger.info(
                            f"{step.provider}: despiked {despiked} pixel(s) deviating more than "
                            f"{float(max_dev)} m from their 3x3 median in cell {cell_bbox}"
                        )

            new_data_mask = aligned_data.notnull().compute()

            aligned_source_id = None
            if provider_source_id_da is not None:
                try:
                    from rasterio.enums import Resampling

                    aligned_source_id = provider_source_id_da.rio.reproject_match(
                        elevation, resampling=Resampling.nearest
                    ).compute()
                    # Squeeze any spurious band dimension introduced by reproject
                    if "band" in aligned_source_id.dims:
                        aligned_source_id = aligned_source_id.squeeze("band", drop=True)
                    if aligned_source_id.ndim != 2:
                        aligned_source_id = aligned_source_id.squeeze(drop=True)
                except Exception as e:
                    logger.warning(f"Failed to reproject source_id mask for {step.provider}: {e}")

            # Use provider data wherever elevation is valid. Source IDs may be missing after
            # reprojection, so we can fall back to provider-level IDs for attribution.
            provider_valid_mask = new_data_mask

            step_status = "partial" if fetch_errors else "ok"
            step_error = f"{fetch_errors} tile/project fetch error(s)" if fetch_errors else None
            valid_pixels = int(provider_valid_mask.sum())
            if not valid_pixels:
                _record(
                    step.provider,
                    "partial" if fetch_errors else "nodata",
                    error=step_error,
                    despiked=despiked,
                )
                continue
            _record(step.provider, step_status, pixels=valid_pixels, error=step_error, despiked=despiked)

            # 4. Generate Provider Legend & IDs
            legend = generate_provider_legend(policy)
            provider_to_id = {v: k for k, v in legend.items()}

            provider_id = provider_to_id.get(step.provider, 0)

            # --- Pairwise Logic ---
            # 1. Initialize mask of pixels to be processed by default rule
            remaining_mask = provider_valid_mask.copy()

            # Apply Transitions (Specific Overrides)
            for rule in step.transitions:
                target_id = provider_to_id.get(rule.target_provider)
                if target_id is None:
                    continue

                # Identify overlap: Valid New Data AND Underlying Source == TargetID
                transition_mask = provider_valid_mask & (source_elevation == target_id)

                if not transition_mask.any():
                    continue

                # Apply specific operator
                if rule.operator == OperatorType.metric_feather and rule.blend_distance:
                    blended = metric_feather(elevation, aligned_data, rule.blend_distance)
                elif rule.operator == OperatorType.overwrite:
                    blended = overwrite(elevation, aligned_data, transition_mask)
                else:
                    # Fallback to overwrite for unsupported operators in V0.
                    # Edge Case: If operator is 'metric_feather' but blend_distance is None,
                    # we fall through to here (Safe Default: Hard Cut).
                    blended = overwrite(elevation, aligned_data, transition_mask)

                # Update Canvas for this transition region
                elevation = elevation.where(~transition_mask, blended)

                # Remove these pixels from the default processing list
                remaining_mask = remaining_mask & (~transition_mask)

            # Apply Default Operation to remaining pixels
            if remaining_mask.any():
                if step.operation == OperatorType.metric_feather and step.blend_distance:
                    blended_default = metric_feather(elevation, aligned_data, step.blend_distance)
                elif step.operation == OperatorType.overwrite:
                    blended_default = overwrite(elevation, aligned_data, remaining_mask)
                else:
                    blended_default = overwrite(elevation, aligned_data, remaining_mask)

                elevation = elevation.where(~remaining_mask, blended_default)

            # Update Provenance (Global update for all valid pixels)
            if aligned_source_id is not None:
                source_detail_mask = (
                    provider_valid_mask & aligned_source_id.notnull() & (aligned_source_id > 0)
                )
                source_elevation = xr.where(
                    source_detail_mask,
                    aligned_source_id,
                    source_elevation,  # type: ignore
                )
                source_fallback_mask = provider_valid_mask & (~source_detail_mask)
                source_elevation = xr.where(source_fallback_mask, provider_id, source_elevation)  # type: ignore
            else:
                source_elevation = xr.where(provider_valid_mask, provider_id, source_elevation)  # type: ignore
            # Enforce 2D — xr.where can reintroduce a band dim from aligned_source_id
            if "band" in source_elevation.dims:
                source_elevation = source_elevation.squeeze("band", drop=True)

    # 6. Clip Halo Away! We only want the exact cell bounds
    # The cell bounds are clip_min_x, clip_min_y, clip_max_x, clip_max_y
    # We clip cleanly before saving to cache. Because we are projecting/translating,
    # pixel alignment might mean we should slice cleanly.

    # We slice based on 'x' and 'y' coordinates of the dataset.
    # Note: ys are usually decreasing (max_y to min_y) in raster.

    # Use epsilon > 0.5 * res_x so that any pixel center within one step of a cell boundary is
    # always included by at least one adjacent cell.  The 0.05° WGS84 cell grid does not align
    # exactly with the EPSG:3857 pixel grid, so the clip boundary can fall between two pixel
    # centers leaving a 1-pixel gap.  0.6 * res_x guarantees overlap (handled by drop_duplicates).
    epsilon = res_x * 0.6
    ds = xr.Dataset({"elevation": elevation, "source_elevation": source_elevation})

    # Slice X
    ds = ds.sel(x=slice(clip_min_x - epsilon, clip_max_x + epsilon))

    # Slice Y (Handle both Ascending and Descending Y coordinates)
    if len(ds.y) > 0 and ds.y[0] > ds.y[-1]:
        # Descending (North Up)
        ds = ds.sel(y=slice(clip_max_y + epsilon, clip_min_y - epsilon))
    else:
        # Ascending (South Up)
        ds = ds.sel(y=slice(clip_min_y - epsilon, clip_max_y + epsilon))

    # Ensure valid dimensions
    if ds.x.size == 0 or ds.y.size == 0:
        logger.warning(f"Clip resulted in empty dimensions for cell {cell_bbox}")

    logger.debug(f"_run_cell returning provenance: {cell_provenance_dict}")
    ds.attrs["provenance_dict"] = cell_provenance_dict
    # Serialised as JSON because zarr and netCDF attributes cannot hold a list of dicts.
    ds.attrs["step_outcomes_json"] = json.dumps(step_outcomes)

    return cast(xr.Dataset, ds)


# Default ceiling on the NaN fraction of a cell before hydrate refuses to cache it.
# Complete cells have no NaN when the policy has full coverage; the 2026-05 WLIS
# incident left 0.07% to 24.5% NaN in cells that dropped a step.
DEFAULT_MAX_CELL_NAN_FRACTION = 0.01


def _max_cell_nan_fraction() -> float:
    """Return the NaN ceiling for caching, from TOPOBATHY_MAX_CELL_NAN_FRACTION if set."""
    raw = os.environ.get("TOPOBATHY_MAX_CELL_NAN_FRACTION")
    if raw is None or raw.strip() == "":
        return DEFAULT_MAX_CELL_NAN_FRACTION
    try:
        return float(raw)
    except ValueError:
        logger.warning(f"Ignoring invalid TOPOBATHY_MAX_CELL_NAN_FRACTION={raw!r}")
        return DEFAULT_MAX_CELL_NAN_FRACTION


def _cell_integrity_problems(ds: xr.Dataset, max_nan_fraction: float | None) -> list[str]:
    """Return the reasons a freshly computed cell must not be cached (empty when complete).

    A cell is incomplete when any policy step raised a fetch error or returned a
    partial result, or (unless `max_nan_fraction` is None) when it is all NaN or its
    NaN fraction exceeds `max_nan_fraction`. Steps that raised `ProviderNoDataError`
    (no coverage) do not count against the cell.
    """
    problems: list[str] = []
    try:
        outcomes = json.loads(ds.attrs.get("step_outcomes_json", "[]"))
    except (TypeError, ValueError):
        outcomes = []
    for outcome in outcomes:
        if outcome.get("status") in ("error", "partial"):
            detail = f": {outcome['error']}" if outcome.get("error") else ""
            problems.append(f"step {outcome.get('provider')} {outcome.get('status')}{detail}")

    if "elevation" not in ds:
        problems.append("no elevation variable")
        return problems
    total = int(ds["elevation"].size)
    valid = int(ds["elevation"].notnull().sum())
    nan_fraction = 1.0 - (valid / total) if total else 1.0
    ds.attrs["nan_fraction"] = round(nan_fraction, 6)
    if max_nan_fraction is None:
        return problems
    if valid == 0:
        problems.append("all-NaN elevation")
    elif nan_fraction > max_nan_fraction:
        problems.append(f"NaN fraction {nan_fraction:.4f} exceeds {max_nan_fraction:.4f}")
    return problems


def _resolve_policy(policy_input: str | Path) -> Any:
    """
    Resolve policy input to a FusionPolicy object.
    Accepts a filepath (str/Path) OR a raw YAML string.
    """
    # Naive check: if it looks like a path and exists, load it
    s_input = str(policy_input)
    if (s_input.endswith(".yaml") or s_input.endswith(".yml")) and os.path.exists(s_input):
        return load_policy(s_input)

    # Otherwise treat as raw content
    return load_policy_from_str(s_input)


def get_fused_cache_info(
    policy_input: str | Path,
    bbox: tuple[float, float, float, float],
    resolution: float = 30.0,
) -> tuple[Path, str]:
    """
    Returns the expected Path and the hash key for a fused Zarr cache file.
    Hashes the POLICY CONTENT + BBOX + RES to create a unique ID.
    This centralized function ensures metadata parity across all endpoints.
    """
    import json

    policy = _resolve_policy(policy_input)
    target_crs = policy.crs

    cache_dir = get_cache_root() / "fused_zarr"

    # MD5 Hashing logic (Standardized Precision: 8 decimals)
    # We hash the policy content digest itself to keep the key short but content-addressed
    policy_content_hash = hash_policy(policy.model_dump())

    # Save policy to cache if it doesn't exist
    policies_dir = cache_dir.parent / "policies"
    policies_dir.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(c for c in getattr(policy, "name", "Unnamed") if c.isalnum()) or "Unnamed"
    short_hash = policy_content_hash[:8]
    policy_cache_path = policies_dir / f"{safe_name}.{short_hash}.yaml"
    if not policy_cache_path.exists():
        import yaml

        with open(policy_cache_path, "w") as f:
            yaml.dump(policy.model_dump(mode="json"), f)

    # Quantize resolution to nearest 0.5m for the cache key.
    # This absorbs latitude-dependent variation within a zoom level
    # (e.g. z=13 gives 14.44-14.48m across a region → all key to 14.5m)
    # while keeping distinct zoom levels separate (14.5 vs 7.0 vs 29.0).
    # The actual computation still uses the full-precision resolution.
    quantized_res = round(resolution * 2) / 2

    key_dict = {
        "policy_hash": policy_content_hash,
        "cell": [round(x, 8) for x in bbox],
        "res": quantized_res,
        "crs": target_crs,
    }
    key_str = json.dumps(key_dict, sort_keys=True, default=str)
    key_hash = hashlib.md5(key_str.encode()).hexdigest()

    return cache_dir / short_hash / f"{key_hash}.zarr", key_hash


def get_cache_info_for_point(
    policy_path: str,
    lon: float,
    lat: float,
    resolution: float,
) -> tuple[Path, str]:
    """
    Finds the fused cache file corresponding to the standard grid cell containing the point.
    This ensures alignment with the tiling engine which caches by standard grid cells.
    """
    policy = load_policy(policy_path)
    # Treat point as a degenerate bbox to verify grid cell alignment
    cells, _ = _get_grid_cells((lon, lat, lon, lat), policy.crs)

    if not cells:
        # Should be covered by at least one cell if valid
        # If outside area of use (rarely checked here), fallback
        raise ValueError(f"Point {lon},{lat} not covered by any '{policy.crs}' grid cell.")

    # There should only be one cell for a point
    cell_bbox = cells[0]
    return get_fused_cache_info(policy_path, cell_bbox, resolution)


def get_fused_cache_path(
    policy_input: str | Path,
    bbox: tuple[float, float, float, float],
    resolution: float = 30.0,
) -> Path:
    """Legacy wrapper for get_fused_cache_info returning only the Path."""
    path, _ = get_fused_cache_info(policy_input, bbox, resolution)
    return path


def _zoom_resolution(zoom: int, center_lat: float) -> float:
    """Compute tile-endpoint-compatible resolution for a given zoom and latitude.

    Uses the same formula as the /elevation and /tiles endpoints:
        res = 156543.03 * cos(center_lat_rad) / 2^zoom
    """
    import math

    return round(156543.03 * math.cos(math.radians(center_lat)) / (2.0**zoom), 8)


def hydrate(
    policy_input: str | Path | Any,
    bbox: tuple[float, float, float, float],
    resolution: float | None = None,
    time: datetime | None = None,
    max_workers: int | None = 2,
    on_progress: Any | None = None,
    target_zoom: int | None = None,
    max_nan_fraction: float | None = None,
) -> dict[str, int]:
    """
    Hydrate the cache for a given bbox and resolution by processing all covering grid cells.
    Does not return the fused dataset, only ensures cells are cached.

    Args:
        on_progress: Optional callback called with (stats_dict) after each cell completes.
                     Used by the service to write progress to a state file.
        target_zoom: If set, the hydration was requested at a specific slippy-map zoom level
                     rather than an explicit resolution in meters.
        max_nan_fraction: Cells whose NaN fraction exceeds this are counted as failed and
                     not cached. Defaults to TOPOBATHY_MAX_CELL_NAN_FRACTION, else 0.01.
                     Pass 1.0 to accept any cell that is not entirely NaN.
    """

    policy = _resolve_policy(policy_input)
    target_crs = policy.crs
    res = resolution if resolution else 30.0
    nan_ceiling = _max_cell_nan_fraction() if max_nan_fraction is None else max_nan_fraction

    # Determine Grid Cells
    cells, grid_cell_size = _get_grid_cells(bbox, target_crs)
    legend = generate_provider_legend(policy)

    stats = {"total": len(cells), "cached": 0, "processed": 0, "failed": 0}
    if on_progress:
        on_progress(stats)

    policy_display_name = getattr(policy, "name", "Unnamed") or "Unnamed"
    logger.info(
        f"Starting hydration for {len(cells)} cells [{policy_display_name}]. BBox: {bbox}, Res: {res}"
    )

    def _process_cell(i: int, cell_bbox: tuple[float, float, float, float]) -> str:
        # Use centralized cache helper
        cache_path, key_hash = get_fused_cache_info(policy_input, cell_bbox, resolution=res)

        # Ensure parent dir exists (lazy creation)
        if not cache_path.parent.exists():
            cache_path.parent.mkdir(parents=True, exist_ok=True)

        is_cached = False
        if cache_path.exists():
            try:
                # Lightweight check: valid Zarr?
                ds_meta = xr.open_dataset(cache_path, engine="zarr", decode_coords="all")

                valid_grid = ds_meta.attrs.get("grid_cell_size") == grid_cell_size
                # We trust our hash includes policy content, so a hit is a hit on policy content.
                # But we double check legend just in case of collisions or old data.
                cached_legend = ds_meta.attrs.get("policy_legend")
                valid_legend = not (cached_legend and cached_legend != str(legend))

                if valid_grid and valid_legend:
                    is_cached = True

                ds_meta.close()
            except Exception as e:
                logger.warning(f"Cache check failed for {cache_path}: {e}")

        if is_cached:
            if i % 10 == 0:
                logger.info(f"Hydration progress: {i}/{len(cells)} (Cached)")
            return "cached"

        policy_display_name = getattr(policy, "name", "Unnamed") or "Unnamed"
        logger.info(
            f"Hydrating cell {i}/{len(cells)} [{policy_display_name}]: {cell_bbox} -> {cache_path.name}"
        )

        # Process and Cache
        try:
            ds_cell = _run_cell(
                policy=policy,
                cell_bbox=cell_bbox,
                resolution=res,
                time=time,
                target_crs=target_crs,
                halo_pct=0.10,
            )

            # Refuse to cache an incomplete cell. A cached zarr is treated as a valid
            # hit on every later hydrate run, so caching a cell in which a step failed
            # (network timeout, VDatum outage, MemoryError) or that is mostly NaN would
            # permanently lock in the hole. Leaving it uncached retries it next time.
            problems = _cell_integrity_problems(ds_cell, nan_ceiling)
            if problems:
                logger.warning(
                    f"Cell {cell_bbox} is incomplete; skipping cache write so it will be "
                    f"retried on next hydrate: {'; '.join(problems)}"
                )
                return "failed"

            new_attrs: dict[str, object] = {
                "policy_hash": key_hash,
                "policy_legend": str(legend),
                "grid_cell_size": grid_cell_size,
                "crs": target_crs,
                "created_at": datetime.utcnow().isoformat(),
                "cell_bbox": list(cell_bbox),
                "hydrated": "true",
                "source": "hydrate",
                "resolution_m": round(res, 4),
            }
            if target_zoom is not None:
                new_attrs["target_zoom"] = int(target_zoom)
                new_attrs["resolution_origin"] = "zoom"
            else:
                new_attrs["resolution_origin"] = "meters"
            ds_cell.attrs.update(new_attrs)
            if "provenance_dict" in ds_cell.attrs:
                ds_cell.attrs["provenance_dict_json"] = json.dumps(ds_cell.attrs["provenance_dict"])

            # Save to Cache
            ds_chunked = ds_cell.chunk({"y": 2048, "x": 2048})
            tmp_path = cache_path.with_suffix(f".tmp.{key_hash}.zarr")

            if tmp_path.exists():
                import shutil

                shutil.rmtree(tmp_path)

            ds_chunked.to_zarr(tmp_path, mode="w", consolidated=should_consolidate())

            if cache_path.exists():
                import shutil

                shutil.rmtree(cache_path)

            import shutil

            shutil.move(str(tmp_path), str(cache_path))

            del ds_chunked  # release reference before close
            ds_cell.close()
            del ds_cell
            gc.collect()
            return "processed"

        except Exception as e:
            logger.error(f"Failed to hydrate cell {cell_bbox}: {e}", exc_info=True)
            return "failed"

    import concurrent.futures

    # Process cells in small batches using ProcessPoolExecutor.
    # Each cell runs in its own subprocess with isolated memory, so:
    # - A cell that hits MemoryError dies without affecting others
    # - GC of large provider arrays is guaranteed on process exit
    # - No GIL contention between cells doing CPU-heavy reprojection
    # Falls back to ThreadPoolExecutor if fork is unavailable.
    batch_size = max(max_workers or 2, 2) * 2
    cell_items = list(enumerate(cells, 1))

    # Daemon processes cannot spawn children (ProcessPoolExecutor would fail with
    # "daemonic processes are not allowed to have children"). This happens when
    # hydrate() runs inside the /hydrate endpoint's subprocess. Fall back to threads.
    current = multiprocessing.current_process()
    is_daemon = getattr(current, "daemon", False)

    pool_cls: type = concurrent.futures.ThreadPoolExecutor
    pool_kwargs: dict[str, Any] = {"max_workers": max_workers}

    if not is_daemon:
        try:
            mp_ctx = multiprocessing.get_context("fork")
            pool_cls = concurrent.futures.ProcessPoolExecutor
            pool_kwargs = {"max_workers": max_workers, "mp_context": mp_ctx}
        except ValueError:
            logger.info("ProcessPoolExecutor(fork) unavailable, falling back to threads")
    else:
        logger.info("Running inside daemon process, using ThreadPoolExecutor")

    for batch_start in range(0, len(cell_items), batch_size):
        batch = cell_items[batch_start : batch_start + batch_size]

        with pool_cls(**pool_kwargs) as executor:
            futures = {executor.submit(_process_cell, i, cb): cb for i, cb in batch}
            for future in concurrent.futures.as_completed(futures):
                try:
                    status = future.result()
                except Exception as e:
                    logger.error(f"Cell worker crashed: {e}")
                    status = "failed"
                stats[status] += 1
                if on_progress:
                    on_progress(stats)

        gc.collect()

    logger.info(f"Hydration complete. Stats: {stats}")
    return stats


def _collect_provenance(datasets: Iterable[xr.Dataset]) -> dict[int, dict[str, str]]:
    """Merge the per-cell provenance maps into one id -> source-info mapping.

    Zarr attributes cannot hold a dict, so cells written to cache carry the map
    serialised as `provenance_dict_json`; freshly computed cells carry the dict
    directly as `provenance_dict`. Accept either form.
    """
    merged: dict[int, dict[str, str]] = {}
    for ds in datasets:
        raw = ds.attrs.get("provenance_dict")
        if raw is None and "provenance_dict_json" in ds.attrs:
            try:
                raw = json.loads(ds.attrs["provenance_dict_json"])
            except (TypeError, ValueError) as e:
                logger.warning(f"Could not parse cached provenance: {e}")
                continue
        if isinstance(raw, dict):
            for k, v in raw.items():
                merged[int(k)] = v
    return merged


def run(
    policy_input: str | Path,
    bbox: tuple[float, float, float, float],
    resolution: float | None = None,
    time: datetime | None = None,
    use_cache: bool = True,
    target_zoom: int | None = None,
    source: str = "fuse",
) -> xr.Dataset:
    """
    Execute a fusion policy to generate a topobathymetric dataset.
    This public entrypoint chunks arbitrary bboxes into standard grid cells for caching.
    """
    policy = _resolve_policy(policy_input)
    target_crs = policy.crs

    # If the bbox exactly matches a single grid cell, we can bypass the expensive tiling logic
    # and directly check/write the singular cache entry.
    # This is critical for /elevation hits because it requests the exact PADDED tile bbox.
    is_standard_cell = False
    try:
        # We check if the requested bbox matches what _get_grid_cells would produce for a single tile
        cells, _ = _get_grid_cells(bbox, target_crs)
        if len(cells) == 1:
            c = cells[0]
            # Floating point tolerance
            if all(abs(a - b) < 1e-10 for a, b in zip(bbox, c, strict=False)):
                is_standard_cell = True
    except Exception:
        pass

    if is_standard_cell:
        # FAST PATH: Single cache-hit check for the whole request
        cache_path, key_hash = get_fused_cache_info(
            policy_input,
            bbox,
            resolution=resolution if resolution else 30.0,
        )
        if use_cache and cache_path.exists():
            logger.info(f"Fast Path Cache Hit: {cache_path.name}")
            ds = cast(xr.Dataset, xr.open_dataset(cache_path, engine="zarr", decode_coords="all").load())
            # Restore `provenance_dict` from its serialised form so a cache hit
            # exposes the same attributes as a freshly computed result.
            if "provenance_dict" not in ds.attrs:
                ds.attrs["provenance_dict"] = _collect_provenance([ds])
            return ds

    start_lon, start_lat, end_lon, end_lat = bbox

    # Validate CRS Area of Use
    try:
        crs_obj = CRS(target_crs)
        if crs_obj.area_of_use:
            # Area of Use bounds are (west, south, east, north)
            aou_w, aou_s, aou_e, aou_n = crs_obj.area_of_use.bounds

            # Simple intersection check
            if start_lon > aou_e or end_lon < aou_w or start_lat > aou_n or end_lat < aou_s:
                error_msg = (
                    f"Requested bounds {bbox} are outside the validity area for policy CRS {target_crs}. "
                    f"Area of Use: {crs_obj.area_of_use.name} ({aou_w}, {aou_s}, {aou_e}, {aou_n})"
                )
                raise ValueError(error_msg)
    except ImportError:
        pass
    except ValueError as e:
        raise e
    except Exception:
        pass

    # Determine Grid Cells
    cells, grid_cell_size = _get_grid_cells(bbox, target_crs)

    # For caching

    cache_dir = get_cache_root() / "fused_zarr"

    cache_dir.mkdir(parents=True, exist_ok=True)

    legend = generate_provider_legend(policy)

    # Process all cells
    cell_datasets = []

    for cell_bbox in cells:
        # Use centralized cache helper to ensure metadata parity across all endpoints
        cache_path, key_hash = get_fused_cache_info(
            policy_input,
            cell_bbox,
            resolution=resolution if resolution else 30.0,
        )

        ds_cell = None
        if use_cache and cache_path.exists():
            try:
                # We use chunks="auto" to use dask, but unify it immediately
                # to prevent Dask task graph explosion during combine_by_coords.
                ds_cell = xr.open_dataset(cache_path, engine="zarr", decode_coords="all")
                # Removed chunk re-alignment to dodge Dask entirely

                valid_grid = ds_cell.attrs.get("grid_cell_size") == grid_cell_size
                # If cached grid size doesn't match current logic, invalidate
                if not valid_grid:
                    logger.info(f"Cache Grid Mismatch for {cache_path.name}. Invalidating.")
                    ds_cell = None
                else:
                    # Clean up random dimensions
                    for var in ["elevation", "source_elevation"]:
                        if var in ds_cell and "band" in ds_cell[var].dims:
                            ds_cell[var] = ds_cell[var].squeeze("band", drop=True)

                    # Double check legend
                    cached_legend = ds_cell.attrs.get("policy_legend")
                    if cached_legend and cached_legend != str(legend):
                        logger.info(f"Policy Legend Mismatch for {cache_path.name}. Invalidating Cache.")
                        ds_cell = None

            except Exception as e:
                logger.warning(f"Failed to load cache {cache_path}: {e}")
                ds_cell = None

        if ds_cell is None:
            # Cache miss! Execute the cell
            ds_cell = _run_cell(
                policy=policy,
                cell_bbox=cell_bbox,
                resolution=resolution,
                time=time,
                target_crs=target_crs,
                halo_pct=0.10,
            )

            # Attributes
            new_attrs: dict[str, object] = {
                "policy_hash": hash_policy(policy.model_dump()),  # type: ignore
                "policy_legend": str(legend),
                "grid_cell_size": grid_cell_size,
                "crs": target_crs,
                "created_at": datetime.utcnow().isoformat(),
                "cell_bbox": list(cell_bbox),
                "source": source,
                "resolution_m": round(resolution if resolution else 30.0, 4),
            }
            if target_zoom is not None:
                new_attrs["target_zoom"] = int(target_zoom)
                new_attrs["resolution_origin"] = "zoom"
            else:
                new_attrs["resolution_origin"] = "meters"
            ds_cell.attrs.update(new_attrs)

            if "provenance_dict" in ds_cell.attrs:
                # Keep it as is or JSON dump it
                ds_cell.attrs["provenance_dict_json"] = json.dumps(ds_cell.attrs["provenance_dict"])

            # Serve a cell with a failed step, but do not cache it (see hydrate). The NaN
            # ceiling is not applied here: tile requests routinely cover areas outside
            # every provider, and refusing to cache those would recompute them per request.
            problems = _cell_integrity_problems(ds_cell, None) if use_cache else []
            if problems:
                logger.warning(
                    f"Cell {cell_bbox} is incomplete; returning it without caching: {'; '.join(problems)}"
                )

            # Attempt to save to cache
            if use_cache and not problems:
                try:
                    logger.info(f"Writing Fused Grid Cell Zarr to {cache_path}")
                    ds_chunked = ds_cell.chunk({"y": 2048, "x": 2048})
                    tmp_path = cache_path.with_suffix(f".tmp.{key_hash}.zarr")

                    if tmp_path.exists():
                        import shutil

                        shutil.rmtree(tmp_path)

                    ds_chunked.to_zarr(tmp_path, mode="w", consolidated=should_consolidate())

                    if cache_path.exists():
                        import shutil

                        shutil.rmtree(cache_path)

                    import shutil

                    shutil.move(str(tmp_path), str(cache_path))
                except Exception as e:
                    logger.error(f"Failed to cache cell {cache_path}: {e}")

        # No need to unify chunks since we bypassed Dask combination entirely!
        ds_cell = ds_cell.assign_coords(x=np.round(ds_cell.x.values, 6), y=np.round(ds_cell.y.values, 6))
        cell_datasets.append(ds_cell)

    # Collect provenance up front: the mosaic branch below clears
    # `cell_datasets` to free memory, so this cannot be deferred to the end.
    global_provenance = _collect_provenance(cell_datasets)

    # Mosaic the cells together
    if len(cell_datasets) == 1:
        merged_ds = cell_datasets[0]
    else:
        logger.info(
            f"Mosaicking {len(cell_datasets)} disjoint tiles via preallocated NumPy arrays "
            "(Avoiding Dask Chunk Explosion)"
        )

        # Sort coordinates to ensure monotonic indexes before combining.
        sorted_cells = [ds.sortby("y", ascending=False).sortby("x", ascending=True) for ds in cell_datasets]

        min_x = min(float(ds.x.min()) for ds in sorted_cells)
        max_x = max(float(ds.x.max()) for ds in sorted_cells)
        min_y = min(float(ds.y.min()) for ds in sorted_cells)
        max_y = max(float(ds.y.max()) for ds in sorted_cells)

        # Derive the mosaic step from the cells' own pixel spacing rather than the
        # `resolution` argument. Cells are emitted on a uniform grid in the policy
        # CRS, so in production this equals `resolution`. But `resolution` is
        # expressed in metres while the coordinates are in CRS units (degrees for
        # a geographic CRS); using it directly as a coordinate step collapses the
        # target grid to a handful of pixels. x and y are derived separately.
        def _pixel_spacing(vals: np.ndarray) -> float | None:
            d = np.abs(np.diff(np.asarray(vals, dtype=np.float64)))
            d = d[d > 0]
            return float(np.median(d)) if d.size else None

        fallback_res = float(resolution) if resolution is not None else 1.0
        x_spacings = [s for s in (_pixel_spacing(ds.x.values) for ds in sorted_cells) if s]
        y_spacings = [s for s in (_pixel_spacing(ds.y.values) for ds in sorted_cells) if s]
        res_x = float(np.median(x_spacings)) if x_spacings else fallback_res
        res_y = float(np.median(y_spacings)) if y_spacings else fallback_res

        x_coords_arr = np.arange(min_x, max_x + res_x / 2.0, res_x)
        y_coords_arr = np.arange(max_y, min_y - res_y / 2.0, -res_y)

        logger.info(
            f"Output grid dimensions: {len(y_coords_arr)}x{len(x_coords_arr)} "
            f"(Allocating {(len(y_coords_arr) * len(x_coords_arr) * 4) / 1024**2:.2f} MB per array)"
        )
        out_elev = np.full((len(y_coords_arr), len(x_coords_arr)), np.nan, dtype=np.float32)
        out_src = np.zeros((len(y_coords_arr), len(x_coords_arr)), dtype=np.uint32)
        has_src = any("source_elevation" in ds for ds in sorted_cells)

        cell_datasets.clear()  # Free the original list references immediately

        while sorted_cells:
            ds = sorted_cells.pop(0).load()  # Load individual tile, popping off list to free RAM immediately
            elev_data = ds["elevation"].values
            src_data = ds["source_elevation"].values if "source_elevation" in ds else None

            # Calculate precise insertion boundaries
            x_start_idx = round((float(ds.x[0]) - min_x) / res_x)
            y_start_idx = round((max_y - float(ds.y[0])) / res_y)

            h, w = elev_data.shape
            x_end = min(x_start_idx + w, len(x_coords_arr))
            y_end = min(y_start_idx + h, len(y_coords_arr))
            slice_w = x_end - x_start_idx
            slice_h = y_end - y_start_idx

            elev_slice = elev_data[:slice_h, :slice_w]
            existing_elev = out_elev[y_start_idx:y_end, x_start_idx:x_end]

            mask = ~np.isnan(elev_slice)
            existing_elev[mask] = elev_slice[mask]

            if has_src and src_data is not None:
                src_slice = src_data[:slice_h, :slice_w]
                existing_src = out_src[y_start_idx:y_end, x_start_idx:x_end]
                existing_src[mask] = src_slice[mask]

            ds.close()  # Free memory immediately for this chunk!
            import gc

            gc.collect()

        ds_dict: dict[str, tuple[tuple[str, str], np.ndarray]] = {"elevation": (("y", "x"), out_elev)}
        if has_src:
            ds_dict["source_elevation"] = (("y", "x"), out_src)
        merged_ds = xr.Dataset(ds_dict, coords={"x": x_coords_arr, "y": y_coords_arr})

        # Drop float duplicate coordinates that can arise at grid cell boundaries to prevent
        # downstream "Reindexing only valid with uniquely valued Index" errors.
        merged_ds = merged_ds.drop_duplicates(dim="x").drop_duplicates(dim="y")

        # Adjacent cells are computed independently and may have y-pixel grids that are
        # slightly offset from each other.  combine_by_coords unions both coordinate arrays
        # rather than aligning them, producing interleaved valid/NaN rows at cell boundaries.
        # Fill those stripes now so downstream linear interp doesn't propagate them.
        # Note: interpolate_na requires a monotonically increasing index; y is north-up
        # (descending), so sort ascending, fill, then restore.
        if "elevation" in merged_ds:
            elev = merged_ds["elevation"]
            if bool(elev.isnull().any()):
                merged_ds["elevation"] = (
                    elev.sortby("y", ascending=True)
                    .interpolate_na(dim="y", method="linear", limit=2)
                    .sortby("y", ascending=False)
                    .interpolate_na(dim="x", method="linear", limit=2)
                )

        logger.info(f"Mosaic complete. Final combined shape: {merged_ds.dims}")

    # Clip to the exact requested bbox, reprojecting bounds to target CRS if needed.
    if target_crs != "EPSG:4326":
        transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
        r_min_x, r_min_y = transformer.transform(start_lon, start_lat)
        r_max_x, r_max_y = transformer.transform(end_lon, end_lat)
        req_min_x, req_max_x = min(r_min_x, r_max_x), max(r_min_x, r_max_x)
        req_min_y, req_max_y = min(r_min_y, r_max_y), max(r_min_y, r_max_y)
    else:
        req_min_x, req_min_y = start_lon, start_lat
        req_max_x, req_max_y = end_lon, end_lat

    # Use item access for coordinates to avoid potential AttributeError on property access
    # if dataset structure is irregular
    x_coords = merged_ds["x"]
    # Fallback for single pixel or malformed grids
    epsilon = float(abs((x_coords[1] - x_coords[0]).values * 0.1)) if len(x_coords) > 1 else 0.0001

    # Ensure strictly python floats for slicing
    req_min_x = float(req_min_x)
    req_max_x = float(req_max_x)
    req_min_y = float(req_min_y)
    req_max_y = float(req_max_y)

    # Re-verify numeric index to prevent slicing errors
    from pandas.api.types import is_numeric_dtype

    if not is_numeric_dtype(merged_ds.indexes["x"].dtype):
        logger.warning(f"X Index is not numeric: {merged_ds.indexes['x'].dtype}. Slicing may fail.")

    merged_ds = merged_ds.sel(x=slice(req_min_x - epsilon, req_max_x + epsilon))
    if len(merged_ds.y) > 0 and merged_ds.y[0] > merged_ds.y[-1]:
        merged_ds = merged_ds.sel(y=slice(req_max_y + epsilon, req_min_y - epsilon))
    else:
        merged_ds = merged_ds.sel(y=slice(req_min_y - epsilon, req_max_y + epsilon))

    # Add combined attributes (provenance was collected before the mosaic freed
    # the cell list).
    merged_ds.attrs = {
        "policy_hash": hash_policy(policy.model_dump()),  # type: ignore
        "policy_legend": str(legend),
        "crs": target_crs,
        "mosaiced_from_cells": len(cells),
        "provenance_dict": global_provenance,
    }
    # Final mosaic diagnostics. Guard the percentage: an empty merged grid
    # (size 0) is a legitimate outcome for some cell sets and must not raise.
    m_elev = merged_ds["elevation"]
    m_nans = int(np.isnan(m_elev).sum())
    m_total = m_elev.size
    nan_pct = (100.0 * m_nans / m_total) if m_total else 0.0
    logger.info(f"Final mosaic: shape={m_elev.shape}, NaNs={m_nans}/{m_total} ({nan_pct:.1f}%)")
    return cast(xr.Dataset, merged_ds)
