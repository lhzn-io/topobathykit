"""DEM Viewer API — serves fused zarr cells as mosaiced Float32 binaries for 3D visualization."""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import xarray as xr
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from topobathyserve.models import DEMQualityReport
from topobathyserve.policy_catalog import NO_POLICY_SUMMARY, policy_summary, saved_policies

from topobathykit.config import get_cache_root
from topobathykit.quality import calculate_spatial_stats
from topobathykit.runtime import fill_short_gaps

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/dem", tags=["DEM Viewer"])


# Tolerance for resolution matching: 5% handles latitude-dependent pixel size
# variation in projected CRS (e.g. EPSG:3857 L13 tiles: dx ≈ 14.44-14.46m).
_RES_TOLERANCE = 0.05


def _fused_zarr_root() -> Path:
    return get_cache_root() / "fused_zarr"


def _read_cell_attrs(cell_path: Path) -> dict[str, Any] | None:
    """Read attrs + coordinate extents from a zarr cell without loading data."""
    try:
        ds = xr.open_dataset(cell_path, engine="zarr", decode_coords="all")
        attrs = dict(ds.attrs)
        # Grab coordinate spacing and extents in native CRS
        x_vals = ds.x.values if "x" in ds.coords else None
        y_vals = ds.y.values if "y" in ds.coords else None
        if x_vals is not None and len(x_vals) > 1:
            attrs["_dx"] = abs(float(x_vals[1] - x_vals[0]))
            attrs["_x_min"] = float(min(x_vals))
            attrs["_x_max"] = float(max(x_vals))
            attrs["_nx"] = len(x_vals)
        if y_vals is not None and len(y_vals) > 1:
            attrs["_dy"] = abs(float(y_vals[1] - y_vals[0]))
            attrs["_y_min"] = float(min(y_vals))
            attrs["_y_max"] = float(max(y_vals))
            attrs["_ny"] = len(y_vals)
        ds.close()
        return attrs
    except Exception:
        return None


def _cell_mtime(cell_path: Path) -> float:
    """Return the latest modification time of a zarr cell directory.

    A cell rewritten in place by hydrate is moved into position as a whole
    directory, so its own mtime and that of its metadata files reflect the write.
    """
    mtime = 0.0
    for p in (cell_path, cell_path / ".zmetadata", cell_path / ".zattrs", cell_path / "zarr.json"):
        try:
            mtime = max(mtime, p.stat().st_mtime)
        except OSError:
            continue
    return mtime


def _cells_newer_than(cell_mtimes: list[float], cache_file: Path) -> bool:
    """True when any cell was written after `cache_file`, so the mosaic sidecar is stale."""
    try:
        built = cache_file.stat().st_mtime
    except OSError:
        return True
    return any(m > built for m in cell_mtimes)


def _parse_cell_bbox(attrs: dict[str, Any]) -> list[float] | None:
    cb = attrs.get("cell_bbox")
    if cb is None:
        return None
    if isinstance(cb, str):
        cb = json.loads(cb)
    return list(cb)


def _cluster_cells(
    group: list[tuple[Path, dict[str, Any]]],
    threshold: float,
) -> list[list[tuple[Path, dict[str, Any]]]]:
    """Cluster cells by spatial proximity using single-linkage on bbox centroids.

    Two cells are neighbors if their centroid distance (Chebyshev / L-inf) is
    within `threshold`.  Returns a list of clusters, each a list of (path, attrs).
    """
    if len(group) <= 1:
        return [group]

    # Compute centroids from cell_bbox (geographic degrees)
    centroids: list[tuple[float, float]] = []
    for _, attrs in group:
        bb = _parse_cell_bbox(attrs)
        if bb:
            centroids.append(((bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0))
        else:
            centroids.append((0.0, 0.0))

    # Union-find for single-linkage clustering
    parent = list(range(len(group)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        parent[find(i)] = find(j)

    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            dx = abs(centroids[i][0] - centroids[j][0])
            dy = abs(centroids[i][1] - centroids[j][1])
            if max(dx, dy) <= threshold:
                union(i, j)

    # Collect clusters
    clusters_map: dict[int, list[int]] = {}
    for i in range(len(group)):
        root = find(i)
        clusters_map.setdefault(root, []).append(i)

    return [[group[i] for i in indices] for indices in clusters_map.values()]


# Hard cap on the mosaic build canvas.  Prevents OOM for large regions at fine
# resolution (e.g. GBE at 3m yields a ~20kx25k = 2 GB canvas unbuildable in RAM).
# The cached .bin is always at most this many pixels on its longest side.
MOSAIC_BUILD_MAX_DIM = 8192

# Recorded in each mosaic sidecar; a sidecar written by another version is rebuilt.
# 2: short gaps filled as in /fuse (#40).
MOSAIC_VERSION = 2


def _downsample_nearest(arr: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """Nearest-neighbour downsample a 2-D array (works for float32 and uint32, preserves NaN)."""
    row_idx = np.round(np.linspace(0, arr.shape[0] - 1, new_h)).astype(np.intp)
    col_idx = np.round(np.linspace(0, arr.shape[1] - 1, new_w)).astype(np.intp)
    return arr[np.ix_(row_idx, col_idx)]


def _clamp_dims(width: int, height: int, max_dim: int | None) -> tuple[int, int, float]:
    """Return (new_w, new_h, scale) clamped so max(new_w, new_h) <= max_dim."""
    if max_dim is None or max(width, height) <= max_dim:
        return width, height, 1.0
    scale = max_dim / max(width, height)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale))), scale


MaxDim = Annotated[int | None, Query(ge=64, le=16384, description="Cap grid dimensions for LoD rendering")]
BBox = Annotated[
    str | None,
    Query(
        description=(
            "Crop to west,south,east,north in the units of the dataset bounds (degrees for "
            "geographic datasets) before max_dim is applied"
        ),
        examples=["-70.9,41.2,-70.4,41.6"],
    ),
]


def _parse_bbox(bbox: str | None) -> tuple[float, float, float, float] | None:
    """Parse a ``west,south,east,north`` query value; 422 when malformed."""
    if bbox is None:
        return None
    try:
        parts = [float(p) for p in bbox.split(",")]
    except ValueError:
        parts = []
    if len(parts) != 4 or not all(np.isfinite(parts)):
        raise HTTPException(status_code=422, detail="bbox must be four numbers: west,south,east,north")
    west, south, east, north = parts
    if west >= east or south >= north:
        raise HTTPException(status_code=422, detail="bbox needs west < east and south < north")
    return west, south, east, north


def _crop_window(
    bounds: list[float], width: int, height: int, bbox: tuple[float, float, float, float]
) -> tuple[int, int, int, int, list[float]]:
    """Return the canvas window ``(row0, row1, col0, col1)`` (half-open) covering *bbox*.

    The canvas spans *bounds* with its first and last pixels on the bounds' edges, so
    pixel ``i`` sits at ``west + i * (east - west) / (width - 1)``.  The window is the
    smallest one whose pixels enclose the part of *bbox* inside the dataset, and the
    returned bounds are those of its edge pixels.  A bbox that misses the dataset is a 400.
    """
    west, south, east, north = bounds
    bw, bs, be, bn = bbox
    if be < west or bw > east or bn < south or bs > north:
        raise HTTPException(
            status_code=400,
            detail=f"bbox {list(bbox)} does not intersect the dataset bounds {bounds}",
        )
    px = (east - west) / (width - 1) if width > 1 else 0.0
    py = (north - south) / (height - 1) if height > 1 else 0.0

    def span(lo: float, hi: float, step: float, n: int) -> tuple[int, int]:
        """Half-open pixel range enclosing offsets [lo, hi] from the first pixel."""
        if step <= 0:
            return 0, n
        i0 = int(np.floor(lo / step + 1e-9))
        i1 = int(np.ceil(hi / step - 1e-9))
        return max(0, i0), min(n, i1 + 1)

    col0, col1 = span(max(bw, west) - west, min(be, east) - west, px, width)
    # Rows count down from the north edge.
    row0, row1 = span(north - min(bn, north), north - max(bs, south), py, height)
    crop_bounds = [
        west + col0 * px,
        north - (row1 - 1) * py,
        west + (col1 - 1) * px,
        north - row0 * py,
    ]
    return row0, row1, col0, col1, crop_bounds


def _window_grid(
    meta: dict[str, Any], max_dim: int | None, bbox: tuple[float, float, float, float] | None
) -> dict[str, Any]:
    """Apply an optional bbox crop, then the max_dim LoD cap, to a full-dataset meta.

    *meta* describes the build-capped canvas (``full_width`` x ``full_height`` over
    ``bounds`` at ``full_dx`` x ``full_dy``).  The result holds the output ``width``,
    ``height``, ``bounds``, ``dx``, ``dy`` and LoD ``scale``, plus the canvas ``window``
    ``(row0, row1, col0, col1)`` that the binaries slice before downsampling.
    """
    full_w, full_h = meta["full_width"], meta["full_height"]
    bounds = list(meta["bounds"])
    full_dx, full_dy = meta["full_dx"], meta["full_dy"]
    window = (0, full_h, 0, full_w)
    if bbox is not None:
        row0, row1, col0, col1, bounds = _crop_window(bounds, full_w, full_h, bbox)
        window = (row0, row1, col0, col1)
    crop_w, crop_h = window[3] - window[2], window[1] - window[0]
    out_w, out_h, lod_scale = _clamp_dims(crop_w, crop_h, max_dim)
    return {
        "width": out_w,
        "height": out_h,
        "scale": lod_scale,
        "bounds": bounds,
        "dx": full_dx / lod_scale if lod_scale > 0 else full_dx,
        "dy": full_dy / lod_scale if lod_scale > 0 else full_dy,
        "window": window,
    }


def _slice_and_downsample(content: bytes, dtype: type, meta: dict[str, Any], grid: dict[str, Any]) -> bytes:
    """Cut *grid*'s window out of the full canvas bytes and downsample it to *grid*'s size."""
    full_w, full_h = meta["full_width"], meta["full_height"]
    row0, row1, col0, col1 = grid["window"]
    out_h, out_w = grid["height"], grid["width"]
    if (row0, row1, col0, col1, out_h, out_w) == (0, full_h, 0, full_w, full_h, full_w):
        return content
    canvas = np.frombuffer(content, dtype=dtype).reshape(full_h, full_w)[row0:row1, col0:col1]
    if canvas.shape != (out_h, out_w):
        canvas = _downsample_nearest(canvas, out_h, out_w)
    return np.ascontiguousarray(canvas).tobytes()


@router.get("/datasets")
async def list_datasets(source: str | None = None) -> list[dict[str, Any]]:
    """List fused datasets, grouped by policy hash, resolution, and spatial cluster.

    Args:
        source: Comma-separated source filter, e.g. "hydrate,fuse". If omitted, returns all.
    """
    root = _fused_zarr_root()
    sources = set(source.split(",")) if source else None
    return _list_datasets_sync(root, source_filter=sources)


def _mosaic_to_bin(
    policy_dir: Path,
    dx: float,
    dy: float,
    width: int,
    height: int,
    x_min: float,
    y_max: float,
    cluster_bbox: list[float] | None = None,
    build_scale: float = 1.0,
) -> np.ndarray:
    """Paint zarr cells into a pre-allocated numpy canvas (north-first, row-major).

    When *build_scale* < 1 the canvas is smaller than the natural full-resolution
    grid.  Each cell is nearest-neighbour downsampled to match the coarser canvas
    pixel size before blitting, and blit offsets are computed using the scaled dx/dy.
    This avoids ever allocating the full-resolution canvas (which can be >2 GB for
    large fine-resolution regions).
    """
    canvas = np.full((height, width), np.nan, dtype=np.float32)
    cells_painted = 0
    # Canvas pixel size (coarser than native when build_scale < 1)
    cdx = dx / build_scale if build_scale > 0 else dx
    cdy = dy / build_scale if build_scale > 0 else dy

    for cell_path in sorted(policy_dir.glob("*.zarr")):
        try:
            # Quick cluster filter before loading data
            if cluster_bbox is not None:
                attrs = _read_cell_attrs(cell_path)
                if attrs and not _cell_in_cluster(attrs, cluster_bbox):
                    continue

            ds = xr.open_dataset(cell_path, engine="zarr", decode_coords="all")
            x_vals = ds.x.values if "x" in ds.coords else None
            if x_vals is None or len(x_vals) < 2:
                ds.close()
                continue
            cell_dx = abs(float(x_vals[1] - x_vals[0]))
            if abs(cell_dx - dx) / max(dx, 1e-10) > _RES_TOLERANCE:
                ds.close()
                continue

            y_vals = ds.y.values
            elev = ds["elevation"].values.astype(np.float32)
            ds.close()

            # Replace nodata sentinels with NaN.  Zarr cells may store the
            # float32 _FillValue (~-3.4e38) or PDAL nodata (-9999) as real
            # float values instead of NaN, which would collapse the color scale.
            elev[(elev < -20000.0) | (elev > 20000.0)] = np.nan

            # Ensure elevation is north-first (descending y)
            if len(y_vals) > 1 and y_vals[0] < y_vals[-1]:
                elev = elev[::-1]

            # Derive canvas pixel slot from coordinates first — these are the
            # authoritative positions this cell must fill.  Computing col_start
            # and col_end from the same formula guarantees adjacent cells share
            # exactly one boundary pixel (no 1-column NaN gap caused by rounding
            # round(native_w * scale) vs round(offset / cdx) disagreeing by 1).
            col_start = int(round((float(min(x_vals)) - x_min) / cdx))
            col_end_c = int(round((float(max(x_vals)) - x_min) / cdx))
            row_start = int(round((y_max - float(max(y_vals))) / cdy))
            row_end_c = int(round((y_max - float(min(y_vals))) / cdy))

            # Downsample cell to match its coordinate-derived canvas slot.
            if build_scale < 0.999:
                new_w = max(1, col_end_c - col_start + 1)
                new_h = max(1, row_end_c - row_start + 1)
                elev = _downsample_nearest(elev, new_h, new_w)

            # Clip to canvas bounds (handles halo overlap)
            src_r0 = max(0, -row_start)
            src_c0 = max(0, -col_start)
            dst_r0 = max(0, row_start)
            dst_c0 = max(0, col_start)
            copy_h = min(elev.shape[0] - src_r0, height - dst_r0)
            copy_w = min(elev.shape[1] - src_c0, width - dst_c0)

            if copy_h <= 0 or copy_w <= 0:
                continue

            # Blit: only overwrite NaN pixels (first-valid-wins, like combine_first)
            src_slice = elev[src_r0 : src_r0 + copy_h, src_c0 : src_c0 + copy_w]
            dst_slice = canvas[dst_r0 : dst_r0 + copy_h, dst_c0 : dst_c0 + copy_w]
            mask = np.isnan(dst_slice) & ~np.isnan(src_slice)
            dst_slice[mask] = src_slice[mask]
            cells_painted += 1

        except Exception as e:
            logger.warning(f"Failed to paint cell {cell_path.name}: {e}")

    # Fill the short gaps /fuse fills, so the two agree on which pixels are NaN. Under
    # the build cap the two-pixel limit applies to canvas pixels.
    grid = xr.DataArray(
        canvas,
        dims=("y", "x"),
        coords={"y": np.arange(height, 0, -1, dtype=np.float64), "x": np.arange(width, dtype=np.float64)},
    )
    canvas = fill_short_gaps(grid).values.astype(np.float32, copy=False)

    logger.info(
        f"Mosaic complete: {cells_painted} cells -> {width}x{height} canvas (scale={build_scale:.3f})"
    )
    return canvas


def _mosaic_source_to_bin(
    policy_dir: Path,
    dx: float,
    dy: float,
    width: int,
    height: int,
    x_min: float,
    y_max: float,
    cluster_bbox: list[float] | None = None,
    build_scale: float = 1.0,
) -> np.ndarray:
    """Paint source_elevation IDs into a pre-allocated uint32 canvas (north-first).

    Mirrors _mosaic_to_bin but reads source_elevation instead of elevation.
    Each pixel stores the integer ID of the data provider that contributed it.
    Supports *build_scale* for the same OOM-safe downsampled build path.
    """
    canvas = np.zeros((height, width), dtype=np.uint32)
    cells_painted = 0
    cdx = dx / build_scale if build_scale > 0 else dx
    cdy = dy / build_scale if build_scale > 0 else dy

    for cell_path in sorted(policy_dir.glob("*.zarr")):
        try:
            if cluster_bbox is not None:
                attrs = _read_cell_attrs(cell_path)
                if attrs and not _cell_in_cluster(attrs, cluster_bbox):
                    continue

            ds = xr.open_dataset(cell_path, engine="zarr", decode_coords="all")
            x_vals = ds.x.values if "x" in ds.coords else None
            if x_vals is None or len(x_vals) < 2:
                ds.close()
                continue
            cell_dx = abs(float(x_vals[1] - x_vals[0]))
            if abs(cell_dx - dx) / max(dx, 1e-10) > _RES_TOLERANCE:
                ds.close()
                continue

            if "source_elevation" not in ds:
                ds.close()
                continue

            y_vals = ds.y.values
            src = ds["source_elevation"].values.astype(np.uint32)
            ds.close()

            if len(y_vals) > 1 and y_vals[0] < y_vals[-1]:
                src = src[::-1]

            col_start = int(round((float(min(x_vals)) - x_min) / cdx))
            col_end_c = int(round((float(max(x_vals)) - x_min) / cdx))
            row_start = int(round((y_max - float(max(y_vals))) / cdy))
            row_end_c = int(round((y_max - float(min(y_vals))) / cdy))

            if build_scale < 0.999:
                new_w = max(1, col_end_c - col_start + 1)
                new_h = max(1, row_end_c - row_start + 1)
                src = _downsample_nearest(src, new_h, new_w)

            src_r0 = max(0, -row_start)
            src_c0 = max(0, -col_start)
            dst_r0 = max(0, row_start)
            dst_c0 = max(0, col_start)
            copy_h = min(src.shape[0] - src_r0, height - dst_r0)
            copy_w = min(src.shape[1] - src_c0, width - dst_c0)

            if copy_h <= 0 or copy_w <= 0:
                continue

            src_slice = src[src_r0 : src_r0 + copy_h, src_c0 : src_c0 + copy_w]
            dst_slice = canvas[dst_r0 : dst_r0 + copy_h, dst_c0 : dst_c0 + copy_w]
            mask = (dst_slice == 0) & (src_slice > 0)
            dst_slice[mask] = src_slice[mask]
            cells_painted += 1

        except Exception as e:
            logger.warning(f"Failed to paint provenance cell {cell_path.name}: {e}")

    logger.info(
        f"Provenance mosaic complete: {cells_painted} cells -> {width}x{height} (scale={build_scale:.3f})"
    )
    return canvas


def _cell_in_cluster(attrs: dict[str, Any], cluster_bbox: list[float] | None) -> bool:
    """Check if a cell's geographic bbox overlaps the cluster bbox."""
    if cluster_bbox is None:
        return True
    bb = _parse_cell_bbox(attrs)
    if bb is None:
        return True  # can't filter, include it
    # Simple overlap test (bboxes are [west, south, east, north])
    return not (
        bb[2] < cluster_bbox[0]
        or bb[0] > cluster_bbox[2]
        or bb[3] < cluster_bbox[1]
        or bb[1] > cluster_bbox[3]
    )


@router.get("/datasets/{dataset_id}/meta")
async def get_dataset_meta(
    dataset_id: str,
    max_dim: MaxDim = None,
    bbox: BBox = None,
) -> dict[str, Any]:
    """Return metadata for a fused dataset.

    When ``bbox`` is supplied the grid is first cropped to the pixels enclosing it, and
    ``bounds`` reports the cropped extent.  When ``max_dim`` is supplied the returned
    ``width``, ``height``, ``dx``, and ``dy`` reflect the *downsampled* grid (of the crop,
    if any) that ``elevation.bin`` and ``provenance.bin`` will serve for the same
    ``bbox`` and ``max_dim``.  Pass the same values to all three endpoints so
    buffer-length checks remain consistent.  ``full_width`` and ``full_height`` always
    describe the whole build-capped canvas.  Elevations are in ``units`` (metres).
    """
    meta = _full_dataset_meta(dataset_id)
    grid = _window_grid(meta, max_dim, _parse_bbox(bbox))
    meta.update({k: grid[k] for k in ("width", "height", "scale", "bounds", "dx", "dy")})
    del meta["full_dx"], meta["full_dy"]
    return meta


def _full_dataset_meta(dataset_id: str) -> dict[str, Any]:
    """Metadata for the whole build-capped canvas, before any bbox crop or LoD cap."""
    root = _fused_zarr_root()
    policy_prefix, dx, dy, cluster_bbox = _parse_dataset_id(root, dataset_id)
    policy_dir = _resolve_policy_dir(root, policy_prefix)

    cells = sorted(policy_dir.glob("*.zarr"))
    cell_count = 0
    x_min_all = float("inf")
    x_max_all = float("-inf")
    y_min_all = float("inf")
    y_max_all = float("-inf")
    geo_bboxes: list[list[float]] = []
    crs = "EPSG:4326"
    legend = ""
    provenance_dict: dict[str, Any] = {}

    for cell_path in cells:
        attrs = _read_cell_attrs(cell_path)
        if attrs is None:
            continue
        cell_dx = attrs.get("_dx", 0)
        if abs(cell_dx - dx) / max(dx, 1e-10) > _RES_TOLERANCE:
            continue
        if not _cell_in_cluster(attrs, cluster_bbox):
            continue
        cell_count += 1
        x_min_all = min(x_min_all, attrs.get("_x_min", x_min_all))
        x_max_all = max(x_max_all, attrs.get("_x_max", x_max_all))
        y_min_all = min(y_min_all, attrs.get("_y_min", y_min_all))
        y_max_all = max(y_max_all, attrs.get("_y_max", y_max_all))
        bb = _parse_cell_bbox(attrs)
        if bb:
            geo_bboxes.append(bb)
        crs = attrs.get("crs", crs)
        legend = attrs.get("policy_legend", legend)
        cell_prov = attrs.get("provenance_dict", {})
        if isinstance(cell_prov, dict):
            provenance_dict.update(cell_prov)

    if cell_count == 0:
        raise HTTPException(status_code=404, detail="No cells at this resolution")

    width = max(1, int(round((x_max_all - x_min_all) / dx)) + 1)
    height = max(1, int(round((y_max_all - y_min_all) / dy)) + 1)

    if geo_bboxes:
        bounds = [
            min(b[0] for b in geo_bboxes),
            min(b[1] for b in geo_bboxes),
            max(b[2] for b in geo_bboxes),
            max(b[3] for b in geo_bboxes),
        ]
    else:
        bounds = [x_min_all, y_min_all, x_max_all, y_max_all]

    if bounds and width > 1 and height > 1:
        geo_dx = (bounds[2] - bounds[0]) / (width - 1)
        geo_dy = (bounds[3] - bounds[1]) / (height - 1)
    else:
        geo_dx = dx
        geo_dy = dy

    # Inject tier-level fallback entries from policy_legend into provenance_dict so that
    # pixels written with provider_id (tier number) don't show as "undefined".
    # policy_legend is stored as a Python repr string e.g. "{1: 'ncei_bag', 2: 'ncei_cudem'}"
    if legend:
        try:
            import ast

            tier_map = ast.literal_eval(legend) if isinstance(legend, str) else legend
            if isinstance(tier_map, dict):
                for tier_id, provider_name in tier_map.items():
                    key = str(tier_id)
                    if key not in provenance_dict:
                        provenance_dict[key] = {"name": provider_name, "provider": provider_name}
        except Exception:
            pass

    # Apply hard build cap first (matches what the cached .bin is built at)
    build_w, build_h, _build_scale = _clamp_dims(width, height, MOSAIC_BUILD_MAX_DIM)
    build_dx = geo_dx / _build_scale if _build_scale > 0 else geo_dx
    build_dy = geo_dy / _build_scale if _build_scale > 0 else geo_dy

    return {
        "width": build_w,
        "height": build_h,
        "full_width": build_w,
        "full_height": build_h,
        "scale": 1.0,
        "bounds": bounds,
        "dx": build_dx,
        "dy": build_dy,
        "full_dx": build_dx,
        "full_dy": build_dy,
        "units": "m",
        "crs": crs,
        "cell_count": cell_count,
        "policy_legend": legend,
        "provenance_dict": provenance_dict,
    }


@router.get("/datasets/{dataset_id}/elevation.bin")
async def get_elevation_binary(
    dataset_id: str,
    max_dim: MaxDim = None,
    bbox: BBox = None,
) -> Response:
    """Return the mosaiced elevation grid as a raw Float32 binary (north-first).

    On first request, the mosaic is built by painting cells into a numpy canvas
    (no xr.combine_by_coords) and cached to disk as a .bin file.  Subsequent
    requests serve the cached file directly.

    Pass ``max_dim`` to receive a nearest-neighbour downsampled grid whose longest
    dimension is at most ``max_dim`` pixels.  The full-resolution ``.bin`` is always
    cached on disk; downsampling is applied on the fly before sending the response.

    Pass ``bbox`` (``west,south,east,north``) to receive only the pixels enclosing it;
    the crop is cut from the cached canvas before ``max_dim`` is applied, so the result
    matches ``/meta`` for the same ``bbox`` and ``max_dim``.
    """
    # Resolve the output grid first, so a bad bbox is rejected before any mosaic build.
    full_meta = _full_dataset_meta(dataset_id)
    grid = _window_grid(full_meta, max_dim, _parse_bbox(bbox))

    root = _fused_zarr_root()
    policy_prefix, dx, dy, cluster_bbox = _parse_dataset_id(root, dataset_id)
    policy_dir = _resolve_policy_dir(root, policy_prefix)

    # Check for cached binary (invalidate if cell count or canvas changed, or if any
    # cell was rewritten after the binary was built)
    cache_bin = policy_dir / f"_mosaic_{dataset_id}.bin"
    cache_meta_file = policy_dir / f"_mosaic_{dataset_id}.json"

    cache_valid = False
    if cache_bin.exists() and cache_meta_file.exists():
        try:
            cached = json.loads(cache_meta_file.read_text())
            current_cells = 0
            cell_mtimes: list[float] = []
            x_min_check = float("inf")
            x_max_check = float("-inf")
            y_min_check = float("inf")
            y_max_check = float("-inf")
            for cp in policy_dir.glob("*.zarr"):
                a = _read_cell_attrs(cp)
                if not a:
                    continue
                if abs(a.get("_dx", 0) - dx) / max(dx, 1e-10) > _RES_TOLERANCE:
                    continue
                if not _cell_in_cluster(a, cluster_bbox):
                    continue
                current_cells += 1
                cell_mtimes.append(_cell_mtime(cp))
                x_min_check = min(x_min_check, a.get("_x_min", x_min_check))
                x_max_check = max(x_max_check, a.get("_x_max", x_max_check))
                y_min_check = min(y_min_check, a.get("_y_min", y_min_check))
                y_max_check = max(y_max_check, a.get("_y_max", y_max_check))
            if current_cells > 0:
                natural_w = max(1, int(round((x_max_check - x_min_check) / dx)) + 1)
                natural_h = max(1, int(round((y_max_check - y_min_check) / dy)) + 1)
                expected_w, expected_h, _ = _clamp_dims(natural_w, natural_h, MOSAIC_BUILD_MAX_DIM)
            else:
                expected_w = expected_h = 0
            cells_rewritten = _cells_newer_than(cell_mtimes, cache_bin)
            if (
                cached.get("mosaic_version") == MOSAIC_VERSION
                and cached.get("cell_count") == current_cells
                and cached.get("width") == expected_w
                and cached.get("height") == expected_h
                and not cells_rewritten
            ):
                cache_valid = True
            else:
                logger.info(
                    f"DEM mosaic cache stale: {cached.get('cell_count')} -> {current_cells} cells, "
                    f"dims {cached.get('width')}x{cached.get('height')} -> "
                    f"{expected_w}x{expected_h}, cells rewritten since build: {cells_rewritten}, "
                    "rebuilding"
                )
        except Exception:
            pass

    if cache_valid:
        logger.info(f"DEM mosaic cache hit: {cache_bin.name}")
        content = cache_bin.read_bytes()
    else:
        meta = full_meta

        # Scan cells to get canvas extents and natural (uncapped) pixel count
        x_min = float("inf")
        y_max_val = float("-inf")
        x_max_scan = float("-inf")
        y_min_scan = float("inf")
        for cell_path in sorted(policy_dir.glob("*.zarr")):
            attrs = _read_cell_attrs(cell_path)
            if attrs is None:
                continue
            cell_dx = attrs.get("_dx", 0)
            if abs(cell_dx - dx) / max(dx, 1e-10) > _RES_TOLERANCE:
                continue
            if not _cell_in_cluster(attrs, cluster_bbox):
                continue
            x_min = min(x_min, attrs.get("_x_min", x_min))
            y_max_val = max(y_max_val, attrs.get("_y_max", y_max_val))
            x_max_scan = max(x_max_scan, attrs.get("_x_max", x_max_scan))
            y_min_scan = min(y_min_scan, attrs.get("_y_min", y_min_scan))

        # Derive natural (full-resolution) canvas dims, then cap at MOSAIC_BUILD_MAX_DIM
        natural_w = max(1, int(round((x_max_scan - x_min) / dx)) + 1) if x_max_scan > x_min else 1
        natural_h = max(1, int(round((y_max_val - y_min_scan) / dy)) + 1) if y_max_val > y_min_scan else 1
        build_w, build_h, build_scale = _clamp_dims(natural_w, natural_h, MOSAIC_BUILD_MAX_DIM)

        logger.info(
            f"Building mosaic: natural {natural_w}x{natural_h} -> build {build_w}x{build_h} "
            f"(scale={build_scale:.3f})"
        )

        canvas = _mosaic_to_bin(
            policy_dir, dx, dy, build_w, build_h, x_min, y_max_val, cluster_bbox, build_scale
        )
        content = canvas.tobytes()

        try:
            cache_bin.write_bytes(content)
            cache_meta_file.write_text(
                json.dumps(
                    {
                        "mosaic_version": MOSAIC_VERSION,
                        "width": build_w,
                        "height": build_h,
                        "cell_count": meta["cell_count"],
                    }
                )
            )
            logger.info(f"DEM mosaic cached: {cache_bin.name} ({len(content)} bytes)")
            # Invalidate provenance bin — it shares the same meta file for cache
            # validation, so after writing a new cell_count it would falsely appear
            # valid even though it was built from the old (smaller) mosaic.
            prov_bin = policy_dir / f"_mosaic_{dataset_id}_provenance.bin"
            if prov_bin.exists():
                prov_bin.unlink()
                logger.info(f"Provenance mosaic cache invalidated: {prov_bin.name}")
        except Exception as e:
            logger.warning(f"Failed to cache mosaic: {e}")

    # --- bbox crop, then LoD downsampling, on top of the build-capped binary ---
    out_w, out_h = grid["width"], grid["height"]
    content = _slice_and_downsample(content, np.float32, full_meta, grid)
    if bbox is not None or max_dim is not None:
        logger.info(f"Elevation window {grid['window']} -> {out_w}x{out_h} (bbox={bbox}, max_dim={max_dim})")

    return Response(
        content=content,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f"attachment; filename={dataset_id}_elevation.bin",
            "X-Grid-Width": str(out_w),
            "X-Grid-Height": str(out_h),
        },
    )


@router.get("/datasets/{dataset_id}/provenance.bin")
async def get_provenance_binary(
    dataset_id: str,
    max_dim: MaxDim = None,
    bbox: BBox = None,
) -> Response:
    """Return the mosaiced source_elevation grid as a raw Uint32 binary (north-first).

    Each pixel stores a source ID integer that maps to a provider name via the
    policy_legend in the dataset metadata. 0 = no data.

    Pass ``max_dim`` to receive a nearest-neighbour downsampled grid (categorical-safe),
    and ``bbox`` to crop first, as for ``elevation.bin``.
    """
    full_meta = _full_dataset_meta(dataset_id)
    grid = _window_grid(full_meta, max_dim, _parse_bbox(bbox))

    root = _fused_zarr_root()
    policy_prefix, dx, dy, cluster_bbox = _parse_dataset_id(root, dataset_id)
    policy_dir = _resolve_policy_dir(root, policy_prefix)

    cache_bin = policy_dir / f"_mosaic_{dataset_id}_provenance.bin"
    cache_meta_file = policy_dir / f"_mosaic_{dataset_id}.json"

    cache_valid = False
    if cache_bin.exists() and cache_meta_file.exists():
        try:
            cached = json.loads(cache_meta_file.read_text())
            current_cells = 0
            cell_mtimes: list[float] = []
            x_min_check = float("inf")
            x_max_check = float("-inf")
            y_min_check = float("inf")
            y_max_check = float("-inf")
            for cp in policy_dir.glob("*.zarr"):
                a = _read_cell_attrs(cp)
                if not a:
                    continue
                if abs(a.get("_dx", 0) - dx) / max(dx, 1e-10) > _RES_TOLERANCE:
                    continue
                if not _cell_in_cluster(a, cluster_bbox):
                    continue
                current_cells += 1
                cell_mtimes.append(_cell_mtime(cp))
                x_min_check = min(x_min_check, a.get("_x_min", x_min_check))
                x_max_check = max(x_max_check, a.get("_x_max", x_max_check))
                y_min_check = min(y_min_check, a.get("_y_min", y_min_check))
                y_max_check = max(y_max_check, a.get("_y_max", y_max_check))
            if current_cells > 0:
                natural_w = max(1, int(round((x_max_check - x_min_check) / dx)) + 1)
                natural_h = max(1, int(round((y_max_check - y_min_check) / dy)) + 1)
                expected_w, expected_h, _ = _clamp_dims(natural_w, natural_h, MOSAIC_BUILD_MAX_DIM)
            else:
                expected_w = expected_h = 0
            if (
                cached.get("cell_count") == current_cells
                and cached.get("width") == expected_w
                and cached.get("height") == expected_h
                and not _cells_newer_than(cell_mtimes, cache_bin)
            ):
                cache_valid = True
        except Exception:
            pass

    if cache_valid:
        logger.info(f"Provenance mosaic cache hit: {cache_bin.name}")
        content = cache_bin.read_bytes()
    else:
        x_min = float("inf")
        y_max_val = float("-inf")
        x_max_scan = float("-inf")
        y_min_scan = float("inf")
        for cell_path in sorted(policy_dir.glob("*.zarr")):
            attrs = _read_cell_attrs(cell_path)
            if attrs is None:
                continue
            cell_dx = attrs.get("_dx", 0)
            if abs(cell_dx - dx) / max(dx, 1e-10) > _RES_TOLERANCE:
                continue
            if not _cell_in_cluster(attrs, cluster_bbox):
                continue
            x_min = min(x_min, attrs.get("_x_min", x_min))
            y_max_val = max(y_max_val, attrs.get("_y_max", y_max_val))
            x_max_scan = max(x_max_scan, attrs.get("_x_max", x_max_scan))
            y_min_scan = min(y_min_scan, attrs.get("_y_min", y_min_scan))

        natural_w = max(1, int(round((x_max_scan - x_min) / dx)) + 1) if x_max_scan > x_min else 1
        natural_h = max(1, int(round((y_max_val - y_min_scan) / dy)) + 1) if y_max_val > y_min_scan else 1
        build_w, build_h, build_scale = _clamp_dims(natural_w, natural_h, MOSAIC_BUILD_MAX_DIM)

        canvas = _mosaic_source_to_bin(
            policy_dir, dx, dy, build_w, build_h, x_min, y_max_val, cluster_bbox, build_scale
        )
        content = canvas.tobytes()

        try:
            cache_bin.write_bytes(content)
            logger.info(f"Provenance mosaic cached: {cache_bin.name} ({len(content)} bytes)")
        except Exception as e:
            logger.warning(f"Failed to cache provenance mosaic: {e}")

    # --- bbox crop, then LoD downsampling ---
    out_w, out_h = grid["width"], grid["height"]
    content = _slice_and_downsample(content, np.uint32, full_meta, grid)
    if bbox is not None or max_dim is not None:
        logger.info(f"Provenance window {grid['window']} -> {out_w}x{out_h} (bbox={bbox}, max_dim={max_dim})")

    return Response(
        content=content,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f"attachment; filename={dataset_id}_provenance.bin",
            "X-Grid-Width": str(out_w),
            "X-Grid-Height": str(out_h),
        },
    )


@router.get("/datasets/{dataset_id}/quality", response_model=DEMQualityReport)
async def get_dataset_quality(
    dataset_id: str,
    resolution: float = Query(30.0, description="Approximate nominal grid resolution in meters"),
) -> DEMQualityReport:
    """Return quality report computed over the cached full-mosaic DEM without executing any fusion."""
    try:
        # Re-use metadata and trigger mosaic building if missing
        meta = await get_dataset_meta(dataset_id, max_dim=None, bbox=None)
        elev_resp = await get_elevation_binary(dataset_id, max_dim=None, bbox=None)
        prov_resp = await get_provenance_binary(dataset_id, max_dim=None, bbox=None)

        full_h = meta["full_height"]
        full_w = meta["full_width"]

        elevation = np.frombuffer(elev_resp.body, dtype=np.float32).reshape(full_h, full_w)
        source_map = np.frombuffer(prov_resp.body, dtype=np.uint32).reshape(full_h, full_w)

        report_data = calculate_spatial_stats(
            elevation=elevation,
            source_map=source_map,
            resolution_m=resolution,
            bounds=tuple(meta["bounds"]),
            steep_threshold_m=3.0,
            pit_ratio_threshold=3.0,
        )

        return DEMQualityReport(**report_data)

    except Exception as e:
        logger.error(f"Dataset quality reporting failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/datasets/{dataset_id}/cell_at")
async def get_cell_at(
    dataset_id: str,
    lat: float = Query(..., description="Latitude (EPSG:4326)"),
    lon: float = Query(..., description="Longitude (EPSG:4326)"),
) -> dict[str, Any]:
    """Return the zarr cell filename whose cell_bbox contains the given lat/lon."""
    root = _fused_zarr_root()
    policy_prefix, dx, _dy, cluster_bbox = _parse_dataset_id(root, dataset_id)
    policy_dir = _resolve_policy_dir(root, policy_prefix)

    for cell_path in sorted(policy_dir.glob("*.zarr")):
        attrs = _read_cell_attrs(cell_path)
        if attrs is None:
            continue
        if abs(attrs.get("_dx", 0) - dx) / max(dx, 1e-10) > _RES_TOLERANCE:
            continue
        if not _cell_in_cluster(attrs, cluster_bbox):
            continue
        bb = _parse_cell_bbox(attrs)
        if bb is None:
            continue
        # bb = [west, south, east, north]
        if bb[0] <= lon <= bb[2] and bb[1] <= lat <= bb[3]:
            return {"cell": cell_path.name, "bbox": bb}

    raise HTTPException(status_code=404, detail="No cell covers this location")


@router.delete("/datasets/{dataset_id}/cache")
async def clear_dataset_cache(dataset_id: str) -> dict[str, Any]:
    """Delete the cached mosaic binaries so the next load rebuilds from zarr cells."""
    root = _fused_zarr_root()
    policy_prefix, _dx, _dy, _bbox = _parse_dataset_id(root, dataset_id)
    policy_dir = _resolve_policy_dir(root, policy_prefix)

    deleted = []
    for suffix in (".bin", ".json", "_provenance.bin"):
        f = policy_dir / f"_mosaic_{dataset_id}{suffix}"
        if f.exists():
            f.unlink()
            deleted.append(f.name)

    logger.info(f"DEM cache cleared for {dataset_id}: {deleted}")
    return {"status": "cleared", "deleted": deleted}


def _parse_dataset_id(root: Path, dataset_id: str) -> tuple[str, float, float, list[float] | None]:
    """Parse dataset_id like '313a6513_15m' into (policy_prefix, dx, dy, cluster_bbox).

    Uses list_datasets() to find the matching dataset with its cluster bbox.
    The bbox is used to filter cells to the correct spatial cluster.
    """
    datasets = _list_datasets_sync(root)

    for ds in datasets:
        if ds["id"] == dataset_id:
            policy_hash_full = ds.get("policy_hash_full", "")
            # Find the policy dir
            for d in root.iterdir():
                if d.is_dir() and d.name.startswith(ds["policy_hash"][:8]):
                    return d.name, ds["dx"], ds["dy"], ds.get("bbox")
            # Try full hash
            for d in root.iterdir():
                if d.is_dir() and policy_hash_full.startswith(d.name):
                    return d.name, ds["dx"], ds["dy"], ds.get("bbox")

    raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")


def _list_datasets_sync(root: Path, source_filter: str | set[str] | None = None) -> list[dict[str, Any]]:
    """Synchronous version of list_datasets for internal use."""
    if not root.exists():
        return []

    datasets: list[dict[str, Any]] = []
    # Fusion saves each policy beside the cache, named after its fused_zarr directory.
    policy_files = saved_policies(root.parent / "policies")
    for policy_dir in sorted(root.iterdir()):
        if not policy_dir.is_dir():
            continue
        cells = sorted(policy_dir.glob("*.zarr"))
        if not cells:
            continue

        policy_file = policy_files.get(policy_dir.name)
        policy_info = policy_summary(policy_file) if policy_file is not None else dict(NO_POLICY_SUMMARY)

        res_groups: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
        for cell_path in cells:
            attrs = _read_cell_attrs(cell_path)
            if attrs is None:
                continue
            # Filter by source type if specified.
            # Cells without a source attr predate the source tag — treat as hydrate.
            cell_source = attrs.get("source", "hydrate")
            if source_filter and cell_source not in (
                source_filter if isinstance(source_filter, set) else {source_filter}
            ):
                continue
            dx = attrs.get("_dx", 0)
            dy = attrs.get("_dy", 0)
            # For projected CRS (dx in meters), bucket by integer meter to
            # handle latitude-dependent pixel size variation within a tile tier.
            # E.g. L13 tiles at different latitudes have dx ≈ 14.44-14.46m.
            crs = attrs.get("crs", "EPSG:4326")
            if "4326" not in str(crs) and dx > 1:
                res_key = f"{round(dx)}m_{round(dy)}m"
            else:
                res_key = f"{dx:.8f}_{dy:.8f}"
            res_groups.setdefault(res_key, []).append((cell_path, attrs))

        for _res_key, group in res_groups.items():
            first_attrs = group[0][1]
            # Use directory name as the stable dataset identifier, NOT the per-cell
            # policy_hash attr. The directory is named after the policy content and
            # the fusion algorithm version, so the same policy fused by two algorithm
            # versions appears as two datasets.
            policy_hash = policy_dir.name
            dx = first_attrs.get("_dx", 0)
            dy = first_attrs.get("_dy", 0)

            grid_cell_size = first_attrs.get("grid_cell_size", 0.05)
            threshold = grid_cell_size * 1.5
            clusters = _cluster_cells(group, threshold)

            for cluster_idx, cluster in enumerate(clusters):
                cluster_bboxes = [b for b in (_parse_cell_bbox(a) for _, a in cluster) if b is not None]
                if not cluster_bboxes:
                    continue

                union_bbox = [
                    min(b[0] for b in cluster_bboxes),
                    min(b[1] for b in cluster_bboxes),
                    max(b[2] for b in cluster_bboxes),
                    max(b[3] for b in cluster_bboxes),
                ]

                crs = cluster[0][1].get("crs", "EPSG:4326")
                if "4326" in str(crs):
                    lat_center = (union_bbox[1] + union_bbox[3]) / 2.0
                    res_m = dx * 111320 * np.cos(np.radians(lat_center))
                else:
                    res_m = dx

                suffix = f"_{cluster_idx}" if len(clusters) > 1 else ""
                dataset_id = f"{policy_hash[:8]}_{res_m:.0f}m{suffix}"
                # Newest cell write (UTC), to tell re-hydrations of one policy apart.
                newest = max((_cell_mtime(p) for p, _ in cluster), default=0.0)
                updated = (
                    datetime.fromtimestamp(newest, UTC).isoformat(timespec="seconds") if newest else None
                )

                datasets.append(
                    {
                        "id": dataset_id,
                        **policy_info,
                        "updated": updated,
                        "policy_hash": policy_hash[:8],
                        "policy_hash_full": policy_hash,
                        "crs": crs,
                        "cell_count": len(cluster),
                        "resolution_m": round(res_m, 1),
                        "dx": dx,
                        "dy": dy,
                        "bbox": union_bbox,
                        "policy_legend": cluster[0][1].get("policy_legend", ""),
                        "source": cluster[0][1].get("source", "hydrate"),
                        # Cells written before the attribute existed are version 1.
                        "fusion_algo_version": int(cluster[0][1].get("fusion_algo_version", 1)),
                    }
                )

    return datasets


def _resolve_policy_dir(root: Path, prefix: str) -> Path:
    """Find a policy directory by prefix match."""
    exact = root / prefix
    if exact.is_dir():
        return exact

    matches = [d for d in root.iterdir() if d.is_dir() and d.name.startswith(prefix)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise HTTPException(
            status_code=400,
            detail=f"Ambiguous prefix '{prefix}' matches {len(matches)} datasets",
        )
    raise HTTPException(status_code=404, detail=f"Dataset '{prefix}' not found")
