import json
import logging
import math
import multiprocessing
import os
import sys
import threading
import time
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from datetime import datetime, timezone

# Set Matplotlib Backend to Agg (Non-Interactive) for Server Use
import matplotlib

matplotlib.use("Agg")

import contextlib
import shutil
import tempfile
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import xarray as xr
import yaml
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, RedirectResponse, Response
from starlette.background import BackgroundTask

from topobathysim.config import get_cache_root
from topobathysim.policy.loader import hash_policy, load_policy, load_policy_from_str
from topobathysim.quality import calculate_spatial_stats
from topobathysim.runtime import hydrate, run, should_consolidate

from . import job_state

# from topobathysim.quality import source_report # Removed as not directly supported in runtime yet
from .models import (
    DEMQualityReport,
    ElevationResponse,
    FusionRequest,
    HydrateRequest,
    PolicyUpdateRequest,
    TIDReportResponse,
    TileMetadataResponse,
)
from .routers import cache_viewer, dem_viewer

# Configure Logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Parse Debug Level from Env (set by run_server.py or environment)
debug_mode = int(os.environ.get("TOPOBATHYSIM_DEBUG", "0"))
log_level = logging.DEBUG if debug_mode >= 1 else logging.INFO

# We need to aggressively configure logging because Uvicorn may have already set up handlers
# and basicConfig does nothing if handlers exist. This runs at import deliberately:
# run_server.py sets TOPOBATHYSIM_DEBUG before uvicorn imports this module, and each
# worker process imports it afresh, so there is no earlier hook to use.
# Test implication: importing this module pins the "topobathysim" and "topobathyserve"
# loggers to the level above for the rest of the process. Tests that assert on DEBUG
# records must scope caplog to the package logger, for example
# `caplog.set_level(logging.DEBUG, logger="topobathysim")`, rather than relying on root.
root_logger = logging.getLogger()
root_logger.setLevel(log_level)

# Formatter
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

# File Handler (Always add)
file_handler = logging.FileHandler(log_dir / "service.log")
file_handler.setFormatter(formatter)
root_logger.addHandler(file_handler)

# Stream Handler (Add if not present, to avoid duplicates from Uvicorn)
# Uvicorn adds a stream handler to the root logger or 'uvicorn'.
# If we are running under uvicorn, the root logger might already have a handler.
# BUT we want to ensure our format is used and it goes to stdout.
# So we aggressively add it if we don't see one that looks like ours.
has_console = False
for h in root_logger.handlers:
    if isinstance(h, logging.StreamHandler) and h.stream == sys.stdout:
        h.setFormatter(formatter)  # Force our formatter
        h.setLevel(log_level)
        has_console = True
        break

if not has_console:
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(log_level)
    root_logger.addHandler(stream_handler)


# --- Silence External Libraries even in Debug Mode ---
# We generally want to inspect our own logic, but not the internals of rasterio/matplotlib/etc for every tile.
for noisy_logger in [
    "rasterio",
    "fiona",
    "shapely",
    "matplotlib",
    "PIL",
    "botocore",
    "urllib3",
    "asyncio",
    "multipart",
    "uvicorn.access",  # Uvicorn access logs are redundant if we have middleware
]:
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)

# Explicitly set level for our app loggers
logging.getLogger("topobathyserve").setLevel(log_level)
logging.getLogger("topobathysim").setLevel(log_level)

logger = logging.getLogger("topobathyserve")

# Default Policy Path
DEFAULT_POLICY = Path(__file__).resolve().parent.parent.parent / "policies" / "examples" / "wlis.yaml"
POLICY_PATH: Path | None = None


def deg2num(lat_deg: float, lon_deg: float, z: int) -> tuple[int, int]:
    """Converts WGS84 coordinates to tile coordinates."""
    lat_rad = math.radians(lat_deg)
    n = 2.0**z
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)


def num2deg(xtile: int, ytile: int, zoom: int) -> tuple[float, float]:
    """Converts tile coordinates to WGS84 coordinates (top-left)."""
    n = 2.0**zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)


def tile_bounds_calc(x: int, y: int, z: int) -> dict[str, float]:
    """Calculates the bounding box for a given tile."""
    lat_max, lon_min = num2deg(x, y, z)
    lat_min, lon_max = num2deg(x + 1, y + 1, z)
    return {"north": lat_max, "south": lat_min, "west": lon_min, "east": lon_max}


def _parse_bbox_params(
    bbox: str | None,
    west: float | None,
    south: float | None,
    east: float | None,
    north: float | None,
) -> tuple[float, float, float, float]:
    if bbox:
        parts = [p.strip() for p in bbox.split(",") if p.strip()]
        if len(parts) != 4:
            raise HTTPException(
                status_code=400,
                detail="bbox must be 'west,south,east,north' with 4 comma-separated values",
            )
        try:
            west, south, east, north = (float(p) for p in parts)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="bbox values must be numeric") from exc

    if west is None or south is None or east is None or north is None:
        raise HTTPException(
            status_code=400,
            detail="Provide bbox or west/south/east/north query parameters",
        )

    if west >= east or south >= north:
        raise HTTPException(
            status_code=400,
            detail="Invalid bbox ordering (expected west < east and south < north)",
        )

    return west, south, east, north


def _normalize_output_format(format_raw: str) -> str:
    fmt = format_raw.strip().lower()
    if fmt in {"geotiff", "tiff", "tif"}:
        return "geotiff"
    if fmt in {"zarr", "zip"}:
        return "zarr"
    raise HTTPException(status_code=400, detail=f"Unsupported format '{format_raw}'")


def _json_safe(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, default=str))
    except (TypeError, ValueError):
        return str(value)


def _build_zarr_attrs(
    ds: xr.Dataset,
    bbox: tuple[float, float, float, float],
    resolution: float,
    policy_name: str,
) -> dict[str, Any]:
    crs = None
    if hasattr(ds, "rio") and ds.rio.crs:
        crs = ds.rio.crs.to_string()

    provenance = ds.attrs.get("provenance_dict", {})
    providers_used: list[str] = []
    for entry in provenance.values():
        if isinstance(entry, dict):
            provider = entry.get("provider")
            if provider:
                providers_used.append(str(provider))

    return {
        "crs": crs or "EPSG:4326",
        "bbox": [float(v) for v in bbox],
        "resolution_m": float(resolution),
        "vertical_datum": ds.attrs.get("vertical_datum", "Unknown"),
        "policy": policy_name,
        "providers_used": sorted(set(providers_used)),
        "created": datetime.now(timezone.utc).isoformat(),
        "provenance_dict": _json_safe(provenance) if provenance else {},
    }


def _zarr_zip_response(ds_out: xr.Dataset) -> FileResponse:
    """Write `ds_out` as a consolidated Zarr v2 store, zip it, and stream the archive.

    zarr_format=2 keeps the export readable by Zarr.jl, which also needs an explicit
    fill value on the integer source_elevation array. The fill value goes in encoding
    only: xarray refuses to encode a variable that also carries _FillValue in attrs.
    """
    if "source_elevation" in ds_out:
        ds_out["source_elevation"].attrs.pop("_FillValue", None)
        ds_out["source_elevation"].encoding["_FillValue"] = 0

    tmpdir = tempfile.mkdtemp()
    zarr_dir = Path(tmpdir) / "fused.zarr"
    ds_out.to_zarr(zarr_dir, mode="w", consolidated=True, zarr_format=2)
    shutil.make_archive(str(Path(tmpdir) / "fused.zarr"), "zip", str(zarr_dir))

    return FileResponse(
        path=Path(tmpdir) / "fused.zarr.zip",
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=fused.zarr.zip"},
        background=BackgroundTask(shutil.rmtree, tmpdir, ignore_errors=True),
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Load .env (Resolve from topobathysim root)
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    global POLICY_PATH
    policy_env = os.getenv("TOPOBATHY_POLICY")
    POLICY_PATH = Path(policy_env) if policy_env else DEFAULT_POLICY

    if not POLICY_PATH.exists():
        logger.warning(f"Policy file not found: {POLICY_PATH}. Service may fail.")
    else:
        logger.info(f"Using Fusion Policy: {POLICY_PATH}")

    yield
    # clean up logic if needed


app = FastAPI(
    title="BathyServe",
    description="Microservice for Hybrid Bathymetry (GEBCO 2025 + BlueTopo)",
    version="0.2.0",
    lifespan=lifespan,
)

app.include_router(cache_viewer.router)
app.include_router(dem_viewer.router)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> FileResponse:
    return FileResponse(static_dir / "favicon.ico")


@app.middleware("http")
async def log_requests(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(
        f"Request: {request.method} {request.url} | "
        f"Status: {response.status_code} | "
        f"Latency: {process_time:.4f}s"
    )
    return response


# Mount Viewer
static_dir = Path(__file__).resolve().parent.parent / "static"

if static_dir.exists():
    from fastapi.staticfiles import StaticFiles

    app.mount("/viewer", StaticFiles(directory=str(static_dir), html=True), name="static")


@app.get("/", include_in_schema=False)
async def viewer_root() -> RedirectResponse:
    return RedirectResponse("/viewer/")


@app.get("/health")
def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "service": "topobathyserve"}


def get_policy_path() -> Path:
    if POLICY_PATH is None or not POLICY_PATH.exists():
        raise HTTPException(status_code=503, detail="Policy path not initialized")
    return POLICY_PATH


@app.post("/policy/update")
async def update_policy(
    request: PolicyUpdateRequest,
    policy_path: Annotated[Path, Depends(get_policy_path)],
) -> dict[str, str]:
    """Unified endpoint to save a policy. Handles full YAML or 'grounded fragments' (curations)."""
    # 1. Sanitize Name
    safe_name = "".join([c for c in request.name if c.isalnum() or c in ("-", "_")])
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid policy name")

    final_yaml = request.policy_yaml

    # 2. Handle Curation (Grounded Fragment)
    if request.curations is not None:
        if not request.base_yaml:
            raise HTTPException(status_code=400, detail="curations require base_yaml")

        try:
            from topobathysim.policy.schema import FilterConfig

            policy = load_policy_from_str(request.base_yaml)

            for step_idx_str, source_names in request.curations.items():
                step_idx = int(step_idx_str)
                if step_idx < len(policy.variables[0].steps):
                    step = policy.variables[0].steps[step_idx]
                    if step.filter is None:
                        step.filter = FilterConfig()
                    if step.filter.exclude_patterns is None:
                        step.filter.exclude_patterns = []

                    for name in source_names:
                        if name not in step.filter.exclude_patterns:
                            step.filter.exclude_patterns.append(name)

            final_yaml = yaml.dump(policy.model_dump(), sort_keys=False)
        except Exception as e:
            logger.error(f"Curation application failed: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to apply curation: {e}") from e

    if not final_yaml:
        raise HTTPException(status_code=400, detail="Provide either policy_yaml or base_yaml+curations")

    # 3. Final Validation & Save
    try:
        load_policy_from_str(final_yaml)  # Ensure it's valid policy YAML
        dest_path = policy_path.parent / f"{safe_name}.yaml"

        with open(dest_path, "w") as f:
            f.write(final_yaml)

        logger.info(f"Updated policy saved to {dest_path}")
        return {"status": "success", "name": safe_name, "path": str(dest_path)}
    except Exception as e:
        logger.error(f"Policy save failed: {e}")
        raise HTTPException(status_code=500, detail=f"Policy update failed: {e}") from e


@app.get("/fuse")
def fuse(
    policy_path: Annotated[Path, Depends(get_policy_path)],
    bbox: str | None = Query(
        None,
        description="Bounding box as 'west,south,east,north'",
    ),
    west: float | None = Query(None, description="Bounding box west (lon)"),
    south: float | None = Query(None, description="Bounding box south (lat)"),
    east: float | None = Query(None, description="Bounding box east (lon)"),
    north: float | None = Query(None, description="Bounding box north (lat)"),
    resolution: float = Query(30.0, description="Output resolution in meters"),
    format: str = Query("geotiff", description="Output format: geotiff or zarr"),
) -> Response:
    bbox_tuple = _parse_bbox_params(bbox, west, south, east, north)
    fmt = _normalize_output_format(format)

    try:
        _policy = load_policy(str(policy_path))
        _phash = hash_policy(_policy.model_dump())
        logger.info(
            f"GET /fuse policy={policy_path.name} hash={_phash[:8]} bbox={bbox_tuple} res={resolution}m"
        )
        ds = run(str(policy_path), bbox_tuple, resolution=resolution, use_cache=True)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"/fuse failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if "elevation" not in ds or ds["elevation"].size == 0:
        raise HTTPException(status_code=404, detail="Fusion returned no elevation data")

    if fmt == "zarr":
        ds_out = ds.copy()
        keep_vars = ["elevation"]
        if "source_elevation" in ds_out:
            keep_vars.append("source_elevation")
        ds_out = ds_out[keep_vars]
        ds_out.attrs = _build_zarr_attrs(ds, bbox_tuple, resolution, policy_path.name)
        return _zarr_zip_response(ds_out)

    buf = BytesIO()
    ds["elevation"].rio.to_raster(buf, driver="GTiff")
    return Response(content=buf.getvalue(), media_type="image/tiff")


@app.post("/fuse")
def fuse_post(
    request: FusionRequest,
    policy_path: Annotated[Path, Depends(get_policy_path)],
) -> Response:
    fmt = _normalize_output_format(request.format)

    # Dynamic Policy Logic
    policy_input: Path | str = policy_path
    if request.policy_yaml:
        policy_input = request.policy_yaml
    elif request.policy_name:
        # Simple lookup: check if a policy with this name exists in policies dir?
        # For now, let's just log and fallback or fail?
        # Actually, let's look for it in the same directory as default policy if possible.
        # But for strict compatibility with instructions, let's prioritize override string.
        pass

    try:
        _policy = (
            load_policy_from_str(str(policy_input)) if request.policy_yaml else load_policy(str(policy_input))
        )
        _phash = hash_policy(_policy.model_dump())
        logger.info(
            f"POST /fuse policy={getattr(_policy, 'name', str(policy_input))} "
            f"hash={_phash[:8]} bbox={request.bbox} res={request.resolution}m"
        )
        ds = run(
            policy_input=policy_input,
            bbox=request.bbox,
            resolution=request.resolution,
            use_cache=True,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"POST /fuse failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if "elevation" not in ds or ds["elevation"].size == 0:
        raise HTTPException(status_code=404, detail="Fusion returned no elevation data")

    if fmt == "zarr":
        ds_out = ds.copy()
        keep_vars = ["elevation"]
        if "source_elevation" in ds_out:
            keep_vars.append("source_elevation")
        ds_out = ds_out[keep_vars]

        # Policy name for metadata
        p_name = policy_path.name
        if request.policy_name:
            p_name = request.policy_name
        elif request.policy_yaml:
            p_name = "custom_override"

        ds_out.attrs = _build_zarr_attrs(ds, request.bbox, request.resolution, p_name)
        return _zarr_zip_response(ds_out)

    buf = BytesIO()
    ds["elevation"].rio.to_raster(buf, driver="GTiff")
    return Response(content=buf.getvalue(), media_type="image/tiff")


# Visualization Constants
GLOBAL_VMIN = -2000.0
GLOBAL_VMAX = 1200.0

# Feature Flags
SKIP_LAND_BACKGROUND = os.getenv("SKIP_LAND_BACKGROUND", "False").lower() in (
    "true",
    "1",
    "yes",
)

# Tiling Constants
TILE_SIZE_PX = 512
TILE_PADDING_PX = 4  # Standard padding to avoid edge artifacts and ensure cache parity

# Concurrency: limit simultaneous fusion runs per worker to prevent resource exhaustion.
# At high zoom (e.g. z=17, ~0.9m), each run() is CPU/IO-heavy; unbounded concurrency
# causes all requests to compete and none to finish before client timeout.
_FUSION_SEMAPHORE = threading.Semaphore(int(os.environ.get("TOPOBATHY_MAX_CONCURRENT_TILES", "3")))
_TILE_FUSION_TIMEOUT = int(os.environ.get("TOPOBATHY_TILE_TIMEOUT", "120"))  # seconds


# So I can change the signature of render_png to accept an optional 'legend' dict.


def render_png(
    da: xr.DataArray,
    style: str = "default",
    vmin: float | None = None,
    vmax: float | None = None,
    zoom: int = 13,
    legend: dict[int, str] | None = None,  # NEW ARGUMENT
    pad: int = 0,
    provenance_dict: dict[Any, dict[str, str]] | None = None,
) -> bytes:
    """Renders DataArray to PNG bytes using a terrain colormap, optionally with hillshade."""
    import matplotlib.colors as mcolors
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource

    buf = BytesIO()
    vals = da.values.astype(np.float32)

    # Handle 3D Arrays (e.g. (1, h, w) or (C, h, w))
    # Handle 3D Arrays
    if vals.ndim == 3:
        if vals.shape[0] == 1:
            # (1, h, w) -> (h, w)
            vals = vals.squeeze(0)
        elif vals.shape[-1] == 1:
            # (h, w, 1) -> (h, w)
            vals = vals.squeeze(-1)
        else:
            # Ambiguous (C, h, w) or (h, w, C).
            # We assume (C, h, w) standard raster layout if not singleton.
            logger.warning(f"render_png received 3D array {vals.shape}. Using first band.")
            vals = vals[0, :, :]

    # Last resort check
    if vals.ndim != 2:
        logger.error(f"render_png expected 2D array, got {vals.shape}")
        # Try to flatten or reshape? No, just fail gracefully or let it crash with better log
        # But we can try to force 2D if it's (h, w, 1) or similar
        if vals.ndim == 3 and vals.shape[-1] == 1:
            vals = vals.squeeze(-1)

    # Source Visualization Style
    if style == "source":
        # Safeguard: Ensure source IDs are integers. Floating point noise or NaNs
        # can cause np.unique() to explode (O(N^2) loop), leading to timeouts/crashes.
        if not np.issubdtype(vals.dtype, np.integer):
            vals = np.nan_to_num(vals, nan=0).astype(np.int32)

        if pad > 0:
            vals = vals[pad:-pad, pad:-pad]
        h, w = vals.shape

        # Qualitative Palette (Set3 / Paired)
        # https://matplotlib.org/stable/users/explain/colors/colormaps.html#qualitative
        # Set3 has 12 colors. Paired has 12.
        # We map Provider ID -> Color Index

        # If no legend provided, fallback to random/hash
        cmap_name = "Set3"  # Good for up to 12 categories
        base_cmap = plt.get_cmap(cmap_name)

        # Create a custom colormap where specific integer values map to specific colors
        # We know IDs are 1..N
        # We want ID 1 -> Color 0, ID 2 -> Color 1, etc.

        # Helper to get color for an ID
        def get_color(uid: int) -> tuple[float, float, float, float]:
            from typing import cast

            if uid == 0:
                return cast(tuple[float, float, float, float], (0, 0, 0, 1))  # No Data / Background
            if np.isnan(uid):
                return cast(tuple[float, float, float, float], (0, 0, 0, 0))  # Transparent

            # Use provenance_dict to map detailed IDs back to their provider color consistently
            if provenance_dict:
                int_uid = int(uid)
                str_uid = str(int_uid)

                prov_info = provenance_dict.get(int_uid) or provenance_dict.get(str_uid)

                if prov_info:
                    provider_str = prov_info.get("provider", "")

                    if provider_str and legend:
                        # Look up the provider string in the legend map to get its global integer ID
                        for leg_id, leg_name in legend.items():
                            if leg_name == provider_str:
                                idx = (leg_id - 1) % 12
                                return cast(tuple[float, float, float, float], base_cmap(idx))

            # Fallback for generic provider IDs
            try:
                idx = (int(uid) - 1) % 12  # Cycle through 12 colors
            except (ValueError, TypeError):
                idx = 0
            return cast(tuple[float, float, float, float], base_cmap(idx))

        h, w = vals.shape
        rgba = np.zeros((h, w, 4), dtype=np.float32)

        unique_ids = np.unique(vals)
        for uid in unique_ids:
            if np.isnan(uid) or uid == 0:
                continue

            mask = np.isclose(vals, uid)
            rgba[mask] = get_color(uid)

        mpimg.imsave(buf, rgba, format="png")
        buf.seek(0)
        return buf.getvalue()

    # Contours Style
    if style == "contours":
        if pad > 0:
            vals = vals[pad:-pad, pad:-pad]
        h, w = vals.shape

        # Safety Check for NaN
        if np.isnan(vals).all():
            fig = plt.figure(figsize=(1, 1), dpi=100)
            fig.savefig(buf, format="png", transparent=True)
            plt.close(fig)
            buf.seek(0)
            return buf.getvalue()

        dpi = 100
        fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
        ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
        ax.set_axis_off()

        interval = 10.0
        if zoom >= 14:
            interval = 5.0
        if zoom >= 16:
            interval = 2.0
        if zoom >= 18:
            interval = 1.0

        # Catch empty range
        if np.nanmin(vals) == np.nanmax(vals):
            levels = [np.nanmin(vals)]
        else:
            vmin_cnt = math.floor(np.nanmin(vals) / interval) * interval
            vmax_cnt = math.ceil(np.nanmax(vals) / interval) * interval
            levels = np.arange(vmin_cnt, vmax_cnt + interval, interval).tolist()

        cs = ax.contour(vals, levels=levels, colors="black", linewidths=0.5, alpha=0.7, origin="upper")
        ax.contour(vals, levels=[0], colors="red", linewidths=1.5, alpha=0.8, origin="upper")

        if zoom >= 14:
            ax.clabel(cs, inline=True, fontsize=8, fmt="%1.0f")

        fig.savefig(buf, format="png", transparent=True)
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()

    # Hillshade Calculation
    ls = LightSource(azdeg=315, altdeg=45)

    if style == "hillshade":
        hillshade = ls.hillshade(vals, vert_exag=10, dx=1.0, dy=1.0)
        if pad > 0:
            hillshade = hillshade[pad:-pad, pad:-pad]
        mpimg.imsave(buf, hillshade, cmap="gray", format="png")
        buf.seek(0)
        return buf.getvalue()

    # Default: Color + Hillshade Overlay
    eff_vmin = vmin if vmin is not None else GLOBAL_VMIN
    eff_vmax = vmax if vmax is not None else GLOBAL_VMAX

    cmap = plt.get_cmap("terrain")
    norm: mcolors.Normalize | None = None

    if style in ["chart", "default"]:
        # Use standard Blues_r for water, but truncate it before pure white.
        # Rationale: Standard Blues_r hits pure white at 1.0 (0m).
        # This washes out shallow texture because hillshading on white is weak.
        # The standalone "Blues" style looks better because it normalizes VMIN-VMAX linearly,
        # placing 0m (~60% of range) at a nice pale blue (not white).
        # We manually emulate this by sampling Blues_r only up to 0.7 for water.
        blues = plt.get_cmap("Blues_r")

        land_colors = [
            (0.0, "#F7E5B5"),
            (0.2, "lightgreen"),
            (0.5, "forestgreen"),
            (0.8, "sienna"),
            (1.0, "snow"),
        ]
        land_cmap = mcolors.LinearSegmentedColormap.from_list("land_custom", land_colors)

        n_bins = 256
        # Truncate water colormap at 0.7 to keep shallowest water pale blue (not white)
        colors_water = blues(np.linspace(0.0, 0.7, n_bins))
        colors_land = land_cmap(np.linspace(0.0, 1.0, n_bins))

        colors_combined = np.vstack((colors_water, colors_land))
        cmap = mcolors.LinearSegmentedColormap.from_list("chart_hybrid", colors_combined)
        norm = mcolors.TwoSlopeNorm(vmin=eff_vmin, vcenter=0, vmax=eff_vmax)

    elif style == "blues":
        cmap = plt.get_cmap("Blues_r")
        norm = mcolors.Normalize(vmin=eff_vmin, vmax=eff_vmax)

    else:
        if eff_vmax <= 0:
            # Depth only
            try:
                cmap = plt.get_cmap("turbo")
            except ValueError:
                cmap = plt.get_cmap("jet")
            norm = mcolors.Normalize(vmin=eff_vmin, vmax=eff_vmax)
        elif eff_vmin >= 0:
            # Land only
            _terrain = plt.get_cmap("terrain")
            land_rgba = _terrain(np.linspace(0.5, 1.0, 256))
            land_cmap = mcolors.LinearSegmentedColormap.from_list("terrain_land", land_rgba)
            cmap = land_cmap
            norm = mcolors.Normalize(vmin=eff_vmin, vmax=eff_vmax)
        else:
            cmap = plt.get_cmap("terrain")
            norm = mcolors.TwoSlopeNorm(vmin=eff_vmin, vcenter=0, vmax=eff_vmax)

    try:
        rgb = ls.shade(
            vals,
            cmap=cmap,
            norm=norm,
            vert_exag=10,
            dx=1.0,
            dy=1.0,
            blend_mode="overlay",
        )
        if pad > 0:
            rgb = rgb[pad:-pad, pad:-pad]

        mpimg.imsave(buf, rgb, format="png")

    except Exception as e:
        logger.warning(f"Shade failed: {e}")
        if pad > 0:
            vals = vals[pad:-pad, pad:-pad]
        mpimg.imsave(buf, vals, cmap="terrain", format="png")

    buf.seek(0)
    return buf.getvalue()


@app.get("/legend")
async def get_legend(policy_path: Annotated[Path, Depends(get_policy_path)]) -> dict[str, Any]:
    """
    Returns the legend mapping (Provider Name -> Color Hex) for the active policy.
    Used by the frontend to render the dynamic legend.
    """
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    from topobathysim.policy.loader import generate_provider_legend, load_policy

    # Reload policy to be safe (or assume it's loaded in runtime, but runtime doesn't expose object easily)
    # We load it here to get the legend mapping
    try:
        policy = load_policy(str(policy_path))
        legend_map = generate_provider_legend(policy)  # {1: "ProviderA", 2: "ProviderB"}

        # Generate Colors matching render_png logic (Set3)
        cmap = plt.get_cmap("Set3")

        # Build priority map: provider_key -> step_index (0-based, higher = higher priority)
        priority_map: dict[str, int] = {}
        for variable in policy.variables:
            for step_idx, step in enumerate(variable.steps):
                priority_map[step.provider] = step_idx

        items = []
        for pid, name in legend_map.items():
            # ID to Color
            idx = (pid - 1) % 12
            rgba = cmap(idx)
            hex_color = mcolors.to_hex(rgba)

            items.append({"id": pid, "name": name, "color": hex_color, "priority": priority_map.get(name, 0)})

        return {"items": items}

    except Exception as e:
        logger.error(f"Failed to generate legend: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/elevation")
async def get_elevation(
    lat: float,
    lon: float,
    policy_path: Annotated[Path, Depends(get_policy_path)],
    zoom: int | None = Query(None, description="Preferred zoom level/resolution to check in cache"),
) -> ElevationResponse:
    try:
        logger.info(f"Elevation Lookup: lat={lat}, lon={lon}, zoom={zoom}")

        # Check if we have any cached fused datasets for this coordinate
        # We start with the requested zoom, or the highest available in the tile cache
        base_cache_dir = get_cache_root()

        check_zooms = []
        if zoom is not None:
            check_zooms = [zoom]
        else:
            # Dynamically discover available zoom levels from the tile data cache
            tile_data_dir = base_cache_dir / "tiles" / "data"
            if tile_data_dir.exists():
                try:
                    # Collect all zoom level directories (must be numeric)
                    zooms = []
                    for d in tile_data_dir.iterdir():
                        if d.is_dir() and d.name.isdigit():
                            zooms.append(int(d.name))
                    # Sort descending for highest resolution first
                    check_zooms = sorted(zooms, reverse=True)
                except Exception as e:
                    logger.debug(f"Failed to scan tile cache for zooms: {e}")

        # Fallback to defaults if no zooms discovered (Z15 is ~4.7m, Z14 is ~9.5m)
        if not check_zooms:
            check_zooms = [15, 14, 13, 12]

        for z in check_zooms:
            try:
                # 1. Determine Tile Center Resolution
                # We must match the resolution used during tile generation,
                # which is based on the TILE CENTER latitude,
                # not the specific point latitude.
                tx, ty = deg2num(lat, lon, z)
                bounds = tile_bounds_calc(tx, ty, z)
                center_lat = (bounds["north"] + bounds["south"]) / 2.0

                n = 2.0**z
                res_meters = 156543.03 * math.cos(math.radians(center_lat)) / n
                res_meters = round(res_meters, 8)

                # 2. Check Cache for the Standard Grid Cell containing this point
                # We do NOT use padded_bbox here anymore because run() caches
                # by standardized grid cells (0.05 deg), not by arbitrary tile bboxes.
                try:
                    from topobathysim.runtime import get_cache_info_for_point

                    cache_path, key_hash = get_cache_info_for_point(str(policy_path), lon, lat, res_meters)
                except ValueError:
                    continue

                logger.debug(
                    f"Checking Elevation Cache: Z={z} res={res_meters} "
                    f"point=({lon:.5f},{lat:.5f}) key={key_hash}"
                )

                if cache_path.exists():
                    logger.info(f"Elevation Cache Hit (Cell Z{z}): {cache_path.name}")
                    from filelock import FileLock

                    with FileLock(cache_path.with_suffix(".lock")):
                        ds = xr.open_dataset(
                            cache_path,
                            engine="zarr",
                            chunks=None,
                            decode_coords="all",
                            consolidated=should_consolidate(),
                        )
                        ds = ds.load()

                    # Sample closest using projected coordinates (EPSG:3857)
                    earth_r = 6378137.0
                    x_m = earth_r * math.radians(lon)
                    y_m = earth_r * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))

                    val = ds["elevation"].sel(x=x_m, y=y_m, method="nearest").item()
                    return ElevationResponse(elevation=float(val) if not np.isnan(val) else None)

            except Exception as e:
                logger.debug(f"Failed to check/load tile cache at Z{z}: {e}")
                continue

        logger.info(f"Elevation Cache Miss for {lat}, {lon} (Zoom {zoom}). Falling back to 10m global run.")

        # Request a reasonable area to ensure we catch grid cells (GEBCO is ~450m)
        delta = 0.02  # ~2km buffer
        bbox = (lon - delta, lat - delta, lon + delta, lat + delta)

        # Run inference
        # Use 10m resolution for sampling
        ds = run(str(policy_path), bbox, resolution=10.0)

        if ds["elevation"].size == 0:
            return ElevationResponse(elevation=None)  # type: ignore

        # Sample closest using projected coordinates (EPSG:3857)
        earth_r = 6378137.0
        x_m = earth_r * math.radians(lon)
        y_m = earth_r * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))

        val = ds["elevation"].sel(x=x_m, y=y_m, method="nearest").item()

        return ElevationResponse(elevation=float(val) if not np.isnan(val) else None)
    except ValueError as e:
        # Catch CRS validation errors from runtime
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"get_elevation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/source_info")
async def get_source_info(
    lat: float, lon: float, policy_path: Annotated[Path, Depends(get_policy_path)]
) -> dict[str, str]:
    try:
        delta = 0.00015
        bbox = (lon - delta, lat - delta, lon + delta, lat + delta)
        ds = run(str(policy_path), bbox, resolution=10.0)

        if ds["source_elevation"].size == 0:
            return {"source": "No Data", "id": "NaN"}

        # Sample closest using projected coordinates (EPSG:3857)
        earth_r = 6378137.0
        x_m = earth_r * math.radians(lon)
        y_m = earth_r * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))

        sid = ds["source_elevation"].sel(x=x_m, y=y_m, method="nearest").item()

        # TODO: Lookup ID in ds.attrs or legend
        return {"source": f"Provider ID {sid}", "id": str(sid), "policy": policy_path.name}
    except Exception as e:
        return {"error": str(e)}


@app.get("/metadata", response_model=TIDReportResponse)
async def get_metadata(
    north: float,
    south: float,
    west: float,
    east: float,
    policy_path: Annotated[Path, Depends(get_policy_path)],
) -> TIDReportResponse:
    try:
        # Metadata logic in runtime is minimal, mostly about the policy
        # Just run a coarse integration to get attrs?
        # Or just return policy details?
        # For now, return a stub report
        report = {
            "policy": str(policy_path),
            "status": "Runtime Metadata Not Generally Available via API yet",
        }
        return TIDReportResponse(report=report)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/tiles/coverage")
def get_tile_coverage(
    north: float,
    south: float,
    west: float,
    east: float,
    zoom: int,
) -> dict[str, object]:
    """
    Calculates the XYZ tile indices covering a given bounding box.
    Response includes min/max indices and a list of tiles (capped at 100 items).
    """
    try:
        # Get tile coordinates for top-left (NW) and bottom-right (SE)
        # Note: BBox is North, South, West, East
        # In XYZ, y increases southwards

        # Valid Latitude Checks to prevent math domain errors
        north_clamped = min(max(north, -85.0511), 85.0511)
        south_clamped = min(max(south, -85.0511), 85.0511)

        min_x, min_y = deg2num(north_clamped, west, zoom)
        max_x, max_y = deg2num(south_clamped, east, zoom)

        # Handle wrap around or ordering if needed, simplistically:
        # Swap if min > max (though deg2num logic should handle standard west<east)
        if min_x > max_x:
            min_x, max_x = max_x, min_x
        if min_y > max_y:
            min_y, max_y = max_y, min_y

        count = (max_x - min_x + 1) * (max_y - min_y + 1)

        tiles = []
        # Only list specific tiles if count is reasonable
        # Increased limit from 100 to 1000 to support larger viewport debugging
        if count <= 1000:
            for x in range(min_x, max_x + 1):
                for y in range(min_y, max_y + 1):
                    info: dict[str, object] = {"z": zoom, "x": x, "y": y}
                    info["bounds"] = tile_bounds_calc(x, y, zoom)
                    tiles.append(info)

        return {
            "zoom": zoom,
            "bounds": {"north": north, "south": south, "west": west, "east": east},
            "limits": {"x_min": min_x, "x_max": max_x, "y_min": min_y, "y_max": max_y},
            "count": count,
            "tiles": tiles,
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.get("/tiles/{z}/{x}/{y}/metadata", response_model=TileMetadataResponse)
def get_tile_metadata(
    z: int,
    x: int,
    y: int,
    policy_path: Annotated[Path, Depends(get_policy_path)],
    lidar_url: str | None = None,
    ept_url: str | None = None,
    use_seam_blending: bool = True,
    style: str = Query("default", description="Visualization style"),
    vmin: float | None = Query(None, description="Explicit Min Elevation"),
    vmax: float | None = Query(None, description="Explicit Max Elevation"),
    tile_size: int = Query(512, description="Output pixel dimension for standard square XYZ tiles"),
) -> TileMetadataResponse:
    """
    Returns metadata for a specific fused tile (XYZ).
    """
    import datetime
    import hashlib

    n = 2.0**z
    west = x / n * 360.0 - 180.0
    east = (x + 1) / n * 360.0 - 180.0
    lat_rad_north = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    north = math.degrees(lat_rad_north)
    lat_rad_south = math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n)))
    south = math.degrees(lat_rad_south)

    bounds = {"north": north, "south": south, "west": west, "east": east}

    # Check Cache existence
    base_cache_dir = get_cache_root() / "tiles"
    params_str = f"{policy_path.name}|{vmin}|{vmax}|{tile_size}"
    param_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]

    # Check NPY (data) first
    npy_path = base_cache_dir / "data" / str(z) / str(x) / f"{y}_{param_hash}.npy"
    # Then try Visual
    safe_style = "".join(c for c in style if c.isalnum() or c in ("-", "_")) or "default"
    png_path = base_cache_dir / "visual" / safe_style / str(z) / str(x) / f"{y}_{param_hash}.png"

    created_at = None
    cache_status = "miss"
    provenance_dict = None

    for path_to_check in [npy_path, png_path]:
        if path_to_check.exists() and path_to_check.stat().st_size > 0:
            mtime = path_to_check.stat().st_mtime
            dt = datetime.datetime.fromtimestamp(mtime, tz=datetime.timezone.utc)
            created_at = dt.strftime("%Y%m%d-%H:%M:%S UTC")
            cache_status = "hit"

            # Check for sidecar metadata JSON
            import json

            meta_path = path_to_check.parent / f"{y}_{param_hash}_meta.json"
            if meta_path.exists():
                try:
                    with open(meta_path) as f:
                        provenance_dict = json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to read provenance {meta_path}: {e}")

            break

    return TileMetadataResponse(
        z=z,
        x=x,
        y=y,
        bounds=bounds,
        created_at=created_at,
        cache_status=cache_status,
        fusion_sources=policy_path.name,
        request_params=params_str,
        provenance_dict=provenance_dict,
    )


@app.get("/tiles/{z}/{x}/{y}/pixel")
def get_tile_pixel_source(
    z: int,
    x: int,
    y: int,
    lat: float,
    lon: float,
    policy_path: Annotated[Path, Depends(get_policy_path)],
    style: str = Query("default"),
    vmin: float | None = Query(None),
    vmax: float | None = Query(None),
    tile_size: int = Query(512),
) -> dict[str, Any]:
    """
    Returns the source that contributed the pixel at (lat, lon) within tile (z, x, y).

    Requires the PNG tile to have been generated first (reads the ``_src.npz`` sidecar
    written alongside the PNG cache file).
    """
    import hashlib
    import json

    earth_r = 6378137.0
    half = math.pi * earth_r  # 20037508.342789244

    # Tile bounds in Web Mercator (EPSG:3857)
    tile_width_m = 2 * half / (2**z)
    west_m = x * tile_width_m - half
    north_m = half - y * tile_width_m

    # Project lat/lon (EPSG:4326) to Web Mercator
    x_m = earth_r * math.radians(lon)
    y_m = earth_r * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))

    # Map to pixel coordinates in the tile_size x tile_size grid
    col = int((x_m - west_m) / tile_width_m * tile_size)
    row = int((north_m - y_m) / tile_width_m * tile_size)
    col = max(0, min(tile_size - 1, col))
    row = max(0, min(tile_size - 1, row))

    base_cache_dir = get_cache_root() / "tiles"
    safe_style = "".join(c for c in style if c.isalnum() or c in ("-", "_")) or "default"
    tile_dir = base_cache_dir / "visual" / safe_style / str(z) / str(x)
    params_str = f"{policy_path.name}|{vmin}|{vmax}|{tile_size}"
    param_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]

    src_npz_path = tile_dir / f"{y}_{param_hash}_src.npz"
    if not src_npz_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Pixel source data not available. Request the PNG tile first.",
        )

    source_arr = np.load(src_npz_path)["source_elevation"]
    row = min(row, source_arr.shape[0] - 1)
    col = min(col, source_arr.shape[1] - 1)
    source_id = int(source_arr[row, col])

    meta_path = tile_dir / f"{y}_{param_hash}_meta.json"
    provenance_dict: dict[str, Any] = {}
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                provenance_dict = json.load(f)
        except Exception:
            pass

    source_info = provenance_dict.get(str(source_id))
    return {"source_id": source_id, "source": source_info, "pixel": {"row": row, "col": col}}


@app.get("/tiles/{z}/{x}/{y}")
@app.get("/tiles/{z}/{x}/{y}.tif")
@app.get("/tiles/{z}/{x}/{y}.png")
@app.get("/tiles/{z}/{x}/{y}.npy")
@app.get("/tiles/{z}/{x}/{y}.npz")
def get_xyz_tile(
    request: Request,
    z: int,
    x: int,
    y: int,
    policy_path: Annotated[Path, Depends(get_policy_path)],
    format: str = "tiff",
    lidar_url: str | None = None,
    ept_url: str | None = None,
    use_seam_blending: bool = True,
    style: str = Query("default", description="Visualization style"),
    vmin: float | None = Query(None, description="Explicit Min Elevation"),
    vmax: float | None = Query(None, description="Explicit Max Elevation"),
    tile_size: int = Query(512, description="Output pixel dimension for standard square XYZ tiles"),
    nocache: bool = Query(False, description="Bypass tile output cache and force regeneration"),
) -> Response:
    """
    XYZ Tile Endpoint.
    """
    # Infer format
    path = request.url.path
    if format == "tiff":  # Only override if default
        if path.endswith(".png"):
            format = "png"
        elif path.endswith(".npy"):
            format = "npy"
        elif path.endswith(".npz"):
            format = "npz"

    n = 2.0**z
    west = x / n * 360.0 - 180.0
    east = (x + 1) / n * 360.0 - 180.0
    lat_rad_north = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    north = math.degrees(lat_rad_north)
    lat_rad_south = math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n)))
    south = math.degrees(lat_rad_south)

    logger.info(f"XYZ Request: z={z} x={x} y={y} | Format={format} | Style={style} | VMin={vmin} VMax={vmax}")

    # --- Caching Logic ---
    base_cache_dir = get_cache_root() / "tiles"

    # Determine Cache Subdirectory based on format/style (Parity with Legacy)
    if format in ["png", "jpg", "jpeg"]:
        # Visual Cache: tiles/visual/{style}/{z}/{x}/
        safe_style = "".join(c for c in style if c.isalnum() or c in ("-", "_")) or "default"
        tile_dir = base_cache_dir / "visual" / safe_style / str(z) / str(x)
        ext = format
    elif format in ["npy", "npz"]:
        # Data Cache: tiles/data/{z}/{x}/
        tile_dir = base_cache_dir / "data" / str(z) / str(x)
        ext = format
    else:
        # Raw/Other Cache (TIFF): tiles/raw/{z}/{x}/
        tile_dir = base_cache_dir / "raw" / str(z) / str(x)
        ext = "tif" if format == "tiff" else format

    # Hash unique parameters (Policy, VMin/Max) to handle different renderings of same tile
    params_str = f"{policy_path.name}|{vmin}|{vmax}|{tile_size}"
    import hashlib

    param_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]

    tile_dir.mkdir(parents=True, exist_ok=True)
    tile_path = tile_dir / f"{y}_{param_hash}.{ext}"

    media_type = "image/tiff" if format == "tiff" else f"image/{format}"
    if format in ["npy", "npz"]:
        media_type = "application/octet-stream"

    # 1. Check Cache
    if not nocache and tile_path.exists() and tile_path.stat().st_size > 0:
        logger.debug(f"Cache Hit: {tile_path}")
        return FileResponse(tile_path, media_type=media_type)

    start_time = time.time()

    # Calculate Resolution in Meters for this Zoom Level (Approx at center Lat)
    center_lat = (north + south) / 2.0
    # Web Mercator resolution equation
    res_meters = 156543.03 * math.cos(math.radians(center_lat)) / (2**z)
    res_meters = round(res_meters, 8)

    # Pad requested bbox to supply surrounding context for PNG Hillshade rendering.
    # The trim MUST match the fetch halo (TILE_PADDING_PX) so no halo pixels remain
    # in the output — mismatched pad causes 1px seams at tile boundaries.
    pad_px = TILE_PADDING_PX if format == "png" else 0
    dx_deg = (east - west) / tile_size
    dy_deg = (north - south) / tile_size

    # Fetch slightly larger region than strictly necessary to prevent edge artifacts.
    # For source/data outputs, avoid fetch halo to prevent deterministic seam stripes
    # from neighboring-tile context leaking into per-tile provenance.
    fetch_pad = 0 if style == "source" or format in ["npy", "npz"] else TILE_PADDING_PX
    req_west = west - dx_deg * fetch_pad
    req_east = east + dx_deg * fetch_pad
    req_south = south - dy_deg * fetch_pad
    req_north = north + dy_deg * fetch_pad

    logger.info(
        f"Tile Run Call: z={z} x={x} y={y} res={res_meters} bbox={(req_west, req_south, req_east, req_north)}"
    )
    # Acquire semaphore to limit concurrent fusion runs (prevents resource exhaustion at high zoom)
    acquired = _FUSION_SEMAPHORE.acquire(timeout=_TILE_FUSION_TIMEOUT)
    if not acquired:
        logger.warning(f"Tile {z}/{x}/{y} timed out waiting for fusion slot ({_TILE_FUSION_TIMEOUT}s)")
        return Response(content="Server busy - tile fusion backlog", status_code=503)

    # Run Fusion
    try:
        ds = run(
            str(policy_path),
            (req_west, req_south, req_east, req_north),
            resolution=res_meters,
            use_cache=True,
            target_zoom=z,
            source="tile",
        )
        logger.info(f"Tile Fusion Result: res={res_meters} bbox={(req_west, req_south, req_east, req_north)}")
        logger.debug(f"/tiles run() returned ds with attrs: {ds.attrs}")
    except ValueError as e:
        # CRS Validation Error
        logger.warning(f"Tile {z}/{x}/{y} out of bounds for policy: {e}")
        return Response(content=f"Policy/CRS Mismatch: {e}", status_code=400)
    except Exception as e:
        import traceback

        logger.error(f"Runtime failed: {type(e).__name__}: {e}")
        logger.error(traceback.format_exc())
        return Response(content=f"Error: {e}", status_code=500)
    finally:
        _FUSION_SEMAPHORE.release()

    # Force alignment to explicitly bounded 256x256 grid for Leaflet rendering
    # Mathematically resample exact Web Mercator bounds to eliminate visual overlap seams
    try:
        if ds.rio.crs is not None and ds.rio.crs.to_string() == "EPSG:3857":
            # Compute exact EPSG:3857 meter bounds for this XYZ tile
            n_tiles = 2.0**z

            # WGS 84 Semimajor axis
            r_earth = 6378137.0

            # Spherical Mercator math
            min_x_m = x / n_tiles * (2 * math.pi * r_earth) - (math.pi * r_earth)
            max_x_m = (x + 1) / n_tiles * (2 * math.pi * r_earth) - (math.pi * r_earth)

            lat_rad_south = math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n_tiles)))
            min_y_m = r_earth * math.log(math.tan(math.pi / 4 + lat_rad_south / 2))

            lat_rad_north = math.atan(math.sinh(math.pi * (1 - 2 * y / n_tiles)))
            max_y_m = r_earth * math.log(math.tan(math.pi / 4 + lat_rad_north / 2))

            dx_m = (max_x_m - min_x_m) / tile_size
            dy_m = (max_y_m - min_y_m) / tile_size

            # Add 1x pixel padding specifically for PNG Hillshade rendering to avoid 3x3 kernel edge artifacts
            n_dim = tile_size + 2 * pad_px
            target_x = np.linspace(
                min_x_m - dx_m * (pad_px - 0.5),
                max_x_m + dx_m * (pad_px - 0.5),
                n_dim,
            )
            target_y = np.linspace(
                max_y_m + dy_m * (pad_px - 0.5), min_y_m - dy_m * (pad_px - 0.5), n_dim
            )  # North up

            crs = ds.rio.crs
            ds_attrs = ds.attrs
            ds_elev = ds["elevation"].interp(x=target_x, y=target_y, method="linear")
            if "source_elevation" in ds:
                # Ensure integer source IDs after interpolation
                # (nearest maintains values, but NaNs promote to float)
                ds_src = ds["source_elevation"].interp(x=target_x, y=target_y, method="nearest")
                ds_src = ds_src.fillna(0).astype("int32")
                ds = xr.Dataset({"elevation": ds_elev, "source_elevation": ds_src})
            else:
                ds = xr.Dataset({"elevation": ds_elev})
            ds.attrs = ds_attrs
            if crs is not None:
                ds.rio.write_crs(crs, inplace=True)
        else:
            # Fallback for non-EPSG:3857 to explicitly enforce output dimensions using scipy zoom
            target_px = tile_size + 2 * pad_px
            if ds["elevation"].shape != (target_px, target_px):
                import warnings

                from scipy.ndimage import zoom as _zoom

                elev = ds["elevation"].values
                zy = target_px / elev.shape[0]
                zx = target_px / elev.shape[1]

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    elev_resized = _zoom(elev.astype(np.float32), (zy, zx), order=1)

                old_x = ds.coords["x"].values
                old_y = ds.coords["y"].values
                new_x = np.linspace(old_x[0], old_x[-1], target_px)
                new_y = np.linspace(old_y[0], old_y[-1], target_px)

                ds_out = {
                    "elevation": xr.DataArray(elev_resized, dims=("y", "x"), coords={"y": new_y, "x": new_x})
                }

                if "source_elevation" in ds:
                    src_elev = ds["source_elevation"].values
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        src_elev_resized = _zoom(src_elev.astype(np.float32), (zy, zx), order=0).astype(
                            np.uint16
                        )
                    ds_out["source_elevation"] = xr.DataArray(
                        src_elev_resized, dims=("y", "x"), coords={"y": new_y, "x": new_x}
                    )

                crs = ds.rio.crs
                ds_attrs = ds.attrs
                ds = xr.Dataset(ds_out)
                ds.attrs = ds_attrs
                if crs is not None:
                    ds.rio.write_crs(crs, inplace=True)

    except Exception as e:
        logger.warning(f"Failed to cleanly interpolate/resize tile bounds: {e}")

    # Render
    if format == "png":
        if style == "source":
            # Generate Legend Map for Dynamic Coloring
            from topobathysim.policy.loader import generate_provider_legend, load_policy

            # We have policy_path, but need the object.
            # load_policy is cached? Let's hope it's fast enough or LRU cached.
            # In production, we'd cache this at app level.
            # Assuming load_policy is cheap enough for now (parsing YAML)
            try:
                pol = load_policy(str(policy_path))
                legend_map = generate_provider_legend(pol)
            except Exception as e:
                logger.warning(f"Failed to load legend for source rendering: {e}")
                legend_map = None

            data = render_png(
                ds["source_elevation"],
                style="source",
                vmin=vmin,
                vmax=vmax,
                zoom=z,
                legend=legend_map,
                pad=pad_px,
                provenance_dict=ds.attrs.get("provenance_dict"),
            )
        else:
            data = render_png(ds["elevation"], style=style, vmin=vmin, vmax=vmax, zoom=z, pad=pad_px)
        media_type = "image/png"

    elif format in ["npy", "npz"]:
        import io

        buf = io.BytesIO()
        if format == "npy":
            np.save(buf, ds["elevation"].values)
        else:
            np.savez_compressed(
                buf, elevation=ds["elevation"].values, source_elevation=ds["source_elevation"].values
            )
        data = buf.getvalue()
        media_type = "application/octet-stream"

    else:  # TIFF
        import io

        buf = io.BytesIO()
        da_out = ds["elevation"]

        # Ensure proper dimension order (Band, Y, X) for GDAL
        if da_out.ndim == 3 and da_out.shape[-1] == 1:
            # If shape is (H, W, 1), move band to front
            da_out = da_out.transpose("band", "y", "x")
            # If shape is (1, H, W) it's already fine

        # Write to Raster
        da_out.rio.to_raster(buf, driver="GTiff")
        data = buf.getvalue()
        media_type = "image/tiff"

    # 2. Write to Cache
    try:
        with open(tile_path, "wb") as f:
            f.write(data)

        logger.debug(f"Tile attributes before json write: {ds.attrs}")

        # Write Sidecar Metadata if we have provenance, filtered to IDs present in this tile
        if "provenance_dict" in ds.attrs:
            import json

            all_provenance = ds.attrs["provenance_dict"]
            if "source_elevation" in ds:
                present_ids = {int(v) for v in np.unique(ds["source_elevation"].values) if v != 0}
                filtered_provenance = {k: v for k, v in all_provenance.items() if int(k) in present_ids}
            else:
                filtered_provenance = dict(all_provenance)
            meta_path = tile_dir / f"{y}_{param_hash}_meta.json"
            with open(meta_path, "w") as f:
                json.dump(filtered_provenance, f)

        # Write source_elevation NPZ sidecar for pixel-level source lookups
        if format == "png" and "source_elevation" in ds:
            src_npz_path = tile_dir / f"{y}_{param_hash}_src.npz"
            if not src_npz_path.exists():
                src_arr = ds["source_elevation"].values
                # Strip padding applied for hillshade rendering so array aligns with the 512x512 PNG
                if pad_px > 0 and src_arr.ndim == 2 and src_arr.shape[0] == tile_size + 2 * pad_px:
                    src_arr = src_arr[pad_px:-pad_px, pad_px:-pad_px]
                buf = BytesIO()
                np.savez_compressed(buf, source_elevation=src_arr)
                with open(src_npz_path, "wb") as f:
                    f.write(buf.getvalue())

    except Exception as e:
        logger.warning(f"Failed to write cache file {tile_path}: {e}")

    logger.info(f"Tile {z}/{x}/{y} generated in {time.time() - start_time:.2f}s")
    return Response(content=data, media_type=media_type)


@app.get("/policy/reload")
async def reload_policy_cache() -> dict[str, object]:
    from topobathysim.policy.loader import clear_policy_cache

    clear_policy_cache()
    return {
        "status": "success",
        "message": "Policy cache cleared",
    }


@app.post("/cache/clear")
@app.get("/cache/clear")
async def clear_cache(type: str = "output") -> dict[str, object]:
    import shutil

    # Default cache location
    cache_root = get_cache_root()
    deleted = []

    try:
        if type in ["output", "all"]:
            tiles_dir = cache_root / "tiles"
            if tiles_dir.exists():
                shutil.rmtree(tiles_dir)
                deleted.append("tiles")

            fused_dir = cache_root / "fused"
            if fused_dir.exists():
                shutil.rmtree(fused_dir)
                deleted.append("fused")

            fused_zarr = cache_root / "fused_zarr"
            if fused_zarr.exists():
                shutil.rmtree(fused_zarr)
                deleted.append("fused_zarr")

        if type in ["source", "all"]:
            # Clear provider caches
            for subdir in [
                "usgs_3dep",
                "noaa_bluetopo",
                "noaa_topobathy",
                "usgs_lidar",
                "ncei_bag",
                "ncei_cudem",
                "gebco_2025",
            ]:
                p = cache_root / subdir
                if p.exists():
                    shutil.rmtree(p)
                    deleted.append(subdir)

        if type in ["orphaned", "cleanup", "all"]:
            count_locks = 0
            count_tmps = 0
            try:
                for root, _, files in os.walk(cache_root):
                    for file in files:
                        if file.endswith(".lock") or ".tmp" in file:
                            try:
                                fp = Path(root) / file
                                fp.unlink(missing_ok=True)
                                if file.endswith(".lock"):
                                    count_locks += 1
                                else:
                                    count_tmps += 1
                            except Exception as e:
                                logger.warning(f"Failed to delete orphaned file {file}: {e}")

                if count_locks > 0 or count_tmps > 0:
                    deleted.append(f"orphaned_locks({count_locks})")
                    deleted.append(f"orphaned_tmps({count_tmps})")
            except Exception as e:
                logger.warning(f"Failed to cleanup orphaned files: {e}")

        return {
            "status": "success",
            "cleared": deleted,
            "message": f"Cleared cache types: {type}",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


def _hydrate_subprocess(
    job_id: str,
    policy_input: str,
    bbox: tuple[float, float, float, float],
    resolution: float,
    max_workers: int = 2,
    target_zoom: int | None = None,
) -> None:
    """
    Entry point for the hydration subprocess.

    Runs in a separate process (multiprocessing.Process) so that:
    - OOM kills this process, not the web server
    - Progress is written atomically to a JSON file readable by any worker
    - No shared mutable state — all communication via filesystem
    """
    import signal

    # Ignore SIGINT in subprocess — let the parent handle Ctrl+C
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Set a memory ceiling so the subprocess raises MemoryError instead of
    # being OOM-killed. Default: 80% of system RAM. Cells that exceed the
    # limit are marked as failed and the job continues with the next cell.
    try:
        import resource

        mem_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        limit = int(mem_bytes * 0.80)
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        logger.info(f"Hydration subprocess memory ceiling: {limit / 1e9:.1f} GB")
    except Exception as e:
        logger.warning(f"Could not set memory ceiling: {e}")

    state = job_state.read_state(job_id)
    if state is None:
        return
    state["status"] = "running"
    state["pid"] = os.getpid()
    # Start time lets readers tell this worker apart from a later process that
    # reuses the PID (e.g. after a container restart).
    state["pid_start_ticks"] = job_state.process_start_ticks(os.getpid())
    job_state.write_state(job_id, state)

    def _on_progress(stats: dict[str, int]) -> None:
        s = job_state.read_state(job_id)
        if s is None:
            return
        s["total_cells"] = stats["total"]
        s["processed_cells"] = stats["processed"]
        s["cached_cells"] = stats["cached"]
        s["failed_cells"] = stats["failed"]
        job_state.write_state(job_id, s)

    try:
        logger.info(
            f"Hydration subprocess [{job_id}] started (PID {os.getpid()}) "
            f"for {bbox} @ {resolution}m with {max_workers} workers"
        )
        _hydrate_start = time.time()
        stats = hydrate(
            policy_input,
            bbox,
            resolution=resolution,
            max_workers=max_workers,
            on_progress=_on_progress,
            target_zoom=target_zoom,
        )
        _hydrate_elapsed = time.time() - _hydrate_start
        state = job_state.read_state(job_id) or state
        state["status"] = "completed"
        state["stats"] = stats
        job_state.write_state(job_id, state)
        logger.info(f"Hydration [{job_id}] finished in {_hydrate_elapsed:.1f}s: {stats}")
    except Exception as e:
        state = job_state.read_state(job_id) or state
        state["status"] = "failed"
        state["error"] = str(e)
        job_state.write_state(job_id, state)
        logger.error(f"Hydration [{job_id}] failed: {e}", exc_info=True)


# Keep track of spawned subprocesses so we can join them on shutdown
_HYDRATE_PROCESSES: dict[str, "multiprocessing.Process"] = {}


@app.post("/hydrate")
async def trigger_hydrate(
    request: HydrateRequest,
    policy_path: Annotated[Path, Depends(get_policy_path)],
) -> dict[str, Any]:
    """
    Trigger a background hydration process for the specified bounding box and resolution.
    Spawns a dedicated subprocess so OOM cannot kill the web server.
    """
    # Prune old job files
    job_state.list_jobs(max_age_hours=24)

    start_time = datetime.now(timezone.utc)
    job_id = str(uuid.uuid4())

    # Handle custom policy or default to loaded policy
    policy_input = request.policy_yaml if request.policy_yaml else str(policy_path)

    # Compute effective resolution: zoom-based (latitude-aware) or explicit meters
    if request.zoom is not None:
        # Match the tile endpoint's resolution formula:
        # res = 156543.03 * cos(center_lat) / 2^z
        center_lat = (request.bbox[1] + request.bbox[3]) / 2.0
        effective_resolution = round(156543.03 * math.cos(math.radians(center_lat)) / (2.0**request.zoom), 8)
        logger.info(f"Zoom {request.zoom} at lat {center_lat:.4f} -> resolution {effective_resolution:.4f}m")
    else:
        effective_resolution = request.resolution

    try:
        _policy = load_policy_from_str(policy_input) if request.policy_yaml else load_policy(policy_input)
        _phash = hash_policy(_policy.model_dump())
        logger.info(
            f"POST /hydrate policy={getattr(_policy, 'name', policy_input)} "
            f"hash={_phash[:8]} bbox={request.bbox} res={effective_resolution}m"
        )
    except Exception as _log_exc:
        logger.debug(f"POST /hydrate policy hash logging failed: {_log_exc}")

    # Write initial state to disk before spawning
    initial_state: dict[str, Any] = {
        "id": job_id,
        "status": "pending",
        "bbox": list(request.bbox),
        "resolution": effective_resolution,
        "submitted_at": start_time.isoformat(),
        "total_cells": 0,
        "processed_cells": 0,
        "cached_cells": 0,
        "failed_cells": 0,
    }
    if request.zoom is not None:
        initial_state["zoom"] = request.zoom
    job_state.write_state(job_id, initial_state)

    # Spawn subprocess
    proc = multiprocessing.Process(
        target=_hydrate_subprocess,
        args=(
            job_id,
            policy_input,
            request.bbox,
            effective_resolution,
            request.max_workers,
            request.zoom,
        ),
        daemon=True,
    )
    proc.start()

    # Track for cleanup, evict dead processes
    for old_id in list(_HYDRATE_PROCESSES):
        if not _HYDRATE_PROCESSES[old_id].is_alive():
            _HYDRATE_PROCESSES[old_id].join(timeout=0)
            del _HYDRATE_PROCESSES[old_id]
    _HYDRATE_PROCESSES[job_id] = proc

    logger.info(f"Spawned hydration subprocess PID {proc.pid} for job {job_id}")

    return {
        "status": "accepted",
        "message": "Hydration task started in subprocess",
        "job_id": job_id,
        "bbox": list(request.bbox),
        "resolution": request.resolution,
        "timestamp": start_time.isoformat(),
    }


@app.get("/hydrate/{job_id}")
async def get_hydration_status(job_id: str) -> dict[str, Any]:
    """Get the status of a hydration job (reads from persistent JSON state file)."""
    state = job_state.read_state(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return state


@app.delete("/hydrate/{job_id}")
async def cancel_hydration_job(job_id: str) -> dict[str, Any]:
    """Cancel a running hydration job by killing its subprocess."""
    state = job_state.read_state(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if state.get("status") not in ("running", "pending"):
        return {"status": state.get("status"), "message": "Job is not running"}

    # Try in-memory process handle first (same server instance)
    proc = _HYDRATE_PROCESSES.get(job_id)
    if proc and proc.is_alive():
        proc.terminate()
        proc.join(timeout=3)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=2)
        del _HYDRATE_PROCESSES[job_id]
    else:
        # Fall back to PID from state (e.g. after server restart)
        pid = state.get("pid")
        if pid:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, 15)  # SIGTERM

    state["status"] = "cancelled"
    state["error"] = "Cancelled by user"
    job_state.write_state(job_id, state)
    logger.info(f"Hydration job {job_id} cancelled")
    return {"status": "cancelled", "job_id": job_id}


@app.websocket("/hydrate/{job_id}/ws")
async def hydrate_ws(websocket: WebSocket, job_id: str) -> None:
    """
    WebSocket endpoint for live hydration progress.

    Streams JSON state updates every second until the job completes or fails.
    Client receives the same payload as GET /hydrate/{job_id} but in real time.
    """
    import asyncio

    await websocket.accept()
    try:
        prev_snapshot = ""
        while True:
            state = job_state.read_state(job_id)
            if state is None:
                await websocket.send_json({"error": "Job not found", "status": "failed"})
                break

            # Only send when state has changed to reduce noise
            snapshot = json.dumps(state, sort_keys=True)
            if snapshot != prev_snapshot:
                await websocket.send_json(state)
                prev_snapshot = snapshot

            if state.get("status") in ("completed", "failed", "cancelled"):
                break

            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning(f"WebSocket error for job {job_id}: {e}")


@app.get("/hydrate")
async def list_hydration_jobs() -> list[dict[str, Any]]:
    """List recent hydration jobs."""
    return job_state.list_jobs(max_age_hours=24)


@app.get("/api/v1/quality/report", response_model=DEMQualityReport)
async def get_quality_report(
    north: float,
    south: float,
    west: float,
    east: float,
    policy_path: Annotated[Path, Depends(get_policy_path)],
    resolution: float = Query(30.0, description="Exact grid resolution in meters to evaluate fragility at"),
) -> DEMQualityReport:
    """
    Evaluates the DEM for numerical and physical stability exactly at the requested simulation resolution.
    Returns JSON stats and an embedded GeoJSON FeatureCollection of anomalies
    (cliffs, pits, VDatum mismatches).
    """
    try:
        # We need to fetch and fuse the exact area at metric resolution
        # Avoid using map tile logic so we don't smear out the precise gradients
        bbox = (west, south, east, north)

        # Run fusion (returns aligned arrays)
        ds = run(policy_path, bbox, resolution)

        if "elevation" not in ds:
            raise HTTPException(status_code=404, detail="No elevation data generated for region")

        elevation = ds["elevation"]
        source_map = ds.get("source_id")  # Assuming source_id is preserved if generated

        report_data = calculate_spatial_stats(
            elevation=elevation,
            source_map=source_map,
            resolution_m=resolution,
            steep_threshold_m=3.0,
            pit_ratio_threshold=3.0,
        )

        return DEMQualityReport(**report_data)

    except Exception as e:
        logger.error(f"Quality reporting failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/analyze")
async def analyze_coverage(
    request: HydrateRequest,
    policy_path: Annotated[Path, Depends(get_policy_path)],
) -> dict[str, Any]:
    """Dry-run coverage analyzer for a bounding box."""
    from topobathysim.providers.registry import registry

    try:
        policy_input = request.policy_yaml
        if not policy_input:
            if request.policy_name:
                # If they passed just 'wlis', find it next to the default policy
                policy_dir = policy_path.parent
                cand = policy_dir / request.policy_name
                if not cand.exists() and not cand.suffix:
                    cand = policy_dir / f"{request.policy_name}.yaml"
                policy_input = str(cand) if cand.exists() else str(policy_path)
            else:
                policy_input = str(policy_path)

        policy = load_policy_from_str(policy_input) if request.policy_yaml else load_policy(policy_input)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid policy: {e}") from e

    results = []

    for step in policy.variables[0].steps:
        provider_cls = registry.get_provider_class(step.provider)
        provider = provider_cls()

        # Since we can't easily fetch dry-run without an `inventory()` method,
        # implement a pseudo-inventory loop here.
        items = []
        try:
            if step.provider == "usgs_3dep":
                from topobathysim.providers.base import matches_pattern

                raw_items = provider._query_stac_api(request.bbox, collection_id="3dep-seamless") or []  # type: ignore[attr-defined]
                for i in raw_items:
                    name = Path(i["href"].split("?")[0]).name
                    included = True
                    reason = ""

                    if (
                        step.filter
                        and step.filter.include_patterns
                        and not matches_pattern(name, step.filter.include_patterns)
                    ):
                        included = False
                        reason = "Not in include_patterns"
                    if (
                        step.filter
                        and included
                        and step.filter.exclude_patterns
                        and matches_pattern(name, step.filter.exclude_patterns)
                    ):
                        included = False
                        reason = "Matches exclude_pattern"
                    # min_elevation filtering happens at fetch time; warn if set
                    if (
                        step.filter
                        and included
                        and step.filter.min_elevation is not None
                        and step.filter.min_elevation > 0
                    ):
                        reason += " (Warn: Filtered dynamically if < min_elevation)"

                    # Add helpful source links
                    url = "https://planetarycomputer.microsoft.com/dataset/3dep-seamless"
                    items.append({"name": name, "included": included, "reason": reason.strip(), "url": url})

            elif step.provider == "noaa_bluetopo":
                from topobathysim.providers.base import matches_pattern

                w, s, east, n = provider._normalize_bbox(request.bbox)  # type: ignore[attr-defined]

                seen_surveys: set[str] = set()

                # If this is a very small bbox (like a point query from the UI),
                # use the exact coordinate to resolve the specific pixel survey ID
                if (east - w) < 0.01 and (n - s) < 0.01:
                    lon, lat = (w + east) / 2.0, (s + n) / 2.0
                    survey_id = provider.get_source_survey_id(lat, lon)  # type: ignore[attr-defined]
                    tiles: list[str] = []  # Bypass full tile extraction
                    if survey_id:
                        name = survey_id
                        seen_surveys.add(name)
                        included = True
                        reason = ""

                        if (
                            step.filter
                            and step.filter.include_patterns
                            and not matches_pattern(survey_id, step.filter.include_patterns)
                        ):
                            included = False
                            reason = "Not in include_patterns"
                        if (
                            step.filter
                            and step.filter.exclude_patterns
                            and matches_pattern(survey_id, step.filter.exclude_patterns)
                        ):
                            included = False
                            reason = "Matched exclude_patterns"

                        url = f"https://www.ncei.noaa.gov/maps/bathymetry/?search={survey_id.split('.')[0]}"
                        items.append({"name": name, "included": included, "reason": reason, "url": url})
                else:
                    tiles = provider.resolve_tiles_in_bbox(w, s, east, n)  # type: ignore[attr-defined]

                for tile_id in tiles:
                    # Extract inner surveys from cached RAT sidecars
                    surveys = provider.get_all_survey_ids(tile_id)  # type: ignore[attr-defined]

                    if not surveys:
                        # Fallback if no sidecar available
                        name = tile_id
                        if name in seen_surveys:
                            continue
                        seen_surveys.add(name)

                        included = True
                        reason = ""
                        if step.filter and (step.filter.include_patterns or step.filter.exclude_patterns):
                            reason = (
                                "(Dynamic pattern filtering will be applied"
                                " sequentially to internal survey sources)"
                            )
                        url = "https://nauticalcharts.noaa.gov/data/bluetopo.html"
                        items.append({"name": name, "included": included, "reason": reason, "url": url})
                        continue

                    # Process each internal survey independently for the UI
                    for survey_id in surveys:
                        name = survey_id
                        if name in seen_surveys:
                            continue
                        seen_surveys.add(name)

                        included = True
                        reason = ""

                        # Filter explicitly exactly as fusion does
                        if (
                            step.filter
                            and step.filter.include_patterns
                            and not matches_pattern(survey_id, step.filter.include_patterns)
                        ):
                            included = False
                            reason = "Not in include_patterns"
                        if (
                            step.filter
                            and step.filter.exclude_patterns
                            and matches_pattern(survey_id, step.filter.exclude_patterns)
                        ):
                            included = False
                            reason = "Matched exclude_patterns"

                        url = f"https://www.ncei.noaa.gov/maps/bathymetry/?search={survey_id.split('.')[0]}"
                        items.append({"name": name, "included": included, "reason": reason, "url": url})

            elif step.provider == "noaa_topobathy":
                from topobathysim.providers.base import matches_pattern

                w, s, east, n = provider._normalize_bbox(request.bbox)  # type: ignore[attr-defined]
                projects = provider.find_projects_by_box(w, s, east, n)  # type: ignore[attr-defined]
                if not projects:
                    projects = []

                if (east - w) < 0.01 and (n - s) < 0.01:
                    valid_projects = []
                    for p in projects:
                        name = str(p.get("project_id", "Unknown")) if isinstance(p, dict) else str(p)
                        # Skip heavy tile fetch if the filter excludes it anyway
                        if (
                            step.filter
                            and hasattr(step.filter, "project_id")
                            and step.filter.project_id
                            and name not in step.filter.project_id
                        ):
                            continue

                        # Pass the project explicitly: the provider is a process-wide
                        # singleton shared with concurrent requests.
                        provider._load_tile_index(name)  # type: ignore[attr-defined]
                        tiles = provider.resolve_tiles_in_bbox(  # type: ignore[attr-defined]
                            w, s, east, n, project_id=name
                        )
                        has_data = False
                        for t in tiles:
                            try:
                                da = provider.fetch_tile(  # type: ignore[attr-defined]
                                    t, request.bbox, project_id=name
                                )
                                if da is not None and int(da.notnull().sum()) > 0:
                                    has_data = True
                                    break
                            except Exception:
                                pass
                        if has_data:
                            valid_projects.append(p)
                    projects = valid_projects

                for p in projects:
                    name = str(p.get("project_id", "Unknown")) if isinstance(p, dict) else str(p)
                    included = True
                    reason = ""

                    if (
                        step.filter
                        and hasattr(step.filter, "project_id")
                        and step.filter.project_id
                        and name not in step.filter.project_id
                    ):
                        included = False
                        reason = "Not in project_id list"

                    # Generate URL for NOAA Topobathy Lidar Viewer
                    url = (
                        f"https://coast.noaa.gov/dataviewer/#/lidar/search/where:ID={name}"
                        if name.isdigit()
                        else "https://coast.noaa.gov/dataviewer/#/lidar/search/"
                    )
                    items.append(
                        {"name": f"Project: {name}", "included": included, "reason": reason, "url": url}
                    )

            elif step.provider == "ncei_bag":
                from topobathysim.providers.base import matches_pattern
                from topobathysim.providers.ncei_bag import BAGDiscovery

                w, s, east, n = provider._normalize_bbox(request.bbox)  # type: ignore[attr-defined]
                urls = BAGDiscovery.find_bags_by_bbox(w, s, east, n)
                if not urls:
                    urls = []

                if (east - w) < 0.01 and (n - s) < 0.01:
                    valid_urls = []
                    for target_url in urls:
                        name = Path(target_url.split("?")[0]).name

                        # Filter before heavy processing
                        if step.filter:
                            included_pre = True
                            if step.filter.include_patterns and not matches_pattern(
                                name, step.filter.include_patterns
                            ):
                                included_pre = False
                            if (
                                included_pre
                                and step.filter.exclude_patterns
                                and matches_pattern(name, step.filter.exclude_patterns)
                            ):
                                included_pre = False
                            if not included_pre:
                                continue

                        has_data = False
                        try:
                            da = provider.fetch_dataset(target_url, request.bbox)  # type: ignore[attr-defined]
                            if da is not None and int(da.notnull().sum()) > 0:
                                has_data = True
                        except Exception:
                            pass
                        if has_data:
                            valid_urls.append(target_url)
                    urls = valid_urls

                for target_url in urls:
                    name = Path(target_url.split("?")[0]).name
                    included = True
                    reason = ""

                    if (
                        step.filter
                        and step.filter.include_patterns
                        and not matches_pattern(name, step.filter.include_patterns)
                    ):
                        included = False
                        reason = "Not in include_patterns"
                    if (
                        step.filter
                        and included
                        and step.filter.exclude_patterns
                        and matches_pattern(name, step.filter.exclude_patterns)
                    ):
                        included = False
                        reason = "Matches exclude_pattern"

                    import re

                    survey_matches = re.search(r"^[A-Z][0-9]+", name)
                    url = (
                        f"https://www.ncei.noaa.gov/maps/bathymetry/?search={survey_matches.group(0)}"
                        if survey_matches
                        else "https://www.ncei.noaa.gov/maps/bathymetry/"
                    )
                    items.append({"name": name, "included": included, "reason": reason, "url": url})

        except Exception as e:
            items.append({"name": "Error fetching inventory", "included": False, "reason": str(e)})

        results.append(
            {
                "provider": step.provider,
                "step_index": policy.variables[0].steps.index(step),
                "items": items,
            }
        )

    return {"providers": results, "policy_yaml": yaml.dump(policy.model_dump(), sort_keys=False)}
