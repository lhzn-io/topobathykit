"""
topobathykit Cache Manager
==========================

Unified CLI for inspecting, verifying, and purging the topobathykit cache hierarchy.
Organises operations by tier — from the safest (output tiles) to the most destructive
(raw downloaded source files).

When attached to a TTY, an interactive arrow-key menu is presented (powered by
questionary + rich).  In non-TTY environments (CI, piped) the tool falls back to
plain-text output with the same --check / --purge / --clean flags.

Cache Tiers
-----------

+---+-------------------------------+------------------------------------------+-------------------------+
| # | Name                          | Paths                                    | Rebuild cost            |
+===+===============================+==========================================+=========================+
| 1 | Output tiles & metadata       | tiles/  (PNGs, _meta.json, _src.npz)     | Seconds (re-render)     |
+---+-------------------------------+------------------------------------------+-------------------------+
| 2 | Fused zarr                    | fused_zarr/                              | Minutes (re-fuse)       |
+---+-------------------------------+------------------------------------------+-------------------------+
| 3 | Provider zarr                 | {provider}/zarr/*.zarr                   | Minutes-hours           |
+---+-------------------------------+------------------------------------------+-------------------------+
| 4 | Discovery caches              | 5 JSON/GeoJSON metadata files + sqlite   | Seconds-minutes         |
+---+-------------------------------+------------------------------------------+-------------------------+
| 5 | Raw source files              | *.bag, *.tiff, *.tif, *.laz             | Hours (full re-download)|
+---+-------------------------------+------------------------------------------+-------------------------+

Usage
-----

.. code-block:: bash

    python -m topobathykit.scripts.cache_manager            # interactive menu (TTY)
    python -m topobathykit.scripts.cache_manager --check    # integrity audit
    python -m topobathykit.scripts.cache_manager --clean    # delete corrupt/stale files
    python -m topobathykit.scripts.cache_manager --purge 1  # purge output tiles
    python -m topobathykit.scripts.cache_manager --purge 4  # purge discovery caches
    python -m topobathykit.scripts.cache_manager --purge all --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Generator, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any

import laspy
import rasterio
import xarray as xr
from rasterio.errors import RasterioIOError
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
from topobathykit.config import get_cache_root

CACHE_ROOT = get_cache_root()

PROVIDERS = ["ncei_bag", "usgs_lidar", "noaa_bluetopo", "noaa_topobathy", "ncei_cudem", "usgs_3dep"]

# Providers that download raw source files locally before rasterising to zarr.
# Tier 5 only applies to these.
SOURCE_PATTERNS: dict[str, list[str]] = {
    "ncei_bag": ["*.bag"],
    "usgs_lidar": ["*.laz"],
    "ncei_cudem": ["*.tif"],
}

# Providers that stream from cloud COGs/S3 directly into zarr — no raw files
# ever land on disk.  Their tier-3 zarr IS their primary source cache.
# Purging tier 3 for these providers means re-streaming from the network.
_STREAMING_PROVIDERS: frozenset[str] = frozenset({"noaa_bluetopo", "noaa_topobathy", "usgs_3dep"})

_IS_TTY = sys.stdin.isatty() and sys.stdout.isatty()

console = Console(highlight=False)
err_console = Console(stderr=True, highlight=False)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("cache_manager")

# ---------------------------------------------------------------------------
# Pager helper — captures rich output and pipes to `less -R`
# ---------------------------------------------------------------------------


@contextmanager
def _rich_pager() -> Generator[Console, None, None]:
    """
    Capture rich output written to the yielded Console and display it through
    ``less -R``.  Falls back to direct stdout if less is unavailable.

    Usage::

        with _rich_pager() as pager:
            pager.print("[bold]hello[/]")
    """
    buf = StringIO()
    width = console.size.width if _IS_TTY else 120
    pager_console = Console(file=buf, highlight=False, force_terminal=True, width=width)
    yield pager_console
    output = buf.getvalue()
    if not output.strip():
        return
    try:
        # -R  interpret ANSI colour sequences
        # -F  quit immediately if content fits on one screen
        # -X  don't clear screen on exit (content stays visible after q)
        proc = subprocess.Popen(["less", "-R", "-F", "-X"], stdin=subprocess.PIPE)
        proc.stdin.write(output.encode("utf-8", errors="replace"))  # type: ignore[union-attr]
        proc.stdin.close()  # type: ignore[union-attr]
        proc.wait()
    except (FileNotFoundError, OSError):
        sys.stdout.write(output)
    except BrokenPipeError:
        pass  # user pressed q before all content was written


# ---------------------------------------------------------------------------
# Questionary style
# ---------------------------------------------------------------------------

_QSTYLE = None


def _get_qstyle() -> Any:
    global _QSTYLE
    if _QSTYLE is None:
        from questionary import Style

        _QSTYLE = Style(
            [
                ("qmark", "fg:#00d7af bold"),
                ("question", "bold"),
                ("answer", "fg:#00d7af bold"),
                ("pointer", "fg:#00d7af bold"),
                ("highlighted", "fg:#ffffff bold"),
                ("selected", "fg:#00d7af"),
                ("separator", "fg:#444444"),
                ("instruction", "fg:#666666 italic"),
                ("text", ""),
                ("disabled", "fg:#444444 italic"),
            ]
        )
    return _QSTYLE


# ---------------------------------------------------------------------------
# Integrity check helpers
# ---------------------------------------------------------------------------


@contextmanager
def suppress_fd_stderr() -> Iterator[None]:
    """Redirect system-level stderr to /dev/null to silence C-level GDAL noise."""
    try:
        if not hasattr(sys.stderr, "fileno"):
            yield
            return
        stderr_fd = sys.stderr.fileno()
        with open(os.devnull, "w") as devnull:
            saved = os.dup(stderr_fd)
            try:
                os.dup2(devnull.fileno(), stderr_fd)
                yield
            finally:
                os.dup2(saved, stderr_fd)
                os.close(saved)
    except Exception:
        yield


def verify_raster_integrity(file_path: Path) -> bool:
    try:
        with suppress_fd_stderr(), rasterio.Env(CPL_LOG_ERRORS=False), rasterio.open(file_path) as src:
            w, h = src.width, src.height
            if w > 0 and h > 0:
                src.read(1, window=rasterio.windows.Window(w - 1, h - 1, 1, 1))
        return True
    except (RasterioIOError, Exception):
        return False


def verify_laz_integrity(file_path: Path) -> bool:
    try:
        with laspy.open(file_path) as fh:
            _ = fh.header
        return True
    except Exception:
        return False


def verify_zarr_integrity(zarr_path: Path) -> bool:
    try:
        with xr.open_dataarray(zarr_path, engine="zarr", chunks="auto", consolidated=False) as da:
            _ = da.shape
        return True
    except Exception:
        try:
            with xr.open_dataset(zarr_path, engine="zarr", chunks="auto", consolidated=False) as ds:
                _ = ds.dims
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Size / age helpers
# ---------------------------------------------------------------------------


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            with suppress(OSError):
                total += f.stat().st_size
    return total


def _fmt_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024**2:
        return f"{n / 1024:.1f} KB"
    if n < 1024**3:
        return f"{n / 1024**2:.1f} MB"
    return f"{n / 1024**3:.2f} GB"


def _fmt_age(secs: float) -> str:
    if secs < 60:
        return f"{int(secs)}s ago"
    if secs < 3600:
        return f"{int(secs // 60)}m ago"
    if secs < 86400:
        return f"{secs / 3600:.1f}h ago"
    return f"{secs / 86400:.1f}d ago"


def _fused_zarr_extents(
    path: Path,
) -> tuple[float, float, float, float, float, str | None, int | None, float | None] | None:
    """Return (x_min, x_max, y_min, y_max, resolution, resolution_origin, target_zoom, resolution_m)
    from a zarr store's coordinate arrays and attrs."""
    try:
        import zarr  # type: ignore[import]

        z = zarr.open_group(str(path), mode="r")
        x_arr = z["x"]
        y_arr = z["y"]
        x0, x1 = float(x_arr[0]), float(x_arr[-1])  # type: ignore[index, arg-type]
        y0, y1 = float(y_arr[0]), float(y_arr[-1])  # type: ignore[index, arg-type]

        # Calculate resolution (meters/pixel or degrees/pixel)
        count = x_arr.shape[0]  # type: ignore[union-attr]
        width = abs(x1 - x0)
        # Assuming cell centers, span is width. Res is width / (count-1)
        # If count=1, we can't really know, assume 0 or handle separately
        res = getattr(x_arr, "attrs", {}).get("resolution")
        if res is None:
            res = width / (count - 1) if count > 1 else 0.0

        # Read provenance attrs (may not exist on legacy zarr files)
        root_attrs = z.attrs
        raw_origin = root_attrs.get("resolution_origin")
        resolution_origin: str | None = str(raw_origin) if raw_origin is not None else None
        raw_tz = root_attrs.get("target_zoom")
        target_zoom_val: int | None = int(str(raw_tz)) if raw_tz is not None else None
        raw_rm = root_attrs.get("resolution_m")
        resolution_m_val: float | None = float(str(raw_rm)) if raw_rm is not None else None

        return (
            min(x0, x1),
            max(x0, x1),
            min(y0, y1),
            max(y0, y1),
            res,
            resolution_origin,
            target_zoom_val,
            resolution_m_val,
        )
    except Exception:
        return None


def _proj_to_zoom_and_lonlat(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    resolution: float = 0.0,
) -> tuple[int | float, float, float, float, float]:
    """Return (zoom, lon_min, lon_max, lat_min, lat_max) from coordinate extents.

    Detects Web Mercator (|x| > 360) vs. geographic degrees automatically.
    Returns zoom as float based on resolution if provided, else fallback to width.
    """
    import math

    if abs(x_max) > 360:
        # Web Mercator (EPSG:3857) - convert to WGS84
        earth_r = 6378137.0
        lon_min = x_min / 20037508.34 * 180.0
        lon_max = x_max / 20037508.34 * 180.0
        lat_min = math.degrees(2 * math.atan(math.exp(y_min / earth_r)) - math.pi / 2)
        lat_max = math.degrees(2 * math.atan(math.exp(y_max / earth_r)) - math.pi / 2)

        # Calculate zoom from RESOLUTION (meters/pixel), not extent width
        # Base resolution at z=0 is EarthCircumference / 256 ~= 156543.03392
        if resolution > 0:
            flt_z = math.log2(156543.03392 / resolution)
            z: int | float = round(flt_z)
            if abs(flt_z - z) > 0.05:
                z = round(flt_z, 2)
        else:
            # Fallback (legacy/single-tile specific logic)
            x_span = x_max - x_min
            z = round(math.log2(40075016.686 / x_span)) if x_span > 0 else 0

    else:
        # Already in degrees
        lon_min, lon_max, lat_min, lat_max = x_min, x_max, y_min, y_max
        if resolution > 0:
            # Base resolution at z=0 is 360 / 256 = 1.40625 deg/px
            flt_z = math.log2(1.40625 / resolution)
            z = round(flt_z)
            if abs(flt_z - z) > 0.05:
                z = round(flt_z, 2)
        else:
            lon_span = x_max - x_min
            z = round(math.log2(360.0 / lon_span)) if lon_span > 0 else 0

    return max(0, min(24, z)), lon_min, lon_max, lat_min, lat_max


# ---------------------------------------------------------------------------
# Cache tier definitions
# ---------------------------------------------------------------------------


@dataclass
class CacheTier:
    number: int
    name: str
    short_desc: str
    long_desc: str
    purge_reason: str
    warning: bool = False

    def purge_paths(self) -> list[Path]:
        paths: list[Path] = []

        if self.number == 1:
            tiles_root = CACHE_ROOT / "tiles"
            if tiles_root.exists():
                paths.extend(tiles_root.iterdir())

        elif self.number == 2:
            fused_root = CACHE_ROOT / "fused_zarr"
            if fused_root.exists():
                paths.extend(fused_root.iterdir())

        elif self.number == 3:
            for provider in PROVIDERS:
                zarr_dir = CACHE_ROOT / provider / "zarr"
                if zarr_dir.exists():
                    paths.extend(zarr_dir.glob("*.zarr"))
                prov_dir = CACHE_ROOT / provider
                if prov_dir.exists():
                    paths.extend(p for p in prov_dir.glob("*.zarr"))

        elif self.number == 4:
            for p in [
                CACHE_ROOT / "ncei_bag" / "discovery_cache.json",
                CACHE_ROOT / "noaa_bluetopo" / "tile_url_cache.json",
                CACHE_ROOT / "noaa_bluetopo" / "BlueTopo_Tile_Scheme.gpkg",
                CACHE_ROOT / "noaa_bluetopo" / "sidecars",
                CACHE_ROOT / "metadata" / "ncei_bag_redirects.json",
                CACHE_ROOT / "metadata" / "noaa_coastal_lidar.json",
                CACHE_ROOT / "metadata" / "noaa_project_extents.geojson",
                CACHE_ROOT / "vdatum.sqlite",
            ]:
                if p.exists():
                    paths.append(p)
            # NOAA Topobathy per-project tile-index spatial files (*.zip, *.gpkg)
            topo_dir = CACHE_ROOT / "noaa_topobathy"
            if topo_dir.exists():
                for pat in ("*.zip", "*.gpkg"):
                    paths.extend(topo_dir.glob(pat))

        elif self.number == 5:
            for provider, patterns in SOURCE_PATTERNS.items():
                prov_dir = CACHE_ROOT / provider
                if prov_dir.exists():
                    for pat in patterns:
                        paths.extend(prov_dir.rglob(pat))

        return paths

    def size_bytes(self) -> int:
        total = 0
        for p in self.purge_paths():
            total += _dir_size_bytes(p) if p.is_dir() else (p.stat().st_size if p.exists() else 0)
        return total

    def item_count(self) -> int:
        return len(self.purge_paths())


TIERS: list[CacheTier] = [
    CacheTier(
        number=1,
        name="Output tiles & metadata",
        short_desc="Rendered PNGs, NPY/NPZ, sidecars",
        long_desc=(
            "Visual PNG tiles (all styles), raw data tiles (.npy/.npz), and per-tile\n"
            "sidecar files (_meta.json provenance, _src.npz pixel source lookup).\n"
            "Stored under tiles/visual/, tiles/data/, tiles/raw/.\n\n"
            "Purge when: rendered tiles are stale, colormap/hillshade style changed,\n"
            "or you want to force a full visual re-render.\n\n"
            "Rebuild cost: seconds — the next tile request re-renders on demand."
        ),
        purge_reason="Rendered tiles are stale or styles have changed.",
        warning=False,
    ),
    CacheTier(
        number=2,
        name="Fused zarr",
        short_desc="Multi-provider composite zarr stores",
        long_desc=(
            "Zarr stores produced by the fusion pipeline combining data from multiple\n"
            "providers into a single composite grid. Stored under fused_zarr/.\n\n"
            "Purge when: the fusion policy (wlis.yaml) has changed, a provider's\n"
            "underlying zarr was updated, or you see unexpected fusion artifacts.\n\n"
            "Rebuild cost: minutes — re-fuses from provider zarr on next request."
        ),
        purge_reason="Fusion policy changed or provider zarr was updated.",
        warning=False,
    ),
    CacheTier(
        number=3,
        name="Provider zarr",
        short_desc="Per-provider rasterised zarr grids",
        long_desc=(
            "Zarr grids derived from each provider's raw source files (rasterised\n"
            "BAGs, CUDEM tiles, BlueTopo GeoTIFFs, lidar point clouds).\n"
            "Stored in {provider}/zarr/*.zarr.\n\n"
            "Purge when: a provider's raw source files have been updated/replaced, or\n"
            "you suspect corrupt rasterisation.\n\n"
            "Rebuild cost: minutes to hours — re-rasterises from raw files on next request."
        ),
        purge_reason="Raw source files were replaced or rasterisation is suspect.",
        warning=False,
    ),
    CacheTier(
        number=4,
        name="Metadata & Discovery caches",
        short_desc="JSON/GeoJSON/SQLite indices (bbox→URLs, API offsets)",
        long_desc=(
            "Six lightweight metadata databases recording URL coverage and API lookups.\n"
            "Populated on-demand; survive restarts for fully offline tile generation.\n\n"
            "  ncei_bag/discovery_cache.json        bbox → BAG download-URL list\n"
            "  noaa_bluetopo/tile_url_cache.json    tile_id → HTTPS asset URL (S3 glob)\n"
            "  metadata/ncei_bag_redirects.json     landing-page URL → .bag file URL\n"
            "  metadata/noaa_coastal_lidar.json     project ID → folder name\n"
            "  metadata/noaa_project_extents.geojson  project spatial bbox index\n"
            "  vdatum.sqlite                        Local sqlite WAL cache for NOAA API conversions\n\n"
            "Purge when: NCEI published new surveys, NOAA updated BlueTopo tile\n"
            "filenames (cached URLs will 404), or a new topobathy lidar project was added.\n\n"
            "Rebuild cost: seconds to minutes — re-queried from network on next request.\n"
            "If provider zarr is still cached, no network calls are needed."
        ),
        purge_reason="New surveys published; stale discovery data causing 404s.",
        warning=False,
    ),
    CacheTier(
        number=5,
        name="Raw source files",
        short_desc="Downloaded BAG, TIFF, LAZ source files",
        long_desc=(
            "Original survey files downloaded from NCEI, NOAA, and USGS:\n\n"
            "  ncei_bag/      *.bag   — multibeam bathymetry surveys\n"
            "  noaa_bluetopo/ *.tiff  — BlueTopo GeoTIFF tiles\n"
            "  usgs_lidar/    *.laz   — 3DEP lidar point clouds\n"
            "  ncei_cudem/    *.tif   — CUDEM merged elevation tiles\n"
            "  noaa_topobathy/*.tif   — Topobathy lidar rasters\n"
            "  usgs_3dep/     *.tif   — 3DEP 10m DEM tiles\n\n"
            "Purge when: reclaiming disk space or a file is known to be corrupt\n"
            "and must be re-fetched from the source.\n\n"
            "Rebuild cost: hours — full re-download from NCEI/NOAA/USGS APIs.\n\n"
            "WARNING: Deleting raw files means the next tile request must re-download\n"
            "all surveys that cover the requested area. This can be very slow."
        ),
        purge_reason="Reclaim disk space or force re-download of corrupt files.",
        warning=True,
    ),
]

# ---------------------------------------------------------------------------
# Metadata & Discovery cache file registry
# ---------------------------------------------------------------------------

_DISCOVERY_FILES: list[tuple[str, Path]] = [
    ("ncei_bag/discovery_cache.json", CACHE_ROOT / "ncei_bag" / "discovery_cache.json"),
    ("noaa_bluetopo/tile_url_cache.json", CACHE_ROOT / "noaa_bluetopo" / "tile_url_cache.json"),
    ("metadata/ncei_bag_redirects.json", CACHE_ROOT / "metadata" / "ncei_bag_redirects.json"),
    ("metadata/noaa_coastal_lidar.json", CACHE_ROOT / "metadata" / "noaa_coastal_lidar.json"),
    ("metadata/noaa_project_extents.geojson", CACHE_ROOT / "metadata" / "noaa_project_extents.geojson"),
    ("vdatum.sqlite", CACHE_ROOT / "vdatum.sqlite"),
]

# ---------------------------------------------------------------------------
# Rich display — compact summary table (shown at top of TUI loop)
# ---------------------------------------------------------------------------


def _render_status_table() -> Table:
    """Compact 5-row tier summary used as the TUI header."""
    total = sum(t.size_bytes() for t in TIERS)
    table = Table(
        title=(f"[bold cyan]topobathykit Cache[/]  [dim]{CACHE_ROOT}  ·  {_fmt_size(total)} total[/]"),
        box=box.ROUNDED,
        show_header=True,
        header_style="bold dim",
        expand=False,
        min_width=72,
    )
    table.add_column("#", style="bold", width=3, justify="right")
    table.add_column("Tier", min_width=30)
    table.add_column("Summary", min_width=26, style="dim")
    table.add_column("Size", justify="right", width=10)
    table.add_column("Items", justify="right", width=6)

    for tier in TIERS:
        items = tier.item_count()
        sz = tier.size_bytes()
        num_style = "bold yellow" if tier.warning else "bold cyan"
        name_text = Text()
        if tier.warning:
            name_text.append("!  ", style="bold yellow")
        name_text.append(tier.name, style="bold yellow" if tier.warning else "")
        dot = Text("●  ", style="green" if items else "dim")
        size_t = Text(_fmt_size(sz) if items else "—", style="dim" if not items else "")
        items_t = Text(str(items) if items else "—", style="dim" if not items else "")
        table.add_row(
            Text(str(tier.number), style=num_style),
            Text.assemble(dot, name_text),
            tier.short_desc,
            size_t,
            items_t,
        )
    return table


# ---------------------------------------------------------------------------
# Rich display — detailed status report (shown in pager)
# ---------------------------------------------------------------------------


@dataclass
class _StatusData:
    """Pre-computed status data so directory walks happen before entering the pager."""

    total_bytes: int = 0
    # Tier 1 - tiles
    tile_subdirs: list[tuple[str, int, int]] = field(default_factory=list)  # (label, files, bytes)
    tile_meta_count: int = 0
    tile_npz_count: int = 0
    # Tier 2 - fused zarr (aggregate only; individual filenames are hashes and not useful)
    fused_count: int = 0
    fused_total_bytes: int = 0
    fused_newest: float = 0.0
    fused_oldest: float = float("inf")
    # policy_hash -> {"yaml_snippet": str/None, "count": int, "bytes": int, "by_zoom": dict}
    fused_policies: dict[str, dict] = field(default_factory=dict)
    # Tier 3 - provider zarr
    provider_zarr: list[tuple[str, int, int, float, float]] = field(
        default_factory=list
    )  # (prov, count, bytes, newest, oldest)
    # Tier 4 - discovery
    discovery: list[tuple[str, bool, int, int, float]] = field(
        default_factory=list
    )  # (label, ok, bytes, entries, mtime)
    # Tier 4 supplementary - streaming-provider spatial indices (not JSON-parseable)
    bluetopo_scheme: tuple[bool, int, float] = (False, 0, 0.0)  # (exists, bytes, mtime)
    bluetopo_sidecars: tuple[int, int, int, float] = (0, 0, 0, 0.0)  # (ok_count, failed_count, bytes, mtime)
    topobathy_zips: tuple[int, int, float] = (0, 0, 0.0)  # (count, bytes, newest_mtime)
    # Tier 5 - raw files (download providers only; streaming providers never write raw files)
    raw_files: list[tuple[str, int, int, float, float]] = field(
        default_factory=list
    )  # (prov, count, bytes, newest, oldest)


def _gather_status_data() -> _StatusData:
    """Walk cache directories and collect all data for the detailed report."""
    d = _StatusData()

    # ── Tier 1: tiles ────────────────────────────────────────────
    tiles_root = CACHE_ROOT / "tiles"
    if tiles_root.exists():
        visual_root = tiles_root / "visual"
        if visual_root.exists():
            for style_dir in sorted(visual_root.iterdir()):
                if style_dir.is_dir():
                    pngs = list(style_dir.rglob("*.png"))
                    sz = sum(f.stat().st_size for f in pngs if f.is_file())
                    d.tile_subdirs.append((f"visual/{style_dir.name}", len(pngs), sz))

        for subdir_name, globs in [("data", ["*.npy", "*.npz"]), ("raw", ["*.tif", "*.tiff"])]:
            sub = tiles_root / subdir_name
            if sub.exists():
                files: list[Path] = []
                for g in globs:
                    files.extend(sub.rglob(g))
                sz = sum(f.stat().st_size for f in files if f.is_file())
                if files:
                    d.tile_subdirs.append((subdir_name, len(files), sz))

        d.tile_meta_count = sum(1 for _ in tiles_root.rglob("*_meta.json"))
        d.tile_npz_count = sum(1 for _ in tiles_root.rglob("*_src.npz"))

    # ── Tier 2: fused zarr ────────────────────────────────────────
    fused_root = CACHE_ROOT / "fused_zarr"
    policies_root = CACHE_ROOT / "policies"
    if fused_root.exists():
        # Iterate over policy directories (or direct .zarr in legacy format)
        for item in fused_root.iterdir():
            if item.is_dir():
                if item.suffix == ".zarr":
                    # Legacy zarr right in fused_zarr
                    stores = [item]
                    phash = "legacy"
                else:
                    # New style: hash subdirectories
                    stores = [p for p in item.iterdir() if p.suffix == ".zarr"]
                    phash = item.name

                if not stores:
                    continue

                if phash not in d.fused_policies:
                    yaml_snippet = None
                    if phash != "legacy":
                        # Look for either strictly {phash}.yaml or glob *.{phash}.yaml
                        yaml_path = policies_root / f"{phash}.yaml"
                        yaml_match = None

                        if yaml_path.exists():
                            yaml_match = yaml_path
                        else:
                            matches = list(policies_root.glob(f"*.{phash}.yaml"))
                            if matches:
                                yaml_match = matches[0]

                        if yaml_match:
                            import contextlib

                            with contextlib.suppress(Exception):
                                yaml_snippet = yaml_match.read_text()
                    d.fused_policies[phash] = {
                        "yaml_snippet": yaml_snippet,
                        "count": 0,
                        "bytes": 0,
                        "by_zoom": {},
                    }

                p_dict = d.fused_policies[phash]

                for z_item in stores:
                    d.fused_count += 1
                    sz = _dir_size_bytes(z_item)
                    mtime = z_item.stat().st_mtime
                    d.fused_total_bytes += sz
                    p_dict["bytes"] += sz
                    p_dict["count"] += 1
                    d.fused_newest = max(d.fused_newest, mtime)
                    d.fused_oldest = min(d.fused_oldest, mtime)

                    extents_res = _fused_zarr_extents(z_item)
                    if extents_res is not None:
                        x0, x1, y0, y1, res, res_origin, _tgt_zoom, res_m = extents_res
                        zoom, lon_min, lon_max, lat_min, lat_max = _proj_to_zoom_and_lonlat(
                            x0, x1, y0, y1, resolution=res
                        )
                        # Use 2-decimal precision to accurately group custom resolutions
                        if isinstance(zoom, float):
                            zoom = round(zoom, 2)

                        b_z = p_dict["by_zoom"]
                        if zoom not in b_z:
                            b_z[zoom] = {
                                "count": 0,
                                "bytes": 0,
                                "lon_min": lon_min,
                                "lon_max": lon_max,
                                "lat_min": lat_min,
                                "lat_max": lat_max,
                                "origin_zoom": 0,
                                "origin_meters": 0,
                                "origin_unknown": 0,
                                "resolution_m": res_m,
                            }
                        b = b_z[zoom]
                        b["count"] += 1
                        b["bytes"] += sz
                        b["lon_min"] = min(b["lon_min"], lon_min)
                        b["lon_max"] = max(b["lon_max"], lon_max)
                        b["lat_min"] = min(b["lat_min"], lat_min)
                        b["lat_max"] = max(b["lat_max"], lat_max)
                        # Track provenance origin counts
                        if res_origin == "zoom":
                            b["origin_zoom"] += 1
                        elif res_origin == "meters":
                            b["origin_meters"] += 1
                        else:
                            b["origin_unknown"] += 1
                        # Keep the resolution_m from the first cell with a value
                        if res_m is not None and b.get("resolution_m") is None:
                            b["resolution_m"] = res_m

    # ── Tier 3: provider zarr ─────────────────────────────────────
    for provider in PROVIDERS:
        zarrs: list[Path] = []
        for z_root in [CACHE_ROOT / provider / "zarr", CACHE_ROOT / provider]:
            if z_root.exists():
                zarrs.extend(z_root.glob("*.zarr"))
        newest: float = 0.0
        oldest: float = 0.0
        if zarrs:
            sz = sum(_dir_size_bytes(z) for z in zarrs)
            mtimes = [float(z.stat().st_mtime) for z in zarrs]
            if mtimes:
                newest = float(max(mtimes))
                oldest = float(min(mtimes))
        else:
            sz = 0
        d.provider_zarr.append((provider, len(zarrs), sz, newest, oldest))

    # ── Tier 4: discovery caches ──────────────────────────────────
    for label, path in _DISCOVERY_FILES:
        if not path.exists():
            d.discovery.append((label, False, 0, 0, 0.0))
            continue
        try:
            stat = path.stat()
            entries = -1
            if path.suffix == ".sqlite":
                import contextlib
                import sqlite3

                with contextlib.closing(sqlite3.connect(path)) as conn:
                    cur = conn.execute("SELECT COUNT(*) FROM vdatum_offsets")
                    entries = cur.fetchone()[0]
            else:
                with open(path) as f:
                    data = json.load(f)
                entries = len(data) if isinstance(data, dict | list) else -1
            d.discovery.append((label, True, stat.st_size, entries, stat.st_mtime))
        except Exception:
            d.discovery.append((label, False, 0, 0, 0.0))
    # BlueTopo Tile Scheme (.gpkg — spatial index of tile footprints, not JSON)
    scheme = CACHE_ROOT / "noaa_bluetopo" / "BlueTopo_Tile_Scheme.gpkg"
    if scheme.exists():
        stat = scheme.stat()
        d.bluetopo_scheme = (True, stat.st_size, stat.st_mtime)
    # BlueTopo sidecar RAT files (*.aux.xml downloaded per-tile + *.failed sentinels)
    sidecar_dir = CACHE_ROOT / "noaa_bluetopo" / "sidecars"
    if sidecar_dir.exists():
        all_files = list(sidecar_dir.iterdir())
        failed = [f for f in all_files if f.name.endswith(".failed")]
        ok = [f for f in all_files if not f.name.endswith(".failed")]
        sz = sum(f.stat().st_size for f in all_files)
        newest = max((f.stat().st_mtime for f in all_files), default=0.0)
        d.bluetopo_sidecars = (len(ok), len(failed), sz, newest)
    # NOAA Topobathy per-project tile index files (*.zip / *.gpkg)
    topo_dir = CACHE_ROOT / "noaa_topobathy"
    if topo_dir.exists():
        zips = [f for pat in ("*.zip", "*.gpkg") for f in topo_dir.glob(pat)]
        if zips:
            sz = sum(f.stat().st_size for f in zips)
            newest = max(f.stat().st_mtime for f in zips)
            d.topobathy_zips = (len(zips), sz, newest)

    # ── Tier 5: raw source files ──────────────────────────────────
    for provider, patterns in SOURCE_PATTERNS.items():
        prov_dir = CACHE_ROOT / provider
        raw_files: list[Path] = []
        if prov_dir.exists():
            for pat in patterns:
                raw_files.extend(prov_dir.rglob(pat))
        raw_newest: float = 0.0
        raw_oldest: float = 0.0
        if raw_files:
            file_stats = [f.stat() for f in raw_files if f.is_file()]
            sz = sum(s.st_size for s in file_stats)
            file_mtimes = [float(s.st_mtime) for s in file_stats]
            if file_mtimes:
                raw_newest = float(max(file_mtimes))
                raw_oldest = float(min(file_mtimes))
        else:
            sz = 0
        d.raw_files.append((provider, len(raw_files), sz, raw_newest, raw_oldest))

    d.total_bytes = sum(t.size_bytes() for t in TIERS)
    return d


def _render_status_report(c: Console, d: _StatusData) -> None:
    """Render the full detailed status report to *c* (call inside a pager context)."""
    now = time.time()

    c.print()
    c.print(
        Panel(
            f"[bold]Cache root:[/]  [cyan]{CACHE_ROOT}[/]\n"
            f"[bold]Total size:[/]  [bold green]{_fmt_size(d.total_bytes)}[/]",
            title="[bold cyan]topobathykit — Detailed Cache Status[/]",
            border_style="cyan",
            padding=(1, 2),
        )
    )

    # ── Tier 1 ────────────────────────────────────────────────────
    c.print()
    c.rule("[bold cyan]Tier 1 — Output Tiles & Metadata[/]  [dim](safe to purge, re-rendered on request)[/]")
    if d.tile_subdirs or d.tile_meta_count:
        t = Table(box=box.SIMPLE, show_header=True, header_style="dim")
        t.add_column("Subdirectory", style="cyan", min_width=24)
        t.add_column("Files", justify="right", width=8)
        t.add_column("Size", justify="right", width=10)
        for label, count, sz in d.tile_subdirs:
            t.add_row(label, str(count), _fmt_size(sz))
        if d.tile_meta_count:
            t.add_row(
                "[dim]  _meta.json sidecars[/]",
                Text(str(d.tile_meta_count), style="dim"),
                Text("—", style="dim"),
            )
        if d.tile_npz_count:
            t.add_row(
                "[dim]  _src.npz sidecars[/]",
                Text(str(d.tile_npz_count), style="dim"),
                Text("—", style="dim"),
            )
        c.print(t)
    else:
        c.print("  [dim](empty — tiles are generated on first request)[/]")

    # ── Tier 2 ────────────────────────────────────────────────────
    c.print()
    c.rule("[bold cyan]Tier 2 — Fused Zarr[/]  [dim](re-fused from provider zarr on next request)[/]")
    if d.fused_count:
        age_parts: list[str] = []
        if d.fused_oldest != float("inf"):
            age_parts.append(f"oldest {_fmt_age(now - d.fused_oldest)}")
        if d.fused_newest:
            age_parts.append(f"newest {_fmt_age(now - d.fused_newest)}")
        age_str = "  [dim]" + " · ".join(age_parts) + "[/]" if age_parts else ""
        c.print(
            f"  [bold]{d.fused_count}[/] stores · [bold green]{_fmt_size(d.fused_total_bytes)}[/]{age_str}"
        )
        for phash, p_dict in d.fused_policies.items():
            c.print(
                f"  [bold cyan]Policy Hash:[/] {phash} "
                f"({p_dict['count']} stores, {_fmt_size(p_dict['bytes'])})"
            )
            t = Table(box=box.SIMPLE, show_header=True, header_style="dim")
            t.add_column("Zoom", justify="right", width=6)
            t.add_column("Stores", justify="right", width=7)
            t.add_column("Size", justify="right", width=10)
            t.add_column("Lon range", justify="center", min_width=22)
            t.add_column("Lat range", justify="center", min_width=18)
            for zoom in sorted(p_dict["by_zoom"]):
                b = p_dict["by_zoom"][zoom]
                lon_str = f"{b['lon_min']:.2f}°  …  {b['lon_max']:.2f}°"
                lat_str = f"{b['lat_min']:.2f}°  …  {b['lat_max']:.2f}°"
                t.add_row(str(zoom), str(b["count"]), _fmt_size(b["bytes"]), lon_str, lat_str)
            c.print(t)
    else:
        c.print("  [dim](empty)[/]")

    # ── Tier 3 ────────────────────────────────────────────────────
    c.print()
    c.rule("[bold cyan]Tier 3 — Provider Zarr[/]  [dim](re-rasterised from raw files; slow to rebuild)[/]")
    t = Table(box=box.SIMPLE, show_header=True, header_style="dim")
    t.add_column("Provider", style="cyan", min_width=18)
    t.add_column("Zarr stores", justify="right", width=12)
    t.add_column("Size", justify="right", width=10)
    t.add_column("Newest", justify="right", width=12)
    t.add_column("Oldest", justify="right", width=12)
    for prov, count, sz, newest, oldest in d.provider_zarr:
        if count == 0:
            t.add_row(
                prov,
                Text("—", style="dim"),
                Text("—", style="dim"),
                Text("—", style="dim"),
                Text("—", style="dim"),
            )
        else:
            t.add_row(
                prov,
                str(count),
                _fmt_size(sz),
                _fmt_age(now - newest),
                _fmt_age(now - oldest),
            )
    c.print(t)

    # ── Tier 4 ────────────────────────────────────────────────────
    c.print()
    c.rule("[bold cyan]Tier 4 — Discovery Caches[/]  [dim](re-queried from network on next request)[/]")
    t = Table(box=box.SIMPLE, show_header=True, header_style="dim")
    t.add_column("File", style="cyan", min_width=44)
    t.add_column("Size", justify="right", width=10)
    t.add_column("Entries", justify="right", width=8)
    t.add_column("Age", justify="right", width=11)
    t.add_column("Status", width=10)
    for label, ok, sz, entries, mtime in d.discovery:
        if not ok:
            t.add_row(label, "—", "—", "—", Text("not found", style="dim"))
        else:
            ec_str = str(entries) if entries >= 0 else "?"
            t.add_row(label, _fmt_size(sz), ec_str, _fmt_age(now - mtime), Text("OK", style="green"))
    # BlueTopo Tile Scheme (.gpkg spatial index — no entry count)
    scheme_ok, scheme_sz, scheme_mtime = d.bluetopo_scheme
    if scheme_ok:
        t.add_row(
            "noaa_bluetopo/BlueTopo_Tile_Scheme.gpkg",
            _fmt_size(scheme_sz),
            "—",
            _fmt_age(now - scheme_mtime),
            Text("OK", style="green"),
        )
    else:
        t.add_row("noaa_bluetopo/BlueTopo_Tile_Scheme.gpkg", "—", "—", "—", Text("not found", style="dim"))
    # BlueTopo sidecar RAT files
    sc_ok, sc_fail, sc_bytes, sc_mtime = d.bluetopo_sidecars
    if sc_ok + sc_fail > 0:
        fail_note = f"  [yellow]{sc_fail} failed[/]" if sc_fail else ""
        label = f"noaa_bluetopo/sidecars  [dim]({sc_ok} RAT{fail_note}[dim])[/]"
        t.add_row(
            label,
            _fmt_size(sc_bytes),
            str(sc_ok + sc_fail),
            _fmt_age(now - sc_mtime),
            Text("OK", style="green"),
        )
    else:
        t.add_row("noaa_bluetopo/sidecars", "—", "—", "—", Text("not found", style="dim"))
    # NOAA Topobathy per-project tile index files (aggregate)
    zip_count, zip_bytes, zip_newest = d.topobathy_zips
    if zip_count:
        t.add_row(
            f"noaa_topobathy/ tile indices  [dim]({zip_count} files)[/]",
            _fmt_size(zip_bytes),
            "—",
            _fmt_age(now - zip_newest),
            Text("OK", style="green"),
        )
    else:
        t.add_row("noaa_topobathy/ tile indices", "—", "—", "—", Text("not found", style="dim"))
    c.print(t)

    # ── Tier 5 ────────────────────────────────────────────────────
    c.print()
    c.rule("[bold yellow]Tier 5 ! — Raw Source Files[/]  [dim](download providers only)[/]")
    t = Table(box=box.SIMPLE, show_header=True, header_style="dim")
    t.add_column("Provider", style="cyan", min_width=18)
    t.add_column("Files", justify="right", width=8)
    t.add_column("Size", justify="right", width=10)
    t.add_column("Newest", justify="right", width=12)
    t.add_column("Oldest", justify="right", width=12)
    any_raw = False
    for prov, count, sz, newest, oldest in d.raw_files:
        if count == 0:
            t.add_row(
                prov,
                Text("—", style="dim"),
                Text("—", style="dim"),
                Text("—", style="dim"),
                Text("—", style="dim"),
            )
        else:
            any_raw = True
            t.add_row(
                prov,
                str(count),
                _fmt_size(sz),
                _fmt_age(now - newest),
                _fmt_age(now - oldest),
            )
    c.print(t)
    if not any_raw:
        c.print("  [dim](no raw files cached yet)[/]")
    streaming = sorted(_STREAMING_PROVIDERS)
    c.print(
        f"\n  [dim]Streaming providers (no local raw files — pull direct from cloud → zarr):\n"
        f"  {', '.join(streaming)}\n"
        f"  Purge tier 3 to force re-streaming for these providers.[/]"
    )

    c.print()
    c.print("[dim]  Press [bold]q[/] to return to the menu.[/]")
    c.print()


# ---------------------------------------------------------------------------
# Rich display — tier info panels (shown in pager)
# ---------------------------------------------------------------------------


def _render_tier_panel(tier: CacheTier) -> Panel:
    title_color = "yellow" if tier.warning else "cyan"
    warn_tag = "  [bold yellow]!  destructive / slow to rebuild[/]" if tier.warning else ""
    content = Text.from_markup(tier.long_desc.replace("WARNING:", "[bold red]WARNING:[/]"))
    return Panel(
        content,
        title=f"[bold {title_color}][{tier.number}] {tier.name}[/]{warn_tag}",
        border_style=title_color,
        padding=(1, 2),
    )


# ---------------------------------------------------------------------------
# Integrity check
# ---------------------------------------------------------------------------


def _scan_provider_integrity(
    provider_name: str,
    clean: bool = False,
    lock_timeout: int = 3600,
) -> dict[str, int]:
    counts = {"stale_locks": 0, "corrupt_src": 0, "corrupt_zarr": 0}
    p_dir = CACHE_ROOT / provider_name
    if not p_dir.exists():
        return counts

    # Note: global .lock/.tmp cleanup is handled by _scan_global_orphans,
    # but we keep this here for provider-scoped checks just in case.
    for t in list(p_dir.glob(".tmp_*")) + list(p_dir.glob("*.lock")):
        age = time.time() - t.stat().st_mtime
        if age > lock_timeout:
            counts["stale_locks"] += 1
            if clean:
                shutil.rmtree(t) if t.is_dir() else t.unlink()

    for pat in SOURCE_PATTERNS.get(provider_name, []):
        for f in p_dir.rglob(pat):
            ok = verify_laz_integrity(f) if f.suffix == ".laz" else verify_raster_integrity(f)
            if not ok:
                counts["corrupt_src"] += 1
                if clean:
                    f.unlink()

    for z_root in [p_dir / "zarr", p_dir]:
        if not z_root.exists():
            continue
        for z in z_root.glob("*.zarr"):
            if not verify_zarr_integrity(z):
                counts["corrupt_zarr"] += 1
                if clean:
                    shutil.rmtree(z)

    return counts


def _scan_global_orphans(clean: bool = False, lock_timeout: int = 3600) -> tuple[int, int]:
    """Scan and purge orphaned .lock and .tmp_* files across the ENTIRE cache root."""
    mode = "[bold red]DELETE[/]" if clean else "[bold green]SCAN[/]"
    console.print(f"\n[bold]Orphaned File {mode} (timeout={lock_timeout}s)...[/]")

    stale_count = 0
    removed_count = 0
    failed_count = 0
    now = time.time()

    # We target:
    # 1. *.lock anywhere (e.g. tiles/visual/default/13/123/456.lock)
    # 2. .tmp* (usually directories, sometimes files) anywhere
    patterns = ["*.lock", ".tmp*", "__temp__*"]
    candidates: list[Path] = []

    with console.status("Scanning file system (this may take a moment)...", spinner="dots"):
        for pat in patterns:
            candidates.extend(CACHE_ROOT.rglob(pat))

    # Filter by age and uniqueness
    candidates = sorted(list(set(candidates)), key=lambda p: str(p))

    for p in candidates:
        if not p.exists():
            continue
        try:
            stat = p.stat()
            age = now - stat.st_mtime
            if age > lock_timeout:
                stale_count += 1
                try:
                    rel_path = p.relative_to(CACHE_ROOT)
                except ValueError:
                    rel_path = p

                if clean:
                    if p.is_dir():
                        shutil.rmtree(p)
                    else:
                        p.unlink()
                    console.print(f"  [green]✓[/] {rel_path}  [dim]({_fmt_age(age)})[/]")
                    removed_count += 1
                else:
                    console.print(f"  [yellow]![/] {rel_path}  [dim]({_fmt_age(age)})[/]")
        except Exception as e:
            console.print(f"  [red]x[/] {p.name}: {e}")
            failed_count += 1

    if stale_count == 0:
        console.print("  [dim]No orphaned files found.[/]")
    else:
        suffix = f" -> {removed_count} removed" if clean else ""
        console.print(f"\n  Found {stale_count} orphaned items{suffix}.")

    return stale_count, removed_count


def cmd_check(clean: bool = False, lock_timeout: int = 3600) -> None:
    mode = "[bold red]CLEAN[/]" if clean else "[bold green]CHECK[/]"
    console.print()
    console.rule(f"[bold]Cache Integrity — {mode} mode[/]")

    results_table = Table(box=box.SIMPLE, show_header=True, header_style="bold dim")
    results_table.add_column("Provider", style="cyan", min_width=18)
    results_table.add_column("Stale locks", justify="right", width=12)
    results_table.add_column("Corrupt src", justify="right", width=12)
    results_table.add_column("Corrupt zarr", justify="right", width=13)

    def _cell(n: int) -> Text:
        if n == 0:
            return Text("—", style="dim")
        return Text(str(n), style="red bold" if not clean else "yellow")

    with Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Scanning…", total=len(PROVIDERS) + 1)
        for p in PROVIDERS:
            progress.update(task, description=f"Scanning [cyan]{p}[/]…")
            counts = _scan_provider_integrity(p, clean=clean, lock_timeout=lock_timeout)
            results_table.add_row(
                p, _cell(counts["stale_locks"]), _cell(counts["corrupt_src"]), _cell(counts["corrupt_zarr"])
            )
            progress.advance(task)
        progress.update(task, description="Scanning discovery caches…")
        progress.advance(task)

    console.print(results_table)

    _scan_global_orphans(clean=clean, lock_timeout=lock_timeout)

    disco_table = Table(
        box=box.SIMPLE,
        show_header=True,
        header_style="bold dim",
        title="Metadata & Discovery Caches",
    )
    disco_table.add_column("File", min_width=44)
    disco_table.add_column("Size", justify="right", width=10)
    disco_table.add_column("Entries", justify="right", width=8)
    disco_table.add_column("Status", width=14)

    any_found = False
    for label, path in _DISCOVERY_FILES:
        if not path.exists():
            disco_table.add_row(label, "—", "—", Text("not found", style="dim"))
            continue
        any_found = True
        try:
            ec = "?"
            if path.suffix == ".sqlite":
                import contextlib
                import sqlite3

                with contextlib.closing(sqlite3.connect(path)) as conn:
                    cur = conn.execute("SELECT COUNT(*) FROM vdatum_offsets")
                    ec = str(cur.fetchone()[0])
            else:
                with open(path) as f:
                    data = json.load(f)
                ec = str(len(data)) if isinstance(data, dict | list) else "?"
            disco_table.add_row(
                label, _fmt_size(path.stat().st_size), str(ec), Text("OK", style="green bold")
            )
        except Exception as e:
            if clean:
                path.unlink()
                disco_table.add_row(label, "—", "—", Text("DELETED", style="red bold"))
            else:
                disco_table.add_row(label, "—", "—", Text(f"CORRUPT: {e}", style="red bold"))

    console.print(disco_table)
    if not any_found:
        console.print("[dim]  (no discovery cache files found — created on first tile request)[/]")

    suffix = "Corrupt files and stale locks deleted." if clean else "Use [bold]--clean[/] to delete."
    console.print(f"\n[green]✓[/] Integrity check complete. {suffix}\n")


# ---------------------------------------------------------------------------
# Purge
# ---------------------------------------------------------------------------


def _do_purge(tier: CacheTier, dry_run: bool = False) -> tuple[int, int]:
    paths = tier.purge_paths()
    removed = failed = 0
    for p in paths:
        if dry_run:
            console.print(f"  [yellow]~[/] [dim]{p}[/]")
            removed += 1
            continue
        try:
            shutil.rmtree(p) if p.is_dir() else p.unlink()
            console.print(f"  [green]✓[/] [dim]{p}[/]")
            removed += 1
        except Exception as e:
            console.print(f"  [red]x[/] {p}: {e}")
            failed += 1
    return removed, failed


def cmd_purge(tier_numbers: list[int | str], dry_run: bool = False, yes: bool = False) -> None:
    targets = (
        list(TIERS)
        if "all" in tier_numbers
        else [TIERS[int(n) - 1] for n in tier_numbers if str(n).isdigit() and 1 <= int(n) <= 5]
    )
    if not targets:
        console.print("[yellow]No valid tier numbers specified.[/]")
        return

    label = "[dim]DRY RUN[/] — " if dry_run else ""
    names = ", ".join(f"[cyan]{t.number}[/]:{t.name}" for t in targets)
    console.print(f"\n{label}Purging tiers: {names}\n")
    for tier in targets:
        _purge_tier_with_confirm(tier, dry_run=dry_run, yes=yes)


def _purge_tier_with_confirm(tier: CacheTier, dry_run: bool = False, yes: bool = False) -> bool:
    paths = tier.purge_paths()
    if not paths:
        console.print(f"  [dim]Tier {tier.number} is already empty — nothing to delete.[/]")
        return False

    warn_tag = "  [bold yellow]!  HOURS to rebuild[/]" if tier.warning else ""
    console.print(f"\n[bold]Tier {tier.number}: {tier.name}[/]{warn_tag}")
    console.print(f"  {len(paths)} item(s) — [bold]{_fmt_size(tier.size_bytes())}[/]")

    if not dry_run and not yes:
        if tier.warning:
            console.print()
            console.print("[bold yellow]  This tier contains raw downloaded source files.[/]")
            console.print("[yellow]  Re-downloading may take many hours.[/]")
            try:
                answer = input('\n  Type "yes" to confirm deletion: ').strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n  [dim]Aborted.[/]")
                return False
            if answer != "yes":
                console.print("  [dim]Kept.[/]")
                return False
        else:
            try:
                answer = input(f"\n  Delete {len(paths)} item(s)? [y/N]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                console.print("\n  [dim]Aborted.[/]")
                return False
            if answer not in ("y", "yes"):
                console.print("  [dim]Kept.[/]")
                return False

    console.print()
    removed, failed = _do_purge(tier, dry_run=dry_run)
    suffix = " [dim](dry run)[/]" if dry_run else ""
    if failed:
        console.print(f"\n  [yellow]Done{suffix}: {removed} removed, {failed} failed.[/]")
    else:
        console.print(f"\n  [green]Done{suffix}: {removed}/{len(paths)} item(s) removed.[/]")
    if removed and not dry_run:
        console.print("  [dim]Re-built or re-downloaded on next tile request.[/]")
    return removed > 0


# ---------------------------------------------------------------------------
# Interactive TUI
# ---------------------------------------------------------------------------

_MAIN_CHOICES = [
    ("Detailed status report", "status"),
    ("Tier info  (when & why to purge)", "info"),
    ("Purge tier(s)", "purge"),
    ("Check integrity  (read-only audit)", "check"),
    ("Clean integrity  (delete corrupt/stale)", "clean"),
    ("List orphaned locks/tmp (fast)", "list-orphans"),
    ("Clean orphaned locks/tmp (fast)", "clean-orphans"),
    ("Quit", "quit"),
]


def _tui_main_menu() -> str | None:
    import questionary

    choices = [questionary.Choice(label, value=val) for label, val in _MAIN_CHOICES]
    result: str | None = questionary.select(
        "What would you like to do?",
        choices=choices,
        style=_get_qstyle(),
        use_shortcuts=False,
        use_indicator=True,
    ).ask()
    return result


def _tui_tier_checkbox(prompt: str) -> list[CacheTier] | None:
    import questionary

    choices = []
    for tier in TIERS:
        items = tier.item_count()
        sz = _fmt_size(tier.size_bytes()) if items else "empty"
        warn = " !" if tier.warning else ""
        label = f"[{tier.number}]{warn}  {tier.name:<30}  {sz:>9}  ({items} items)"
        choices.append(questionary.Choice(label, value=tier, disabled=None if items else "empty"))
    result: list[CacheTier] | None = questionary.checkbox(prompt, choices=choices, style=_get_qstyle()).ask()
    return result


def _tui_confirm(msg: str) -> bool | None:
    import questionary

    result: bool | None = questionary.confirm(msg, style=_get_qstyle(), default=False).ask()
    return result


def _tui_text(msg: str) -> str | None:
    import questionary

    result: str | None = questionary.text(msg, style=_get_qstyle()).ask()
    return result


def cmd_interactive(lock_timeout: int = 3600) -> None:
    import questionary  # noqa: F401

    console.print()
    console.print(
        Panel(
            "[bold cyan]topobathykit Cache Manager[/]\n"
            "[dim]Arrow keys to navigate · Enter to select · q to quit[/]",
            border_style="cyan",
            padding=(0, 2),
        )
    )

    while True:
        # Compact summary table is always shown before the menu prompt
        console.print(_render_status_table())

        action = _tui_main_menu()

        if action is None or action == "quit":
            console.print("\n[dim]Goodbye.[/]\n")
            break

        elif action == "status":
            # Gather data with a spinner (directory walks can be slow), then page
            with console.status("[cyan]Building status report…[/]", spinner="dots"):
                data = _gather_status_data()
            with _rich_pager() as pager:
                _render_status_report(pager, data)

        elif action == "info":
            with _rich_pager() as pager:
                pager.print()
                pager.print(
                    Panel(
                        "[dim]Arrow keys / j k to scroll · [bold]q[/] to return[/]",
                        border_style="dim",
                        padding=(0, 2),
                    )
                )
                for tier in TIERS:
                    pager.print()
                    pager.print(_render_tier_panel(tier))
                pager.print()

        elif action == "purge":
            selected = _tui_tier_checkbox("Select tier(s) to purge")
            if not selected:
                console.print("[dim]  Nothing selected.[/]")
                continue

            # Preview table
            console.print()
            summary = Table(box=box.SIMPLE, show_header=False)
            summary.add_column("", style="cyan", width=10)
            summary.add_column("", min_width=30)
            summary.add_column("", justify="right", width=10, style="bold")
            for tier in selected:
                warn = " [bold yellow]![/]" if tier.warning else ""
                summary.add_row(f"  Tier {tier.number}{warn}", tier.name, _fmt_size(tier.size_bytes()))
            console.print(summary)

            if not _tui_confirm("Proceed with purge?"):
                console.print("[dim]  Cancelled.[/]")
                continue

            for tier in selected:
                if tier.warning:
                    console.print()
                    console.print("[bold yellow]  !  Tier 5 contains raw downloaded source files.[/]")
                    console.print("[yellow]  Re-downloading may take many hours.[/]")
                    answer = _tui_text('  Type "yes" to confirm deletion of raw files')
                    if answer != "yes":
                        console.print("[dim]  Skipped Tier 5.[/]")
                        continue

                console.print()
                paths = tier.purge_paths()
                with Progress(
                    SpinnerColumn(style="cyan"),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                    transient=True,
                ) as progress:
                    task = progress.add_task(f"Purging Tier {tier.number}…", total=len(paths))
                    removed = failed = 0
                    for p in paths:
                        progress.update(task, description=f"Deleting [dim]{p.name}[/]…")
                        try:
                            shutil.rmtree(p) if p.is_dir() else p.unlink()
                            removed += 1
                        except Exception as e:
                            console.print(f"  [red]x[/] {p}: {e}")
                            failed += 1
                        progress.advance(task)

                if failed:
                    console.print(f"  [yellow]Tier {tier.number}: {removed} removed, {failed} failed.[/]")
                else:
                    console.print(f"  [green]✓[/]  Tier {tier.number}: {removed}/{len(paths)} removed.")

            console.print("[dim]  Re-building happens automatically on next tile request.[/]")

        elif action in ("check", "clean"):
            cmd_check(clean=(action == "clean"), lock_timeout=lock_timeout)

        elif action in ("list-orphans", "clean-orphans"):
            _scan_global_orphans(clean=(action == "clean-orphans"), lock_timeout=lock_timeout)

    console.print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="cache_manager",
        description=(
            "Inspect, verify, and purge the topobathykit cache hierarchy.\n\n"
            "Run without arguments for the interactive menu (requires TTY)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  %(prog)s                       # interactive menu (TTY)\n"
            "  %(prog)s --check               # integrity audit (read-only)\n"
            "  %(prog)s --clean               # delete corrupt/stale files\n"
            "  %(prog)s --orphans             # list orphaned .lock/.tmp files\n"
            "  %(prog)s --clean-orphans       # delete orphaned .lock/.tmp files\n"
            "  %(prog)s --purge 1             # purge output tiles\n"
            "  %(prog)s --purge 4             # purge discovery caches\n"
            "  %(prog)s --purge 1 2 4         # purge multiple tiers\n"
            "  %(prog)s --purge all --dry-run # preview full cache wipe\n"
            "  %(prog)s --purge 5 --yes       # purge raw files, no prompt\n"
        ),
    )
    parser.add_argument("--check", action="store_true", help="Integrity audit: read-only scan.")
    parser.add_argument("--clean", action="store_true", help="Integrity check + delete corrupt/stale files.")
    parser.add_argument(
        "--orphans",
        action="store_true",
        help="List orphaned .lock and .tmp* files across the cache.",
    )
    parser.add_argument(
        "--clean-orphans",
        action="store_true",
        help="Delete orphaned .lock and .tmp* files across the cache.",
    )
    parser.add_argument("--purge", nargs="+", metavar="TIER", help="Tier number(s) to purge (1-5) or 'all'.")
    parser.add_argument("--dry-run", action="store_true", help="Preview without deleting.")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompts.")
    parser.add_argument(
        "--lock-timeout",
        type=int,
        default=3600,
        metavar="SECONDS",
        help="Seconds before a .lock file is considered stale (default: 3600).",
    )

    args = parser.parse_args()

    if args.orphans or args.clean_orphans:
        _scan_global_orphans(clean=args.clean_orphans, lock_timeout=args.lock_timeout)
    elif args.check or args.clean:
        cmd_check(clean=args.clean, lock_timeout=args.lock_timeout)
    elif args.purge:
        cmd_purge(args.purge, dry_run=args.dry_run, yes=args.yes)
    elif _IS_TTY:
        try:
            import questionary  # noqa: F401

            cmd_interactive(lock_timeout=args.lock_timeout)
        except ImportError:
            console.print("[yellow]questionary not installed — falling back to status display.[/]")
            console.print(_render_status_table())
    else:
        console.print(_render_status_table())
        console.print("[dim]Run with --check, --clean, or --purge <tier> for actions.[/]")


if __name__ == "__main__":
    main()
