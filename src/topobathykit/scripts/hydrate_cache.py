#!/usr/bin/env python3
"""
Hydrate the topobathykit cache.

Subcommands:
    fuse   — Hydrate fused zarr cells for a bbox + resolution via the /hydrate API.
    tiles  — Hydrate rendered PNG tiles for a bbox + zoom level via the /tiles API.

Usage:
    python -m topobathykit.scripts.hydrate_cache fuse  WEST SOUTH EAST NORTH -r 15 -c policy.yaml
    python -m topobathykit.scripts.hydrate_cache tiles WEST SOUTH EAST NORTH -z 13
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import sys
import time
import urllib.error
import urllib.request

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _validate_bbox(bbox: list[float]) -> tuple[float, float, float, float]:
    west, south, east, north = bbox
    if west >= east or south >= north:
        print(f"Invalid bbox: west({west}) >= east({east}) or south({south}) >= north({north})")
        sys.exit(1)
    return west, south, east, north


# ---------------------------------------------------------------------------
# fuse subcommand — hydrate fused zarr cells via /hydrate endpoint
# ---------------------------------------------------------------------------


def _print_fuse_status(status: dict) -> None:
    processed = status.get("processed_cells", 0) + status.get("cached_cells", 0)
    total = status.get("total_cells", 0)
    failed = status.get("failed_cells", 0)
    pct = int(100 * processed / total) if total else 0
    parts = [f"Status: {status['status']} | Progress: {pct}% | Cells: {processed}/{total}"]
    if failed:
        parts.append(f" | Failed: {failed}")
    if status.get("error"):
        parts.append(f" | Error: {status['error']}")
    print("".join(parts))

    if status.get("status") in ("completed", "failed") and failed > 0:
        print("\n" + "=" * 80)
        print("!  WARNING: Some cells failed to hydrate. This is commonly caused by")
        print("   Out-Of-Memory (OOM) / ArrayMemoryError exceptions when SciPy interpolates")
        print("   massive gaps across too many concurrent workers.")
        print("   ")
        print("   STRATEGY: The successfully processed cells are already cached! You can")
        print("   safely re-run the exact same command. Or, to reduce memory pressure,")
        print("   re-run with a lower worker limit (e.g., `-w 1` or `--workers 1`) to let")
        print("   the remaining cells process sequentially with full system RAM.")
        print("=" * 80 + "\n")


def _monitor_ws(ws_url: str) -> None:
    import websockets.sync.client  # type: ignore[import-untyped]

    print("Connecting via WebSocket...")
    with websockets.sync.client.connect(ws_url) as ws:
        while True:
            msg = ws.recv()
            status = json.loads(msg)
            _print_fuse_status(status)
            if status.get("status") in ("completed", "failed"):
                return


def _monitor_poll(base_url: str, job_id: str) -> None:
    print("Monitoring via HTTP polling...")
    while True:
        try:
            req = urllib.request.Request(f"{base_url}/hydrate/{job_id}")
            with urllib.request.urlopen(req) as response:
                status = json.loads(response.read().decode("utf-8"))
                _print_fuse_status(status)
                if status["status"] in ("completed", "failed"):
                    return
        except urllib.error.HTTPError as e:
            if e.code != 404:
                print(f"HTTP Error: {e.code}")
        time.sleep(2)


def cmd_fuse(args: argparse.Namespace) -> None:
    """Submit a fused-zarr hydration job and monitor progress."""
    west, south, east, north = _validate_bbox(args.bbox)

    payload: dict = {
        "bbox": [west, south, east, north],
        "resolution": args.resolution,
    }
    if args.config:
        with open(args.config) as f:
            payload["policy_yaml"] = f.read()
    if args.workers:
        payload["max_workers"] = args.workers

    req = urllib.request.Request(
        f"{args.url}/hydrate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    print(f"Submitting hydration: bbox=[{west}, {south}, {east}, {north}] res={args.resolution}m")
    try:
        with urllib.request.urlopen(req) as response:
            resp = json.loads(response.read().decode("utf-8"))
            job_id = resp["job_id"]
            ws_url = resp.get("ws")
            print(f"Job started: {job_id}")
    except urllib.error.HTTPError as e:
        print(f"Error: {e}")
        print(e.read().decode("utf-8"))
        sys.exit(1)

    if ws_url and not args.no_ws:
        try:
            _monitor_ws(ws_url)
            return
        except ImportError:
            print("websockets not installed, falling back to HTTP polling")
        except Exception as e:
            print(f"WebSocket failed ({e}), falling back to HTTP polling")

    _monitor_poll(args.url, job_id)


# ---------------------------------------------------------------------------
# tiles subcommand — hydrate rendered PNG tiles via /tiles endpoint
# ---------------------------------------------------------------------------


def _deg2num(lat_deg: float, lon_deg: float, zoom: int) -> tuple[int, int]:
    """Convert geographic coordinates to XYZ tile index."""
    lat_rad = math.radians(lat_deg)
    n = 2.0**zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile


def _fetch_tile(base_url: str, z: int, x: int, y: int) -> tuple[int, float]:
    """Fetch a single tile from the service."""
    url = f"{base_url}/tiles/{z}/{x}/{y}?format=png"
    try:
        t0 = time.time()
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=60) as resp:
            resp.read()  # consume body to trigger server-side caching
            dur = time.time() - t0
            return resp.status, dur
    except Exception as e:
        print(f"Failed {z}/{x}/{y}: {e}")
        return 0, 0.0


def cmd_tiles(args: argparse.Namespace) -> None:
    """Hydrate rendered PNG tile cache for a bbox + zoom level."""
    west, south, east, north = _validate_bbox(args.bbox)
    zoom = args.zoom

    x_min, y_min = _deg2num(north, west, zoom)
    x_max, y_max = _deg2num(south, east, zoom)
    x_start, x_end = min(x_min, x_max), max(x_min, x_max)
    y_start, y_end = min(y_min, y_max), max(y_min, y_max)

    tiles = [(zoom, x, y) for x in range(x_start, x_end + 1) for y in range(y_start, y_end + 1)]

    print(f"Hydrating tile cache: bbox=[{west}, {south}, {east}, {north}]")
    print(f"Zoom: {zoom} | Tiles: {len(tiles)} | Workers: {args.jobs}")

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        future_map = {executor.submit(_fetch_tile, args.url, z, x, y): (z, x, y) for z, x, y in tiles}
        for i, future in enumerate(concurrent.futures.as_completed(future_map), 1):
            status_code, dur = future.result()
            results.append((status_code, dur))
            if i % 50 == 0 or i == len(tiles):
                ok = sum(1 for s, _ in results if s == 200)
                print(f"  Progress: {i}/{len(tiles)} ({ok} OK)")

    success = [r for r in results if r[0] == 200]
    avg_time = sum(r[1] for r in success) / len(success) if success else 0
    print(f"Done. {len(success)}/{len(results)} tiles OK, avg {avg_time:.2f}s/tile")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="hydrate_cache",
        description="Hydrate the topobathykit cache (fused zarr cells or rendered tiles).",
    )
    parser.add_argument(
        "--url",
        default="http://localhost:9595",
        help="Base URL of the topobathykit service (default: http://localhost:9595)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- fuse subcommand ---
    p_fuse = sub.add_parser("fuse", help="Hydrate fused zarr cells for a bbox + resolution")
    p_fuse.add_argument(
        "bbox",
        type=float,
        nargs=4,
        metavar="BBOX",
        help="Bounding box in WGS84 decimal degrees (WEST SOUTH EAST NORTH)",
    )
    p_fuse.add_argument(
        "-r",
        "--resolution",
        type=float,
        default=30.0,
        help="Resolution in meters (default: 30)",
    )
    p_fuse.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="YAML policy config file",
    )
    p_fuse.add_argument(
        "-w", "--workers", type=int, default=None, help="Parallel cell workers (default: server default)"
    )
    p_fuse.add_argument("--no-ws", action="store_true", help="Disable WebSocket, use HTTP polling")
    p_fuse.set_defaults(func=cmd_fuse)

    # --- tiles subcommand ---
    p_tiles = sub.add_parser("tiles", help="Hydrate rendered PNG tiles for a bbox + zoom level")
    p_tiles.add_argument(
        "bbox",
        type=float,
        nargs=4,
        metavar="BBOX",
        help="Bounding box in WGS84 decimal degrees (WEST SOUTH EAST NORTH)",
    )
    p_tiles.add_argument("-z", "--zoom", type=int, default=13, help="Zoom level (default: 13)")
    p_tiles.add_argument("-j", "--jobs", type=int, default=4, help="Parallel fetch workers (default: 4)")
    p_tiles.set_defaults(func=cmd_tiles)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
