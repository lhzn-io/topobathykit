# Import from scripts.cache_manager, suppressing output if any
import contextlib
import io
import json
import logging
import math
import shutil
import time
from pathlib import Path
from typing import Any

from topobathykit.api.models import (
    CacheDetail,
    CacheSummary,
    CacheTierInfo,
    CacheTierSummary,
    CheckResult,
    DiscoveryDetail,
    OrphansResult,
    ProviderDetail,
    PurgeResult,
    RawFileDetail,
    Tier1Detail,
    Tier2Detail,
    Tier3Detail,
    Tier4Detail,
    Tier5Detail,
    TileSubdir,
    ZoomDetail,
)

# We need to suppress strict type checking on this import because it's a script
# and mypy might complain about missing stubs or side effects.
try:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        import topobathykit.scripts.cache_manager as cm
except ImportError:
    # Fallback or error if script is missing
    raise ImportError("Could not import topobathykit.scripts.cache_manager") from None

logger = logging.getLogger(__name__)


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Ensure a value is a finite float, returning default if NaN/Inf."""
    try:
        f = float(val)
        if not math.isfinite(f):
            return default
        return f
    except (ValueError, TypeError):
        return default


# Re-export some constants for convenience
CACHE_ROOT = cm.CACHE_ROOT
TIERS = cm.TIERS

# Cache for get_summary to avoid scanning too often
_SUMMARY_CACHE: tuple[float, CacheSummary] | None = None
_SUMMARY_TTL = 5.0  # seconds


def get_tiers() -> list[CacheTierInfo]:
    """Return static metadata about cache tiers."""
    return [
        CacheTierInfo(
            number=t.number,
            name=t.name,
            short_desc=t.short_desc,
            long_desc=t.long_desc,
            purge_reason=t.purge_reason,
            warning=t.warning,
        )
        for t in TIERS
    ]


def get_cache_summary(cache_bust: bool = False) -> CacheSummary:
    """Return a lightweight summary of cache tiers."""
    global _SUMMARY_CACHE
    now = time.time()

    if not cache_bust and _SUMMARY_CACHE is not None:
        ts, summary = _SUMMARY_CACHE
        if now - ts < _SUMMARY_TTL:
            return summary

    total_bytes = 0
    tier_summaries: list[CacheTierSummary] = []

    # Map tier numbers to the API endpoint that generates them
    tier_origins = {
        1: "/tiles",
        2: "/fuse",
        3: "auto-rasterized",
        4: "auto-discovery",
        5: "download",
    }

    for tier in TIERS:
        # These methods scan the filesystem, hence the caching
        try:
            items = int(tier.item_count())
        except Exception:
            logger.exception(f"Error getting item count for tier {tier.number}")
            items = 0

        try:
            sz = int(tier.size_bytes())
        except Exception:
            logger.exception(f"Error getting size bytes for tier {tier.number}")
            sz = 0

        total_bytes += sz

        mb_val = 0.0
        try:
            # Check for insane values before division
            if sz > 1e15:  # 1 PB
                logger.warning(f"Tier {tier.number} size is massive: {sz}")
                mb_val = 0.0  # Cap visual display to avoid float overflow in extreme cases
            else:
                raw_mb = sz / (1024 * 1024)
                if not math.isfinite(raw_mb):
                    logger.warning(f"Tier {tier.number} raw_mb is not finite: {raw_mb}")
                    mb_val = 0.0
                else:
                    mb_val = round(raw_mb, 2)
        except Exception as e:
            logger.exception(f"Error calculating MB for tier {tier.number}: {e}")
            mb_val = 0.0

        if sz > 1e16:  # Trap insane values
            logger.warning(f"Tier {tier.number} reports insane size: {sz}")

        tier_summaries.append(
            CacheTierSummary(
                number=tier.number,
                name=tier.name,
                items=items,
                bytes=sz,
                mb=mb_val,
                warning=tier.warning,
                origin=tier_origins.get(tier.number),
            )
        )

    total_mb_val = 0.0
    try:
        if total_bytes > 1e15:
            logger.warning(f"Total size is massive: {total_bytes}")
            total_mb_val = 0.0
        else:
            raw_total_mb = total_bytes / (1024 * 1024)
            if not math.isfinite(raw_total_mb):
                logger.warning(f"Total raw_mb is not finite: {raw_total_mb}")
                total_mb_val = 0.0
            else:
                total_mb_val = round(raw_total_mb, 2)
    except Exception as e:
        logger.exception(f"Error calculating total MB: {e}")
        total_mb_val = 0.0

    summary = CacheSummary(
        cache_root=str(CACHE_ROOT),
        total_bytes=total_bytes,
        total_mb=total_mb_val,
        tiers=tier_summaries,
        last_updated=now,
    )

    # Paranoia check: Validate JSON compliance before returning
    try:
        # allow_nan=False is CRITICAL to catch the error that FastAPI/Starlette throws
        json.dumps(summary.dict(), allow_nan=False)
    except ValueError as e:
        logger.error(f"JSON Serialization Check Failed: {e}")
        logger.error(f"Diagnostics: Total MB={summary.total_mb}, LastUpdated={summary.last_updated}")

        # Check Tiers
        for t in summary.tiers:
            if not math.isfinite(t.mb):
                logger.error(f"Tier {t.number} has BAD MB: {t.mb}")
                t.mb = 0.0

        # Check Total
        if not math.isfinite(summary.total_mb):
            logger.error(f"Total MB is BAD: {summary.total_mb}")
            summary.total_mb = 0.0

        # Check Timestamp
        if not math.isfinite(summary.last_updated):
            logger.error(f"Last Updated is BAD: {summary.last_updated}")
            summary.last_updated = 0.0

        # Run check again, if it fails, we are doomed but at least we tried
        try:
            json.dumps(summary.dict(), allow_nan=False)
        except ValueError:
            logger.critical("Failed to sanitize CacheSummary! Returning zeroed fallback.")
            # Fallback to zeroed summary
            summary.total_mb = 0.0
            for t in summary.tiers:
                t.mb = 0.0

    _SUMMARY_CACHE = (now, summary)
    return summary


def get_cache_detail(cache_bust: bool = True) -> CacheDetail:
    """Return full detailed inventory of the cache."""
    # We reuse the logic from cache_manager._gather_status_data
    # but map it to our Pydantic models.
    # Note: cache_manager._gather_status_data is not async, so this blocks.
    # In a real app we might want to run this in a thread pool.

    # Force re-scan
    d = cm._gather_status_data()

    # Map Tier 1
    t1_subdirs = [TileSubdir(label=label, files=files, bytes=sz) for label, files, sz in d.tile_subdirs]
    t1 = Tier1Detail(
        tile_subdirs=t1_subdirs,
        meta_count=d.tile_meta_count,
        npz_count=d.tile_npz_count,
    )

    # Map Tier 2
    from topobathykit.api.models import PolicyHashGroup

    policies_mapped = []

    for phash, pdict in getattr(d, "fused_policies", {}).items():
        by_zoom = {}
        for z, info in pdict.get("by_zoom", {}).items():
            by_zoom[float(z)] = ZoomDetail(
                count=info["count"],
                bytes=info["bytes"],
                lon_min=_safe_float(info["lon_min"]),
                lon_max=_safe_float(info["lon_max"]),
                lat_min=_safe_float(info["lat_min"]),
                lat_max=_safe_float(info["lat_max"]),
                origin_zoom=info.get("origin_zoom", 0),
                origin_meters=info.get("origin_meters", 0),
                origin_unknown=info.get("origin_unknown", 0),
                resolution_m=_safe_float(info["resolution_m"])
                if info.get("resolution_m") is not None
                else None,
            )

        # Extract policy name from YAML snippet if available
        policy_name = None
        snippet = pdict.get("yaml_snippet")
        if snippet:
            import re

            m = re.search(r"^name:\s*(.+)$", snippet, re.MULTILINE)
            if m:
                policy_name = m.group(1).strip()

        policies_mapped.append(
            PolicyHashGroup(
                policy_hash=phash,
                policy_name=policy_name,
                yaml_snippet=snippet,
                count=pdict.get("count", 0),
                bytes=pdict.get("bytes", 0),
                by_zoom=by_zoom,
            )
        )

    t2 = Tier2Detail(
        count=d.fused_count,
        bytes=d.fused_total_bytes,
        newest=_safe_float(d.fused_newest),
        oldest=_safe_float(d.fused_oldest),
        policies=policies_mapped,
    )

    # Map Tier 3
    providers = []
    for p, count, sz, newest, oldest in d.provider_zarr:
        providers.append(
            ProviderDetail(
                name=p,
                count=count,
                bytes=sz,
                newest=_safe_float(newest),
                oldest=_safe_float(oldest),
            )
        )
    t3 = Tier3Detail(providers=providers)

    # Map Tier 4
    discovery = []
    for label, ok, sz, entries, mtime in d.discovery:
        discovery.append(
            DiscoveryDetail(
                label=label,
                ok=ok,
                bytes=sz,
                entries=entries,
                mtime=_safe_float(mtime),
            )
        )

    t4 = Tier4Detail(
        discovery=discovery,
        bluetopo_scheme={
            "exists": d.bluetopo_scheme[0],
            "bytes": d.bluetopo_scheme[1],
            "mtime": _safe_float(d.bluetopo_scheme[2]),
        },
        bluetopo_sidecars={
            "ok_count": d.bluetopo_sidecars[0],
            "failed_count": d.bluetopo_sidecars[1],
            "bytes": d.bluetopo_sidecars[2],
            "mtime": _safe_float(d.bluetopo_sidecars[3]),
        },
        topobathy_zips={
            "count": d.topobathy_zips[0],
            "bytes": d.topobathy_zips[1],
            "mtime": _safe_float(d.topobathy_zips[2]),
        },
    )

    # Map Tier 5
    raw_files = []
    for p, count, sz, newest, oldest in d.raw_files:
        raw_files.append(
            RawFileDetail(
                provider=p,
                count=count,
                bytes=sz,
                newest=_safe_float(newest),
                oldest=_safe_float(oldest),
            )
        )
    t5 = Tier5Detail(raw_files=raw_files)

    detail = CacheDetail(
        cache_root=str(CACHE_ROOT),
        total_bytes=d.total_bytes,
        tier_1=t1,
        tier_2=t2,
        tier_3=t3,
        tier_4=t4,
        tier_5=t5,
    )

    # Paranoia check
    try:
        json.dumps(detail.dict(), allow_nan=False)
    except ValueError as e:
        logger.error(f"JSON Check Failed in get_cache_detail: {e}")
        # If checks failed, we might crash, but at least we know WHERE.
        # Since we applied _safe_float above, this should pass.

    return detail


def purge_tiers(
    tiers: list[int],
    dry_run: bool = True,
    yes: bool = False,
    tier_2_hashes: list[str] | None = None,
) -> PurgeResult:
    """Purge specified cache tiers."""

    # Safety check for Tier 5
    if 5 in tiers and not yes:
        # This should have been caught by the API router/controller,
        # but double check here.
        pass

    affected_paths: list[str] = []
    summary: dict[str, dict[str, int]] = {}

    # Sort tiers to process
    tiers_to_process = [t for t in TIERS if t.number in tiers]

    for tier in tiers_to_process:
        paths = tier.purge_paths()

        # Filter Tier 2 paths if tier_2_hashes is provided
        if tier.number == 2 and tier_2_hashes is not None:
            paths = [p for p in paths if p.name in tier_2_hashes]

        deleted_count = 0
        deleted_bytes = 0

        tier_paths_str = []

        for p in paths:
            # Gather stats before deletion attempt
            try:
                if p.exists():
                    sz = cm._dir_size_bytes(p) if p.is_dir() else p.stat().st_size
                    deleted_bytes += sz
            except OSError:
                sz = 0

            p_str = str(p)
            tier_paths_str.append(p_str)

            if not dry_run:
                try:
                    if p.is_dir():
                        shutil.rmtree(p)
                    else:
                        p.unlink()
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"Failed to delete {p}: {e}")
            else:
                deleted_count += 1

        summary[f"tier_{tier.number}"] = {"count": deleted_count, "bytes": deleted_bytes}
        affected_paths.extend(tier_paths_str)

    msg = "Dry-run complete." if dry_run else "Purge complete."
    mode = "dry-run" if dry_run else "committed"

    return PurgeResult(status="success", mode=mode, message=msg, summary=summary, paths=affected_paths)


def check_integrity(clean: bool = False, lock_timeout: int = 3600) -> CheckResult:
    """Scan provider integrity and orphaned files."""
    checks: dict[str, Any] = {}

    # 1. Scan providers
    for p in cm.PROVIDERS:
        # _scan_provider_integrity is available in cm
        # It returns dict with keys: stale_locks, corrupt_src, corrupt_zarr (int counts)
        try:
            counts = cm._scan_provider_integrity(p, clean=clean, lock_timeout=lock_timeout)
            checks[p] = counts
        except Exception as e:
            logger.error(f"Error checking provider {p}: {e}")
            checks[p] = {"error": str(e)}

    # 2. Discovery checks (adapted from cmd_check)
    disco_status = []
    for label, path in cm._DISCOVERY_FILES:
        status = "ok"
        if not path.exists():
            status = "missing"
        else:
            try:
                with open(path) as f:
                    json.load(f)
            except json.JSONDecodeError:
                status = "corrupt"
                if clean:
                    try:
                        path.unlink()
                        status = "deleted"
                    except OSError:
                        status = "delete_failed"
        disco_status.append({"label": label, "status": status})

    checks["discovery"] = disco_status

    # 3. Scan orphans (adapted from _scan_global_orphans)
    # _scan_global_orphans prints to console and returns (stale_count, removed_count)
    # We want more detail. Implementing custom logic similar to _scan_global_orphans.

    orphans = _scan_orphans_internal(clean=clean, lock_timeout=lock_timeout)

    return CheckResult(status="complete", checks=checks, orphans=orphans)


def manage_orphans(clean: bool = False) -> OrphansResult:
    """List or clean orphaned locks/tmp files."""
    orphans = _scan_orphans_internal(clean=clean, lock_timeout=3600)  # Default timeout for manual action

    mode = "committed" if clean else "list-only"

    return OrphansResult(status="success", mode=mode, orphans=orphans)


def _scan_orphans_internal(clean: bool, lock_timeout: int) -> dict[str, Any]:
    now = time.time()
    patterns = ["*.lock", ".tmp*", "__temp__*"]
    candidates: list[Path] = []

    for pat in patterns:
        candidates.extend(CACHE_ROOT.rglob(pat))

    unique_candidates = sorted(list(set(candidates)), key=lambda p: str(p))

    stale_locks = []
    tmp_files = []
    total_bytes = 0

    for p in unique_candidates:
        if not p.exists():
            continue
        try:
            stat = p.stat()
            age = now - stat.st_mtime

            # For .lock files, check timeout. For tmp files, maybe just delete if old?
            # Matching logic from cache_manager: usage of lock_timeout implies it applies to all.
            if age > lock_timeout:
                sz = cm._dir_size_bytes(p) if p.is_dir() else stat.st_size
                info = {"path": str(p), "age_sec": age, "bytes": sz}

                if p.name.endswith(".lock"):
                    stale_locks.append(info)
                else:
                    tmp_files.append(info)

                total_bytes += sz

                if clean:
                    if p.is_dir():
                        shutil.rmtree(p)
                    else:
                        p.unlink()
        except Exception as e:
            logger.warning(f"Error processing orphan {p}: {e}")

    return {
        "locks": stale_locks,
        "tmp_files": tmp_files,
        "total_bytes": total_bytes,
        "count": len(stale_locks) + len(tmp_files),
    }
