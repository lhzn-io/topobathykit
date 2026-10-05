import logging
import os

from fastapi import APIRouter, HTTPException, status

from topobathykit.api import cache_api
from topobathykit.api.models import (
    CacheDetail,
    CacheSummary,
    CacheTierInfo,
    CheckRequest,
    CheckResult,
    OrphansRequest,
    OrphansResult,
    PurgeRequest,
    PurgeResult,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/cache", tags=["Cache Viewer"])

# Security Guard
ENABLE_DESTRUCTIVE = os.environ.get("CACHE_UI_ENABLE_DESTRUCTIVE", "false").lower() == "true"


@router.get("/tiers", response_model=list[CacheTierInfo])
async def get_tiers() -> list[CacheTierInfo]:
    """Return static metadata about cache tiers."""
    return cache_api.get_tiers()


@router.get("/summary", response_model=CacheSummary)
async def get_summary(cache_bust: bool = False) -> CacheSummary:
    """Return a lightweight summary of cache tiers."""
    return cache_api.get_cache_summary(cache_bust=cache_bust)


@router.get("/detail", response_model=CacheDetail)
async def get_detail(cache_bust: bool = True) -> CacheDetail:
    """Return full detailed inventory of the cache."""
    return cache_api.get_cache_detail(cache_bust=cache_bust)


@router.post("/purge", response_model=PurgeResult)
async def purge_cache(request: PurgeRequest) -> PurgeResult:
    """Purge specified cache tiers."""

    # 1. Validation
    if not request.tiers:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No tiers specified.")

    for t in request.tiers:
        if not (1 <= t <= 5):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid tier number: {t}")

    # 2. Safety Checks
    is_destructive = not request.dry_run

    # Check if destructive actions are enabled generally
    if is_destructive and not ENABLE_DESTRUCTIVE:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Destructive operations are disabled on this server. Set CACHE_UI_ENABLE_DESTRUCTIVE=true."
            ),
        )

    # Check specifically for Tier 5 (Raw Files) warning
    if 5 in request.tiers and is_destructive and not request.yes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tier 5 (Raw source files) is destructive. You must set 'yes=true' to confirm.",
        )

    # 3. Execution
    try:
        if is_destructive:
            logger.warning(f"Purging cache tiers {request.tiers} (user request)")

        result = cache_api.purge_tiers(
            tiers=request.tiers,
            dry_run=request.dry_run,
            yes=request.yes,
            tier_2_hashes=request.tier_2_hashes,
        )
        return result

    except Exception as e:
        logger.error(f"Purge failed: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e


@router.post("/check", response_model=CheckResult)
async def check_integrity(request: CheckRequest) -> CheckResult:
    """Audit cache integrity (read-only by default; optional clean)."""

    if request.clean and not ENABLE_DESTRUCTIVE:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Clean operations are disabled. Set CACHE_UI_ENABLE_DESTRUCTIVE=true.",
        )

    try:
        return cache_api.check_integrity(clean=request.clean, lock_timeout=request.lock_timeout)
    except Exception as e:
        logger.error(f"Integrity check failed: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e


@router.post("/orphans", response_model=OrphansResult)
async def manage_orphans(request: OrphansRequest) -> OrphansResult:
    """List or delete orphaned lock files and tmp artifacts."""

    if request.clean and not ENABLE_DESTRUCTIVE:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Clean operations are disabled. Set CACHE_UI_ENABLE_DESTRUCTIVE=true.",
        )

    try:
        return cache_api.manage_orphans(clean=request.clean)
    except Exception as e:
        logger.error(f"Orphan management failed: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e
