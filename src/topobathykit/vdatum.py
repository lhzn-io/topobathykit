import logging
import os
import sqlite3
import time
from collections.abc import Callable
from functools import lru_cache, wraps
from pathlib import Path
from typing import Any

import requests  # type: ignore

from .config import get_cache_root

logger = logging.getLogger(__name__)

# Respects TOPOBATHYKIT_CACHE_DIR so the db lands on the mounted volume in Docker
VDATUM_CACHE_DIR = get_cache_root()
VDATUM_DB_PATH = VDATUM_CACHE_DIR / "vdatum.sqlite"

_DDL = """
    CREATE TABLE IF NOT EXISTS vdatum_offsets (
        offset_type TEXT,
        lat REAL,
        lon REAL,
        offset REAL,
        PRIMARY KEY (offset_type, lat, lon)
    )
"""


def _connect_and_create(timeout: float) -> None:
    with sqlite3.connect(VDATUM_DB_PATH, timeout=timeout) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute(_DDL)


def _init_db() -> None:
    """Initialize the SQLite WAL database and layout schemas."""
    VDATUM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        _connect_and_create(timeout=30.0)
    except sqlite3.OperationalError as e:
        if "locked" not in str(e):
            raise
        # Stale WAL/SHM files from a killed container; safe to remove since the
        # previous process is gone and the kernel has already released its locks.
        for suffix in ("-shm", "-wal"):
            stale = Path(str(VDATUM_DB_PATH) + suffix)
            if stale.exists():
                stale.unlink(missing_ok=True)
        _connect_and_create(timeout=30.0)


# Initialize on module load
try:
    _init_db()
except sqlite3.Error as e:
    logger.warning(f"Failed to initialize VDatum SQLite cache: {e}")


def sqlite_vdatum_cache(offset_type: str) -> Callable:
    """
    Decorator for L2 SQLite caching of VDatum API calls.
    Mitigates completely the file lock contention seen with .lock files
    by utilizing SQLite WAL-mode concurrency.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(lat: float, lon: float) -> float:
            try:
                # 1. Read L2 hit
                with sqlite3.connect(VDATUM_DB_PATH, timeout=30.0) as conn:
                    crs = conn.execute(
                        "SELECT offset FROM vdatum_offsets WHERE offset_type=? AND lat=? AND lon=?",
                        (offset_type, lat, lon),
                    )
                    row = crs.fetchone()
                    if row is not None:
                        return float(row[0])
            except sqlite3.Error as e:
                logger.debug(f"SQLite read bypassed for ({lat},{lon}): {e}")

            # 2. Fetch value natively through HTTP (this throws VDatumNoDataError gracefully)
            val = func(lat, lon)

            # 3. Write L2 entry asynchronously capable via WAL
            try:
                with sqlite3.connect(VDATUM_DB_PATH, timeout=30.0) as conn:
                    conn.execute("PRAGMA journal_mode=WAL")
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO vdatum_offsets
                        (offset_type, lat, lon, offset) VALUES (?, ?, ?, ?)
                        """,
                        (offset_type, lat, lon, val),
                    )
            except sqlite3.Error as e:
                logger.debug(f"SQLite write bypassed for ({lat},{lon}): {e}")

            return float(val)

        return wrapper

    return decorator


class VDatumError(Exception):
    """Base class for VDatum errors."""

    pass


class VDatumNoDataError(VDatumError):
    """Raised when VDatum returns a NoData value (outside valid region)."""

    pass


class VDatumUnavailableError(VDatumError):
    """Raised when the VDatum API stays unreachable (timeouts, 5xx) after bounded retries.

    Unlike `VDatumNoDataError`, this says nothing about the location, so callers
    should treat it as transient rather than as a lack of coverage.
    """


# Bounded retry for the VDatum API. Worst case per request with the defaults:
# 3 attempts x 10 s read timeout + 2 s + 4 s backoff = 36 s.
VDATUM_TIMEOUT_S = 10.0
VDATUM_BACKOFF_S = 2.0
_VDATUM_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


def _vdatum_max_attempts() -> int:
    """Number of VDatum API attempts, from TOPOBATHY_VDATUM_ATTEMPTS (default 3, minimum 1)."""
    try:
        return max(1, int(os.environ.get("TOPOBATHY_VDATUM_ATTEMPTS", "3")))
    except ValueError:
        return 3


class VDatumResolver:
    """
    Resolves vertical datum offsets between NAVD88 and Local Mean Sea Level (LMSL)
    using NOAA's VDatum API. Supports GEOID18 for Ellipsoid to NAVD88 conversions.
    """

    VDATUM_API = "https://vdatum.noaa.gov/vdatumweb/api/convert"

    @staticmethod
    def _get_session() -> requests.Session:
        """Creates a session for VDatum API calls.

        Retries are handled in `_get_json` rather than by a urllib3 adapter, so
        that timeouts are retried a bounded number of times and not multiplied
        by a second retry layer.
        """
        session = requests.Session()
        ua = (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(HTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        session.headers.update({"User-Agent": ua})
        return session

    @staticmethod
    def _get_json(params: dict[str, Any]) -> Any:
        """GET the VDatum API and return the decoded JSON body.

        Timeouts, connection errors and retryable HTTP statuses are retried with
        exponential backoff up to `_vdatum_max_attempts()` attempts, after which
        `VDatumUnavailableError` is raised. Other HTTP errors raise immediately.
        """
        attempts = _vdatum_max_attempts()
        session = VDatumResolver._get_session()
        last_exc: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                response = session.get(VDatumResolver.VDATUM_API, params=params, timeout=VDATUM_TIMEOUT_S)
                response.raise_for_status()
                return response.json()
            except (requests.Timeout, requests.ConnectionError) as e:
                last_exc = e
            except requests.HTTPError as e:
                status = getattr(e.response, "status_code", None)
                if status not in _VDATUM_RETRY_STATUS:
                    raise
                last_exc = e

            if attempt < attempts:
                delay = VDATUM_BACKOFF_S * (2 ** (attempt - 1))
                logger.warning(
                    f"VDatum API attempt {attempt}/{attempts} failed ({type(last_exc).__name__}: "
                    f"{last_exc}); retrying in {delay:.0f} s"
                )
                time.sleep(delay)

        raise VDatumUnavailableError(
            f"VDatum API unavailable after {attempts} attempt(s): {type(last_exc).__name__}: {last_exc}"
        ) from last_exc

    @staticmethod
    @lru_cache(maxsize=1024)
    @sqlite_vdatum_cache("navd88_lmsl")
    def get_navd88_to_lmsl_offset(lat: float, lon: float) -> float:
        """
        Calculates the vertical shift to convert NAVD88 elevation to LMSL.
        LMSL = NAVD88 - Offset
        """
        # API requires source/target/region
        params: dict[str, Any] = {
            "s_x": lon,
            "s_y": lat,
            "s_z": 0.0,  # Explicitly convert 0
            "s_hframe": "NAD83_2011",
            "s_vframe": "NAVD88",
            "t_hframe": "NAD83_2011",
            "t_vframe": "LMSL",
            "region": "contiguous",
            "result_format": "json",
        }

        try:
            data = VDatumResolver._get_json(params)

            if "t_z" in data:
                val = float(data["t_z"])
                if val < -90000.0 or val > 90000.0:
                    raise VDatumNoDataError(f"VDatum API returned NoData value: {val}")
                return val

            raise ValueError(f"VDatum API returned no elevation data: {data}")

        except VDatumNoDataError:
            raise
        except Exception as e:
            logger.error(f"Datum conversion failed for ({lat}, {lon}): {e}")
            raise

    @staticmethod
    @lru_cache(maxsize=1024)
    @sqlite_vdatum_cache("mllw_navd88")
    def get_mllw_to_navd88_offset(lat: float, lon: float) -> float:
        """
        Calculates shift from MLLW to NAVD88.
        NAVD88 = MLLW + Offset
        (Actually VDatum output is Target Height, so if Input=0 MLLW, Output=Z NAVD88)
        """
        params: dict[str, Any] = {
            "s_x": lon,
            "s_y": lat,
            "s_z": 0.0,
            "s_hframe": "NAD83_2011",
            "s_vframe": "MLLW",
            "t_hframe": "NAD83_2011",
            "t_vframe": "NAVD88",
            "region": "contiguous",
            "result_format": "json",
        }

        try:
            data = VDatumResolver._get_json(params)

            if "t_z" in data:
                # If s_z=0, t_z is the height of MLLW zero in NAVD88 frame.
                val = float(data["t_z"])
                if val < -90000.0 or val > 90000.0:
                    # Don't log error here, let the caller handle it (it's expected for inland points)
                    raise VDatumNoDataError(f"VDatum API returned NoData value: {val}")
                return val

            raise ValueError(f"VDatum API returned no elevation data: {data}")

        except VDatumNoDataError:
            raise
        except Exception as e:
            logger.error(f"Datum (MLLW->NAVD88) conversion failed for ({lat}, {lon}): {e}")
            raise

    @staticmethod
    @lru_cache(maxsize=1024)
    @sqlite_vdatum_cache("ellipsoid_navd88")
    def get_ellipsoid_to_navd88_offset(lat: float, lon: float) -> float:
        """
        Calculates shift from NAD83(2011) Ellipsoid to NAVD88 using GEOID18.
        NAVD88 = Ellipsoid - GeoidHeight
        """
        params: dict[str, Any] = {
            "s_x": lon,
            "s_y": lat,
            "s_z": 0.0,
            "s_hframe": "NAD83_2011",
            "s_vframe": "NAD83_2011",  # Ellipsoid
            "t_hframe": "NAD83_2011",
            "t_vframe": "NAVD88",
            "region": "contiguous",
            "geoid": "GEOID18",  # Explicitly request GEOID18
            "result_format": "json",
        }

        try:
            data = VDatumResolver._get_json(params)

            if "t_z" in data:
                val = float(data["t_z"])
                if val < -90000.0 or val > 90000.0:
                    raise VDatumNoDataError(f"VDatum API returned NoData value: {val}")
                return val

            raise ValueError(f"VDatum API returned no elevation data: {data}")

        except VDatumNoDataError:
            raise
        except Exception as e:
            logger.error(f"Datum (Ellipsoid->NAVD88) conversion failed for ({lat}, {lon}): {e}")
            raise

    @staticmethod
    @lru_cache(maxsize=1024)
    @sqlite_vdatum_cache("egm2008_navd88")
    def get_egm2008_to_navd88_offset(lat: float, lon: float) -> float:
        """
        Calculates shift from EGM2008 to NAVD88.
        NAVD88 = EGM2008 + Offset (t_z)
        """
        params: dict[str, Any] = {
            "s_x": lon,
            "s_y": lat,
            "s_z": 0.0,
            "s_hframe": "WGS84_G1674",  # EGM2008 implies WGS84 usually
            "s_vframe": "EGM2008",
            "t_hframe": "NAD83_2011",
            "t_vframe": "NAVD88",
            "region": "contiguous",
            "result_format": "json",
        }

        try:
            data = VDatumResolver._get_json(params)

            if "t_z" in data:
                val = float(data["t_z"])
                if val < -90000.0 or val > 90000.0:
                    raise VDatumNoDataError(f"VDatum API returned NoData value: {val}")
                return val

            raise ValueError(f"VDatum API returned no elevation data: {data}")

        except VDatumNoDataError:
            raise
        except Exception as e:
            logger.error(f"Datum (EGM2008->NAVD88) conversion failed for ({lat}, {lon}): {e}")
            raise

    @staticmethod
    def get_robust_ellipsoid_to_navd88_offset(
        lat: float, lon: float, search_radius: float = 0.05, steps: int = 5
    ) -> float:
        """
        Attempts to find a valid Ellipsoid->NAVD88 offset by searching spirally outward.
        Returns the average of valid points found at the nearest valid radius.
        """
        try:
            return float(VDatumResolver.get_ellipsoid_to_navd88_offset(lat, lon))
        except VDatumNoDataError:
            pass

        step_deg = search_radius / steps
        for r in range(1, steps + 1):
            curr_dist = r * step_deg
            offsets = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
            valid_vals = []
            for dy, dx in offsets:
                t_lat = lat + (dy * curr_dist)
                t_lon = lon + (dx * curr_dist)
                try:
                    val = VDatumResolver.get_ellipsoid_to_navd88_offset(t_lat, t_lon)
                    valid_vals.append(val)
                except VDatumNoDataError:
                    continue
            if valid_vals:
                return float(sum(valid_vals) / len(valid_vals))

        raise VDatumNoDataError(f"VDatum robust search failed within {search_radius} deg of ({lat}, {lon})")

    @staticmethod
    def get_robust_mllw_to_navd88_offset(
        lat: float, lon: float, search_radius: float = 0.05, steps: int = 5
    ) -> float:
        """
        Attempts to find a valid MLLW->NAVD88 offset by searching spirally outward.
        Useful when the exact point (e.g. tile center) falls on land or a hole in the model.
        Returns the average of valid points found at the nearest valid radius.
        """
        # 1. Try exact point first
        try:
            return float(VDatumResolver.get_mllw_to_navd88_offset(lat, lon))
        except VDatumNoDataError:
            pass

        # 2. Spiral Search
        step_deg = search_radius / steps
        for r in range(1, steps + 1):
            curr_dist = r * step_deg
            # Check 8 directions
            offsets = [
                (0, 1),
                (0, -1),
                (1, 0),
                (-1, 0),  # Cardinal
                (1, 1),
                (1, -1),
                (-1, 1),
                (-1, -1),  # Diagonals
            ]

            valid_vals = []
            for dy, dx in offsets:
                t_lat = lat + (dy * curr_dist)
                t_lon = lon + (dx * curr_dist)
                try:
                    val = VDatumResolver.get_mllw_to_navd88_offset(t_lat, t_lon)
                    valid_vals.append(val)
                except VDatumNoDataError:
                    continue

            if valid_vals:
                # Return average of valid samples at this ring
                return float(sum(valid_vals) / len(valid_vals))

        # Fallback for very specific cases or raise
        raise VDatumNoDataError(f"VDatum robust search failed within {search_radius} deg of ({lat}, {lon})")
