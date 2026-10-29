"""
GDAL HTTP settings for remote raster reads.

Providers open cloud-hosted COGs through GDAL's /vsicurl/ (via rioxarray), and dask
performs the ranged reads later, on whatever thread computes the chunk. A
`rasterio.Env` block around the open would not cover those reads, so the settings
are applied as process environment defaults instead: GDAL reads its config options
from the environment at request time, and spawned hydration workers inherit them.

Without them GDAL uses its own defaults (no retry, no transfer timeout), so one
stalled range request can hold a hydration worker indefinitely. Any variable already
set in the environment wins over these defaults, which is also how to try transport
options such as GDAL_HTTP_VERSION=1.1 or GDAL_HTTP_MULTIPLEX=NO.
"""

import logging
import os
import threading

GDAL_HTTP_DEFAULTS: dict[str, str] = {
    # Seconds for a whole request, and for establishing the connection.
    "GDAL_HTTP_TIMEOUT": "120",
    "GDAL_HTTP_CONNECTTIMEOUT": "20",
    # Retries for transient HTTP errors (429, 5xx), with an initial delay in seconds.
    "GDAL_HTTP_MAX_RETRY": "4",
    "GDAL_HTTP_RETRY_DELAY": "2",
    # Do not list the remote "directory" when opening a single file.
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
}

# Messages GDAL emits when its curl state has failed. After one failed range read a
# worker has logged hundreds of thousands of them, so they are rate-limited.
_CURL_FAILURE_MARKERS = ("curl_multi_poll() failed", "failed with response_code=0")
CURL_FAILURE_LOG_LIMIT = 20


def apply_gdal_http_defaults() -> None:
    """Set GDAL_HTTP_DEFAULTS in os.environ, keeping any value already set."""
    for key, value in GDAL_HTTP_DEFAULTS.items():
        os.environ.setdefault(key, value)


class RepeatedCurlFailureFilter(logging.Filter):
    """Pass the first `limit` GDAL curl failure messages in a process, then drop them.

    The message after the limit is passed once more with a note that later ones are
    suppressed. Other messages are unaffected.
    """

    def __init__(self, limit: int = CURL_FAILURE_LOG_LIMIT) -> None:
        super().__init__()
        self.limit = limit
        self.count = 0
        self._lock = threading.Lock()

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if not any(marker in message for marker in _CURL_FAILURE_MARKERS):
            return True
        with self._lock:
            self.count += 1
            count = self.count
        if count <= self.limit:
            return True
        if count == self.limit + 1:
            record.msg = f"{message} (further GDAL curl failure messages in this process suppressed)"
            record.args = ()
            return True
        return False


def install_curl_failure_filter() -> None:
    """Attach one RepeatedCurlFailureFilter to the logger rasterio reports GDAL errors on."""
    gdal_logger = logging.getLogger("rasterio._env")
    if not any(isinstance(f, RepeatedCurlFailureFilter) for f in gdal_logger.filters):
        gdal_logger.addFilter(RepeatedCurlFailureFilter())
