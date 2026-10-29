import contextlib
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env variables
env_path = Path(__file__).resolve().parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Disable PROJ network to avoid HTTP 404s for obscure regional datum shift grids
# (e.g. Canadian shifts like NS778301.gsb or NVI27_05.GSB) that fail to download.
# We don't need mm-level horizontal shifts for our 15m resolution bathymetry tiles.
# NOTE: Forced overwrite because Conda activate scripts explicitly set PROJ_NETWORK=ON.
os.environ["PROJ_NETWORK"] = "OFF"
with contextlib.suppress(ImportError):
    import pyproj

    pyproj.network.set_network_enabled(False)

# Timeouts and retries for remote COG reads, and a cap on GDAL curl failure spam
# (see gdal_http). Set after .env so values there take precedence.
from .gdal_http import apply_gdal_http_defaults, install_curl_failure_filter  # noqa: E402

apply_gdal_http_defaults()
install_curl_failure_filter()

from .providers.registry import registry  # noqa: E402
from .quality import TIDClassifier, source_report  # noqa: E402

# Set up NullHandler to prevent "No handler found" warnings
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "TIDClassifier",
    "registry",
    "source_report",
]
