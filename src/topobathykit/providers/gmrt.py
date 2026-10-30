"""
GMRT Provider module.

This module implements the provider for the Global Multi-Resolution Topography synthesis (GMRT),
maintained at Lamont-Doherty Earth Observatory. GMRT merges processed ship multibeam over a global
base; its ``topo-mask`` layer keeps only the cells backed by multibeam, so a fusion step with this
provider adds measured relief on the outer shelf and slope and lets lower steps show through
elsewhere.

Grids come from the GridServer web service (https://www.gmrt.org/services/) as GeoTIFF and are
cached by bounding box, node spacing, layer and GMRT version, so a new GMRT release starts a new
cache instead of mixing versions.

Vertical datum: GMRT depths are taken as relative to mean sea level and no VDatum conversion is
applied. Offshore, where this provider is meant to contribute, the chart-datum offsets are small
against depths of hundreds of metres.
"""

import logging
import re
import threading
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, ClassVar, cast

import numpy as np
import requests  # type: ignore
import rioxarray
import xarray as xr

from ..config import get_cache_root
from .base import Provider, ProviderFetchError, ProviderNoDataError, sanitize_elevation_nodata, source_uid
from .registry import registry

logger = logging.getLogger(__name__)

GRIDSERVER_URL = "https://www.gmrt.org/services/GridServer"
METADATA_URL = f"{GRIDSERVER_URL}/metadata"

LAYERS = ("topo-mask", "topo")
DEFAULT_LAYER = "topo-mask"

CITATION = (
    "Ryan, W.B.F., S.M. Carbotte, J.O. Coplan, S. O'Hara, A. Melkonian, R. Arko, R.A. Weissel, "
    "V. Ferrini, A. Goodwillie, F. Nitsche, J. Bonczkowski, and R. Zemsky (2009), Global "
    "Multi-Resolution Topography synthesis, Geochem. Geophys. Geosyst., 10, Q03014, "
    "doi:10.1029/2008GC002332."
)
LICENSE = "CC BY 4.0"

REQUEST_TIMEOUT_S = 120


def _bbox_key(bbox: tuple[float, float, float, float]) -> str:
    """Filesystem-safe key for a WGS84 bbox, to 1e-5 degrees (about 1 m)."""
    return "_".join(f"{v:+.5f}" for v in bbox).replace("+", "p").replace("-", "m").replace(".", "d")


def _version_dir(version: str) -> str:
    return "v" + re.sub(r"[^0-9A-Za-z]+", "_", version)


class GMRTProvider(Provider):
    """
    Provider for GMRT (Global Multi-Resolution Topography) from the GridServer web service.

    Policy options, given under the step's ``filter``:

    - ``layer``: ``topo-mask`` (default; multibeam cells only) or ``topo`` (multibeam over the
      global base).
    """

    # GridServer metadata per (bbox, resolution) for the life of the process. Not persisted, so
    # every new process learns the current GMRT version before it trusts the cache.
    _metadata_memo: ClassVar[dict[tuple[str, str], dict[str, Any]]] = {}
    _locks: ClassVar[dict[str, threading.Lock]] = {}
    _locks_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, cache_dir: str | Path | None = None):
        """
        Initialize the GMRT provider.

        Args:
            cache_dir: Cache root; grids are stored under ``<cache_dir>/gmrt``.
        """
        if cache_dir is None:
            cache_dir = get_cache_root()
        self.cache_dir = Path(cache_dir).expanduser() / "gmrt"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _query(bbox: tuple[float, float, float, float], resolution: float | None) -> dict[str, Any]:
        """GridServer query parameters for a bbox and a requested node spacing in metres.

        ``mresolution`` makes the server pick the coarsest of its levels that is at least as fine
        as the request; with no resolution the finest level (``max``) is asked for.
        """
        west, south, east, north = bbox
        params: dict[str, Any] = {
            "minlongitude": west,
            "maxlongitude": east,
            "minlatitude": south,
            "maxlatitude": north,
        }
        if resolution is not None and resolution > 0:
            params["mresolution"] = float(resolution)
        else:
            params["resolution"] = "max"
        return params

    def _metadata(self, bbox: tuple[float, float, float, float], resolution: float | None) -> dict[str, Any]:
        """GMRT version and node spacing the GridServer will deliver for this request."""
        res_key = "max" if resolution is None or resolution <= 0 else f"{float(resolution):g}"
        memo_key = (_bbox_key(bbox), res_key)
        cached = GMRTProvider._metadata_memo.get(memo_key)
        if cached is not None:
            return cached

        resp = requests.get(METADATA_URL, params=self._query(bbox, resolution), timeout=REQUEST_TIMEOUT_S)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)

        def text(tag: str) -> str | None:
            node = root.find(tag)
            return node.text.strip() if node is not None and node.text else None

        version = text("gmrt_version")
        mpn = text("meters_per_node")
        if not version or not mpn:
            raise ValueError(f"GMRT metadata lacks gmrt_version or meters_per_node: {resp.text[:200]}")
        meta: dict[str, Any] = {
            "gmrt_version": version,
            "gmrt_release_date": text("gmrt_release_date"),
            "meters_per_node": float(mpn),
        }
        GMRTProvider._metadata_memo[memo_key] = meta
        return meta

    def _cache_path(self, bbox: tuple[float, float, float, float], layer: str, meta: dict[str, Any]) -> Path:
        return (
            self.cache_dir
            / _version_dir(meta["gmrt_version"])
            / layer
            / f"{_bbox_key(bbox)}_{meta['meters_per_node']:g}m.tif"
        )

    def _offline_fallback(self, bbox: tuple[float, float, float, float], layer: str) -> Path | None:
        """Newest cached grid for this bbox and layer, of any version, when the server is unreachable."""
        hits = sorted(
            self.cache_dir.glob(f"v*/{layer}/{_bbox_key(bbox)}_*m.tif"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return hits[0] if hits else None

    def _download(
        self, bbox: tuple[float, float, float, float], resolution: float | None, layer: str, dest: Path
    ) -> None:
        params = self._query(bbox, resolution)
        params.update({"format": "geotiff", "layer": layer})
        resp = requests.get(GRIDSERVER_URL, params=params, timeout=REQUEST_TIMEOUT_S)
        resp.raise_for_status()
        # A GeoTIFF starts with a TIFF byte-order mark; the server answers errors with text.
        if resp.content[:4] not in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"):
            raise ValueError(f"GMRT GridServer did not return a GeoTIFF: {resp.content[:200]!r}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".tif.part")
        tmp.write_bytes(resp.content)
        tmp.replace(dest)

    def fetch_layer(
        self,
        bbox: tuple[float, float, float, float],
        resolution: float | None = None,
        crs: str = "EPSG:4326",
        **kwargs: Any,
    ) -> xr.Dataset:
        """
        Fetch GMRT elevation for the bbox at the GMRT level nearest to (and not coarser than)
        ``resolution`` metres.

        Returns a Dataset with ``elevation`` (float32, metres relative to MSL, NaN outside the
        layer's coverage) and ``source_id`` (uint32), on the GridServer's EPSG:4326 grid.
        """
        options = kwargs.get("filter") or {}
        layer = str(options.get("layer") or DEFAULT_LAYER)
        if layer not in LAYERS:
            raise ValueError(f"GMRT layer must be one of {LAYERS}, got {layer!r}")

        wgs_bbox = tuple(float(v) for v in self._normalize_bbox(bbox, crs))
        bbox4 = cast(tuple[float, float, float, float], wgs_bbox)

        try:
            meta = self._metadata(bbox4, resolution)
        except (requests.RequestException, ET.ParseError, ValueError) as e:
            fallback = self._offline_fallback(bbox4, layer)
            if fallback is None:
                raise ProviderFetchError(f"GMRT metadata request failed: {e}") from e
            logger.warning(f"GMRT metadata unavailable ({e}); using cached {fallback.name}")
            path = fallback
            meta = {
                "gmrt_version": fallback.parent.parent.name.removeprefix("v").replace("_", "."),
                "gmrt_release_date": None,
                "meters_per_node": float(fallback.stem.rsplit("_", 1)[-1].removesuffix("m")),
            }
        else:
            path = self._cache_path(bbox4, layer, meta)
            if path.exists():
                logger.info(f"GMRT Cache Hit: {path.name} (v{meta['gmrt_version']}, {layer})")
            else:
                with GMRTProvider._locks_lock:
                    thread_lock = GMRTProvider._locks.setdefault(str(path), threading.Lock())
                from filelock import FileLock

                path.parent.mkdir(parents=True, exist_ok=True)
                with thread_lock, FileLock(path.with_suffix(".tif.lock")):
                    if not path.exists():
                        logger.info(
                            f"GMRT Cache Miss: downloading {layer} at {meta['meters_per_node']:g} m "
                            f"(v{meta['gmrt_version']})"
                        )
                        try:
                            self._download(bbox4, resolution, layer, path)
                        except (requests.RequestException, ValueError, OSError) as e:
                            raise ProviderFetchError(f"GMRT GridServer request failed: {e}") from e

        try:
            with cast(xr.DataArray, rioxarray.open_rasterio(path, masked=True)) as raw:
                da = raw.squeeze("band", drop=True).load()
        except Exception as e:
            path.unlink(missing_ok=True)
            raise ProviderFetchError(f"Corrupt GMRT cache file {path.name}: {e}") from e

        if da.rio.crs is None:
            da = da.rio.write_crs("EPSG:4326")
        elevation = sanitize_elevation_nodata(da).astype(np.float32)
        elevation.name = "elevation"
        valid = np.isfinite(elevation.values)
        if not valid.any():
            raise ProviderNoDataError(f"GMRT {layer} has no data in {bbox4}")

        version = meta["gmrt_version"]
        sid = source_uid("gmrt", f"{version}:{layer}")
        source_id = xr.DataArray(
            np.where(valid, sid, 0).astype(np.uint32),
            coords=elevation.coords,
            dims=elevation.dims,
            name="source_id",
        )

        ds = xr.Dataset({"elevation": elevation, "source_id": source_id})
        ds = ds.rio.write_crs("EPSG:4326")
        # GMRT's GridServer does not list the contributing cruises for a bbox, so provenance
        # records the release and layer only.
        ds.attrs["provenance_dict"] = {
            sid: {
                "name": f"GMRT {version} ({layer})",
                "provider": "gmrt",
                "product": layer,
                "gmrt_version": version,
                "meters_per_node": f"{meta['meters_per_node']:g}",
                "vertical_datum": "MSL",
                "citation": self.citation(version),
                "license": LICENSE,
            }
        }
        ds.attrs["vertical_datum"] = "MSL"
        ds.attrs["gmrt_version"] = version
        logger.debug(f"GMRT {layer}: {valid.mean():.1%} valid, {meta['meters_per_node']:g} m/node")
        return cast(xr.Dataset, ds)

    @staticmethod
    def citation(version: str | None = None) -> str:
        """The required GMRT citation, naming the release when known."""
        release = f" GMRT version {version}." if version else ""
        return f"{CITATION}{release}"

    def get_metadata(self) -> dict[str, Any]:
        """
        Return metadata for the GMRT provider.

        Returns:
            Dictionary containing provider name, citation, licence, resolution, and URL.
        """
        return {
            "name": "GMRT",
            "citation": self.citation(),
            "license": LICENSE,
            "resolution": "variable (multibeam to about 30 m; global base about 500 m)",
            "url": GRIDSERVER_URL,
            "vertical_datum": "MSL",
        }


# Register the provider
registry.register(Path(__file__).stem, GMRTProvider)
