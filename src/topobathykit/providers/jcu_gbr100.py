"""
JCU GBR100 Provider module.

This module implements the provider for the Great Barrier Reef (GBR) 100m bathymetry grid
hosted by AusSeabed. It handles downloading, caching, and serving the data as xarray DataArrays.
"""

import logging
import zipfile
from pathlib import Path
from typing import Any, cast

import requests  # type: ignore
import rioxarray
import xarray as xr

from ..config import get_cache_root
from .base import Provider
from .registry import registry

logger = logging.getLogger(__name__)


class GBR100Provider(Provider):
    """
    Provider for GBR100 (Great Barrier Reef 100m Bathymetry).
    Source: AusSeabed / Geoscience Australia
    URL: https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/133163
    """

    DOWNLOAD_URL = (
        "https://files.ausseabed.gov.au/survey/Great%20Barrier%20Reef%20Bathymetry%202020%20100m.zip"
    )
    ZIP_FILENAME = "Great_Barrier_Reef_Bathymetry_2020_100m.zip"

    def __init__(self, cache_dir: str | Path | None = None):
        """
        Initialize the GBR100 provider.

        Args:
            cache_dir: Directory to store cached data files.
        """
        if cache_dir is None:
            cache_dir = get_cache_root()

        self.cache_dir = Path(cache_dir).expanduser() / "gbr100"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.zip_path = self.cache_dir / self.ZIP_FILENAME
        self.extracted_dir = self.cache_dir / "extracted"

    def _ensure_data_ready(self) -> Path:
        """
        Ensures data is downloaded and extracted.
        Returns path to the main grid file (nc/grd/tif).
        """
        from filelock import FileLock

        lock_path = self.zip_path.with_suffix(".lock")

        with FileLock(lock_path):
            # 1. Download if missing
            if not self.zip_path.exists():
                logger.info(f"Downloading GBR100 from {self.DOWNLOAD_URL}...")
                try:
                    with requests.get(self.DOWNLOAD_URL, stream=True, timeout=120) as r:
                        r.raise_for_status()
                        with open(self.zip_path, "wb") as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)
                    logger.info("GBR100 download complete.")
                except Exception as e:
                    # Clean up partial download
                    if self.zip_path.exists():
                        self.zip_path.unlink()
                    raise RuntimeError(f"Failed to download GBR100: {e}") from e

            # 2. Extract
            if not self.extracted_dir.exists() or not any(self.extracted_dir.iterdir()):
                logger.info(f"Extracting GBR100 to {self.extracted_dir}...")
                self.extracted_dir.mkdir(parents=True, exist_ok=True)
                try:
                    with zipfile.ZipFile(self.zip_path, "r") as z:
                        z.extractall(self.extracted_dir)
                except Exception as e:
                    import shutil

                    shutil.rmtree(self.extracted_dir, ignore_errors=True)
                    raise RuntimeError(f"Failed to unzip GBR100: {e}") from e

        # 3. Find Data File (.nc, .grd, .tif)
        candidates = (
            list(self.extracted_dir.glob("*.nc"))
            + list(self.extracted_dir.glob("*.grd"))
            + list(self.extracted_dir.glob("*.tif"))
        )

        if not candidates:
            # Check subdirectories
            candidates = list(self.extracted_dir.rglob("*.nc")) + list(self.extracted_dir.rglob("*.grd"))

        if not candidates:
            raise FileNotFoundError("No recognized grid file (.nc, .grd, .tif) found in GBR100 zip.")

        # Return first candidate
        return candidates[0]

    def fetch_layer(
        self,
        bbox: tuple[float, float, float, float],
        resolution: float | None = None,
        crs: str = "EPSG:4326",
        **kwargs: Any,
    ) -> xr.DataArray:
        """
        Fetches GBR100 data for the given bounding box.
        """
        try:
            data_path = self._ensure_data_ready()

            # Load lazily
            da_raw = rioxarray.open_rasterio(data_path, chunks="auto")  # type: ignore

            # Handle return types
            if isinstance(da_raw, list):
                da = cast(xr.DataArray, da_raw[0])
            elif isinstance(da_raw, xr.Dataset):
                da = da_raw.to_array().isel(variable=0)
            else:
                da = cast(xr.DataArray, da_raw)

            if "band" in da.dims:
                da = da.squeeze("band", drop=True)

            # GBR100 is typically EPSG:4326
            if da.rio.crs is None:
                da.rio.write_crs("EPSG:4326", inplace=True)

            # Reproject
            if crs and da.rio.crs != crs:
                logger.debug(f"Reprojecting GBR100 to {crs}")
                da = da.rio.reproject(crs)

            # Clip
            wd, sd, ed, nd = bbox
            da = da.rio.clip_box(minx=wd, miny=sd, maxx=ed, maxy=nd, allow_one_dimensional_raster=True)
            da.name = "elevation"

            return cast(xr.DataArray, da)

        except Exception as e:
            logger.error(f"Failed to load GBR100 data: {e}")
            raise e

    def get_metadata(self) -> dict[str, Any]:
        """
        Return metadata for the GBR100 provider.

        Returns:
            Dictionary containing provider name, citation, resolution, and URL.
        """
        return {
            "name": "GBR100 v6",
            "citation": "Deepreef Explorer (2020). Great Barrier Reef 100m Bathymetry.",
            "resolution": "100m",
            "url": "https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/133163",
        }


# Register the provider
registry.register(Path(__file__).stem, GBR100Provider)
