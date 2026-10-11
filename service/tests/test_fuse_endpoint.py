"""Tests for the /fuse endpoint with Zarr and GeoTIFF output formats."""

# Add parent package to path for imports
import sys
from pathlib import Path
from pathlib import Path as PathlibPath
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr
from fastapi import HTTPException
from fastapi.testclient import TestClient

sys.path.insert(0, str(PathlibPath(__file__).parent.parent))

from topobathyserve.main import _normalize_output_format, _parse_bbox_params, app


class TestBboxParsing:
    """Test bbox parameter parsing and validation."""

    def test_bbox_string_valid(self) -> None:
        """Test parsing valid bbox string."""
        bbox_str = "-74.0,40.5,-73.5,41.0"
        west, south, east, north = _parse_bbox_params(bbox_str, None, None, None, None)
        assert west == -74.0
        assert south == 40.5
        assert east == -73.5
        assert north == 41.0

    def test_bbox_individual_params_valid(self) -> None:
        """Test parsing individual bbox parameters."""
        west, south, east, north = _parse_bbox_params(None, -74.0, 40.5, -73.5, 41.0)
        assert west == -74.0
        assert south == 40.5
        assert east == -73.5
        assert north == 41.0

    def test_bbox_invalid_ordering(self) -> None:
        """Test that invalid bbox ordering raises error."""
        with pytest.raises(HTTPException):
            _parse_bbox_params(None, -73.5, 40.5, -74.0, 41.0)  # west > east

        with pytest.raises(HTTPException):
            _parse_bbox_params(None, -74.0, 41.0, -73.5, 40.5)  # south > north

    def test_bbox_missing_params(self) -> None:
        """Test that missing bbox parameters raise error."""
        with pytest.raises(HTTPException):
            _parse_bbox_params(None, -74.0, 40.5, None, 41.0)

    def test_bbox_string_malformed(self) -> None:
        """Test that malformed bbox string raises error."""
        with pytest.raises(HTTPException):
            _parse_bbox_params("-74.0,40.5,-73.5", None, None, None, None)

        with pytest.raises(HTTPException):
            _parse_bbox_params("-74.0,40.5,invalid,-73.5,41.0", None, None, None, None)


class TestOutputFormatNormalization:
    """Test output format parameter normalization."""

    def test_geotiff_variants(self) -> None:
        """Test GeoTIFF format variants are normalized."""
        assert _normalize_output_format("geotiff") == "geotiff"
        assert _normalize_output_format("tiff") == "geotiff"
        assert _normalize_output_format("tif") == "geotiff"
        assert _normalize_output_format("GEOTIFF") == "geotiff"
        assert _normalize_output_format("TIF") == "geotiff"

    def test_zarr_variants(self) -> None:
        """Test Zarr format variants are normalized."""
        assert _normalize_output_format("zarr") == "zarr"
        assert _normalize_output_format("zip") == "zarr"
        assert _normalize_output_format("ZARR") == "zarr"

    def test_unsupported_format(self) -> None:
        """Test unsupported format raises error."""
        with pytest.raises(HTTPException):
            _normalize_output_format("netcdf")


class TestFuseEndpoint:
    """Test the /fuse endpoint."""

    def _create_mock_dataset(self, bbox: tuple[float, float, float, float]) -> xr.Dataset:
        """Create a mock elevation dataset for testing."""
        west, south, east, north = bbox
        # Simple 10x10 grid
        x = np.linspace(west, east, 10)
        y = np.linspace(south, north, 10)
        elev = np.random.rand(10, 10) * 100 - 50  # Random elevation -50 to 50m

        da_elev = xr.DataArray(
            elev,
            coords={"x": x, "y": y},
            dims=("y", "x"),
            name="elevation",
        )
        da_elev.rio.write_crs("EPSG:4326", inplace=True)

        src = np.ones((10, 10), dtype=np.uint32)
        da_src = xr.DataArray(src, coords={"x": x, "y": y}, dims=("y", "x"), name="source_elevation")

        ds = xr.Dataset({"elevation": da_elev, "source_elevation": da_src})
        ds.attrs["provenance_dict"] = {1: {"provider": "gebco", "product": "2025"}}
        ds.rio.write_crs("EPSG:4326", inplace=True)

        return ds

    @patch("topobathyserve.main.run")
    def test_fuse_geotiff_format(self, mock_run: MagicMock) -> None:
        """Test /fuse endpoint with GeoTIFF output."""
        bbox = (-74.0, 40.5, -73.5, 41.0)
        mock_run.return_value = self._create_mock_dataset(bbox)

        with TestClient(app) as client:
            response = client.get(
                "/fuse",
                params={
                    "west": -74.0,
                    "south": 40.5,
                    "east": -73.5,
                    "north": 41.0,
                    "format": "geotiff",
                },
            )

        assert response.status_code == 200
        assert response.headers["content-type"] == "image/tiff"
        assert len(response.content) > 0

    @patch("topobathyserve.main.run")
    def test_fuse_zarr_format(self, mock_run: MagicMock) -> None:
        """Test /fuse endpoint with Zarr ZIP output."""
        bbox = (-74.0, 40.5, -73.5, 41.0)
        ds = self._create_mock_dataset(bbox)
        mock_run.return_value = ds

        with TestClient(app) as client:
            response = client.get(
                "/fuse",
                params={
                    "west": -74.0,
                    "south": 40.5,
                    "east": -73.5,
                    "north": 41.0,
                    "format": "zarr",
                },
            )

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/zip"
        assert len(response.content) > 0
        assert "fused.zarr.zip" in response.headers.get("Content-Disposition", "")

        # Verify Zarr file content
        with TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.zarr.zip"
            zarr_path.write_bytes(response.content)

            import zarr.storage

            store = zarr.storage.ZipStore(str(zarr_path), mode="r")
            ds_loaded = xr.open_dataset(store, engine="zarr")  # type: ignore[arg-type]

            assert "elevation" in ds_loaded
            assert ds_loaded.rio.crs.to_string() == "EPSG:4326"
            # Check metadata attributes
            assert "crs" in ds_loaded.attrs
            assert "bbox" in ds_loaded.attrs
            assert "resolution_m" in ds_loaded.attrs
            assert "policy" in ds_loaded.attrs
            assert "created" in ds_loaded.attrs
            assert "providers_used" in ds_loaded.attrs

            store.close()

    @patch("topobathyserve.main.run")
    def test_fuse_zarr_metadata_attrs(self, mock_run: MagicMock) -> None:
        """Test that Zarr output includes proper metadata."""
        bbox = (-74.0, 40.5, -73.5, 41.0)
        ds = self._create_mock_dataset(bbox)
        mock_run.return_value = ds

        with TestClient(app) as client:
            response = client.get(
                "/fuse",
                params={
                    "bbox": "-74.0,40.5,-73.5,41.0",
                    "format": "zarr",
                    "resolution": 25.0,
                },
            )

        assert response.status_code == 200

        with TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.zarr.zip"
            zarr_path.write_bytes(response.content)

            import zarr.storage

            store = zarr.storage.ZipStore(str(zarr_path), mode="r")
            ds_loaded = xr.open_dataset(store, engine="zarr")  # type: ignore[arg-type]

            # Verify metadata
            assert ds_loaded.attrs["crs"] == "EPSG:4326"
            assert ds_loaded.attrs["bbox"] == [-74.0, 40.5, -73.5, 41.0]
            assert ds_loaded.attrs["resolution_m"] == 25.0
            assert ds_loaded.attrs["vertical_datum"] == "NAVD88"
            assert "wlis.yaml" in ds_loaded.attrs["policy"]  # Default policy
            assert isinstance(ds_loaded.attrs["created"], str)
            assert isinstance(ds_loaded.attrs["providers_used"], list)
            assert "gebco" in ds_loaded.attrs["providers_used"]

            store.close()

    @patch("topobathyserve.main.run")
    def test_fuse_bbox_string_format(self, mock_run: MagicMock) -> None:
        """Test /fuse endpoint with bbox string parameter."""
        bbox = (-74.0, 40.5, -73.5, 41.0)
        mock_run.return_value = self._create_mock_dataset(bbox)

        with TestClient(app) as client:
            response = client.get(
                "/fuse",
                params={"bbox": "-74.0,40.5,-73.5,41.0", "format": "geotiff"},
            )

        assert response.status_code == 200
        assert response.headers["content-type"] == "image/tiff"

    @patch("topobathyserve.main.run")
    def test_fuse_invalid_bbox_ordering(self, mock_run: MagicMock) -> None:
        """Test /fuse endpoint rejects invalid bbox ordering."""
        mock_run.return_value = self._create_mock_dataset((-74.0, 40.5, -73.5, 41.0))
        with TestClient(app) as client:
            response = client.get(
                "/fuse",
                params={
                    "west": -73.5,
                    "south": 40.5,
                    "east": -74.0,  # west > east (invalid)
                    "north": 41.0,
                    "format": "geotiff",
                },
            )

        assert response.status_code == 400

    @patch("topobathyserve.main.run")
    def test_fuse_missing_bbox(self, mock_run: MagicMock) -> None:
        """Test /fuse endpoint requires bbox."""
        with TestClient(app) as client:
            response = client.get("/fuse", params={"format": "geotiff"})

        assert response.status_code == 400

    @patch("topobathyserve.main.run")
    def test_fuse_unsupported_format(self, mock_run: MagicMock) -> None:
        """Test /fuse endpoint rejects unsupported format."""
        bbox = (-74.0, 40.5, -73.5, 41.0)
        mock_run.return_value = self._create_mock_dataset(bbox)

        with TestClient(app) as client:
            response = client.get(
                "/fuse",
                params={
                    "west": -74.0,
                    "south": 40.5,
                    "east": -73.5,
                    "north": 41.0,
                    "format": "netcdf",
                },
            )

        assert response.status_code == 400

    @patch("topobathyserve.main.run")
    def test_fuse_empty_elevation(self, mock_run: MagicMock) -> None:
        """Test /fuse endpoint handles empty elevation gracefully."""
        ds_empty = xr.Dataset(
            {
                "elevation": xr.DataArray(
                    np.array([]).reshape(0, 0),
                    dims=("y", "x"),
                )
            }
        )
        mock_run.return_value = ds_empty

        with TestClient(app) as client:
            response = client.get(
                "/fuse",
                params={
                    "west": -74.0,
                    "south": 40.5,
                    "east": -73.5,
                    "north": 41.0,
                    "format": "geotiff",
                },
            )

        assert response.status_code == 404


class TestQualityReport:
    """Test the /api/v1/quality/report endpoint."""

    @patch("topobathyserve.main.run")
    def test_vdatum_risk_counts_steps_at_source_seams(self, mock_run: MagicMock) -> None:
        """A 10 m step where two sources meet must register as VDatum risk, not 0."""
        x = np.linspace(-74.0, -73.99, 10)
        y = np.linspace(40.71, 40.70, 10)
        elev = np.zeros((10, 10), dtype=np.float32)
        elev[:, 5:] = -10.0
        src = np.full((10, 10), 5_000_001, dtype=np.uint32)
        src[:, 5:] = 6_000_001
        mock_run.return_value = xr.Dataset(
            {
                "elevation": xr.DataArray(elev, coords={"y": y, "x": x}, dims=("y", "x")),
                "source_elevation": xr.DataArray(src, coords={"y": y, "x": x}, dims=("y", "x")),
            }
        )

        with TestClient(app) as client:
            response = client.get(
                "/api/v1/quality/report",
                params={"north": 40.71, "south": 40.70, "west": -74.0, "east": -73.99, "resolution": 10.0},
            )

        assert response.status_code == 200
        stats = response.json()["stats"]
        assert stats["vdatum_risk_score"] > 0
        assert stats["steep_cliffs_count"] == 0
