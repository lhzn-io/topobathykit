from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from topobathykit.policy.schema import CompositionStep, FusionPolicy, OperatorType, VariableStrategy
from topobathykit.providers.base import Provider
from topobathykit.runtime import run


# Mock Provider that returns 3D data
class Mock3DProvider(Provider):
    def get_metadata(self) -> dict[str, Any]:
        return {}

    def fetch_layer(
        self,
        bbox: tuple[float, float, float, float],
        resolution: float | None = None,
        crs: str = "EPSG:4326",
        **kwargs: Any,
    ) -> xr.DataArray:
        # Return (1, 10, 10) array
        data = np.full((1, 10, 10), 5.0)
        da = xr.DataArray(
            data,
            coords={"band": [1], "y": np.arange(10), "x": np.arange(10)},
            dims=("band", "y", "x"),
        )
        da.rio.write_crs("EPSG:4326", inplace=True)
        return da


class MockMultiBandProvider(Provider):
    def get_metadata(self) -> dict[str, Any]:
        return {}

    def fetch_layer(
        self,
        bbox: tuple[float, float, float, float],
        resolution: float | None = None,
        crs: str = "EPSG:4326",
        **kwargs: Any,
    ) -> xr.DataArray:
        # Return (3, 10, 10) array
        data = np.full((3, 10, 10), 10.0)
        # Make bands distinct
        data[0, :, :] = 1.0
        data[1, :, :] = 2.0
        data[2, :, :] = 3.0

        da = xr.DataArray(
            data,
            coords={"band": [1, 2, 3], "y": np.arange(10), "x": np.arange(10)},
            dims=("band", "y", "x"),
        )
        da.rio.write_crs("EPSG:4326", inplace=True)
        return da


class MockChannelLastProvider(Provider):
    def get_metadata(self) -> dict[str, Any]:
        return {}

    def fetch_layer(
        self,
        bbox: tuple[float, float, float, float],
        resolution: float | None = None,
        crs: str = "EPSG:4326",
        **kwargs: Any,
    ) -> xr.DataArray:
        # Return (10, 10, 1) array
        data = np.full((10, 10, 1), 7.0)
        da = xr.DataArray(
            data,
            coords={"y": np.arange(10), "x": np.arange(10), "band": [1]},
            dims=("y", "x", "band"),
        )
        da.rio.write_crs("EPSG:4326", inplace=True)
        return da


@pytest.fixture
def mock_policy() -> FusionPolicy:
    return FusionPolicy(
        name="test_policy",
        crs="EPSG:4326",
        variables=[
            VariableStrategy(
                name="elevation",
                background=np.nan,
                steps=[
                    CompositionStep(provider="mock_3d", operation=OperatorType.overwrite),
                    CompositionStep(provider="mock_multiband", operation=OperatorType.overwrite),
                    CompositionStep(provider="mock_channel_last", operation=OperatorType.overwrite),
                ],
            )
        ],
    )


def test_runtime_enforces_2d(mock_policy: FusionPolicy) -> None:
    # REGISTER MOCKS
    with patch("topobathykit.runtime.registry.get_provider_class") as mock_get_cls:
        # We need to dispatch based on provider name
        def get_cls(name: str) -> type[Provider]:
            if name == "mock_3d":
                return Mock3DProvider
            if name == "mock_multiband":
                return MockMultiBandProvider
            if name == "mock_channel_last":
                return MockChannelLastProvider
            return Mock3DProvider

        mock_get_cls.side_effect = get_cls

        # We must patch _resolve_policy because run() calls it now
        with patch("topobathykit.runtime._resolve_policy", return_value=mock_policy):
            # 1. Test Single Band 3D (1, H, W) -> Should reduce to (H, W)
            # We construct a policy with just that step
            mock_policy.variables[0].steps = [
                CompositionStep(
                    provider="mock_3d",
                    operation=OperatorType.overwrite,
                    transitions=[],
                    filter=None,
                )
            ]

            # Use small bbox to avoid processing hundreds of tiles
            ds = run("dummy_path", bbox=(0, 0, 0.04, 0.04), resolution=10000, use_cache=False)
            elevation = ds["elevation"]

            assert elevation.ndim == 2, f"Expected 2D array, got {elevation.ndim}"
            valid_vals = elevation.values[~np.isnan(elevation.values)]
            assert len(valid_vals) > 0, "Expected valid values but array is all NaNs"
            assert valid_vals[0] == 5.0

            # 2. Test Multi Band (3, H, W) -> Should pick Band 1 (Value 1.0)
            mock_policy.variables[0].steps = [
                CompositionStep(provider="mock_multiband", operation=OperatorType.overwrite)
            ]

            ds = run("dummy_path", bbox=(0, 0, 0.04, 0.04), resolution=10000, use_cache=False)
            elevation = ds["elevation"]

            assert elevation.ndim == 2
            # Should be 1.0 (first band) not 2.0 or 3.0
            # Note: We align to canvas, so check valid pixels
            valid_vals = elevation.values[~np.isnan(elevation.values)]
            assert len(valid_vals) > 0, "Expected valid values but array is all NaNs"
            assert valid_vals[0] == 1.0

            # 3. Test Channel Last (H, W, 1) -> Should reduce to (H, W)
            mock_policy.variables[0].steps = [
                CompositionStep(provider="mock_channel_last", operation=OperatorType.overwrite)
            ]

            ds = run("dummy_path", bbox=(0, 0, 0.04, 0.04), resolution=10000, use_cache=False)
            elevation = ds["elevation"]

            assert elevation.ndim == 2
            valid_vals = elevation.values[~np.isnan(elevation.values)]
            assert len(valid_vals) > 0, "Expected valid values but array is all NaNs"
            assert valid_vals[0] == 7.0


if __name__ == "__main__":
    # Allow running directly
    pytest.main([__file__])
