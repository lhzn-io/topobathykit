from typing import Any

import numpy as np
import pytest
import xarray as xr
import yaml
from affine import Affine

from topobathykit.providers.base import Provider
from topobathykit.providers.registry import registry
from topobathykit.runtime import run


class MockProvider(Provider):
    def __init__(self, value: float = 10.0):
        self.value = value

    def fetch_layer(
        self,
        bbox: tuple[float, float, float, float],
        resolution: float | None = None,
        crs: str = "EPSG:4326",
        **kwargs: Any,
    ) -> xr.DataArray:
        min_lon, min_lat, max_lon, max_lat = bbox

        if resolution is not None:
            res_x = resolution / 111320.0 if crs == "EPSG:4326" else resolution
            res_y = res_x

            x_idx_start = int(np.floor(min_lon / res_x))
            x_idx_end = int(np.ceil(max_lon / res_x))
            width = max(1, x_idx_end - x_idx_start)

            y_idx_start = int(np.floor(min_lat / res_y))
            y_idx_end = int(np.ceil(max_lat / res_y))
            height = max(1, y_idx_end - y_idx_start)

            snapped_min_x = x_idx_start * res_x
            snapped_max_y = y_idx_end * res_y
            transform = Affine.translation(snapped_min_x, snapped_max_y) * Affine.scale(res_x, -res_y)

            xs = (np.arange(x_idx_start, x_idx_start + width) + 0.5) * res_x
            ys = (np.arange(y_idx_end - 1, y_idx_end - 1 - height, -1) + 0.5) * res_y

            xs = np.round(xs, 8)
            ys = np.round(ys, 8)
        else:
            width, height = 10, 10
            res_x = (max_lon - min_lon) / width
            res_y = (max_lat - min_lat) / height
            transform = Affine.translation(min_lon, max_lat) * Affine.scale(res_x, -res_y)
            xs = np.linspace(min_lon + res_x / 2, max_lon - res_x / 2, width)
            ys = np.linspace(max_lat - res_y / 2, min_lat + res_y / 2, height)

        data = np.full((height, width), self.value, dtype=np.float32)

        coords = {
            "y": ys,
            "x": xs,
        }

        da = xr.DataArray(data, coords=coords, dims=("y", "x"))
        da.rio.write_crs(crs, inplace=True)
        da.rio.write_transform(transform, inplace=True)

        return da

    def get_metadata(self) -> dict[str, Any]:
        return {"name": "MockProvider"}


@pytest.fixture
def policy_with_filters(tmp_path: Any) -> str:
    """Creates a policy that applies min_elevation and max_elevation filters."""
    policy_content = {
        "name": "FilterPolicy",
        "crs": "EPSG:4326",
        "variables": [
            {
                "name": "elevation",
                "background": -999.0,
                "steps": [
                    {"provider": "mock_base", "operation": "overwrite"},
                    {
                        "provider": "mock_filtered",
                        "operation": "overwrite",
                        "filter": {"min_elevation": -15.0, "max_elevation": 5.0},
                    },
                ],
            }
        ],
    }
    policy_path = tmp_path / "filter_policy.yaml"
    with open(policy_path, "w") as f:
        yaml.dump(policy_content, f)
    return str(policy_path)


def test_elevation_constraints(policy_with_filters: str) -> None:
    pass


def test_filter_bounds(policy_with_filters: str) -> None:
    """Test that runtime respects min_elevation and max_elevation filters."""

    # Base Provider returns a flat 0.0 surface
    class BaseProvider(MockProvider):
        def fetch_layer(self, *args: Any, **kwargs: Any) -> xr.DataArray:
            da = super().fetch_layer(*args, **kwargs)
            da[:] = 0.0
            return da

    # Filtered Provider returns a gradient from -20 to 20 mapped exactly from -74.0 to -73.0 longitude
    class FilteredProvider(MockProvider):
        def fetch_layer(
            self,
            bbox: tuple[float, float, float, float],
            resolution: float | None = None,
            crs: str = "EPSG:4326",
            **kwargs: Any,
        ) -> xr.DataArray:
            da = super().fetch_layer(bbox=bbox, resolution=resolution, crs=crs, **kwargs)
            # Elev = (lon - (-74)) / 1.0 * 40 - 20
            da_vals = (da.x - (-74.0)) * 40.0 - 20.0
            da.values[:] = np.broadcast_to(da_vals.values, da.shape).astype(np.float32)
            return da

    registry.register("mock_base", BaseProvider)
    registry.register("mock_filtered", FilteredProvider)

    bbox = (-74.0, 40.0, -73.0, 41.0)

    # Run the fusion engine without cache to force fetch
    ds = run(policy_with_filters, bbox, resolution=1000.0, use_cache=False)

    # Analyze the result
    elev = ds["elevation"].values

    # The filter bounds are min=-15.0, max=5.0
    # Where filtered provider exceeds bounds, it should fall back to base provider (0.0)

    # Ensure no values exist outside the constraint boundaries
    # Any value that was filtered out will be 0.0 (Base provider)
    assert not np.any(elev < -15.0)
    assert not np.any(elev > 5.0)

    # Ensure the filtered data successfully blended inside the bounds
    # (i.e. it's not simply an empty map of all 0.0)
    assert np.any(elev < -0.1)  # Some negative gradient survived (> -15.0)
    assert np.any(elev > 0.1)  # Some positive gradient survived (< 5.0)
