from typing import Any, cast

import numpy as np
import pytest
import xarray as xr
import yaml
from affine import Affine
from pyproj import Transformer

from topobathykit.providers.base import Provider
from topobathykit.providers.registry import registry
from topobathykit.runtime import run


# Mock Provider for Testing
class MockParamsProvider(Provider):
    """
    Mock Provider that returns a constant value.
    """

    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def get_metadata(self) -> dict[str, Any]:
        return {"name": "MockParamsProvider"}

    def fetch_layer(
        self,
        bbox: tuple[float, float, float, float],
        resolution: float | None = None,
        crs: str = "EPSG:4326",
        **kwargs: Any,
    ) -> xr.DataArray:
        min_lon, min_lat, max_lon, max_lat = bbox

        # Dynamic high-resolution grid (approx 10m in degrees)
        res_deg = 0.0001
        width = max(int((max_lon - min_lon) / res_deg), 10)
        height = max(int((max_lat - min_lat) / res_deg), 10)
        res_x = (max_lon - min_lon) / width
        res_y = (max_lat - min_lat) / height

        transform = Affine.translation(min_lon, max_lat) * Affine.scale(res_x, -res_y)

        data = np.full((height, width), self.value, dtype=np.float32)

        coords = {
            "y": np.linspace(max_lat - res_y / 2, min_lat + res_y / 2, height),
            "x": np.linspace(min_lon + res_x / 2, max_lon - res_x / 2, width),
        }

        da = xr.DataArray(data, coords=coords, dims=("y", "x"))
        da.rio.write_crs("EPSG:4326", inplace=True)
        da.rio.write_nodata(np.nan, inplace=True)
        da.rio.write_transform(transform, inplace=True)

        return da


# Define Specific Providers
class BaseProvider(MockParamsProvider):
    def __init__(self) -> None:
        super().__init__(value=0.0)


class LidarProvider(MockParamsProvider):
    def __init__(self) -> None:
        super().__init__(value=10.0)

    def fetch_layer(self, *args: Any, **kwargs: Any) -> xr.DataArray:
        da = super().fetch_layer(*args, **kwargs)
        # Occupy ORIGINAL LEFT half of the (-74.01 to -74.00) test box
        da = cast(xr.DataArray, da.where(da.x < -74.005, np.nan))
        print("Lidar raw vals:", np.unique(da.values))
        return da


class StructureProvider(MockParamsProvider):
    def __init__(self) -> None:
        super().__init__(value=20.0)

    def fetch_layer(self, *args: Any, **kwargs: Any) -> xr.DataArray:
        da = super().fetch_layer(*args, **kwargs)
        # Occupy ORIGINAL CENTER (-74.0075 to -74.0025)
        mask = (da.x >= -74.0075) & (da.x <= -74.0025)
        da = cast(xr.DataArray, da.where(mask, np.nan))
        return da


# Register them
# Register mock providers
# Registry is a singleton, so these persist for the session unless cleared.


@pytest.fixture
def pairwise_policy_file(tmp_path: Any) -> str:
    # Setup Registry
    registry.register("base", BaseProvider)
    registry.register("lidar", LidarProvider)
    registry.register("structure", StructureProvider)

    # 3 Layers:
    # 1. Base (Value 0).
    # 2. Lidar (Value 10) on Left.
    # 3. Structure (Value 20) in Center.
    #    - Overlaps Lidar (Left-Center): Transition Rule -> Overwrite
    #    - Overlaps Base (Right-Center): Default Rule -> Feather

    policy_content = {
        "name": "PairwiseTest",
        "crs": "EPSG:32618",  # UTM 18N (Projected)
        "defaults": {
            "resolution": 10.0  # 10 meters
        },
        "variables": [
            {
                "name": "elevation",
                "steps": [
                    {"provider": "base", "operation": "overwrite"},
                    {"provider": "lidar", "operation": "overwrite"},
                    {
                        "provider": "structure",
                        "operation": "metric_feather",
                        "blend_distance": 50.0,  # 5 pixels if res=10m
                        "transitions": [{"target_provider": "lidar", "operator": "overwrite"}],
                    },
                ],
            }
        ],
    }

    p_path = tmp_path / "pairwise.yaml"
    with open(p_path, "w") as f:
        yaml.dump(policy_content, f)

    return str(p_path)


def test_pairwise_fusion_logic(pairwise_policy_file: str) -> None:
    # BBox in 4326.
    # Must pick a bbox that is valid for UTM 18N (e.g., NYC area).
    # NYC is approx 74W, 40.7N.
    bbox = (-74.01, 40.70, -74.00, 40.71)

    ds = run(pairwise_policy_file, bbox, resolution=10.0, use_cache=False)

    assert isinstance(ds, xr.Dataset)
    assert ds.attrs["crs"] == "EPSG:32618"

    # Verify Dimensions/Resolution
    # 0.01 deg width / 10m resolution -> approx 100 pixels

    # Check IDs
    # Base=1, Lidar=2, Structure=3
    # (assuming order in generate_provider_legend matches policy order or alphabet?)
    # generate_provider_legend usually iterates steps.
    # Let's inspect source_elevation to confirm IDs.

    elev = ds["elevation"]

    # Expected Composition:
    # Left: Lidar (10)
    # Center-Left: Structure (20) covering Lidar [Overwrite]
    # Center-Right: Structure (20->0) fading onto Base [Feather]
    # Right: Base (0)

    # 1. Check Output CRS
    assert ds.rio.crs.to_string() == "EPSG:32618"

    # 2. Check Overwrite Transition (Structure on Lidar)
    # Finding a point where structure overlaps lidar.
    # Lidar is strictly valid on left half. Structure is valid in center.
    # So overlap is the intersection.
    # Because logic is Overwrite, values should be exactly 20.0, no blending from 10.0.

    # Find mask where source is structure
    # We need to know the IDs.
    # Legend is stored in attrs.
    # Use pyproj to convert 4326 test points to UTM 18N for querying
    trans = Transformer.from_crs("EPSG:4326", "EPSG:32618", always_xy=True)

    # Test Points (Longitudes):
    # A: Pure Lidar (-74.009) - Left side
    # B: Structure on Lidar (-74.006) - Center-Left Overlap
    # C: Structure on Base (-74.0027) - Center-Right Overlap (Feather Zone)

    lons = [-74.009, -74.006, -74.0027]
    lats = [40.705, 40.705, 40.705]

    xs, ys = trans.transform(lons, lats)

    # query with expected tolerance
    # Point A: Pure Lidar
    val_a = elev.sel(x=xs[0], y=ys[0], method="nearest").item()
    val_b = elev.sel(x=xs[1], y=ys[1], method="nearest").item()
    val_c = elev.sel(x=xs[2], y=ys[2], method="nearest").item()
    assert val_a == 10.0, f"Expected Pure Lidar (10.0), got {val_a}"

    # Point B: Overwrite Zone (Structure on Lidar)
    val_b = elev.sel(x=xs[1], y=ys[1], method="nearest").item()
    assert val_b == 20.0, f"Expected Overwrite (20.0), got {val_b}"

    # Point C: Feather Zone (Structure on Base)
    # Should be blended between Base(0) and Structure(20)
    val_c = elev.sel(x=xs[2], y=ys[2], method="nearest").item()
    assert 0.0 < val_c < 20.0, f"Expected Feathering (0 < x < 20), got {val_c}"
    assert val_c != 20.0
    assert val_c != 0.0

    print("Integration Test Passed: Pairwise logic confirmed.")
