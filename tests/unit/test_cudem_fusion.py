import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import xarray as xr
import yaml
from affine import Affine

from topobathykit.providers.base import Provider
from topobathykit.providers.registry import registry
from topobathykit.runtime import run


# Define Mock Providers (outside of class to avoid pickle issues if any)
class MockProvider(Provider):
    def __init__(self, val: float = 0.0):
        self.val = val

    def fetch_layer(
        self,
        bbox: tuple[float, float, float, float],
        resolution: float | None = None,
        crs: str = "EPSG:4326",
        **kwargs: Any,
    ) -> xr.Dataset:
        west, south, east, north = bbox

        # Simple 10x10 relative to bbox
        width = 10
        height = 10

        res_x = (east - west) / width
        res_y = (north - south) / height

        # Affine transform for creating coords
        transform = Affine.translation(west, north) * Affine.scale(res_x, -res_y)

        # Pixel centers
        xs = (np.arange(width) + 0.5) * res_x + west
        ys = (np.arange(height) + 0.5) * -res_y + north

        data = np.full((height, width), self.val, dtype=np.float32)

        # Add some NaNs to test fusion if value is specific (e.g. CUDEM=10.0)
        if self.val == 10.0:
            data[:, 5:] = np.nan

        da = xr.DataArray(data, coords={"y": ys, "x": xs}, dims=("y", "x"), name="elevation")
        if hasattr(da.rio, "write_crs"):
            da.rio.write_crs("EPSG:4326", inplace=True)
        if hasattr(da.rio, "write_transform"):
            da.rio.write_transform(transform, inplace=True)

        source_id = xr.DataArray(
            np.full((height, width), 1, dtype=np.uint8),
            coords={"y": ys, "x": xs},
            dims=("y", "x"),
            name="source_id",
        )

        ds = da.to_dataset()
        ds["source_id"] = source_id
        ds.attrs["provenance_dict"] = {1: "mock"}
        return ds

    def get_metadata(self) -> dict[str, Any]:
        return {"name": "MockProvider", "version": "0.1"}


class TestRuntimeFusion(unittest.TestCase):
    def setUp(self) -> None:
        # Register mock providers dynamically
        # Since registry is global, we should be careful about cleaning up,
        # but for unit tests it's usually fine as long as names don't clash.
        registry.register("mock_cudem_runtime", MockProvider)
        registry.register("mock_bluetopo_runtime", MockProvider)

        self.tmp_dir = TemporaryDirectory()
        self.policy_path = Path(self.tmp_dir.name) / "test_policy.yaml"

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_bluetopo_overrides_cudem(self) -> None:
        """
        Verify that BlueTopo overrides CUDEM in the fusion stack using runtime.
        """
        # Create Policy Data
        policy_data = {
            "name": "Test Fusion",
            "crs": "EPSG:4326",
            "variables": [
                {
                    "name": "elevation",
                    "steps": [
                        # Step 1: CUDEM (Value 10.0, partial NaN)
                        {
                            "provider": "mock_cudem_runtime",
                            "operator": "overwrite",
                            # We can't pass constructor args easily here
                            # through registry without extra logic in registry
                            # Registry instantiates with no args: cls()
                            # So MockProvider default is 0.0.
                            # We need separate classes or a way to config.
                        },
                        # Step 2: BlueTopo (Value 50.0)
                        {"provider": "mock_bluetopo_runtime", "operator": "overwrite"},
                    ],
                }
            ],
        }

        # We need distinct classes for registry to instantiate different values
        class MockCUDEM(MockProvider):
            def __init__(self) -> None:
                super().__init__(val=10.0)

        class MockBlueTopo(MockProvider):
            def __init__(self) -> None:
                super().__init__(val=50.0)

        registry.register("mock_cudem_runtime", MockCUDEM)
        registry.register("mock_bluetopo_runtime", MockBlueTopo)

        with open(self.policy_path, "w") as f:
            yaml.dump(policy_data, f)

        # Define bbox
        bbox = (-70.1, 40.0, -70.0, 40.1)

        # Run Fusion
        # resolution=0.01 degrees ~ 1km. Bbox is 0.1 deg ~ 11km.
        # use_cache=False to avoid disk IO issues
        # 0.01 degrees is roughly 1113 meters.
        # Passing resolution=1113.0 to get close to what MockProvider produces (10x10 over 0.1 deg)
        ds = run(str(self.policy_path), bbox, resolution=1113.0, use_cache=False)

        # Verify
        # Expected: 50.0 everywhere (MockBlueTopo)
        vals = ds["elevation"].values
        np.testing.assert_allclose(vals, 50.0)
