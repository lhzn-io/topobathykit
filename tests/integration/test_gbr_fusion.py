import logging
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from topobathykit.runtime import run

# Create a custom logger
logger = logging.getLogger(__name__)


@pytest.mark.integration
def test_gbr_overlay_policy_execution(persistent_cache_dir: Path) -> None:
    """
    Integration test for the 'Hello-Coast' GBR fusion policy.

    This test executes the actual `policies/examples/gbr_overlay.yaml` policy.
    It requires network access (or valid cache) for GEBCO/GBR100 data.

    We use a small bounding box to minimize data fetching time while ensuring
    coverage from both sources.
    """

    # Locate the policy file relative to project root
    # This assumes the test is in tests/integration/
    project_root = Path(__file__).parents[2]
    policy_path = project_root / "policies" / "examples" / "gbr_overlay.yaml"

    assert policy_path.exists(), f"Policy file not found at {policy_path}"

    # Define a small BBox within the GBR region (approx 5km x 5km)
    # 145.5, -17.0 is the bottom-left of the user's manual run, so usage should hit cache.
    bbox = (145.5, -17.0, 145.55, -16.95)

    logger.info(f"Running GBR Fusion Policy from {policy_path} with bbox {bbox}")

    # Run the pipeline
    # Note: We rely on the default resolution logic (or we can force it)
    # The policy might not specify resolution, so run() defaults it or takes an arg.
    # We'll use 30m to match GBR100 roughly.
    ds = run(str(policy_path), bbox, resolution=30.0)

    # Verify Output Structure
    assert isinstance(ds, xr.Dataset)
    assert "elevation" in ds
    assert "source_elevation" in ds
    assert "crs" in ds.attrs or ds.rio.crs is not None

    # Verify CRS
    # Policy specifies EPSG:4326
    assert ds.rio.crs.to_string() == "EPSG:4326"

    # Verify Content
    elev = ds["elevation"]
    source = ds["source_elevation"]

    # Check for NaNs (should be fully filled by GEBCO at least)
    assert not np.isnan(elev.values).all(), "Elevation is entirely NaN"

    # Check Dimensions
    # 0.05 deg / ~30m (approx 0.00027 deg) -> approx 185 pixels
    # Just check it's not empty
    assert elev.size > 0

    # Verify Source IDs
    # Expecting GEBCO (usually ID 2 per alphabetic or step order) and GBR100 (ID 1)
    # We should see at least one source.
    unique_sources = np.unique(source.values)
    logger.info(f"Unique Sources found in integration test: {unique_sources}")

    assert len(unique_sources) >= 1, "No data sources found in output"

    # Optional: Check if we have both sources (might depend on exact bbox overlap)
    # If the bbox covers land and sea, we might see different sources.
    # GEBCO covers everything. GBR100 covers the reef.

    print("GBR Fusion Integration Test Passed")
