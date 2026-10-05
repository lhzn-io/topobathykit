from pathlib import Path

import numpy as np
import pytest

from topobathykit.runtime import run


@pytest.mark.integration
def test_wlis_fusion_policy() -> None:
    """
    Integration test for the WLIS fusion policy.
    Verifies that the runtime correctly parses the policy, queries providers,
    handles missing data (BAG/Topobathy), and produces a valid 7-layer fused output.
    """
    policy_path = "policies/examples/wlis.yaml"
    if not Path(policy_path).exists():
        pytest.skip(f"Policy file {policy_path} not found.")

    # Small test patch to avoid OOM and long runtimes
    # (-72.5, 41.0) to (-72.45, 41.05) covers a coastal area in WLIS
    bbox = (-72.5, 41.0, -72.45, 41.05)
    resolution = 100.0  # Coarse resolution for speed

    print(f"Running WLIS Fusion on patch: {bbox} @ {resolution}m")

    ds = run(policy_path, bbox, resolution=resolution)

    # 1. Check Output Structure
    assert "elevation" in ds
    assert "source_elevation" in ds
    assert ds.rio.crs.to_string() == "EPSG:3857"

    # 2. Check Data Validity
    elev = ds.elevation.values
    source = ds.source_elevation.values

    # basic ranges
    assert np.nanmin(elev) > -100  # Not deeper than 100m
    assert np.nanmax(elev) < 200  # Not higher than 200m

    # 3. Check Source Provenance
    # We expect a mix of sources.
    # BAG (1) might be missing.
    # Lidar (5), BlueTopo (2), CUDEM (3) are likely present.
    unique_sources = np.unique(source)
    print(f"Unique Sources Found: {unique_sources}")

    # Ensure at least some data comes from non-default (0) sources
    assert len(unique_sources) > 1

    # Check specifically for BlueTopo or CUDEM which should cover this marine area.
    # We now check using the detailed provenance dictionary since IDs are dynamic assets.
    prov_dict = ds.attrs.get("provenance_dict", {})
    providers_used = [info.get("provider") for info in prov_dict.values() if isinstance(info, dict)]
    assert "noaa_bluetopo" in providers_used or "ncei_cudem" in providers_used

    # 4. Check Attributes
    assert "policy_hash" in ds.attrs
    assert "policy_legend" in ds.attrs
