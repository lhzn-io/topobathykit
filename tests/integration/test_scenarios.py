import logging
import math
from pathlib import Path

import numpy as np
import pytest

from topobathykit.runtime import run

# Create a custom logger for this test module to ensure we see our own debugs
logger = logging.getLogger(__name__)

POLICY_PATH = Path(__file__).parent.parent.parent / "policies" / "examples" / "wlis.yaml"
logger.setLevel(logging.DEBUG)


@pytest.fixture(autouse=True, scope="module")
def silence_chatty_libraries() -> None:
    """
    Configures logging to silence noisy third-party libraries while
    keeping 'topobathykit' and local tests verbose.
    """
    # List of libraries to silence (set to WARNING or higher)
    chatty_loggers = [
        "rasterio",
        "botocore",
        "s3fs",
        "fsspec",
        "urllib3",
        "fiona",
        "shapely",
        "matplotlib",
        "PIL",
        "asyncio",
        "azure",
        "geopandas",
        "pyproj",
    ]

    for log_name in chatty_loggers:
        logging.getLogger(log_name).setLevel(logging.WARNING)

    # Ensure topobathykit is loud and clear
    logging.getLogger("topobathykit").setLevel(logging.DEBUG)


def xyz_to_bounds(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    """Convert XYZ tile coordinates to (south, north, west, east) bounding box."""
    n = 2.0**z
    west = x / n * 360.0 - 180.0
    east = (x + 1) / n * 360.0 - 180.0

    lat_rad_north = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    north = math.degrees(lat_rad_north)

    lat_rad_south = math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n)))
    south = math.degrees(lat_rad_south)
    return west, south, east, north


@pytest.mark.integration
@pytest.mark.network
@pytest.mark.parametrize(
    "z, x, y, expected_source_substr",
    [
        (13, 2418, 3075, "BAG"),  # Scenario from verify_tile.py
        (
            15,
            9672,
            12300,
            ["BAG", "LIDAR", "TOPO", "USGS"],
        ),  # Scenario from verify_user_tile.py (matches any of these)
    ],
)
def test_real_world_tile_scenarios(
    z: int, x: int, y: int, expected_source_substr: str | list[str], persistent_cache_dir: Path
) -> None:
    """
    Verify specific real-world tiles known to contain specific data types.
    This replaces the ad-hoc scripts in tests/scripts/.
    """
    import os

    os.environ["TOPOBATHYKIT_CACHE_DIR"] = str(persistent_cache_dir)

    west, south, east, north = xyz_to_bounds(z, x, y)
    logger.info(f"Testing Tile Z={z} X={x} Y={y} -> BBox: {west:.4f}, {south:.4f}, {east:.4f}, {north:.4f}")

    # Calculate approximate resolution to match tile zoom level roughly
    resolution = 10.0

    try:
        ds = run(str(POLICY_PATH), (west, south, east, north), resolution=resolution, use_cache=True)
    except Exception as e:
        pytest.fail(f"Runtime execution failed: {e}")

    da = ds["elevation"]
    assert da is not None, "Elevation data missing"

    # Check provenance
    if "provenance_dict" in ds.attrs:
        provenance = ds.attrs["provenance_dict"]
        sources_text = " ".join([str(v).upper() for v in provenance.values()])
    else:
        sources_text = "UNKNOWN"

    logger.info(f"Sources found: {sources_text}")

    # Verification Logic
    if isinstance(expected_source_substr, list):
        # Match ANY of the expected substrings
        found = any(sub in sources_text for sub in expected_source_substr)
        assert found, f"Expected one of {expected_source_substr} in sources, got: {sources_text}"
    else:
        assert expected_source_substr.upper() in sources_text, (
            f"Expected {expected_source_substr} in sources, got: {sources_text}"
        )

    # Data content validation
    valid_data = da.values[~np.isnan(da.values)]
    assert len(valid_data) > 0, "Grid contains only NaNs"

    min_val = np.min(valid_data)
    max_val = np.max(valid_data)
    logger.info(f"Elevation Range: {min_val:.2f} to {max_val:.2f}")

    # Heuristic checks based on known tile locations
    if z == 15 and x == 9672:
        # Known detailed tile with land and water
        assert max_val > 0, "Should have land (elevation > 0)"
        assert min_val < -2.0, "Should have deep water (elevation < -2.0)"
