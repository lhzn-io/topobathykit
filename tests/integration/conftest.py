from pathlib import Path

import pytest

from topobathykit.config import get_cache_root


@pytest.fixture(scope="session")
def persistent_cache_dir() -> Path:
    """
    Returns a persistent cache directory for integration tests.
    This avoids re-downloading large datasets (Lidar/BlueTopo) across test runs,
    significantly reducing test suite runtime.
    """
    # Use a subdirectory in the standard user cache to separate test data from production data
    path = get_cache_root() / "tests"
    path.mkdir(parents=True, exist_ok=True)
    return path
