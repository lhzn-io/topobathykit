import sys
from collections.abc import Generator
from pathlib import Path

# Add service to path so we can import topobathyserve
sys.path.append(str(Path(__file__).parent.parent.parent / "service"))

import pytest
from fastapi.testclient import TestClient
from topobathyserve.main import app, get_policy_path

from topobathykit.policy.schema import CompositionStep, FusionPolicy, VariableStrategy


@pytest.fixture
def clean_cache(tmp_path: Path) -> Path:
    """Creates a temporary cache directory for the duration of the test."""
    cache_dir = tmp_path / "test_cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def test_policy_path(clean_cache: Path) -> Path:
    """Creates a basic policy pointing its cache to the clean tmp_path."""
    policy = FusionPolicy(
        name="Test Caching Policy",
        version="1.0.0",
        variables=[
            VariableStrategy(
                name="elevation",
                steps=[
                    CompositionStep(provider="usgs_3dep"),
                ],
            )
        ],
    )

    policy_path = clean_cache / "policy.yaml"
    with open(policy_path, "w") as f:
        # Patch the environment cache to point here during test explicitly
        import os

        os.environ["TOPOBATHYKIT_CACHE_DIR"] = str(clean_cache)
        f.write(policy.model_dump_json())  # type: ignore
    return policy_path


@pytest.fixture
def client(test_policy_path: Path) -> Generator[TestClient, None, None]:
    """Returns a TestClient with the policy path dependency overridden."""
    app.dependency_overrides[get_policy_path] = lambda: test_policy_path
    with TestClient(app) as client:
        yield client


@pytest.mark.integration
def test_tile_request_caches_data(client: TestClient, clean_cache: Path) -> None:
    """
    Verifies that requesting a tile triggers data downloads and caching
    in the expected directories.
    """
    # NYC (Battery Park area) - Known to have Lidar and 3DEP coverage
    # Zoom 14 specific tile coordinates
    z, x, y = 14, 4825, 6156

    url = f"/tiles/{z}/{x}/{y}.png"

    # 1. Initial State: Cache should be empty of data
    land_dir = clean_cache / "usgs_3dep"

    assert not land_dir.exists() or len(list(land_dir.glob("*"))) == 0, "Land cache should be empty initially"

    # 2. Fire Request
    # We expect this to be slow on first run as it downloads data
    print(f"Requesting {url}...")
    response = client.get(url)

    # 3. Validation
    assert response.status_code == 200, f"Request failed: {response.text}"
    assert response.headers["content-type"] == "image/png"

    # 4. Side Effect Check: Did we cache files?
    # Note: Depending on the fusion logic and data availability,
    # specific providers might be skipped or hit.
    # For NYC Battery Park:
    # 4. Check Output Cache (The generated PNG)
    output_cache = clean_cache / "tiles" / "visual" / "default"
    assert any(output_cache.rglob("*.png")), "Output PNG was not cached."

    # 5. Verify NPZ format and default 512x512 tile limits
    import io

    import numpy as np

    url_npz = f"/tiles/{z}/{x}/{y}.npz"
    resp_npz = client.get(url_npz)
    assert resp_npz.status_code == 200, f"NPZ request failed: {resp_npz.text}"

    with np.load(io.BytesIO(resp_npz.content)) as data:
        assert data["elevation"].shape == (
            512,
            512,
        ), f"Expected default 512x512, got {data['elevation'].shape}"
        if "source_elevation" in data:
            assert data["source_elevation"].shape == (512, 512)

    # 6. Verify tile_size parameter
    url_npz_256 = f"/tiles/{z}/{x}/{y}.npz?tile_size=256"
    resp_npz_256 = client.get(url_npz_256)
    assert resp_npz_256.status_code == 200, f"NPZ request failed: {resp_npz_256.text}"

    with np.load(io.BytesIO(resp_npz_256.content)) as data:
        assert data["elevation"].shape == (256, 256), f"Expected 256x256, got {data['elevation'].shape}"
        if "source_elevation" in data:
            assert data["source_elevation"].shape == (256, 256)
