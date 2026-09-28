from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import xarray as xr
from fastapi.testclient import TestClient
from topobathyserve.main import app, get_policy_path

client = TestClient(app)

_DEFAULT_POLICY = """
name: default_policy
crs: EPSG:4326
variables:
  - name: elevation
    steps:
      - provider: gebco_2025
"""


@pytest.fixture(autouse=True)
def _policy_path(tmp_path: Path) -> Iterator[None]:
    """Point get_policy_path at a real policy file for these tests only.

    The /fuse handlers load the policy to log its hash before calling run(), so the
    file must exist. The override is removed afterwards so it cannot leak into other
    test modules that share the app.
    """
    policy = tmp_path / "default_policy.yaml"
    policy.write_text(_DEFAULT_POLICY)
    app.dependency_overrides[get_policy_path] = lambda: policy
    yield
    app.dependency_overrides.pop(get_policy_path, None)


@pytest.fixture
def mock_run_dataset() -> xr.Dataset:
    """Create a minimal xarray dataset for mock returns."""
    bbox = (-74.0, 40.0, -73.0, 41.0)
    width, height = 10, 10

    # Create coords using simple linspace
    import numpy as np

    x = np.linspace(bbox[0], bbox[2], width)
    y = np.linspace(bbox[3], bbox[1], height)  # North up (descending Y) usually

    data = np.zeros((height, width))

    ds = xr.Dataset(data_vars={"elevation": (("y", "x"), data)}, coords={"y": y, "x": x})
    if not hasattr(ds, "rio"):
        pass
    ds.rio.write_crs("EPSG:4326", inplace=True)
    return ds


@patch("topobathyserve.main.run")
def test_post_fuse_with_policy_yaml(mock_run: MagicMock, mock_run_dataset: xr.Dataset) -> None:
    """Test POST /fuse with policy_yaml string."""
    mock_run.return_value = mock_run_dataset

    override_policy = """
    name: override_policy
    crs: EPSG:4326
    variables:
      - name: elevation
        steps:
          - provider: noaa_bluetopo
            filter:
              min_elevation: -50.0
    """

    payload = {
        "bbox": [-74.0, 40.0, -73.0, 41.0],
        "resolution": 30.0,
        "format": "geotiff",
        "policy_yaml": override_policy,
    }

    response = client.post("/fuse", json=payload)

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/tiff"

    # Check that run was called with the override string as 'policy_input'
    mock_run.assert_called_once()
    call_args = mock_run.call_args
    # signature: run(policy_input, bbox, resolution=..., use_cache=...)
    assert call_args.kwargs["policy_input"] == override_policy
    assert call_args.kwargs["use_cache"] is True


@patch("topobathyserve.main.run")
def test_post_fuse_with_policy_name(mock_run: MagicMock, mock_run_dataset: xr.Dataset) -> None:
    """Test POST /fuse with policy_name (currently minimal impl)."""
    mock_run.return_value = mock_run_dataset

    payload = {
        "bbox": [-74.0, 40.0, -73.0, 41.0],
        "resolution": 30.0,
        "format": "geotiff",
        "policy_name": "special_policy",
    }

    # Default behavior for now is to use default path but maybe logic differs
    # The handler implementation currently prioritizes default path if override is missing/name handling
    # is basic.
    # Actually current implementation in main.py does:
    # policy_input: Path | str = policy_path (from dependency)
    # if request.policy_override: ...
    # elif request.policy_name: ... pass

    # So it falls back to default policy path if only name is provided (as per 'pass').
    # We just want to ensure it doesn't crash.

    response = client.post("/fuse", json=payload)
    assert response.status_code == 200
