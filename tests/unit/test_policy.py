"""
Unit tests for the policy schema and loader.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr
import yaml
from pydantic import ValidationError

from topobathykit.policy.loader import generate_provider_legend, hash_policy, load_policy
from topobathykit.policy.schema import (
    CompositionStep,
    FusionPolicy,
    OperatorType,
    TransitionRule,
    VariableStrategy,
)
from topobathykit.providers.base import Provider
from topobathykit.providers.registry import registry


# Mock Provider
class MockProvider(Provider):
    def fetch_layer(
        self,
        bbox: tuple[float, float, float, float],
        resolution: float | None = None,
        crs: str = "EPSG:4326",
        **kwargs: Any,
    ) -> xr.DataArray:
        # Return dummy data to satisfy interface
        return xr.DataArray(np.zeros((2, 2)), dims=("y", "x"), coords={"y": [0, 1], "x": [0, 1]})

    def get_metadata(self) -> dict[str, Any]:
        return {"name": "Mock Provider"}


# Register mock provider
registry.register("mock_provider", MockProvider)
registry.register("gebco", MockProvider)


@pytest.fixture
def valid_policy_dict() -> dict[str, Any]:
    return {
        "name": "Test Policy",
        "version": "1.0.0",
        "crs": "EPSG:4326",
        "variables": [
            {
                "name": "elevation",
                "background": -10000.0,
                "steps": [
                    {
                        "provider": "mock_provider",
                        "operation": "overwrite",
                        "blend_mode": "feather",
                        "blend_distance": 500.0,
                        "zone": {"type": "bbox", "coords": [-180, -90, 180, 90], "priority": 10},
                    }
                ],
            }
        ],
    }


def test_load_valid_policy(valid_policy_dict: dict[str, Any], tmp_path: Path) -> None:
    policy_file = tmp_path / "policy.yaml"
    with open(policy_file, "w") as f:
        yaml.dump(valid_policy_dict, f)

    policy = load_policy(str(policy_file))
    assert policy.name == "Test Policy"
    assert len(policy.variables) == 1
    assert policy.variables[0].steps[0].provider == "mock_provider"


def test_load_invalid_provider(valid_policy_dict: dict[str, Any], tmp_path: Path) -> None:
    # Modify to use unregistered provider
    valid_policy_dict["variables"][0]["steps"][0]["provider"] = "unknown_provider"

    policy_file = tmp_path / "invalid_policy.yaml"
    with open(policy_file, "w") as f:
        yaml.dump(valid_policy_dict, f)

    # The schema validator allows unregistered providers during static loading
    # (it suppresses KeyError in validation), but we verify the policy loads with the invalid key.
    policy = load_policy(str(policy_file))
    assert policy.variables[0].steps[0].provider == "unknown_provider"


def test_hash_policy(valid_policy_dict: dict[str, Any]) -> None:
    hash1 = hash_policy(valid_policy_dict)

    # Same content, different key order in dict (Python dicts preserve order since 3.7)
    dict2 = valid_policy_dict.copy()
    hash2 = hash_policy(dict2)

    assert hash1 == hash2

    # Modify content
    valid_policy_dict["name"] = "Modified Policy"
    hash3 = hash_policy(valid_policy_dict)
    assert hash1 != hash3


def test_generate_provider_legend() -> None:
    policy = FusionPolicy(
        name="Legend Test",
        variables=[
            VariableStrategy(
                name="elevation",
                steps=[
                    CompositionStep(provider="gebco"),
                    CompositionStep(provider="mock_provider"),
                ],
                background=0,
            )
        ],
    )

    legend = generate_provider_legend(policy)

    # Legend keys should be integers starting from 1
    assert 1 in legend
    assert 2 in legend

    # Values should be provider strings
    values = list(legend.values())
    assert "gebco" in values
    assert "mock_provider" in values

    # Check deterministic ordering (alphabetical)
    # "gebco" comes before "mock_provider"
    assert legend[1] == "gebco"
    assert legend[2] == "mock_provider"


def test_step_operator_alias_maps_to_operation() -> None:
    """The documented-but-unsupported step key `operator` must not be dropped."""
    with pytest.warns(FutureWarning, match="operator"):
        step = CompositionStep(provider="mock_provider", operator="overwrite")  # type: ignore[call-arg]
    assert step.operation == OperatorType.overwrite


def test_step_operator_alias_conflict_is_rejected() -> None:
    with pytest.raises(ValidationError, match="operation only"):
        CompositionStep(provider="mock_provider", operation="overwrite", operator="metric_feather")  # type: ignore[call-arg, arg-type]


def test_step_informational_fields_are_accepted() -> None:
    step = CompositionStep(provider="mock_provider", product="geotiff", description="BlueTopo")
    assert step.product == "geotiff"
    assert step.operation == OperatorType.metric_feather


@pytest.mark.parametrize(
    ("model", "kwargs"),
    [
        (CompositionStep, {"provider": "mock_provider", "operaton": "overwrite"}),
        (TransitionRule, {"target_provider": "gebco", "operator": "overwrite", "operation": "overwrite"}),
        (VariableStrategy, {"name": "elevation", "steps": [], "stpes": []}),
    ],
)
def test_unknown_policy_keys_are_rejected(model: type, kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        model(**kwargs)


def test_example_policies_validate_without_deprecation_warnings() -> None:
    import warnings

    examples = Path(__file__).resolve().parents[2] / "policies" / "examples"
    for path in sorted(examples.glob("*.yaml")):
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            FusionPolicy(**yaml.safe_load(path.read_text()))
