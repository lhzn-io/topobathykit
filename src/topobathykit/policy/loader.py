import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from .schema import FusionPolicy


def hash_policy(content: dict[str, Any]) -> str:
    """
    Compute a stable SHA256 hash of the policy content.
    Sorts keys to ensure determinism.
    """
    # Use json.dumps with sort_keys to ensure deterministic serialization
    serialized = json.dumps(content, sort_keys=True).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


@lru_cache(maxsize=32)
def load_policy(path: str) -> FusionPolicy:
    """
    Load a policy from a YAML file.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Policy file not found: {path}")

    with open(p) as f:
        content = yaml.safe_load(f)

    # Validate against schema
    policy = FusionPolicy(**content)

    return policy


def load_policy_from_str(yaml_content: str) -> FusionPolicy:
    """
    Load a policy directly from a YAML string.
    """
    content = yaml.safe_load(yaml_content)

    # Validate against schema
    policy = FusionPolicy(**content)

    return policy


def clear_policy_cache() -> None:
    """Clear the in-memory policy cache."""
    load_policy.cache_clear()


def generate_provider_legend(policy: FusionPolicy) -> dict[int, str]:
    """
    Generate a legend mapping integer IDs to provider keys.
    Extracts all unique provider keys referenced in the policy variables.
    IDs start at 1 (0 is reserved for background/no-data).
    """
    providers = set()
    for variable in policy.variables:
        for step in variable.steps:
            providers.add(step.provider)

    # Sort providers for deterministic ID assignment
    sorted_providers = sorted(list(providers))

    legend = {}
    for idx, provider in enumerate(sorted_providers, start=1):
        legend[idx] = provider

    return legend
