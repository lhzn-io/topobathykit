"""Policy files the viewers can name: the served policy directory and the cache's saved policies.

Fusion saves a copy of each policy it runs to ``<cache>/policies/<Name>.<hash>.yaml``, where
``<hash>`` is the name of the ``<cache>/fused_zarr/<hash>`` directory holding its cells. These
helpers pair the two, so the viewers can label a fused dataset with its policy name and let a
viewer pick a policy by file name without accepting arbitrary paths.
"""

from pathlib import Path
from typing import Any

import yaml

POLICY_SUFFIXES = (".yaml", ".yml")

# Summary fields for a fused directory with no saved policy file.
NO_POLICY_SUMMARY: dict[str, Any] = {
    "policy_file": None,
    "policy_name": None,
    "policy_version": None,
    "policy_description": None,
}


def policy_summary(path: Path) -> dict[str, Any]:
    """Name, version and description of a policy file; empty fields when it cannot be read."""
    doc: Any = None
    try:
        doc = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError):
        doc = None
    if not isinstance(doc, dict):
        doc = {}

    def text(key: str) -> str | None:
        value = doc.get(key)
        return " ".join(str(value).split()) if value not in (None, "") else None

    return {
        "policy_file": path.name,
        "policy_name": text("name"),
        "policy_version": text("version"),
        "policy_description": text("description"),
    }


def cache_dir_of(policy_file: str) -> str | None:
    """The fused cache directory a saved policy file belongs to (``Name.<hash>.yaml``)."""
    stem = Path(policy_file).stem
    if "." in stem:
        return stem.rsplit(".", 1)[1]
    return stem or None


def saved_policies(policies_dir: Path) -> dict[str, Path]:
    """Map fused cache directory name to the policy file saved for it."""
    found: dict[str, Path] = {}
    if not policies_dir.is_dir():
        return found
    for path in sorted(policies_dir.iterdir()):
        if path.suffix not in POLICY_SUFFIXES or not path.is_file():
            continue
        key = cache_dir_of(path.name)
        # A bare "<hash>.yaml" wins over "Name.<hash>.yaml" when both exist.
        if key and (key not in found or path.stem == key):
            found[key] = path
    return found


def resolve_policy_file(name: str, search_dirs: list[Path]) -> Path | None:
    """Find a policy by bare file name in ``search_dirs``; None for anything that is not one."""
    if not name or Path(name).name != name or not name.endswith(POLICY_SUFFIXES):
        return None
    for directory in search_dirs:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return None
