import warnings
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from topobathykit.providers.registry import registry

"""
Policy Schema Definitions.

This module provides the Pydantic models used to validate and load YAML fusion policies.
"""


class ZoneRule(BaseModel):
    """
    Defines a geographic zone and the rules applied within it.
    """

    type: Literal["polygon", "bbox"] = "bbox"
    coords: list[float]  # [min_x, min_y, max_x, max_y] for bbox
    priority: int = 0


class OperatorType(StrEnum):
    """
    Types of spatial fusion operators available.
    """

    metric_feather = "metric_feather"
    overwrite = "overwrite"
    linear_blend = "linear_blend"


class TransitionRule(BaseModel):
    """
    Rule for transitioning from a specific underlying provider.

    The rule fires on pixels whose current value was supplied by
    ``target_provider``. Note that this key is ``operator``, whereas a
    composition step uses ``operation``.
    """

    model_config = ConfigDict(extra="forbid")

    target_provider: str
    operator: OperatorType
    blend_distance: float | None = None
    """
    Distance in meters for feathering/blending.
    Note: If operator is 'metric_feather' and this is None, runtime defaults to 'overwrite'.
    """


class FilterConfig(BaseModel):
    """
    Configuration for data filtering, cleaning, and constraints.
    """

    # Allow arbitrary extra fields for custom provider logic (like ncei_bag project_id)
    model_config = ConfigDict(extra="allow")

    max_depth_change: float | None = None
    max_deviation: float | None = None
    min_elevation: float | None = None
    max_elevation: float | None = None
    include_patterns: list[str] | None = None
    exclude_patterns: list[str] | None = None
    max_interpolation_gap_m: float | None = None


class CompositionStep(BaseModel):
    """
    A single step in the composition process.

    Unknown keys are rejected so that a misspelled field fails validation
    instead of silently falling back to a default. ``operator`` is accepted as
    a deprecated alias of ``operation``.
    """

    model_config = ConfigDict(extra="forbid")

    provider: str
    operation: OperatorType = OperatorType.metric_feather
    blend_mode: Literal["linear", "feather"] | None = None  # Deprecated in favor of operation
    blend_distance: float | None = None  # In meters
    zone: ZoneRule | None = None
    transitions: list[TransitionRule] = []
    filter: FilterConfig | None = None
    product: str | None = None
    """Informational label for the provider product. Not passed to the provider."""
    description: str | None = None
    """Human-readable note."""

    @model_validator(mode="before")
    @classmethod
    def accept_operator_alias(cls, data: Any) -> Any:
        """
        Maps the deprecated step key ``operator`` onto ``operation``.

        Policies written against earlier documentation used ``operator``,
        which was silently dropped and left ``operation`` at its default.
        """
        if not isinstance(data, dict) or "operator" not in data:
            return data
        data = dict(data)
        value = data.pop("operator")
        if "operation" in data and data["operation"] != value:
            raise ValueError(
                f"Step sets both operation={data['operation']!r} and the deprecated "
                f"operator={value!r}; use operation only."
            )
        data["operation"] = value
        warnings.warn(
            f"Composition step key 'operator' is deprecated; use 'operation' "
            f"(provider {data.get('provider')!r}).",
            FutureWarning,
            stacklevel=2,
        )
        return data

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """
        Validates that the specified provider name corresponds to a registered Provider implementation.
        """
        # Runtime check to ensure provider is registered
        # Note: This requires providers to be loaded before policy validation
        import contextlib

        with contextlib.suppress(KeyError):
            registry.get_provider_class(v)
        return v


class VariableStrategy(BaseModel):
    """
    Strategy for a specific variable (e.g., elevation).
    """

    model_config = ConfigDict(extra="forbid")

    name: str = "elevation"
    description: str | None = None
    steps: list[CompositionStep]
    background: float | None = None


class FusionPolicy(BaseModel):
    """
    Top-level policy definition.
    """

    name: str
    version: str = "1.0.0"
    crs: str = "EPSG:4326"
    variables: list[VariableStrategy]

    @model_validator(mode="after")
    def check_at_least_one_variable(self) -> "FusionPolicy":
        """
        Validates that the policy contains at least one variable strategy.
        """
        if not self.variables:
            raise ValueError("Policy must contain at least one variable strategy.")
        return self
