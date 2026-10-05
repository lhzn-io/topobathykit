from enum import StrEnum
from typing import Literal

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
    """

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
    """

    provider: str
    operation: OperatorType = OperatorType.metric_feather
    blend_mode: Literal["linear", "feather"] | None = None  # Deprecated in favor of operation
    blend_distance: float | None = None  # In meters
    zone: ZoneRule | None = None
    transitions: list[TransitionRule] = []
    filter: FilterConfig | None = None

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

    name: str = "elevation"
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
