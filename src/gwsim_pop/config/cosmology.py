"""Configuration for the cosmology."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CosmologicalArguments(BaseModel):
    """Arguments for the cosmological model."""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True, validate_default=True)
    """Allow additional parameters."""


class CosmologyConfiguration(BaseModel):
    """Configuration for the cosmology."""

    model: str = Field(default="Planck18", description="Cosmological model.")
    """Cosmological model."""

    arguments: CosmologicalArguments = Field(
        default_factory=CosmologicalArguments, description="Arguments for the model."
    )
    """Parameters for the model."""
