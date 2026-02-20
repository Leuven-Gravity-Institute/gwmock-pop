"""Configuration for the population."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PopulationArguments(BaseModel):
    """Population arguments."""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True, validate_default=True)
    """Allow additional arguments."""


class PopulationConfiguration(BaseModel):
    """Configuration for the population."""

    model: str = Field(..., description="Name of the model.")
    """Name of the model."""

    arguments: PopulationArguments = Field(
        default_factory=PopulationArguments, description="Arguments for the population."
    )
    """Arguments for the population."""
