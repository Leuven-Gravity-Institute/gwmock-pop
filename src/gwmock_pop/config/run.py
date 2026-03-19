"""Configuration for the run control."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from gwmock_pop.config.logging import LoggingConfiguration
from gwmock_pop.config.output import OutputConfiguration


class RunConfiguration(BaseModel):
    """Configuration for the run control."""

    name: str = Field(default="simulation", description="Run identifier.")
    """Run identifier."""

    seed: int = Field(default=42, description="Global RNG seed (overridable via CLI).")
    """Global RNG seed (overridable via CLI)."""

    mode: Literal["fixed_n_samples", "duration"] = Field(
        default="fixed_n_samples",
        description="Mode of simulation. Supported: 'fixed_n_samples', 'duration'.",
    )
    """Mode of simulation. Supported: 'fixed_n_samples', 'duration'."""

    n_samples: int = Field(
        default=1_000_000,
        gt=0,
        description="Number of samples. Only used when 'mode' is 'fixed_n_samples'.",
    )
    """Number of samples. Only used when 'mode' is 'fixed_n_samples'."""

    duration: float = Field(
        default=1.0,
        gt=0.0,
        description="Duration in year (365 days). This is used when 'mode' is 'duration'.",
    )
    """Duration in year (365 days). This is used when 'mode' is 'duration'."""

    output: OutputConfiguration = Field(
        default_factory=OutputConfiguration, description="Configuration for the output."
    )
    """Configuration for the output."""

    logging: LoggingConfiguration = Field(
        default_factory=LoggingConfiguration, description="Configuration for the logging."
    )
    """Configuration for the logging."""
