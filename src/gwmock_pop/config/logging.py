"""Configuration for logging."""

from __future__ import annotations

from pydantic import BaseModel, Field

from gwmock_pop.utils.log import LoggingLevel


class LoggingConfiguration(BaseModel):
    """Configuration for the logging."""

    level: LoggingLevel = Field(default=LoggingLevel.INFO, description="Logging level.")
    """Logging level."""

    filename: str = Field(default="simulation.log", description="Name of the log file.")
    """Name of the log file."""
