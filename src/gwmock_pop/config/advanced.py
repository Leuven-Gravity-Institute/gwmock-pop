"""Configuration for the advanced features."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class AdvancedConfiguration(BaseModel):
    """Configuration for the advanced features."""

    backend: Literal["jax", "numpy"] = Field(default="jax", description="Backend for the computation.")
    """Backend for the computation."""

    jit: bool = Field(default=True, description="Whether to jit critical functions.")
    """Whether to jit critical functions."""

    vectorization: Literal["vmap", "manual loop", "pmap"] = Field(default="vmap", description="Vectorization method.")
    """Vectorization method."""

    device: Literal["cpu", "gpu", "tpu", "auto"] = Field(default="auto", description="Device to use for the run.")
    """Device to use for the run."""

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, value: Literal["jax", "numpy"]) -> Literal["jax", "numpy"]:
        """Validate the backend to use.

        Args:
            value: Name of the backend.

        Returns:
            Name of the validated backend.

        Raises:
            ValueError: If value is numpy.
        """
        if value != "jax":
            raise ValueError(f"backend='{value}' is not implemented yet.")
        return value
