"""Configuration for the post-processing."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class HookArguments(BaseModel):
    """Arguments for the hook."""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True, validate_default=True)
    """Allow additional arguments."""


class HookConfiguration(BaseModel):
    """Configuration for the hook."""

    name: str = Field(..., description="Name of the hook.")
    """Name of the hook."""

    arguments: HookArguments = Field(default_factory=HookArguments, description="Arguments for the hook.")
    """Arguments for the hook."""


class PostProcessingConfiguration(BaseModel):
    """Configuration for post-processing."""

    hooks: list[HookConfiguration] = Field(
        default_factory=list, description="A list of functions to run in post-processing."
    )
    """A list of functions to run in post-processing."""

    # TODO: After implementing this feature, remove this validator and remove `exclude=True` in main.py.
    @field_validator("hooks")
    @classmethod
    def validate_hooks(cls, value: list[HookConfiguration]) -> list[HookConfiguration]:
        """Validate the hooks input.

        Args:
            value: Input list of hooks.

        Returns:
            A list of hook configuration.
        """
        if len(value) > 0:
            raise ValueError("Post-processing hook is not implemented yet.")
        return value
