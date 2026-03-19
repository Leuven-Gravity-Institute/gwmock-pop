"""Configuration for the selection."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class SelectionConfiguration(BaseModel):
    """Configuration for the selection."""

    enabled: bool = Field(default=False, description="Whether to apply detectability.")
    """Whether to apply detectability."""

    method: Literal["snr"] = Field(default="snr", description="Method to apply detectability.")
    """Method to apply detectability."""

    arguments: dict = Field(default_factory=dict, description="Arguments for the method to apply detectability.")
    """Arguments for the method to apply detectability."""

    # TODO: After implementing this feature, remove this validator and remove `exclude=True` in main.py.
    @field_validator("enabled")
    @classmethod
    def validate_enabled(cls, value: bool) -> bool:
        """Validate the field 'enabled'.

        Args:
            value: Value of 'enabled'.

        Returns:
            Value of 'enabled'.

        Raises:
            ValueError: If value is True.
        """
        if value is True:
            raise ValueError("The selection feature is not available yet. Set it to 'False' and rerun.")
        return value
