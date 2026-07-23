"""Configuration for the output."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class OutputConfiguration(BaseModel):
    """Configuration for the output."""

    directory: str = Field(default=".", description="Output directory.")
    """Output directory."""

    format: Literal["hdf5", "npz", "csv"] = Field(default="hdf5", description="Storage format.")
    """Storage format."""

    compression: Literal["gzip", "lzf", "szip"] | None = Field(default="gzip", description="Compression filter.")
    """Compression filter."""

    overwrite: bool = Field(default=False, description="Whether or not to overwrite existing files.")
    """Whether or not to overwrite existing files."""

    save_metadata: bool = Field(default=True, description="Whether or not to save metadata.")
    """Whether or not to save metadata."""
