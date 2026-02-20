"""Models for the configuration."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from gwsim_pop.config.advanced import AdvancedConfiguration
from gwsim_pop.config.cosmology import CosmologyConfiguration
from gwsim_pop.config.post_processing import PostProcessingConfiguration
from gwsim_pop.config.run import RunConfiguration
from gwsim_pop.config.selection import SelectionConfiguration
from gwsim_pop.utils.yaml import read_yaml, write_yaml


class MainConfiguration(BaseModel):
    """Configuration of simulation."""

    config_version: str = Field(default="0.1.0", description="Version of the configuration file.")
    """Version of the configuration file."""

    run: RunConfiguration = Field(default_factory=RunConfiguration, description="Configuration for the run.")
    """Configuration for the run."""

    cosmology: CosmologyConfiguration = Field(
        default_factory=CosmologyConfiguration, description="Configuration for the cosmology."
    )
    """Configuration for the cosmology."""

    selection: SelectionConfiguration = Field(
        default_factory=SelectionConfiguration, description="Configuration for the selection.", exclude=True
    )
    """Configuration for the selection."""

    post_processing: PostProcessingConfiguration = Field(
        default_factory=PostProcessingConfiguration, description="Configuration for the post-processing.", exclude=True
    )
    """Configuration for the post-processing."""

    advanced: AdvancedConfiguration = Field(
        default_factory=AdvancedConfiguration, description="Configuration fro the advanced features."
    )
    """Configuration for the advanced features."""

    @classmethod
    def from_file(cls, filename: str | Path, encoding: str = "utf-8") -> MainConfiguration:
        """Read from file.

        Args:
            filename: File name.
            encoding: File encoding.

        Returns:
            Configuration.

        """
        data = read_yaml(filename=filename, encoding=encoding)
        return cls(**data)

    def to_file(
        self,
        filename: str | Path,
        encoding: str = "utf-8",
        exclude_none: bool = True,
        exclude_defaults: bool = False,
        round_trip: bool = False,
        **kwargs,
    ) -> None:
        """Write to file.

        Args:
            filename: File name.
            encoding: Encoding of the file.
            exclude_none: Exclude None entries.
            exclude_defaults: Exclude entries set to default.
            round_trip: If True, use ruyaml to preserve comments/order.
            **kwargs: Extra arguments passed to yaml.safe_dump (or ruyaml.YAML.dump).
        """
        data = self.model_dump(mode="python", exclude_none=exclude_none, exclude_defaults=exclude_defaults)
        write_yaml(filename=filename, data=data, encoding=encoding, round_trip=round_trip, **kwargs)
