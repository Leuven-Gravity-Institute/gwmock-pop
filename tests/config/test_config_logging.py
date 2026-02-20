"""Tests for the logging configuration."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from gwsim_pop.config.logging import LoggingConfiguration
from gwsim_pop.utils.log import LoggingLevel


class TestLoggingConfiguration:
    """Test cases for LoggingConfiguration."""

    def test_default_values(self) -> None:
        """Test that default values are correctly set."""
        config = LoggingConfiguration()
        assert config.level == LoggingLevel.INFO
        assert config.filename == "simulation.log"

    def test_explicit_values(self) -> None:
        """Test that explicit values are correctly set."""
        config = LoggingConfiguration(level=LoggingLevel.DEBUG, filename="custom.log")
        assert config.level == LoggingLevel.DEBUG
        assert config.filename == "custom.log"

    def test_level_field_description(self) -> None:
        """Test that level field has correct description."""
        config = LoggingConfiguration()
        # Access field description through model_fields
        fields = config.__class__.model_fields
        assert fields["level"].description == "Logging level."

    def test_filename_field_description(self) -> None:
        """Test that filename field has correct description."""
        config = LoggingConfiguration()
        # Access field description through model_fields
        fields = config.__class__.model_fields
        assert fields["filename"].description == "Name of the log file."

    def test_configuration_from_dict(self) -> None:
        """Test that configuration can be created from a dictionary."""
        config_dict = {"level": "DEBUG", "filename": "test.log"}
        config = LoggingConfiguration(**config_dict)
        assert config.level == LoggingLevel.DEBUG
        assert config.filename == "test.log"

    def test_invalid_level_value(self) -> None:
        """Test that invalid level values raise ValidationError."""
        # Test with a level that is not in LoggingLevel enum
        with pytest.raises(ValidationError):
            LoggingConfiguration(level="INVALID_LEVEL")

    def test_level_enum_values(self) -> None:
        """Test that all valid level enum values work correctly."""
        # Test that all enum values can be used
        valid_levels = [
            LoggingLevel.DEBUG,
            LoggingLevel.INFO,
            LoggingLevel.WARNING,
            LoggingLevel.ERROR,
            LoggingLevel.CRITICAL,
        ]

        for level in valid_levels:
            config = LoggingConfiguration(level=level)
            assert config.level == level

    def test_filename_default_value(self) -> None:
        """Test that filename default value is correct."""
        config = LoggingConfiguration()
        assert config.filename == "simulation.log"

    def test_filename_custom_value(self) -> None:
        """Test that custom filename values work correctly."""
        config = LoggingConfiguration(filename="custom.log")
        assert config.filename == "custom.log"

    def test_partial_configuration(self) -> None:
        """Test that partial configuration works correctly."""
        config = LoggingConfiguration(level=LoggingLevel.WARNING)
        assert config.level == LoggingLevel.WARNING
        assert config.filename == "simulation.log"

    def test_empty_configuration(self) -> None:
        """Test that empty configuration works correctly."""
        config = LoggingConfiguration()
        assert config.level == LoggingLevel.INFO
        assert config.filename == "simulation.log"
