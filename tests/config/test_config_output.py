"""Tests for the output configuration."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from gwmock_pop.config.output import OutputConfiguration


class TestOutputConfiguration:
    """Test cases for OutputConfiguration."""

    def test_default_values(self) -> None:
        """Test that default values are correctly set."""
        config = OutputConfiguration()
        assert config.directory == "."
        assert config.format == "hdf5"
        assert config.compression == "gzip"
        assert config.overwrite is False
        assert config.save_metadata is True

    def test_explicit_values(self) -> None:
        """Test that explicit values are correctly set."""
        config = OutputConfiguration(
            directory="/custom/path", format="npz", compression="lzf", overwrite=True, save_metadata=False
        )
        assert config.directory == "/custom/path"
        assert config.format == "npz"
        assert config.compression == "lzf"
        assert config.overwrite is True
        assert config.save_metadata is False

    def test_directory_field_description(self) -> None:
        """Test that directory field has correct description."""
        config = OutputConfiguration()
        # Access field description through model_fields
        fields = config.__class__.model_fields
        assert fields["directory"].description == "Output directory."

    def test_format_field_description(self) -> None:
        """Test that format field has correct description."""
        config = OutputConfiguration()
        # Access field description through model_fields
        fields = config.__class__.model_fields
        assert fields["format"].description == "Storage format."

    def test_compression_field_description(self) -> None:
        """Test that compression field has correct description."""
        config = OutputConfiguration()
        # Access field description through model_fields
        fields = config.__class__.model_fields
        assert fields["compression"].description == "Compression filter."

    def test_overwrite_field_description(self) -> None:
        """Test that overwrite field has correct description."""
        config = OutputConfiguration()
        # Access field description through model_fields
        fields = config.__class__.model_fields
        assert fields["overwrite"].description == "Whether or not to overwrite existing files."

    def test_save_metadata_field_description(self) -> None:
        """Test that save_metadata field has correct description."""
        config = OutputConfiguration()
        # Access field description through model_fields
        fields = config.__class__.model_fields
        assert fields["save_metadata"].description == "Whether or not to save metadata."

    def test_configuration_from_dict(self) -> None:
        """Test that configuration can be created from a dictionary."""
        config_dict = {
            "directory": "/custom/path",
            "format": "csv",
            "compression": "szip",
            "overwrite": True,
            "save_metadata": False,
        }
        config = OutputConfiguration(**config_dict)
        assert config.directory == "/custom/path"
        assert config.format == "csv"
        assert config.compression == "szip"
        assert config.overwrite is True
        assert config.save_metadata is False

    def test_valid_format_values(self) -> None:
        """Test that all valid format values work correctly."""
        valid_formats = ["hdf5", "npz", "csv"]

        for fmt in valid_formats:
            config = OutputConfiguration(format=fmt)
            assert config.format == fmt

    def test_valid_compression_values(self) -> None:
        """Test that all valid compression values work correctly."""
        valid_compressions = [None, "gzip", "lzf", "szip"]

        for comp in valid_compressions:
            config = OutputConfiguration(compression=comp)
            assert config.compression == comp

    def test_directory_default_value(self) -> None:
        """Test that directory default value is correct."""
        config = OutputConfiguration()
        assert config.directory == "."

    def test_format_default_value(self) -> None:
        """Test that format default value is correct."""
        config = OutputConfiguration()
        assert config.format == "hdf5"

    def test_compression_default_value(self) -> None:
        """Test that compression default value is correct."""
        config = OutputConfiguration()
        assert config.compression == "gzip"

    def test_overwrite_default_value(self) -> None:
        """Test that overwrite default value is correct."""
        config = OutputConfiguration()
        assert config.overwrite is False

    def test_save_metadata_default_value(self) -> None:
        """Test that save_metadata default value is correct."""
        config = OutputConfiguration()
        assert config.save_metadata is True

    def test_partial_configuration(self) -> None:
        """Test that partial configuration works correctly."""
        config = OutputConfiguration(directory="/custom", overwrite=True)
        assert config.directory == "/custom"
        assert config.format == "hdf5"  # default
        assert config.compression == "gzip"  # default
        assert config.overwrite is True
        assert config.save_metadata is True  # default

    def test_empty_configuration(self) -> None:
        """Test that empty configuration works correctly."""
        config = OutputConfiguration()
        assert config.directory == "."
        assert config.format == "hdf5"
        assert config.compression == "gzip"
        assert config.overwrite is False
        assert config.save_metadata is True

    def test_invalid_format_value(self) -> None:
        """Test that invalid format values raise ValidationError."""
        with pytest.raises(ValidationError):
            OutputConfiguration(format="invalid_format")

    def test_invalid_compression_value(self) -> None:
        """Test that invalid compression values raise ValidationError."""
        with pytest.raises(ValidationError):
            OutputConfiguration(compression="invalid_compression")
