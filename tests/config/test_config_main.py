"""Tests for main configuration."""

import tempfile
from pathlib import Path

from gwsim_pop.config.main import MainConfiguration


class TestMainConfiguration:
    """Tests for MainConfiguration class."""

    def test_initialization(self):
        """Test that MainConfiguration can be initialized."""
        config = MainConfiguration()
        assert isinstance(config, MainConfiguration)
        assert config.config_version == "0.1.0"

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = MainConfiguration()
        assert config.config_version == "0.1.0"
        assert hasattr(config, "run")
        assert hasattr(config, "cosmology")
        assert hasattr(config, "selection")
        assert hasattr(config, "post_processing")
        assert hasattr(config, "advanced")

    def test_initialization_with_values(self):
        """Test that MainConfiguration can be initialized with specific values."""
        config = MainConfiguration(config_version="1.0.0")
        assert config.config_version == "1.0.0"

    def test_from_file_method(self):
        """Test that from_file method works correctly."""
        # Create a temporary YAML file with minimal configuration
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
config_version: "1.0.0"
run:
  name: test_run
""")
            temp_file = f.name

        try:
            config = MainConfiguration.from_file(temp_file)
            assert config.config_version == "1.0.0"
            assert config.run.name == "test_run"
        finally:
            Path(temp_file).unlink()

    def test_to_file_method(self):
        """Test that to_file method works correctly."""
        config = MainConfiguration(config_version="1.0.0")

        # Create a temporary file to write to
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_file = f.name

        try:
            config.to_file(temp_file)
            # Verify file was created
            assert Path(temp_file).exists()
        finally:
            Path(temp_file).unlink()
