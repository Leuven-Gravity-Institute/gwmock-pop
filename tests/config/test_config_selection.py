"""Tests for selection configuration."""

import pytest
from pydantic import ValidationError

from gwmock_pop.config.selection import SelectionConfiguration


class TestSelectionConfiguration:
    """Tests for SelectionConfiguration class."""

    def test_initialization(self):
        """Test that SelectionConfiguration can be initialized."""
        config = SelectionConfiguration()
        assert isinstance(config, SelectionConfiguration)

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = SelectionConfiguration()
        assert config.enabled is False
        assert config.method == "snr"
        assert config.arguments == {}

    def test_initialization_with_default_values(self):
        """Test that SelectionConfiguration can be initialized with default values."""
        config = SelectionConfiguration()
        assert config.enabled is False
        assert config.method == "snr"
        assert config.arguments == {}

    def test_enabled_validation_raises_error(self):
        """Test that enabling selection raises validation error."""
        with pytest.raises(ValidationError, match="The selection feature is not available yet"):
            SelectionConfiguration(enabled=True)

    def test_enabled_false_works(self):
        """Test that setting enabled to False works."""
        config = SelectionConfiguration(enabled=False)
        assert config.enabled is False

    def test_initialization_with_arguments(self):
        """Test that SelectionConfiguration can be initialized with arguments."""
        config = SelectionConfiguration(arguments={"threshold": 5.0})
        assert config.enabled is False
        assert config.arguments == {"threshold": 5.0}
