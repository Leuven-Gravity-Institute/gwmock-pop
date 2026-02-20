"""Tests for population configuration."""

import pytest
from pydantic import ValidationError

from gwsim_pop.config.population import PopulationArguments, PopulationConfiguration


class TestPopulationArguments:
    """Tests for PopulationArguments class."""

    def test_initialization(self):
        """Test that PopulationArguments can be initialized."""
        args = PopulationArguments()
        assert isinstance(args, PopulationArguments)


class TestPopulationConfiguration:
    """Tests for PopulationConfiguration class."""

    def test_initialization_without_arguments(self):
        """Test that PopulationConfiguration can be initialized without arguments."""
        config = PopulationConfiguration(model="test_model")
        assert config.model == "test_model"
        assert isinstance(config.arguments, PopulationArguments)

    def test_initialization_with_arguments(self):
        """Test that PopulationConfiguration can be initialized with arguments."""
        # Test with default arguments
        config = PopulationConfiguration(model="test_model")
        assert config.model == "test_model"
        assert isinstance(config.arguments, PopulationArguments)

    def test_model_validation(self):
        """Test that model validation works correctly."""
        with pytest.raises(ValidationError):
            PopulationConfiguration()

    def test_valid_model_name(self):
        """Test that valid model names work."""
        config = PopulationConfiguration(model="valid_model_name")
        assert config.model == "valid_model_name"
