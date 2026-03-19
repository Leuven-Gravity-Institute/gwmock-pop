"""Tests for the cosmology configuration."""

from __future__ import annotations

from gwsim_pop.config.cosmology import CosmologicalArguments, CosmologyConfiguration


class TestCosmologicalArguments:
    """Test cases for CosmologicalArguments."""

    def test_default_values(self) -> None:
        """Test that default values are correctly set."""
        args = CosmologicalArguments()
        assert args.model_config == {"extra": "allow", "arbitrary_types_allowed": True, "validate_default": True}

    def test_extra_fields_allowed(self) -> None:
        """Test that extra fields are allowed."""
        args = CosmologicalArguments(extra_field="test_value")
        assert args.model_extra is not None
        assert args.model_extra.get("extra_field") == "test_value"

    def test_arbitrary_types_allowed(self) -> None:
        """Test that arbitrary types are allowed."""
        # The CosmologicalArguments is a generic class, so we test by passing various valid arguments
        args = CosmologicalArguments()
        assert args is not None


class TestCosmologyConfiguration:
    """Test cases for CosmologyConfiguration."""

    def test_default_values(self) -> None:
        """Test that default values are correctly set."""
        config = CosmologyConfiguration()
        assert config.model == "Planck18"
        assert isinstance(config.arguments, CosmologicalArguments)
        assert config.arguments.model_config == {
            "extra": "allow",
            "arbitrary_types_allowed": True,
            "validate_default": True,
        }

    def test_explicit_values(self) -> None:
        """Test that explicit values are correctly set."""
        config = CosmologyConfiguration(model="WMAP9", arguments=CosmologicalArguments())
        assert config.model == "WMAP9"
        assert isinstance(config.arguments, CosmologicalArguments)

    def test_arguments_default_factory(self) -> None:
        """Test that arguments uses the default factory correctly."""
        config = CosmologyConfiguration()
        assert isinstance(config.arguments, CosmologicalArguments)
        # Should have the default model_config
        assert config.arguments.model_config == {
            "extra": "allow",
            "arbitrary_types_allowed": True,
            "validate_default": True,
        }

    def test_model_field_description(self) -> None:
        """Test that model field has correct description."""
        config = CosmologyConfiguration()
        # Access field description through model_fields
        fields = config.__class__.model_fields
        assert fields["model"].description == "Cosmological model."

    def test_arguments_field_description(self) -> None:
        """Test that arguments field has correct description."""
        config = CosmologyConfiguration()
        # Access field description through model_fields
        fields = config.__class__.model_fields
        assert fields["arguments"].description == "Arguments for the model."

    def test_configuration_from_dict(self) -> None:
        """Test that configuration can be created from a dictionary."""
        config_dict = {"model": "WMAP9", "arguments": {}}
        config = CosmologyConfiguration(**config_dict)
        assert config.model == "WMAP9"
        assert isinstance(config.arguments, CosmologicalArguments)

    def test_nested_configuration(self) -> None:
        """Test nested configuration with arguments."""
        args = CosmologicalArguments()
        config = CosmologyConfiguration(model="CustomModel", arguments=args)
        assert config.model == "CustomModel"
        assert isinstance(config.arguments, CosmologicalArguments)

    def test_empty_arguments(self) -> None:
        """Test configuration with empty arguments."""
        config = CosmologyConfiguration(arguments=CosmologicalArguments())
        assert config.model == "Planck18"
        assert isinstance(config.arguments, CosmologicalArguments)
        # Check that arguments can accept extra fields
        extra_args = CosmologicalArguments(extra_test="value")
        assert extra_args.model_extra is not None
        assert extra_args.model_extra.get("extra_test") == "value"
