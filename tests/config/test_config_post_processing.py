"""Tests for post-processing configuration."""

import pytest
from pydantic import ValidationError

from gwsim_pop.config.post_processing import HookArguments, HookConfiguration, PostProcessingConfiguration


class TestHookArguments:
    """Tests for HookArguments class."""

    def test_initialization(self):
        """Test that HookArguments can be initialized."""
        args = HookArguments()
        assert isinstance(args, HookArguments)


class TestHookConfiguration:
    """Tests for HookConfiguration class."""

    def test_initialization_without_arguments(self):
        """Test that HookConfiguration can be initialized without arguments."""
        config = HookConfiguration(name="test_hook")
        assert config.name == "test_hook"
        assert isinstance(config.arguments, HookArguments)

    def test_initialization_with_arguments(self):
        """Test that HookConfiguration can be initialized with arguments."""
        args = HookArguments()
        config = HookConfiguration(name="test_hook", arguments=args)
        assert config.name == "test_hook"
        assert isinstance(config.arguments, HookArguments)

    def test_name_validation(self):
        """Test that name validation works correctly."""
        with pytest.raises(ValidationError):
            HookConfiguration()


class TestPostProcessingConfiguration:
    """Tests for PostProcessingConfiguration class."""

    def test_initialization(self):
        """Test that PostProcessingConfiguration can be initialized."""
        config = PostProcessingConfiguration()
        assert isinstance(config, PostProcessingConfiguration)
        assert config.hooks == []

    def test_hooks_validation_empty(self):
        """Test that empty hooks list passes validation."""
        config = PostProcessingConfiguration(hooks=[])
        assert config.hooks == []

    def test_hooks_validation_non_empty_raises_error(self):
        """Test that non-empty hooks list raises validation error."""
        with pytest.raises(ValidationError, match=r"Post-processing hook is not implemented yet."):
            PostProcessingConfiguration(hooks=[HookConfiguration(name="test_hook")])
