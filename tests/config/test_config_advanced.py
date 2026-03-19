"""Tests for the advanced configuration."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from gwmock_pop.config.advanced import AdvancedConfiguration


class TestAdvancedConfiguration:
    """Test cases for AdvancedConfiguration."""

    def test_default_values(self) -> None:
        """Test that the default values are correctly set."""
        config = AdvancedConfiguration()
        assert config.backend == "jax"
        assert config.jit is True
        assert config.vectorization == "vmap"
        assert config.device == "auto"

    def test_explicit_values(self) -> None:
        """Test that explicit values are correctly set."""
        config = AdvancedConfiguration(backend="jax", jit=False, vectorization="manual loop", device="cpu")
        assert config.backend == "jax"
        assert config.jit is False
        assert config.vectorization == "manual loop"
        assert config.device == "cpu"

    def test_backend_validation_success(self) -> None:
        """Test that jax backend validation passes."""
        config = AdvancedConfiguration(backend="jax")
        assert config.backend == "jax"

    def test_backend_validation_failure(self) -> None:
        """Test that numpy backend validation fails as expected."""
        with pytest.raises(ValidationError) as exc_info:
            AdvancedConfiguration(backend="numpy")

        assert "backend='numpy' is not implemented yet" in str(exc_info.value)

    def test_field_descriptions(self) -> None:
        """Test that field descriptions are correctly set."""
        config = AdvancedConfiguration()
        # Access field descriptions through model_fields
        fields = config.__class__.model_fields

        assert fields["backend"].description == "Backend for the computation."
        assert fields["jit"].description == "Whether to jit critical functions."
        assert fields["vectorization"].description == "Vectorization method."
        assert fields["device"].description == "Device to use for the run."

    def test_all_valid_backends(self) -> None:
        """Test that all valid backend values work correctly."""
        # Test that jax is valid
        config = AdvancedConfiguration(backend="jax")
        assert config.backend == "jax"

    def test_partial_configuration(self) -> None:
        """Test that partial configuration works correctly."""
        config = AdvancedConfiguration(jit=False)
        assert config.backend == "jax"
        assert config.jit is False
        assert config.vectorization == "vmap"
        assert config.device == "auto"

    def test_configuration_from_dict(self) -> None:
        """Test that configuration can be created from a dictionary."""
        config_dict = {"backend": "jax", "jit": True, "vectorization": "pmap", "device": "gpu"}
        config = AdvancedConfiguration(**config_dict)
        assert config.backend == "jax"
        assert config.jit is True
        assert config.vectorization == "pmap"
        assert config.device == "gpu"
