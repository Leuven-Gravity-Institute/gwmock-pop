"""Tests for GraphSimulator."""

from __future__ import annotations

import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
import yaml

import gwsim_pop.simulators.graph as graph_module
from gwsim_pop.simulators.graph import GraphSimulator

# Constants for test data
DEFAULT_N_SAMPLES = 100
DEFAULT_N_PARAMETERS = 2
DEFAULT_N_PARAMETERS_MULTI = 3


class TestGraphSimulator:
    """Test suite for GraphSimulator class."""

    def test_init_with_config(self) -> None:
        """Test initialization with a simple config."""
        config = {
            "mass_1": {
                "sampler": {
                    "function": "gwsim_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
                    "arguments": {
                        "alpha_1": 1.72,
                        "alpha_2": 4.51,
                        "transition": 35.6,
                        "minimum": 5.06,
                        "maximum": 300.0,
                        "mean_1": 9.76,
                        "sigma_1": 0.649,
                        "mean_2": 32.8,
                        "sigma_2": 3.92,
                        "taper_range": 4.32,
                        "lambda_0": 0.361,
                        "lambda_1": 0.586,
                    },
                },
            },
        }

        simulator = GraphSimulator(config=config)
        assert simulator is not None
        assert simulator.parameter_names == ["mass_1"]

    def test_init_with_multiple_parameters(self) -> None:
        """Test initialization with multiple parameters."""
        config = {
            "mass_1": {
                "sampler": {
                    "function": "gwsim_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
                    "arguments": {
                        "alpha_1": 1.72,
                        "alpha_2": 4.51,
                        "transition": 35.6,
                        "minimum": 5.06,
                        "maximum": 300.0,
                        "mean_1": 9.76,
                        "sigma_1": 0.649,
                        "mean_2": 32.8,
                        "sigma_2": 3.92,
                        "taper_range": 4.32,
                        "lambda_0": 0.361,
                        "lambda_1": 0.586,
                    },
                },
            },
            "mass_ratio": {
                "sampler": {
                    "function": "gwsim_pop.samplers.planck_tapered_conditional_ratio_power_law",
                    "arguments": {"denominator": "@mass_1"},
                },
            },
        }

        simulator = GraphSimulator(config=config)
        assert simulator.parameter_names == ["mass_1", "mass_ratio"]

    def test_graph_nodes(self) -> None:
        """Test that graph nodes are correctly built."""
        config = {
            "mass_1": {
                "sampler": {
                    "function": "gwsim_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
                    "arguments": {
                        "alpha_1": 1.72,
                        "alpha_2": 4.51,
                        "transition": 35.6,
                        "minimum": 5.06,
                        "maximum": 300.0,
                        "mean_1": 9.76,
                        "sigma_1": 0.649,
                        "mean_2": 32.8,
                        "sigma_2": 3.92,
                        "taper_range": 4.32,
                        "lambda_0": 0.361,
                        "lambda_1": 0.586,
                    },
                },
            },
            "mass_ratio": {
                "sampler": {
                    "function": "gwsim_pop.samplers.planck_tapered_conditional_ratio_power_law",
                    "arguments": {"denominator": "@mass_1"},
                },
            },
        }

        simulator = GraphSimulator(config=config)
        graph_nodes = list(simulator._graph.nodes())
        assert "mass_1" in graph_nodes
        assert "mass_ratio" in graph_nodes

    def test_source_type(self) -> None:
        """Test that source_type returns 'bbh'."""
        config = {
            "mass_1": {
                "sampler": {
                    "function": "gwsim_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
                    "arguments": {
                        "alpha_1": 1.72,
                        "alpha_2": 4.51,
                        "transition": 35.6,
                        "minimum": 5.06,
                        "maximum": 300.0,
                        "mean_1": 9.76,
                        "sigma_1": 0.649,
                        "mean_2": 32.8,
                        "sigma_2": 3.92,
                        "taper_range": 4.32,
                        "lambda_0": 0.361,
                        "lambda_1": 0.586,
                    },
                },
            },
        }

        simulator = GraphSimulator(config=config)
        assert simulator.source_type == "bbh"

    def test_reset(self) -> None:
        """Test that reset clears sampled values."""
        config = {
            "mass_1": {
                "sampler": {
                    "function": "gwsim_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
                    "arguments": {
                        "alpha_1": 1.72,
                        "alpha_2": 4.51,
                        "transition": 35.6,
                        "minimum": 5.06,
                        "maximum": 300.0,
                        "mean_1": 9.76,
                        "sigma_1": 0.649,
                        "mean_2": 32.8,
                        "sigma_2": 3.92,
                        "taper_range": 4.32,
                        "lambda_0": 0.361,
                        "lambda_1": 0.586,
                    },
                },
            },
        }

        simulator = GraphSimulator(config=config)
        _ = simulator()

        assert len(simulator._sampled_values) > 0

        simulator.reset()

        assert len(simulator._sampled_values) == 0

    def test_from_config_file_yaml(self) -> None:
        """Test loading config from YAML file."""
        config = {
            "mass_1": {
                "sampler": {
                    "function": "gwsim_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
                    "arguments": {
                        "alpha_1": 1.72,
                        "alpha_2": 4.51,
                        "transition": 35.6,
                        "minimum": 5.06,
                        "maximum": 300.0,
                        "mean_1": 9.76,
                        "sigma_1": 0.649,
                        "mean_2": 32.8,
                        "sigma_2": 3.92,
                        "taper_range": 4.32,
                        "lambda_0": 0.361,
                        "lambda_1": 0.586,
                    },
                },
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            temp_path = f.name

        try:
            simulator = GraphSimulator.from_config_file(temp_path)
            assert simulator is not None
            assert simulator.parameter_names == ["mass_1"]
        finally:
            Path(temp_path).unlink()

    def test_from_config_file_toml(self) -> None:
        """Test loading config from TOML file."""
        config = """
[parameters.mass_1.sampler]
function = "gwsim_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks"

[parameters.mass_1.sampler.arguments]
alpha_1 = 1.72
alpha_2 = 4.51
transition = 35.6
minimum = 5.06
maximum = 300.0
mean_1 = 9.76
sigma_1 = 0.649
mean_2 = 32.8
sigma_2 = 3.92
taper_range = 4.32
lambda_0 = 0.361
lambda_1 = 0.586
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config)
            temp_path = f.name

        try:
            simulator = GraphSimulator.from_config_file(temp_path)
            assert simulator is not None
            assert simulator.parameter_names == ["mass_1"]
        finally:
            Path(temp_path).unlink()

    def test_from_config_file_unsupported_format(self) -> None:
        """Test that unsupported file format raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{}")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Suffix of config_path"):
                GraphSimulator.from_config_file(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_from_config_file_with_parameters_section(self) -> None:
        """Test loading config with parameters section."""
        config = {
            "parameters": {
                "mass_1": {
                    "sampler": {
                        "function": "gwsim_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
                        "arguments": {
                            "alpha_1": 1.72,
                            "alpha_2": 4.51,
                            "transition": 35.6,
                            "minimum": 5.06,
                            "maximum": 300.0,
                            "mean_1": 9.76,
                            "sigma_1": 0.649,
                            "mean_2": 32.8,
                            "sigma_2": 3.92,
                            "taper_range": 4.32,
                            "lambda_0": 0.361,
                            "lambda_1": 0.586,
                        },
                    },
                },
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            temp_path = f.name

        try:
            simulator = GraphSimulator.from_config_file(temp_path)
            assert simulator is not None
            assert simulator.parameter_names == ["mass_1"]
        finally:
            Path(temp_path).unlink()

    def test_init_with_custom_n_samples(self) -> None:
        """Test initialization with custom n_samples."""
        config = {
            "mass_1": {
                "sampler": {
                    "function": "gwsim_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
                    "arguments": {
                        "alpha_1": 1.72,
                        "alpha_2": 4.51,
                        "transition": 35.6,
                        "minimum": 5.06,
                        "maximum": 300.0,
                        "mean_1": 9.76,
                        "sigma_1": 0.649,
                        "mean_2": 32.8,
                        "sigma_2": 3.92,
                        "taper_range": 4.32,
                        "lambda_0": 0.361,
                        "lambda_1": 0.586,
                    },
                },
            },
        }

        n_samples = 50
        simulator = GraphSimulator(config=config, n_samples=n_samples)
        assert simulator is not None
        assert simulator._n_samples == n_samples

        result = simulator()
        assert result.shape[0] == n_samples
        assert result.shape[1] == len(simulator.parameter_names)

    def test_simulate_with_multiple_parameters(self) -> None:
        """Test simulation with multiple parameters and dependencies."""
        config = {
            "mass_1": {
                "sampler": {
                    "function": "gwsim_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
                    "arguments": {
                        "alpha_1": 1.72,
                        "alpha_2": 4.51,
                        "transition": 35.6,
                        "minimum": 5.06,
                        "maximum": 300.0,
                        "mean_1": 9.76,
                        "sigma_1": 0.649,
                        "mean_2": 32.8,
                        "sigma_2": 3.92,
                        "taper_range": 4.32,
                        "lambda_0": 0.361,
                        "lambda_1": 0.586,
                    },
                },
            },
            "mass_ratio": {
                "sampler": {
                    "function": "gwsim_pop.samplers.planck_tapered_conditional_ratio_power_law",
                    "arguments": {"denominator": "@mass_1"},
                },
            },
        }

        simulator = GraphSimulator(config=config, n_samples=20)
        result = simulator()

        assert result.shape[0] == 20  # noqa: PLR2004  # n_samples
        assert result.shape[1] == DEFAULT_N_PARAMETERS  # mass_1 and mass_ratio
        assert "mass_1" in simulator.parameter_names
        assert "mass_ratio" in simulator.parameter_names

    def test_output_shape_validation(self) -> None:
        """Test that output shape matches expected dimensions."""
        config = {
            "mass_1": {
                "sampler": {
                    "function": "gwsim_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
                    "arguments": {
                        "alpha_1": 1.72,
                        "alpha_2": 4.51,
                        "transition": 35.6,
                        "minimum": 5.06,
                        "maximum": 300.0,
                        "mean_1": 9.76,
                        "sigma_1": 0.649,
                        "mean_2": 32.8,
                        "sigma_2": 3.92,
                        "taper_range": 4.32,
                        "lambda_0": 0.361,
                        "lambda_1": 0.586,
                    },
                },
            },
        }

        n_samples = 75
        simulator = GraphSimulator(config=config, n_samples=n_samples)
        result = simulator()

        assert result.ndim == 2  # noqa: PLR2004
        assert result.shape[0] == n_samples
        assert result.shape[1] == len(simulator.parameter_names)

    def test_reset_with_custom_seed(self) -> None:
        """Test that reset preserves custom seed behavior."""
        config = {
            "mass_1": {
                "sampler": {
                    "function": "gwsim_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
                    "arguments": {
                        "alpha_1": 1.72,
                        "alpha_2": 4.51,
                        "transition": 35.6,
                        "minimum": 5.06,
                        "maximum": 300.0,
                        "mean_1": 9.76,
                        "sigma_1": 0.649,
                        "mean_2": 32.8,
                        "sigma_2": 3.92,
                        "taper_range": 4.32,
                        "lambda_0": 0.361,
                        "lambda_1": 0.586,
                    },
                },
            },
        }

        simulator = GraphSimulator(config=config, seed=123)
        result1 = simulator()
        simulator.reset()
        result2 = simulator()

        # Results should be identical after reset with same seed
        assert jnp.allclose(result1, result2)

    def test_simulator_callable_with_args(self) -> None:
        """Test that simulator can be called with positional arguments."""
        config = {
            "mass_1": {
                "sampler": {
                    "function": "gwsim_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
                    "arguments": {
                        "alpha_1": 1.72,
                        "alpha_2": 4.51,
                        "transition": 35.6,
                        "minimum": 5.06,
                        "maximum": 300.0,
                        "mean_1": 9.76,
                        "sigma_1": 0.649,
                        "mean_2": 32.8,
                        "sigma_2": 3.92,
                        "taper_range": 4.32,
                        "lambda_0": 0.361,
                        "lambda_1": 0.586,
                    },
                },
            },
        }

        simulator = GraphSimulator(config=config)
        # simulator() should work without arguments
        result = simulator()
        assert result is not None
        assert result.shape[0] == DEFAULT_N_SAMPLES  # Default n_samples

    def test_parameter_names_from_config(self) -> None:
        """Test that parameter_names matches config keys."""
        config = {
            "mass_1": {
                "sampler": {
                    "function": "gwsim_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
                    "arguments": {
                        "alpha_1": 1.72,
                        "alpha_2": 4.51,
                        "transition": 35.6,
                        "minimum": 5.06,
                        "maximum": 300.0,
                        "mean_1": 9.76,
                        "sigma_1": 0.649,
                        "mean_2": 32.8,
                        "sigma_2": 3.92,
                        "taper_range": 4.32,
                        "lambda_0": 0.361,
                        "lambda_1": 0.586,
                    },
                },
            },
            "mass_ratio": {
                "sampler": {
                    "function": "gwsim_pop.samplers.planck_tapered_conditional_ratio_power_law",
                    "arguments": {"denominator": "@mass_1"},
                },
            },
            "redshift": {
                "sampler": {
                    "function": "gwsim_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
                    "arguments": {
                        "alpha_1": 1.0,
                        "alpha_2": 2.0,
                        "transition": 10.0,
                        "minimum": 0.01,
                        "maximum": 2.0,
                        "mean_1": 0.5,
                        "sigma_1": 0.1,
                        "mean_2": 1.0,
                        "sigma_2": 0.2,
                        "taper_range": 0.5,
                        "lambda_0": 0.5,
                        "lambda_1": 0.5,
                    },
                },
            },
        }

        simulator = GraphSimulator(config=config)
        assert set(simulator.parameter_names) == set(config.keys())
        assert len(simulator.parameter_names) == DEFAULT_N_PARAMETERS_MULTI

    def test_source_type_is_constant(self) -> None:
        """Test that source_type always returns 'bbh'."""
        config = {
            "mass_1": {
                "sampler": {
                    "function": "gwsim_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
                    "arguments": {
                        "alpha_1": 1.72,
                        "alpha_2": 4.51,
                        "transition": 35.6,
                        "minimum": 5.06,
                        "maximum": 300.0,
                        "mean_1": 9.76,
                        "sigma_1": 0.649,
                        "mean_2": 32.8,
                        "sigma_2": 3.92,
                        "taper_range": 4.32,
                        "lambda_0": 0.361,
                        "lambda_1": 0.586,
                    },
                },
            },
        }

        simulator = GraphSimulator(config=config)
        assert simulator.source_type == "bbh"

        # Test that it's consistent across multiple calls
        assert simulator.source_type == "bbh"
        assert simulator.source_type == "bbh"

    def test_graph_nodes_match_parameters(self) -> None:
        """Test that graph nodes match parameter names."""
        config = {
            "mass_1": {
                "sampler": {
                    "function": "gwsim_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
                    "arguments": {
                        "alpha_1": 1.72,
                        "alpha_2": 4.51,
                        "transition": 35.6,
                        "minimum": 5.06,
                        "maximum": 300.0,
                        "mean_1": 9.76,
                        "sigma_1": 0.649,
                        "mean_2": 32.8,
                        "sigma_2": 3.92,
                        "taper_range": 4.32,
                        "lambda_0": 0.361,
                        "lambda_1": 0.586,
                    },
                },
            },
            "mass_ratio": {
                "sampler": {
                    "function": "gwsim_pop.samplers.planck_tapered_conditional_ratio_power_law",
                    "arguments": {"denominator": "@mass_1"},
                },
            },
        }

        simulator = GraphSimulator(config=config)
        graph_nodes = list(simulator._graph.nodes())

        # All parameter names should be in graph nodes
        for param_name in simulator.parameter_names:
            assert param_name in graph_nodes

        # All graph nodes should be parameter names
        for node in graph_nodes:
            assert node in simulator.parameter_names

    def test_sampled_values_after_simulation(self) -> None:
        """Test that sampled_values is populated after simulation."""
        config = {
            "mass_1": {
                "sampler": {
                    "function": "gwsim_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
                    "arguments": {
                        "alpha_1": 1.72,
                        "alpha_2": 4.51,
                        "transition": 35.6,
                        "minimum": 5.06,
                        "maximum": 300.0,
                        "mean_1": 9.76,
                        "sigma_1": 0.649,
                        "mean_2": 32.8,
                        "sigma_2": 3.92,
                        "taper_range": 4.32,
                        "lambda_0": 0.361,
                        "lambda_1": 0.586,
                    },
                },
            },
            "mass_ratio": {
                "sampler": {
                    "function": "gwsim_pop.samplers.planck_tapered_conditional_ratio_power_law",
                    "arguments": {"denominator": "@mass_1"},
                },
            },
        }

        simulator = GraphSimulator(config=config, n_samples=30)
        _ = simulator()

        # Check that sampled_values is populated
        assert len(simulator._sampled_values) == 2  # noqa: PLR2004
        assert "mass_1" in simulator._sampled_values
        assert "mass_ratio" in simulator._sampled_values

        # Check that values are JAX arrays
        assert hasattr(simulator._sampled_values["mass_1"], "shape")
        assert hasattr(simulator._sampled_values["mass_ratio"], "shape")

        # Check shapes
        assert simulator._sampled_values["mass_1"].shape[0] == 30  # noqa: PLR2004
        assert simulator._sampled_values["mass_ratio"].shape[0] == 30  # noqa: PLR2004

    def test_dependency_not_sampled_raises_error(self) -> None:
        """Test that missing dependency raises ValueError."""
        config = {
            "mass_ratio": {
                "sampler": {
                    "function": "gwsim_pop.samplers.planck_tapered_conditional_ratio_power_law",
                    "arguments": {"denominator": "@mass_1"},  # mass_1 not defined
                },
            },
        }

        simulator = GraphSimulator(config=config, n_samples=10)
        with pytest.raises(ValueError, match="Dependency 'mass_1' not sampled yet"):
            _ = simulator()

    def test_sampler_with_explicit_key(self) -> None:
        """Test that sampler works with explicit key in arguments."""
        config = {
            "mass_1": {
                "sampler": {
                    "function": "gwsim_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
                    "arguments": {
                        "alpha_1": 1.72,
                        "alpha_2": 4.51,
                        "transition": 35.6,
                        "minimum": 5.06,
                        "maximum": 300.0,
                        "mean_1": 9.76,
                        "sigma_1": 0.649,
                        "mean_2": 32.8,
                        "sigma_2": 3.92,
                        "taper_range": 4.32,
                        "lambda_0": 0.361,
                        "lambda_1": 0.586,
                        "key": jax.random.key(42),  # Explicit key
                    },
                },
            },
        }

        simulator = GraphSimulator(config=config, n_samples=10)
        result = simulator()
        assert result.shape[0] == 0

    def test_sampler_with_explicit_n_samples(self) -> None:
        """Test that sampler works with explicit n_samples in arguments."""
        config = {
            "mass_1": {
                "sampler": {
                    "function": "gwsim_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
                    "arguments": {
                        "alpha_1": 1.72,
                        "alpha_2": 4.51,
                        "transition": 35.6,
                        "minimum": 5.06,
                        "maximum": 300.0,
                        "mean_1": 9.76,
                        "sigma_1": 0.649,
                        "mean_2": 32.8,
                        "sigma_2": 3.92,
                        "taper_range": 4.32,
                        "lambda_0": 0.361,
                        "lambda_1": 0.586,
                        "n_samples": 15,  # Explicit n_samples
                    },
                },
            },
        }

        simulator = GraphSimulator(config=config, n_samples=100)  # Different default
        result = simulator()
        assert result.shape[0] == 15  # Should use explicit value  # noqa: PLR2004

    def test_transform_execution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that transforms resolve dependencies and execute correctly."""
        config = {
            "mass_1": {
                "sampler": {
                    "function": "gwsim_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
                    "arguments": {
                        "alpha_1": 1.72,
                        "alpha_2": 4.51,
                        "transition": 35.6,
                        "minimum": 5.06,
                        "maximum": 300.0,
                        "mean_1": 9.76,
                        "sigma_1": 0.649,
                        "mean_2": 32.8,
                        "sigma_2": 3.92,
                        "taper_range": 4.32,
                        "lambda_0": 0.361,
                        "lambda_1": 0.586,
                    },
                },
            },
        }

        def fake_import_from_string(object_path: str, default_module: str | None = None):
            assert object_path == "double_values"
            assert default_module == "gwsim_pop.transforms"

            def fake_transform(*, values: jnp.ndarray, scale: int) -> jnp.ndarray:
                return values * scale

            return fake_transform

        simulator = GraphSimulator(config=config, n_samples=10)
        simulator._sampled_values["mass_1"] = jnp.array([1.0, 2.0, 3.0])
        monkeypatch.setattr(graph_module, "import_from_string", fake_import_from_string)

        result = simulator._execute_transform(
            {
                "function": "double_values",
                "arguments": {"values": "@mass_1", "scale": 2},
            }
        )

        assert jnp.array_equal(result, jnp.array([2.0, 4.0, 6.0]))

    def test_transform_execution_missing_dependency(self) -> None:
        """Test that transforms raise when a dependency has not been sampled."""
        config = {
            "mass_1": {
                "sampler": {
                    "function": "gwsim_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
                    "arguments": {
                        "alpha_1": 1.72,
                        "alpha_2": 4.51,
                        "transition": 35.6,
                        "minimum": 5.06,
                        "maximum": 300.0,
                        "mean_1": 9.76,
                        "sigma_1": 0.649,
                        "mean_2": 32.8,
                        "sigma_2": 3.92,
                        "taper_range": 4.32,
                        "lambda_0": 0.361,
                        "lambda_1": 0.586,
                    },
                },
            },
        }

        simulator = GraphSimulator(config=config, n_samples=20)

        with pytest.raises(ValueError, match="Dependency 'mass_1' not available for transform"):
            simulator._execute_transform(
                {
                    "function": "double_values",
                    "arguments": {"values": "@mass_1"},
                }
            )

    def test_simulate_with_transform(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test simulation with both sampler and transform."""
        config = {
            "mass_1": {
                "sampler": {
                    "function": "gwsim_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
                    "arguments": {
                        "alpha_1": 1.72,
                        "alpha_2": 4.51,
                        "transition": 35.6,
                        "minimum": 5.06,
                        "maximum": 300.0,
                        "mean_1": 9.76,
                        "sigma_1": 0.649,
                        "mean_2": 32.8,
                        "sigma_2": 3.92,
                        "taper_range": 4.32,
                        "lambda_0": 0.361,
                        "lambda_1": 0.586,
                    },
                },
            },
            "mass_1_transformed": {
                "transform": {
                    "function": "offset_values",
                    "arguments": {"values": "@mass_1"},
                },
                "intermediate": True,
            },
            "mass_2": {
                "sampler": {
                    "function": "gwsim_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
                    "arguments": {
                        "alpha_1": 1.72,
                        "alpha_2": 4.51,
                        "transition": 35.6,
                        "minimum": 5.06,
                        "maximum": 300.0,
                        "mean_1": 9.76,
                        "sigma_1": 0.649,
                        "mean_2": 32.8,
                        "sigma_2": 3.92,
                        "taper_range": 4.32,
                        "lambda_0": 0.361,
                        "lambda_1": 0.586,
                    },
                },
            },
        }

        real_import_from_string = graph_module.import_from_string

        def fake_import_from_string(object_path: str, default_module: str | None = None):
            if object_path == "offset_values":
                assert default_module == "gwsim_pop.transforms"

                def fake_transform(*, values: jnp.ndarray) -> jnp.ndarray:
                    return values + 1

                return fake_transform

            return real_import_from_string(object_path=object_path, default_module=default_module)

        simulator = GraphSimulator(config=config, n_samples=20)
        monkeypatch.setattr(graph_module, "import_from_string", fake_import_from_string)
        monkeypatch.setattr(
            simulator,
            "_get_ordered_parameters",
            lambda: ["mass_1", "mass_1_transformed", "mass_2"],
        )
        result = simulator()

        assert result.shape == (20, 3)
        assert jnp.allclose(result[:, 1], result[:, 0] + 1)

    def test_reset_without_rng_manager(self) -> None:
        """Test that reset works even without _rng_manager attribute."""
        config = {
            "mass_1": {
                "sampler": {
                    "function": "gwsim_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
                    "arguments": {
                        "alpha_1": 1.72,
                        "alpha_2": 4.51,
                        "transition": 35.6,
                        "minimum": 5.06,
                        "maximum": 300.0,
                        "mean_1": 9.76,
                        "sigma_1": 0.649,
                        "mean_2": 32.8,
                        "sigma_2": 3.92,
                        "taper_range": 4.32,
                        "lambda_0": 0.361,
                        "lambda_1": 0.586,
                    },
                },
            },
        }

        simulator = GraphSimulator(config=config, n_samples=10)
        _ = simulator()

        # Manually remove _rng_manager to test reset without it
        if hasattr(simulator, "_rng_manager"):
            delattr(simulator, "_rng_manager")

        # Reset should not raise error
        simulator.reset()
        assert len(simulator._sampled_values) == 0
