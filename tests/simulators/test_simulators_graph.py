"""Tests for GraphSimulator."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import pytest
import yaml

import gwmock_pop.simulators.graph as graph_module
from gwmock_pop.graph.validation import ConfigValidationError
from gwmock_pop.protocols import GWPopSimulator
from gwmock_pop.simulators.graph import GraphSimulator

# Constants for test data
DEFAULT_N_SAMPLES = 100
DEFAULT_N_PARAMETERS = 2
DEFAULT_N_PARAMETERS_MULTI = 3


def _result_n_samples(result: dict[str, jnp.ndarray]) -> int:
    """Return sample count from mapping output."""
    return len(next(iter(result.values()))) if result else 0


class TestGraphSimulator:
    """Test suite for GraphSimulator class."""

    def test_init_with_config(self) -> None:
        """Test initialization with a simple config."""
        config = {
            "mass_1": {
                "sampler": {
                    "function": "gwmock_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
                    "arguments": {
                        "n_samples": 100,
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
                    "function": "gwmock_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
                    "arguments": {
                        "n_samples": 100,
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
                    "function": "gwmock_pop.samplers.planck_tapered_conditional_ratio_power_law",
                    "arguments": {
                        "denominator": "@mass_1",
                        "beta": 1.5,
                        "numerator_minimum": 0.5,
                        "taper_range": 0.5,
                        "minimum": 0.1,
                        "maximum": 1.0,
                        "n_grids": 100,
                    },
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
                    "function": "gwmock_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
                    "arguments": {
                        "n_samples": 100,
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
                    "function": "gwmock_pop.samplers.planck_tapered_conditional_ratio_power_law",
                    "arguments": {
                        "denominator": "@mass_1",
                        "beta": 1.5,
                        "numerator_minimum": 0.5,
                        "taper_range": 0.5,
                        "minimum": 0.1,
                        "maximum": 1.0,
                        "n_grids": 100,
                    },
                },
            },
        }

        simulator = GraphSimulator(config=config)
        graph_nodes = list(simulator._graph.nodes())
        assert "mass_1" in graph_nodes
        assert "mass_ratio" in graph_nodes

    def test_graph_simulator_simulate_raises_without_source_type(self) -> None:
        """Test that simulate() raises ValueError when source_type was not set."""
        config = {
            "mass_1": {
                "sampler": {
                    "function": "gwmock_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
                    "arguments": {
                        "n_samples": 100,
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
        with pytest.raises(ValueError, match="source_type must be set before calling simulate"):
            simulator.simulate(10)

    def test_source_type_can_be_configured(self) -> None:
        """Test that source_type can be provided by the caller."""
        config = {
            "mass_1": {
                "sampler": {
                    "function": "gwmock_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
                    "arguments": {
                        "n_samples": 100,
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

        simulator = GraphSimulator(config=config, source_type="generic-source")
        assert simulator.source_type == "generic-source"

    def test_source_type_rejects_empty_value(self) -> None:
        """Test that source_type rejects empty values."""
        config = {
            "mass_1": {
                "sampler": {
                    "function": "gwmock_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
                    "arguments": {
                        "n_samples": 100,
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

        with pytest.raises(ValueError, match="source_type must be a non-empty string"):
            GraphSimulator(config=config, source_type=" ")

    def test_reset(self) -> None:
        """Test that reset clears sampled values."""
        config = {
            "mass_1": {
                "sampler": {
                    "function": "gwmock_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
                    "arguments": {
                        "n_samples": 100,
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

        simulator = GraphSimulator(config=config, source_type="population")
        _ = simulator()

        assert len(simulator._sampled_values) > 0

        simulator.reset()

        assert len(simulator._sampled_values) == 0

    def test_from_config_file_yaml(self) -> None:
        """Test loading config from YAML file."""
        config = {
            "mass_1": {
                "sampler": {
                    "function": "gwmock_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
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
function = "gwmock_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks"

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
                        "function": "gwmock_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
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

    def test_from_config_file_missing_required_key_raises_config_validation_error(self) -> None:
        """Missing schema-required keys surface as ``ConfigValidationError``."""
        config = {
            "parameters": {
                "bad_node": {
                    "transform": {
                        "arguments": {
                            "value": 1.0,
                        },
                    },
                },
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            temp_path = f.name

        try:
            with pytest.raises(ConfigValidationError, match=r"transform\.function"):
                GraphSimulator.from_config_file(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_from_config_file_unknown_transform_suggests_close_match(self) -> None:
        """Unknown transform names include a close-match suggestion."""
        config = {
            "parameters": {
                "source": {
                    "sampler": {
                        "function": "uniform_comoving_volume_distance",
                        "arguments": {
                            "d_max": 1000.0,
                        },
                    },
                },
                "bad_node": {
                    "transform": {
                        # typos:off
                        "function": "multply",
                        # typos:on
                        "arguments": {
                            "left": "@source",
                            "right": "@source",
                        },
                    },
                },
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            temp_path = f.name

        try:
            with pytest.raises(ConfigValidationError, match="Did you mean 'multiply'\\?"):
                GraphSimulator.from_config_file(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_init_with_custom_n_samples(self) -> None:
        """Test initialization with custom n_samples passed to sampler arguments."""
        config = {
            "mass_1": {
                "sampler": {
                    "function": "gwmock_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
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
                        "n_samples": 50,
                    },
                },
            },
        }

        simulator = GraphSimulator(config=config, source_type="population")
        assert simulator is not None

        result = simulator()
        assert _result_n_samples(result) == 50
        assert set(result.keys()) == set(simulator.parameter_names)

    def test_simulate_injects_n_samples_when_not_in_sampler_arguments(self) -> None:
        """Test that simulate(n_samples=...) injects the sample count into samplers."""
        config = {
            "mass_1": {
                "sampler": {
                    "function": "gwmock_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
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

        simulator = GraphSimulator(config=config, source_type="population")
        result = simulator.simulate(n_samples=12)

        assert _result_n_samples(result) == 12
        assert set(result.keys()) == {"mass_1"}

    def test_simulate_with_multiple_parameters(self) -> None:
        """Test simulation with multiple parameters and dependencies."""
        config = {
            "mass_1": {
                "sampler": {
                    "function": "gwmock_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
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
                        "n_samples": 20,
                    },
                },
            },
            "mass_ratio": {
                "sampler": {
                    "function": "gwmock_pop.samplers.planck_tapered_conditional_ratio_power_law",
                    "arguments": {
                        "denominator": "@mass_1",
                        "beta": 1.5,
                        "numerator_minimum": 0.5,
                        "taper_range": 0.5,
                        "minimum": 0.1,
                        "maximum": 1.0,
                        "n_grids": 100,
                        "n_samples": 20,
                    },
                },
            },
        }

        simulator = GraphSimulator(config=config, source_type="population")
        result = simulator()

        assert _result_n_samples(result) == 20  # n_samples
        assert len(result) == DEFAULT_N_PARAMETERS  # mass_1 and mass_ratio
        assert "mass_1" in simulator.parameter_names
        assert "mass_ratio" in simulator.parameter_names

    def test_output_shape_validation(self) -> None:
        """Test that output shape matches expected dimensions."""
        config = {
            "mass_1": {
                "sampler": {
                    "function": "gwmock_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
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
                        "n_samples": 75,
                    },
                },
            },
        }

        simulator = GraphSimulator(config=config, source_type="population")
        result = simulator()

        assert set(result.keys()) == set(simulator.parameter_names)
        assert all(value.ndim == 1 for value in result.values())
        assert _result_n_samples(result) == 75

    def test_reset_with_custom_seed(self) -> None:
        """Test that reset preserves custom seed behavior."""
        config = {
            "mass_1": {
                "sampler": {
                    "function": "gwmock_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
                    "arguments": {
                        "n_samples": 100,
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

        simulator = GraphSimulator(config=config, seed=123, source_type="population")
        result1 = simulator()
        simulator.reset()
        result2 = simulator()

        # Results should be identical after reset with same seed
        for name in simulator.parameter_names:
            assert jnp.allclose(result1[name], result2[name])

    def test_simulator_callable_without_args(self) -> None:
        """Test that simulator can be called with positional arguments."""
        config = {
            "mass_1": {
                "sampler": {
                    "function": "gwmock_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
                    "arguments": {
                        "n_samples": 100,
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

        simulator = GraphSimulator(config=config, source_type="population")
        # simulator() should work without arguments
        result = simulator()
        assert result is not None
        assert _result_n_samples(result) == DEFAULT_N_SAMPLES  # Default n_samples

    def test_parameter_names_from_config(self) -> None:
        """Test that parameter_names matches config keys."""
        config = {
            "mass_1": {
                "sampler": {
                    "function": "gwmock_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
                    "arguments": {
                        "n_samples": 100,
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
                    "function": "gwmock_pop.samplers.planck_tapered_conditional_ratio_power_law",
                    "arguments": {
                        "denominator": "@mass_1",
                        "beta": 1.5,
                        "numerator_minimum": 0.5,
                        "taper_range": 0.5,
                        "minimum": 0.1,
                        "maximum": 1.0,
                        "n_grids": 100,
                    },
                },
            },
            "redshift": {
                "sampler": {
                    "function": "gwmock_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
                    "arguments": {
                        "n_samples": 100,
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

    def test_excluded_parameters_are_not_returned(self) -> None:
        """Test that parameters marked as excluded are omitted from the output."""
        config = {
            "mass_1": {
                "sampler": {
                    "function": "gwmock_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
                    "arguments": {
                        "n_samples": 10,
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
            "hidden_mass": {
                "exclude": True,
                "sampler": {
                    "function": "gwmock_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
                    "arguments": {
                        "n_samples": 10,
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

        simulator = GraphSimulator(config=config, source_type="population")
        result = simulator()

        assert simulator.parameter_names == ["mass_1"]
        assert _result_n_samples(result) == 10
        assert set(result.keys()) == {"mass_1"}

    def test_source_type_is_stable(self) -> None:
        """Test that source_type remains stable across property access."""
        config = {
            "mass_1": {
                "sampler": {
                    "function": "gwmock_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
                    "arguments": {
                        "n_samples": 100,
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

        simulator = GraphSimulator(config=config, source_type="population")
        assert simulator.source_type == "population"

        # Test that it's consistent across multiple calls
        assert simulator.source_type == "population"
        assert simulator.source_type == "population"

    def test_graph_nodes_match_parameters(self) -> None:
        """Test that graph nodes match parameter names."""
        config = {
            "mass_1": {
                "sampler": {
                    "function": "gwmock_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
                    "arguments": {
                        "n_samples": 100,
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
                    "function": "gwmock_pop.samplers.planck_tapered_conditional_ratio_power_law",
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
                    "function": "gwmock_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
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
                        "n_samples": 30,
                    },
                },
            },
            "mass_ratio": {
                "sampler": {
                    "function": "gwmock_pop.samplers.planck_tapered_conditional_ratio_power_law",
                    "arguments": {
                        "denominator": "@mass_1",
                        "beta": 1.5,
                        "numerator_minimum": 0.5,
                        "taper_range": 0.5,
                        "minimum": 0.1,
                        "maximum": 1.0,
                        "n_grids": 100,
                        "n_samples": 30,
                    },
                },
            },
        }

        simulator = GraphSimulator(config=config, source_type="population")
        _ = simulator()

        # Check that sampled_values is populated
        assert len(simulator._sampled_values) == 2
        assert "mass_1" in simulator._sampled_values
        assert "mass_ratio" in simulator._sampled_values

        # Check that values are JAX arrays
        assert hasattr(simulator._sampled_values["mass_1"], "shape")
        assert hasattr(simulator._sampled_values["mass_ratio"], "shape")

        # Check shapes
        assert simulator._sampled_values["mass_1"].shape[0] == 30
        assert simulator._sampled_values["mass_ratio"].shape[0] == 30

    def test_dependency_not_sampled_raises_error(self) -> None:
        """Test that missing dependency raises ValueError."""
        config = {
            "mass_ratio": {
                "sampler": {
                    "function": "gwmock_pop.samplers.planck_tapered_conditional_ratio_power_law",
                    "arguments": {
                        "denominator": "@mass_1",  # mass_1 not defined
                        "beta": 1.5,
                        "numerator_minimum": 0.5,
                        "taper_range": 0.5,
                        "minimum": 0.1,
                        "maximum": 1.0,
                        "n_grids": 100,
                    },
                },
            },
        }

        simulator = GraphSimulator(config=config, source_type="population")
        with pytest.raises(ValueError, match="Undefined parameter dependencies"):
            _ = simulator()

    def test_sampler_with_explicit_key(self) -> None:
        """Test that sampler works with explicit key in arguments."""
        config = {
            "mass_1": {
                "sampler": {
                    "function": "gwmock_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
                    "arguments": {
                        "n_samples": 100,
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

        simulator = GraphSimulator(config=config, source_type="population")
        result = simulator()
        assert _result_n_samples(result) == 100  # default n_samples

    def test_sampler_with_explicit_n_samples(self) -> None:
        """Test that sampler works with explicit n_samples in arguments."""
        config = {
            "mass_1": {
                "sampler": {
                    "function": "gwmock_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
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

        simulator = GraphSimulator(config=config, source_type="population")
        result = simulator()
        assert _result_n_samples(result) == 15  # Should use explicit value

    def test_transform_execution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that transforms resolve dependencies and execute correctly."""
        config = {
            "mass_1": {
                "sampler": {
                    "function": "gwmock_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
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
            assert default_module == "gwmock_pop.transforms"

            def fake_transform(*, values: jnp.ndarray, scale: int, key: jax.Array) -> jnp.ndarray:
                del key
                return values * scale

            return fake_transform

        simulator = GraphSimulator(config=config)
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
                    "function": "gwmock_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
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

        with pytest.raises(ValueError, match="Dependency 'mass_1' not available for transform"):
            simulator._execute_transform(
                {
                    "function": "double_values",
                    "arguments": {"values": "@mass_1"},
                }
            )

    def test_transform_execution_injects_rng_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that stochastic transforms receive an injected RNG key."""
        simulator = GraphSimulator(config={"mass_1": {"transform": {"function": "stochastic_transform"}}}, seed=1234)
        simulator._sampled_values["mass_1"] = jnp.array([1.0, 2.0, 3.0])

        def fake_import_from_string(object_path: str, default_module: str | None = None):
            assert object_path == "stochastic_transform"
            assert default_module == "gwmock_pop.transforms"

            def fake_transform(*, reference: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
                assert reference.shape == (3,)
                assert str(key.dtype).startswith("key<")
                return jax.random.uniform(key, shape=reference.shape)

            return fake_transform

        monkeypatch.setattr(graph_module, "import_from_string", fake_import_from_string)

        result = simulator._execute_transform(
            {
                "function": "stochastic_transform",
                "arguments": {"reference": "@mass_1"},
            }
        )

        assert result.shape == (3,)

    def test_simulate_with_transform(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test simulation with both sampler and transform."""
        config = {
            "mass_1": {
                "sampler": {
                    "function": "gwmock_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
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
                        "n_samples": 20,
                    },
                },
            },
            "mass_1_transformed": {
                "transform": {
                    "function": "offset_values",
                    "arguments": {"values": "@mass_1"},
                },
            },
            "mass_2": {
                "sampler": {
                    "function": "gwmock_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
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
                        "n_samples": 20,
                    },
                },
            },
        }

        real_import_from_string = graph_module.import_from_string

        def fake_import_from_string(object_path: str, default_module: str | None = None):
            if object_path == "offset_values":
                assert default_module == "gwmock_pop.transforms"

                def fake_transform(*, values: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
                    del key
                    return values + 1

                return fake_transform

            return real_import_from_string(object_path=object_path, default_module=default_module)

        simulator = GraphSimulator(config=config, source_type="population")
        monkeypatch.setattr(graph_module, "import_from_string", fake_import_from_string)
        monkeypatch.setattr(
            simulator,
            "_get_ordered_parameters",
            lambda: ["mass_1", "mass_1_transformed", "mass_2"],
        )
        result = simulator()

        assert _result_n_samples(result) == 20
        assert set(result.keys()) == {"mass_1", "mass_1_transformed", "mass_2"}
        assert jnp.allclose(result["mass_1_transformed"], result["mass_1"] + 1)

    def test_reset_without_rng_manager(self) -> None:
        """Test that reset works even without _rng_manager attribute."""
        config = {
            "mass_1": {
                "sampler": {
                    "function": "gwmock_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
                    "arguments": {
                        "n_samples": 100,
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

        simulator = GraphSimulator(config=config, source_type="population")
        _ = simulator()

        # Manually remove _rng_manager to test reset without it
        if hasattr(simulator, "_rng_manager"):
            delattr(simulator, "_rng_manager")

        # Reset should not raise error
        simulator.reset()
        assert len(simulator._sampled_values) == 0

    def test_execute_sampler_with_none_arguments(self) -> None:
        """Test that _execute_sampler handles None arguments correctly."""
        config = {
            "mass_1": {
                "sampler": {
                    "function": "gwmock_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
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

        # Test with None arguments - should be converted to empty dict
        # This tests the code path where arguments is None and gets converted to {}
        # We use a minimal sampler that doesn't require arguments
        def minimal_sampler(**kwargs: Any) -> jnp.ndarray:
            return jnp.array([1.0, 2.0, 3.0])

        # Monkeypatch import_from_string to return our minimal sampler
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(graph_module, "import_from_string", lambda *args, **kwargs: minimal_sampler)

        result = simulator._execute_sampler(
            "minimal_sampler",
            None,  # type: ignore[arg-type]
        )
        assert result is not None
        assert isinstance(result, jnp.ndarray)
        monkeypatch.undo()

    def test_execute_sampler_with_invalid_arguments_type(self) -> None:
        """Test that _execute_sampler raises error for invalid arguments type."""
        config = {
            "mass_1": {
                "sampler": {
                    "function": "gwmock_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
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
        # Test with string arguments - should raise ValueError
        with pytest.raises(ValueError, match="Sampler arguments must be a mapping"):
            simulator._execute_sampler(
                "gwmock_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
                "invalid_arguments",  # type: ignore[arg-type]
            )

    def test_execute_sampler_with_unresolved_dependency(self) -> None:
        """Test that _execute_sampler raises error for unresolved dependency."""
        config = {
            "mass_1": {
                "sampler": {
                    "function": "gwmock_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
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
        # Test with dependency that hasn't been sampled
        with pytest.raises(ValueError, match="Dependency 'nonexistent' not sampled yet"):
            simulator._execute_sampler(
                "gwmock_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
                {"alpha_1": "@nonexistent"},
            )

    def test_execute_transform_with_string_expression(self) -> None:
        """Test that _execute_transform raises error for string transform."""
        config = {
            "mass_1": {
                "sampler": {
                    "function": "gwmock_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
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
        # Test with string transform - should raise ValueError
        with pytest.raises(ValueError, match="String transform expressions are not supported"):
            simulator._execute_transform("@mass_1 * 2")  # type: ignore[arg-type]

    def test_execute_transform_with_none_arguments(self) -> None:
        """Test that _execute_transform handles None arguments correctly."""
        config = {
            "mass_1": {
                "sampler": {
                    "function": "gwmock_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
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

        # Mock transform that expects arguments
        def fake_transform(**kwargs: Any) -> jnp.ndarray:
            return jnp.array([1.0, 2.0, 3.0])

        # Monkeypatch import_from_string to return our fake transform
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(graph_module, "import_from_string", lambda *args, **kwargs: fake_transform)

        # Test with None arguments - should be converted to empty dict
        result = simulator._execute_transform(
            {
                "function": "fake_transform",
                "arguments": None,
            }
        )
        assert result is not None
        assert isinstance(result, jnp.ndarray)
        monkeypatch.undo()

    def test_execute_transform_with_invalid_arguments_type(self) -> None:
        """Test that _execute_transform raises error for invalid arguments type."""
        config = {
            "mass_1": {
                "sampler": {
                    "function": "gwmock_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
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
        # Test with string arguments - should raise ValueError
        with pytest.raises(ValueError, match="Only mapping-style transform arguments are currently supported"):
            simulator._execute_transform(
                {
                    "function": "some_transform",
                    "arguments": "invalid_arguments",
                }
            )

    def test_from_config_file_with_invalid_root(self) -> None:
        """Test that config file with non-dict root raises ValueError."""
        # Create a YAML file with a list at root
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("- item1\n- item2\n")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Config file must contain a mapping"):
                GraphSimulator.from_config_file(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_from_config_file_with_list_root(self) -> None:
        """Test that config file with list root raises ValueError."""
        # Create a YAML file with a list at root (TOML cannot have list at root)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("- 1\n- 2\n- 3\n")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Config file must contain a mapping"):
                GraphSimulator.from_config_file(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_coerce_output_column_scalar_raises_error(self) -> None:
        """Test that _coerce_output_column raises error for scalar output."""
        config = {
            "mass_1": {
                "sampler": {
                    "function": "gwmock_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
                    "arguments": {
                        "n_samples": 100,
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
        # Test with scalar value - should raise ValueError
        with pytest.raises(ValueError, match="produced a scalar output"):
            simulator._coerce_output_column("mass_1", jnp.array(5.0), None)

    def test_coerce_output_column_sample_count_mismatch(self) -> None:
        """Test that _coerce_output_column raises error for sample count mismatch."""
        config = {
            "mass_1": {
                "sampler": {
                    "function": "gwmock_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
                    "arguments": {
                        "n_samples": 100,
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
        # Test with wrong sample count - should raise ValueError
        wrong_count_array = jnp.array([1.0, 2.0, 3.0])  # 3 samples instead of 100
        with pytest.raises(ValueError, match="produced 3 samples, expected 100"):
            simulator._coerce_output_column("mass_1", wrong_count_array, 100)

    def test_simulate_with_transform_branch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that transform branch is executed correctly in _simulate_impl."""

        # Create a simple transform that accepts any kwargs
        def dummy_transform(**kwargs: Any) -> jnp.ndarray:
            return jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

        # Monkeypatch to use our dummy transform
        monkeypatch.setattr(
            graph_module,
            "import_from_string",
            lambda object_path, default_module=None: dummy_transform,
        )

        config = {
            "mass_1": {
                "sampler": {
                    "function": "gwmock_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
                    "arguments": {
                        "n_samples": 10,
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
                    "function": "dummy_transform",
                    "arguments": {"values": "@mass_1"},
                },
            },
        }

        simulator = GraphSimulator(config=config, source_type="population")
        # Just verify that the transform branch is reachable
        # The actual output is not important, we just want to ensure
        # the transform branch in _simulate_impl is executed
        result = simulator()

        # Should have 2 parameters: mass_1 and mass_1_transformed
        expected_n_parameters = 2
        assert len(result) == expected_n_parameters
        assert "mass_1" in simulator.parameter_names
        assert "mass_1_transformed" in simulator.parameter_names

    def test_bbh_config_with_spin_transform_nodes_is_protocol_conformant(self) -> None:
        """Test a BBH graph config that composes the stochastic spin transform nodes."""
        n_samples = 128
        config = {
            "detector_frame_mass_1": {
                "sampler": {
                    "function": "gwmock_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
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
                }
            },
            "mass_ratio": {
                "sampler": {
                    "function": "gwmock_pop.samplers.planck_tapered_conditional_ratio_power_law",
                    "arguments": {
                        "denominator": "@detector_frame_mass_1",
                        "beta": 1.5,
                        "numerator_minimum": 0.5,
                        "taper_range": 0.5,
                        "minimum": 0.1,
                        "maximum": 1.0,
                        "n_grids": 128,
                    },
                },
                "intermediate": True,
            },
            "detector_frame_mass_2": {
                "transform": {
                    "function": "multiply",
                    "arguments": {
                        "left": "@detector_frame_mass_1",
                        "right": "@mass_ratio",
                    },
                }
            },
            "spin_1_magnitude": {
                "transform": {
                    "function": "beta_spin_magnitude",
                    "arguments": {
                        "reference": "@detector_frame_mass_1",
                        "alpha": 2.0,
                        "beta": 5.0,
                    },
                },
                "intermediate": True,
            },
            "spin_2_magnitude": {
                "transform": {
                    "function": "beta_spin_magnitude",
                    "arguments": {
                        "reference": "@detector_frame_mass_1",
                        "alpha": 2.5,
                        "beta": 4.0,
                    },
                },
                "intermediate": True,
            },
            "chi_eff": {
                "transform": {
                    "function": "gaussian_chi_eff",
                    "arguments": {
                        "reference": "@detector_frame_mass_1",
                        "mean": 0.05,
                        "sigma": 0.15,
                    },
                },
                "intermediate": True,
            },
            "spin_1x": {
                "transform": {
                    "function": "constant_like",
                    "arguments": {"reference": "@detector_frame_mass_1", "value": 0.0},
                }
            },
            "spin_1y": {
                "transform": {
                    "function": "constant_like",
                    "arguments": {"reference": "@detector_frame_mass_1", "value": 0.0},
                }
            },
            "spin_1z": {
                "transform": {
                    "function": "gaussian_chi_eff",
                    "arguments": {
                        "chi_eff": "@chi_eff",
                        "component": "primary",
                        "mass_1": "@detector_frame_mass_1",
                        "mass_2": "@detector_frame_mass_2",
                        "spin_magnitude_1": "@spin_1_magnitude",
                        "spin_magnitude_2": "@spin_2_magnitude",
                    },
                }
            },
            "spin_2x": {
                "transform": {
                    "function": "constant_like",
                    "arguments": {"reference": "@detector_frame_mass_1", "value": 0.0},
                }
            },
            "spin_2y": {
                "transform": {
                    "function": "constant_like",
                    "arguments": {"reference": "@detector_frame_mass_1", "value": 0.0},
                }
            },
            "spin_2z": {
                "transform": {
                    "function": "gaussian_chi_eff",
                    "arguments": {
                        "chi_eff": "@chi_eff",
                        "component": "secondary",
                        "mass_1": "@detector_frame_mass_1",
                        "mass_2": "@detector_frame_mass_2",
                        "spin_magnitude_1": "@spin_1_magnitude",
                        "spin_magnitude_2": "@spin_2_magnitude",
                    },
                }
            },
            "eccentricity": {
                "transform": {
                    "function": "constant_like",
                    "arguments": {"reference": "@detector_frame_mass_1", "value": 0.0},
                }
            },
            "distance": {
                "sampler": {
                    "function": "uniform_comoving_volume_distance",
                    "arguments": {
                        "d_max": 5_000.0,
                    },
                }
            },
            "coa_phase": {
                "transform": {
                    "function": "constant_like",
                    "arguments": {"reference": "@detector_frame_mass_1", "value": 0.0},
                }
            },
            "inclination": {
                "transform": {
                    "function": "isotropic_spin_orientation",
                    "arguments": {"reference": "@detector_frame_mass_1"},
                }
            },
            "theta_jn": {
                "transform": {
                    "function": "isotropic_spin_orientation",
                    "arguments": {"reference": "@detector_frame_mass_1"},
                }
            },
            "long_asc_node": {
                "transform": {
                    "function": "constant_like",
                    "arguments": {"reference": "@detector_frame_mass_1", "value": 0.0},
                }
            },
            "mean_per_ano": {
                "transform": {
                    "function": "constant_like",
                    "arguments": {"reference": "@detector_frame_mass_1", "value": 0.0},
                }
            },
            "coa_time": {
                "transform": {
                    "function": "constant_like",
                    "arguments": {"reference": "@detector_frame_mass_1", "value": 0.0},
                }
            },
            "right_ascension": {
                "transform": {
                    "function": "constant_like",
                    "arguments": {"reference": "@detector_frame_mass_1", "value": 0.0},
                }
            },
            "declination": {
                "transform": {
                    "function": "constant_like",
                    "arguments": {"reference": "@detector_frame_mass_1", "value": 0.0},
                }
            },
            "polarization_angle": {
                "transform": {
                    "function": "constant_like",
                    "arguments": {"reference": "@detector_frame_mass_1", "value": 0.0},
                }
            },
            "redshift": {
                "transform": {
                    "function": "luminosity_distance_to_redshift",
                    "arguments": {"luminosity_distance": "@distance"},
                }
            },
            "f_ref": {
                "transform": {
                    "function": "constant_like",
                    "arguments": {"reference": "@detector_frame_mass_1", "value": 20.0},
                }
            },
        }

        simulator: GWPopSimulator = GraphSimulator(config=config, source_type="bbh", seed=123)

        assert isinstance(simulator, GWPopSimulator)
        assert simulator.source_type == "bbh"

        result = simulator.simulate(n_samples=n_samples)

        assert list(result.keys()) == list(simulator.parameter_names)
        assert all(values.shape == (n_samples,) for values in result.values())
        assert all(bool(jnp.all(jnp.isfinite(values))) for values in result.values())

    def test_simulate_empty_parameter_names_raises_error(self) -> None:
        """Test that _simulate_impl raises error when no output parameters defined."""
        config = {
            "intermediate_param": {
                "sampler": {
                    "function": "gwmock_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
                    "arguments": {
                        "n_samples": 100,
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
                "exclude": True,  # Exclude from output
                "intermediate": True,  # Mark as intermediate
            },
        }

        simulator = GraphSimulator(config=config, source_type="population")
        # parameter_names will be empty because all params are excluded/intermediate
        assert simulator.parameter_names == []

        # Simulate should raise ValueError when no output parameters
        with pytest.raises(ValueError, match="does not define any output parameters"):
            simulator()

    def test_from_config_file_with_unsupported_suffix_other_error(self) -> None:
        """Test that from_config_file re-raises non-matching errors."""
        # Create a file with unsupported suffix that triggers a different error
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            # Write content that will cause a different ValueError
            f.write("invalid yaml content {{")
            temp_path = f.name

        try:
            # This should raise a ValueError with unsupported suffix message
            with pytest.raises(ValueError, match="Suffix of config_path"):
                GraphSimulator.from_config_file(temp_path)
        finally:
            Path(temp_path).unlink()


# ---------------------------------------------------------------------------
# GWTC-4 BBH mass model parameters (BrokenPowerLawTwoPeaks from GWTC-4.0)
# ---------------------------------------------------------------------------
_GWTC4_MASS1_CONFIG: dict[str, Any] = {
    "mass_1": {
        "sampler": {
            "function": "gwmock_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks",
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
                "key": jax.random.key(42),
            },
        },
    },
}


class TestGraphSimulatorProtocolConformance:
    """Protocol conformance and regression tests for GraphSimulator."""

    def test_isinstance_gwpopsimulator(self) -> None:
        """GraphSimulator satisfies the GWPopSimulator runtime-checkable protocol."""
        sim = GraphSimulator(config=_GWTC4_MASS1_CONFIG, source_type="bbh")
        assert isinstance(sim, GWPopSimulator)

    def test_simulate_returns_mapping_with_correct_keys(self) -> None:
        """simulate() returns a Mapping whose keys equal parameter_names."""
        sim = GraphSimulator(config=_GWTC4_MASS1_CONFIG, source_type="bbh")
        result = sim.simulate(n_samples=50)
        assert set(result.keys()) == set(sim.parameter_names)

    def test_simulate_returns_1d_arrays(self) -> None:
        """Each value in the simulate() result is a 1-D array."""
        sim = GraphSimulator(config=_GWTC4_MASS1_CONFIG, source_type="bbh")
        result = sim.simulate(n_samples=50)
        assert all(v.ndim == 1 for v in result.values())

    def test_simulate_respects_n_samples(self) -> None:
        """Each array in the result has the requested number of samples."""
        n = 75
        sim = GraphSimulator(config=_GWTC4_MASS1_CONFIG, source_type="bbh")
        result = sim.simulate(n_samples=n)
        assert all(v.shape[0] == n for v in result.values())

    def test_gwtc4_mass1_mean_regression(self) -> None:
        """Mass-1 mean from GWTC-4 config matches golden value to 3 significant figures.

        Golden values computed with key=jax.random.key(42) and n_samples=1000:
            mean ≈ 18.14, std ≈ 11.28
        """
        sim = GraphSimulator(config=_GWTC4_MASS1_CONFIG, source_type="bbh")
        result = sim.simulate(n_samples=1000)
        mass_1 = result["mass_1"]
        assert abs(float(jnp.mean(mass_1)) - 18.141) / 18.141 < 5e-3, (
            f"mass_1 mean regression failed: got {float(jnp.mean(mass_1)):.4f}, expected ~18.141"
        )
        assert abs(float(jnp.std(mass_1)) - 11.284) / 11.284 < 5e-3, (
            f"mass_1 std regression failed: got {float(jnp.std(mass_1)):.4f}, expected ~11.284"
        )
