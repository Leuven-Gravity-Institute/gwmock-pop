"""Tests for BBHSimulator."""

from __future__ import annotations

import jax.numpy as jnp

from gwsim_pop.simulators.bbh.base import BBHSimulator


class ConcreteBBHSimulator(BBHSimulator):
    """Concrete implementation of BBHSimulator for testing."""

    def __init__(self, *args, **kwargs):
        """Initialize the instance."""
        super().__init__(*args, **kwargs)

    def _simulate_impl(self, *args, **kwargs):
        """Implement simulation (returns dummy data)."""
        return jnp.ones((1, len(self.parameter_names)))


class TestBBHSimulator:
    """Test suite for BBHSimulator base class."""

    def test_parameter_names_returns_correct_list(self) -> None:
        """Test that parameter_names returns the expected list of parameters."""
        simulator = ConcreteBBHSimulator()
        expected_params = [
            "detector_frame_mass_1",
            "detector_frame_mass_2",
            "spin_1x",
            "spin_1y",
            "spin_1z",
            "spin_2x",
            "spin_2y",
            "spin_2z",
            "eccentricity",
            "distance",
            "coa_phase",
            "inclination",
            "theta_jn",
            "long_asc_node",
            "mean_per_ano",
            "coa_time",
            "right_ascension",
            "declination",
            "polarization_angle",
            "redshift",
            "f_ref",
        ]

        assert simulator.parameter_names == expected_params

    def test_parameter_names_is_list(self) -> None:
        """Test that parameter_names returns a list."""
        simulator = ConcreteBBHSimulator()
        assert isinstance(simulator.parameter_names, list)

    def test_parameter_names_is_not_empty(self) -> None:
        """Test that parameter_names is not empty."""
        simulator = ConcreteBBHSimulator()
        assert len(simulator.parameter_names) > 0

    def test_parameter_names_unique(self) -> None:
        """Test that parameter names are unique."""
        simulator = ConcreteBBHSimulator()
        params = simulator.parameter_names
        assert len(params) == len(set(params))

    def test_parameter_names_all_strings(self) -> None:
        """Test that all parameter names are strings."""
        simulator = ConcreteBBHSimulator()
        for param in simulator.parameter_names:
            assert isinstance(param, str)

    def test_parameter_names_has_expected_count(self) -> None:
        """Test that parameter_names has the expected number of parameters."""
        simulator = ConcreteBBHSimulator()
        num_bbh_parameters = 21
        assert len(simulator.parameter_names) == num_bbh_parameters

    def test_source_type_is_bbh(self) -> None:
        """Test that source_type returns 'bbh'."""
        simulator = ConcreteBBHSimulator()
        assert simulator.source_type == "bbh"

    def test_source_type_is_string(self) -> None:
        """Test that source_type returns a string."""
        simulator = ConcreteBBHSimulator()
        assert isinstance(simulator.source_type, str)

    def test_source_type_is_not_empty(self) -> None:
        """Test that source_type is not empty."""
        simulator = ConcreteBBHSimulator()
        assert len(simulator.source_type) > 0

    def test_source_type_lowercase(self) -> None:
        """Test that source_type is lowercase."""
        simulator = ConcreteBBHSimulator()
        assert simulator.source_type == simulator.source_type.lower()

    def test_instantiation(self) -> None:
        """Test that ConcreteBBHSimulator can be instantiated."""
        simulator = ConcreteBBHSimulator()
        assert simulator is not None

    def test_parameter_names_consistent_across_instances(self) -> None:
        """Test that parameter_names is consistent across instances."""
        sim1 = ConcreteBBHSimulator()
        sim2 = ConcreteBBHSimulator()

        assert sim1.parameter_names == sim2.parameter_names

    def test_source_type_consistent_across_instances(self) -> None:
        """Test that source_type is consistent across instances."""
        sim1 = ConcreteBBHSimulator()
        sim2 = ConcreteBBHSimulator()

        assert sim1.source_type == sim2.source_type

    def test_parameter_names_has_mass_parameters(self) -> None:
        """Test that parameter_names includes mass-related parameters."""
        simulator = ConcreteBBHSimulator()
        params = simulator.parameter_names
        assert "detector_frame_mass_1" in params
        assert "detector_frame_mass_2" in params

    def test_parameter_names_has_spin_parameters(self) -> None:
        """Test that parameter_names includes spin-related parameters."""
        simulator = ConcreteBBHSimulator()
        params = simulator.parameter_names
        assert "spin_1x" in params
        assert "spin_1y" in params
        assert "spin_1z" in params
        assert "spin_2x" in params
        assert "spin_2y" in params
        assert "spin_2z" in params

    def test_parameter_names_has_phase_parameters(self) -> None:
        """Test that parameter_names includes phase-related parameters."""
        simulator = ConcreteBBHSimulator()
        params = simulator.parameter_names
        assert "coa_phase" in params
        assert "coa_time" in params

    def test_parameter_names_has_position_parameters(self) -> None:
        """Test that parameter_names includes position-related parameters."""
        simulator = ConcreteBBHSimulator()
        params = simulator.parameter_names
        assert "right_ascension" in params
        assert "declination" in params
        assert "polarization_angle" in params

    def test_parameter_names_has_distance_parameters(self) -> None:
        """Test that parameter_names includes distance-related parameters."""
        simulator = ConcreteBBHSimulator()
        params = simulator.parameter_names
        assert "distance" in params
        assert "redshift" in params

    def test_parameter_names_has_inclination_parameters(self) -> None:
        """Test that parameter_names includes inclination-related parameters."""
        simulator = ConcreteBBHSimulator()
        params = simulator.parameter_names
        assert "inclination" in params
        assert "theta_jn" in params

    def test_parameter_names_has_frequency_parameters(self) -> None:
        """Test that parameter_names includes frequency-related parameters."""
        simulator = ConcreteBBHSimulator()
        params = simulator.parameter_names
        assert "f_ref" in params

    def test_parameter_names_has_eccentricity_parameter(self) -> None:
        """Test that parameter_names includes eccentricity parameter."""
        simulator = ConcreteBBHSimulator()
        params = simulator.parameter_names
        assert "eccentricity" in params

    def test_parameter_names_has_ascending_nodes_parameter(self) -> None:
        """Test that parameter_names includes long_asc_node parameter."""
        simulator = ConcreteBBHSimulator()
        params = simulator.parameter_names
        assert "long_asc_node" in params

    def test_parameter_names_has_periastron_parameter(self) -> None:
        """Test that parameter_names includes mean_per_ano parameter."""
        simulator = ConcreteBBHSimulator()
        params = simulator.parameter_names
        assert "mean_per_ano" in params

    def test_source_type_matches_class(self) -> None:
        """Test that source_type correctly identifies the source type."""
        simulator = ConcreteBBHSimulator()
        assert simulator.source_type == "bbh"

    def test_instantiation_with_kwargs(self) -> None:
        """Test that ConcreteBBHSimulator can be instantiated with kwargs."""
        simulator = ConcreteBBHSimulator(test_kwargs="input")
        assert simulator is not None
