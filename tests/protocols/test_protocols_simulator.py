"""Protocol conformance tests for GWPopSimulator.

Two test classes verify complementary properties of ``GWPopSimulator``:

``TestGWPopSimulatorStructure``
    Plain pytest tests that verify the structural contract: a complete
    implementation satisfies ``isinstance(obj, GWPopSimulator)``, and any
    class missing one of the three required members (``parameter_names``,
    ``source_type``, ``simulate``) fails the check.

``TestGWPopSimulatorContracts``
    Hypothesis-driven tests parametrized over both minimal fixture classes.
    They exercise the runtime invariants that ``gwmock`` depends on: key
    completeness, array shape, ``source_type`` validity, ``parameter_names``
    stability, and value types.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import jax.numpy as jnp
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from jax import Array

from gwmock_pop.protocols import GWPopSimulator

# ---------------------------------------------------------------------------
# Minimal conformant implementations (plain classes, not pytest fixtures)
# ---------------------------------------------------------------------------


class _MinimalBBHSimulator:
    """Minimal BBH-like simulator satisfying ``GWPopSimulator``."""

    parameter_names: ClassVar[list[str]] = ["mass_1", "mass_2", "distance", "redshift"]
    source_type = "bbh"

    def simulate(self, n_samples: int, **kwargs: Any) -> Mapping[str, Array]:
        """Return constant 1-D arrays of length ``n_samples``."""
        return {k: jnp.ones(n_samples) for k in self.parameter_names}


class _MinimalStochasticSimulator:
    """Minimal stochastic-background simulator satisfying ``GWPopSimulator``."""

    parameter_names: ClassVar[list[str]] = ["omega_gw", "spectral_index"]
    source_type = "stochastic"

    def simulate(self, n_samples: int, **kwargs: Any) -> Mapping[str, Array]:
        """Return constant 1-D arrays of length ``n_samples``."""
        return {k: jnp.ones(n_samples) for k in self.parameter_names}


# Shared list used by ``TestGWPopSimulatorContracts`` parametrize.
_SIMULATORS = [_MinimalBBHSimulator(), _MinimalStochasticSimulator()]

# ---------------------------------------------------------------------------
# Structural tests
# ---------------------------------------------------------------------------


class TestGWPopSimulatorStructure:
    """Verify that ``GWPopSimulator`` structural membership checks are correct."""

    def test_bbh_impl_satisfies_protocol(self) -> None:
        """Test that a complete BBH implementation satisfies the protocol."""
        assert isinstance(_MinimalBBHSimulator(), GWPopSimulator)

    def test_stochastic_impl_satisfies_protocol(self) -> None:
        """Test that a complete stochastic implementation satisfies the protocol."""
        assert isinstance(_MinimalStochasticSimulator(), GWPopSimulator)

    def test_missing_parameter_names_fails(self) -> None:
        """Test that a class without parameter_names fails the isinstance check."""

        class _NoParameterNames:
            source_type = "bbh"

            def simulate(self, n_samples: int, **kwargs: Any) -> Mapping[str, Array]:
                """Return empty mapping."""
                return {}

        assert not isinstance(_NoParameterNames(), GWPopSimulator)

    def test_missing_source_type_fails(self) -> None:
        """Test that a class without source_type fails the isinstance check."""

        class _NoSourceType:
            parameter_names: ClassVar[list[str]] = ["mass_1"]

            def simulate(self, n_samples: int, **kwargs: Any) -> Mapping[str, Array]:
                """Return empty mapping."""
                return {}

        assert not isinstance(_NoSourceType(), GWPopSimulator)

    def test_missing_simulate_fails(self) -> None:
        """Test that a class without simulate fails the isinstance check."""

        class _NoSimulate:
            parameter_names: ClassVar[list[str]] = ["mass_1"]
            source_type = "bbh"

        assert not isinstance(_NoSimulate(), GWPopSimulator)

    def test_all_three_members_required(self) -> None:
        """Test that a class with all three members satisfies the protocol."""

        class _Complete:
            parameter_names: ClassVar[list[str]] = ["x"]
            source_type = "test"

            def simulate(self, n_samples: int, **kwargs: Any) -> Mapping[str, Array]:
                """Return constant array."""
                return {"x": jnp.ones(n_samples)}

        assert isinstance(_Complete(), GWPopSimulator)


# ---------------------------------------------------------------------------
# Contract invariant tests
# ---------------------------------------------------------------------------


class TestGWPopSimulatorContracts:
    """Hypothesis-driven contract invariants, parametrized over both simulators."""

    @pytest.mark.parametrize("simulator", _SIMULATORS, ids=["bbh", "stochastic"])
    @settings(max_examples=25)
    @given(n_samples=st.integers(min_value=1, max_value=500))
    def test_simulate_keys_match_parameter_names(self, simulator: GWPopSimulator, n_samples: int) -> None:
        """Test that simulate() returns exactly the keys in parameter_names."""
        result = simulator.simulate(n_samples)
        assert set(result.keys()) == set(simulator.parameter_names)

    @pytest.mark.parametrize("simulator", _SIMULATORS, ids=["bbh", "stochastic"])
    @settings(max_examples=25)
    @given(n_samples=st.integers(min_value=1, max_value=500))
    def test_simulate_arrays_have_n_samples_leading_dim(self, simulator: GWPopSimulator, n_samples: int) -> None:
        """Test that every output array has leading dimension equal to n_samples."""
        result = simulator.simulate(n_samples)
        for array in result.values():
            assert array.shape[0] == n_samples

    @pytest.mark.parametrize("simulator", _SIMULATORS, ids=["bbh", "stochastic"])
    @settings(max_examples=20)
    @given(n_samples=st.integers(min_value=1, max_value=500))
    def test_source_type_is_non_empty_string(self, simulator: GWPopSimulator, n_samples: int) -> None:
        """Test that source_type is a non-empty string."""
        assert isinstance(simulator.source_type, str)
        assert len(simulator.source_type) > 0

    @pytest.mark.parametrize("simulator", _SIMULATORS, ids=["bbh", "stochastic"])
    def test_parameter_names_is_stable(self, simulator: GWPopSimulator) -> None:
        """Test that parameter_names returns the same sequence on repeated access."""
        assert list(simulator.parameter_names) == list(simulator.parameter_names)

    @pytest.mark.parametrize("simulator", _SIMULATORS, ids=["bbh", "stochastic"])
    @settings(max_examples=20)
    @given(n_samples=st.integers(min_value=1, max_value=500))
    def test_no_extra_keys_beyond_parameter_names(self, simulator: GWPopSimulator, n_samples: int) -> None:
        """Test that simulate() produces no keys beyond parameter_names."""
        result = simulator.simulate(n_samples)
        assert set(result.keys()) <= set(simulator.parameter_names)

    @pytest.mark.parametrize("simulator", _SIMULATORS, ids=["bbh", "stochastic"])
    @settings(max_examples=20)
    @given(n_samples=st.integers(min_value=1, max_value=500))
    def test_simulate_result_values_are_jax_arrays(self, simulator: GWPopSimulator, n_samples: int) -> None:
        """Test that all values in the simulate() result are jax.Array instances."""
        result = simulator.simulate(n_samples)
        for value in result.values():
            assert isinstance(value, Array)
