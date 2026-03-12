"""Tests for Simulator."""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from jax import Array

from gwsim_pop.simulators.simulator import Simulator


class ConcreteSimulator(Simulator):
    """Concrete implementation of Simulator for testing."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the concrete simulator."""
        super().__init__()
        self._parameter_names = ["mass", "spin", "redshift"]
        self._source_type = "BBH"
        super().__init__(*args, **kwargs)

    @property
    def parameter_names(self) -> list[str]:
        """Get the names of the parameters."""
        return self._parameter_names

    @property
    def source_type(self) -> str:
        """Get the source type."""
        return self._source_type

    def _simulate_impl(self, *args, **kwargs) -> Array:
        """Implement simulation for subclass."""
        # Create dummy data: (n_samples, n_parameters)
        data = jnp.ones((10, 3))
        return data


class TestSimulator:
    """Test suite for Simulator base class."""

    @pytest.fixture
    def simulator(self) -> ConcreteSimulator:
        """Create a concrete simulator instance for testing."""
        return ConcreteSimulator()

    def test_register_node_no_dependencies(self, simulator: ConcreteSimulator) -> None:
        """Test register_node method with no dependencies."""

        def dummy_func() -> jnp.ndarray:
            return jnp.array([[1.0, 2.0, 3.0]])

        simulator.register_node("test_node", dummy_func)

        assert "test_node" in simulator._node_funcs
        assert "test_node" in simulator._node_depends
        assert simulator._node_funcs["test_node"] == dummy_func

    def test_register_node_with_dependencies(self, simulator: ConcreteSimulator) -> None:
        """Test register_node method with dependencies."""

        def dummy_func() -> jnp.ndarray:
            return jnp.array([[1.0, 2.0, 3.0]])

        simulator.register_node("test_node", dummy_func, depends_on=["mass"])

        assert "test_node" in simulator._node_funcs
        assert "test_node" in simulator._node_depends
        assert simulator._node_funcs["test_node"] == dummy_func
        assert simulator._node_depends["test_node"] == ["mass"]

    def test_node_decorator(self, simulator: ConcreteSimulator) -> None:
        """Test node decorator."""

        def mass() -> jnp.ndarray:
            return jnp.array([[1.0, 2.0, 3.0]])

        _decorated_mass = simulator.node()(mass)

        assert "mass" in simulator._node_funcs
        assert "mass" in simulator._node_depends
        assert simulator._node_funcs["mass"] is mass

    def test_node_decorator_with_dependencies(self, simulator: ConcreteSimulator) -> None:
        """Test node decorator with dependencies."""

        def spin() -> jnp.ndarray:
            return jnp.array([[1.0, 2.0, 3.0]])

        _decorated_spin = simulator.node(depends_on=["mass"])(spin)

        assert "spin" in simulator._node_funcs
        assert "spin" in simulator._node_depends
        assert simulator._node_funcs["spin"] is spin
        assert simulator._node_depends["spin"] == ["mass"]

    def test_node_decorator_with_multiple_dependencies(self, simulator: ConcreteSimulator) -> None:
        """Test node decorator with multiple dependencies."""

        def spin() -> jnp.ndarray:
            return jnp.array([[1.0, 2.0, 3.0]])

        _decorated_spin = simulator.node(depends_on=["mass", "redshift"])(spin)

        assert "spin" in simulator._node_funcs
        assert "spin" in simulator._node_depends
        assert simulator._node_funcs["spin"] is spin
        assert simulator._node_depends["spin"] == ["mass", "redshift"]

    def test_node_decorator_with_no_dependencies(self, simulator: ConcreteSimulator) -> None:
        """Test node decorator with no dependencies."""

        def mass() -> jnp.ndarray:
            return jnp.array([[1.0, 2.0, 3.0]])

        _decorated_mass = simulator.node()(mass)

        assert "mass" in simulator._node_funcs
        assert "mass" in simulator._node_depends
        assert simulator._node_funcs["mass"] is mass

    def test_node_decorator_with_empty_dependencies_list(self, simulator: ConcreteSimulator) -> None:
        """Test node decorator with empty dependencies list."""

        def mass() -> jnp.ndarray:
            return jnp.array([[1.0, 2.0, 3.0]])

        _decorated_mass = simulator.node(depends_on=[])(mass)

        assert "mass" in simulator._node_funcs
        assert "mass" in simulator._node_depends
        assert simulator._node_funcs["mass"] is mass
