"""Tests for Simulator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from gwmock_pop import read_population_catalogue
from gwmock_pop.provenance import read_provenance
from gwmock_pop.simulators.simulator import Simulator


class ConcreteSimulator(Simulator):
    """Concrete implementation of Simulator for testing."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the concrete simulator."""
        super().__init__(*args, **kwargs)
        self._parameter_names = ["mass", "spin", "redshift"]

    @property
    def parameter_names(self) -> list[str]:
        """Get the names of the parameters."""
        return self._parameter_names

    def _simulate_impl(self, *args: Any, **kwargs: Any) -> dict[str, Array]:
        """Implement simulation for subclass."""
        # Create dummy data with 1D per-parameter arrays.
        return {name: jnp.ones((10,)) for name in self.parameter_names}


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

    def test_node_decorator_wrapper_calls_original_function(self, simulator: ConcreteSimulator) -> None:
        """Test that calling the decorated function invokes the original function."""
        call_count = 0

        def counter() -> jnp.ndarray:
            nonlocal call_count
            call_count += 1
            return jnp.array([[1.0, 2.0, 3.0]])

        decorated = simulator.node()(counter)
        result = decorated()

        assert call_count == 1
        assert result.shape == (1, 3)

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

    def test_register_node_replaces_existing_node(self, simulator: ConcreteSimulator) -> None:
        """Test register_node removes edges from existing node when re-registering."""

        def func1() -> jnp.ndarray:
            return jnp.array([[1.0, 2.0, 3.0]])

        def func2() -> jnp.ndarray:
            return jnp.array([[4.0, 5.0, 6.0]])

        # Register initial node with dependency
        simulator.register_node("test_node", func1, depends_on=["mass"])
        assert list(simulator.graph.predecessors("test_node")) == ["mass"]

        # Re-register node without dependency (should remove existing edge)
        simulator.register_node("test_node", func2, depends_on=[])
        assert list(simulator.graph.predecessors("test_node")) == []
        assert simulator._node_funcs["test_node"] is func2

    def test_simulate(self, simulator: ConcreteSimulator) -> None:
        """Test simulate method."""
        result = simulator.simulate()

        assert set(result.keys()) == set(simulator.parameter_names)
        assert all(value.shape == (10,) for value in result.values())
        assert all(jnp.all(value == 1.0) for value in result.values())
        assert simulator._last_data is result

    def test_call(self, simulator: ConcreteSimulator) -> None:
        """Test __call__ method."""
        result = simulator()

        assert set(result.keys()) == set(simulator.parameter_names)
        assert all(value.shape == (10,) for value in result.values())
        assert all(jnp.all(value == 1.0) for value in result.values())

    @pytest.mark.parametrize("suffix", ["csv", "hdf5", "h5"])
    def test_save_catalogue_round_trips_through_the_package_reader(
        self, simulator: ConcreteSimulator, tmp_path: Path, suffix: str
    ) -> None:
        """A saved catalogue is one this package's own reader accepts."""
        expected = simulator.simulate()
        output_path = tmp_path / f"population.{suffix}"

        simulator.save_catalogue(output_path)

        catalogue = read_population_catalogue(output_path)
        assert list(catalogue) == simulator.parameter_names
        for name, values in catalogue.items():
            assert np.allclose(values, np.asarray(expected[name]), atol=0.0)

    def test_save_catalogue_writes_columns_in_parameter_order(
        self, simulator: ConcreteSimulator, tmp_path: Path
    ) -> None:
        """Column order follows ``parameter_names``, whatever order the data arrives in."""
        shuffled = {name: jnp.full((4,), index + 1.0) for index, name in enumerate(reversed(simulator.parameter_names))}
        output_path = tmp_path / "population.hdf5"

        simulator.save_catalogue(output_path, data=shuffled)

        assert list(read_population_catalogue(output_path)) == simulator.parameter_names

    def test_save_catalogue_carries_a_provenance_record(self, simulator: ConcreteSimulator, tmp_path: Path) -> None:
        """A saved catalogue says which code produced it and with which seed."""
        simulator.simulate()
        output_path = tmp_path / "population.hdf5"

        simulator.save_catalogue(output_path)

        record = read_provenance(output_path)
        assert record is not None
        assert record["catalogue"]["parameter_names"] == simulator.parameter_names
        assert record["catalogue"]["n_samples"] == 10
        assert record["origin"]["kind"] == "simulator"
        assert record["origin"]["engine"]["component"].endswith("ConcreteSimulator")

    def test_save_catalogue_stores_a_supplied_record_unchanged(
        self, simulator: ConcreteSimulator, tmp_path: Path
    ) -> None:
        """A caller who has already built a record gets that one stored."""
        simulator.simulate()
        output_path = tmp_path / "population.hdf5"
        record = {"schema_version": "1.0", "marker": "supplied"}

        simulator.save_catalogue(output_path, provenance=record)

        assert read_provenance(output_path) == record

    def test_save_catalogue_compresses_when_asked(self, simulator: ConcreteSimulator, tmp_path: Path) -> None:
        """The compression filter reaches the dataset."""
        simulator.simulate()
        output_path = tmp_path / "population.hdf5"

        simulator.save_catalogue(output_path, compression="gzip")

        with h5py.File(output_path, "r") as handle:
            assert handle["data"].compression == "gzip"

    def test_save_catalogue_without_data(self, simulator: ConcreteSimulator, tmp_path: Path) -> None:
        """Saving before simulating is an error, not an empty file."""
        with pytest.raises(ValueError, match="No data provided and no last simulated data available"):
            simulator.save_catalogue(tmp_path / "population.hdf5")

    def test_save_catalogue_rejects_unnamed_data(self, simulator: ConcreteSimulator, tmp_path: Path) -> None:
        """A bare array has no column names, so it is not a catalogue."""
        with pytest.raises(TypeError, match="mapping of parameter names"):
            simulator.save_catalogue(tmp_path / "population.hdf5", data=jnp.ones((3, 3)))

    def test_save_catalogue_rejects_an_unsupported_suffix(self, simulator: ConcreteSimulator, tmp_path: Path) -> None:
        """Only the catalogue formats this package can read back are written."""
        simulator.simulate()

        with pytest.raises(ValueError, match="Unsupported population-file format"):
            simulator.save_catalogue(tmp_path / "population.npy")

    def test_validate_output_valid(self, simulator: ConcreteSimulator) -> None:
        """Test _validate_output with valid mapping output."""
        valid_data = {name: jnp.ones((10,)) for name in simulator.parameter_names}
        simulator._validate_output(valid_data)

    def test_validate_output_wrong_ndim(self, simulator: ConcreteSimulator) -> None:
        """Test _validate_output raises error for non-1D parameter arrays."""
        invalid_data = {
            "mass": jnp.ones((10, 1)),
            "spin": jnp.ones((10,)),
            "redshift": jnp.ones((10,)),
        }

        with pytest.raises(ValueError, match="Expected 1D array for parameter 'mass', got 2D array"):
            simulator._validate_output(invalid_data)

    def test_validate_output_wrong_n_parameters(self, simulator: ConcreteSimulator) -> None:
        """Test _validate_output raises error for wrong parameter keys."""
        invalid_data = {
            "mass": jnp.ones((10,)),
            "spin": jnp.ones((10,)),
        }

        with pytest.raises(ValueError, match="Expected keys"):
            simulator._validate_output(invalid_data)
