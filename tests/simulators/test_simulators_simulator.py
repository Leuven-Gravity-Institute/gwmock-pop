"""Tests for Simulator."""

from __future__ import annotations

from pathlib import Path

import h5py
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from gwsim_pop.simulators.simulator import Simulator


class ConcreteSimulator(Simulator):
    """Concrete implementation of Simulator for testing."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the concrete simulator."""
        super().__init__(*args, **kwargs)
        self._parameter_names = ["mass", "spin", "redshift"]
        self._source_type = "BBH"

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

        assert result.shape == (10, 3)
        assert jnp.all(result == 1.0)
        assert simulator._last_data is result

    def test_call(self, simulator: ConcreteSimulator) -> None:
        """Test __call__ method."""
        result = simulator()

        assert result.shape == (10, 3)
        assert jnp.all(result == 1.0)

    @pytest.mark.parametrize("file_format", ["npy", "csv", "hdf5"])
    def test_save(self, simulator: ConcreteSimulator, tmp_path: Path, file_format: str) -> None:
        """Test save method with different formats."""
        data = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        simulator._last_data = data

        output_path = tmp_path / f"test.{file_format}"
        simulator.save(output_path)

        assert output_path.exists()

        if file_format == "npy":
            loaded = jnp.load(output_path)
            assert jnp.allclose(loaded, data)
        elif file_format == "csv":
            loaded = np.loadtxt(output_path, delimiter=",")
            assert np.allclose(loaded, data)
        elif file_format == "hdf5":
            with h5py.File(output_path, "r") as f:
                loaded = jnp.array(f["data"])
            assert jnp.allclose(loaded, data)

    def test_save_with_explicit_data(self, simulator: ConcreteSimulator, tmp_path: Path) -> None:
        """Test save method with explicit data argument."""
        data = jnp.array([[7.0, 8.0, 9.0]])
        output_path = tmp_path / "test.npy"
        simulator.save(output_path, data=data)

        loaded = jnp.load(output_path)
        assert jnp.allclose(loaded, data)

    def test_save_explicit_file_format(self, simulator: ConcreteSimulator, tmp_path: Path) -> None:
        """Test save method with explicit file_format (bypasses extension inference)."""
        data = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        simulator._last_data = data
        # Use a path without extension, but specify format explicitly
        output_path = tmp_path / "testfile"
        simulator.save(output_path, file_format="csv")

        loaded = np.loadtxt(output_path, delimiter=",")
        assert np.allclose(loaded, data)

    def test_save_no_last_data(self, simulator: ConcreteSimulator, tmp_path: Path) -> None:
        """Test save method raises error when no data available."""
        simulator._last_data = None
        output_path = tmp_path / "test.npy"

        with pytest.raises(ValueError, match="No data provided and no last simulated data available"):
            simulator.save(output_path)

    def test_save_unsupported_format(self, simulator: ConcreteSimulator, tmp_path: Path) -> None:
        """Test save method raises error for unsupported format."""
        data = jnp.array([[1.0, 2.0, 3.0]])
        simulator._last_data = data
        output_path = tmp_path / "test.txt"

        with pytest.raises(ValueError, match="Unsupported format: txt"):
            simulator.save(output_path)

    def test_load(self, simulator: ConcreteSimulator, tmp_path: Path) -> None:
        """Test load method with different formats."""
        original_data = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        for file_format in ["npy", "csv", "hdf5"]:
            output_path = tmp_path / f"test_load.{file_format}"

            if file_format == "npy":
                jnp.save(output_path, original_data)
            elif file_format == "csv":
                np.savetxt(output_path, original_data, delimiter=",")
            elif file_format == "hdf5":
                with h5py.File(output_path, "w") as f:
                    f.create_dataset("data", data=original_data)

            loaded = simulator.load(output_path)
            assert jnp.allclose(loaded, original_data)

    def test_load_unsupported_format(self, simulator: ConcreteSimulator, tmp_path: Path) -> None:
        """Test load method raises error for unsupported format."""
        output_path = tmp_path / "test.txt"
        output_path.touch()

        with pytest.raises(ValueError, match="Unsupported format: txt"):
            simulator.load(output_path)

    def test_validate_output_valid(self, simulator: ConcreteSimulator) -> None:
        """Test _validate_output with valid 2D array."""
        valid_data = jnp.ones((10, 3))
        simulator._validate_output(valid_data)

    def test_validate_output_wrong_ndim(self, simulator: ConcreteSimulator) -> None:
        """Test _validate_output raises error for wrong number of dimensions."""
        invalid_data = jnp.ones((10,))

        with pytest.raises(ValueError, match="Expected 2D array, got 1D array"):
            simulator._validate_output(invalid_data)

    def test_validate_output_wrong_n_parameters(self, simulator: ConcreteSimulator) -> None:
        """Test _validate_output raises error for wrong number of parameters."""
        invalid_data = jnp.ones((10, 2))

        with pytest.raises(ValueError, match="Expected 3 parameters, got 2"):
            simulator._validate_output(invalid_data)

    def test_save_hdf5_static(self, tmp_path: Path) -> None:
        """Test _save_hdf5 static method."""
        data = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        path = tmp_path / "test.h5"

        Simulator._save_hdf5(path, data)

        with h5py.File(path, "r") as f:
            loaded = jnp.array(f["data"])
            assert jnp.allclose(loaded, data)

    def test_load_hdf5_static(self, tmp_path: Path) -> None:
        """Test _load_hdf5 static method."""
        data = jnp.array([[5.0, 6.0], [7.0, 8.0]])
        path = tmp_path / "test2.h5"

        with h5py.File(path, "w") as f:
            f.create_dataset("data", data=data)

        loaded = Simulator._load_hdf5(path)
        assert jnp.allclose(loaded, data)

    def test_load_hdf5_invalid_dataset(self, tmp_path: Path) -> None:
        """Test _load_hdf5 raises error for non-dataset object."""
        path = tmp_path / "test3.h5"

        with h5py.File(path, "w") as f:
            f.create_group("data")

        with pytest.raises(TypeError, match=r'f\["data"\] is not a h5py\.Dataset instance'):
            Simulator._load_hdf5(path)
