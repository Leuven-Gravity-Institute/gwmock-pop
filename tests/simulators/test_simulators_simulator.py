"""Tests for Simulator."""

from __future__ import annotations

import tempfile
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

    def _simulate_impl(self, n_samples: int = 10) -> Array:
        """Implement simulation for subclass."""
        # Create dummy data: (n_samples, n_parameters)
        data = jnp.ones((n_samples, 3))
        return data


class TestSimulator:
    """Test suite for Simulator base class."""

    @pytest.fixture
    def simulator(self) -> ConcreteSimulator:
        """Create a concrete simulator instance for testing."""
        return ConcreteSimulator()

    def test_parameter_names(self, simulator: ConcreteSimulator) -> None:
        """Test that parameter_names returns the correct list."""
        assert simulator.parameter_names == ["mass", "spin", "redshift"]

    def test_source_type(self, simulator: ConcreteSimulator) -> None:
        """Test that source_type returns the correct string."""
        assert simulator.source_type == "BBH"

    def test_simulate_default_samples(self, simulator: ConcreteSimulator) -> None:
        """Test simulate with default number of samples."""
        result = simulator.simulate()

        assert result.shape == (10, 3)
        assert jnp.all(result == 1.0)
        assert isinstance(result, jnp.ndarray)

    def test_simulate_custom_samples(self, simulator: ConcreteSimulator) -> None:
        """Test simulate with custom number of samples."""
        n_samples = 20
        result = simulator.simulate(n_samples)

        assert result.shape == (n_samples, 3)
        assert isinstance(result, jnp.ndarray)

    def test_call_method(self, simulator: ConcreteSimulator) -> None:
        """Test __call__ method delegates to simulate."""
        result = simulator(15)

        assert result.shape == (15, 3)
        assert isinstance(result, jnp.ndarray)

    def test_validate_output_2d_array(self, simulator: ConcreteSimulator) -> None:
        """Test validation passes for 2D array."""
        data = jnp.ones((10, 3))
        simulator._validate_output(data)

    def test_validate_output_1d_array(self, simulator: ConcreteSimulator) -> None:
        """Test validation fails for 1D array."""
        data = jnp.ones(3)

        with pytest.raises(ValueError, match="Expected 2D array, got 1D array"):
            simulator._validate_output(data)

    def test_validate_output_3d_array(self, simulator: ConcreteSimulator) -> None:
        """Test validation fails for 3D array."""
        data = jnp.ones((10, 3, 2))

        with pytest.raises(ValueError, match="Expected 2D array, got 3D array"):
            simulator._validate_output(data)

    def test_save_load_npy(self, simulator: ConcreteSimulator, tmp_path: Path) -> None:
        """Test saving and loading data in npy format."""
        data = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        simulator._last_data = data

        save_path = tmp_path / "test_data.npy"
        simulator.save(save_path)

        loaded_data = simulator.load(save_path)

        assert jnp.array_equal(data, loaded_data)

    def test_save_load_csv(self, simulator: ConcreteSimulator, tmp_path: Path) -> None:
        """Test saving and loading data in csv format."""
        data = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        simulator._last_data = data

        save_path = tmp_path / "test_data.csv"
        simulator.save(save_path)

        loaded_data = simulator.load(save_path)

        assert jnp.array_equal(data, loaded_data)

    def test_save_load_hdf5(self, simulator: ConcreteSimulator, tmp_path: Path) -> None:
        """Test saving and loading data in hdf5 format."""
        data = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        simulator._last_data = data

        save_path = tmp_path / "test_data.hdf5"
        simulator.save(save_path)

        loaded_data = simulator.load(save_path)

        assert jnp.array_equal(data, loaded_data)

    def test_save_with_explicit_format(self, simulator: ConcreteSimulator, tmp_path: Path) -> None:
        """Test saving with explicit format specification."""
        data = jnp.array([[1.0, 2.0, 3.0]])
        simulator._last_data = data

        # Test with npy (use correct extension)
        save_path = tmp_path / "test_data_correct.npy"
        simulator.save(save_path, file_format="npy")

        loaded_data = jnp.load(save_path)
        assert jnp.array_equal(data, loaded_data)

    def test_save_with_custom_data(self, simulator: ConcreteSimulator, tmp_path: Path) -> None:
        """Test saving with custom data provided."""
        custom_data = jnp.array([[7.0, 8.0, 9.0]])
        save_path = tmp_path / "test_custom.npy"

        simulator.save(save_path, data=custom_data)

        loaded_data = jnp.load(save_path)
        assert jnp.array_equal(custom_data, loaded_data)

    def test_save_no_data_and_no_last_data(self, simulator: ConcreteSimulator) -> None:
        """Test save raises error when no data and no last data."""
        simulator._last_data = None

        with pytest.raises(ValueError, match="No data provided and no last simulated data"):
            simulator.save("test.npy")

    def test_save_unsupported_format(self, simulator: ConcreteSimulator, tmp_path: Path) -> None:
        """Test save with unsupported format raises error."""
        data = jnp.ones((2, 3))
        simulator._last_data = data

        save_path = tmp_path / "test_data.txt"

        with pytest.raises(ValueError, match="Unsupported format: txt"):
            simulator.save(save_path)

    def test_load_npy(self, simulator: ConcreteSimulator, tmp_path: Path) -> None:
        """Test loading data from npy file."""
        data = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        save_path = tmp_path / "load_test.npy"
        jnp.save(save_path, data)

        loaded_data = simulator.load(save_path)

        assert jnp.array_equal(data, loaded_data)

    def test_load_csv(self, simulator: ConcreteSimulator, tmp_path: Path) -> None:
        """Test loading data from csv file."""
        data = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        save_path = tmp_path / "load_test.csv"
        np.savetxt(save_path, data, delimiter=",")

        loaded_data = simulator.load(save_path)

        assert jnp.array_equal(data, loaded_data)

    def test_load_hdf5(self, simulator: ConcreteSimulator, tmp_path: Path) -> None:
        """Test loading data from hdf5 file."""
        data = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        save_path = tmp_path / "load_test.hdf5"

        with h5py.File(save_path, "w") as f:
            f.create_dataset("data", data=data)

        loaded_data = simulator.load(save_path)

        assert jnp.array_equal(data, loaded_data)

    def test_load_unsupported_format(self, simulator: ConcreteSimulator, tmp_path: Path) -> None:
        """Test load with unsupported format raises error."""
        save_path = tmp_path / "test.txt"

        with pytest.raises(ValueError, match="Unsupported format: txt"):
            simulator.load(save_path)

    def test_save_hdf5_static_method(self, tmp_path: Path) -> None:
        """Test the static _save_hdf5 method."""
        data = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        save_path = tmp_path / "static_hdf5.hdf5"

        Simulator._save_hdf5(save_path, data)

        with h5py.File(save_path, "r") as f:
            loaded_data = jnp.array(f["data"])

        assert jnp.array_equal(data, loaded_data)

    def test_load_hdf5_static_method(self, tmp_path: Path) -> None:
        """Test the static _load_hdf5 method."""
        data = jnp.array([[5.0, 6.0], [7.0, 8.0]])
        save_path = tmp_path / "static_load_hdf5.hdf5"

        with h5py.File(save_path, "w") as f:
            f.create_dataset("data", data=data)

        loaded_data = Simulator._load_hdf5(save_path)

        assert jnp.array_equal(data, loaded_data)

    def test_load_hdf5_with_dataset(self, tmp_path: Path) -> None:
        """Test loading HDF5 with correct dataset."""
        data = jnp.array([[1.0]])
        save_path = tmp_path / "dataset_test.hdf5"

        with h5py.File(save_path, "w") as f:
            f.create_dataset("data", data=data)

        loaded_data = Simulator._load_hdf5(save_path)

        assert jnp.array_equal(data, loaded_data)

    def test_load_hdf5_missing_dataset(self, tmp_path: Path) -> None:
        """Test loading HDF5 with missing dataset raises KeyError."""
        save_path = tmp_path / "missing_dataset.hdf5"

        with h5py.File(save_path, "w") as f:
            f.create_group("not_data")

        with pytest.raises(KeyError, match="'data'"):
            Simulator._load_hdf5(save_path)

    def test_last_data_attribute_after_simulate(self, simulator: ConcreteSimulator) -> None:
        """Test that _last_data is set after simulate."""
        simulator._last_data = None

        result = simulator.simulate(5)

        assert simulator._last_data is not None
        assert jnp.array_equal(result, simulator._last_data)

    def test_last_data_attribute_after_call(self, simulator: ConcreteSimulator) -> None:
        """Test that _last_data is set after __call__."""
        simulator._last_data = None

        result = simulator(5)

        assert simulator._last_data is not None
        assert jnp.array_equal(result, simulator._last_data)

    def test_load_with_string_path(self, simulator: ConcreteSimulator, tmp_path: Path) -> None:
        """Test loading with string path instead of Path object."""
        data = jnp.array([[1.0, 2.0]])
        save_path = str(tmp_path / "string_path.npy")
        jnp.save(save_path, data)

        loaded_data = simulator.load(save_path)

        assert jnp.array_equal(data, loaded_data)

    def test_save_with_string_path(self, simulator: ConcreteSimulator, tmp_path: Path) -> None:
        """Test saving with string path instead of Path object."""
        data = jnp.array([[1.0, 2.0]])
        simulator._last_data = data

        save_path = str(tmp_path / "string_path_save.npy")
        simulator.save(save_path)

        loaded_data = jnp.load(save_path)
        assert jnp.array_equal(data, loaded_data)

    def test_simulate_returns_correct_shape(self, simulator: ConcreteSimulator) -> None:
        """Test that simulate returns correct shape for various sample sizes."""
        for n_samples in [1, 10, 100]:
            result = simulator.simulate(n_samples)
            assert result.shape == (n_samples, 3)

    def test_inferred_format_from_extension(self, simulator: ConcreteSimulator, tmp_path: Path) -> None:
        """Test that format is inferred from file extension."""
        data = jnp.array([[1.0, 2.0]])
        simulator._last_data = data

        # Test .npy extension
        save_path_npy = tmp_path / "test.npy"
        simulator.save(save_path_npy)
        assert save_path_npy.exists()

        # Test .csv extension
        save_path_csv = tmp_path / "test.csv"
        simulator.save(save_path_csv)
        assert save_path_csv.exists()

        # Test .hdf5 extension
        save_path_hdf5 = tmp_path / "test.hdf5"
        simulator.save(save_path_hdf5)
        assert save_path_hdf5.exists()

    def test_simulate_sets_last_data(self, simulator: ConcreteSimulator) -> None:
        """Test that simulate correctly sets _last_data."""
        # Clear any existing data
        simulator._last_data = None

        # Simulate
        result = simulator.simulate(5)

        # Verify _last_data was set
        assert simulator._last_data is not None
        assert jnp.array_equal(result, simulator._last_data)

    def test_context_manager_usage(self, simulator: ConcreteSimulator) -> None:
        """Test simulator can be used with context manager."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            data = jnp.array([[1.0, 2.0, 3.0]])
            simulator._last_data = data

            # Use context manager
            path = Path(tmp_dir) / "data.npy"
            simulator.save(path)

            loaded = simulator.load(path)
            assert jnp.array_equal(data, loaded)
