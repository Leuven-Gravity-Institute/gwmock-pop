"""Base class of the simulators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import cast

import h5py
import jax.numpy as jnp
import numpy as np
from jax import Array


class Simulator(ABC):
    """A simulator base class for generating gravitational wave source populations."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the instance.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        self._last_data: Array | None = None

    @property
    @abstractmethod
    def parameter_names(self) -> list[str]:
        """Get the names of the parameters.

        Returns:
            List of parameter names.

        """

    @property
    @abstractmethod
    def source_type(self) -> str:
        """Get the source type.

        Returns:
            Source type string.

        """

    @abstractmethod
    def _simulate_impl(self, *args, **kwargs) -> Array:
        """Implement simulation for subclass.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            2D array of shape (n_samples, n_parameters).

        """

    def simulate(self, *args, **kwargs) -> Array:
        """Simulate a population of sources.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            2D array of shape (n_samples, n_parameters).

        """
        result = self._simulate_impl(*args, **kwargs)
        self._validate_output(result)
        self._last_data = result
        return result

    def __call__(self, *args, **kwargs) -> Array:
        """Call simulate() with n_samples.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            2D array of shape (n_samples, n_parameters).

        """
        return self.simulate(*args, **kwargs)

    def save(
        self,
        output_path: str | Path,
        file_format: str | None = None,
        data: Array | None = None,
    ) -> None:
        """Save the simulated data to a file.

        Args:
            output_path: Path to save the file.
            file_format: File format (npy, csv, hdf5). If None, infers from extension.
            data: Data to save. If None, uses last simulated data.

        """
        output_path = Path(output_path)

        if file_format is None:
            file_format = output_path.suffix.lstrip(".")

        if data is None:
            if self._last_data is None:
                raise ValueError("No data provided and no last simulated data available.")
            data = cast(Array, self._last_data)

        if file_format == "npy":
            jnp.save(output_path, data)
        elif file_format == "csv":
            np.savetxt(output_path, data, delimiter=",")
        elif file_format == "hdf5":
            self._save_hdf5(output_path, data)
        else:
            raise ValueError(f"Unsupported format: {file_format}")

    def load(self, input_path: str | Path) -> Array:
        """Load data from a file.

        Args:
            input_path: Path to load the file from.

        Returns:
            Loaded data as numpy array.

        """
        input_path = Path(input_path)

        suffix = input_path.suffix.lstrip(".")

        if suffix == "npy":
            return jnp.load(input_path)
        elif suffix == "csv":
            return jnp.array(np.loadtxt(input_path, delimiter=","))
        elif suffix == "hdf5":
            return self._load_hdf5(input_path)
        else:
            raise ValueError(f"Unsupported format: {suffix}")

    @staticmethod
    def _save_hdf5(path: Path, data: Array) -> None:
        """Save data to HDF5 format.

        Args:
            path: Path to save the file.
            data: Data to save.

        """
        with h5py.File(path, "w") as f:
            f.create_dataset("data", data=data)

    @staticmethod
    def _load_hdf5(path: Path) -> Array:
        """Load data from HDF5 format.

        Args:
            path: Path to load the file from.

        Returns:
            Loaded data as numpy array.

        """
        with h5py.File(path, "r") as f:
            data_obj = f["data"]
            if not isinstance(data_obj, h5py.Dataset):
                raise TypeError('f["data"] is not a h5py.Dataset instance.')
            return jnp.array(data_obj[()])

    def _validate_output(self, array: Array) -> None:
        """Validate the output of simulate().

        Args:
            array: Array to validate.

        Raises:
            ValueError: If array shape is invalid.

        """
        expected_ndim = 2
        if array.ndim != expected_ndim:
            raise ValueError(f"Expected 2D array, got {array.ndim}D array")

        expected_n_parameters = len(self.parameter_names)
        if array.shape[1] != expected_n_parameters:
            raise ValueError(f"Expected {expected_n_parameters} parameters, got {array.shape[1]}")
