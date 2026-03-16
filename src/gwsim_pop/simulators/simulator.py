"""Base class of the simulators."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any

import h5py
import jax.numpy as jnp
import networkx as nx
import numpy as np
from jax import Array


class Simulator(ABC):
    """Abstract base class for generating simulated populations."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialize the instance.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        self.graph = nx.DiGraph()
        self._last_data: Array | None = None
        self._node_funcs: dict[str, Callable] = {}
        self._node_depends: dict[str, list[str]] = {}

    @property
    @abstractmethod
    def parameter_names(self) -> list[str]:
        """Get the names of the parameters.

        Returns:
            List of parameter names.

        """

    def register_node(self, name: str, func: Callable, depends_on: list[str] | None = None) -> None:
        """Register a node function on this instance.

        Args:
            name: Parameter name.
            func: A function to simulate this parameter.
            depends_on: A list of dependent parameters.
        """
        dependencies = list(depends_on or [])
        if name in self.graph:
            self.graph.remove_edges_from((dep, name) for dep in list(self.graph.predecessors(name)))
        self._node_funcs[name] = func
        self._node_depends[name] = dependencies
        self.graph.add_node(name, func=func)
        for dep in dependencies:
            self.graph.add_edge(dep, name)

    def node(self, depends_on: list[str] | None = None) -> Callable:
        """Implement a decorator to bind a node to this instance.

        Args:
            depends_on: A list of dependencies.

        Returns:
            A callable.
        """

        def decorator(func: Callable):
            node_name = func.__name__  # ty:ignore[unresolved-attribute]
            self.register_node(node_name, func, depends_on)

            @wraps(func)
            def wrapper(*args: object, **kwargs: object):
                return func(*args, **kwargs)

            return wrapper

        return decorator

    @abstractmethod
    def _simulate_impl(self, *args: object, **kwargs: object) -> Array:
        """Implement simulation for subclass.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            2D array of shape (n_samples, n_parameters).

        """

    def simulate(self, *args: object, **kwargs: object) -> Array:
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

    def __call__(self, *args: object, **kwargs: object) -> Array:
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
        compression: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save the simulated data to a file.

        Args:
            output_path: Path to save the file.
            file_format: File format (npy, npz, csv, hdf5). If None, infers from extension.
            data: Data to save. If None, uses last simulated data.
            compression: Optional compression setting for supported formats.
                For HDF5, this specifies the compression filter (e.g., "gzip").
                For NPZ, any non-None value enables default zlib compression.
            metadata: Optional metadata to store alongside the samples.

        """
        output_path = Path(output_path)

        if file_format is None:
            file_format = output_path.suffix.lstrip(".")

        if data is None:
            if self._last_data is None:
                raise ValueError("No data provided and no last simulated data available.")
            data = self._last_data

        if file_format == "npy":
            jnp.save(output_path, data)
        elif file_format == "npz":
            if metadata:
                payload = {"data": np.asarray(data), "__metadata__": np.array(json.dumps(metadata))}
            else:
                payload = {"data": np.asarray(data)}
            if compression is None:
                np.savez(output_path, **payload)
            else:
                np.savez_compressed(output_path, **payload)
        elif file_format == "csv":
            np.savetxt(output_path, data, delimiter=",")
        elif file_format == "hdf5":
            self._save_hdf5(output_path, data, compression=compression, metadata=metadata)
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
        elif suffix == "npz":
            with np.load(input_path) as loaded:
                return jnp.array(loaded["data"])
        elif suffix == "csv":
            return jnp.array(np.loadtxt(input_path, delimiter=","))
        elif suffix == "hdf5":
            return self._load_hdf5(input_path)
        else:
            raise ValueError(f"Unsupported format: {suffix}")

    @staticmethod
    def _save_hdf5(
        path: Path, data: Array, compression: str | None = None, metadata: dict[str, Any] | None = None
    ) -> None:
        """Save data to HDF5 format.

        Args:
            path: Path to save the file.
            data: Data to save.
            compression: Optional HDF5 compression filter.
            metadata: Optional metadata to store in the file.

        """
        with h5py.File(path, "w") as f:
            f.create_dataset("data", data=data, compression=compression)
            if metadata:
                metadata_group = f.create_group("metadata")
                for key, value in metadata.items():
                    metadata_group.attrs[key] = json.dumps(value)

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
