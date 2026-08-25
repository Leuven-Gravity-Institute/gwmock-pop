"""Base class of the simulators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np

from gwmock_pop.loaders.file_loader import infer_population_file_format, write_population_catalogue
from gwmock_pop.provenance import build_provenance_record, run_metadata, simulator_origin

if TYPE_CHECKING:
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
        self._last_data: Mapping[str, Array] | Array | None = None
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
    def _simulate_impl(self, *args: object, **kwargs: object) -> Mapping[str, Array]:
        """Implement simulation for subclass.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Mapping from parameter names to 1D arrays of length n_samples.

        """

    def simulate(self, *args: object, **kwargs: object) -> Mapping[str, Array]:
        """Simulate a population of sources.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Mapping from parameter names to 1D arrays of length n_samples.

        """
        result = self._simulate_impl(*args, **kwargs)
        self._validate_output(result)
        self._last_data = result
        return result

    def __call__(self, *args: object, **kwargs: object) -> Mapping[str, Array]:
        """Call simulate() with n_samples.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Mapping from parameter names to 1D arrays of length n_samples.

        """
        return self.simulate(*args, **kwargs)

    def save_catalogue(
        self,
        output_path: str | Path,
        *,
        data: Mapping[str, Array] | None = None,
        provenance: Mapping[str, Any] | None = None,
        compression: str | None = None,
    ) -> None:
        """Persist a simulated population as a named-column catalogue.

        Persistence goes through :func:`~gwmock_pop.loaders.write_population_catalogue`,
        the one writer in this package, so a file written here is one the
        package's own readers accept and it carries the same provenance record
        as a file written by the CLI.

        Args:
            output_path: Destination ``.csv``, ``.h5``, or ``.hdf5`` file.
            data: Population to write. Defaults to the last simulated population.
            provenance: Record to store with the catalogue. Defaults to the one
                this simulator can describe itself with.
            compression: Optional HDF5 compression filter.

        Raises:
            ValueError: If no population is given and none has been simulated.
            TypeError: If the population is not a mapping of named columns.
        """
        population = self._last_data if data is None else data
        if population is None:
            raise ValueError("No data provided and no last simulated data available.")
        if not isinstance(population, Mapping):
            raise TypeError("A catalogue is written from a mapping of parameter names to columns.")

        ordered = {name: population[name] for name in self.parameter_names}
        n_samples = 0 if not ordered else int(np.asarray(next(iter(ordered.values()))).shape[0])
        record = (
            self.build_provenance_record(
                n_samples=n_samples,
                file_format=infer_population_file_format(output_path),
                writer=f"{type(self).__module__}.{type(self).__qualname__}.save_catalogue",
            )
            if provenance is None
            else dict(provenance)
        )
        write_population_catalogue(
            output_path=output_path, population=ordered, provenance=record, compression=compression
        )

    def build_provenance_record(
        self,
        *,
        n_samples: int,
        file_format: str,
        parameter_names: Sequence[str] | None = None,
        run: Mapping[str, Any] | None = None,
        writer: str | None = None,
    ) -> dict[str, Any]:
        """Build the provenance record describing a catalogue from this simulator.

        This is the single record builder behind both persistence paths: the CLI
        calls it with the run settings it resolved, and :meth:`save_catalogue`
        calls it with what the simulator knows about itself. Neither assembles a
        record of its own, so the two cannot drift apart.

        Args:
            n_samples: Number of rows being written.
            file_format: Format the catalogue is written in.
            parameter_names: Column names in output order. Defaults to this
                simulator's parameter names.
            run: Block from :func:`gwmock_pop.provenance.run_metadata`. Defaults
                to what the simulator's own generator reports.
            writer: Import path of the code writing the file.

        Returns:
            The record.
        """
        return build_provenance_record(
            origin=self._provenance_origin(),
            source_type=self._provenance_source_type(),
            parameter_names=self.parameter_names if parameter_names is None else parameter_names,
            n_samples=n_samples,
            file_format=file_format,
            writer=writer or f"{type(self).__module__}.{type(self).__qualname__}.save_catalogue",
            run=self._provenance_run() if run is None else run,
        )

    def _provenance_source_type(self) -> str | None:
        """Return the source type to record, or ``None`` when it is not configured.

        Returns:
            The simulator's source type when it has one.
        """
        try:
            return self.source_type  # ty:ignore[unresolved-attribute]
        except (AttributeError, RuntimeError, ValueError):
            return None

    def _provenance_origin(self) -> dict[str, Any]:
        """Return the origin block describing how this simulator produced its data.

        Returns:
            An origin block. The base class can name the code that ran but not
            the configuration that drove it, so the record it yields is not
            replayable; subclasses that know their configuration override this.
        """
        return simulator_origin(component=f"{type(self).__module__}.{type(self).__qualname__}")

    def _provenance_run(self) -> dict[str, Any] | None:
        """Return the block describing the run, when the seed in use is knowable.

        Returns:
            The run block, or ``None`` for a simulator with no seeded generator.
        """
        rng_manager = getattr(self, "rng_manager", None)
        if rng_manager is None:
            return None
        seed_source = "library" if rng_manager.requested_seed is not None else "drawn"
        return run_metadata(name=None, seed=rng_manager.resolved_seed, seed_source=seed_source)

    def _validate_output(self, result: Mapping[str, Array]) -> None:
        """Validate the output of simulate().

        Args:
            result: Mapping to validate.

        Raises:
            ValueError: If output is invalid.

        """
        import jax.numpy as jnp  # noqa: PLC0415  # deferred JAX import

        expected_keys = list(self.parameter_names)
        actual_keys = list(result.keys())
        if set(actual_keys) != set(expected_keys):
            raise ValueError(f"Expected keys {expected_keys}, got {actual_keys}.")

        n_samples: int | None = None
        for parameter_name in expected_keys:
            if parameter_name not in result:
                raise ValueError(f"Missing parameter '{parameter_name}' in simulator output.")

            array = jnp.asarray(result[parameter_name])
            if array.ndim != 1:
                raise ValueError(f"Expected 1D array for parameter '{parameter_name}', got {array.ndim}D array.")

            parameter_n_samples = int(array.shape[0])
            if n_samples is None:
                n_samples = parameter_n_samples
            elif parameter_n_samples != n_samples:
                raise ValueError(
                    f"Parameter '{parameter_name}' has {parameter_n_samples} samples, expected {n_samples}."
                )
