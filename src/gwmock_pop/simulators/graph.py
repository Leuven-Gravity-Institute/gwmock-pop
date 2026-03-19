"""Graph-based population simulator."""

from __future__ import annotations

import secrets
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import networkx as nx
from jax import Array

from gwmock_pop.graph.build import build_dependency_graph
from gwmock_pop.mixins.random import RandomMixin
from gwmock_pop.simulators.simulator import Simulator
from gwmock_pop.utils.import_utils import import_from_string
from gwmock_pop.utils.yaml import read_data_file


class GraphSimulator(RandomMixin, Simulator):
    """Graph-based population simulator.

    This simulator uses a probabilistic graphical model to generate populations.
    Parameters are defined in a configuration with dependencies, and the simulator
    executes sampling in topological order based on the dependency graph.

    Args:
        config: Configuration dictionary defining parameters and their sampling/transform rules.
        **kwargs: Additional arguments passed to parent class.

    Example:
        >>> config = {
        ...     "mass_1": {
        ...         "sampler": {
        ...             "function": "planck_tapered_broken_power_law_plus_two_peaks",
        ...             "arguments": {
        ...                 "alpha_1": 1.72,
        ...                 "alpha_2": 4.51,
        ...                 "transition": 35.6,
        ...                 "minimum": 5.06,
        ...                 "maximum": 300.0,
        ...             },
        ...         },
        ...     },
        ...     "mass_ratio": {
        ...         "sampler": {
        ...             "function": "planck_tapered_conditional_ratio_power_law",
        ...             "arguments": {"denominator": "@mass_1"},
        ...         },
        ...     },
        ... }
        >>> simulator = GraphSimulator(config=config)
        >>> population = simulator()
    """

    def __init__(
        self,
        config: dict[str, Any],
        source_type: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the graph-based simulator.

        Args:
            config: Configuration dictionary with parameter definitions.
            source_type: Optional logical source identifier for higher-level orchestration.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self._config = config
        self._source_type = source_type
        self._sampled_values: dict[str, Any] = {}
        self._parameter_names = [name for name, spec in config.items() if self._include_in_output(spec)]
        self._build_graph()

    @property
    def parameter_names(self) -> list[str]:
        """Get the names of the parameters.

        Returns:
            List of parameter names.
        """
        return self._parameter_names

    @property
    def source_type(self) -> str | None:
        """Get the optional logical source type.

        Returns:
            Source type string, when configured.
        """
        return self._source_type

    @staticmethod
    def _include_in_output(spec: dict[str, Any]) -> bool:
        """Determine whether a parameter should be included in the final output."""
        return not bool(spec.get("exclude", False) or spec.get("intermediate", False))

    def _build_graph(self) -> None:
        """Build the dependency graph from the configuration."""
        # Extract only parameters with samplers (not transforms)
        self._graph = build_dependency_graph(self._config)

    def _get_ordered_parameters(self) -> list[str]:
        """Get parameters in topological order.

        Returns:
            List of parameter names in sampling order.
        """
        ordered = list(nx.topological_sort(self._graph))
        undefined = [name for name in ordered if name not in self._config]
        if undefined:
            raise ValueError(f"Undefined parameter dependencies: {undefined}")
        return ordered

    def _execute_sampler(
        self,
        sampler_name: str,
        arguments: dict[str, Any],
        n_samples: int | None = None,
    ) -> Array:
        """Execute a sampler with given arguments.

        Args:
            sampler_name: Name of the sampler function.
            arguments: Arguments for the sampler.
            n_samples: Number of samples to inject when the sampler arguments omit it.

        Returns:
            Sampled array.
        """
        # Resolve dependencies in arguments
        if arguments is None:
            arguments = {}
        elif not isinstance(arguments, dict):
            raise ValueError("Sampler arguments must be a mapping (or null).")
        resolved_args: dict[str, Any] = {}
        for key, value in arguments.items():
            if isinstance(value, str) and value.startswith("@"):
                dep_name = value[1:]  # Remove @ prefix
                if dep_name in self._sampled_values:
                    resolved_args[key] = self._sampled_values[dep_name]
                else:
                    raise ValueError(f"Dependency '{dep_name}' not sampled yet")
            else:
                resolved_args[key] = value

        if n_samples is not None and "n_samples" not in resolved_args:
            resolved_args["n_samples"] = n_samples

        # Add random key if needed
        if "key" not in resolved_args:
            resolved_args["key"] = self.rng_manager.new_key

        # Dispatch to sampler function
        sampler_func = import_from_string(object_path=sampler_name, default_module="gwmock_pop.samplers")
        return sampler_func(**resolved_args)

    def _execute_transform(
        self,
        transform_spec: dict[str, Any],
    ) -> Array:
        """Execute a transform.

        Args:
            transform_spec: Transform specification.

        Returns:
            Transformed array.
        """
        if isinstance(transform_spec, str):
            raise ValueError("String transform expressions are not supported by GraphSimulator yet.")

        function_name = transform_spec.get("function", "")
        arguments = transform_spec.get("arguments")
        if arguments is None:
            arguments = {}
        elif not isinstance(arguments, dict):
            raise ValueError("Only mapping-style transform arguments are currently supported.")

        # Resolve dependencies
        resolved_args: dict[str, Any] = {}
        for key, value in arguments.items():
            if isinstance(value, str) and value.startswith("@"):
                dep_name = value[1:]
                if dep_name in self._sampled_values:
                    resolved_args[key] = self._sampled_values[dep_name]
                else:
                    raise ValueError(f"Dependency '{dep_name}' not available for transform")
            else:
                resolved_args[key] = value

        # Dispatch to transform function
        transform_func = import_from_string(object_path=function_name, default_module="gwmock_pop.transforms")
        return transform_func(**resolved_args)

    def _coerce_output_column(
        self,
        parameter_name: str,
        value: Any,
        expected_n_samples: int | None,
    ) -> tuple[Array, int]:
        """Convert a sampled value into a validated output column."""
        array = jnp.asarray(value)
        if array.ndim == 0:
            raise ValueError(f"Parameter '{parameter_name}' produced a scalar output, expected an array of samples.")

        current_n_samples = int(array.shape[0])
        if expected_n_samples is not None and current_n_samples != expected_n_samples:
            raise ValueError(
                f"Parameter '{parameter_name}' produced {current_n_samples} samples, expected {expected_n_samples}."
            )
        return array, current_n_samples

    def _simulate_impl(self, n_samples: int | None = None, **kwargs: Any) -> Array:
        """Implement simulation using graph traversal.

        Args:
            n_samples: Number of samples to inject into samplers when not explicitly configured.
            **kwargs: Keyword arguments.

        Returns:
            2D array of shape (n_samples, n_parameters).
        """
        del kwargs

        self._sampled_values = {}

        # Get ordered parameters
        ordered_params = self._get_ordered_parameters()

        # Sample each parameter in order
        for param_name in ordered_params:
            spec = self._config[param_name]

            if "sampler" in spec:
                sampler_spec = spec["sampler"]
                sampler_name = sampler_spec.get("function", "")
                sampler_args = sampler_spec.get("arguments", {})
                samples = self._execute_sampler(sampler_name, sampler_args, n_samples=n_samples)
                self._sampled_values[param_name] = samples

            elif "transform" in spec:
                transform_spec = spec["transform"]
                transformed = self._execute_transform(transform_spec)
                self._sampled_values[param_name] = transformed

        if not self.parameter_names:
            raise ValueError("GraphSimulator configuration does not define any output parameters.")

        expected_n_samples = n_samples
        output_columns: list[Array] = []
        for parameter_name in self.parameter_names:
            output_column, expected_n_samples = self._coerce_output_column(
                parameter_name=parameter_name,
                value=self._sampled_values[parameter_name],
                expected_n_samples=expected_n_samples,
            )
            output_columns.append(output_column)

        # Build output array in parameter order
        output = jnp.column_stack(output_columns)
        return output

    @classmethod
    def from_config_file(cls, config_path: str | Path, encoding: str = "utf-8", **kwargs: Any) -> GraphSimulator:
        """Create simulator from configuration file.

        Args:
            config_path: Path to YAML/TOML configuration file.
            encoding: Encoding of the file.
            **kwargs: Additional arguments passed to __init__.

        Returns:
            Configured simulator instance.
        """
        config_path = Path(config_path)
        try:
            config = read_data_file(config_path, encoding=encoding)
        except ValueError as error:
            message = str(error)
            if "Suffix of filename=" in message:
                raise ValueError(
                    f"Suffix of config_path={config_path} is not supported. Only '.yaml', '.yml', and '.toml' are supported."
                ) from error
            if "mapping at the top level" in message:
                raise ValueError("Config file must contain a mapping at the root.") from error
            raise

        # Extract parameters section if present
        if "parameters" in config:
            config = config["parameters"]

        return cls(config=config, **kwargs)

    def reset(self) -> None:
        """Reset the simulator state."""
        self._sampled_values = {}
        # Reset RNG by using a fresh key
        if hasattr(self, "_rng_manager"):
            seed = getattr(self._rng_manager, "_seed", None)
            if seed is not None:
                self._rng_manager.key = jax.random.key(seed)
            else:
                self._rng_manager.key = jax.random.key(secrets.randbelow(2**63))
