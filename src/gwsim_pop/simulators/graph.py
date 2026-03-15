"""Graph-based population simulator."""

from __future__ import annotations

import secrets
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import networkx as nx
from jax import Array

from gwsim_pop.graph.build import build_dependency_graph
from gwsim_pop.mixins.random import RandomMixin
from gwsim_pop.simulators.simulator import Simulator
from gwsim_pop.utils.import_utils import import_from_string


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
        **kwargs: Any,
    ) -> None:
        """Initialize the graph-based BBH simulator.

        Args:
            config: Configuration dictionary with parameter definitions.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self._config = config
        self._sampled_values: dict[str, Array] = {}
        self._parameter_names = [name for name, spec in config.items() if not spec.get("intermediate", False)]
        self._build_graph()

    @property
    def parameter_names(self) -> list[str]:
        """Get the names of the parameters.

        Returns:
            List of parameter names.
        """
        return self._parameter_names

    @property
    def source_type(self) -> str:
        """Get the source type.

        Returns:
            Source type string.
        """
        return "bbh"

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
    ) -> Array:
        """Execute a sampler with given arguments.

        Args:
            sampler_name: Name of the sampler function.
            arguments: Arguments for the sampler.

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

        # Add random key if needed
        if "key" not in resolved_args:
            resolved_args["key"] = self.rng_manager.new_key

        # Dispatch to sampler function
        sampler_func = import_from_string(object_path=sampler_name, default_module="gwsim_pop.samplers")
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
        transform_func = import_from_string(object_path=function_name, default_module="gwsim_pop.transforms")
        return transform_func(**resolved_args)

    def _simulate_impl(self, *args: Any, **kwargs: Any) -> Array:
        """Implement simulation using graph traversal.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            2D array of shape (n_samples, n_parameters).
        """
        # Get ordered parameters
        ordered_params = self._get_ordered_parameters()

        # Sample each parameter in order
        for param_name in ordered_params:
            spec = self._config[param_name]

            if "sampler" in spec:
                sampler_spec = spec["sampler"]
                sampler_name = sampler_spec.get("function", "")
                sampler_args = sampler_spec.get("arguments", {})
                samples = self._execute_sampler(sampler_name, sampler_args)
                self._sampled_values[param_name] = samples

            elif "transform" in spec:
                transform_spec = spec["transform"]
                transformed = self._execute_transform(transform_spec)
                self._sampled_values[param_name] = transformed

        # Build output array in parameter order
        # Include parameters that don't have intermediate=False (i.e., default to True)
        output = jnp.column_stack([self._sampled_values[name] for name in self.parameter_names])
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

        if config_path.suffix in [".yaml", ".yml"]:
            import yaml  # noqa: PLC0415

            with open(config_path, encoding=encoding) as f:
                config = yaml.safe_load(f)
        elif config_path.suffix == ".toml":
            import tomllib  # noqa: PLC0415

            # Binary mode does not require encoding
            with open(config_path, "rb") as f:
                config = tomllib.load(f)
        else:
            raise ValueError(
                f"Suffix of config_path={config_path} is not supported. Only '.yaml', '.yml', and '.toml' are supported."
            )

        if not isinstance(config, dict):
            raise ValueError("Config file must contain a mapping at the root.")

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
