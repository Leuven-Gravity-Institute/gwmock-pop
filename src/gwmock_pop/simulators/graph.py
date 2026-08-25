"""Graph-based population simulator."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import TYPE_CHECKING, Any

import networkx as nx

from gwmock_pop.config.simulation import (
    SimulationTarget,
    simulation_target_from_file,
    simulation_target_from_preset,
)
from gwmock_pop.graph.build import build_dependency_graph
from gwmock_pop.mixins.random import RandomMixin
from gwmock_pop.provenance import graph_simulation_origin, resolved_configuration_payload
from gwmock_pop.simulators.simulator import Simulator
from gwmock_pop.utils.import_utils import import_from_string

if TYPE_CHECKING:
    from jax import Array


class GraphSimulator(RandomMixin, Simulator):
    """Graph-based population simulator.

    This simulator uses a probabilistic graphical model to generate populations.
    Parameters are defined in a configuration with dependencies, and the simulator
    executes sampling in topological order based on the dependency graph.

    Args:
        config: Configuration dictionary defining parameters and their sampling/transform rules.
        **kwargs: Additional arguments passed to parent class.

    Note:
        ``source_type`` must be set before calling ``simulate()``. Construction
        without ``source_type`` is allowed (e.g. for builder patterns), but
        ``simulate()`` raises :exc:`ValueError` if ``source_type`` is ``None``
        at call time. Pass ``source_type=<str>`` to the constructor to avoid
        this.

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
            source_type: Logical source identifier for higher-level orchestration.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)
        if source_type is not None and not source_type.strip():
            raise ValueError("source_type must be a non-empty string.")
        self._config = config
        self._source_type: str | None = source_type
        self._sampled_values: dict[str, Any] = {}
        self._parameter_names = [name for name, spec in config.items() if self._include_in_output(spec)]
        self._target: SimulationTarget | None = None
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
        """Get the logical source type.

        Returns:
            Source type string.
        """
        if self._source_type is None:
            raise RuntimeError(
                "GraphSimulator.source_type is not configured; expected a non-empty string to satisfy GWPopSimulator."
            )
        return self._source_type

    @staticmethod
    def _include_in_output(spec: dict[str, Any]) -> bool:
        """Determine whether a parameter should be included in the final output."""
        return not bool(spec.get("exclude") or spec.get("intermediate"))

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
            raise TypeError("String transform expressions are not supported by GraphSimulator yet.")

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
        transform_signature = inspect.signature(transform_func)
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in transform_signature.parameters.values()
        )
        if "key" not in resolved_args and ("key" in transform_signature.parameters or accepts_kwargs):
            resolved_args["key"] = self.rng_manager.new_key
        return transform_func(**resolved_args)

    def _coerce_output_column(
        self,
        parameter_name: str,
        value: Any,
        expected_n_samples: int | None,
    ) -> tuple[Array, int]:
        """Convert a sampled value into a validated output column."""
        import jax.numpy as jnp  # noqa: PLC0415  # deferred JAX import

        array = jnp.asarray(value)
        if array.ndim == 0:
            raise ValueError(f"Parameter '{parameter_name}' produced a scalar output, expected an array of samples.")
        if array.ndim > 1:
            array = array.reshape(-1)

        current_n_samples = int(array.shape[0])
        if expected_n_samples is not None and current_n_samples != expected_n_samples:
            raise ValueError(
                f"Parameter '{parameter_name}' produced {current_n_samples} samples, expected {expected_n_samples}."
            )
        return array, current_n_samples

    def _simulate_impl(self, n_samples: int | None = None, **kwargs: Any) -> dict[str, Array]:
        """Implement simulation using graph traversal.

        Args:
            n_samples: Number of samples to inject into samplers when not explicitly configured.
            **kwargs: Keyword arguments.

        Returns:
            Mapping from parameter names to 1D arrays of shape ``(n_samples,)``.
        """
        if self._source_type is None:
            raise ValueError(
                "source_type must be set before calling simulate(). "
                "Pass source_type=<str> to the constructor or set it "
                "via the builder API."
            )

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
        output_mapping: dict[str, Array] = {}
        for parameter_name in self.parameter_names:
            output_column, expected_n_samples = self._coerce_output_column(
                parameter_name=parameter_name,
                value=self._sampled_values[parameter_name],
                expected_n_samples=expected_n_samples,
            )
            output_mapping[parameter_name] = output_column

        return output_mapping

    @classmethod
    def from_target(cls, target: SimulationTarget, **kwargs: Any) -> GraphSimulator:
        """Create a simulator from an already-resolved simulation target.

        The target is kept, so the simulator can describe the configuration it
        was built from when a catalogue is written.

        Args:
            target: Resolved preset or configuration file.
            **kwargs: Additional arguments passed to __init__. A ``source_type``
                given here wins over the one the target declares.

        Returns:
            Configured simulator instance.
        """
        options = dict(kwargs)
        if target.source_type is not None:
            options.setdefault("source_type", target.source_type)
        simulator = cls(config=target.graph_config, **options)
        simulator._target = target
        return simulator

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
        return cls.from_target(simulation_target_from_file(config_path, encoding=encoding), **kwargs)

    @classmethod
    def from_preset(cls, preset_name: str, **kwargs: Any) -> GraphSimulator:
        """Create a graph simulator from a packaged preset."""
        return cls.from_target(simulation_target_from_preset(preset_name), **kwargs)

    def _provenance_origin(self) -> dict[str, Any]:
        """Return the origin block for a run of this parameter graph.

        Returns:
            An origin block carrying the parameter graph with its defaults filled
            in, so the run can be reconstructed from the record alone.
        """
        from gwmock_pop.config.main import MainConfiguration  # noqa: PLC0415  # avoids an import cycle

        configuration = MainConfiguration() if self._target is None else self._target.configuration
        config_path = (
            None if self._target is None or self._target.config_path is None else str(self._target.config_path)
        )
        return graph_simulation_origin(
            graph_config=self._config,
            configuration=resolved_configuration_payload(configuration),
            preset=None if self._target is None else self._target.preset,
            config_path=config_path,
        )

    def reset(self) -> None:
        """Reset the simulator state.

        The random stream is rewound to the seed the simulator was built with,
        including a seed that was drawn rather than requested. Anything else
        would make the state after a reset undescribable.
        """
        import jax  # noqa: PLC0415  # deferred JAX import

        self._sampled_values = {}
        if hasattr(self, "_rng_manager"):
            self._rng_manager.key = jax.random.key(self._rng_manager.resolved_seed)
