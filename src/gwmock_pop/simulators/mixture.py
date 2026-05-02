"""Mixture wrapper for population simulators."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from gwmock_pop.mixins.random import RandomMixin
from gwmock_pop.protocols import GWPopSimulator
from gwmock_pop.simulators.simulator import Simulator


class MixtureSimulator(RandomMixin, Simulator):
    """Draw samples from a weighted mixture of ``GWPopSimulator`` components."""

    def __init__(
        self,
        simulators: Sequence[GWPopSimulator],
        weights: Sequence[float],
        *,
        seed: int | None = None,
    ) -> None:
        """Initialize the mixture simulator."""
        super().__init__(seed=seed)
        self._simulators = tuple(simulators)
        self._weights = self._normalize_weights(weights)
        self._parameter_names = self._validate_parameter_names()
        self._source_type = self._validate_source_type()

    @property
    def parameter_names(self) -> list[str]:
        """Return the shared parameter names of the component simulators."""
        return list(self._parameter_names)

    @property
    def source_type(self) -> str:
        """Return the shared source type of the component simulators."""
        return self._source_type

    def _normalize_weights(self, weights: Sequence[float]) -> tuple[float, ...]:
        """Validate and normalize the component weights."""
        if not self._simulators:
            raise ValueError("simulators must contain at least one component.")
        if len(weights) != len(self._simulators):
            raise ValueError("weights must have the same length as simulators.")

        normalized_weights: list[float] = []
        for index, simulator in enumerate(self._simulators):
            if not isinstance(simulator, GWPopSimulator):
                raise TypeError(f"simulators[{index}] must satisfy GWPopSimulator.")

            weight = float(weights[index])
            if not math.isfinite(weight):
                raise ValueError(f"weights[{index}] must be finite.")
            if weight < 0.0:
                raise ValueError(f"weights[{index}] must be greater than or equal to 0.")
            normalized_weights.append(weight)

        total_weight = sum(normalized_weights)
        if total_weight <= 0.0:
            raise ValueError("weights must sum to a positive value.")

        return tuple(weight / total_weight for weight in normalized_weights)

    def _validate_parameter_names(self) -> tuple[str, ...]:
        """Validate that every component exposes the same parameter-name set."""
        reference = tuple(self._simulators[0].parameter_names)
        reference_set = set(reference)

        for index, simulator in enumerate(self._simulators[1:], start=1):
            if set(simulator.parameter_names) != reference_set:
                raise ValueError(
                    "All component simulators must expose the same parameter_names set. "
                    f"Mismatch found at simulators[{index}]."
                )

        return reference

    def _validate_source_type(self) -> str:
        """Validate that every component exposes the same source type."""
        reference = self._simulators[0].source_type

        for index, simulator in enumerate(self._simulators[1:], start=1):
            if simulator.source_type != reference:
                raise ValueError(
                    f"All component simulators must share the same source_type. Mismatch found at simulators[{index}]."
                )

        return reference

    def _simulate_impl(
        self,
        n_samples: int,
        *,
        seed: int | None = None,
        **kwargs: Any,
    ) -> Mapping[str, Array]:
        """Draw ``n_samples`` from the configured mixture."""
        if n_samples < 0:
            raise ValueError(f"n_samples must be >= 0, got {n_samples}.")
        if n_samples == 0:
            return {name: jnp.empty((0,)) for name in self.parameter_names}

        key = self.rng_manager.new_key if seed is None else jax.random.PRNGKey(int(seed))
        assignment_key, component_seed_key = jax.random.split(key)
        probabilities = jnp.asarray(self._weights)
        assignments = jax.random.categorical(assignment_key, logits=jnp.log(probabilities), shape=(n_samples,))
        sorted_indices = jnp.argsort(assignments, stable=True)
        inverse_indices = jnp.argsort(sorted_indices)
        component_seed_keys = jax.random.split(component_seed_key, len(self._simulators))

        component_samples: list[Mapping[str, Array]] = []
        for index, simulator in enumerate(self._simulators):
            count = int(jnp.sum(assignments == index))
            if count == 0:
                continue

            component_seed = int(
                jax.random.randint(
                    component_seed_keys[index],
                    shape=(),
                    minval=0,
                    maxval=jnp.iinfo(jnp.int32).max,
                    dtype=jnp.int32,
                )
            )
            component_samples.append(simulator.simulate(count, seed=component_seed, **kwargs))

        return {
            name: jnp.concatenate([sample[name] for sample in component_samples], axis=0)[inverse_indices]
            for name in self.parameter_names
        }
