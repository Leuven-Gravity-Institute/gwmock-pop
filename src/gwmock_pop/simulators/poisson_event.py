"""Poisson-count wrapper for population simulators."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from gwmock_pop.mixins.random import RandomMixin
from gwmock_pop.protocols import GWPopSimulator
from gwmock_pop.simulators.simulator import Simulator


class PoissonEventSampler(RandomMixin, Simulator):
    """Wrap a ``GWPopSimulator`` and draw a Poisson-distributed event count."""

    def __init__(
        self,
        simulator: GWPopSimulator,
        rate_gpc3_yr: float,
        t_obs_yr: float,
        seed: int | None = None,
    ) -> None:
        """Initialize the Poisson event sampler.

        Args:
            simulator: Wrapped simulator that generates per-event parameters.
            rate_gpc3_yr: Volumetric merger rate factor used in the Poisson mean.
            t_obs_yr: Observation time in years used in the Poisson mean.
            seed: Optional seed for deterministic event-count draws.
        """
        super().__init__(seed=seed)
        if not isinstance(simulator, GWPopSimulator):
            raise TypeError("simulator must satisfy GWPopSimulator.")

        self._simulator = simulator
        self._rate_gpc3_yr = float(rate_gpc3_yr)
        self._t_obs_yr = float(t_obs_yr)
        self._validate_configuration()

    @property
    def parameter_names(self) -> list[str]:
        """Return the wrapped simulator parameter names."""
        return list(self._simulator.parameter_names)

    @property
    def source_type(self) -> str:
        """Return the wrapped simulator source type."""
        return self._simulator.source_type

    @property
    def mean_event_count(self) -> float:
        """Return the Poisson mean ``rate_gpc3_yr * t_obs_yr``."""
        return self._rate_gpc3_yr * self._t_obs_yr

    def _validate_configuration(self) -> None:
        """Validate the wrapper configuration."""
        for name, value in (("rate_gpc3_yr", self._rate_gpc3_yr), ("t_obs_yr", self._t_obs_yr)):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
            if value < 0.0:
                raise ValueError(f"{name} must be greater than or equal to 0.")

    def _simulate_impl(
        self,
        n_samples: int | None = None,
        *,
        seed: int | None = None,
        **kwargs: Any,
    ) -> Mapping[str, Array]:
        """Draw a Poisson event count, then delegate sample generation.

        ``n_samples`` is accepted for protocol compatibility but ignored because
        the realized event count is drawn internally from the configured Poisson
        process.
        """
        del n_samples

        key = self.rng_manager.new_key if seed is None else jax.random.PRNGKey(int(seed))
        count_key, wrapped_seed_key = jax.random.split(key)
        n_events = int(jax.random.poisson(count_key, lam=self.mean_event_count))

        if n_events == 0:
            return {name: jnp.empty((0,)) for name in self.parameter_names}

        wrapped_seed = int(
            jax.random.randint(
                wrapped_seed_key,
                shape=(),
                minval=0,
                maxval=jnp.iinfo(jnp.int32).max,
                dtype=jnp.int32,
            )
        )
        return self._simulator.simulate(n_events, seed=wrapped_seed, **kwargs)
