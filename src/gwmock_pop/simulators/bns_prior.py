"""Lightweight BNS prior simulator."""

from __future__ import annotations

import math
from collections.abc import Mapping

import jax
import jax.numpy as jnp
from jax import Array

from gwmock_pop.simulators.cbc_prior import CBCPriorSimulator


class BNSPriorSimulator(CBCPriorSimulator):
    """Sample BNS source parameters from lightweight analytic priors.

    This simulator reuses the distance, redshift, and extrinsic-parameter priors
    from :class:`~gwmock_pop.simulators.cbc_prior.CBCPriorSimulator` while
    specializing the intrinsic prior to neutron-star component masses and
    low-spin, aligned configurations by default.

    Args:
        lambda_max: Upper bound for the flat tidal deformability prior applied to
            both neutron-star components. Both ``lambda_1`` and ``lambda_2`` are
            sampled independently from ``Uniform[0, lambda_max]``.
    """

    def __init__(  # noqa: PLR0913
        self,
        source_type: str = "bns",
        *,
        m_min: float = 1.0,
        m_max: float = 3.0,
        d_min: float = 0.0,
        d_max: float = 5_000.0,
        chi_max: float = 0.05,
        aligned_spins: bool = True,
        gps_start: float = 0.0,
        gps_end: float = 1.0,
        total_mass_max: float | None = None,
        f_ref: float = 20.0,
        lambda_max: float = 3000.0,
        seed: int | None = None,
    ) -> None:
        """Initialize the BNS prior simulator."""
        self._lambda_max = float(lambda_max)
        super().__init__(
            source_type=source_type,
            m_min=m_min,
            m_max=m_max,
            d_min=d_min,
            d_max=d_max,
            chi_max=chi_max,
            aligned_spins=aligned_spins,
            gps_start=gps_start,
            gps_end=gps_end,
            total_mass_max=total_mass_max,
            f_ref=f_ref,
            seed=seed,
        )

    def _validate_configuration(self) -> None:
        """Validate constructor arguments."""
        super()._validate_configuration()
        if not math.isfinite(self._lambda_max):
            raise ValueError("lambda_max must be finite.")
        if self._lambda_max < 0.0:
            raise ValueError("lambda_max must be non-negative.")

    def _simulate_impl(self, n_samples: int, *, seed: int | None = None) -> Mapping[str, Array]:
        """Draw ``n_samples`` BNS prior samples with tidal deformability."""
        base = super()._simulate_impl(n_samples, seed=seed)
        key = jax.random.PRNGKey(0 if seed is None else int(seed))
        lambda_key_1, lambda_key_2 = jax.random.split(jax.random.fold_in(key, 99), 2)
        return {
            **base,
            "lambda_1": jax.random.uniform(lambda_key_1, shape=(n_samples,), minval=0.0, maxval=self._lambda_max),
            "lambda_2": jax.random.uniform(lambda_key_2, shape=(n_samples,), minval=0.0, maxval=self._lambda_max),
        }

    def _sample_component_masses(self, key: Array, n_samples: int) -> tuple[Array, Array]:
        """Draw ordered BNS component masses with ``m1 >= m2``."""
        mass_1, mass_2 = super()._sample_component_masses(key, n_samples)
        return jnp.maximum(mass_1, mass_2), jnp.minimum(mass_1, mass_2)
