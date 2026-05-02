"""Lightweight BNS prior simulator."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from gwmock_pop.simulators.cbc_prior import CBCPriorSimulator


class BNSPriorSimulator(CBCPriorSimulator):
    """Sample BNS source parameters from lightweight analytic priors.

    This simulator reuses the distance, redshift, and extrinsic-parameter priors
    from :class:`~gwmock_pop.simulators.cbc_prior.CBCPriorSimulator` while
    specializing the intrinsic prior to neutron-star component masses and
    low-spin, aligned configurations by default.
    """

    def __init__(  # noqa: PLR0913
        self,
        source_type: str = "bns",
        *,
        m_min: float = 1.0,
        m_max: float = 3.0,
        d_max: float = 5_000.0,
        chi_max: float = 0.05,
        aligned_spins: bool = True,
        gps_start: float = 0.0,
        gps_end: float = 1.0,
        total_mass_max: float | None = None,
        f_ref: float = 20.0,
        seed: int | None = None,
    ) -> None:
        """Initialize the BNS prior simulator."""
        super().__init__(
            source_type=source_type,
            m_min=m_min,
            m_max=m_max,
            d_max=d_max,
            chi_max=chi_max,
            aligned_spins=aligned_spins,
            gps_start=gps_start,
            gps_end=gps_end,
            total_mass_max=total_mass_max,
            f_ref=f_ref,
            seed=seed,
        )

    def _sample_component_masses(self, key: Array, n_samples: int) -> tuple[Array, Array]:
        """Draw ordered BNS component masses with ``m1 >= m2``."""
        mass_1, mass_2 = super()._sample_component_masses(key, n_samples)
        return jnp.maximum(mass_1, mass_2), jnp.minimum(mass_1, mass_2)
