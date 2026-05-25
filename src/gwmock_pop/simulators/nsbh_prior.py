"""Lightweight NSBH prior simulator."""

from __future__ import annotations

import math
from collections.abc import Mapping

import jax
import jax.numpy as jnp
from jax import Array

from gwmock_pop.simulators.cbc_prior import CBCPriorSimulator


class NSBHPriorSimulator(CBCPriorSimulator):
    """Sample NSBH source parameters from lightweight analytic priors.

    This simulator reuses the distance, redshift, and extrinsic-parameter priors
    from :class:`~gwmock_pop.simulators.cbc_prior.CBCPriorSimulator` while
    specializing the intrinsic prior to black-hole primary masses, neutron-star
    secondary masses, and component-specific spin-magnitude bounds.
    """

    def __init__(  # noqa: PLR0913
        self,
        source_type: str = "nsbh",
        *,
        bh_mass_min: float = 3.0,
        bh_mass_max: float = 50.0,
        ns_mass_min: float = 1.0,
        ns_mass_max: float = 3.0,
        d_max: float = 5_000.0,
        bh_chi_max: float = 1.0,
        ns_chi_max: float = 0.05,
        aligned_spins: bool = False,
        gps_start: float = 0.0,
        gps_end: float = 1.0,
        total_mass_max: float | None = None,
        f_ref: float = 20.0,
        ns_lambda_max: float = 3000.0,
        seed: int | None = None,
    ) -> None:
        """Initialize the NSBH prior simulator."""
        self._bh_mass_min = float(bh_mass_min)
        self._bh_mass_max = float(bh_mass_max)
        self._ns_mass_min = float(ns_mass_min)
        self._ns_mass_max = float(ns_mass_max)
        self._bh_chi_max = float(bh_chi_max)
        self._ns_chi_max = float(ns_chi_max)
        self._ns_lambda_max = float(ns_lambda_max)
        super().__init__(
            source_type=source_type,
            m_min=min(self._bh_mass_min, self._ns_mass_min),
            m_max=max(self._bh_mass_max, self._ns_mass_max),
            d_max=d_max,
            chi_max=max(self._bh_chi_max, self._ns_chi_max),
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
        for name, value in (
            ("bh_mass_min", self._bh_mass_min),
            ("bh_mass_max", self._bh_mass_max),
            ("ns_mass_min", self._ns_mass_min),
            ("ns_mass_max", self._ns_mass_max),
            ("bh_chi_max", self._bh_chi_max),
            ("ns_chi_max", self._ns_chi_max),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
        if self._bh_mass_min <= 0.0:
            raise ValueError("bh_mass_min must be positive.")
        if self._bh_mass_max <= self._bh_mass_min:
            raise ValueError("bh_mass_max must be greater than bh_mass_min.")
        if self._ns_mass_min <= 0.0:
            raise ValueError("ns_mass_min must be positive.")
        if self._ns_mass_max <= self._ns_mass_min:
            raise ValueError("ns_mass_max must be greater than ns_mass_min.")
        if self._bh_mass_min < self._ns_mass_max:
            raise ValueError("bh_mass_min must be greater than or equal to ns_mass_max.")
        if not (0.0 <= self._bh_chi_max <= 1.0):
            raise ValueError("bh_chi_max must be in [0, 1].")
        if not (0.0 <= self._ns_chi_max <= 1.0):
            raise ValueError("ns_chi_max must be in [0, 1].")
        if self._total_mass_max is not None and self._total_mass_max <= self._bh_mass_min + self._ns_mass_min:
            raise ValueError("total_mass_max must be greater than bh_mass_min + ns_mass_min to admit any samples.")
        if not math.isfinite(self._ns_lambda_max):
            raise ValueError("ns_lambda_max must be finite.")
        if self._ns_lambda_max < 0.0:
            raise ValueError("ns_lambda_max must be non-negative.")

    def _simulate_impl(self, n_samples: int, *, seed: int | None = None) -> Mapping[str, Array]:
        """Draw ``n_samples`` NSBH prior samples using JAX PRNGs."""
        if n_samples < 0:
            raise ValueError(f"n_samples must be >= 0, got {n_samples}.")

        key = jax.random.PRNGKey(0 if seed is None else int(seed))
        (
            mass_key,
            distance_key,
            ra_key,
            dec_key,
            inclination_key,
            polarization_key,
            phase_key,
            time_key,
            spin1_key,
            spin2_key,
            asc_node_key,
            periastron_key,
            lambda_key,
        ) = jax.random.split(key, 13)

        mass_1, mass_2 = self._sample_component_masses(mass_key, n_samples)
        distance = self._sample_distance(distance_key, n_samples)
        inclination = self._sample_isotropic_polar_angle(inclination_key, n_samples)
        theta_jn = (
            inclination
            if self._aligned_spins
            else self._sample_isotropic_polar_angle(jax.random.fold_in(inclination_key, 1), n_samples)
        )

        spin_1x, spin_1y, spin_1z = self._sample_spin_components_with_max(spin1_key, n_samples, self._bh_chi_max)
        spin_2x, spin_2y, spin_2z = self._sample_spin_components_with_max(spin2_key, n_samples, self._ns_chi_max)

        return {
            "detector_frame_mass_1": mass_1,
            "detector_frame_mass_2": mass_2,
            "spin_1x": spin_1x,
            "spin_1y": spin_1y,
            "spin_1z": spin_1z,
            "spin_2x": spin_2x,
            "spin_2y": spin_2y,
            "spin_2z": spin_2z,
            "lambda_1": jnp.zeros(n_samples),
            "lambda_2": jax.random.uniform(lambda_key, shape=(n_samples,), minval=0.0, maxval=self._ns_lambda_max),
            "eccentricity": jnp.zeros(n_samples),
            "distance": distance,
            "coa_phase": jax.random.uniform(phase_key, shape=(n_samples,), minval=0.0, maxval=2.0 * math.pi),
            "inclination": inclination,
            "theta_jn": theta_jn,
            "long_asc_node": jax.random.uniform(asc_node_key, shape=(n_samples,), minval=0.0, maxval=2.0 * math.pi),
            "mean_per_ano": jax.random.uniform(
                periastron_key,
                shape=(n_samples,),
                minval=0.0,
                maxval=2.0 * math.pi,
            ),
            "coa_time": jax.random.uniform(
                time_key,
                shape=(n_samples,),
                minval=self._gps_start,
                maxval=self._gps_end,
            ),
            "right_ascension": jax.random.uniform(ra_key, shape=(n_samples,), minval=0.0, maxval=2.0 * math.pi),
            "declination": self._sample_declination(dec_key, n_samples),
            "polarization_angle": jax.random.uniform(
                polarization_key,
                shape=(n_samples,),
                minval=0.0,
                maxval=math.pi,
            ),
            "redshift": self._distance_to_redshift(distance),
            "f_ref": jnp.full(n_samples, self._f_ref),
        }

    def _sample_component_masses(self, key: Array, n_samples: int) -> tuple[Array, Array]:
        """Draw BH-primary and NS-secondary masses with an optional total-mass cut."""
        if self._total_mass_max is None:
            bh_key, ns_key = jax.random.split(key)
            return (
                self._sample_uniform(bh_key, n_samples, self._bh_mass_min, self._bh_mass_max),
                self._sample_uniform(ns_key, n_samples, self._ns_mass_min, self._ns_mass_max),
            )

        if n_samples == 0:
            empty = jnp.empty((0,))
            return empty, empty

        accepted_bh_masses: list[Array] = []
        accepted_ns_masses: list[Array] = []
        remaining = n_samples
        current_key = key

        while remaining > 0:
            current_key, bh_key, ns_key = jax.random.split(current_key, 3)
            batch_size = max(remaining * 4, 256)
            batch_bh_masses = self._sample_uniform(bh_key, batch_size, self._bh_mass_min, self._bh_mass_max)
            batch_ns_masses = self._sample_uniform(ns_key, batch_size, self._ns_mass_min, self._ns_mass_max)
            accepted = (batch_bh_masses + batch_ns_masses) <= self._total_mass_max
            if not bool(jnp.any(accepted)):
                continue

            accepted_bh_masses.append(batch_bh_masses[accepted][:remaining])
            accepted_ns_masses.append(batch_ns_masses[accepted][:remaining])
            remaining -= int(accepted_bh_masses[-1].shape[0])

        return jnp.concatenate(accepted_bh_masses), jnp.concatenate(accepted_ns_masses)

    def _sample_spin_components_with_max(
        self, key: Array, n_samples: int, chi_max: float
    ) -> tuple[Array, Array, Array]:
        """Draw dimensionless spin components with a component-specific bound."""
        return self._sample_spin_components(key, n_samples, chi_max=chi_max)
