"""Lightweight CBC prior simulator."""

from __future__ import annotations

import math
from collections.abc import Mapping

import jax
import jax.numpy as jnp
from jax import Array

from gwmock_pop.simulators.bbh.base import BBHSimulator

_TWO_PI = 2.0 * math.pi
_PLANCK18_H0_KM_S_MPC = 67.66
_SPEED_OF_LIGHT_KM_S = 299792.458


class CBCPriorSimulator(BBHSimulator):
    """Sample CBC source parameters from lightweight analytic priors.

    The simulator is intentionally independent of the graph engine and provides a
    small configurable prior surface for testing, injection studies, and simple
    baseline populations. Distances are sampled with a Euclidean
    ``p(d) proportional to d^2`` approximation on ``[0, d_max]`` and converted to
    redshift via the low-redshift Hubble-law relation
    ``z ~= H0 * d / c`` using the Planck18 value ``H0 = 67.66 km s^-1 Mpc^-1``.

    Component masses are sampled independently and are not reordered after the
    draw. When ``total_mass_max`` is set, the simulator rejects samples with
    ``m1 + m2 > total_mass_max`` and keeps drawing until enough accepted pairs are
    collected.

    Args:
        source_type: Non-empty routing key used by downstream orchestration.
            Defaults to ``"bbh"``.
        m_min: Lower bound for each component mass in solar masses.
        m_max: Upper bound for each component mass in solar masses.
        d_max: Maximum luminosity distance in Mpc for the Euclidean-volume prior.
        chi_max: Maximum dimensionless spin magnitude.
        aligned_spins: If ``True``, only the ``z`` spin components are populated
            and the in-plane spin components are set to zero.
        gps_start: Lower bound for the coalescence-time prior.
        gps_end: Upper bound for the coalescence-time prior.
        total_mass_max: Optional upper bound on the sum of the two component
            masses.
        f_ref: Constant reference frequency assigned to every sample.
        seed: Optional integer seed forwarded to the parent :class:`BBHSimulator`
            so :class:`~gwmock_pop.mixins.random.RandomMixin` initializes its
            RNG manager for callers that rely on mixin-based randomness.
    """

    def __init__(  # noqa: PLR0913
        self,
        source_type: str = "bbh",
        *,
        m_min: float = 5.0,
        m_max: float = 100.0,
        d_max: float = 5_000.0,
        chi_max: float = 0.99,
        aligned_spins: bool = False,
        gps_start: float = 0.0,
        gps_end: float = 1.0,
        total_mass_max: float | None = None,
        f_ref: float = 20.0,
        seed: int | None = None,
    ) -> None:
        """Initialize the prior simulator."""
        super().__init__(seed=seed)
        self._source_type = source_type
        self._m_min = float(m_min)
        self._m_max = float(m_max)
        self._d_max = float(d_max)
        self._chi_max = float(chi_max)
        self._aligned_spins = aligned_spins
        self._gps_start = float(gps_start)
        self._gps_end = float(gps_end)
        self._total_mass_max = None if total_mass_max is None else float(total_mass_max)
        self._f_ref = float(f_ref)
        self._validate_configuration()

    @property
    def source_type(self) -> str:
        """Return the routing key for the sampled source population."""
        return self._source_type

    def _validate_configuration(self) -> None:
        """Validate constructor arguments."""
        if not self._source_type.strip():
            raise ValueError("source_type must be a non-empty string.")
        for name, value in (
            ("m_min", self._m_min),
            ("m_max", self._m_max),
            ("d_max", self._d_max),
            ("chi_max", self._chi_max),
            ("gps_start", self._gps_start),
            ("gps_end", self._gps_end),
            ("f_ref", self._f_ref),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
        if self._total_mass_max is not None and not math.isfinite(self._total_mass_max):
            raise ValueError("total_mass_max must be finite when provided.")
        if self._m_min <= 0.0:
            raise ValueError("m_min must be positive.")
        if self._m_max <= self._m_min:
            raise ValueError("m_max must be greater than m_min.")
        if self._d_max <= 0.0:
            raise ValueError("d_max must be positive.")
        if not (0.0 <= self._chi_max <= 1.0):
            raise ValueError("chi_max must be in [0, 1].")
        if self._gps_end <= self._gps_start:
            raise ValueError("gps_end must be greater than gps_start.")
        if self._total_mass_max is not None and self._total_mass_max <= 2.0 * self._m_min:
            raise ValueError("total_mass_max must be greater than 2 * m_min to admit any samples.")
        if self._f_ref <= 0.0:
            raise ValueError("f_ref must be positive.")

    def _simulate_impl(self, n_samples: int, *, seed: int | None = None) -> Mapping[str, Array]:
        """Draw ``n_samples`` CBC prior samples using JAX PRNGs."""
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
        ) = jax.random.split(key, 12)

        mass_1, mass_2 = self._sample_component_masses(mass_key, n_samples)
        distance = self._sample_distance(distance_key, n_samples)
        inclination = self._sample_isotropic_polar_angle(inclination_key, n_samples)
        theta_jn = (
            inclination
            if self._aligned_spins
            else self._sample_isotropic_polar_angle(jax.random.fold_in(inclination_key, 1), n_samples)
        )

        spin_1x, spin_1y, spin_1z = self._sample_spin_components(spin1_key, n_samples)
        spin_2x, spin_2y, spin_2z = self._sample_spin_components(spin2_key, n_samples)

        return {
            "detector_frame_mass_1": mass_1,
            "detector_frame_mass_2": mass_2,
            "spin_1x": spin_1x,
            "spin_1y": spin_1y,
            "spin_1z": spin_1z,
            "spin_2x": spin_2x,
            "spin_2y": spin_2y,
            "spin_2z": spin_2z,
            "eccentricity": jnp.zeros(n_samples),
            "distance": distance,
            "coa_phase": jax.random.uniform(phase_key, shape=(n_samples,), minval=0.0, maxval=_TWO_PI),
            "inclination": inclination,
            "theta_jn": theta_jn,
            "long_asc_node": jax.random.uniform(asc_node_key, shape=(n_samples,), minval=0.0, maxval=_TWO_PI),
            "mean_per_ano": jax.random.uniform(periastron_key, shape=(n_samples,), minval=0.0, maxval=_TWO_PI),
            "coa_time": jax.random.uniform(
                time_key,
                shape=(n_samples,),
                minval=self._gps_start,
                maxval=self._gps_end,
            ),
            "right_ascension": jax.random.uniform(ra_key, shape=(n_samples,), minval=0.0, maxval=_TWO_PI),
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
        """Draw component masses with an optional total-mass rejection cut."""
        if self._total_mass_max is None:
            key_1, key_2 = jax.random.split(key)
            return self._sample_uniform(key_1, n_samples, self._m_min, self._m_max), self._sample_uniform(
                key_2, n_samples, self._m_min, self._m_max
            )

        if n_samples == 0:
            empty = jnp.empty((0,))
            return empty, empty

        accepted_mass_1: list[Array] = []
        accepted_mass_2: list[Array] = []
        remaining = n_samples
        current_key = key

        while remaining > 0:
            current_key, key_1, key_2 = jax.random.split(current_key, 3)
            batch_size = max(remaining * 4, 256)
            batch_mass_1 = self._sample_uniform(key_1, batch_size, self._m_min, self._m_max)
            batch_mass_2 = self._sample_uniform(key_2, batch_size, self._m_min, self._m_max)
            accepted = (batch_mass_1 + batch_mass_2) <= self._total_mass_max
            if not bool(jnp.any(accepted)):
                continue

            accepted_mass_1.append(batch_mass_1[accepted][:remaining])
            accepted_mass_2.append(batch_mass_2[accepted][:remaining])
            remaining -= int(accepted_mass_1[-1].shape[0])

        return jnp.concatenate(accepted_mass_1), jnp.concatenate(accepted_mass_2)

    def _sample_distance(self, key: Array, n_samples: int) -> Array:
        """Draw luminosity distance from a Euclidean-volume prior."""
        unit_uniform = jax.random.uniform(key, shape=(n_samples,))
        return self._d_max * jnp.cbrt(unit_uniform)

    @staticmethod
    def _sample_uniform(key: Array, n_samples: int, lower: float, upper: float) -> Array:
        """Draw from a uniform distribution on ``[lower, upper)``."""
        return jax.random.uniform(key, shape=(n_samples,), minval=lower, maxval=upper)

    @staticmethod
    def _sample_declination(key: Array, n_samples: int) -> Array:
        """Draw an isotropic declination in ``[-pi/2, pi/2]``."""
        sin_declination = jax.random.uniform(key, shape=(n_samples,), minval=-1.0, maxval=1.0)
        return jnp.arcsin(sin_declination)

    @staticmethod
    def _sample_isotropic_polar_angle(key: Array, n_samples: int) -> Array:
        """Draw a polar angle with ``cos(theta)`` uniform on ``[-1, 1]``."""
        cos_theta = jax.random.uniform(key, shape=(n_samples,), minval=-1.0, maxval=1.0)
        return jnp.arccos(cos_theta)

    def _sample_spin_components(
        self, key: Array, n_samples: int, *, chi_max: float | None = None
    ) -> tuple[Array, Array, Array]:
        """Draw dimensionless spin components."""
        magnitude_key, orientation_key, azimuth_key, sign_key = jax.random.split(key, 4)
        spin_bound = self._chi_max if chi_max is None else chi_max
        magnitude = jax.random.uniform(magnitude_key, shape=(n_samples,), minval=0.0, maxval=spin_bound)

        if self._aligned_spins:
            sign = jnp.where(jax.random.bernoulli(sign_key, p=0.5, shape=(n_samples,)), 1.0, -1.0)
            zeros = jnp.zeros(n_samples)
            return zeros, zeros, sign * magnitude

        polar = self._sample_isotropic_polar_angle(orientation_key, n_samples)
        azimuth = jax.random.uniform(azimuth_key, shape=(n_samples,), minval=0.0, maxval=_TWO_PI)
        sin_polar = jnp.sin(polar)
        return (
            magnitude * sin_polar * jnp.cos(azimuth),
            magnitude * sin_polar * jnp.sin(azimuth),
            magnitude * jnp.cos(polar),
        )

    @staticmethod
    def _distance_to_redshift(distance: Array) -> Array:
        """Approximate redshift from luminosity distance using the Hubble law."""
        return distance * (_PLANCK18_H0_KM_S_MPC / _SPEED_OF_LIGHT_KM_S)
