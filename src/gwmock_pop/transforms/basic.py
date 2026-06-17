"""Basic transform helpers for graph-based simulator configs."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from gwmock_pop.cosmology.flat_lambda_cdm import (
    DEFAULT_LOOKUP_GRID_SIZE,
    DEFAULT_MAX_REDSHIFT,
    PLANCK18_H0_KM_S_MPC,
    PLANCK18_OMEGA_M,
    build_distance_lookup,
    compute_redshift_from_luminosity_distance,
)


def _resolve_output_shape(reference: Array | None, n_samples: int | None) -> tuple[int, ...]:
    """Resolve the output shape for stochastic transform nodes."""
    if reference is not None:
        shape = jnp.asarray(reference).shape
        if not shape:
            raise ValueError("reference must be array-like with at least one dimension.")
        return shape

    if n_samples is None:
        raise ValueError("Either reference or n_samples must be provided.")
    if n_samples < 0:
        raise ValueError(f"n_samples must be >= 0, got {n_samples}.")
    return (n_samples,)


def constant_like(reference: Array, value: float) -> Array:
    """Return a constant array with the same shape as ``reference``.

    Args:
        reference: Array providing the target shape.
        value: Scalar value to broadcast across the output.

    Returns:
        An array with the same shape as ``reference`` filled with ``value``.
    """
    dtype = jnp.result_type(reference, jnp.asarray(value))
    return jnp.full(shape=reference.shape, fill_value=value, dtype=dtype)


def multiply(left: Array, right: Array) -> Array:
    """Multiply two array-like inputs elementwise.

    Args:
        left: Left multiplicand.
        right: Right multiplicand.

    Returns:
        Elementwise product of ``left`` and ``right``.
    """
    return jnp.asarray(left) * jnp.asarray(right)


def maximum(left: Array, right: Array) -> Array:
    """Return the elementwise maximum of two array-like inputs."""
    return jnp.maximum(jnp.asarray(left), jnp.asarray(right))


def minimum(left: Array, right: Array) -> Array:
    """Return the elementwise minimum of two array-like inputs."""
    return jnp.minimum(jnp.asarray(left), jnp.asarray(right))


def identity(*, value: Array) -> Array:
    """Return ``value`` unchanged.

    Useful for aliasing one parameter to another in a graph config (e.g. setting
    ``theta_jn`` equal to ``inclination`` for aligned-spin populations).

    Args:
        value: Array-like value to pass through.

    Returns:
        ``value`` as a JAX array.
    """
    return jnp.asarray(value)


def take_row(*, matrix: Array, index: int) -> Array:
    """Select one row from a 2-D array.

    Used to split a jointly sampled ``(k, n_samples)`` array (e.g. a component
    mass pair) into individual 1-D parameter columns.

    Args:
        matrix: Array whose leading axis is indexed.
        index: Row index to select.

    Returns:
        The selected row as a 1-D array.
    """
    return jnp.asarray(matrix)[index]


def isotropic_spin_orientation(*, key: Array, reference: Array | None = None, n_samples: int | None = None) -> Array:
    """Draw polar angles with ``cos(tilt)`` uniform on ``[-1, 1]``."""
    shape = _resolve_output_shape(reference=reference, n_samples=n_samples)
    cos_tilt = jax.random.uniform(key, shape=shape, minval=-1.0, maxval=1.0)
    return jnp.arccos(cos_tilt)


def isotropic_declination(*, key: Array, reference: Array | None = None, n_samples: int | None = None) -> Array:
    """Draw declinations with ``sin(declination)`` uniform on ``[-1, 1]``.

    Produces an isotropic sky distribution in declination, i.e. angles in
    ``[-pi/2, pi/2]`` with ``arcsin`` of a uniform variate.

    Args:
        key: JAX PRNG key.
        reference: Optional array providing the output shape.
        n_samples: Number of samples, used when ``reference`` is omitted.

    Returns:
        Declination angles in radians on ``[-pi/2, pi/2]``.
    """
    shape = _resolve_output_shape(reference=reference, n_samples=n_samples)
    sin_declination = jax.random.uniform(key, shape=shape, minval=-1.0, maxval=1.0)
    return jnp.arcsin(sin_declination)


def spherical_to_cartesian_x(*, magnitude: Array, tilt: Array, azimuth: Array) -> Array:
    """Project spherical coordinates onto the Cartesian ``x`` component."""
    return jnp.asarray(magnitude) * jnp.sin(jnp.asarray(tilt)) * jnp.cos(jnp.asarray(azimuth))


def spherical_to_cartesian_y(*, magnitude: Array, tilt: Array, azimuth: Array) -> Array:
    """Project spherical coordinates onto the Cartesian ``y`` component."""
    return jnp.asarray(magnitude) * jnp.sin(jnp.asarray(tilt)) * jnp.sin(jnp.asarray(azimuth))


def spherical_to_cartesian_z(*, magnitude: Array, tilt: Array) -> Array:
    """Project spherical coordinates onto the Cartesian ``z`` component."""
    return jnp.asarray(magnitude) * jnp.cos(jnp.asarray(tilt))


def gaussian_chi_eff(  # noqa: PLR0913
    *,
    key: Array | None = None,
    reference: Array | None = None,
    n_samples: int | None = None,
    mean: float = 0.0,
    sigma: float = 0.1,
    minimum: float = -1.0,
    maximum: float = 1.0,
    chi_eff: Array | None = None,
    component: str | None = None,
    mass_1: Array | None = None,
    mass_2: Array | None = None,
    spin_magnitude_1: Array | None = None,
    spin_magnitude_2: Array | None = None,
) -> Array:
    """Draw or project an effective aligned-spin distribution.

    When ``component`` is omitted, this returns ``chi_eff`` samples drawn from a
    clipped Gaussian. When ``component`` is ``"primary"`` or ``"secondary"``,
    the function projects the supplied or sampled ``chi_eff`` values onto the
    requested spin-z component while respecting the available spin magnitudes.
    """
    if sigma <= 0.0:
        raise ValueError(f"sigma must be > 0, got {sigma}.")
    if minimum > maximum:
        raise ValueError(f"minimum must be <= maximum, got {minimum} > {maximum}.")

    if chi_eff is None:
        if key is None:
            raise ValueError("key is required when chi_eff is not provided.")
        shape = _resolve_output_shape(reference=reference, n_samples=n_samples)
        chi_eff_values = mean + sigma * jax.random.normal(key, shape=shape)
    else:
        chi_eff_values = jnp.asarray(chi_eff)

    chi_eff_values = jnp.clip(jnp.asarray(chi_eff_values), minimum, maximum)
    if component is None:
        return chi_eff_values

    if component not in {"primary", "secondary"}:
        raise ValueError(f"component must be 'primary' or 'secondary', got {component!r}.")
    required = {
        "mass_1": mass_1,
        "mass_2": mass_2,
        "spin_magnitude_1": spin_magnitude_1,
        "spin_magnitude_2": spin_magnitude_2,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        missing_args = ", ".join(missing)
        raise ValueError(f"Projection requires {missing_args}.")

    mass_1_array = jnp.asarray(mass_1)
    mass_2_array = jnp.asarray(mass_2)
    spin_magnitude_1_array = jnp.asarray(spin_magnitude_1)
    spin_magnitude_2_array = jnp.asarray(spin_magnitude_2)

    total_mass = mass_1_array + mass_2_array
    aligned_capacity = mass_1_array * spin_magnitude_1_array + mass_2_array * spin_magnitude_2_array
    max_abs_chi_eff = jnp.where(total_mass > 0.0, aligned_capacity / total_mass, 0.0)
    bounded_chi_eff = jnp.clip(chi_eff_values, -max_abs_chi_eff, max_abs_chi_eff)

    spin_magnitude = spin_magnitude_1_array if component == "primary" else spin_magnitude_2_array
    projected = jnp.where(
        aligned_capacity > 0.0,
        bounded_chi_eff * total_mass * spin_magnitude / aligned_capacity,
        0.0,
    )
    return projected


def beta_spin_magnitude(  # noqa: PLR0913
    *,
    alpha: float,
    beta: float,
    key: Array,
    reference: Array | None = None,
    n_samples: int | None = None,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> Array:
    """Draw spin magnitudes from a rescaled Beta distribution."""
    if alpha <= 0.0:
        raise ValueError(f"alpha must be > 0, got {alpha}.")
    if beta <= 0.0:
        raise ValueError(f"beta must be > 0, got {beta}.")
    if minimum > maximum:
        raise ValueError(f"minimum must be <= maximum, got {minimum} > {maximum}.")

    shape = _resolve_output_shape(reference=reference, n_samples=n_samples)
    unit_interval_samples = jax.random.beta(key, a=alpha, b=beta, shape=shape)
    return minimum + (maximum - minimum) * unit_interval_samples


def luminosity_distance_to_redshift(
    luminosity_distance: Array,
    *,
    hubble_constant: float = PLANCK18_H0_KM_S_MPC,
    omega_m: float = PLANCK18_OMEGA_M,
    max_redshift: float = DEFAULT_MAX_REDSHIFT,
    n_grid: int = DEFAULT_LOOKUP_GRID_SIZE,
) -> Array:
    """Convert luminosity distance to redshift with a flat-ΛCDM lookup.

    Args:
        luminosity_distance: Luminosity distance in Mpc.
        hubble_constant: Hubble constant in km / s / Mpc.
        omega_m: Matter density.
        max_redshift: Largest redshift supported by the lookup table.
        n_grid: Number of tabulation points.

    Returns:
        Redshift inferred from ``luminosity_distance``.
    """
    return compute_redshift_from_luminosity_distance(
        luminosity_distance=luminosity_distance,
        hubble_constant=hubble_constant,
        omega_m=omega_m,
        max_redshift=max_redshift,
        n_grid=n_grid,
    )


def redshift_to_luminosity_distance(
    redshift: Array,
    *,
    hubble_constant: float = PLANCK18_H0_KM_S_MPC,
    omega_m: float = PLANCK18_OMEGA_M,
    max_redshift: float = DEFAULT_MAX_REDSHIFT,
    n_grid: int = DEFAULT_LOOKUP_GRID_SIZE,
) -> Array:
    """Convert redshift to luminosity distance with a flat-ΛCDM lookup.

    Inverse of :func:`luminosity_distance_to_redshift`; lets a graph sample
    redshift directly (e.g. via ``madau_dickinson_redshift``) and derive the
    luminosity distance the signal pipeline needs.

    Args:
        redshift: Redshift.
        hubble_constant: Hubble constant in km / s / Mpc.
        omega_m: Matter density.
        max_redshift: Largest redshift supported by the lookup table.
        n_grid: Number of tabulation points.

    Returns:
        Luminosity distance in Mpc inferred from ``redshift``.
    """
    redshift_grid, _, luminosity_distance_grid = build_distance_lookup(
        hubble_constant=hubble_constant,
        omega_m=omega_m,
        max_redshift=max_redshift,
        n_grid=n_grid,
    )
    return jnp.interp(jnp.asarray(redshift), redshift_grid, luminosity_distance_grid)
