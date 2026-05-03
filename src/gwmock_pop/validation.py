"""Physical-consistency validators for sampled populations."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping

import jax.numpy as jnp
from jax import Array

_MASS_1_KEYS = ("detector_frame_mass_1", "mass_1", "m1")
_MASS_2_KEYS = ("detector_frame_mass_2", "mass_2", "m2")
_ORDERED_MASS_1_KEYS = ("mass_1", "m1")
_ORDERED_MASS_2_KEYS = ("mass_2", "m2")
_TWO_PI = 2.0 * math.pi
_HALF_PI = 0.5 * math.pi


def validate_sample(sample: Mapping[str, Array]) -> list[str]:
    """Return human-readable physical-consistency violations for a population sample.

    The validator checks only the recognized parameters present in ``sample`` and
    ignores extra keys. Mass-ordering checks accept either the package's
    canonical detector-frame mass names or the shorter ``m1``/``m2`` aliases.
    """
    violations: list[str] = []

    _check_positive(violations, sample, label="m1", keys=_MASS_1_KEYS)
    _check_positive(violations, sample, label="m2", keys=_MASS_2_KEYS)
    _check_mass_ordering(violations, sample)

    _check_simple_constraint(
        violations,
        sample,
        key="spin_1z",
        requirement="must lie in [-1, 1]",
        invalid_mask_fn=lambda values: jnp.abs(values) > 1.0,
    )
    _check_simple_constraint(
        violations,
        sample,
        key="spin_2z",
        requirement="must lie in [-1, 1]",
        invalid_mask_fn=lambda values: jnp.abs(values) > 1.0,
    )
    _check_simple_constraint(
        violations,
        sample,
        key="distance",
        requirement="must be > 0",
        invalid_mask_fn=lambda values: values <= 0.0,
    )
    _check_simple_constraint(
        violations,
        sample,
        key="redshift",
        requirement="must be >= 0",
        invalid_mask_fn=lambda values: values < 0.0,
    )
    _check_simple_constraint(
        violations,
        sample,
        key="right_ascension",
        requirement="must lie in [0, 2*pi]",
        invalid_mask_fn=lambda values: (values < 0.0) | (values > _TWO_PI),
    )
    _check_simple_constraint(
        violations,
        sample,
        key="declination",
        requirement="must lie in [-pi/2, pi/2]",
        invalid_mask_fn=lambda values: (values < -_HALF_PI) | (values > _HALF_PI),
    )

    return violations


def _check_positive(violations: list[str], sample: Mapping[str, Array], *, label: str, keys: tuple[str, ...]) -> None:
    """Check positivity for the first matching key."""
    values = _find_first_present(sample, keys)
    if values is not None:
        _append_mask_violation(
            violations,
            values,
            label=label,
            invalid_mask=values[1] <= 0.0,
            requirement="must be > 0",
        )


def _check_mass_ordering(violations: list[str], sample: Mapping[str, Array]) -> None:
    """Check m1 >= m2 only for explicitly ordered mass aliases."""
    mass_1 = _find_first_present(sample, _ORDERED_MASS_1_KEYS)
    mass_2 = _find_first_present(sample, _ORDERED_MASS_2_KEYS)
    if mass_1 is None or mass_2 is None:
        return

    mass_1_key, mass_1_values = mass_1
    mass_2_key, mass_2_values = mass_2
    if mass_1_values.shape != mass_2_values.shape:
        violations.append(
            "m1 and m2 must have matching shapes; "
            f"got {mass_1_key} shape {mass_1_values.shape} and {mass_2_key} shape {mass_2_values.shape}."
        )
        return

    delta = mass_1_values - mass_2_values
    if bool(jnp.any(delta < 0.0)):
        violations.append(
            f"m1 ({mass_1_key}) must be >= m2 ({mass_2_key}); "
            f"offending (m1 - m2) range {_format_range(delta[delta < 0.0])}."
        )


def _check_simple_constraint(
    violations: list[str],
    sample: Mapping[str, Array],
    *,
    key: str,
    requirement: str,
    invalid_mask_fn: Callable[[Array], Array],
) -> None:
    """Check a single named parameter against a predicate."""
    values = _get_values(sample, key)
    if values is not None:
        _append_mask_violation(
            violations,
            (key, values),
            label=key,
            invalid_mask=invalid_mask_fn(values),
            requirement=requirement,
        )


def _find_first_present(sample: Mapping[str, Array], keys: tuple[str, ...]) -> tuple[str, Array] | None:
    """Return the first matching sample entry for a list of candidate keys."""
    for key in keys:
        values = _get_values(sample, key)
        if values is not None:
            return key, values
    return None


def _get_values(sample: Mapping[str, Array], key: str) -> Array | None:
    """Normalize one sample entry to a flattened JAX array."""
    if key not in sample:
        return None
    return jnp.ravel(jnp.asarray(sample[key]))


def _append_mask_violation(
    violations: list[str],
    field: tuple[str, Array],
    *,
    label: str,
    invalid_mask: Array,
    requirement: str,
) -> None:
    """Append a formatted violation when any values fail the mask."""
    if bool(jnp.any(invalid_mask)):
        key, values = field
        invalid_values = values[invalid_mask]
        violations.append(f"{label} ({key}) {requirement}; offending range {_format_range(invalid_values)}.")


def _format_range(values: Array) -> str:
    """Format the numeric range of the offending values."""
    minimum = float(jnp.min(values))
    maximum = float(jnp.max(values))
    return f"[{minimum:.6g}, {maximum:.6g}]"
