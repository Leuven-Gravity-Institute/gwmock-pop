"""Tests for transform helper functions."""

from __future__ import annotations

import math

import jax
import numpy as np
import pytest
from scipy.stats import beta as scipy_beta

from gwmock_pop.transforms import (
    beta_spin_magnitude,
    gaussian_chi_eff,
    isotropic_spin_orientation,
)
from gwmock_pop.utils.import_utils import import_from_string


def _ks_p_value(samples: np.ndarray, cdf) -> float:
    """Return the asymptotic one-sample Kolmogorov-Smirnov p-value."""
    sample = np.sort(np.asarray(samples, dtype=float))
    n_samples = sample.size
    empirical_upper = np.arange(1, n_samples + 1) / n_samples
    empirical_lower = np.arange(0, n_samples) / n_samples
    theoretical = np.clip(cdf(sample), 0.0, 1.0)
    statistic = max(
        float(np.max(empirical_upper - theoretical)),
        float(np.max(theoretical - empirical_lower)),
    )
    lam = (math.sqrt(n_samples) + 0.12 + 0.11 / math.sqrt(n_samples)) * statistic
    p_value = 2.0 * sum(((-1) ** (term - 1)) * math.exp(-2.0 * (lam**2) * (term**2)) for term in range(1, 101))
    return max(0.0, min(1.0, p_value))


def test_isotropic_spin_orientation_draws_uniform_cosine() -> None:
    """The sampled tilt angles are isotropic in cosine."""
    tilt = isotropic_spin_orientation(key=jax.random.PRNGKey(0), n_samples=10_000)
    cosine = np.asarray(np.cos(np.asarray(tilt)))

    p_value = _ks_p_value(cosine, lambda x: np.clip((x + 1.0) / 2.0, 0.0, 1.0))
    assert p_value > 0.05, p_value


def test_beta_spin_magnitude_matches_beta_reference_distribution() -> None:
    """The spin magnitudes follow the configured Beta prior."""
    samples = beta_spin_magnitude(alpha=2.0, beta=5.0, key=jax.random.PRNGKey(1), n_samples=10_000)

    p_value = _ks_p_value(np.asarray(samples), scipy_beta(a=2.0, b=5.0).cdf)
    assert p_value > 0.05, p_value


def test_gaussian_chi_eff_projects_to_bounded_spin_z_components() -> None:
    """Projected spin-z components reproduce the bounded effective aligned spin."""
    mass_1 = np.asarray([30.0, 35.0, 40.0])
    mass_2 = np.asarray([20.0, 15.0, 10.0])
    spin_magnitude_1 = np.asarray([0.6, 0.4, 0.3])
    spin_magnitude_2 = np.asarray([0.5, 0.5, 0.2])
    chi_eff = np.asarray([0.4, -0.3, 0.9])

    spin_1z = np.asarray(
        gaussian_chi_eff(
            chi_eff=chi_eff,
            component="primary",
            mass_1=mass_1,
            mass_2=mass_2,
            spin_magnitude_1=spin_magnitude_1,
            spin_magnitude_2=spin_magnitude_2,
        )
    )
    spin_2z = np.asarray(
        gaussian_chi_eff(
            chi_eff=chi_eff,
            component="secondary",
            mass_1=mass_1,
            mass_2=mass_2,
            spin_magnitude_1=spin_magnitude_1,
            spin_magnitude_2=spin_magnitude_2,
        )
    )

    recovered = (mass_1 * spin_1z + mass_2 * spin_2z) / (mass_1 + mass_2)
    max_abs_chi_eff = (mass_1 * spin_magnitude_1 + mass_2 * spin_magnitude_2) / (mass_1 + mass_2)

    np.testing.assert_allclose(recovered, np.clip(chi_eff, -max_abs_chi_eff, max_abs_chi_eff))
    assert np.all(np.abs(spin_1z) <= spin_magnitude_1 + 1e-6)
    assert np.all(np.abs(spin_2z) <= spin_magnitude_2 + 1e-6)


def test_stochastic_transforms_require_reference_or_sample_count() -> None:
    """Random transforms need a shape source."""
    with pytest.raises(ValueError, match="Either reference or n_samples must be provided"):
        isotropic_spin_orientation(key=jax.random.PRNGKey(0))

    with pytest.raises(ValueError, match="Either reference or n_samples must be provided"):
        beta_spin_magnitude(alpha=2.0, beta=3.0, key=jax.random.PRNGKey(1))


def test_transform_exports_are_discoverable_via_default_module_lookup() -> None:
    """The graph engine can resolve the new transform names."""
    assert import_from_string("isotropic_spin_orientation", default_module="gwmock_pop.transforms") is (
        isotropic_spin_orientation
    )
    assert import_from_string("gaussian_chi_eff", default_module="gwmock_pop.transforms") is gaussian_chi_eff
    assert import_from_string("beta_spin_magnitude", default_module="gwmock_pop.transforms") is beta_spin_magnitude
