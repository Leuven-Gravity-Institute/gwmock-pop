"""Tests for the Madau-Dickinson redshift rate model."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from gwmock_pop.distributions.madau_dickinson import (
    madau_dickinson_rate,
    madau_dickinson_redshift_pdf,
)


def test_rate_is_unity_at_zero() -> None:
    """The rate shape is normalised so that ``psi(0) = 1``."""
    assert jnp.isclose(madau_dickinson_rate(0.0), 1.0, atol=1e-12)


def test_rate_matches_madau_dickinson_2014_value_at_z2() -> None:
    """psi(2) reproduces the Madau & Dickinson (2014) shape (external hand calc).

    psi(2) = C * 3^2.7 / (1 + (3/2.9)^5.6) with C = 1 + 2.9^-5.6 ~= 8.815.
    """
    assert jnp.isclose(madau_dickinson_rate(2.0), 8.815, atol=0.01)


def test_rate_peaks_near_redshift_1p86() -> None:
    """The default rate peaks at the analytic turnover z ~= 1.86."""
    grid = jnp.linspace(0.0, 6.0, 6001)
    peak_redshift = float(grid[int(jnp.argmax(madau_dickinson_rate(grid)))])
    assert 1.80 < peak_redshift < 1.95, peak_redshift


def test_rate_is_vectorised() -> None:
    """The rate accepts and returns arrays elementwise."""
    grid = jnp.linspace(0.0, 4.0, 17)
    values = madau_dickinson_rate(grid)
    assert values.shape == grid.shape
    assert np.all(np.asarray(values) > 0.0)


def test_redshift_pdf_normalises_to_unity() -> None:
    """The redshift density integrates to one over ``[z_min, z_max]``."""
    z_max = 2.0
    grid = jnp.linspace(0.0, z_max, 4096)
    density = madau_dickinson_redshift_pdf(grid, z_max=z_max)
    integral = jnp.trapezoid(y=density, x=grid)
    assert jnp.isclose(integral, 1.0, atol=1e-3), float(integral)


def test_redshift_pdf_is_zero_outside_support() -> None:
    """The density vanishes outside ``[z_min, z_max]``."""
    density = madau_dickinson_redshift_pdf(jnp.array([-0.5, 0.05, 1.5, 2.5]), z_min=0.1, z_max=2.0)
    assert float(density[0]) == 0.0
    assert float(density[1]) == 0.0  # below z_min
    assert float(density[2]) > 0.0
    assert float(density[3]) == 0.0  # above z_max


def test_redshift_pdf_rejects_degenerate_range() -> None:
    """The density raises when ``z_max`` does not exceed ``z_min``."""
    with pytest.raises(ValueError, match="z_max must be strictly greater"):
        madau_dickinson_redshift_pdf(jnp.array([0.5]), z_min=1.0, z_max=1.0)
