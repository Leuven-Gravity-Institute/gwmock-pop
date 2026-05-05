"""Tests for NumPy coercion helpers."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from gwmock_pop import coerce_to_numpy


def test_coerce_to_numpy_converts_array_columns_to_ndarrays() -> None:
    """Batch population columns are materialized as NumPy arrays."""
    record = {
        "detector_frame_mass_1": jnp.array([30.0, 32.0]),
        "detector_frame_mass_2": np.array([20.0, 21.0]),
    }

    result = coerce_to_numpy(record)

    assert isinstance(result["detector_frame_mass_1"], np.ndarray)
    assert isinstance(result["detector_frame_mass_2"], np.ndarray)
    np.testing.assert_allclose(result["detector_frame_mass_1"], np.array([30.0, 32.0]))
    np.testing.assert_allclose(result["detector_frame_mass_2"], np.array([20.0, 21.0]))


def test_coerce_to_numpy_unwraps_scalar_arrays_and_preserves_plain_values() -> None:
    """Scalar array entries become Python scalars, while plain values pass through."""
    record = {
        "redshift": jnp.array(0.1),
        "coa_time": np.float64(1000.0),
        "source_type": "bbh",
    }

    result = coerce_to_numpy(record)

    assert isinstance(result["redshift"], float)
    assert isinstance(result["coa_time"], float)
    assert result["redshift"] == pytest.approx(0.1)
    assert result["coa_time"] == pytest.approx(1000.0)
    assert result["source_type"] == "bbh"
