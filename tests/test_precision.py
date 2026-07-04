"""Tests for the default 64-bit precision configuration."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import jax.numpy as jnp
import numpy as np

from gwmock_pop import CBCSimulator

GPS_MINIMUM = 1_577_491_250.0
GPS_MAXIMUM = 1_577_491_320.0


def test_import_enables_x64_by_default():
    """Importing gwmock_pop enables 64-bit JAX floats."""
    assert jnp.asarray(1.0).dtype == jnp.float64


def test_gps_scale_coa_time_sampling_is_not_quantized():
    """Regression: GPS-scale coa_time samples must be distinct and in range.

    Under JAX's float32 default the spacing at ~1.6e9 s is 128 s, so uniform
    samples over a 70 s window all collapsed onto the single representable
    value 1_577_491_328.0 — identical for every event and outside the
    requested range.
    """
    simulator = CBCSimulator(
        seed=11,
        parameters={
            "coa_time": {
                "sampler": {
                    "function": "uniform",
                    "arguments": {"minimum": GPS_MINIMUM, "maximum": GPS_MAXIMUM},
                }
            }
        },
    )
    coa_time = np.asarray(simulator.simulate(16)["coa_time"])

    assert coa_time.dtype == np.float64
    assert len(np.unique(coa_time)) == len(coa_time)
    assert coa_time.min() >= GPS_MINIMUM
    assert coa_time.max() <= GPS_MAXIMUM


def test_disable_x64_env_opt_out():
    """GWMOCK_POP_DISABLE_X64=1 keeps JAX's 32-bit default.

    The x64 flag is process-global JAX state, so the opt-out is exercised in a
    subprocess with a clean interpreter.
    """
    script = textwrap.dedent(
        """
        import jax.numpy as jnp

        import gwmock_pop  # noqa: F401

        assert jnp.asarray(1.0).dtype == jnp.float32, jnp.asarray(1.0).dtype
        """
    )
    env = dict(os.environ, GWMOCK_POP_DISABLE_X64="1")
    subprocess.run([sys.executable, "-c", script], env=env, check=True)  # noqa: S603
