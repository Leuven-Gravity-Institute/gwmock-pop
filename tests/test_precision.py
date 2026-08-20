"""Tests for the default 64-bit precision configuration."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import jax.numpy as jnp
import numpy as np
import pytest

from gwmock_pop import CBCSimulator

GPS_MINIMUM = 1_577_491_250.0
GPS_MAXIMUM = 1_577_491_320.0


def test_import_enables_x64_by_default():
    """Importing gwmock_pop enables 64-bit JAX floats.

    This module imports ``jax.numpy`` above, so it covers the branch where JAX
    is already imported when ``gwmock_pop`` configures the flag.
    """
    assert jnp.asarray(1.0).dtype == jnp.float64


def test_x64_enabled_when_gwmock_pop_is_imported_before_jax():
    """The other branch: gwmock_pop is imported first and JAX arrives later.

    Importing ``gwmock_pop`` no longer imports JAX, so the flag is requested
    through ``JAX_ENABLE_X64`` and only takes effect when JAX is imported. Run
    in a subprocess because import order is process-global.
    """
    script = textwrap.dedent(
        """
        import gwmock_pop  # noqa: F401

        import jax.numpy as jnp

        assert jnp.asarray(1.0).dtype == jnp.float64, jnp.asarray(1.0).dtype
        """
    )
    # Both variables are dropped, not just the one under test: inheriting an
    # exported GWMOCK_POP_DISABLE_X64 would turn the default path into the
    # opt-out path and fail the assertion for a reason the test is not about.
    env = {k: v for k, v in os.environ.items() if k not in {"JAX_ENABLE_X64", "GWMOCK_POP_DISABLE_X64"}}
    subprocess.run([sys.executable, "-c", script], env=env, check=True)  # noqa: S603


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


@pytest.mark.parametrize("import_jax_first", [False, True], ids=["jax-after", "jax-before"])
def test_disable_x64_overrides_an_inherited_jax_enable_x64(import_jax_first):
    """The opt-out wins over ``JAX_ENABLE_X64=1`` in the environment.

    Importing gwmock_pop exports ``JAX_ENABLE_X64=1`` (that is how the default
    is applied without importing JAX), so a child process inherits it. The
    opt-out has to override the variable, not merely decline to set it, or it
    would be ineffective in exactly the processes most likely to use it.
    Both import orders are covered: the variable is read when JAX is imported,
    so only the second order goes through the configuration API.
    """
    imports = (
        "import jax.numpy as jnp\nimport gwmock_pop  # noqa: F401"
        if import_jax_first
        else "import gwmock_pop  # noqa: F401\nimport jax.numpy as jnp"
    )
    script = f"{imports}\nassert jnp.asarray(1.0).dtype == jnp.float32, jnp.asarray(1.0).dtype\n"
    env = dict(os.environ, GWMOCK_POP_DISABLE_X64="1", JAX_ENABLE_X64="1")

    subprocess.run([sys.executable, "-c", script], env=env, check=True)  # noqa: S603
