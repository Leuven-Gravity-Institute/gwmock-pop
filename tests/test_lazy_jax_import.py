"""Importing gwmock_pop must not import JAX.

``import jax`` is where a process that only wanted to read a population file
starts paying for a numerical runtime it never uses: about 240 ms of import
time, and the first JAX computation after it starts XLA's thread pools, which
makes the process unsafe to fork. See the invariant note in
``gwmock_pop/__init__.py``.

The check is on ``sys.modules`` rather than on a thread count, because the
threads are a downstream consequence: ``import jax`` on its own starts none of
them, so a thread count cannot tell a deferred import from an eager one.

The assertions run in subprocesses because the rest of this suite imports JAX,
so an in-process ``sys.modules`` check would pass or fail on collection order.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

#: Modules in the import closure of ``gwmock_pop`` that had to defer JAX.
LAZY_MODULES = (
    "gwmock_pop",
    "gwmock_pop._precision",
    "gwmock_pop.cli.inspect",
    "gwmock_pop.cli.main",
    "gwmock_pop.loaders.file_loader",
    "gwmock_pop.mixins.random",
    "gwmock_pop.protocols.simulator",
    "gwmock_pop.rng.rng",
    "gwmock_pop.simulators.graph",
    "gwmock_pop.simulators.mixture",
    "gwmock_pop.simulators.poisson_event",
    "gwmock_pop.simulators.simulator",
    "gwmock_pop.validation",
)


def _run(script: str) -> str:
    """Run *script* in a clean interpreter and return its stdout."""
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


@pytest.mark.parametrize("module_name", LAZY_MODULES)
def test_module_imports_without_jax(module_name):
    """Importing the module leaves ``jax`` and ``jaxlib`` out of ``sys.modules``."""
    loaded = _run(
        f"""
        import sys

        import {module_name}  # noqa: F401

        print(sorted(m for m in sys.modules if m.split(".")[0] in {{"jax", "jaxlib"}}))
        """
    )
    assert loaded == "[]"
