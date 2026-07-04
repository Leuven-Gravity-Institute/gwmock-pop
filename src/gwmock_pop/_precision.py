"""Floating-point precision configuration for gwmock_pop.

JAX defaults to 32-bit floats. At GPS scale (~1.6e9 s) the float32 spacing is
128 s, so sampling coalescence times in float32 collapses every event onto a
coarse grid — and can even land outside the requested range. Population
catalogues are the whole point of this package, so 64-bit precision is enabled
by default when the package is imported.

Set the environment variable ``GWMOCK_POP_DISABLE_X64=1`` before importing
``gwmock_pop`` to keep JAX's 32-bit default (e.g. for GPU-throughput studies
that do not sample absolute times).
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("gwmock_pop")

_DISABLE_X64_ENV = "GWMOCK_POP_DISABLE_X64"
_TRUTHY = {"1", "true", "yes", "on"}


def enable_x64_by_default() -> None:
    """Enable 64-bit JAX floats unless explicitly opted out.

    Respects ``GWMOCK_POP_DISABLE_X64`` (set to ``1``/``true``/``yes``/``on``
    to keep JAX's 32-bit default). The flag is global JAX state: it affects
    the dtype of arrays created after this call, in this package and beyond.
    """
    if os.environ.get(_DISABLE_X64_ENV, "").strip().lower() in _TRUTHY:
        logger.debug("%s is set: leaving JAX x64 mode unchanged.", _DISABLE_X64_ENV)
        return

    import jax  # noqa: PLC0415  # deferred so the opt-out path never initializes JAX config

    jax.config.update("jax_enable_x64", True)
