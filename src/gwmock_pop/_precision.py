"""Floating-point precision configuration for gwmock_pop.

JAX defaults to 32-bit floats. At GPS scale (~1.6e9 s) the float32 spacing is
128 s, so sampling coalescence times in float32 collapses every event onto a
coarse grid — and can even land outside the requested range. Population
catalogues are the whole point of this package, so 64-bit precision is enabled
by default when the package is imported.

Set the environment variable ``GWMOCK_POP_DISABLE_X64=1`` before importing
``gwmock_pop`` to keep JAX's 32-bit default (e.g. for GPU-throughput studies
that do not sample absolute times).

The flag is requested through JAX's own ``JAX_ENABLE_X64`` environment
variable rather than ``jax.config.update``, because reaching the config object
means importing ``jax`` — which importing ``gwmock_pop`` deliberately does not
do (see the note in :mod:`gwmock_pop`). When JAX is already imported the
environment is no longer read, so the configuration API is used as well.

Going through the environment means child processes inherit the request, so the
opt-out has to withdraw it rather than merely decline to add it — otherwise
``GWMOCK_POP_DISABLE_X64=1`` would be silently ineffective in any child of a
process that did not opt out, whichever order the two packages are imported in.
The opt-out therefore takes precedence over ``JAX_ENABLE_X64`` whoever set it:
it unsets that variable and, when JAX is already imported, turns the flag off.
Call ``jax.config.update("jax_enable_x64", True)`` yourself if you want x64
while opting out of this package's default.
"""

from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger("gwmock_pop")

_DISABLE_X64_ENV = "GWMOCK_POP_DISABLE_X64"
_JAX_X64_ENV = "JAX_ENABLE_X64"
_TRUTHY = {"1", "true", "yes", "on"}


def enable_x64_by_default() -> None:
    """Enable 64-bit JAX floats unless explicitly opted out.

    Respects ``GWMOCK_POP_DISABLE_X64`` (set to ``1``/``true``/``yes``/``on``
    to keep JAX's 32-bit default). The flag is global JAX state: it affects the
    dtype of arrays created after this call, in this package and beyond, and
    ``JAX_ENABLE_X64`` is set in ``os.environ``, so child processes spawned
    afterwards inherit the same precision. The opt-out overrides that variable
    whoever set it, so that a value inherited from a parent process cannot
    defeat it.
    """
    enable = os.environ.get(_DISABLE_X64_ENV, "").strip().lower() not in _TRUTHY

    if enable:
        os.environ[_JAX_X64_ENV] = "1"
    else:
        logger.debug("%s is set: keeping JAX's 32-bit default.", _DISABLE_X64_ENV)
        os.environ.pop(_JAX_X64_ENV, None)

    jax = sys.modules.get("jax")
    if jax is not None:
        # JAX read the environment when it was imported, so the variable above
        # came too late for this process and the config API has to be used too.
        jax.config.update("jax_enable_x64", enable)
