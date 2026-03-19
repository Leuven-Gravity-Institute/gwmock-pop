"""Top-level package for gwsim_pop."""

from __future__ import annotations

import warnings

from gwsim_pop.version import __version__

warnings.warn(
    "This package `gwsim-pop` has been renamed to `gwmock-pop`. Please update your requirements and imports.",
    DeprecationWarning,
    stacklevel=2,
)

try:
    from gwmock_pop import *  # noqa: F403
except ImportError as e:
    # This handles the edge case where the dependency installation failed
    raise ImportError("gwsim-pop requires gwmock-pop to be installed. Please run 'pip install gwmock-pop'.") from e

__all__ = ["__version__"]
