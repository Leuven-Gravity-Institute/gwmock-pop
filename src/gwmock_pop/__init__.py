"""Top-level package for gwmock_pop."""

from __future__ import annotations

from gwmock_pop.protocols import ExternalPopulationLoader, GWPopSimulator
from gwmock_pop.version import __version__

__all__ = ["ExternalPopulationLoader", "GWPopSimulator", "__version__"]
