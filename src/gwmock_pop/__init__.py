"""Top-level package for gwmock_pop."""

from __future__ import annotations

from gwmock_pop.loaders import FilePopulationLoader
from gwmock_pop.protocols import ExternalPopulationLoader, GWPopSimulator
from gwmock_pop.simulators import CBCPriorSimulator
from gwmock_pop.version import __version__

__all__ = ["CBCPriorSimulator", "ExternalPopulationLoader", "FilePopulationLoader", "GWPopSimulator", "__version__"]
