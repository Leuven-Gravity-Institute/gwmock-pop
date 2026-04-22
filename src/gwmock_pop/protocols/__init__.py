"""Public protocols for gwmock_pop."""

from __future__ import annotations

from gwmock_pop.protocols.loader import ExternalPopulationLoader
from gwmock_pop.protocols.simulator import GWPopSimulator

__all__ = ["ExternalPopulationLoader", "GWPopSimulator"]
