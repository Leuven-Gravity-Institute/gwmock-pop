"""API for the simulator classes."""

from __future__ import annotations

from gwsim_pop.simulators.factory import get_simulator
from gwsim_pop.simulators.simulator import Simulator

__all__ = ["Simulator", "get_simulator"]
