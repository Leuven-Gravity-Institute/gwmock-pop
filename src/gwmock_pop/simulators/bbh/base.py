"""Backwards-compatible import location for :class:`BBHSimulator`.

The binary-black-hole simulator now lives in
:mod:`gwmock_pop.simulators.cbc` alongside the other graph-backed CBC
simulators. This module re-exports it so existing
``gwmock_pop.simulators.bbh.base`` import paths keep working.
"""

from __future__ import annotations

from gwmock_pop.simulators.cbc import BBHSimulator

__all__ = ["BBHSimulator"]
