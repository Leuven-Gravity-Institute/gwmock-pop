"""Conversion functions for parameters."""

from __future__ import annotations

from gwmock_pop.conversion.cbc import (
    compute_chirp_mass_from_mass_1_mass_2,
    compute_mass_ratio_from_mass_1_mass_2,
    compute_source_frame_mass_from_detector_frame_mass,
    compute_symmetric_mass_ratio_from_mass_1_mass_2,
    compute_total_mass_from_mass_1_mass_2,
)

__all__ = [
    "compute_chirp_mass_from_mass_1_mass_2",
    "compute_mass_ratio_from_mass_1_mass_2",
    "compute_source_frame_mass_from_detector_frame_mass",
    "compute_symmetric_mass_ratio_from_mass_1_mass_2",
    "compute_total_mass_from_mass_1_mass_2",
]
