"""Physical constants and public parameter-name contracts."""

from __future__ import annotations

_CBC_PARAMETER_NAME_SEQUENCE = (
    "detector_frame_mass_1",
    "detector_frame_mass_2",
    "spin_1x",
    "spin_1y",
    "spin_1z",
    "spin_2x",
    "spin_2y",
    "spin_2z",
    "lambda_1",
    "lambda_2",
    "eccentricity",
    "distance",
    "coa_phase",
    "inclination",
    "theta_jn",
    "long_asc_node",
    "mean_per_ano",
    "coa_time",
    "right_ascension",
    "declination",
    "polarization_angle",
    "redshift",
    "f_ref",
)

# Canonical gwmock-* CBC parameter names used at the public package boundary.
CBC_PARAMETER_NAMES = frozenset(_CBC_PARAMETER_NAME_SEQUENCE)

# Speed of light in m/s.
# Reference: CODATA 2018
SPEED_OF_LIGHT = 299792458.0

__all__ = ["CBC_PARAMETER_NAMES", "SPEED_OF_LIGHT"]
