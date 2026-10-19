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

# Planck 2018 flat Lambda-CDM parameters: Hubble constant in km/s/Mpc and matter density.
PLANCK18_H0_KM_S_MPC = 67.66
PLANCK18_OMEGA_M = 0.3111

# Madau & Dickinson (2014) cosmic star-formation-rate parameters: psi ~
# (1+z)^2.7 / [1 + ((1+z)/2.9)^5.6], i.e. gamma=2.7, gamma+kappa=5.6 (kappa=2.9),
# 1+z_peak=2.9 (z_peak=1.9).
MADAU_DICKINSON_GAMMA = 2.7
MADAU_DICKINSON_KAPPA = 2.9
MADAU_DICKINSON_Z_PEAK = 1.9

__all__ = [
    "CBC_PARAMETER_NAMES",
    "MADAU_DICKINSON_GAMMA",
    "MADAU_DICKINSON_KAPPA",
    "MADAU_DICKINSON_Z_PEAK",
    "PLANCK18_H0_KM_S_MPC",
    "PLANCK18_OMEGA_M",
    "SPEED_OF_LIGHT",
]
