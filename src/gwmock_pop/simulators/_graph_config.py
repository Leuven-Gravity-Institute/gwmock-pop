"""Default graph-config builders and deep-merge logic for CBC simulators.

These helpers translate the physics-level constructor arguments of the
graph-backed CBC family (:mod:`gwmock_pop.simulators.cbc`) into a ``parameters``
graph config consumable by :class:`~gwmock_pop.simulators.graph.GraphSimulator`,
and merge a user-supplied override on top of the defaults so that any single
parameter's distribution can be replaced.

The builders emit the canonical CBC output nodes in
:data:`~gwmock_pop.constants._CBC_PARAMETER_NAME_SEQUENCE` order. Intermediate
helper nodes (mass pair, spin magnitude/tilt/azimuth) are flagged
``intermediate: true`` and dropped from the simulator output, so the surviving
output keys preserve the canonical ordering required by the protocol.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any

_TWO_PI = 2.0 * math.pi

GraphConfig = dict[str, dict[str, Any]]

# Shape/dependency reference used by transform nodes that only need an array of
# the right length. ``detector_frame_mass_1`` is always an output node.
_SHAPE_REFERENCE = "@detector_frame_mass_1"


def _uniform(minimum: float, maximum: float, *, intermediate: bool = False) -> dict[str, Any]:
    """Build a ``uniform`` sampler node."""
    node: dict[str, Any] = {"sampler": {"function": "uniform", "arguments": {"minimum": minimum, "maximum": maximum}}}
    if intermediate:
        node["intermediate"] = True
    return node


def _constant(value: float, *, reference: str = _SHAPE_REFERENCE) -> dict[str, Any]:
    """Build a ``constant_like`` transform node."""
    return {"transform": {"function": "constant_like", "arguments": {"reference": reference, "value": value}}}


def _spin_group(component: int, chi_max: float, *, aligned_spins: bool) -> GraphConfig:
    """Build the ``spin_<i>x/y/z`` node group for one binary component.

    For precessing spins the magnitude, tilt, and azimuth are sampled as
    intermediate nodes and projected onto Cartesian components. For aligned
    spins only the ``z`` component carries spin (uniform on ``[-chi_max,
    chi_max]``, equivalent to a signed magnitude draw) and the in-plane
    components are zero.
    """
    spin_x = f"spin_{component}x"
    spin_y = f"spin_{component}y"
    spin_z = f"spin_{component}z"

    if aligned_spins:
        # Output keys must appear in canonical x, y, z order; spin_z is the
        # sampler the in-plane constants depend on (resolved by topological sort).
        return {
            spin_x: _constant(0.0, reference=f"@{spin_z}"),
            spin_y: _constant(0.0, reference=f"@{spin_z}"),
            spin_z: _uniform(-chi_max, chi_max),
        }

    magnitude = f"spin_{component}_magnitude"
    tilt = f"spin_{component}_tilt"
    azimuth = f"spin_{component}_azimuth"
    return {
        magnitude: _uniform(0.0, chi_max, intermediate=True),
        tilt: {
            "intermediate": True,
            "transform": {"function": "isotropic_spin_orientation", "arguments": {"reference": f"@{magnitude}"}},
        },
        azimuth: _uniform(0.0, _TWO_PI, intermediate=True),
        spin_x: {
            "transform": {
                "function": "spherical_to_cartesian_x",
                "arguments": {"magnitude": f"@{magnitude}", "tilt": f"@{tilt}", "azimuth": f"@{azimuth}"},
            }
        },
        spin_y: {
            "transform": {
                "function": "spherical_to_cartesian_y",
                "arguments": {"magnitude": f"@{magnitude}", "tilt": f"@{tilt}", "azimuth": f"@{azimuth}"},
            }
        },
        spin_z: {
            "transform": {
                "function": "spherical_to_cartesian_z",
                "arguments": {"magnitude": f"@{magnitude}", "tilt": f"@{tilt}"},
            }
        },
    }


def _tidal_node(lambda_max: float | None) -> dict[str, Any]:
    """Build a tidal-deformability node (constant zero or uniform)."""
    if lambda_max is None:
        return _constant(0.0)
    return _uniform(0.0, lambda_max)


def build_cbc_graph_config(  # noqa: PLR0913
    *,
    m1_min: float,
    m1_max: float,
    m2_min: float,
    m2_max: float,
    ordered_masses: bool,
    total_mass_max: float | None,
    chi_1_max: float,
    chi_2_max: float,
    aligned_spins: bool,
    d_min: float,
    d_max: float,
    gps_start: float,
    gps_end: float,
    f_ref: float,
    lambda_1_max: float | None,
    lambda_2_max: float | None,
    eccentricity: float = 0.0,
) -> GraphConfig:
    """Build the default ``parameters`` graph config for a CBC population.

    Args:
        m1_min: Lower bound for the primary component mass.
        m1_max: Upper bound for the primary component mass.
        m2_min: Lower bound for the secondary component mass.
        m2_max: Upper bound for the secondary component mass.
        ordered_masses: If ``True``, enforce ``mass_1 >= mass_2`` per sample.
        total_mass_max: Optional upper bound on ``mass_1 + mass_2``.
        chi_1_max: Maximum dimensionless spin magnitude for component 1.
        chi_2_max: Maximum dimensionless spin magnitude for component 2.
        aligned_spins: If ``True``, populate only the ``z`` spin components.
        d_min: Minimum luminosity distance in Mpc.
        d_max: Maximum luminosity distance in Mpc.
        gps_start: Lower bound for the coalescence-time prior.
        gps_end: Upper bound for the coalescence-time prior.
        f_ref: Constant reference frequency assigned to every sample.
        lambda_1_max: Upper bound for ``lambda_1`` (``None`` for a zero constant).
        lambda_2_max: Upper bound for ``lambda_2`` (``None`` for a zero constant).
        eccentricity: Constant eccentricity assigned to every sample.

    Returns:
        A ``parameters`` graph config dict in canonical CBC output order.
    """
    config: GraphConfig = {
        "mass_pair": {
            "intermediate": True,
            "sampler": {
                "function": "joint_uniform_mass_pair",
                "arguments": {
                    "m1_min": m1_min,
                    "m1_max": m1_max,
                    "m2_min": m2_min,
                    "m2_max": m2_max,
                    "total_mass_max": total_mass_max,
                    "ordered": ordered_masses,
                },
            },
        },
        "detector_frame_mass_1": {
            "transform": {"function": "take_row", "arguments": {"matrix": "@mass_pair", "index": 0}}
        },
        "detector_frame_mass_2": {
            "transform": {"function": "take_row", "arguments": {"matrix": "@mass_pair", "index": 1}}
        },
        **_spin_group(1, chi_1_max, aligned_spins=aligned_spins),
        **_spin_group(2, chi_2_max, aligned_spins=aligned_spins),
        "lambda_1": _tidal_node(lambda_1_max),
        "lambda_2": _tidal_node(lambda_2_max),
        "eccentricity": _constant(eccentricity),
        "distance": {
            "sampler": {
                "function": "uniform_comoving_volume_distance",
                "arguments": {"d_min": d_min, "d_max": d_max},
            }
        },
        "coa_phase": _uniform(0.0, _TWO_PI),
        "inclination": {
            "transform": {"function": "isotropic_spin_orientation", "arguments": {"reference": _SHAPE_REFERENCE}}
        },
        "theta_jn": (
            {"transform": {"function": "identity", "arguments": {"value": "@inclination"}}}
            if aligned_spins
            else {"transform": {"function": "isotropic_spin_orientation", "arguments": {"reference": _SHAPE_REFERENCE}}}
        ),
        "long_asc_node": _uniform(0.0, _TWO_PI),
        "mean_per_ano": _uniform(0.0, _TWO_PI),
        "coa_time": _uniform(gps_start, gps_end),
        "right_ascension": _uniform(0.0, _TWO_PI),
        "declination": {
            "transform": {"function": "isotropic_declination", "arguments": {"reference": _SHAPE_REFERENCE}}
        },
        "polarization_angle": _uniform(0.0, math.pi),
        "redshift": {
            "transform": {
                "function": "luminosity_distance_to_redshift",
                "arguments": {"luminosity_distance": "@distance"},
            }
        },
        "f_ref": _constant(f_ref),
    }
    return config


def _merge_dict(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (mapping values merge, others replace)."""
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(value, Mapping) and isinstance(existing, dict):
            merged[key] = _merge_dict(existing, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _merge_node(default_node: Any, override_node: Any) -> Any:
    """Merge a single node spec, replacing the node when the block type switches."""
    if not isinstance(default_node, dict) or not isinstance(override_node, Mapping):
        return copy.deepcopy(override_node)

    switches_block = ("sampler" in override_node and "transform" in default_node) or (
        "transform" in override_node and "sampler" in default_node
    )
    if switches_block:
        preserved = {key: default_node[key] for key in ("intermediate", "exclude") if key in default_node}
        preserved.update(copy.deepcopy(dict(override_node)))
        return preserved
    return _merge_dict(default_node, override_node)


def deep_merge_graph_config(defaults: GraphConfig, override: Mapping[str, Any] | None) -> GraphConfig:
    """Deep-merge a user ``parameters`` override onto a default graph config.

    Existing nodes are merged recursively (a node's ``arguments`` can be patched
    without restating ``function``); switching a node's block type
    (``sampler`` <-> ``transform``) replaces the node while preserving its
    ``intermediate``/``exclude`` flags unless overridden. Nodes not present in
    the defaults are appended, extending the output parameter set.

    Args:
        defaults: The default graph config produced by a builder.
        override: Optional user-supplied partial graph config.

    Returns:
        A new merged graph config dict (the inputs are not mutated).
    """
    merged: GraphConfig = copy.deepcopy(defaults)
    if not override:
        return merged
    for name, override_spec in override.items():
        if name in merged:
            merged[name] = _merge_node(merged[name], override_spec)
        else:
            merged[name] = copy.deepcopy(override_spec)
    return merged
