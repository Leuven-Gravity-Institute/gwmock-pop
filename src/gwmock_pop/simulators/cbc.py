"""Graph-backed compact-binary-coalescence population simulators.

These simulators express the CBC parameter priors as a config-driven
:class:`~gwmock_pop.simulators.graph.GraphSimulator` graph rather than
hard-coded distributions. Each class builds a default ``parameters`` graph from
its physics-level constructor arguments and deep-merges a user-supplied
``parameters`` override, so the distribution of any single parameter can be
swapped without subclassing. They satisfy the ``GWPopSimulator`` protocol via
the inherited :class:`GraphSimulator` machinery.

The default output reproduces the legacy lightweight prior simulators: uniform
component masses, isotropic spins and sky position, distance uniform in comoving
volume, and a flat-LambdaCDM distance-to-redshift conversion.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from gwmock_pop.graph.validation import ConfigValidationError, validate_graph_config
from gwmock_pop.simulators._graph_config import GraphConfig, build_cbc_graph_config, deep_merge_graph_config
from gwmock_pop.simulators.graph import GraphSimulator


class _CBCGraphSimulator(GraphSimulator):
    """Base class for graph-backed CBC simulators.

    Subclasses build a default ``parameters`` graph config from their physics
    arguments and call ``super().__init__``, which merges the user override,
    validates the resulting graph, and hands it to :class:`GraphSimulator`.
    """

    def __init__(
        self,
        *,
        default_config: GraphConfig,
        parameters: Mapping[str, Any] | None,
        source_type: str,
        seed: int | None,
    ) -> None:
        """Merge defaults with the override, validate, and build the graph."""
        merged = deep_merge_graph_config(default_config, parameters)
        report = validate_graph_config(merged)
        if not report.is_valid:
            raise ConfigValidationError(report.issues)
        super().__init__(config=merged, source_type=source_type, seed=seed)

    @classmethod
    def from_config_file(cls, config_path: str | Path, encoding: str = "utf-8", **kwargs: Any) -> GraphSimulator:
        """Build a plain :class:`GraphSimulator` from a config file.

        The CBC subclasses take physics-level constructor arguments rather than a
        raw ``config`` mapping, so config-file and preset loading delegate to
        :class:`GraphSimulator` and return a graph simulator directly.
        """
        return GraphSimulator.from_config_file(config_path, encoding=encoding, **kwargs)

    @classmethod
    def from_preset(cls, preset_name: str, **kwargs: Any) -> GraphSimulator:
        """Build a plain :class:`GraphSimulator` from a packaged preset."""
        return GraphSimulator.from_preset(preset_name, **kwargs)


class CBCSimulator(_CBCGraphSimulator):
    """Configurable graph-backed compact-binary population simulator.

    Component masses are drawn uniformly and independently on ``[m_min, m_max]``
    (not reordered, matching the legacy ``CBCPriorSimulator``); spins are
    isotropic with magnitude uniform on ``[0, chi_max]`` (or aligned-only when
    ``aligned_spins`` is set); distance is uniform in comoving volume; sky
    position and orientation angles are isotropic. Pass ``parameters`` to
    override the distribution of any node (see
    :func:`gwmock_pop.simulators._graph_config.deep_merge_graph_config`).

    Args:
        source_type: Routing key for downstream orchestration. Defaults to
            ``"bbh"``.
        m_min: Lower bound for each component mass in solar masses.
        m_max: Upper bound for each component mass in solar masses.
        d_min: Minimum luminosity distance in Mpc.
        d_max: Maximum luminosity distance in Mpc.
        chi_max: Maximum dimensionless spin magnitude for both components.
        aligned_spins: If ``True``, populate only the ``z`` spin components.
        gps_start: Lower bound for the coalescence-time prior.
        gps_end: Upper bound for the coalescence-time prior.
        total_mass_max: Optional upper bound on ``mass_1 + mass_2``.
        f_ref: Constant reference frequency assigned to every sample.
        lambda_max: Upper bound for the uniform tidal-deformability prior applied
            to both components. ``None`` (default) sets both to zero.
        ordered_masses: If ``True``, enforce ``mass_1 >= mass_2`` per sample.
        parameters: Optional partial graph config merged onto the defaults.
        seed: Optional RNG seed forwarded to the graph simulator.
    """

    def __init__(  # noqa: PLR0913
        self,
        source_type: str = "bbh",
        *,
        m_min: float = 5.0,
        m_max: float = 100.0,
        d_min: float = 0.0,
        d_max: float = 5_000.0,
        chi_max: float = 0.99,
        aligned_spins: bool = False,
        gps_start: float = 0.0,
        gps_end: float = 1.0,
        total_mass_max: float | None = None,
        f_ref: float = 20.0,
        lambda_max: float | None = None,
        ordered_masses: bool = False,
        parameters: Mapping[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize the configurable CBC simulator."""
        default_config = build_cbc_graph_config(
            m1_min=m_min,
            m1_max=m_max,
            m2_min=m_min,
            m2_max=m_max,
            ordered_masses=ordered_masses,
            total_mass_max=total_mass_max,
            chi_1_max=chi_max,
            chi_2_max=chi_max,
            aligned_spins=aligned_spins,
            d_min=d_min,
            d_max=d_max,
            gps_start=gps_start,
            gps_end=gps_end,
            f_ref=f_ref,
            lambda_1_max=lambda_max,
            lambda_2_max=lambda_max,
        )
        super().__init__(
            default_config=default_config,
            parameters=parameters,
            source_type=source_type,
            seed=seed,
        )


class BBHSimulator(_CBCGraphSimulator):
    """Graph-backed binary-black-hole population simulator.

    Defaults to ordered component masses on ``[5, 100]`` solar masses, isotropic
    precessing spins, zero tidal deformability, and ``source_type="bbh"``.

    Args:
        m_min: Lower bound for each component mass in solar masses.
        m_max: Upper bound for each component mass in solar masses.
        d_min: Minimum luminosity distance in Mpc.
        d_max: Maximum luminosity distance in Mpc.
        chi_max: Maximum dimensionless spin magnitude for both components.
        aligned_spins: If ``True``, populate only the ``z`` spin components.
        gps_start: Lower bound for the coalescence-time prior.
        gps_end: Upper bound for the coalescence-time prior.
        total_mass_max: Optional upper bound on ``mass_1 + mass_2``.
        f_ref: Constant reference frequency assigned to every sample.
        parameters: Optional partial graph config merged onto the defaults.
        seed: Optional RNG seed forwarded to the graph simulator.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        m_min: float = 5.0,
        m_max: float = 100.0,
        d_min: float = 0.0,
        d_max: float = 5_000.0,
        chi_max: float = 0.99,
        aligned_spins: bool = False,
        gps_start: float = 0.0,
        gps_end: float = 1.0,
        total_mass_max: float | None = None,
        f_ref: float = 20.0,
        parameters: Mapping[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize the BBH simulator."""
        default_config = build_cbc_graph_config(
            m1_min=m_min,
            m1_max=m_max,
            m2_min=m_min,
            m2_max=m_max,
            ordered_masses=True,
            total_mass_max=total_mass_max,
            chi_1_max=chi_max,
            chi_2_max=chi_max,
            aligned_spins=aligned_spins,
            d_min=d_min,
            d_max=d_max,
            gps_start=gps_start,
            gps_end=gps_end,
            f_ref=f_ref,
            lambda_1_max=None,
            lambda_2_max=None,
        )
        super().__init__(
            default_config=default_config,
            parameters=parameters,
            source_type="bbh",
            seed=seed,
        )


class BNSSimulator(_CBCGraphSimulator):
    """Graph-backed binary-neutron-star population simulator.

    Defaults to ordered component masses on ``[1, 3]`` solar masses, aligned
    low-magnitude spins, uniform tidal deformability on ``[0, lambda_max]`` for
    both components, and ``source_type="bns"``.

    Args:
        m_min: Lower bound for each component mass in solar masses.
        m_max: Upper bound for each component mass in solar masses.
        d_min: Minimum luminosity distance in Mpc.
        d_max: Maximum luminosity distance in Mpc.
        chi_max: Maximum dimensionless spin magnitude for both components.
        aligned_spins: If ``True``, populate only the ``z`` spin components.
        gps_start: Lower bound for the coalescence-time prior.
        gps_end: Upper bound for the coalescence-time prior.
        total_mass_max: Optional upper bound on ``mass_1 + mass_2``.
        f_ref: Constant reference frequency assigned to every sample.
        lambda_max: Upper bound for the uniform tidal-deformability prior.
        parameters: Optional partial graph config merged onto the defaults.
        seed: Optional RNG seed forwarded to the graph simulator.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        m_min: float = 1.0,
        m_max: float = 3.0,
        d_min: float = 0.0,
        d_max: float = 5_000.0,
        chi_max: float = 0.05,
        aligned_spins: bool = True,
        gps_start: float = 0.0,
        gps_end: float = 1.0,
        total_mass_max: float | None = None,
        f_ref: float = 20.0,
        lambda_max: float = 3_000.0,
        parameters: Mapping[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize the BNS simulator."""
        default_config = build_cbc_graph_config(
            m1_min=m_min,
            m1_max=m_max,
            m2_min=m_min,
            m2_max=m_max,
            ordered_masses=True,
            total_mass_max=total_mass_max,
            chi_1_max=chi_max,
            chi_2_max=chi_max,
            aligned_spins=aligned_spins,
            d_min=d_min,
            d_max=d_max,
            gps_start=gps_start,
            gps_end=gps_end,
            f_ref=f_ref,
            lambda_1_max=lambda_max,
            lambda_2_max=lambda_max,
        )
        super().__init__(
            default_config=default_config,
            parameters=parameters,
            source_type="bns",
            seed=seed,
        )


class NSBHSimulator(_CBCGraphSimulator):
    """Graph-backed neutron-star--black-hole population simulator.

    Defaults to a black-hole primary on ``[bh_mass_min, bh_mass_max]`` solar
    masses, a neutron-star secondary on ``[ns_mass_min, ns_mass_max]``,
    component-specific isotropic spin bounds, zero primary tidal deformability,
    uniform secondary tidal deformability, and ``source_type="nsbh"``.

    Args:
        bh_mass_min: Lower bound for the black-hole primary mass.
        bh_mass_max: Upper bound for the black-hole primary mass.
        ns_mass_min: Lower bound for the neutron-star secondary mass.
        ns_mass_max: Upper bound for the neutron-star secondary mass.
        d_min: Minimum luminosity distance in Mpc.
        d_max: Maximum luminosity distance in Mpc.
        bh_chi_max: Maximum dimensionless spin magnitude for the primary.
        ns_chi_max: Maximum dimensionless spin magnitude for the secondary.
        aligned_spins: If ``True``, populate only the ``z`` spin components.
        gps_start: Lower bound for the coalescence-time prior.
        gps_end: Upper bound for the coalescence-time prior.
        total_mass_max: Optional upper bound on ``mass_1 + mass_2``.
        f_ref: Constant reference frequency assigned to every sample.
        ns_lambda_max: Upper bound for the secondary tidal-deformability prior.
        parameters: Optional partial graph config merged onto the defaults.
        seed: Optional RNG seed forwarded to the graph simulator.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        bh_mass_min: float = 3.0,
        bh_mass_max: float = 50.0,
        ns_mass_min: float = 1.0,
        ns_mass_max: float = 3.0,
        d_min: float = 0.0,
        d_max: float = 5_000.0,
        bh_chi_max: float = 1.0,
        ns_chi_max: float = 0.05,
        aligned_spins: bool = False,
        gps_start: float = 0.0,
        gps_end: float = 1.0,
        total_mass_max: float | None = None,
        f_ref: float = 20.0,
        ns_lambda_max: float = 3_000.0,
        parameters: Mapping[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize the NSBH simulator."""
        default_config = build_cbc_graph_config(
            m1_min=bh_mass_min,
            m1_max=bh_mass_max,
            m2_min=ns_mass_min,
            m2_max=ns_mass_max,
            ordered_masses=False,
            total_mass_max=total_mass_max,
            chi_1_max=bh_chi_max,
            chi_2_max=ns_chi_max,
            aligned_spins=aligned_spins,
            d_min=d_min,
            d_max=d_max,
            gps_start=gps_start,
            gps_end=gps_end,
            f_ref=f_ref,
            lambda_1_max=None,
            lambda_2_max=ns_lambda_max,
        )
        super().__init__(
            default_config=default_config,
            parameters=parameters,
            source_type="nsbh",
            seed=seed,
        )
