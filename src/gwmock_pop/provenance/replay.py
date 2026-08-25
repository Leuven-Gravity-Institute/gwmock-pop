"""Rebuild a simulation run from its provenance record.

A recorded seed is worth nothing if replaying it produces a different
catalogue, so the record is designed to be sufficient on its own: given one,
these functions rebuild the simulator and redraw the samples without the
configuration file, the preset or the catalogue being present.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from gwmock_pop.provenance.record import GRAPH_SIMULATION
from gwmock_pop.version import __version__

if TYPE_CHECKING:
    from jax import Array

    from gwmock_pop.simulators.graph import GraphSimulator

logger = logging.getLogger("gwmock_pop")


def reconstruct_run(record: Mapping[str, Any]) -> tuple[GraphSimulator, int]:
    """Rebuild the simulator and sample count a record describes.

    Args:
        record: A provenance record with a ``graph_simulation`` origin.

    Returns:
        The reconstructed simulator and the number of samples the recorded run
        drew from it.

    Raises:
        ValueError: If the record does not describe a run this package can
            replay, or if the reconstructed simulator disagrees with the
            recorded column list.
    """
    origin = record.get("origin") or {}
    kind = origin.get("kind")
    if kind != GRAPH_SIMULATION:
        raise ValueError(
            f"Cannot reconstruct a run from a record with origin kind {kind!r}; "
            f"only {GRAPH_SIMULATION!r} records describe a run this package can replay."
        )

    configuration = origin.get("configuration") or {}
    parameters = configuration.get("parameters")
    if not parameters:
        raise ValueError("The record carries no parameter graph, so the run cannot be reconstructed.")

    run = record.get("run")
    if not run or run.get("seed") is None:
        raise ValueError("The record carries no seed, so the run cannot be reconstructed.")

    catalogue = record.get("catalogue") or {}
    recorded_version = (record.get("tool") or {}).get("version")
    if recorded_version and recorded_version != __version__:
        logger.warning(
            "The record was written by gwmock-pop %s but this is %s; a replay may differ.",
            recorded_version,
            __version__,
        )

    from gwmock_pop.simulators.graph import GraphSimulator  # noqa: PLC0415  # avoids an import cycle

    simulator = GraphSimulator(
        config=dict(parameters),
        source_type=catalogue.get("source_type"),
        seed=int(run["seed"]),
    )

    recorded_names = catalogue.get("parameter_names")
    if recorded_names is not None and list(recorded_names) != simulator.parameter_names:
        raise ValueError(
            "The recorded parameter_names cannot be produced by the recorded parameter graph: "
            f"the record lists {list(recorded_names)} but the graph yields {simulator.parameter_names}."
        )

    return simulator, int(catalogue["n_samples"])


def replay_catalogue(record: Mapping[str, Any]) -> dict[str, Array]:
    """Redraw the catalogue a record describes.

    Args:
        record: A provenance record with a ``graph_simulation`` origin.

    Returns:
        Mapping from parameter name to the redrawn column, in recorded order.

    Raises:
        ValueError: If the record does not describe a replayable run.
    """
    simulator, n_samples = reconstruct_run(record)
    return dict(simulator.simulate(n_samples))
