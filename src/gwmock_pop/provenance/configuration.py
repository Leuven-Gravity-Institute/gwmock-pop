"""Render a resolved simulation configuration for a provenance record."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gwmock_pop.config.main import MainConfiguration

_EFFECTIVE_RUN_FIELDS = ("name", "seed", "n_samples")


def resolved_configuration_payload(configuration: MainConfiguration) -> dict[str, Any]:
    """Return a configuration as the complete, JSON-ready block of a record.

    Two things are deliberate here. Every block is present, including the two the
    model excludes from an ordinary dump, because a record that claims to carry
    the complete configuration has to carry all of it. And the run name, seed and
    sample count are removed, because a record states those once -- in its own
    ``run`` and ``catalogue`` blocks, as the run actually used them. Leaving the
    declared copies in as well would let a reader find two answers to the same
    question and no way to tell which one ran.

    Args:
        configuration: The resolved configuration, defaults filled in.

    Returns:
        The configuration block of a provenance record.
    """
    payload = configuration.model_dump(mode="json")
    payload["selection"] = configuration.selection.model_dump(mode="json")
    payload["post_processing"] = configuration.post_processing.model_dump(mode="json")
    run = payload.get("run")
    if isinstance(run, dict):
        for field in _EFFECTIVE_RUN_FIELDS:
            run.pop(field, None)
    return payload
