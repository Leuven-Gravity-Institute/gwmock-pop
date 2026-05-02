"""List packaged presets and public simulator classes."""

from __future__ import annotations

import inspect
from dataclasses import dataclass

import typer

import gwmock_pop
from gwmock_pop.configs import PackagedPreset, iter_packaged_presets
from gwmock_pop.simulators.simulator import Simulator


@dataclass(frozen=True)
class SimulatorSummary:
    """Rendered metadata for one public simulator class."""

    name: str
    description: str


def _first_docstring_line(obj: object) -> str:
    """Return the first non-empty docstring line for display."""
    docstring = inspect.getdoc(obj) or ""
    return next((line.strip() for line in docstring.splitlines() if line.strip()), "")


def _render_table(header: list[str], rows: list[list[str]]) -> str:
    """Render a plain-text table with aligned columns."""
    widths = [max(len(cell) for cell in column) for column in zip(header, *rows, strict=False)]

    def render_row(cells: list[str]) -> str:
        return "  ".join(cell.ljust(width) for cell, width in zip(cells, widths, strict=True))

    separator = "  ".join("-" * width for width in widths)
    return "\n".join([render_row(header), separator, *(render_row(row) for row in rows)])


def _discover_public_simulator_classes() -> list[SimulatorSummary]:
    """Return top-level public simulator classes exposed by ``gwmock_pop``."""
    simulators: list[SimulatorSummary] = []
    for public_name in getattr(gwmock_pop, "__all__", []):
        candidate = getattr(gwmock_pop, public_name)
        if not inspect.isclass(candidate) or not issubclass(candidate, Simulator) or candidate is Simulator:
            continue
        simulators.append(SimulatorSummary(name=public_name, description=_first_docstring_line(candidate)))
    return sorted(simulators, key=lambda simulator: simulator.name)


def _render_presets_table(presets: list[PackagedPreset]) -> str:
    """Render the packaged preset table."""
    rows = [[preset.name, preset.source_type, preset.description] for preset in presets]
    return _render_table(header=["Name", "Source type", "Description"], rows=rows)


def _render_simulators_table(simulators: list[SimulatorSummary]) -> str:
    """Render the public simulator-class table."""
    rows = [[simulator.name, simulator.description] for simulator in simulators]
    return _render_table(header=["Class", "Description"], rows=rows)


def list_command() -> None:
    """Print packaged preset metadata and public simulator classes."""
    import logging  # noqa: PLC0415

    logger = logging.getLogger("gwmock_pop")

    try:
        presets = iter_packaged_presets()
        simulators = _discover_public_simulator_classes()
    except Exception as error:
        logger.error("%s", error)
        raise typer.Exit(1) from error

    typer.echo("Presets")
    typer.echo(_render_presets_table(presets))
    typer.echo("")
    typer.echo("Simulator classes")
    typer.echo(_render_simulators_table(simulators))
