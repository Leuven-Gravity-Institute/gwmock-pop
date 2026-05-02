"""Inspect populations from packaged presets or graph config files."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated

import numpy as np
import typer
from jax import Array

from gwmock_pop.cli.common import resolve_simulator
from gwmock_pop.protocols import GWPopSimulator


@dataclass(frozen=True)
class ParameterSummary:
    """Summary statistics for one sampled parameter."""

    parameter: str
    mean: float
    std: float
    p5: float
    p95: float
    minimum: float
    maximum: float


def _ordered_parameter_names(simulator: GWPopSimulator, population: Mapping[str, Array]) -> list[str]:
    """Return a stable output order for the inspected population."""
    preferred = [name for name in simulator.parameter_names if name in population]
    remaining = [name for name in population if name not in preferred]
    return preferred + remaining


def _summarize_population(simulator: GWPopSimulator, population: Mapping[str, Array]) -> list[ParameterSummary]:
    """Compute scalar summary statistics for each sampled parameter."""
    summaries: list[ParameterSummary] = []
    for parameter in _ordered_parameter_names(simulator=simulator, population=population):
        values = np.asarray(population[parameter], dtype=float)
        p5, p95 = np.percentile(values, [5, 95])
        summaries.append(
            ParameterSummary(
                parameter=parameter,
                mean=float(np.mean(values)),
                std=float(np.std(values)),
                p5=float(p5),
                p95=float(p95),
                minimum=float(np.min(values)),
                maximum=float(np.max(values)),
            )
        )
    return summaries


def _format_float(value: float) -> str:
    """Format a summary-statistic value for tabular output."""
    return f"{value:.6g}"


def render_summary_table(summaries: list[ParameterSummary]) -> str:
    """Render a parameter-summary table for stdout output."""
    header = ["Parameter", "Mean", "Std", "P5", "P95", "Min", "Max"]
    rows = [
        [
            summary.parameter,
            _format_float(summary.mean),
            _format_float(summary.std),
            _format_float(summary.p5),
            _format_float(summary.p95),
            _format_float(summary.minimum),
            _format_float(summary.maximum),
        ]
        for summary in summaries
    ]
    widths = [max(len(cell) for cell in column) for column in zip(header, *rows, strict=False)]

    def render_row(cells: list[str]) -> str:
        return "  ".join(cell.ljust(width) for cell, width in zip(cells, widths, strict=True))

    separator = "  ".join("-" * width for width in widths)
    return "\n".join([render_row(header), separator, *(render_row(row) for row in rows)])


def inspect_command(
    config: Annotated[str, typer.Option("--config", help="Preset name or YAML/TOML config-file path.")],
    n: Annotated[int, typer.Option("--n", min=1, help="Number of events to sample.")] = 1000,
    seed: Annotated[int | None, typer.Option("--seed", help="Optional random seed.")] = None,
) -> None:
    """Draw a quick sample and print per-parameter summary statistics."""
    import logging  # noqa: PLC0415

    logger = logging.getLogger("gwmock_pop")

    try:
        simulator = resolve_simulator(config=config, seed=seed)
        population = simulator.simulate(n)
        summaries = _summarize_population(simulator=simulator, population=population)
    except Exception as error:
        logger.error("%s", error)
        raise typer.Exit(1) from error

    typer.echo(render_summary_table(summaries))
