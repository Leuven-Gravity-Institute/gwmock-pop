"""Tests for the CLI inspect command."""

from __future__ import annotations

import re

import numpy as np
from typer.testing import CliRunner

from gwmock_pop.cli.main import app
from gwmock_pop.simulators.bbh.base import BBHSimulator

_RUNNER = CliRunner()


def _parse_summary_rows(output: str) -> dict[str, list[float]]:
    """Parse the inspect command's summary table into numeric rows."""
    lines = [line for line in output.splitlines() if line.strip()]
    data_lines = lines[2:]
    rows: dict[str, list[float]] = {}
    for line in data_lines:
        columns = re.split(r"\s{2,}", line.strip())
        assert len(columns) == 7, line
        rows[columns[0]] = [float(value) for value in columns[1:]]
    return rows


def test_inspect_command_prints_all_canonical_bbh_parameters_with_finite_statistics() -> None:
    """The inspect CLI prints one finite summary row per canonical BBH parameter."""
    result = _RUNNER.invoke(app, ["inspect", "--config", "gwtc4", "--seed", "42"])

    assert result.exit_code == 0, result.output
    assert "Parameter" in result.output
    assert "P95" in result.output

    rows = _parse_summary_rows(result.output)
    expected_parameters = set(BBHSimulator.from_preset("gwtc4").parameter_names)

    assert set(rows) == expected_parameters
    for statistics in rows.values():
        assert np.isfinite(np.asarray(statistics)).all()


def test_inspect_command_rejects_unknown_config_target() -> None:
    """Unknown preset names and missing config files fail clearly."""
    result = _RUNNER.invoke(app, ["inspect", "--config", "/bad/path"])

    assert result.exit_code == 1
    assert "Unknown preset or configuration path '/bad/path'." in result.output
