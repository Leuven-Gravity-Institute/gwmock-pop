"""Tests for the CLI list command."""

from __future__ import annotations

from typer.testing import CliRunner

from gwmock_pop.cli.main import app

_RUNNER = CliRunner()


def test_list_command_prints_packaged_presets_and_public_simulator_classes() -> None:
    """The list CLI prints preset metadata and public simulator-class descriptions."""
    result = _RUNNER.invoke(app, ["list"])

    assert result.exit_code == 0, result.output
    assert "Presets" in result.output
    assert "Name" in result.output
    assert "Source type" in result.output
    assert "Description" in result.output
    assert "bbh_gwtc4" in result.output
    assert "bbh_flat" in result.output
    assert "bns_flat" in result.output
    assert "power_law_plus_peak" in result.output
    assert "GWTC-4-inspired BBH graph preset" in result.output
    assert "Flat-in-log BBH graph preset" in result.output
    assert "Flat BNS graph preset" in result.output
    assert "Talbot-Thrane-inspired BBH mass graph preset" in result.output
    assert "Simulator classes" in result.output
    assert "BBHSimulator" in result.output
    assert "CBCSimulator" in result.output
