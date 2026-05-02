"""Tests for the CLI validate command."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from gwmock_pop.cli.main import app

_RUNNER = CliRunner()


def test_validate_command_prints_summary_for_gwtc4_example() -> None:
    """The validator prints a stable node summary for the repository example config."""
    result = _RUNNER.invoke(app, ["validate", "examples/gwtc4/bbh_population.yaml"])

    assert result.exit_code == 0, result.output
    assert "Node" in result.output
    assert "Transform" in result.output
    assert "Output key(s)" in result.output
    assert "detector_frame_mass_1" in result.output
    assert "planck_tapered_broken_power_law_plus_two_peaks" in result.output


def test_validate_command_reports_unknown_transform_with_node_name(tmp_path: Path) -> None:
    """Unknown transforms fail with a per-node error message."""
    config_path = tmp_path / "unknown_transform.yaml"
    config_path.write_text(
        """
parameters:
  bad_node:
    transform:
      function: does_not_exist
      arguments: {}
""".strip(),
        encoding="utf-8",
    )

    result = _RUNNER.invoke(app, ["validate", str(config_path)])

    assert result.exit_code == 1
    assert "bad_node" in result.output
    assert "does_not_exist" in result.output


def test_validate_command_reports_missing_required_field(tmp_path: Path) -> None:
    """Missing required transform arguments fail with the field name."""
    config_path = tmp_path / "missing_field.yaml"
    config_path.write_text(
        """
parameters:
  bad_node:
    transform:
      function: constant_like
      arguments:
        value: 1.0
""".strip(),
        encoding="utf-8",
    )

    result = _RUNNER.invoke(app, ["validate", str(config_path)])

    assert result.exit_code == 1
    assert "bad_node" in result.output
    assert "reference" in result.output
