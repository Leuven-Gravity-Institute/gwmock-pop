"""Unit tests for CLI simulate command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import typer

from gwsim_pop.cli.simulate import simulate_command


class TestSimulateCommand:
    """Test class for simulate command."""

    def test_simulate_command_writes_population(self, tmp_path: Path) -> None:
        """Test that simulate_command runs GraphSimulator and writes an output file."""
        output_dir = tmp_path / "outputs"
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            f"""
config_version: "1.0.0"
run:
  name: cli_test
  seed: 123
  mode: fixed_n_samples
  n_samples: 5
  output:
    directory: "{output_dir.as_posix()}"
    format: csv
    overwrite: true
    save_metadata: false
parameters:
  mass_1:
    sampler:
      function: gwsim_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks
      arguments:
        alpha_1: 1.72
        alpha_2: 4.51
        transition: 35.6
        minimum: 5.06
        maximum: 300.0
        mean_1: 9.76
        sigma_1: 0.649
        mean_2: 32.8
        sigma_2: 3.92
        taper_range: 4.32
        lambda_0: 0.361
        lambda_1: 0.586
""",
            encoding="utf-8",
        )

        simulate_command(str(config_path))

        output_path = output_dir / "cli_test.csv"
        assert output_path.exists()
        loaded = np.loadtxt(output_path, delimiter=",")
        assert loaded.shape == (5,)

    @patch("logging.getLogger")
    def test_simulate_command_rejects_duration_mode(self, mock_get_logger: MagicMock, tmp_path: Path) -> None:
        """Test that the MVP CLI rejects duration mode explicitly."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            """
run:
  mode: duration
parameters:
  mass_1:
    sampler:
      function: gwsim_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks
      arguments:
        alpha_1: 1.72
        alpha_2: 4.51
        transition: 35.6
        minimum: 5.06
        maximum: 300.0
        mean_1: 9.76
        sigma_1: 0.649
        mean_2: 32.8
        sigma_2: 3.92
        taper_range: 4.32
        lambda_0: 0.361
        lambda_1: 0.586
""",
            encoding="utf-8",
        )

        with pytest.raises(typer.Exit):
            simulate_command(str(config_path))

        mock_logger.error.assert_called_once_with("Only run.mode='fixed_n_samples' is supported by the CLI MVP.")

    @patch("logging.getLogger")
    def test_simulate_command_requires_parameters(self, mock_get_logger: MagicMock, tmp_path: Path) -> None:
        """Test that the simulate command requires parameter graph configuration."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            """
run:
  mode: fixed_n_samples
  n_samples: 10
""",
            encoding="utf-8",
        )

        with pytest.raises(typer.Exit):
            simulate_command(str(config_path))

        mock_logger.error.assert_called_once_with("No parameter graph configuration found under 'parameters'.")
