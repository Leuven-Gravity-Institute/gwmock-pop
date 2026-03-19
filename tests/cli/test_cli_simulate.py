"""Unit tests for CLI simulate command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import typer
from typer.testing import CliRunner

from gwmock_pop.cli.main import app, setup_logging
from gwmock_pop.cli.simulate import _build_output_metadata, simulate_command
from gwmock_pop.config.main import MainConfiguration
from gwmock_pop.simulators.graph import GraphSimulator
from gwmock_pop.utils.log import LoggingLevel


class TestSimulateCommand:
    """Test class for simulate command."""

    # Constants for test configuration
    TEST_N_SAMPLES = 20
    TEST_SEED = 789

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
      function: gwmock_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks
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
      function: gwmock_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks
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

    @patch("gwmock_pop.cli.simulate._build_output_metadata")
    def test_simulate_command_with_metadata(self, mock_build_metadata: MagicMock, tmp_path: Path) -> None:
        """Test that simulate_command builds and saves metadata when requested."""
        mock_build_metadata.return_value = {"run_name": "cli_test_metadata"}

        output_dir = tmp_path / "outputs"
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            f"""
config_version: "1.0.0"
run:
  name: cli_test_metadata
  seed: 456
  mode: fixed_n_samples
  n_samples: 10
  output:
    directory: "{output_dir.as_posix()}"
    format: csv
    overwrite: true
    save_metadata: true
parameters:
  mass_1:
    sampler:
      function: gwmock_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks
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
        mock_build_metadata.assert_called_once()

        output_path = output_dir / "cli_test_metadata.csv"
        assert output_path.exists()

    def test_build_output_metadata(self, tmp_path: Path) -> None:
        """Test that _build_output_metadata builds correct metadata."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            f"""
config_version: "1.0.0"
run:
  name: test_metadata
  seed: {self.TEST_SEED}
  mode: fixed_n_samples
  n_samples: {self.TEST_N_SAMPLES}
  output:
    directory: "/tmp/test"
    format: csv
    overwrite: true
    save_metadata: false
parameters:
  mass_1:
    sampler:
      function: gwmock_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks
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

        config = MainConfiguration.from_file(str(config_path))
        simulator = GraphSimulator(config=config.parameters, seed=config.run.seed)

        metadata = _build_output_metadata(config, simulator)

        assert metadata["config_version"] == "1.0.0"
        assert metadata["run_name"] == "test_metadata"
        assert metadata["n_samples"] == self.TEST_N_SAMPLES
        assert metadata["seed"] == self.TEST_SEED
        assert "mass_1" in metadata["parameter_names"]

    def test_simulate_command_overwrite_rejection(self, tmp_path: Path) -> None:
        """Test that simulate_command rejects overwriting without explicit permission."""
        output_dir = tmp_path / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "existing.csv"
        output_path.write_text("1,2,3\n", encoding="utf-8")

        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            f"""
config_version: "1.0.0"
run:
  name: existing
  seed: 111
  mode: fixed_n_samples
  n_samples: 5
  output:
    directory: "{output_dir.as_posix()}"
    format: csv
    overwrite: false
    save_metadata: false
parameters:
  mass_1:
    sampler:
      function: gwmock_pop.samplers.planck_tapered_broken_power_law_plus_two_peaks
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

        assert output_path.exists()


class TestSetupLogging:
    """Test class for setup_logging function."""

    def test_setup_logging_with_info_level(self) -> None:
        """Test that setup_logging sets up logging with INFO level."""
        setup_logging(LoggingLevel.INFO)
        # Just verify it doesn't raise an exception
        assert True

    def test_setup_logging_with_debug_level(self) -> None:
        """Test that setup_logging sets up logging with DEBUG level."""
        setup_logging(LoggingLevel.DEBUG)
        # Just verify it doesn't raise an exception
        assert True

    def test_setup_logging_with_error_level(self) -> None:
        """Test that setup_logging sets up logging with ERROR level."""
        setup_logging(LoggingLevel.ERROR)
        # Just verify it doesn't raise an exception
        assert True


class TestVersionCallback:
    """Test class for version callback function."""

    def test_version_callback_logs_version(self) -> None:
        """Test that version_callback logs version and raises Exit."""
        runner = CliRunner()
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        # Version output goes to stderr via logging
        assert (
            "gwmock_pop version:" in result.stdout
            or "gwmock_pop version:" in result.stderr
            or "gwmock_pop version:" in str(result)
        )


class TestSimulateCommandErrorHandling:
    """Test class for simulate command error handling."""

    def test_simulate_command_file_not_found(self) -> None:
        """Test that simulate_command raises typer.Exit for non-existent file."""
        with pytest.raises(typer.Exit):
            simulate_command("nonexistent_config.yaml")

    @patch("logging.getLogger")
    def test_simulate_command_validation_error(self, mock_get_logger: MagicMock, tmp_path: Path) -> None:
        """Test that simulate_command handles validation errors."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        config_path = tmp_path / "invalid_config.yaml"
        config_path.write_text(
            """
# Invalid config - missing required fields
run:
  mode: fixed_n_samples
""",
            encoding="utf-8",
        )

        with pytest.raises(typer.Exit):
            simulate_command(str(config_path))

    def test_simulate_command_empty_config(self, tmp_path: Path) -> None:
        """Test that simulate_command handles empty config file."""
        config_path = tmp_path / "empty_config.yaml"
        config_path.write_text("", encoding="utf-8")

        with pytest.raises(typer.Exit):
            simulate_command(str(config_path))
