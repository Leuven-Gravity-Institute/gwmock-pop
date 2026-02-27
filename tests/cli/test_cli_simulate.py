"""Unit tests for CLI simulate command."""

from unittest.mock import MagicMock, patch

import pytest
import typer

from gwsim_pop.cli.simulate import simulate_command


class TestSimulateCommand:
    """Test class for simulate command."""

    @patch("logging.getLogger")
    def test_simulate_command_function(self, mock_get_logger):
        """Test simulate_command function behavior."""
        # Mock the logger
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Test that the function exits with code 1
        with pytest.raises(typer.Exit):
            simulate_command("test_config.yaml")
        # Verify that the logger error message was called
        mock_logger.error.assert_called_once_with("The simulate command has not been implemented yet.")

    @patch("logging.getLogger")
    def test_simulate_command_with_different_filename(self, mock_get_logger):
        """Test simulate_command function with different filename."""
        # Mock the logger
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Test with a different filename
        with pytest.raises(typer.Exit):
            simulate_command("config.json")

        # Verify that the logger error message was called
        mock_logger.error.assert_called_once_with("The simulate command has not been implemented yet.")

    @patch("logging.getLogger")
    def test_simulate_command_logging(self, mock_get_logger):
        """Test that simulate_command logs the correct message."""
        # Mock the logger
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Test that the function exits and logs the expected error message
        with pytest.raises(typer.Exit):
            simulate_command("test_config.yaml")

        # Verify the exact error message was logged
        mock_logger.error.assert_called_once_with("The simulate command has not been implemented yet.")
