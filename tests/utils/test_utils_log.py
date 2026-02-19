"""Tests for logging utility functions."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import pytest

from gwsim_pop.utils.log import LoggingLevel, get_version_information, setup_logger


@pytest.fixture(autouse=True)
def cleanup_logger() -> None:
    """Clean up logger state before and after each test."""
    logger = logging.getLogger("gwsim_pop")
    logger.handlers.clear()
    logger.setLevel(logging.NOTSET)


class TestLoggingLevel:
    """Test LoggingLevel enum."""

    def test_enum_values(self) -> None:
        """Test that enum values match logging module."""
        assert LoggingLevel.NOTSET.value == "NOTSET"
        assert LoggingLevel.DEBUG.value == "DEBUG"
        assert LoggingLevel.INFO.value == "INFO"
        assert LoggingLevel.WARNING.value == "WARNING"
        assert LoggingLevel.ERROR.value == "ERROR"
        assert LoggingLevel.CRITICAL.value == "CRITICAL"

    def test_enum_is_str(self) -> None:
        """Test that LoggingLevel is a string enum."""
        assert isinstance(LoggingLevel.INFO, str)
        assert LoggingLevel.INFO == "INFO"


class TestGetVersionInformation:
    """Test get_version_information function."""

    def test_returns_string(self) -> None:
        """Test that function returns a string."""
        version = get_version_information()
        assert isinstance(version, str)
        assert len(version) > 0


class TestSetupLogger:
    """Test setup_logger function."""

    def test_default_parameters(self) -> None:
        """Test setup_logger with default parameters."""
        setup_logger()
        logger = logging.getLogger("gwsim_pop")
        assert logger.level == logging.INFO

    def test_custom_log_level_enum(self) -> None:
        """Test setup_logger with LoggingLevel enum."""
        setup_logger(log_level=LoggingLevel.DEBUG)
        logger = logging.getLogger("gwsim_pop")
        assert logger.level == logging.DEBUG

    def test_custom_log_level_string(self) -> None:
        """Test setup_logger with log level as string."""
        setup_logger(log_level="WARNING")
        logger = logging.getLogger("gwsim_pop")
        assert logger.level == logging.WARNING

    def test_custom_log_level_integer(self) -> None:
        """Test setup_logger with log level as integer."""
        setup_logger(log_level=logging.ERROR)
        logger = logging.getLogger("gwsim_pop")
        assert logger.level == logging.ERROR

    def test_invalid_log_level_string(self) -> None:
        """Test setup_logger with invalid log level string."""
        with pytest.raises(ValueError, match="log_level invalid_level not understood"):
            setup_logger(log_level="invalid_level")

    def test_no_file_handler_without_filename(self) -> None:
        """Test that no file handler is created without filename."""
        setup_logger(filename=None)
        logger = logging.getLogger("gwsim_pop")
        file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 0

    def test_file_handler_with_filename(self) -> None:
        """Test that file handler is created with filename."""
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_logger(outdir=tmpdir, filename="test.log")
            logger = logging.getLogger("gwsim_pop")
            file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
            assert len(file_handlers) == 1
            log_file = Path(tmpdir) / "test.log"
            assert log_file.exists()

    def test_creates_output_directory(self) -> None:
        """Test that output directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = Path(tmpdir) / "new_subdir"
            setup_logger(outdir=str(new_dir), filename="test.log")
            logger = logging.getLogger("gwsim_pop")
            file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
            assert len(file_handlers) == 1
            assert new_dir.exists()

    def test_stream_handler_added(self) -> None:
        """Test that stream handler is added."""
        setup_logger()
        logger = logging.getLogger("gwsim_pop")
        stream_handlers = [
            h
            for h in logger.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        ]
        assert len(stream_handlers) >= 1

    def test_no_duplicate_stream_handler(self) -> None:
        """Test that duplicate stream handlers are not added."""
        setup_logger()
        setup_logger()
        logger = logging.getLogger("gwsim_pop")
        stream_handlers = [
            h
            for h in logger.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        ]
        assert len(stream_handlers) == 1

    def test_no_duplicate_file_handler(self) -> None:
        """Test that duplicate file handlers are not added."""
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_logger(outdir=tmpdir, filename="test.log")
            setup_logger(outdir=tmpdir, filename="test.log")
            logger = logging.getLogger("gwsim_pop")
            file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
            assert len(file_handlers) == 1

    def test_print_version(self) -> None:
        """Test that version is printed when print_version=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_logger(outdir=tmpdir, filename="test.log", print_version=True)
            logger = logging.getLogger("gwsim_pop")
            for handler in logger.handlers:
                handler.flush()
            log_file = Path(tmpdir) / "test.log"
            content = log_file.read_text()
            assert "Running gwsim_pop version:" in content

    def test_log_message_format(self) -> None:
        """Test that log messages have correct format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_logger(outdir=tmpdir, filename="test.log", log_level=logging.INFO)
            logger = logging.getLogger("gwsim_pop")
            logger.info("Test message")
            for handler in logger.handlers:
                handler.flush()
            log_file = Path(tmpdir) / "test.log"
            content = log_file.read_text()
            assert "Test message" in content
            assert "INFO" in content
