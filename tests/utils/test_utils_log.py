"""Tests for logging utility functions."""

from __future__ import annotations

import logging
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pytest

from gwsim_pop.utils.log import LoggingLevel, get_version_information, setup_logger


@contextmanager
def _temp_file_logger(log_file: Path, level=logging.INFO):
    """Context manager: adds & removes FileHandler cleanly."""
    logger = logging.getLogger("test_logger")  # ← use a dedicated name to avoid pollution
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))  # your format

    logger.addHandler(handler)
    logger.setLevel(level)

    try:
        yield handler, log_file  # or just yield log_file if you don't need the handler
    finally:
        # Critical cleanup — do this BEFORE temp dir is removed
        handler.flush()
        handler.close()
        logger.removeHandler(handler)
        # Optional: logger.setLevel(logging.NOTSET) or remove if you reset everything


@pytest.fixture
def temp_log_file(tmp_path: Path):
    """Fixture: provides a temp directory + sets up logging to test.log inside it.

    Automatically cleans up the FileHandler so Windows doesn't complain.
    """
    log_path = tmp_path / "test.log"

    with _temp_file_logger(log_path) as (_handler, path):
        yield path  # ← tests receive only the Path to the log file


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

    def test_file_handler_with_filename(self, temp_log_file) -> None:
        """Test that file handler is created with filename."""
        setup_logger(outdir=temp_log_file.parent, filename=temp_log_file.name)
        logger = logging.getLogger("gwsim_pop")
        file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 1
        assert temp_log_file.exists()

    def test_creates_output_directory(self) -> None:
        """Test that output directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = Path(tmpdir) / "new_subdir"
            setup_logger(outdir=str(new_dir), filename="test.log")
            logger = logging.getLogger("gwsim_pop")
            file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
            assert len(file_handlers) == 1
            assert new_dir.exists()

            if file_handlers:
                handler = file_handlers[0]
                handler.flush()  # ensure all writes are done
                handler.close()  # closes the file descriptor
                logger.removeHandler(handler)  # detach from logger

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

    def test_no_duplicate_file_handler(self, temp_log_file) -> None:
        """Test that duplicate file handlers are not added."""
        setup_logger(outdir=temp_log_file.parent, filename=temp_log_file.name)
        setup_logger(outdir=temp_log_file.parent, filename=temp_log_file.name)
        logger = logging.getLogger("gwsim_pop")
        file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 1

    def test_print_version(self, temp_log_file) -> None:
        """Test that version is printed when print_version=True."""
        setup_logger(outdir=temp_log_file.parent, filename=temp_log_file.name, print_version=True)
        _logger = logging.getLogger("gwsim_pop")
        content = temp_log_file.read_text()
        assert "Running gwsim_pop version:" in content

    def test_log_message_format(self, temp_log_file) -> None:
        """Test that log messages have correct format."""
        setup_logger(outdir=temp_log_file.parent, filename=temp_log_file.name, log_level=logging.INFO)
        logger = logging.getLogger("gwsim_pop")
        logger.info("Test message")
        content = temp_log_file.read_text()
        assert "Test message" in content
        assert "INFO" in content
