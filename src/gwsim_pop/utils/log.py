"""Utility functions for logging."""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path

from gwsim_pop.version import __version__


class LoggingLevel(str, Enum):
    """Logging level."""

    NOTSET = "NOTSET"
    """When set on a logger, indicates that ancestor loggers are to be consulted to determine the effective level.
    If that still resolves to NOTSET, then all events are logged. When set on a handler, all events are handled."""

    DEBUG = "DEBUG"
    """Detailed information, typically only of interest to a developer trying to diagnose a problem."""

    INFO = "INFO"
    """Confirmation that things are working as expected."""

    WARNING = "WARNING"
    """An indication that something unexpected happened, or that a problem might occur in the near future (e.g. 'disk space low'). The software is still working as expected."""

    ERROR = "ERROR"
    """Due to a more serious problem, the software has not been able to perform some function."""

    CRITICAL = "CRITICAL"
    """A serious error, indicating that the program itself may be unable to continue running."""


def get_version_information() -> str:
    """Get the version information.

    Returns:
        str: Version information.

    """
    return __version__


def setup_logger(
    outdir: str = ".",
    filename: str | None = None,
    log_level: LoggingLevel | str | int = "INFO",
    print_version: bool = False,
) -> None:
    """Set up logging output: call at the start of the script to use.

    Args:
        outdir: Output directory for log file.
        filename: Name of the log file. If None, no log file is created.
        log_level: Logging level as string or integer.
        print_version: Whether to print version information to the log.

    """
    if isinstance(log_level, LoggingLevel):
        level = log_level.value
    elif isinstance(log_level, str):
        try:
            level = getattr(logging, log_level.upper())
        except AttributeError as e:
            raise ValueError(f"log_level {log_level} not understood") from e
    else:
        level = int(log_level)

    logger = logging.getLogger("gwsim_pop")
    logger.propagate = False
    logger.setLevel(level)

    if not any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in logger.handlers
    ):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(
            logging.Formatter("%(asctime)s %(name)s %(levelname)-8s: %(message)s", datefmt="%H:%M")
        )
        stream_handler.setLevel(level)
        logger.addHandler(stream_handler)

    if filename:
        outdir_path = Path(outdir)
        outdir_path.mkdir(parents=True, exist_ok=True)
        log_file = outdir_path / filename
        if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
            file_handler = logging.FileHandler(log_file, mode="a")
            file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s: %(message)s", datefmt="%H:%M"))
            file_handler.setLevel(level)
            logger.addHandler(file_handler)

    for handler in logger.handlers:
        handler.setLevel(level)

    if print_version:
        version = get_version_information()
        logger.info("Running gwsim_pop version: %s", version)
