"""Simulate populations."""

from __future__ import annotations

from typing import Annotated

import typer


def simulate_command(filename: Annotated[str, typer.Argument(help="File name of the configuration file.")]) -> None:
    """Implement the simulate command.

    Args:
        filename: File name of the configuration file.

    """
    import logging  # noqa: PLC0415

    logger = logging.getLogger("gwsim_pop")
    logger.error("The simulate command has not been implemented yet.")
    typer.Exit(1)
