"""Validate graph configs without executing sampling."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from gwmock_pop.graph.validation import ConfigValidationError, render_validation_summary, validate_graph_config_file


def validate_command(
    config: Annotated[str, typer.Argument(help="Path to a YAML/TOML graph config file.")],
) -> None:
    """Validate a graph config and print a node summary on success."""
    import logging  # noqa: PLC0415

    logger = logging.getLogger("gwmock_pop")
    config_path = Path(config).expanduser()

    try:
        report = validate_graph_config_file(config_path)
    except ConfigValidationError as error:
        for issue in error.issues:
            logger.error("%s", issue.render())
        raise typer.Exit(1) from error
    except Exception as error:
        logger.error("%s", error)
        raise typer.Exit(1) from error

    typer.echo(render_validation_summary(report.summaries))
