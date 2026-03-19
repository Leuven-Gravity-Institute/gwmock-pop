"""Main entry point for the gwsim_pop CLI application."""

from __future__ import annotations

from typing import Annotated

import typer

from gwsim_pop.utils.log import LoggingLevel

# Create the main Typer app
app = typer.Typer(
    name="gwsim_pop",
    help="Main CLI for gwsim_pop.",
    rich_markup_mode="rich",
)


def version_callback(value: bool) -> None:
    """A callback function to log version information.

    Args:
        value: A boolean input. If True, log the version information.
    """
    import logging  # noqa: PLC0415

    from gwsim_pop.version import __version__  # noqa: PLC0415

    logger = logging.getLogger("gwsim_pop")
    setup_logging(level=LoggingLevel.INFO)

    if value:
        logger.info(f"gwsim_pop version: {__version__}")
        raise typer.Exit()


def setup_logging(level: LoggingLevel = LoggingLevel.INFO) -> None:
    """Set up logging with Rich handler.

    Args:
        level: Logging level.
    """
    import logging  # noqa: PLC0415

    from rich.console import Console  # noqa: PLC0415
    from rich.logging import RichHandler  # noqa: PLC0415

    logger = logging.getLogger("gwsim_pop")

    logger.setLevel(level.value)

    console = Console(stderr=True)

    # Remove any existing handlers to ensure RichHandler is used
    for h in logger.handlers[:]:  # Use slice copy to avoid modification during iteration
        logger.removeHandler(h)
    # Add the RichHandler

    handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        show_time=True,
        show_level=True,  # Keep level (e.g., DEBUG, INFO) for clarity
        markup=True,  # Enable Rich markup in messages for styling
        level=level.value,  # Ensure handler respects the level
        omit_repeated_times=False,
        log_time_format="%H:%M",
    )
    handler.setLevel(level.value)
    logger.addHandler(handler)

    # Prevent propagation to root logger to avoid duplicate output
    logger.propagate = False


@app.callback()
def main(
    verbose: Annotated[
        LoggingLevel,
        typer.Option("--verbose", help="Set verbosity level."),
    ] = LoggingLevel.INFO,
    version: Annotated[
        bool | None, typer.Option("--version", "-v", callback=version_callback, help="Log the version information.")
    ] = None,
) -> None:
    """Main entry point for the CLI application.

    Args:
        verbose: Verbosity level for logging.
        version: Log the version information.
    """
    setup_logging(verbose)


def register_commands() -> None:
    """Register CLI commands."""
    from gwsim_pop.cli.simulate import simulate_command  # noqa: PLC0415

    app.command("simulate", help="Simulate populations.")(simulate_command)


register_commands()
