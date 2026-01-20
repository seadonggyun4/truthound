"""Command-line interface for Truthound.

This module provides the main CLI entry point for Truthound.
All commands are registered through the modular CLI architecture
in `truthound.cli_modules`.
"""

import typer

from truthound.cli_modules.registry import auto_register


def _version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        from truthound import __version__
        typer.echo(f"truthound {__version__}")
        raise typer.Exit()


app = typer.Typer(
    name="truthound",
    help="Zero-Configuration Data Quality Framework Powered by Polars",
    add_completion=False,
)


@app.callback()
def main_callback(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """Truthound - Zero-Configuration Data Quality Framework."""
    pass


# Register all CLI modules (core, checkpoint, profiler, advanced, scaffolding, plugins)
auto_register(app)


def _discover_and_register_plugins() -> None:
    """Discover and register CLI plugins from entry points.

    This function discovers plugins registered under the 'truthound.cli'
    entry point group. This allows external packages (like truthound-dashboard)
    to extend the CLI with additional commands.

    Entry point format in pyproject.toml:
        [project.entry-points."truthound.cli"]
        serve = "truthound_dashboard.cli:register_commands"

    The registered module must have either:
        - A `register_commands(app: typer.Typer)` function
        - An `app` attribute that is a Typer instance
    """
    import logging

    logger = logging.getLogger(__name__)

    try:
        from importlib.metadata import entry_points

        # Get entry points for truthound.cli group
        eps = entry_points(group="truthound.cli")

        for ep in eps:
            try:
                # Load the module
                module = ep.load()

                # Check for register_commands function
                if hasattr(module, "register_commands"):
                    module.register_commands(app)
                    logger.debug(f"Registered CLI plugin via register_commands: {ep.name}")

                # Check for app attribute (sub-typer)
                elif hasattr(module, "app"):
                    app.add_typer(module.app, name=ep.name)
                    logger.debug(f"Registered CLI plugin via app typer: {ep.name}")

                # Check if the module itself is a callable (register function)
                elif callable(module):
                    module(app)
                    logger.debug(f"Registered CLI plugin via callable: {ep.name}")

                else:
                    logger.warning(
                        f"CLI plugin '{ep.name}' has no register_commands, app, "
                        "or is not callable"
                    )

            except Exception as e:
                # Log but don't fail - plugins shouldn't break core functionality
                logger.debug(f"Failed to load CLI plugin '{ep.name}': {e}")

    except Exception as e:
        # Entry points not available or other error - silently continue
        logger.debug(f"Entry point discovery not available: {e}")


# Discover and register CLI plugins from entry points
# This runs at module load time to ensure plugins are available
# when the CLI is invoked
_discover_and_register_plugins()


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
