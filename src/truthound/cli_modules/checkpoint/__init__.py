"""Checkpoint CLI commands for Truthound.

This package contains CI/CD integration commands:
    - run: Run a checkpoint validation pipeline
    - list: List available checkpoints
    - validate: Validate checkpoint configuration
    - init: Initialize sample checkpoint configuration
"""

import typer

from truthound.cli_modules.checkpoint.run import run_cmd
from truthound.cli_modules.checkpoint.list import list_cmd
from truthound.cli_modules.checkpoint.validate import validate_cmd
from truthound.cli_modules.checkpoint.init import init_cmd

# Checkpoint app for subcommands
app = typer.Typer(
    name="checkpoint",
    help="Checkpoint and CI/CD integration commands",
)

# Register subcommands
app.command(name="run")(run_cmd)
app.command(name="list")(list_cmd)
app.command(name="validate")(validate_cmd)
app.command(name="init")(init_cmd)


def register_commands(parent_app: typer.Typer) -> None:
    """Register checkpoint commands with the parent app.

    Args:
        parent_app: Parent Typer app to register commands to
    """
    parent_app.add_typer(app, name="checkpoint")


__all__ = [
    "app",
    "register_commands",
    "run_cmd",
    "list_cmd",
    "validate_cmd",
    "init_cmd",
]
