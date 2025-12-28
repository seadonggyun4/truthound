"""Checkpoint list command.

This module implements the `truthound checkpoint list` command for
listing available checkpoints.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Optional

import typer

from truthound.cli_modules.common.errors import error_boundary, require_file


@error_boundary
def list_cmd(
    config_file: Annotated[
        Optional[Path],
        typer.Option("--config", "-c", help="Checkpoint configuration file"),
    ] = None,
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format (console, json)"),
    ] = "console",
) -> None:
    """List available checkpoints.

    This command displays all checkpoints defined in a configuration file.

    Examples:
        truthound checkpoint list --config checkpoints.yaml
        truthound checkpoint list -c checkpoints.json --format json
    """
    from truthound.checkpoint import CheckpointRegistry

    try:
        registry = CheckpointRegistry()

        if config_file:
            require_file(config_file, "Config file")

            if config_file.suffix in (".yaml", ".yml"):
                registry.load_from_yaml(config_file)
            else:
                registry.load_from_json(config_file)

        checkpoints = registry.list_all()

        if not checkpoints:
            typer.echo("No checkpoints registered.")
            return

        if format == "json":
            result = json.dumps([cp.to_dict() for cp in checkpoints], indent=2)
            typer.echo(result)
        else:
            typer.echo(f"Checkpoints ({len(checkpoints)}):")
            for cp in checkpoints:
                typer.echo(f"  - {cp.name}")
                typer.echo(f"      Data: {cp.config.data_source}")
                typer.echo(f"      Actions: {len(cp.actions)}")
                typer.echo(f"      Triggers: {len(cp.triggers)}")

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
