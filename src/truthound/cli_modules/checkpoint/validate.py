"""Checkpoint validate command.

This module implements the `truthound checkpoint validate` command for
validating checkpoint configuration files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from truthound.cli_modules.common.errors import error_boundary, require_file


@error_boundary
def validate_cmd(
    config_file: Annotated[
        Path,
        typer.Argument(help="Checkpoint configuration file to validate"),
    ],
) -> None:
    """Validate a checkpoint configuration file.

    This command parses and validates a checkpoint configuration file,
    reporting any errors found.

    Examples:
        truthound checkpoint validate checkpoints.yaml
        truthound checkpoint validate ci_config.json
    """
    from truthound.checkpoint import CheckpointRegistry

    try:
        require_file(config_file, "Config file")

        registry = CheckpointRegistry()

        if config_file.suffix in (".yaml", ".yml"):
            checkpoints = registry.load_from_yaml(config_file)
        else:
            checkpoints = registry.load_from_json(config_file)

        all_valid = True

        for cp in checkpoints:
            errors = cp.validate()
            if errors:
                all_valid = False
                typer.echo(f"Checkpoint '{cp.name}' has errors:")
                for err in errors:
                    typer.echo(f"  - {err}")
            else:
                typer.echo(f"Checkpoint '{cp.name}' is valid")

        if all_valid:
            typer.echo(f"\nAll {len(checkpoints)} checkpoint(s) are valid.")
        else:
            typer.echo("\nSome checkpoints have validation errors.", err=True)
            raise typer.Exit(1)

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
