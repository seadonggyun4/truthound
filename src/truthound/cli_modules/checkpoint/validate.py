"""Checkpoint validate command.

This module implements the `truthound checkpoint validate` command for
validating checkpoint configuration files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from truthound.cli_modules.common.errors import error_boundary, require_file


def _validate_config_structure(config: dict[str, Any]) -> list[str]:
    """Validate the raw configuration structure before loading.

    Args:
        config: Raw configuration dictionary.

    Returns:
        List of structural validation errors.
    """
    errors = []

    # Check for required top-level key
    if "checkpoints" not in config:
        errors.append("Missing required key 'checkpoints' at root level")
        return errors

    checkpoints = config["checkpoints"]

    # checkpoints can be a list or dict
    if isinstance(checkpoints, dict):
        checkpoints = [{"name": k, **v} for k, v in checkpoints.items()]
    elif not isinstance(checkpoints, list):
        errors.append(
            f"'checkpoints' must be a list or dict, got {type(checkpoints).__name__}"
        )
        return errors

    for i, cp in enumerate(checkpoints):
        prefix = f"checkpoints[{i}]"

        if not isinstance(cp, dict):
            errors.append(f"{prefix}: Expected dict, got {type(cp).__name__}")
            continue

        # Required fields
        if "name" not in cp:
            errors.append(f"{prefix}: Missing required field 'name'")

        if "data_source" not in cp:
            errors.append(f"{prefix}: Missing required field 'data_source'")

        # Validate validators field
        if "validators" in cp:
            validators = cp["validators"]
            if not isinstance(validators, list):
                errors.append(
                    f"{prefix}.validators: Expected list, got {type(validators).__name__}"
                )

        # Validate validator_config field
        if "validator_config" in cp:
            vc = cp["validator_config"]
            if not isinstance(vc, dict):
                errors.append(
                    f"{prefix}.validator_config: Expected dict, got {type(vc).__name__}"
                )
            else:
                for vname, vconf in vc.items():
                    if not isinstance(vconf, dict):
                        errors.append(
                            f"{prefix}.validator_config.{vname}: "
                            f"Expected dict, got {type(vconf).__name__}"
                        )

        # Validate actions field
        if "actions" in cp:
            actions = cp["actions"]
            if not isinstance(actions, list):
                errors.append(
                    f"{prefix}.actions: Expected list, got {type(actions).__name__}"
                )
            else:
                for j, action in enumerate(actions):
                    if not isinstance(action, dict):
                        errors.append(
                            f"{prefix}.actions[{j}]: Expected dict, got {type(action).__name__}"
                        )
                    elif "type" not in action:
                        errors.append(f"{prefix}.actions[{j}]: Missing required field 'type'")

        # Validate triggers field
        if "triggers" in cp:
            triggers = cp["triggers"]
            if not isinstance(triggers, list):
                errors.append(
                    f"{prefix}.triggers: Expected list, got {type(triggers).__name__}"
                )
            else:
                for j, trigger in enumerate(triggers):
                    if not isinstance(trigger, dict):
                        errors.append(
                            f"{prefix}.triggers[{j}]: Expected dict, got {type(trigger).__name__}"
                        )
                    elif "type" not in trigger:
                        errors.append(f"{prefix}.triggers[{j}]: Missing required field 'type'")

        # Validate min_severity if present
        if "min_severity" in cp:
            severity = cp["min_severity"]
            valid_severities = {"low", "medium", "high", "critical"}
            if not isinstance(severity, str):
                errors.append(
                    f"{prefix}.min_severity: Expected string, got {type(severity).__name__}"
                )
            elif severity.lower() not in valid_severities:
                errors.append(
                    f"{prefix}.min_severity: Invalid value '{severity}'. "
                    f"Must be one of: {', '.join(sorted(valid_severities))}"
                )

    return errors


@error_boundary
def validate_cmd(
    config_file: Annotated[
        Path,
        typer.Argument(help="Checkpoint configuration file to validate"),
    ],
    strict: Annotated[
        bool,
        typer.Option("--strict", "-s", help="Enable strict validation (check file existence)"),
    ] = False,
) -> None:
    """Validate a checkpoint configuration file.

    This command parses and validates a checkpoint configuration file,
    reporting any errors found. Validation includes:

    - YAML/JSON syntax validation
    - Configuration structure validation
    - Validator name validation
    - Severity value validation
    - Action and trigger configuration validation

    With --strict flag, also validates:
    - Data source file existence
    - Schema file existence

    Examples:
        truthound checkpoint validate checkpoints.yaml
        truthound checkpoint validate ci_config.json
        truthound checkpoint validate checkpoints.yaml --strict
    """
    import yaml

    from truthound.checkpoint import CheckpointRegistry

    try:
        require_file(config_file, "Config file")

        # Phase 1: Parse file
        typer.echo(f"Validating {config_file}...")
        typer.echo("")

        try:
            with open(config_file) as f:
                if config_file.suffix in (".yaml", ".yml"):
                    config = yaml.safe_load(f)
                else:
                    config = json.load(f)
        except yaml.YAMLError as e:
            typer.echo(f"YAML syntax error: {e}", err=True)
            raise typer.Exit(1)
        except json.JSONDecodeError as e:
            typer.echo(f"JSON syntax error: {e}", err=True)
            raise typer.Exit(1)

        if config is None:
            typer.echo("Error: Config file is empty", err=True)
            raise typer.Exit(1)

        # Phase 2: Validate structure
        structure_errors = _validate_config_structure(config)
        if structure_errors:
            typer.echo("Configuration structure errors:", err=True)
            for err in structure_errors:
                typer.echo(f"  - {err}", err=True)
            typer.echo("")
            raise typer.Exit(1)

        # Phase 3: Load and validate checkpoints
        registry = CheckpointRegistry()

        if config_file.suffix in (".yaml", ".yml"):
            checkpoints = registry.load_from_yaml(config_file)
        else:
            checkpoints = registry.load_from_json(config_file)

        if not checkpoints:
            typer.echo("Warning: No checkpoints defined in configuration", err=True)
            raise typer.Exit(1)

        all_valid = True
        total_errors = 0

        for cp in checkpoints:
            errors = cp.validate()

            # Filter out file existence errors unless strict mode
            if not strict:
                errors = [
                    e for e in errors
                    if "file not found" not in e.lower()
                ]

            if errors:
                all_valid = False
                total_errors += len(errors)
                typer.echo(f"[FAIL] Checkpoint '{cp.name}':")
                for err in errors:
                    typer.echo(f"       - {err}")
            else:
                typer.echo(f"[OK]   Checkpoint '{cp.name}'")

        typer.echo("")

        if all_valid:
            typer.echo(
                f"Validation passed: {len(checkpoints)} checkpoint(s) are valid."
            )
            if not strict:
                typer.echo("(Use --strict to also validate file existence)")
        else:
            typer.echo(
                f"Validation failed: {total_errors} error(s) found in "
                f"{len([cp for cp in checkpoints if cp.validate()])} checkpoint(s).",
                err=True,
            )
            raise typer.Exit(1)

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
