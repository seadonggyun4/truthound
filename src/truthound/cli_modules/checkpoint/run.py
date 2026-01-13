"""Checkpoint run command.

This module implements the `truthound checkpoint run` command for
running checkpoint validation pipelines.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Optional

import typer

from truthound.cli_modules.common.errors import error_boundary, require_file
from truthound.cli_modules.common.options import parse_list_callback


@error_boundary
def run_cmd(
    name: Annotated[str, typer.Argument(help="Name of checkpoint to run")],
    config_file: Annotated[
        Optional[Path],
        typer.Option("--config", "-c", help="Checkpoint configuration file (YAML/JSON)"),
    ] = None,
    data_source: Annotated[
        Optional[Path],
        typer.Option("--data", "-d", help="Override data source path"),
    ] = None,
    validators: Annotated[
        Optional[list[str]],
        typer.Option("--validators", "-v", help="Override validators (comma-separated)"),
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output file for results (JSON)"),
    ] = None,
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format (console, json)"),
    ] = "console",
    strict: Annotated[
        bool,
        typer.Option("--strict", help="Exit with code 1 if issues are found"),
    ] = False,
    store_result: Annotated[
        Optional[Path],
        typer.Option("--store", help="Store results to directory"),
    ] = None,
    notify_slack: Annotated[
        Optional[str],
        typer.Option("--slack", help="Slack webhook URL for notifications"),
    ] = None,
    notify_webhook: Annotated[
        Optional[str],
        typer.Option("--webhook", help="Webhook URL for notifications"),
    ] = None,
    github_summary: Annotated[
        bool,
        typer.Option("--github-summary", help="Write GitHub Actions job summary"),
    ] = False,
) -> None:
    """Run a checkpoint validation pipeline.

    This command executes a configured checkpoint validation and optionally
    triggers actions like notifications or result storage.

    Examples:
        truthound checkpoint run daily_check --config checkpoints.yaml
        truthound checkpoint run quick_check --data data.csv --validators null,range
        truthound checkpoint run ci_check --strict --github-summary
        truthound checkpoint run prod_check --slack https://hooks.slack.com/...
    """
    from truthound.checkpoint import Checkpoint, CheckpointRegistry
    from truthound.checkpoint.actions import (
        StoreValidationResult,
        SlackNotification,
        WebhookAction,
        GitHubAction,
    )

    try:
        # Load from config file or create ad-hoc
        if config_file:
            require_file(config_file, "Config file")

            registry = CheckpointRegistry()
            if config_file.suffix in (".yaml", ".yml"):
                registry.load_from_yaml(config_file)
            else:
                registry.load_from_json(config_file)

            if name not in registry:
                typer.echo(f"Error: Checkpoint '{name}' not found in config", err=True)
                typer.echo(f"Available: {', '.join(registry.list_names())}")
                raise typer.Exit(1)

            checkpoint = registry.get(name)
        else:
            # Create ad-hoc checkpoint
            if not data_source:
                typer.echo("Error: --data is required when not using config file", err=True)
                raise typer.Exit(1)

            require_file(data_source, "Data file")

            validator_list = parse_list_callback(validators) if validators else None

            actions = []

            # Add actions based on CLI options
            if store_result:
                actions.append(StoreValidationResult(store_path=str(store_result)))

            if notify_slack:
                actions.append(SlackNotification(
                    webhook_url=notify_slack,
                    notify_on="failure",
                ))

            if notify_webhook:
                actions.append(WebhookAction(url=notify_webhook))

            if github_summary:
                actions.append(GitHubAction(
                    set_summary=True,
                    set_output=True,
                ))

            checkpoint = Checkpoint(
                name=name,
                data_source=str(data_source),
                validators=validator_list,
                actions=actions,
            )

        # Run checkpoint
        result = checkpoint.run()

        # Output results
        if format == "json":
            result_json = json.dumps(result.to_dict(), indent=2, default=str)
            if output:
                output.write_text(result_json)
                typer.echo(f"Results written to {output}")
            else:
                typer.echo(result_json)
        else:
            typer.echo(result.summary())

        # Exit code based on status
        # --strict: exit 1 if any issues are found (regardless of severity)
        if strict:
            stats = result.validation_result.statistics
            total_issues = getattr(stats, "total_issues", 0) if stats else 0
            if total_issues > 0 or result.status.value in ("failure", "error"):
                raise typer.Exit(1)

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
