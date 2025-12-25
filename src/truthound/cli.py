"""Command-line interface for Truthound."""

import json
from pathlib import Path
from typing import Annotated, Optional

import typer

from truthound.api import check, mask, profile, scan
from truthound.drift import compare
from truthound.schema import learn

app = typer.Typer(
    name="truthound",
    help="Zero-configuration data quality toolkit powered by Polars",
    add_completion=False,
)

# Create checkpoint subcommand group
checkpoint_app = typer.Typer(
    name="checkpoint",
    help="Checkpoint and CI/CD integration commands",
)
app.add_typer(checkpoint_app, name="checkpoint")


@app.command(name="learn")
def learn_cmd(
    file: Annotated[Path, typer.Argument(help="Path to the data file to learn from")],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output schema file path"),
    ] = Path("schema.yaml"),
    no_constraints: Annotated[
        bool,
        typer.Option("--no-constraints", help="Don't infer constraints from data"),
    ] = False,
) -> None:
    """Learn schema from a data file."""
    if not file.exists():
        typer.echo(f"Error: File not found: {file}", err=True)
        raise typer.Exit(1)

    try:
        schema = learn(str(file), infer_constraints=not no_constraints)
        schema.save(output)
        typer.echo(f"Schema saved to {output}")
        typer.echo(f"  Columns: {len(schema.columns)}")
        typer.echo(f"  Rows: {schema.row_count:,}")
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command(name="check")
def check_cmd(
    file: Annotated[Path, typer.Argument(help="Path to the data file")],
    validators: Annotated[
        Optional[list[str]],
        typer.Option("--validators", "-v", help="Comma-separated list of validators"),
    ] = None,
    min_severity: Annotated[
        Optional[str],
        typer.Option("--min-severity", "-s", help="Minimum severity level (low, medium, high, critical)"),
    ] = None,
    schema_file: Annotated[
        Optional[Path],
        typer.Option("--schema", help="Schema file for validation"),
    ] = None,
    auto_schema: Annotated[
        bool,
        typer.Option("--auto-schema", help="Auto-learn and cache schema (zero-config mode)"),
    ] = False,
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format (console, json, html)"),
    ] = "console",
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output file path"),
    ] = None,
    strict: Annotated[
        bool,
        typer.Option("--strict", help="Exit with code 1 if issues are found"),
    ] = False,
) -> None:
    """Validate data quality in a file."""
    if not file.exists():
        typer.echo(f"Error: File not found: {file}", err=True)
        raise typer.Exit(1)

    if schema_file and not schema_file.exists():
        typer.echo(f"Error: Schema file not found: {schema_file}", err=True)
        raise typer.Exit(1)

    # Parse validators if provided
    validator_list = None
    if validators:
        validator_list = [v.strip() for v in ",".join(validators).split(",")]

    try:
        report = check(
            str(file),
            validators=validator_list,
            min_severity=min_severity,
            schema=schema_file,
            auto_schema=auto_schema,
        )
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    # Output the report
    if format == "json":
        result = report.to_json()
        if output:
            output.write_text(result)
            typer.echo(f"Report written to {output}")
        else:
            typer.echo(result)
    elif format == "html":
        if not output:
            typer.echo("Error: --output is required for HTML format", err=True)
            raise typer.Exit(1)
        # HTML output requires jinja2
        try:
            from truthound.html_report import generate_html_report

            html = generate_html_report(report)
            output.write_text(html)
            typer.echo(f"HTML report written to {output}")
        except ImportError:
            typer.echo("Error: HTML reports require jinja2. Install with: pip install truthound[reports]", err=True)
            raise typer.Exit(1)
    else:
        report.print()

    # Exit with error if strict mode and issues found
    if strict and report.has_issues:
        raise typer.Exit(1)


@app.command(name="scan")
def scan_cmd(
    file: Annotated[Path, typer.Argument(help="Path to the data file")],
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format (console, json)"),
    ] = "console",
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output file path"),
    ] = None,
) -> None:
    """Scan for personally identifiable information (PII)."""
    if not file.exists():
        typer.echo(f"Error: File not found: {file}", err=True)
        raise typer.Exit(1)

    try:
        pii_report = scan(str(file))
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    if format == "json":
        result = pii_report.to_json()
        if output:
            output.write_text(result)
            typer.echo(f"Report written to {output}")
        else:
            typer.echo(result)
    else:
        pii_report.print()


@app.command(name="mask")
def mask_cmd(
    file: Annotated[Path, typer.Argument(help="Path to the data file")],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output file path"),
    ],
    columns: Annotated[
        Optional[list[str]],
        typer.Option("--columns", "-c", help="Columns to mask (comma-separated)"),
    ] = None,
    strategy: Annotated[
        str,
        typer.Option("--strategy", "-s", help="Masking strategy (redact, hash, fake)"),
    ] = "redact",
) -> None:
    """Mask sensitive data in a file."""
    if not file.exists():
        typer.echo(f"Error: File not found: {file}", err=True)
        raise typer.Exit(1)

    # Parse columns if provided
    column_list = None
    if columns:
        column_list = [c.strip() for c in ",".join(columns).split(",")]

    try:
        masked_df = mask(str(file), columns=column_list, strategy=strategy)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    # Write output based on file extension
    suffix = output.suffix.lower()
    if suffix == ".csv":
        masked_df.write_csv(output)
    elif suffix == ".parquet":
        masked_df.write_parquet(output)
    elif suffix == ".json":
        masked_df.write_json(output)
    else:
        masked_df.write_csv(output)

    typer.echo(f"Masked data written to {output}")


@app.command(name="profile")
def profile_cmd(
    file: Annotated[Path, typer.Argument(help="Path to the data file")],
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format (console, json)"),
    ] = "console",
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output file path"),
    ] = None,
) -> None:
    """Generate a statistical profile of the data."""
    if not file.exists():
        typer.echo(f"Error: File not found: {file}", err=True)
        raise typer.Exit(1)

    try:
        profile_report = profile(str(file))
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    if format == "json":
        result = profile_report.to_json()
        if output:
            output.write_text(result)
            typer.echo(f"Profile written to {output}")
        else:
            typer.echo(result)
    else:
        profile_report.print()


@app.command(name="compare")
def compare_cmd(
    baseline: Annotated[Path, typer.Argument(help="Baseline (reference) data file")],
    current: Annotated[Path, typer.Argument(help="Current data file to compare")],
    columns: Annotated[
        Optional[list[str]],
        typer.Option("--columns", "-c", help="Columns to compare (comma-separated)"),
    ] = None,
    method: Annotated[
        str,
        typer.Option("--method", "-m", help="Detection method (auto, ks, psi, chi2, js)"),
    ] = "auto",
    threshold: Annotated[
        Optional[float],
        typer.Option("--threshold", "-t", help="Custom drift threshold"),
    ] = None,
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format (console, json)"),
    ] = "console",
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output file path"),
    ] = None,
    strict: Annotated[
        bool,
        typer.Option("--strict", help="Exit with code 1 if drift is detected"),
    ] = False,
) -> None:
    """Compare two datasets and detect data drift."""
    if not baseline.exists():
        typer.echo(f"Error: Baseline file not found: {baseline}", err=True)
        raise typer.Exit(1)

    if not current.exists():
        typer.echo(f"Error: Current file not found: {current}", err=True)
        raise typer.Exit(1)

    # Parse columns if provided
    column_list = None
    if columns:
        column_list = [c.strip() for c in ",".join(columns).split(",")]

    try:
        drift_report = compare(
            str(baseline),
            str(current),
            columns=column_list,
            method=method,
            threshold=threshold,
        )
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    if format == "json":
        result = drift_report.to_json()
        if output:
            output.write_text(result)
            typer.echo(f"Drift report written to {output}")
        else:
            typer.echo(result)
    else:
        drift_report.print()

    # Exit with error if strict mode and drift found
    if strict and drift_report.has_drift:
        raise typer.Exit(1)


# =============================================================================
# Checkpoint Commands
# =============================================================================


@checkpoint_app.command(name="run")
def checkpoint_run_cmd(
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
    """Run a checkpoint validation pipeline."""
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
            if not config_file.exists():
                typer.echo(f"Error: Config file not found: {config_file}", err=True)
                raise typer.Exit(1)

            registry = CheckpointRegistry()
            registry.load_from_yaml(config_file) if config_file.suffix in (".yaml", ".yml") else registry.load_from_json(config_file)

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

            if not data_source.exists():
                typer.echo(f"Error: Data file not found: {data_source}", err=True)
                raise typer.Exit(1)

            validator_list = None
            if validators:
                validator_list = [v.strip() for v in ",".join(validators).split(",")]

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
        if strict and result.status.value in ("failure", "error"):
            raise typer.Exit(1)

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@checkpoint_app.command(name="list")
def checkpoint_list_cmd(
    config_file: Annotated[
        Optional[Path],
        typer.Option("--config", "-c", help="Checkpoint configuration file"),
    ] = None,
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format (console, json)"),
    ] = "console",
) -> None:
    """List available checkpoints."""
    from truthound.checkpoint import CheckpointRegistry

    try:
        registry = CheckpointRegistry()

        if config_file:
            if not config_file.exists():
                typer.echo(f"Error: Config file not found: {config_file}", err=True)
                raise typer.Exit(1)

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


@checkpoint_app.command(name="validate")
def checkpoint_validate_cmd(
    config_file: Annotated[
        Path,
        typer.Argument(help="Checkpoint configuration file to validate"),
    ],
) -> None:
    """Validate a checkpoint configuration file."""
    from truthound.checkpoint import CheckpointRegistry

    try:
        if not config_file.exists():
            typer.echo(f"Error: Config file not found: {config_file}", err=True)
            raise typer.Exit(1)

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


@checkpoint_app.command(name="init")
def checkpoint_init_cmd(
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output file path"),
    ] = Path("truthound.yaml"),
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Config format (yaml, json)"),
    ] = "yaml",
) -> None:
    """Initialize a sample checkpoint configuration file."""
    import yaml

    sample_config = {
        "checkpoints": [
            {
                "name": "daily_data_validation",
                "data_source": "data/production.csv",
                "validators": ["null", "duplicate", "range", "regex"],
                "min_severity": "medium",
                "auto_schema": True,
                "tags": {
                    "environment": "production",
                    "team": "data-platform",
                },
                "actions": [
                    {
                        "type": "store_result",
                        "store_path": "./truthound_results",
                        "partition_by": "date",
                    },
                    {
                        "type": "update_docs",
                        "site_path": "./truthound_docs",
                        "include_history": True,
                    },
                    {
                        "type": "slack",
                        "webhook_url": "https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
                        "notify_on": "failure",
                        "channel": "#data-quality",
                    },
                ],
                "triggers": [
                    {
                        "type": "schedule",
                        "interval_hours": 24,
                        "run_on_weekdays": [0, 1, 2, 3, 4],  # Mon-Fri
                    },
                ],
            },
            {
                "name": "hourly_metrics_check",
                "data_source": "data/metrics.parquet",
                "validators": ["null", "range"],
                "actions": [
                    {
                        "type": "webhook",
                        "url": "https://api.example.com/data-quality/events",
                        "auth_type": "bearer",
                        "auth_credentials": {"token": "${API_TOKEN}"},
                    },
                ],
                "triggers": [
                    {
                        "type": "cron",
                        "expression": "0 * * * *",  # Every hour
                    },
                ],
            },
        ],
    }

    if format == "json":
        output = output.with_suffix(".json")
        output.write_text(json.dumps(sample_config, indent=2))
    else:
        output = output.with_suffix(".yaml")
        import yaml
        output.write_text(yaml.dump(sample_config, default_flow_style=False, sort_keys=False))

    typer.echo(f"Sample checkpoint config created: {output}")
    typer.echo("\nEdit the file to configure your checkpoints, then run:")
    typer.echo(f"  truthound checkpoint run <checkpoint_name> --config {output}")


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
