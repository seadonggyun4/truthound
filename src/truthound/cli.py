"""Command-line interface for Truthound."""

import json
from pathlib import Path
from typing import Annotated, Optional

import typer

from truthound.api import check, mask, profile, scan
from truthound.drift import compare
from truthound.schema import learn

# Phase 7: Auto-profiling imports (lazy loaded to avoid startup overhead)

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


# =============================================================================
# Auto-Profiling Commands (Phase 7)
# =============================================================================


@app.command(name="auto-profile")
def auto_profile_cmd(
    file: Annotated[Path, typer.Argument(help="Path to the data file")],
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output file path for profile JSON"),
    ] = None,
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format (console, json, yaml)"),
    ] = "console",
    include_patterns: Annotated[
        bool,
        typer.Option("--patterns/--no-patterns", help="Include pattern detection"),
    ] = True,
    include_correlations: Annotated[
        bool,
        typer.Option("--correlations/--no-correlations", help="Include correlation analysis"),
    ] = False,
    sample_size: Annotated[
        Optional[int],
        typer.Option("--sample", "-s", help="Sample size for profiling (default: all rows)"),
    ] = None,
    top_n: Annotated[
        int,
        typer.Option("--top-n", help="Number of top/bottom values to include"),
    ] = 10,
) -> None:
    """Profile data with auto-detection of types and patterns.

    This performs comprehensive profiling including:
    - Column statistics (null ratio, unique ratio, distribution)
    - Type inference (email, phone, UUID, etc.)
    - Pattern detection
    - Suggested validation rules
    """
    if not file.exists():
        typer.echo(f"Error: File not found: {file}", err=True)
        raise typer.Exit(1)

    try:
        from truthound.profiler import (
            DataProfiler,
            ProfilerConfig,
            profile_file,
            save_profile,
        )

        config = ProfilerConfig(
            include_patterns=include_patterns,
            include_correlations=include_correlations,
            sample_size=sample_size,
            top_n_values=top_n,
        )

        profiler = DataProfiler(config=config)

        typer.echo(f"Profiling {file}...")
        profile_result = profiler.profile(
            _read_file_as_lazy(file),
            name=file.stem,
            source=str(file),
        )

        if format == "json":
            import json as json_mod
            result = json_mod.dumps(profile_result.to_dict(), indent=2, default=str)
            if output:
                with open(output, "w", encoding="utf-8") as f:
                    f.write(result)
                typer.echo(f"Profile saved to {output}")
            else:
                typer.echo(result)

        elif format == "yaml":
            import yaml
            result = yaml.dump(profile_result.to_dict(), default_flow_style=False)
            if output:
                with open(output, "w", encoding="utf-8") as f:
                    f.write(result)
                typer.echo(f"Profile saved to {output}")
            else:
                typer.echo(result)

        else:  # console
            _print_profile_summary(profile_result)
            if output:
                save_profile(profile_result, output)
                typer.echo(f"\nFull profile saved to {output}")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command(name="generate-suite")
def generate_suite_cmd(
    profile_file: Annotated[
        Path,
        typer.Argument(help="Path to profile JSON file (from auto-profile)"),
    ],
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output file path"),
    ] = None,
    format: Annotated[
        str,
        typer.Option(
            "--format", "-f",
            help="Output format (yaml, json, python, toml, checkpoint)",
        ),
    ] = "yaml",
    strictness: Annotated[
        str,
        typer.Option("--strictness", "-s", help="Rule strictness (loose, medium, strict)"),
    ] = "medium",
    include: Annotated[
        Optional[list[str]],
        typer.Option("--include", "-i", help="Include only these categories"),
    ] = None,
    exclude: Annotated[
        Optional[list[str]],
        typer.Option("--exclude", "-e", help="Exclude these categories"),
    ] = None,
    min_confidence: Annotated[
        Optional[str],
        typer.Option("--min-confidence", help="Minimum rule confidence (low, medium, high)"),
    ] = None,
    name: Annotated[
        Optional[str],
        typer.Option("--name", "-n", help="Name for the validation suite"),
    ] = None,
    preset: Annotated[
        Optional[str],
        typer.Option(
            "--preset", "-p",
            help="Configuration preset (default, strict, loose, minimal, comprehensive, ci_cd)",
        ),
    ] = None,
    config: Annotated[
        Optional[Path],
        typer.Option("--config", "-c", help="Path to configuration file"),
    ] = None,
    group_by_category: Annotated[
        bool,
        typer.Option("--group-by-category", help="Group rules by category in output"),
    ] = False,
    code_style: Annotated[
        str,
        typer.Option(
            "--code-style",
            help="Python code style (functional, class_based, declarative)",
        ),
    ] = "functional",
) -> None:
    """Generate validation rules from a profile.

    This creates a validation suite based on the data profile.
    Categories available: schema, completeness, uniqueness, format,
    distribution, pattern, temporal, relationship, anomaly

    Output formats:
        - yaml: Human-readable YAML (default)
        - json: Machine-readable JSON
        - python: Executable Python code
        - toml: TOML configuration
        - checkpoint: Truthound checkpoint format for CI/CD

    Examples:
        # Generate from profile
        truthound generate-suite profile.json -o rules.yaml

        # Only schema and format rules
        truthound generate-suite profile.json -i schema -i format

        # Strict mode with preset
        truthound generate-suite profile.json --preset strict

        # Generate Python code with class-based style
        truthound generate-suite profile.json -f python --code-style class_based

        # Generate CI/CD checkpoint
        truthound generate-suite profile.json -f checkpoint -o ci_rules.yaml

        # Use configuration file
        truthound generate-suite profile.json --config suite_config.yaml
    """
    if not profile_file.exists():
        typer.echo(f"Error: Profile file not found: {profile_file}", err=True)
        raise typer.Exit(1)

    try:
        from truthound.profiler import (
            run_generate_suite,
            get_available_formats,
            get_available_presets,
        )

        # Validate format
        available_formats = get_available_formats()
        if format not in available_formats:
            typer.echo(
                f"Error: Invalid format '{format}'. "
                f"Available: {', '.join(available_formats)}",
                err=True,
            )
            raise typer.Exit(1)

        # Validate preset
        if preset:
            available_presets = get_available_presets()
            if preset not in available_presets:
                typer.echo(
                    f"Error: Invalid preset '{preset}'. "
                    f"Available: {', '.join(available_presets)}",
                    err=True,
                )
                raise typer.Exit(1)

        # Parse categories
        include_cats = None
        if include:
            include_cats = [c.strip() for c in ",".join(include).split(",")]

        exclude_cats = None
        if exclude:
            exclude_cats = [c.strip() for c in ",".join(exclude).split(",")]

        # Run generation using the new handler
        exit_code = run_generate_suite(
            profile_file=profile_file,
            output=output,
            format=format,
            strictness=strictness,
            include=include_cats,
            exclude=exclude_cats,
            min_confidence=min_confidence,
            name=name,
            preset=preset,
            config=config,
            group_by_category=group_by_category,
            echo=typer.echo,
            verbose=True,
        )

        if exit_code != 0:
            raise typer.Exit(exit_code)

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command(name="quick-suite")
def quick_suite_cmd(
    file: Annotated[Path, typer.Argument(help="Path to the data file")],
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output file path"),
    ] = None,
    format: Annotated[
        str,
        typer.Option(
            "--format", "-f",
            help="Output format (yaml, json, python, toml, checkpoint)",
        ),
    ] = "yaml",
    strictness: Annotated[
        str,
        typer.Option("--strictness", "-s", help="Rule strictness (loose, medium, strict)"),
    ] = "medium",
    include: Annotated[
        Optional[list[str]],
        typer.Option("--include", "-i", help="Include only these categories"),
    ] = None,
    exclude: Annotated[
        Optional[list[str]],
        typer.Option("--exclude", "-e", help="Exclude these categories"),
    ] = None,
    min_confidence: Annotated[
        Optional[str],
        typer.Option("--min-confidence", help="Minimum rule confidence (low, medium, high)"),
    ] = None,
    name: Annotated[
        Optional[str],
        typer.Option("--name", "-n", help="Name for the validation suite"),
    ] = None,
    preset: Annotated[
        Optional[str],
        typer.Option(
            "--preset", "-p",
            help="Configuration preset (default, strict, loose, minimal, comprehensive, ci_cd)",
        ),
    ] = None,
    sample_size: Annotated[
        Optional[int],
        typer.Option("--sample-size", help="Sample size for profiling (default: auto)"),
    ] = None,
) -> None:
    """Profile data and generate validation rules in one step.

    This is a convenience command that combines auto-profile and generate-suite.

    Output formats:
        - yaml: Human-readable YAML (default)
        - json: Machine-readable JSON
        - python: Executable Python code
        - toml: TOML configuration
        - checkpoint: Truthound checkpoint format for CI/CD

    Examples:
        # Basic usage
        truthound quick-suite data.parquet -o rules.yaml

        # Strict mode with Python output
        truthound quick-suite data.csv -s strict -f python -o validators.py

        # CI/CD checkpoint
        truthound quick-suite data.parquet --preset ci_cd -o ci_rules.yaml

        # With sampling for large files
        truthound quick-suite large_data.parquet --sample-size 10000
    """
    if not file.exists():
        typer.echo(f"Error: File not found: {file}", err=True)
        raise typer.Exit(1)

    try:
        from truthound.profiler import (
            run_quick_suite,
            get_available_formats,
            get_available_presets,
        )

        # Validate format
        available_formats = get_available_formats()
        if format not in available_formats:
            typer.echo(
                f"Error: Invalid format '{format}'. "
                f"Available: {', '.join(available_formats)}",
                err=True,
            )
            raise typer.Exit(1)

        # Validate preset
        if preset:
            available_presets = get_available_presets()
            if preset not in available_presets:
                typer.echo(
                    f"Error: Invalid preset '{preset}'. "
                    f"Available: {', '.join(available_presets)}",
                    err=True,
                )
                raise typer.Exit(1)

        # Parse categories
        include_cats = None
        if include:
            include_cats = [c.strip() for c in ",".join(include).split(",")]

        exclude_cats = None
        if exclude:
            exclude_cats = [c.strip() for c in ",".join(exclude).split(",")]

        # Run quick suite using the new handler
        exit_code = run_quick_suite(
            file=file,
            output=output,
            format=format,
            strictness=strictness,
            include=include_cats,
            exclude=exclude_cats,
            min_confidence=min_confidence,
            name=name,
            preset=preset,
            sample_size=sample_size,
            echo=typer.echo,
            verbose=True,
        )

        if exit_code != 0:
            raise typer.Exit(exit_code)

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command(name="list-formats")
def list_formats_cmd() -> None:
    """List available output formats for suite generation."""
    try:
        from truthound.profiler import get_available_formats

        typer.echo("Available output formats:")
        typer.echo("")
        formats_info = {
            "yaml": "Human-readable YAML format (default)",
            "json": "Machine-readable JSON format",
            "python": "Executable Python code with validators",
            "toml": "TOML configuration format",
            "checkpoint": "Truthound checkpoint format for CI/CD",
        }

        for fmt in get_available_formats():
            desc = formats_info.get(fmt, "")
            typer.echo(f"  {fmt:12} - {desc}")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command(name="list-presets")
def list_presets_cmd() -> None:
    """List available configuration presets for suite generation."""
    try:
        from truthound.profiler import get_available_presets

        typer.echo("Available configuration presets:")
        typer.echo("")
        presets_info = {
            "default": "Balanced settings (medium strictness, all categories)",
            "strict": "Strict validation rules with high confidence",
            "loose": "Relaxed validation for flexible data",
            "minimal": "Only high-confidence schema rules",
            "comprehensive": "All generators with detailed output",
            "schema_only": "Schema and completeness rules only",
            "format_only": "Format and pattern rules only",
            "ci_cd": "Optimized for CI/CD pipelines (checkpoint format)",
            "development": "Development-friendly (Python code output)",
            "production": "Production-ready (strict, high confidence)",
        }

        for preset in get_available_presets():
            desc = presets_info.get(preset, "")
            typer.echo(f"  {preset:16} - {desc}")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command(name="list-categories")
def list_categories_cmd() -> None:
    """List available rule categories for suite generation."""
    try:
        from truthound.profiler import get_available_categories

        typer.echo("Available rule categories:")
        typer.echo("")
        categories_info = {
            "schema": "Column existence, types, and structure",
            "completeness": "Null values and data completeness",
            "uniqueness": "Unique constraints and cardinality",
            "format": "Data format validation (email, phone, etc.)",
            "distribution": "Statistical distribution checks",
            "pattern": "Regex pattern matching",
            "temporal": "Date/time validation",
            "relationship": "Cross-column relationships",
            "anomaly": "Anomaly detection rules",
        }

        for cat in get_available_categories():
            desc = categories_info.get(cat, "")
            typer.echo(f"  {cat:14} - {desc}")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


# =============================================================================
# Helper Functions
# =============================================================================


def _read_file_as_lazy(path: Path):
    """Read a file as a Polars LazyFrame."""
    import polars as pl

    suffix = path.suffix.lower()
    readers = {
        ".parquet": pl.scan_parquet,
        ".csv": pl.scan_csv,
        ".json": pl.scan_ndjson,
        ".ndjson": pl.scan_ndjson,
    }

    if suffix not in readers:
        raise ValueError(
            f"Unsupported file type: {suffix}. "
            f"Supported: {list(readers.keys())}"
        )

    return readers[suffix](path)


# =============================================================================
# Benchmark Commands
# =============================================================================

benchmark_app = typer.Typer(
    name="benchmark",
    help="Performance benchmarking commands",
)
app.add_typer(benchmark_app, name="benchmark")


@benchmark_app.command(name="run")
def benchmark_run_cmd(
    benchmark: Annotated[
        Optional[str],
        typer.Argument(help="Benchmark name to run (e.g., 'profile', 'check')"),
    ] = None,
    suite: Annotated[
        Optional[str],
        typer.Option("--suite", "-s", help="Predefined suite to run (quick, ci, full, profiling, validation)"),
    ] = None,
    size: Annotated[
        str,
        typer.Option("--size", help="Data size (tiny, small, medium, large, xlarge)"),
    ] = "medium",
    rows: Annotated[
        Optional[int],
        typer.Option("--rows", "-r", help="Custom row count (overrides size)"),
    ] = None,
    iterations: Annotated[
        int,
        typer.Option("--iterations", "-i", help="Number of measurement iterations"),
    ] = 5,
    warmup: Annotated[
        int,
        typer.Option("--warmup", "-w", help="Number of warmup iterations"),
    ] = 2,
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output file path"),
    ] = None,
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format (console, json, markdown, html)"),
    ] = "console",
    save_baseline: Annotated[
        bool,
        typer.Option("--save-baseline", help="Save results as baseline for regression detection"),
    ] = False,
    compare_baseline: Annotated[
        bool,
        typer.Option("--compare-baseline", help="Compare against saved baseline"),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Verbose output"),
    ] = False,
) -> None:
    """Run performance benchmarks.

    Examples:
        # Run a single benchmark
        truthound benchmark run profile --size medium

        # Run a predefined suite
        truthound benchmark run --suite quick

        # Run with custom row count
        truthound benchmark run check --rows 1000000

        # Save as baseline
        truthound benchmark run --suite ci --save-baseline

        # Compare against baseline
        truthound benchmark run --suite ci --compare-baseline
    """
    from truthound.benchmark import (
        BenchmarkRunner,
        BenchmarkSuite,
        BenchmarkConfig,
        BenchmarkSize,
        RunnerConfig,
        ConsoleReporter,
        JSONReporter,
        MarkdownReporter,
        HTMLReporter,
        RegressionDetector,
    )

    try:
        # Determine row count
        size_map = {
            "tiny": BenchmarkSize.TINY,
            "small": BenchmarkSize.SMALL,
            "medium": BenchmarkSize.MEDIUM,
            "large": BenchmarkSize.LARGE,
            "xlarge": BenchmarkSize.XLARGE,
        }
        benchmark_size = size_map.get(size, BenchmarkSize.MEDIUM)
        row_count = rows if rows else benchmark_size.row_count

        # Configure benchmark
        benchmark_config = BenchmarkConfig(
            warmup_iterations=warmup,
            measure_iterations=iterations,
            default_size=benchmark_size,
            verbose=verbose,
        )

        runner_config = RunnerConfig(
            size_override=benchmark_size if not rows else None,
            verbose=verbose,
        )

        runner = BenchmarkRunner(
            config=runner_config,
            benchmark_config=benchmark_config,
        )

        # Determine what to run
        if suite:
            suite_map = {
                "quick": BenchmarkSuite.quick,
                "ci": BenchmarkSuite.ci,
                "full": lambda: BenchmarkSuite.full(benchmark_size),
                "profiling": lambda: BenchmarkSuite.profiling(benchmark_size),
                "validation": lambda: BenchmarkSuite.validation(benchmark_size),
            }
            if suite not in suite_map:
                typer.echo(f"Unknown suite: {suite}. Available: {list(suite_map.keys())}", err=True)
                raise typer.Exit(1)

            benchmark_suite = suite_map[suite]()
            results = runner.run_suite(benchmark_suite)

        elif benchmark:
            result = runner.run(benchmark, row_count=row_count)
            # Wrap single result in suite result for consistent handling
            from truthound.benchmark.base import EnvironmentInfo
            from truthound.benchmark.runner import SuiteResult
            results = SuiteResult(
                suite_name=f"single:{benchmark}",
                results=[result],
                environment=EnvironmentInfo.capture(),
            )
            results.completed_at = result.completed_at

        else:
            typer.echo("Specify either a benchmark name or --suite", err=True)
            raise typer.Exit(1)

        # Compare against baseline if requested
        if compare_baseline:
            detector = RegressionDetector()
            report = detector.generate_report(results)
            typer.echo(report)

            regressions = detector.check(results)
            if regressions:
                typer.echo("\n⚠️  Performance regressions detected!", err=True)
                raise typer.Exit(1)

        # Generate output
        reporters = {
            "console": ConsoleReporter(use_colors=True),
            "json": JSONReporter(pretty=True),
            "markdown": MarkdownReporter(),
            "html": HTMLReporter(),
        }

        reporter = reporters.get(format, ConsoleReporter())
        report_content = reporter.report_suite(results)

        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(report_content)
            typer.echo(f"Results saved to: {output}")
        elif format == "console":
            typer.echo(report_content)
        else:
            typer.echo(report_content)

        # Save baseline if requested
        if save_baseline:
            detector = RegressionDetector()
            detector.save_baseline(results)
            typer.echo(f"Baseline saved to: {detector.history_path}")

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        if verbose:
            import traceback
            typer.echo(traceback.format_exc(), err=True)
        raise typer.Exit(1)


@benchmark_app.command(name="list")
def benchmark_list_cmd(
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format (console, json)"),
    ] = "console",
) -> None:
    """List available benchmarks."""
    from truthound.benchmark import benchmark_registry

    benchmarks = benchmark_registry.list_all()

    if format == "json":
        data = [
            {
                "name": b.name,
                "category": b.category.value,
                "description": b.description,
            }
            for b in benchmarks
        ]
        typer.echo(json.dumps(data, indent=2))
    else:
        typer.echo("\nAvailable Benchmarks:")
        typer.echo("=" * 60)

        # Group by category
        from collections import defaultdict
        by_category = defaultdict(list)
        for b in benchmarks:
            by_category[b.category.value].append(b)

        for category in sorted(by_category.keys()):
            typer.echo(f"\n[{category.upper()}]")
            for b in by_category[category]:
                typer.echo(f"  {b.name:20} - {b.description}")


@benchmark_app.command(name="compare")
def benchmark_compare_cmd(
    baseline: Annotated[
        Path,
        typer.Argument(help="Baseline results JSON file"),
    ],
    current: Annotated[
        Path,
        typer.Argument(help="Current results JSON file"),
    ],
    threshold: Annotated[
        float,
        typer.Option("--threshold", "-t", help="Regression threshold percentage"),
    ] = 10.0,
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format (console, json, markdown)"),
    ] = "console",
) -> None:
    """Compare two benchmark results for regressions."""
    from truthound.benchmark import BenchmarkComparator
    from truthound.benchmark.runner import SuiteResult

    if not baseline.exists():
        typer.echo(f"Baseline file not found: {baseline}", err=True)
        raise typer.Exit(1)

    if not current.exists():
        typer.echo(f"Current file not found: {current}", err=True)
        raise typer.Exit(1)

    try:
        baseline_data = json.loads(baseline.read_text())
        current_data = json.loads(current.read_text())

        comparator = BenchmarkComparator(regression_threshold=threshold / 100)

        # This is a simplified comparison - full implementation would
        # reconstruct SuiteResult objects
        typer.echo("\nBenchmark Comparison")
        typer.echo("=" * 60)
        typer.echo(f"Baseline: {baseline}")
        typer.echo(f"Current:  {current}")
        typer.echo(f"Threshold: {threshold}%")
        typer.echo("-" * 60)

        baseline_results = {r["benchmark_name"]: r for r in baseline_data.get("results", [])}
        current_results = {r["benchmark_name"]: r for r in current_data.get("results", [])}

        regressions = []
        improvements = []

        for name, curr in current_results.items():
            if name not in baseline_results:
                continue

            base = baseline_results[name]
            base_duration = base["metrics"]["timing"]["mean_seconds"]
            curr_duration = curr["metrics"]["timing"]["mean_seconds"]

            if base_duration > 0:
                pct_change = ((curr_duration - base_duration) / base_duration) * 100

                if pct_change > threshold:
                    regressions.append((name, base_duration, curr_duration, pct_change))
                elif pct_change < -threshold:
                    improvements.append((name, base_duration, curr_duration, pct_change))

        if regressions:
            typer.echo("\n🔴 REGRESSIONS:")
            for name, base_d, curr_d, pct in regressions:
                typer.echo(f"  {name}: {base_d:.3f}s -> {curr_d:.3f}s ({pct:+.1f}%)")

        if improvements:
            typer.echo("\n🟢 IMPROVEMENTS:")
            for name, base_d, curr_d, pct in improvements:
                typer.echo(f"  {name}: {base_d:.3f}s -> {curr_d:.3f}s ({pct:+.1f}%)")

        if not regressions and not improvements:
            typer.echo("\n✅ No significant changes detected.")

        typer.echo("")

        if regressions:
            raise typer.Exit(1)

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


def _print_profile_summary(profile) -> None:
    """Print a summary of the profile to console."""
    typer.echo(f"\n{'='*60}")
    typer.echo(f"Profile: {profile.name or 'unnamed'}")
    typer.echo(f"{'='*60}")
    typer.echo(f"Rows: {profile.row_count:,}")
    typer.echo(f"Columns: {profile.column_count}")
    typer.echo(f"Estimated Memory: {profile.estimated_memory_bytes / 1024 / 1024:.2f} MB")

    if profile.duplicate_row_ratio > 0:
        typer.echo(f"Duplicate Rows: {profile.duplicate_row_count:,} ({profile.duplicate_row_ratio*100:.1f}%)")

    typer.echo(f"\n{'Column Details':^60}")
    typer.echo("-" * 60)

    for col in profile.columns:
        typer.echo(f"\n{col.name}")
        typer.echo(f"  Type: {col.physical_type} -> {col.inferred_type.value}")
        typer.echo(f"  Nulls: {col.null_count:,} ({col.null_ratio*100:.1f}%)")
        typer.echo(f"  Unique: {col.distinct_count:,} ({col.unique_ratio*100:.1f}%)")

        if col.distribution:
            dist = col.distribution
            typer.echo(f"  Range: [{dist.min}, {dist.max}]")
            if dist.mean is not None:
                typer.echo(f"  Mean: {dist.mean:.2f}, Std: {dist.std:.2f}")

        if col.min_length is not None:
            typer.echo(f"  Length: [{col.min_length}, {col.max_length}], avg={col.avg_length:.1f}")

        if col.detected_patterns:
            patterns = [p.pattern for p in col.detected_patterns[:3]]
            typer.echo(f"  Patterns: {', '.join(patterns)}")

        if col.suggested_validators:
            typer.echo(f"  Suggested: {len(col.suggested_validators)} validators")


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
