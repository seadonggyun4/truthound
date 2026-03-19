"""Check command - Validate data quality.

This module implements the ``truthound check`` command for validating
data quality in files and database tables.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Optional

import typer

from truthound.cli_modules.common.datasource import (
    ConnectionOpt,
    QueryOpt,
    SourceConfigOpt,
    SourceNameOpt,
    TableOpt,
    resolve_datasource,
)
from truthound.cli_modules.common.errors import error_boundary, require_file
from truthound.cli_modules.common.options import parse_list_callback


@error_boundary
def check_cmd(
    file: Annotated[
        Optional[Path],
        typer.Argument(
            help="Path to the data file (CSV, JSON, Parquet, NDJSON)",
        ),
    ] = None,
    # -- DataSource Options --
    connection: ConnectionOpt = None,
    table: TableOpt = None,
    query: QueryOpt = None,
    source_config: SourceConfigOpt = None,
    source_name: SourceNameOpt = None,
    # -- Validator Options --
    validators: Annotated[
        Optional[list[str]],
        typer.Option("--validators", "-v", help="Comma-separated list of validators"),
    ] = None,
    min_severity: Annotated[
        Optional[str],
        typer.Option(
            "--min-severity",
            "-s",
            help="Minimum severity level (low, medium, high, critical)",
        ),
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
    result_format: Annotated[
        str,
        typer.Option(
            "--result-format",
            "--rf",
            help="Result detail level: boolean_only, basic, summary, complete",
        ),
    ] = "summary",
    include_unexpected_rows: Annotated[
        bool,
        typer.Option("--include-unexpected-rows", help="Include failure rows (SUMMARY+)"),
    ] = False,
    max_unexpected_rows: Annotated[
        int,
        typer.Option("--max-unexpected-rows", help="Max failure rows to return"),
    ] = 1000,
    partial_unexpected_count: Annotated[
        int,
        typer.Option(
            "--partial-unexpected-count",
            help="Maximum number of unexpected values in partial list (BASIC+)",
        ),
    ] = 20,
    include_unexpected_index: Annotated[
        bool,
        typer.Option(
            "--include-unexpected-index",
            help="Include row index for each unexpected value in results",
        ),
    ] = False,
    return_debug_query: Annotated[
        bool,
        typer.Option(
            "--return-debug-query",
            help="Include Polars debug query expression in results (COMPLETE level)",
        ),
    ] = False,
    catch_exceptions: Annotated[
        bool,
        typer.Option(
            "--catch-exceptions/--no-catch-exceptions",
            help="Catch validator exceptions (default: True). Use --no-catch-exceptions to abort on first error.",
        ),
    ] = True,
    max_retries: Annotated[
        int,
        typer.Option("--max-retries", help="Number of retry attempts for transient errors"),
    ] = 0,
    show_exceptions: Annotated[
        bool,
        typer.Option("--show-exceptions", help="Show exception details in console output"),
    ] = False,
    exclude_columns: Annotated[
        Optional[list[str]],
        typer.Option(
            "--exclude-columns",
            "-e",
            help="Columns to exclude from all validators (comma-separated)",
        ),
    ] = None,
    validator_config: Annotated[
        Optional[str],
        typer.Option(
            "--validator-config",
            "-vc",
            help=(
                "Validator configuration as JSON string or path to JSON/YAML file. "
                'Example: \'{"unique": {"exclude_columns": ["first_name"]}}\''
            ),
        ),
    ] = None,
    # -- Execution Options --
    parallel: Annotated[
        bool,
        typer.Option(
            "--parallel/--no-parallel",
            help=(
                "Enable DAG-based parallel execution. "
                "Validators are grouped by dependency level and executed concurrently."
            ),
        ),
    ] = False,
    max_workers: Annotated[
        Optional[int],
        typer.Option(
            "--max-workers",
            help=(
                "Maximum worker threads for parallel execution. "
                "Only effective with --parallel. Defaults to min(32, cpu_count + 4)."
            ),
            min=1,
        ),
    ] = None,
    pushdown: Annotated[
        Optional[bool],
        typer.Option(
            "--pushdown/--no-pushdown",
            help=(
                "Enable query pushdown for SQL data sources. "
                "Validation logic is executed server-side when possible. "
                "Default: auto-detect based on data source type."
            ),
        ),
    ] = None,
) -> None:
    """Validate data quality in a file or database table.

    This command runs data quality validators on the specified data
    and reports any issues found. Supports file paths, database
    connections, and source config files.

    Examples:
        truthound check data.csv
        truthound check data.parquet --validators null,duplicate,range
        truthound check data.csv --min-severity high --strict
        truthound check data.csv --auto-schema
        truthound check data.csv --format json -o report.json
        truthound check data.csv --result-format complete
        truthound check --connection "postgresql://user:pass@host/db" --table users
        truthound check --conn "sqlite:///data.db" --table orders --pushdown
        truthound check --source-config db.yaml --strict
        truthound check data.csv --parallel --max-workers 8
        truthound check data.csv --exclude-columns first_name,last_name
    """
    from truthound.api import check
    from truthound.types import ResultFormatConfig, ResultFormat

    # Resolve data source
    data_path, source = resolve_datasource(
        file=file,
        connection=connection,
        table=table,
        query=query,
        source_config=source_config,
        source_name=source_name,
    )

    if schema_file:
        require_file(schema_file, "Schema file")

    # Parse validators if provided
    validator_list = parse_list_callback(validators) if validators else None

    # Parse exclude_columns if provided
    exclude_cols = parse_list_callback(exclude_columns) if exclude_columns else None

    # Parse validator_config (JSON string or file path)
    v_config: dict[str, dict[str, Any]] | None = None
    if validator_config:
        import json

        config_str = validator_config.strip()
        # Check if it's a file path
        if config_str.endswith((".json", ".yaml", ".yml")):
            config_path = Path(config_str)
            if not config_path.exists():
                typer.echo(f"Error: Config file not found: {config_path}", err=True)
                raise typer.Exit(1)
            content = config_path.read_text(encoding="utf-8")
            if config_str.endswith(".json"):
                try:
                    v_config = json.loads(content)
                except json.JSONDecodeError as e:
                    typer.echo(f"Error: Invalid JSON in config file: {e}", err=True)
                    raise typer.Exit(1)
            else:
                try:
                    import yaml
                    v_config = yaml.safe_load(content)
                except ImportError:
                    typer.echo(
                        "Error: YAML config requires PyYAML. "
                        "Install with: pip install pyyaml",
                        err=True,
                    )
                    raise typer.Exit(1)
                except Exception as e:
                    typer.echo(f"Error: Invalid YAML in config file: {e}", err=True)
                    raise typer.Exit(1)
        else:
            # Parse as inline JSON string
            try:
                v_config = json.loads(config_str)
            except json.JSONDecodeError as e:
                typer.echo(f"Error: Invalid JSON in --validator-config: {e}", err=True)
                raise typer.Exit(1)

        if not isinstance(v_config, dict):
            typer.echo("Error: --validator-config must be a JSON object", err=True)
            raise typer.Exit(1)

    # Build result_format config — include all fine-grained parameters
    has_custom_rf = (
        include_unexpected_rows
        or max_unexpected_rows != 1000
        or partial_unexpected_count != 20
        or include_unexpected_index
        or return_debug_query
    )
    rf_config: str | ResultFormatConfig
    if has_custom_rf:
        rf_config = ResultFormatConfig(
            format=ResultFormat.from_string(result_format),
            partial_unexpected_count=partial_unexpected_count,
            include_unexpected_rows=include_unexpected_rows,
            max_unexpected_rows=max_unexpected_rows,
            include_unexpected_index=include_unexpected_index,
            return_debug_query=return_debug_query,
        )
    else:
        rf_config = result_format

    # Build API call kwargs
    check_kwargs: dict[str, Any] = {
        "validators": validator_list,
        "validator_config": v_config,
        "min_severity": min_severity,
        "schema": schema_file,
        "auto_schema": auto_schema,
        "result_format": rf_config,
        "catch_exceptions": catch_exceptions,
        "max_retries": max_retries,
        "exclude_columns": exclude_cols,
        "parallel": parallel,
        "max_workers": max_workers,
        "pushdown": pushdown,
    }

    try:
        if source is not None:
            run_result = check(source=source, **check_kwargs)
        else:
            run_result = check(data_path, **check_kwargs)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    # Determine label for HTML report title
    report_label = source_name or (source.name if source else str(file))

    # Output the report
    if format == "json":
        result = run_result.to_json()
        if output:
            output.write_text(result)
            typer.echo(f"Report written to {output}")
        else:
            typer.echo(result)

    elif format == "html":
        if not output:
            typer.echo("Error: --output is required for HTML format", err=True)
            raise typer.Exit(1)
        try:
            from truthound.html_reporter import generate_html_report

            html = generate_html_report(run_result, title=f"Validation Report: {report_label}")
            output.write_text(html, encoding="utf-8")
            typer.echo(f"HTML report written to {output}")
        except ImportError as e:
            error_msg = str(e)
            if "jinja2" in error_msg.lower():
                typer.echo(
                    "Error: HTML reports require jinja2. "
                    "Install with: pip install truthound[reports] or pip install jinja2",
                    err=True,
                )
            else:
                typer.echo(f"Error generating HTML report: {e}", err=True)
            raise typer.Exit(1)
        except Exception as e:
            typer.echo(f"Error generating HTML report: {e}", err=True)
            raise typer.Exit(1)

    else:
        run_result.print()

    # Show exception details if requested
    if show_exceptions and run_result.execution_issues:
        typer.echo(f"\nException Details ({len(run_result.execution_issues)} exception(s)):")
        for issue in run_result.execution_issues:
            typer.echo(f"  [{issue.failure_category or 'unknown'}] {issue.exception_type or 'Exception'}: {issue.message}")
            typer.echo(f"    Validator: {issue.check_name}")
            if issue.retry_count > 0:
                typer.echo(f"    Retries: {issue.retry_count}")

    # Exit with error if strict mode and issues found
    if strict and run_result.has_failures:
        raise typer.Exit(1)
