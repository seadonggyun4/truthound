"""Check command - Validate data quality.

This module implements the `truthound check` command for validating
data quality in files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from truthound.cli_modules.common.errors import error_boundary, require_file
from truthound.cli_modules.common.options import parse_list_callback


@error_boundary
def check_cmd(
    file: Annotated[
        Path,
        typer.Argument(help="Path to the data file"),
    ],
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
) -> None:
    """Validate data quality in a file.

    This command runs data quality validators on the specified file and
    reports any issues found.

    Examples:
        truthound check data.csv
        truthound check data.parquet --validators null,duplicate,range
        truthound check data.csv --min-severity high --strict
        truthound check data.csv --auto-schema
        truthound check data.csv --format json -o report.json
    """
    from truthound.api import check

    # Validate files exist
    require_file(file)
    if schema_file:
        require_file(schema_file, "Schema file")

    # Parse validators if provided
    validator_list = parse_list_callback(validators) if validators else None

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
        try:
            from truthound.html_reporter import generate_html_report

            html = generate_html_report(report, title=f"Validation Report: {file.name}")
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
        report.print()

    # Exit with error if strict mode and issues found
    if strict and report.has_issues:
        raise typer.Exit(1)
