"""Scan command - Scan for PII.

This module implements the `truthound scan` command for detecting
personally identifiable information in data files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from truthound.cli_modules.common.errors import error_boundary, require_file


@error_boundary
def scan_cmd(
    file: Annotated[
        Path,
        typer.Argument(help="Path to the data file"),
    ],
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format (console, json, html)"),
    ] = "console",
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output file path"),
    ] = None,
) -> None:
    """Scan for personally identifiable information (PII).

    This command analyzes data files to detect columns that may contain
    PII such as names, emails, phone numbers, SSNs, etc.

    Examples:
        truthound scan data.csv
        truthound scan data.parquet --format json
        truthound scan data.csv -o pii_report.json
        truthound scan data.csv --format html -o pii_report.html
    """
    from truthound.api import scan

    # Validate file exists
    require_file(file)

    try:
        pii_report = scan(str(file))
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    if format == "json":
        result = pii_report.to_json()
        if output:
            output.write_text(result, encoding="utf-8")
            typer.echo(f"Report written to {output}")
        else:
            typer.echo(result)

    elif format == "html":
        if not output:
            typer.echo("Error: --output is required for HTML format", err=True)
            raise typer.Exit(1)
        try:
            from truthound.html_reporter import generate_pii_html_report

            html = generate_pii_html_report(
                pii_report, title=f"PII Scan Report: {file.name}"
            )
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
        pii_report.print()
