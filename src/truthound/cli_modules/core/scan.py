"""Scan command - Scan for PII.

This module implements the ``truthound scan`` command for detecting
personally identifiable information in data files and database tables.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from truthound.cli_modules.common.datasource import (
    ConnectionOpt,
    QueryOpt,
    SourceConfigOpt,
    SourceNameOpt,
    TableOpt,
    resolve_datasource,
)
from truthound.cli_modules.common.errors import error_boundary


@error_boundary
def scan_cmd(
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
    # -- Output Options --
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

    This command analyzes data to detect columns that may contain
    PII such as names, emails, phone numbers, SSNs, etc.

    Examples:
        truthound scan data.csv
        truthound scan data.parquet --format json
        truthound scan data.csv -o pii_report.json
        truthound scan --connection "postgresql://user:pass@host/db" --table users
        truthound scan --source-config db.yaml --format json
    """
    from truthound.api import scan

    # Resolve data source
    data_path, source = resolve_datasource(
        file=file,
        connection=connection,
        table=table,
        query=query,
        source_config=source_config,
        source_name=source_name,
    )

    try:
        if source is not None:
            pii_report = scan(source=source)
        else:
            pii_report = scan(data_path)
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

            report_label = source_name or (source.name if source else str(file))
            html = generate_pii_html_report(
                pii_report, title=f"PII Scan Report: {report_label}"
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
