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
        typer.Option("--format", "-f", help="Output format (console, json)"),
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
            output.write_text(result)
            typer.echo(f"Report written to {output}")
        else:
            typer.echo(result)
    else:
        pii_report.print()
