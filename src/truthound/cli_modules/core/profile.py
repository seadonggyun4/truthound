"""Profile command - Generate data profiles.

This module implements the `truthound profile` command for generating
statistical profiles of data files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from truthound.cli_modules.common.errors import error_boundary, require_file


@error_boundary
def profile_cmd(
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
    """Generate a statistical profile of the data.

    This command analyzes the data file and generates statistics including:
    - Row and column counts
    - Null ratios per column
    - Unique value counts
    - Data type distribution
    - Value ranges for numeric columns

    Examples:
        truthound profile data.csv
        truthound profile data.parquet --format json
        truthound profile data.csv -o profile.json
    """
    from truthound.api import profile

    # Validate file exists
    require_file(file)

    try:
        profile_report = profile(str(file))
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    if format == "json":
        result = profile_report.to_json()
        if output:
            output.write_text(result, encoding="utf-8")
            typer.echo(f"Profile written to {output}")
        else:
            typer.echo(result)
    else:
        # Console format
        if output:
            # Write console-formatted output to file
            import io
            from contextlib import redirect_stdout

            buffer = io.StringIO()
            with redirect_stdout(buffer):
                profile_report.print()
            output.write_text(buffer.getvalue(), encoding="utf-8")
            typer.echo(f"Profile written to {output}")
        else:
            profile_report.print()
