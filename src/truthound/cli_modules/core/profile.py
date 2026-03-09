"""Profile command - Generate data profiles.

This module implements the ``truthound profile`` command for generating
statistical profiles of data files and database tables.
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
def profile_cmd(
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
        typer.Option("--format", "-f", help="Output format (console, json)"),
    ] = "console",
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output file path"),
    ] = None,
) -> None:
    """Generate a statistical profile of the data.

    This command analyzes the data and generates statistics including:
    - Row and column counts
    - Null ratios per column
    - Unique value counts
    - Data type distribution
    - Value ranges for numeric columns

    Examples:
        truthound profile data.csv
        truthound profile data.parquet --format json
        truthound profile data.csv -o profile.json
        truthound profile --connection "postgresql://user:pass@host/db" --table users
        truthound profile --source-config db.yaml --format json
    """
    from truthound.api import profile

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
            profile_report = profile(source=source)
        else:
            profile_report = profile(data_path)
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
