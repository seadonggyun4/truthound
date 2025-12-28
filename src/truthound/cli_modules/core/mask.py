"""Mask command - Mask sensitive data.

This module implements the `truthound mask` command for masking
sensitive data in files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from truthound.cli_modules.common.errors import error_boundary, require_file
from truthound.cli_modules.common.options import parse_list_callback


@error_boundary
def mask_cmd(
    file: Annotated[
        Path,
        typer.Argument(help="Path to the data file"),
    ],
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
    """Mask sensitive data in a file.

    This command creates a copy of the data file with sensitive columns
    masked using the specified strategy.

    Strategies:
        - redact: Replace values with asterisks
        - hash: Replace values with hashed versions
        - fake: Replace values with realistic fake data

    Examples:
        truthound mask data.csv -o masked.csv
        truthound mask data.csv -o masked.csv --columns email,phone
        truthound mask data.csv -o masked.csv --strategy hash
    """
    from truthound.api import mask

    # Validate file exists
    require_file(file)

    # Parse columns if provided
    column_list = parse_list_callback(columns) if columns else None

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
