"""Learn command - Learn schema from data files.

This module implements the `truthound learn` command for inferring
schema from data files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from truthound.cli_modules.common.errors import error_boundary, require_file


@error_boundary
def learn_cmd(
    file: Annotated[
        Path,
        typer.Argument(help="Path to the data file to learn from"),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output schema file path"),
    ] = Path("schema.yaml"),
    no_constraints: Annotated[
        bool,
        typer.Option("--no-constraints", help="Don't infer constraints from data"),
    ] = False,
) -> None:
    """Learn schema from a data file.

    This command analyzes the data file and generates a schema definition
    that captures column types, constraints, and patterns.

    Examples:
        truthound learn data.csv
        truthound learn data.parquet -o my_schema.yaml
        truthound learn data.csv --no-constraints
    """
    from truthound.schema import learn

    # Validate file exists
    require_file(file)

    try:
        schema = learn(str(file), infer_constraints=not no_constraints)
        schema.save(output)

        typer.echo(f"Schema saved to {output}")
        typer.echo(f"  Columns: {len(schema.columns)}")
        typer.echo(f"  Rows: {schema.row_count:,}")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
