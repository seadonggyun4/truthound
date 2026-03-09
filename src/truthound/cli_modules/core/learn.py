"""Learn command - Learn schema from data.

This module implements the ``truthound learn`` command for inferring
schema from data files and database tables.
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
def learn_cmd(
    file: Annotated[
        Optional[Path],
        typer.Argument(
            help="Path to the data file to learn from",
        ),
    ] = None,
    # -- DataSource Options --
    connection: ConnectionOpt = None,
    table: TableOpt = None,
    query: QueryOpt = None,
    source_config: SourceConfigOpt = None,
    source_name: SourceNameOpt = None,
    # -- Schema Options --
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output schema file path"),
    ] = Path("schema.yaml"),
    no_constraints: Annotated[
        bool,
        typer.Option("--no-constraints", help="Don't infer constraints from data"),
    ] = False,
    categorical_threshold: Annotated[
        int,
        typer.Option(
            "--categorical-threshold",
            help=(
                "Maximum unique values to treat a column as categorical "
                "during schema inference (default: 20)"
            ),
            min=1,
        ),
    ] = 20,
) -> None:
    """Learn schema from a data file or database table.

    This command analyzes the data and generates a schema definition
    that captures column types, constraints, and patterns.

    Examples:
        truthound learn data.csv
        truthound learn data.parquet -o my_schema.yaml
        truthound learn data.csv --no-constraints
        truthound learn data.csv --categorical-threshold 50
        truthound learn --connection "postgresql://user:pass@host/db" --table users
        truthound learn --source-config db.yaml -o db_schema.yaml
    """
    from truthound.schema import learn

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
            schema = learn(
                source=source,
                infer_constraints=not no_constraints,
                categorical_threshold=categorical_threshold,
            )
        else:
            schema = learn(
                data_path,
                infer_constraints=not no_constraints,
                categorical_threshold=categorical_threshold,
            )
        schema.save(output)

        typer.echo(f"Schema saved to {output}")
        typer.echo(f"  Columns: {len(schema.columns)}")
        typer.echo(f"  Rows: {schema.row_count:,}")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
