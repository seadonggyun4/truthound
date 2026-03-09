"""Read command - Read and preview data from various sources.

This module implements the ``truthound read`` command for loading,
inspecting, and exporting data from files and database connections.
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
from truthound.cli_modules.common.options import parse_list_callback


@error_boundary
def read_cmd(
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
    # -- Row Selection --
    sample: Annotated[
        Optional[int],
        typer.Option(
            "--sample",
            "-s",
            help="Return a random sample of N rows",
            min=1,
        ),
    ] = None,
    head: Annotated[
        Optional[int],
        typer.Option(
            "--head",
            "-n",
            help="Show only the first N rows",
            min=1,
        ),
    ] = None,
    # -- Column Selection --
    columns: Annotated[
        Optional[list[str]],
        typer.Option(
            "--columns",
            "-c",
            help="Columns to include (comma-separated)",
        ),
    ] = None,
    # -- Output Options --
    format: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="Output format (table, csv, json, parquet, ndjson)",
        ),
    ] = "table",
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output file path"),
    ] = None,
    # -- Inspection Modes --
    schema_only: Annotated[
        bool,
        typer.Option(
            "--schema-only",
            help="Show only column names and types (no data loaded)",
        ),
    ] = False,
    count_only: Annotated[
        bool,
        typer.Option(
            "--count-only",
            help="Show only the row count",
        ),
    ] = False,
) -> None:
    """Read and preview data from files or databases.

    Load data from various sources and display a preview, export to
    another format, or inspect the schema. Supports files (CSV, Parquet,
    JSON) and SQL databases via --connection.

    Examples:
        truthound read data.csv
        truthound read data.parquet --head 20
        truthound read data.csv --format json -o output.json
        truthound read data.csv --columns id,name,age
        truthound read --connection "postgresql://user:pass@host/db" --table users
        truthound read --connection "sqlite:///data.db" --table orders --head 10
        truthound read --source-config db.yaml --sample 1000
        truthound read data.csv --schema-only
        truthound read data.csv --count-only
    """
    import polars as pl

    # Resolve data source
    data_path, source = resolve_datasource(
        file=file,
        connection=connection,
        table=table,
        query=query,
        source_config=source_config,
        source_name=source_name,
    )

    # Load data as LazyFrame
    if source is not None:
        lf = source.to_polars_lazyframe()
        label = source.name
    else:
        from truthound.adapters import to_lazyframe

        lf = to_lazyframe(data_path)
        label = data_path

    # Schema-only mode: no data collection needed
    if schema_only:
        schema = lf.collect_schema()
        typer.echo(f"Source: {label}")
        typer.echo(f"Columns: {len(schema)}\n")
        typer.echo(f"{'Column':<40} {'Type':<20}")
        typer.echo("-" * 60)
        for col_name, col_type in schema.items():
            typer.echo(f"{col_name:<40} {str(col_type):<20}")
        return

    # Count-only mode: minimal collection
    if count_only:
        row_count = lf.select(pl.len()).collect().item()
        typer.echo(f"Source: {label}")
        typer.echo(f"Rows: {row_count:,}")
        return

    # Collect data
    df = lf.collect()

    # Column selection
    column_list = parse_list_callback(columns) if columns else None
    if column_list:
        available = set(df.columns)
        missing = [c for c in column_list if c not in available]
        if missing:
            typer.echo(
                f"Warning: columns not found: {', '.join(missing)}", err=True
            )
        valid_cols = [c for c in column_list if c in available]
        if valid_cols:
            df = df.select(valid_cols)

    # Row selection
    if sample is not None and len(df) > sample:
        df = df.sample(n=sample, seed=42)
    if head is not None:
        df = df.head(head)

    # Output
    if format == "parquet" and output is None:
        typer.echo(
            "Error: --output is required for parquet format", err=True
        )
        raise typer.Exit(1)

    if output:
        _write_output(df, output, format)
        typer.echo(f"Data written to {output} ({len(df):,} rows)")
    else:
        _print_output(df, format, label)


def _write_output(df: "pl.DataFrame", output: Path, fmt: str) -> None:
    """Write DataFrame to a file in the specified format."""
    suffix = output.suffix.lower()
    fmt_lower = fmt.lower()

    if fmt_lower == "parquet" or suffix == ".parquet":
        df.write_parquet(output)
    elif fmt_lower == "csv" or suffix == ".csv":
        df.write_csv(output)
    elif fmt_lower == "json" or suffix == ".json":
        df.write_json(output)
    elif fmt_lower == "ndjson" or suffix == ".ndjson":
        df.write_ndjson(output)
    else:
        # Default: CSV
        df.write_csv(output)


def _print_output(df: "pl.DataFrame", fmt: str, label: str | None) -> None:
    """Print DataFrame to stdout."""
    import polars as pl

    fmt_lower = fmt.lower()

    if fmt_lower == "json":
        typer.echo(df.write_json())
    elif fmt_lower == "csv":
        typer.echo(df.write_csv())
    elif fmt_lower == "ndjson":
        typer.echo(df.write_ndjson())
    else:
        # Table format: use Polars' built-in display
        if label:
            typer.echo(f"Source: {label}")
        typer.echo(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns\n")
        with pl.Config(tbl_rows=50, tbl_cols=20, fmt_str_lengths=80):
            typer.echo(str(df))
