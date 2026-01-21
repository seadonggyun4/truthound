"""Auto-profile command.

This module implements the `truthound auto-profile` command for
comprehensive data profiling with auto-detection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Optional

import typer

from truthound.cli_modules.common.errors import error_boundary, require_file


def _read_json_auto(path: Path):
    """Read JSON file, auto-detecting format (JSON Array or NDJSON).

    JSON Array format: [{}, {}, {}]
    NDJSON format: {}\n{}\n{}

    If JSON Array is detected, converts to NDJSON in a temp file.
    """
    import tempfile

    import polars as pl

    with open(path, "r", encoding="utf-8") as f:
        first_char = f.read(1).strip()
        # Skip whitespace to find first meaningful character
        while first_char and first_char in " \t\n\r":
            first_char = f.read(1)

    if first_char == "[":
        # JSON Array format - convert to NDJSON
        import json

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Write as NDJSON to temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".ndjson", delete=False, encoding="utf-8"
        ) as tmp:
            for record in data:
                tmp.write(json.dumps(record, ensure_ascii=False) + "\n")
            tmp_path = tmp.name

        return pl.scan_ndjson(tmp_path)
    else:
        # NDJSON format - read directly
        return pl.scan_ndjson(path)


def _read_file_as_lazy(path: Path):
    """Read a file as a Polars LazyFrame."""
    import polars as pl

    suffix = path.suffix.lower()

    if suffix == ".json":
        return _read_json_auto(path)

    readers = {
        ".parquet": pl.scan_parquet,
        ".csv": pl.scan_csv,
        ".ndjson": pl.scan_ndjson,
        ".jsonl": pl.scan_ndjson,
    }

    if suffix not in readers:
        raise ValueError(
            f"Unsupported file type: {suffix}. "
            f"Supported: {['.json'] + list(readers.keys())}"
        )

    return readers[suffix](path)


def _print_profile_summary(profile) -> None:
    """Print a summary of the profile to console."""
    typer.echo(f"\n{'='*60}")
    typer.echo(f"Profile: {profile.name or 'unnamed'}")
    typer.echo(f"{'='*60}")
    typer.echo(f"Rows: {profile.row_count:,}")
    typer.echo(f"Columns: {profile.column_count}")
    typer.echo(f"Estimated Memory: {profile.estimated_memory_bytes / 1024 / 1024:.2f} MB")

    if profile.duplicate_row_ratio > 0:
        typer.echo(
            f"Duplicate Rows: {profile.duplicate_row_count:,} "
            f"({profile.duplicate_row_ratio*100:.1f}%)"
        )

    typer.echo(f"\n{'Column Details':^60}")
    typer.echo("-" * 60)

    for col in profile.columns:
        typer.echo(f"\n{col.name}")
        typer.echo(f"  Type: {col.physical_type} -> {col.inferred_type.value}")
        typer.echo(f"  Nulls: {col.null_count:,} ({col.null_ratio*100:.1f}%)")
        typer.echo(f"  Unique: {col.distinct_count:,} ({col.unique_ratio*100:.1f}%)")

        if col.distribution:
            dist = col.distribution
            typer.echo(f"  Range: [{dist.min}, {dist.max}]")
            if dist.mean is not None:
                typer.echo(f"  Mean: {dist.mean:.2f}, Std: {dist.std:.2f}")

        if col.min_length is not None:
            typer.echo(f"  Length: [{col.min_length}, {col.max_length}], avg={col.avg_length:.1f}")

        if col.detected_patterns:
            patterns = [p.pattern for p in col.detected_patterns[:3]]
            typer.echo(f"  Patterns: {', '.join(patterns)}")

        if col.suggested_validators:
            typer.echo(f"  Suggested: {len(col.suggested_validators)} validators")


@error_boundary
def auto_profile_cmd(
    file: Annotated[Path, typer.Argument(help="Path to the data file")],
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output file path for profile JSON"),
    ] = None,
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format (console, json, yaml)"),
    ] = "console",
    include_patterns: Annotated[
        bool,
        typer.Option("--patterns/--no-patterns", help="Include pattern detection"),
    ] = True,
    include_correlations: Annotated[
        bool,
        typer.Option("--correlations/--no-correlations", help="Include correlation analysis"),
    ] = False,
    sample_size: Annotated[
        Optional[int],
        typer.Option("--sample", "-s", help="Sample size for profiling (default: all rows)"),
    ] = None,
    top_n: Annotated[
        int,
        typer.Option("--top-n", help="Number of top/bottom values to include"),
    ] = 10,
) -> None:
    """Profile data with auto-detection of types and patterns.

    This performs comprehensive profiling including:
    - Column statistics (null ratio, unique ratio, distribution)
    - Type inference (email, phone, UUID, etc.)
    - Pattern detection
    - Suggested validation rules

    Examples:
        truthound auto-profile data.csv
        truthound auto-profile data.parquet -o profile.json
        truthound auto-profile data.csv --no-patterns --sample 10000
        truthound auto-profile data.csv --format yaml
    """
    require_file(file)

    try:
        from truthound.profiler import (
            DataProfiler,
            ProfilerConfig,
            save_profile,
        )

        config = ProfilerConfig(
            include_patterns=include_patterns,
            include_correlations=include_correlations,
            sample_size=sample_size,
            top_n_values=top_n,
        )

        profiler = DataProfiler(config=config)

        typer.echo(f"Profiling {file}...")
        profile_result = profiler.profile(
            _read_file_as_lazy(file),
            name=file.stem,
            source=str(file),
        )

        if format == "json":
            result = json.dumps(profile_result.to_dict(), indent=2, default=str)
            if output:
                with open(output, "w", encoding="utf-8") as f:
                    f.write(result)
                typer.echo(f"Profile saved to {output}")
            else:
                typer.echo(result)

        elif format == "yaml":
            import yaml

            result = yaml.dump(profile_result.to_dict(), default_flow_style=False)
            if output:
                with open(output, "w", encoding="utf-8") as f:
                    f.write(result)
                typer.echo(f"Profile saved to {output}")
            else:
                typer.echo(result)

        else:  # console
            _print_profile_summary(profile_result)
            if output:
                save_profile(profile_result, output)
                typer.echo(f"\nFull profile saved to {output}")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
