"""Compare command - Compare datasets for drift.

This module implements the `truthound compare` command for detecting
data drift between two datasets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from truthound.cli_modules.common.errors import error_boundary, require_file
from truthound.cli_modules.common.options import parse_list_callback


@error_boundary
def compare_cmd(
    baseline: Annotated[
        Path,
        typer.Argument(help="Baseline (reference) data file"),
    ],
    current: Annotated[
        Path,
        typer.Argument(help="Current data file to compare"),
    ],
    columns: Annotated[
        Optional[list[str]],
        typer.Option("--columns", "-c", help="Columns to compare (comma-separated)"),
    ] = None,
    method: Annotated[
        str,
        typer.Option("--method", "-m", help="Detection method (auto, ks, psi, chi2, js)"),
    ] = "auto",
    threshold: Annotated[
        Optional[float],
        typer.Option("--threshold", "-t", help="Custom drift threshold"),
    ] = None,
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format (console, json)"),
    ] = "console",
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output file path"),
    ] = None,
    strict: Annotated[
        bool,
        typer.Option("--strict", help="Exit with code 1 if drift is detected"),
    ] = False,
) -> None:
    """Compare two datasets and detect data drift.

    This command compares a baseline dataset with a current dataset and
    detects statistical drift in column distributions.

    Detection Methods:
        - auto: Automatically select best method per column
        - ks: Kolmogorov-Smirnov test (numeric)
        - psi: Population Stability Index
        - chi2: Chi-squared test (categorical)
        - js: Jensen-Shannon divergence

    Examples:
        truthound compare baseline.csv current.csv
        truthound compare ref.parquet new.parquet --method psi
        truthound compare old.csv new.csv --threshold 0.2 --strict
        truthound compare old.csv new.csv --columns price,quantity
    """
    from truthound.drift import compare

    # Validate files exist
    require_file(baseline, "Baseline file")
    require_file(current, "Current file")

    # Parse columns if provided
    column_list = parse_list_callback(columns) if columns else None

    try:
        drift_report = compare(
            str(baseline),
            str(current),
            columns=column_list,
            method=method,
            threshold=threshold,
        )
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    if format == "json":
        result = drift_report.to_json()
        if output:
            output.write_text(result)
            typer.echo(f"Drift report written to {output}")
        else:
            typer.echo(result)
    else:
        drift_report.print()

    # Exit with error if strict mode and drift found
    if strict and drift_report.has_drift:
        raise typer.Exit(1)
