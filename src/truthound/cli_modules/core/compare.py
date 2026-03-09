"""Compare command - Compare datasets for drift.

This module implements the ``truthound compare`` command for detecting
data drift between two datasets from files or database tables.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from truthound.cli_modules.common.datasource import (
    SourceConfigOpt,
    resolve_compare_sources,
)
from truthound.cli_modules.common.errors import error_boundary
from truthound.cli_modules.common.options import parse_list_callback


@error_boundary
def compare_cmd(
    baseline: Annotated[
        Optional[Path],
        typer.Argument(
            help="Baseline (reference) data file",
        ),
    ] = None,
    current: Annotated[
        Optional[Path],
        typer.Argument(
            help="Current data file to compare",
        ),
    ] = None,
    # -- DataSource Config (for database-to-database comparison) --
    source_config: SourceConfigOpt = None,
    # -- Compare Options --
    columns: Annotated[
        Optional[list[str]],
        typer.Option("--columns", "-c", help="Columns to compare (comma-separated)"),
    ] = None,
    method: Annotated[
        str,
        typer.Option(
            "--method",
            "-m",
            help=(
                "Detection method: auto, ks, psi, chi2, js, kl, wasserstein, "
                "cvm, anderson, hellinger, bhattacharyya, tv, energy, mmd"
            ),
        ),
    ] = "auto",
    threshold: Annotated[
        Optional[float],
        typer.Option("--threshold", "-t", help="Custom drift threshold"),
    ] = None,
    sample_size: Annotated[
        Optional[int],
        typer.Option(
            "--sample-size",
            "--sample",
            help="Sample size for large datasets. Uses random sampling for faster comparison.",
            min=1,
        ),
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
    detects statistical drift in column distributions. Supports file
    paths or a --source-config for database-to-database comparison.

    Detection Methods:
        - auto: Automatically select best method per column (recommended)
        - ks: Kolmogorov-Smirnov test (numeric)
        - psi: Population Stability Index (ML monitoring)
        - chi2: Chi-squared test (categorical)
        - js: Jensen-Shannon divergence (any type)
        - kl: Kullback-Leibler divergence (numeric)
        - wasserstein: Earth Mover's distance (numeric)
        - cvm: Cramer-von Mises test (numeric, tail-sensitive)
        - anderson: Anderson-Darling test (numeric, extreme values)
        - hellinger: Hellinger distance (bounded metric)
        - bhattacharyya: Bhattacharyya distance (classification bounds)
        - tv: Total Variation distance (max probability diff)
        - energy: Energy distance (location/scale)
        - mmd: Maximum Mean Discrepancy (high-dimensional)

    Source Config Format (YAML):
        baseline:
            connection: "postgresql://user:pass@host/db"
            table: train_data
        current:
            connection: "postgresql://user:pass@host/db"
            table: production_data

    Examples:
        truthound compare baseline.csv current.csv
        truthound compare ref.parquet new.parquet --method psi
        truthound compare old.csv new.csv --threshold 0.2 --strict
        truthound compare old.csv new.csv --columns price,quantity
        truthound compare --source-config drift_config.yaml --method ks
        truthound compare big_train.csv big_prod.csv --sample-size 10000
    """
    from truthound.drift import compare

    # Resolve both data sources
    (baseline_path, baseline_source), (current_path, current_source) = (
        resolve_compare_sources(
            baseline=baseline,
            current=current,
            source_config=source_config,
        )
    )

    # Parse columns if provided
    column_list = parse_list_callback(columns) if columns else None

    # Determine inputs for the compare API
    baseline_input = (
        baseline_source.to_polars_lazyframe().collect()
        if baseline_source
        else baseline_path
    )
    current_input = (
        current_source.to_polars_lazyframe().collect()
        if current_source
        else current_path
    )

    try:
        drift_report = compare(
            baseline_input,
            current_input,
            columns=column_list,
            method=method,
            threshold=threshold,
            sample_size=sample_size,
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
