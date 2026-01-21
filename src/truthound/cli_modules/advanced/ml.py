"""Machine learning commands.

This module implements ML-based validation commands (Phase 10).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Optional

import typer

from truthound.cli_modules.common.errors import error_boundary, require_file

# ML app for subcommands
app = typer.Typer(
    name="ml",
    help="Machine learning based validation commands",
)


def _read_data(file: Path):
    """Read data file as Polars DataFrame."""
    import polars as pl

    suffix = str(file).lower()
    if suffix.endswith(".csv"):
        return pl.read_csv(file)
    elif suffix.endswith(".parquet"):
        return pl.read_parquet(file)
    elif suffix.endswith(".jsonl") or suffix.endswith(".ndjson"):
        return pl.read_ndjson(file)
    elif suffix.endswith(".json"):
        return pl.read_json(file)
    else:
        raise ValueError(
            f"Unsupported file format: {file.suffix}. "
            f"Supported formats: .csv, .parquet, .json, .jsonl, .ndjson"
        )


@app.command(name="anomaly")
@error_boundary
def anomaly_cmd(
    file: Annotated[Path, typer.Argument(help="Path to the data file")],
    method: Annotated[
        str,
        typer.Option(
            "--method", "-m", help="Detection method (zscore, iqr, mad, isolation_forest)"
        ),
    ] = "zscore",
    contamination: Annotated[
        float,
        typer.Option(
            "--contamination", "-c", help="Expected proportion of outliers (0.0 to 0.5)"
        ),
    ] = 0.1,
    columns: Annotated[
        Optional[str],
        typer.Option("--columns", help="Comma-separated columns to analyze"),
    ] = None,
    sample: Annotated[
        Optional[int],
        typer.Option(
            "--sample", "-s", help="Sample size for processing (default: all rows)", min=1
        ),
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output file path for results"),
    ] = None,
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format (console, json)"),
    ] = "console",
) -> None:
    """Detect anomalies in data using ML methods.

    Methods:
        - zscore: Z-score based detection (fast, requires normal distribution)
        - iqr: Interquartile range (robust to outliers)
        - mad: Median Absolute Deviation (very robust)
        - isolation_forest: Tree-based (best for complex patterns)

    Examples:
        truthound ml anomaly data.csv
        truthound ml anomaly data.csv --method isolation_forest --contamination 0.05
        truthound ml anomaly data.csv --columns "amount,price" --output anomalies.json
        truthound ml anomaly data.parquet --sample 100000
        truthound ml anomaly data.jsonl --method zscore
    """
    from truthound.ml import (
        ZScoreAnomalyDetector,
        IQRAnomalyDetector,
        MADAnomalyDetector,
        IsolationForestDetector,
    )
    from truthound.ml.anomaly_models.statistical import StatisticalConfig
    from truthound.ml.anomaly_models.isolation_forest import IsolationForestConfig

    require_file(file)

    try:
        df = _read_data(file)

        # Apply sampling if specified
        original_rows = len(df)
        if sample is not None and sample < original_rows:
            df = df.sample(n=sample, seed=42)
            typer.echo(f"Sampled {sample:,} rows from {original_rows:,} total rows")

        # Parse columns
        cols = [c.strip() for c in columns.split(",")] if columns else None

        # Select detector and appropriate config
        # Use min_samples=10 for CLI to allow smaller datasets
        if method == "isolation_forest":
            config = IsolationForestConfig(
                contamination=contamination, columns=cols, min_samples=10
            )
            detector = IsolationForestDetector(config=config)
        elif method in ("zscore", "iqr", "mad"):
            config = StatisticalConfig(
                contamination=contamination, columns=cols, min_samples=10
            )
            detector_map = {
                "zscore": ZScoreAnomalyDetector,
                "iqr": IQRAnomalyDetector,
                "mad": MADAnomalyDetector,
            }
            detector = detector_map[method](config=config)
        else:
            typer.echo(
                f"Error: Unknown method '{method}'. "
                f"Available: zscore, iqr, mad, isolation_forest",
                err=True,
            )
            raise typer.Exit(1)
        detector.fit(df.lazy())
        result = detector.predict(df.lazy())

        # Output results
        if format == "json":
            output_data = result.to_dict()
            if output:
                with open(output, "w") as f:
                    json.dump(output_data, f, indent=2)
                typer.echo(f"Results saved to {output}")
            else:
                typer.echo(json.dumps(output_data, indent=2))
        else:
            typer.echo(f"\nAnomaly Detection Results ({method})")
            typer.echo("=" * 50)
            typer.echo(f"Total points: {result.total_points}")
            typer.echo(f"Anomalies found: {result.anomaly_count}")
            typer.echo(f"Anomaly ratio: {result.anomaly_ratio:.2%}")
            typer.echo(f"Threshold used: {result.threshold_used:.4f}")

            if result.anomaly_count > 0:
                typer.echo("\nTop anomalies:")
                anomalies = sorted(
                    result.get_anomalies(), key=lambda x: x.score, reverse=True
                )[:10]
                for a in anomalies:
                    typer.echo(
                        f"  Index {a.index}: score={a.score:.4f}, "
                        f"confidence={a.confidence:.2%}"
                    )

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command(name="drift")
@error_boundary
def drift_cmd(
    baseline: Annotated[Path, typer.Argument(help="Path to baseline/reference data file")],
    current: Annotated[Path, typer.Argument(help="Path to current data file")],
    method: Annotated[
        str,
        typer.Option(
            "--method", "-m", help="Detection method (distribution, feature, multivariate)"
        ),
    ] = "feature",
    threshold: Annotated[
        float,
        typer.Option("--threshold", "-t", help="Drift detection threshold"),
    ] = 0.1,
    columns: Annotated[
        Optional[str],
        typer.Option("--columns", help="Comma-separated columns to analyze"),
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output file path"),
    ] = None,
) -> None:
    """Detect data drift between baseline and current datasets.

    Methods:
        - distribution: Compare statistical distributions
        - feature: Per-feature drift detection
        - multivariate: Consider feature correlations

    Examples:
        truthound ml drift baseline.csv current.csv
        truthound ml drift ref.parquet new.parquet --method multivariate
        truthound ml drift old.csv new.csv --threshold 0.2 --output drift_report.json
    """
    from truthound.ml.drift_detection import (
        DistributionDriftDetector,
        FeatureDriftDetector,
        MultivariateDriftDetector,
    )

    require_file(baseline, "Baseline file")
    require_file(current, "Current file")

    try:
        baseline_df = _read_data(baseline)
        current_df = _read_data(current)

        detector_map = {
            "distribution": DistributionDriftDetector,
            "feature": FeatureDriftDetector,
            "multivariate": MultivariateDriftDetector,
        }

        if method not in detector_map:
            # Check if user might be looking for statistical methods
            statistical_methods = ["psi", "ks", "chi2", "js", "auto"]
            hint = ""
            if method.lower() in statistical_methods:
                hint = (
                    f"\n\nHint: '{method}' is a statistical method available in "
                    f"'truthound compare' command.\n"
                    f"  Example: truthound compare {baseline} {current} --method {method}"
                )

            available_desc = {
                "distribution": "Per-column statistical drift detection",
                "feature": "Feature importance-based drift detection",
                "multivariate": "Multivariate drift detection (PCA/covariance)",
            }
            methods_help = "\n".join(
                f"  - {k}: {v}" for k, v in available_desc.items()
            )

            typer.echo(
                f"Error: Unknown method '{method}'.\n\n"
                f"Available methods for 'ml drift':\n{methods_help}"
                f"{hint}",
                err=True,
            )
            raise typer.Exit(1)

        detector = detector_map[method](threshold=threshold)
        detector.fit(baseline_df.lazy())

        cols = [c.strip() for c in columns.split(",")] if columns else None
        result = detector.detect(baseline_df.lazy(), current_df.lazy(), columns=cols)

        # Output results
        typer.echo(f"\nDrift Detection Results ({method})")
        typer.echo("=" * 50)
        typer.echo(f"Drift detected: {'YES' if result.is_drifted else 'NO'}")
        typer.echo(f"Drift score: {result.drift_score:.4f}")
        typer.echo(f"Drift type: {result.drift_type}")

        if result.column_scores:
            typer.echo("\nPer-column drift scores:")
            for col, score in sorted(
                result.column_scores, key=lambda x: x[1], reverse=True
            ):
                status = "[DRIFTED]" if score >= threshold else ""
                typer.echo(f"  {col}: {score:.4f} {status}")

        if output:
            with open(output, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
            typer.echo(f"\nResults saved to {output}")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command(name="learn-rules")
@error_boundary
def learn_rules_cmd(
    file: Annotated[Path, typer.Argument(help="Path to the data file")],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output file for learned rules"),
    ] = Path("learned_rules.json"),
    strictness: Annotated[
        str,
        typer.Option("--strictness", "-s", help="Rule strictness (loose, medium, strict)"),
    ] = "medium",
    min_confidence: Annotated[
        float,
        typer.Option("--min-confidence", help="Minimum rule confidence"),
    ] = 0.9,
    max_rules: Annotated[
        int,
        typer.Option("--max-rules", help="Maximum number of rules to generate"),
    ] = 100,
) -> None:
    """Learn validation rules from data.

    This command analyzes data patterns and generates validation rules
    automatically based on the observed data characteristics.

    Examples:
        truthound ml learn-rules data.csv
        truthound ml learn-rules data.csv --strictness strict --min-confidence 0.95
        truthound ml learn-rules data.parquet --output my_rules.json
    """
    from truthound.ml.rule_learning import DataProfileRuleLearner

    require_file(file)

    try:
        df = _read_data(file)

        typer.echo(f"Learning rules from {file}...")
        typer.echo(f"  Rows: {len(df):,}, Columns: {len(df.columns)}")

        # Use profile learner
        learner = DataProfileRuleLearner(
            strictness=strictness,
            min_confidence=min_confidence,
            max_rules=max_rules,
        )

        result = learner.learn_rules(df.lazy())

        typer.echo(f"\nLearned {len(result.rules)} rules ({result.filtered_rules} filtered)")
        typer.echo(f"Learning time: {result.learning_time_ms:.1f}ms")

        # Show rules by type
        rule_types = {}
        for rule in result.rules:
            rule_types[rule.rule_type] = rule_types.get(rule.rule_type, 0) + 1

        typer.echo("\nRules by type:")
        for rtype, count in sorted(
            rule_types.items(), key=lambda x: x[1], reverse=True
        ):
            typer.echo(f"  {rtype}: {count}")

        # Save rules
        with open(output, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        typer.echo(f"\nRules saved to {output}")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
