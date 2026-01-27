"""Data comparison and drift detection."""

from __future__ import annotations

from typing import Any

import polars as pl

from truthound.adapters import to_lazyframe
from truthound.drift.detectors import (
    AndersonDarlingDetector,
    BhattacharyyaDetector,
    ChiSquareDetector,
    CramervonMisesDetector,
    DriftDetector,
    EnergyDetector,
    HellingerDetector,
    JensenShannonDetector,
    KLDivergenceDetector,
    KSTestDetector,
    MMDDetector,
    PSIDetector,
    TotalVariationDetector,
    WassersteinDetector,
)
from truthound.drift.report import ColumnDrift, DriftReport


def compare(
    baseline: Any,
    current: Any,
    columns: list[str] | None = None,
    method: str = "auto",
    threshold: float | None = None,
    sample_size: int | None = None,
) -> DriftReport:
    """Compare two datasets and detect data drift.

    Uses statistical tests to identify distribution changes between
    a baseline (reference) dataset and current data.

    Args:
        baseline: Reference data (file path, DataFrame, dict, etc.)
        current: Current data to compare against baseline.
        columns: Optional list of columns to compare. If None, all common columns.
        method: Detection method - "auto", "ks", "psi", "chi2", or "js".
               "auto" selects based on column type.
        threshold: Optional custom threshold for drift detection.
        sample_size: Optional sample size for large datasets. If provided,
                    uses random sampling for faster comparison.

    Returns:
        DriftReport with per-column drift analysis.

    Example:
        >>> import truthound as th
        >>> drift = th.compare("train.csv", "production.csv")
        >>> print(drift)
        >>> if drift.has_high_drift:
        ...     print("Warning: Significant data drift detected!")

        >>> # For large datasets, use sampling
        >>> drift = th.compare("big_train.csv", "big_prod.csv", sample_size=10000)
    """
    # Load data
    baseline_lf = to_lazyframe(baseline)
    current_lf = to_lazyframe(current)

    baseline_df = baseline_lf.collect()
    current_df = current_lf.collect()

    # Apply sampling for large datasets
    if sample_size is not None:
        if len(baseline_df) > sample_size:
            baseline_df = baseline_df.sample(n=sample_size, seed=42)
        if len(current_df) > sample_size:
            current_df = current_df.sample(n=sample_size, seed=42)

    baseline_source = str(baseline) if isinstance(baseline, str) else type(baseline).__name__
    current_source = str(current) if isinstance(current, str) else type(current).__name__

    # Determine columns to compare
    baseline_cols = set(baseline_df.columns)
    current_cols = set(current_df.columns)
    common_cols = baseline_cols & current_cols

    if columns is not None:
        compare_cols = [c for c in columns if c in common_cols]
    else:
        compare_cols = list(common_cols)

    # Prepare detectors
    detectors = _get_detectors(method, threshold)

    # Analyze each column
    column_drifts: list[ColumnDrift] = []

    numeric_types = {
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        pl.Float32, pl.Float64,
    }

    for col in compare_cols:
        baseline_series = baseline_df.get_column(col)
        current_series = current_df.get_column(col)

        dtype = baseline_df.schema[col]
        dtype_str = str(dtype)

        # Select appropriate detector
        if method == "auto":
            if dtype in numeric_types:
                detector = detectors["numeric"]
            else:
                detector = detectors["categorical"]
        else:
            detector = detectors["default"]

        # Run drift detection
        result = detector.detect(baseline_series, current_series)

        # Collect statistics
        baseline_stats = _compute_stats(baseline_series, dtype, numeric_types)
        current_stats = _compute_stats(current_series, dtype, numeric_types)

        column_drifts.append(
            ColumnDrift(
                column=col,
                dtype=dtype_str,
                result=result,
                baseline_stats=baseline_stats,
                current_stats=current_stats,
            )
        )

    return DriftReport(
        baseline_source=baseline_source,
        current_source=current_source,
        baseline_rows=len(baseline_df),
        current_rows=len(current_df),
        columns=column_drifts,
    )


def _get_detectors(method: str, threshold: float | None) -> dict[str, DriftDetector]:
    """Get appropriate detectors based on method."""
    if method == "auto":
        return {
            "numeric": PSIDetector(threshold=threshold or 0.1),
            "categorical": ChiSquareDetector(threshold=threshold or 0.05),
        }
    elif method == "ks":
        return {"default": KSTestDetector(threshold=threshold or 0.05)}
    elif method == "psi":
        return {"default": PSIDetector(threshold=threshold or 0.1)}
    elif method == "chi2":
        return {"default": ChiSquareDetector(threshold=threshold or 0.05)}
    elif method == "js":
        return {"default": JensenShannonDetector(threshold=threshold or 0.1)}
    elif method == "kl":
        return {"default": KLDivergenceDetector(threshold=threshold or 0.1)}
    elif method == "wasserstein":
        return {"default": WassersteinDetector(threshold=threshold or 0.1)}
    elif method == "cvm":
        return {"default": CramervonMisesDetector(threshold=threshold or 0.05)}
    elif method == "anderson":
        return {"default": AndersonDarlingDetector(threshold=threshold or 0.05)}
    elif method == "hellinger":
        return {"default": HellingerDetector(threshold=threshold or 0.1)}
    elif method == "bhattacharyya":
        return {"default": BhattacharyyaDetector(threshold=threshold or 0.1)}
    elif method == "tv" or method == "total_variation":
        return {"default": TotalVariationDetector(threshold=threshold or 0.1)}
    elif method == "energy":
        return {"default": EnergyDetector(threshold=threshold or 0.1)}
    elif method == "mmd":
        return {"default": MMDDetector(threshold=threshold or 0.1)}
    else:
        raise ValueError(
            f"Unknown comparison method: '{method}'\n\n"
            f"Available methods:\n"
            f"  • auto         - Automatically select based on column type (recommended)\n"
            f"  • ks           - Kolmogorov-Smirnov test (numeric columns only)\n"
            f"  • psi          - Population Stability Index (numeric columns only)\n"
            f"  • chi2         - Chi-square test (categorical columns)\n"
            f"  • js           - Jensen-Shannon divergence (any column type)\n"
            f"  • kl           - Kullback-Leibler divergence (numeric columns only)\n"
            f"  • wasserstein  - Wasserstein/Earth Mover's distance (numeric columns only)\n"
            f"  • cvm          - Cramér-von Mises test (numeric columns only)\n"
            f"  • anderson     - Anderson-Darling test (numeric columns only)\n"
            f"  • hellinger    - Hellinger distance (any column type)\n"
            f"  • bhattacharyya - Bhattacharyya distance (any column type)\n"
            f"  • tv           - Total Variation distance (any column type)\n"
            f"  • energy       - Energy distance (numeric columns only)\n"
            f"  • mmd          - Maximum Mean Discrepancy (numeric columns only)\n\n"
            f"Example: truthound compare baseline.csv current.csv --method auto"
        )


def _compute_stats(series: pl.Series, dtype: pl.DataType, numeric_types: set) -> dict:
    """Compute summary statistics for a series."""
    stats: dict = {
        "count": len(series),
        "null_count": series.null_count(),
        "null_ratio": round(series.null_count() / len(series), 4) if len(series) > 0 else 0,
    }

    non_null = series.drop_nulls()

    if len(non_null) == 0:
        return stats

    if dtype in numeric_types:
        stats["min"] = float(non_null.min())  # type: ignore
        stats["max"] = float(non_null.max())  # type: ignore
        stats["mean"] = round(float(non_null.mean()), 4)  # type: ignore
        stats["std"] = round(float(non_null.std()), 4) if len(non_null) > 1 else 0  # type: ignore
        q25 = non_null.quantile(0.25)
        q50 = non_null.quantile(0.50)
        q75 = non_null.quantile(0.75)
        if q25 is not None:
            stats["q25"] = round(float(q25), 4)
        if q50 is not None:
            stats["median"] = round(float(q50), 4)
        if q75 is not None:
            stats["q75"] = round(float(q75), 4)
    else:
        stats["unique"] = non_null.n_unique()
        # Top 5 values
        value_counts = non_null.value_counts().sort("count", descending=True).head(5)
        stats["top_values"] = value_counts.to_dicts()

    return stats
