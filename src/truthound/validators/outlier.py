"""Outlier detection validator using IQR method."""

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue, Validator


class OutlierValidator(Validator):
    """Detects statistical outliers using the IQR (Interquartile Range) method."""

    name = "outlier"

    def __init__(self, iqr_multiplier: float = 1.5):
        """Initialize the outlier validator.

        Args:
            iqr_multiplier: Multiplier for IQR to determine outlier bounds.
                           Default is 1.5 (standard). Use 3.0 for extreme outliers only.
        """
        self.iqr_multiplier = iqr_multiplier

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Check for statistical outliers in numeric columns.

        Uses the IQR method: values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR
        are considered outliers.

        Args:
            lf: Polars LazyFrame to validate.

        Returns:
            List of validation issues for columns with outliers.
        """
        issues: list[ValidationIssue] = []

        schema = lf.collect_schema()

        numeric_types = {
            pl.Int8, pl.Int16, pl.Int32, pl.Int64,
            pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
            pl.Float32, pl.Float64,
        }

        # Get numeric columns only
        numeric_cols = [col for col in schema.names() if schema[col] in numeric_types]

        if not numeric_cols:
            return issues

        # Single optimized query: compute Q1, Q3, and count for all numeric columns
        stats_exprs = [pl.len().alias("_total")]
        for col in numeric_cols:
            stats_exprs.extend([
                pl.col(col).quantile(0.25).alias(f"_q1_{col}"),
                pl.col(col).quantile(0.75).alias(f"_q3_{col}"),
                pl.col(col).count().alias(f"_cnt_{col}"),
            ])

        stats = lf.select(stats_exprs).collect()
        total_rows = stats["_total"][0]

        if total_rows < 4:
            return issues

        # Calculate outliers for each column
        outlier_exprs = []
        bounds_info = {}

        for col in numeric_cols:
            q1 = stats[f"_q1_{col}"][0]
            q3 = stats[f"_q3_{col}"][0]
            cnt = stats[f"_cnt_{col}"][0]

            if q1 is None or q3 is None or cnt < 4:
                continue

            iqr = q3 - q1
            if iqr == 0:
                continue

            lower_bound = q1 - self.iqr_multiplier * iqr
            upper_bound = q3 + self.iqr_multiplier * iqr

            bounds_info[col] = (lower_bound, upper_bound, cnt)
            outlier_exprs.append(
                ((pl.col(col) < lower_bound) | (pl.col(col) > upper_bound))
                .sum()
                .alias(f"_out_{col}")
            )

        if not outlier_exprs:
            return issues

        # Single query to count all outliers
        outlier_counts = lf.select(outlier_exprs).collect()

        for col in bounds_info:
            lower_bound, upper_bound, cnt = bounds_info[col]
            outlier_count = outlier_counts[f"_out_{col}"][0]

            if outlier_count > 0:
                outlier_pct = outlier_count / cnt

                if outlier_pct > 0.1:
                    severity = Severity.MEDIUM
                else:
                    severity = Severity.LOW

                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="outlier",
                        count=outlier_count,
                        severity=severity,
                        details=f"IQR bounds: [{lower_bound:.2f}, {upper_bound:.2f}]",
                    )
                )

        return issues
