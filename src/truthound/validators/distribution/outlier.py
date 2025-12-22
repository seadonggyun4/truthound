"""Outlier detection validators."""

from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import (
    ValidationIssue,
    Validator,
    NumericValidatorMixin,
)
from truthound.validators.registry import register_validator


@register_validator
class OutlierValidator(Validator, NumericValidatorMixin):
    """Detects statistical outliers using the IQR method."""

    name = "outlier"
    category = "distribution"

    def __init__(
        self,
        iqr_multiplier: float = 1.5,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.iqr_multiplier = iqr_multiplier

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        columns = self._get_numeric_columns(lf)

        if not columns:
            return issues

        # Compute Q1, Q3 for all columns
        stats_exprs = [pl.len().alias("_total")]
        for col in columns:
            stats_exprs.extend([
                pl.col(col).quantile(0.25).alias(f"_q1_{col}"),
                pl.col(col).quantile(0.75).alias(f"_q3_{col}"),
                pl.col(col).count().alias(f"_cnt_{col}"),
            ])

        stats = lf.select(stats_exprs).collect()
        total_rows = stats["_total"][0]

        if total_rows < 4:
            return issues

        # Calculate outliers
        outlier_exprs = []
        bounds_info = {}

        for col in columns:
            q1 = stats[f"_q1_{col}"][0]
            q3 = stats[f"_q3_{col}"][0]
            cnt = stats[f"_cnt_{col}"][0]

            if q1 is None or q3 is None or cnt < 4:
                continue

            iqr = q3 - q1
            if iqr == 0:
                continue

            lower = q1 - self.iqr_multiplier * iqr
            upper = q3 + self.iqr_multiplier * iqr

            bounds_info[col] = (lower, upper, cnt)
            outlier_exprs.append(
                ((pl.col(col) < lower) | (pl.col(col) > upper))
                .sum()
                .alias(f"_out_{col}")
            )

        if not outlier_exprs:
            return issues

        outlier_counts = lf.select(outlier_exprs).collect()

        for col, (lower, upper, cnt) in bounds_info.items():
            outlier_count = outlier_counts[f"_out_{col}"][0]

            if outlier_count > 0:
                outlier_pct = outlier_count / cnt
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="outlier",
                        count=outlier_count,
                        severity=Severity.MEDIUM if outlier_pct > 0.1 else Severity.LOW,
                        details=f"IQR bounds: [{lower:.2f}, {upper:.2f}]",
                        expected=f"[{lower:.2f}, {upper:.2f}]",
                    )
                )

        return issues


@register_validator
class ZScoreOutlierValidator(Validator, NumericValidatorMixin):
    """Detects outliers using Z-score method."""

    name = "zscore_outlier"
    category = "distribution"

    def __init__(
        self,
        threshold: float = 3.0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.threshold = threshold

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        columns = self._get_numeric_columns(lf)

        if not columns:
            return issues

        # Compute mean and std
        stats_exprs = [pl.len().alias("_total")]
        for col in columns:
            stats_exprs.extend([
                pl.col(col).mean().alias(f"_mean_{col}"),
                pl.col(col).std().alias(f"_std_{col}"),
                pl.col(col).count().alias(f"_cnt_{col}"),
            ])

        stats = lf.select(stats_exprs).collect()
        total_rows = stats["_total"][0]

        if total_rows < 3:
            return issues

        # Calculate Z-score outliers
        outlier_exprs = []
        stats_info = {}

        for col in columns:
            mean = stats[f"_mean_{col}"][0]
            std = stats[f"_std_{col}"][0]
            cnt = stats[f"_cnt_{col}"][0]

            if mean is None or std is None or std == 0 or cnt < 3:
                continue

            stats_info[col] = (mean, std, cnt)
            z_score = (pl.col(col) - mean).abs() / std
            outlier_exprs.append(
                (z_score > self.threshold).sum().alias(f"_out_{col}")
            )

        if not outlier_exprs:
            return issues

        outlier_counts = lf.select(outlier_exprs).collect()

        for col, (mean, std, cnt) in stats_info.items():
            outlier_count = outlier_counts[f"_out_{col}"][0]

            if outlier_count > 0:
                outlier_pct = outlier_count / cnt
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="zscore_outlier",
                        count=outlier_count,
                        severity=Severity.MEDIUM if outlier_pct > 0.1 else Severity.LOW,
                        details=f"|Z-score| > {self.threshold} (mean={mean:.2f}, std={std:.2f})",
                    )
                )

        return issues
