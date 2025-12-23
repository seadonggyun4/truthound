"""Feature null impact validators.

This module provides validators for analyzing the impact
of null values on ML model features.
"""

from typing import Any

import polars as pl
import numpy as np

from truthound.types import Severity
from truthound.validators.base import ValidationIssue
from truthound.validators.registry import register_validator
from truthound.validators.ml_feature.base import MLFeatureValidator


@register_validator
class FeatureNullImpactValidator(MLFeatureValidator):
    """Validates the impact of null values on features.

    Analyzes:
    - Null ratio per feature (high null ratio = problematic)
    - Null correlation between features (systematic missingness)
    - Null pattern detection (MCAR, MAR, MNAR)
    - Impact on feature distributions

    Example:
        validator = FeatureNullImpactValidator(
            max_null_ratio=0.3,
            detect_null_correlation=True,
        )
    """

    name = "feature_null_impact"

    def __init__(
        self,
        columns: list[str] | None = None,
        max_null_ratio: float = 0.5,
        warn_null_ratio: float = 0.1,
        detect_null_correlation: bool = True,
        null_correlation_threshold: float = 0.7,
        min_samples_for_analysis: int = 100,
        **kwargs: Any,
    ):
        """Initialize null impact validator.

        Args:
            columns: Specific columns to validate
            max_null_ratio: Maximum acceptable null ratio
            warn_null_ratio: Null ratio to trigger warning
            detect_null_correlation: Whether to detect null correlations
            null_correlation_threshold: Threshold for null correlation detection
            min_samples_for_analysis: Minimum samples for correlation analysis
            **kwargs: Additional config
        """
        super().__init__(columns=columns, numeric_only=False, **kwargs)
        self.max_null_ratio = max_null_ratio
        self.warn_null_ratio = warn_null_ratio
        self.detect_null_correlation = detect_null_correlation
        self.null_correlation_threshold = null_correlation_threshold
        self.min_samples_for_analysis = min_samples_for_analysis

    def _compute_null_correlation(
        self, df: pl.DataFrame, col1: str, col2: str
    ) -> float | None:
        """Compute correlation between null patterns of two columns.

        Args:
            df: Input DataFrame
            col1: First column
            col2: Second column

        Returns:
            Correlation of null indicators
        """
        if len(df) < self.min_samples_for_analysis:
            return None

        null1 = df[col1].is_null().cast(pl.Int8).to_numpy()
        null2 = df[col2].is_null().cast(pl.Int8).to_numpy()

        # Both must have some nulls
        if null1.sum() == 0 or null2.sum() == 0:
            return None

        # Compute correlation
        n1_mean = np.mean(null1)
        n2_mean = np.mean(null2)
        n1_std = np.std(null1)
        n2_std = np.std(null2)

        if n1_std == 0 or n2_std == 0:
            return None

        correlation = np.mean((null1 - n1_mean) * (null2 - n2_mean)) / (n1_std * n2_std)
        return float(correlation)

    def _analyze_null_pattern(
        self, df: pl.DataFrame, column: str
    ) -> dict[str, Any]:
        """Analyze the null pattern for a column.

        Attempts to classify as:
        - MCAR: Missing Completely At Random
        - MAR: Missing At Random (depends on other columns)
        - MNAR: Missing Not At Random (depends on own value)

        Args:
            df: Input DataFrame
            column: Column to analyze

        Returns:
            Dictionary with pattern analysis
        """
        result = {
            "column": column,
            "null_ratio": 0.0,
            "pattern": "unknown",
            "correlated_columns": [],
        }

        total = len(df)
        null_count = df[column].null_count()
        result["null_ratio"] = null_count / total if total > 0 else 0.0

        if null_count == 0:
            result["pattern"] = "no_nulls"
            return result

        # Check for correlation with other columns
        if self.detect_null_correlation:
            null_indicator = df[column].is_null().cast(pl.Int8)

            for other_col in df.columns:
                if other_col == column:
                    continue

                # Check if null pattern correlates with other column values
                other_null_count = df[other_col].null_count()
                if other_null_count == total:
                    continue

                corr = self._compute_null_correlation(df, column, other_col)
                if corr is not None and abs(corr) >= self.null_correlation_threshold:
                    result["correlated_columns"].append((other_col, corr))

        if result["correlated_columns"]:
            result["pattern"] = "MAR"
        else:
            result["pattern"] = "possibly_MCAR"

        return result

    def validate_features(
        self, df: pl.DataFrame, columns: list[str]
    ) -> list[ValidationIssue]:
        """Validate null impact for features.

        Args:
            df: Input DataFrame
            columns: List of feature columns

        Returns:
            List of validation issues
        """
        issues: list[ValidationIssue] = []
        total = len(df)

        if total == 0:
            return issues

        high_null_features: list[tuple[str, float]] = []
        warning_null_features: list[tuple[str, float]] = []
        null_correlations: list[tuple[str, str, float]] = []

        for col in columns:
            null_count = df[col].null_count()
            null_ratio = null_count / total

            if null_ratio > self.max_null_ratio:
                high_null_features.append((col, null_ratio))
            elif null_ratio > self.warn_null_ratio:
                warning_null_features.append((col, null_ratio))

        # Check for null correlations
        if self.detect_null_correlation and len(columns) > 1:
            for i, col1 in enumerate(columns):
                for col2 in columns[i + 1 :]:
                    corr = self._compute_null_correlation(df, col1, col2)
                    if corr is not None and abs(corr) >= self.null_correlation_threshold:
                        null_correlations.append((col1, col2, corr))

        # Report high null features
        if high_null_features:
            feature_details = [
                f"{name}: {ratio:.1%}" for name, ratio in high_null_features[:5]
            ]
            issues.append(
                ValidationIssue(
                    column="multiple",
                    issue_type="high_null_ratio",
                    count=len(high_null_features),
                    severity=Severity.HIGH,
                    details=(
                        f"Found {len(high_null_features)} features with high null ratio "
                        f"(>{self.max_null_ratio:.0%}). Features: {', '.join(feature_details)}"
                    ),
                    expected=f"Null ratio <= {self.max_null_ratio:.0%}",
                )
            )

        # Report warning-level null features
        if warning_null_features:
            feature_details = [
                f"{name}: {ratio:.1%}" for name, ratio in warning_null_features[:5]
            ]
            issues.append(
                ValidationIssue(
                    column="multiple",
                    issue_type="elevated_null_ratio",
                    count=len(warning_null_features),
                    severity=Severity.MEDIUM,
                    details=(
                        f"Found {len(warning_null_features)} features with elevated null ratio "
                        f"(>{self.warn_null_ratio:.0%}). Features: {', '.join(feature_details)}"
                    ),
                    expected=f"Null ratio <= {self.warn_null_ratio:.0%}",
                )
            )

        # Report null correlations
        if null_correlations:
            corr_details = [
                f"{c1} <-> {c2}: {corr:.2f}"
                for c1, c2, corr in null_correlations[:5]
            ]
            issues.append(
                ValidationIssue(
                    column="multiple",
                    issue_type="null_pattern_correlation",
                    count=len(null_correlations),
                    severity=Severity.MEDIUM,
                    details=(
                        f"Found {len(null_correlations)} pairs of features with correlated "
                        f"null patterns (MAR indicator). Pairs: {'; '.join(corr_details)}"
                    ),
                    expected="Independent null patterns (MCAR)",
                )
            )

        return issues

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate the LazyFrame.

        Args:
            lf: Input LazyFrame

        Returns:
            List of validation issues
        """
        df = lf.collect()
        columns = self._get_feature_columns(df)

        if not columns:
            return []

        return self.validate_features(df, columns)
