"""Cardinality-based profiling validators.

This module provides validators that analyze column cardinality
(the ratio of unique values to total values).
"""

from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue
from truthound.validators.registry import register_validator
from truthound.validators.profiling.base import (
    ProfileMetrics,
    ProfilingValidator,
)


@register_validator
class CardinalityValidator(ProfilingValidator):
    """Validates column cardinality is within expected range.

    Cardinality = unique_count / total_count

    Use cases:
    - Detect ID columns (cardinality ≈ 1.0)
    - Detect categorical columns (low cardinality)
    - Detect potential data issues (unexpected cardinality changes)

    Example:
        validator = CardinalityValidator(
            column="user_id",
            min_cardinality=0.9,  # Expect high uniqueness
            max_cardinality=1.0,
        )
    """

    name = "cardinality"

    def __init__(
        self,
        column: str,
        min_cardinality: float = 0.0,
        max_cardinality: float = 1.0,
        min_unique_count: int | None = None,
        max_unique_count: int | None = None,
        **kwargs: Any,
    ):
        """Initialize cardinality validator.

        Args:
            column: Column to validate
            min_cardinality: Minimum cardinality ratio (0-1)
            max_cardinality: Maximum cardinality ratio (0-1)
            min_unique_count: Minimum absolute unique count
            max_unique_count: Maximum absolute unique count
            **kwargs: Additional config
        """
        super().__init__(column=column, **kwargs)
        self.min_cardinality = min_cardinality
        self.max_cardinality = max_cardinality
        self.min_unique_count = min_unique_count
        self.max_unique_count = max_unique_count

    def validate_profile(
        self, df: pl.DataFrame, metrics: ProfileMetrics
    ) -> list[ValidationIssue]:
        """Validate cardinality constraints.

        Args:
            df: Input DataFrame
            metrics: Profile metrics

        Returns:
            List of validation issues
        """
        issues: list[ValidationIssue] = []

        if metrics.total_count == 0:
            return issues

        # Check cardinality ratio
        if metrics.cardinality < self.min_cardinality:
            issues.append(
                ValidationIssue(
                    column=self.column,
                    issue_type="cardinality_too_low",
                    count=1,
                    severity=Severity.MEDIUM,
                    details=(
                        f"Cardinality ({metrics.cardinality:.4f}) below minimum "
                        f"({self.min_cardinality:.4f}). "
                        f"Unique: {metrics.unique_count}, Total: {metrics.total_count}"
                    ),
                    expected=f"Cardinality >= {self.min_cardinality}",
                )
            )

        if metrics.cardinality > self.max_cardinality:
            issues.append(
                ValidationIssue(
                    column=self.column,
                    issue_type="cardinality_too_high",
                    count=1,
                    severity=Severity.MEDIUM,
                    details=(
                        f"Cardinality ({metrics.cardinality:.4f}) above maximum "
                        f"({self.max_cardinality:.4f}). "
                        f"Unique: {metrics.unique_count}, Total: {metrics.total_count}"
                    ),
                    expected=f"Cardinality <= {self.max_cardinality}",
                )
            )

        # Check absolute unique count
        if self.min_unique_count is not None:
            if metrics.unique_count < self.min_unique_count:
                issues.append(
                    ValidationIssue(
                        column=self.column,
                        issue_type="unique_count_too_low",
                        count=metrics.unique_count,
                        severity=Severity.MEDIUM,
                        details=(
                            f"Unique count ({metrics.unique_count}) below minimum "
                            f"({self.min_unique_count})."
                        ),
                        expected=f"Unique count >= {self.min_unique_count}",
                    )
                )

        if self.max_unique_count is not None:
            if metrics.unique_count > self.max_unique_count:
                issues.append(
                    ValidationIssue(
                        column=self.column,
                        issue_type="unique_count_too_high",
                        count=metrics.unique_count,
                        severity=Severity.MEDIUM,
                        details=(
                            f"Unique count ({metrics.unique_count}) above maximum "
                            f"({self.max_unique_count})."
                        ),
                        expected=f"Unique count <= {self.max_unique_count}",
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
        df = lf.select(pl.col(self.column)).collect()
        metrics = self._compute_metrics(df)
        return self.validate_profile(df, metrics)


@register_validator
class UniquenessRatioValidator(ProfilingValidator):
    """Validates uniqueness ratio with detailed analysis.

    Extends cardinality validation with:
    - Duplicate detection
    - Top duplicate values
    - Uniqueness trend analysis

    Example:
        validator = UniquenessRatioValidator(
            column="email",
            min_uniqueness=0.99,
            report_top_duplicates=5,
        )
    """

    name = "uniqueness_ratio"

    def __init__(
        self,
        column: str,
        min_uniqueness: float = 0.0,
        max_uniqueness: float = 1.0,
        report_top_duplicates: int = 5,
        **kwargs: Any,
    ):
        """Initialize uniqueness ratio validator.

        Args:
            column: Column to validate
            min_uniqueness: Minimum uniqueness ratio
            max_uniqueness: Maximum uniqueness ratio
            report_top_duplicates: Number of top duplicates to report
            **kwargs: Additional config
        """
        super().__init__(column=column, **kwargs)
        self.min_uniqueness = min_uniqueness
        self.max_uniqueness = max_uniqueness
        self.report_top_duplicates = report_top_duplicates

    def _get_top_duplicates(
        self, df: pl.DataFrame, n: int
    ) -> list[tuple[Any, int]]:
        """Get top n most duplicated values.

        Args:
            df: Input DataFrame
            n: Number of duplicates to return

        Returns:
            List of (value, count) tuples
        """
        value_counts = self._compute_value_counts(df)

        # Filter to only duplicates
        duplicates = value_counts.filter(pl.col("count") > 1).head(n)

        return [
            (row[0], row[1])
            for row in duplicates.iter_rows()
        ]

    def validate_profile(
        self, df: pl.DataFrame, metrics: ProfileMetrics
    ) -> list[ValidationIssue]:
        """Validate uniqueness ratio.

        Args:
            df: Input DataFrame
            metrics: Profile metrics

        Returns:
            List of validation issues
        """
        issues: list[ValidationIssue] = []

        if metrics.total_count == 0:
            return issues

        uniqueness = metrics.cardinality

        if uniqueness < self.min_uniqueness:
            # Get top duplicates for details
            top_dupes = self._get_top_duplicates(df, self.report_top_duplicates)
            duplicate_count = metrics.total_count - metrics.unique_count

            dupe_details = ""
            if top_dupes:
                dupe_strs = [f"{val!r}: {cnt}x" for val, cnt in top_dupes]
                dupe_details = f" Top duplicates: {', '.join(dupe_strs)}"

            issues.append(
                ValidationIssue(
                    column=self.column,
                    issue_type="uniqueness_too_low",
                    count=duplicate_count,
                    severity=self._calculate_severity(1 - uniqueness),
                    details=(
                        f"Uniqueness ratio ({uniqueness:.4f}) below minimum "
                        f"({self.min_uniqueness:.4f}). "
                        f"{duplicate_count} duplicate values.{dupe_details}"
                    ),
                    expected=f"Uniqueness >= {self.min_uniqueness}",
                )
            )

        if uniqueness > self.max_uniqueness:
            issues.append(
                ValidationIssue(
                    column=self.column,
                    issue_type="uniqueness_too_high",
                    count=1,
                    severity=Severity.LOW,
                    details=(
                        f"Uniqueness ratio ({uniqueness:.4f}) above maximum "
                        f"({self.max_uniqueness:.4f}). "
                        f"This column may be more unique than expected."
                    ),
                    expected=f"Uniqueness <= {self.max_uniqueness}",
                )
            )

        return issues

    def _calculate_severity(self, duplicate_ratio: float) -> Severity:
        """Calculate severity based on duplicate ratio."""
        if duplicate_ratio < 0.01:
            return Severity.LOW
        elif duplicate_ratio < 0.05:
            return Severity.MEDIUM
        elif duplicate_ratio < 0.1:
            return Severity.HIGH
        else:
            return Severity.CRITICAL

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate the LazyFrame.

        Args:
            lf: Input LazyFrame

        Returns:
            List of validation issues
        """
        df = lf.select(pl.col(self.column)).collect()
        metrics = self._compute_metrics(df)
        return self.validate_profile(df, metrics)
