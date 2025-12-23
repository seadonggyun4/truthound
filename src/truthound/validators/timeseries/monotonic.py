"""Time series monotonicity validators.

This module provides validators for checking monotonicity constraints
in time series data.
"""

from enum import Enum
from typing import Any

import polars as pl
import numpy as np

from truthound.types import Severity
from truthound.validators.base import ValidationIssue
from truthound.validators.registry import register_validator
from truthound.validators.timeseries.base import (
    TimeFrequency,
    ValueTimeSeriesValidator,
)


class MonotonicityType(str, Enum):
    """Types of monotonicity constraints."""

    STRICTLY_INCREASING = "strictly_increasing"
    STRICTLY_DECREASING = "strictly_decreasing"
    NON_DECREASING = "non_decreasing"
    NON_INCREASING = "non_increasing"


@register_validator
class TimeSeriesMonotonicValidator(ValueTimeSeriesValidator):
    """Validates monotonicity of time series values.

    Checks that values follow a specified monotonic pattern:
    - Strictly increasing: each value > previous
    - Strictly decreasing: each value < previous
    - Non-decreasing: each value >= previous
    - Non-increasing: each value <= previous

    Useful for:
    - Cumulative counters
    - Running totals
    - Ordered sequences

    Example:
        validator = TimeSeriesMonotonicValidator(
            timestamp_column="timestamp",
            value_column="cumulative_sales",
            monotonicity=MonotonicityType.NON_DECREASING,
        )
    """

    name = "timeseries_monotonic"

    def __init__(
        self,
        timestamp_column: str,
        value_column: str,
        monotonicity: MonotonicityType | str = MonotonicityType.NON_DECREASING,
        max_violations: int = 0,
        max_violation_ratio: float = 0.0,
        frequency: TimeFrequency | str | None = None,
        **kwargs: Any,
    ):
        """Initialize monotonicity validator.

        Args:
            timestamp_column: Column containing timestamps
            value_column: Column containing values to check
            monotonicity: Type of monotonicity to enforce
            max_violations: Maximum allowed violations
            max_violation_ratio: Maximum ratio of violations
            frequency: Expected frequency
            **kwargs: Additional config
        """
        super().__init__(
            timestamp_column=timestamp_column,
            value_column=value_column,
            frequency=frequency,
            **kwargs,
        )

        if isinstance(monotonicity, str):
            self.monotonicity = MonotonicityType(monotonicity)
        else:
            self.monotonicity = monotonicity

        self.max_violations = max_violations
        self.max_violation_ratio = max_violation_ratio

    def _check_monotonicity(
        self, values: np.ndarray
    ) -> tuple[np.ndarray, str]:
        """Check monotonicity and return violation mask.

        Args:
            values: Array of values

        Returns:
            Tuple of (violation_mask, description)
        """
        if len(values) < 2:
            return np.array([]), ""

        diffs = np.diff(values)

        if self.monotonicity == MonotonicityType.STRICTLY_INCREASING:
            violations = diffs <= 0
            desc = "strictly increasing (each value > previous)"
        elif self.monotonicity == MonotonicityType.STRICTLY_DECREASING:
            violations = diffs >= 0
            desc = "strictly decreasing (each value < previous)"
        elif self.monotonicity == MonotonicityType.NON_DECREASING:
            violations = diffs < 0
            desc = "non-decreasing (each value >= previous)"
        else:  # NON_INCREASING
            violations = diffs > 0
            desc = "non-increasing (each value <= previous)"

        return violations, desc

    def validate_series(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate monotonicity of time series.

        Args:
            lf: Input LazyFrame

        Returns:
            List of validation issues
        """
        issues: list[ValidationIssue] = []

        df = self._get_sorted_series(lf)
        if len(df) < 2:
            return issues

        values = df[self.value_column].to_numpy()
        timestamps = df[self.timestamp_column].to_numpy()

        # Handle NaN values
        valid_mask = ~np.isnan(values)
        if not np.all(valid_mask):
            values = values[valid_mask]
            timestamps = timestamps[valid_mask]

        if len(values) < 2:
            return issues

        violations, desc = self._check_monotonicity(values)
        violation_count = int(np.sum(violations))
        violation_ratio = violation_count / len(violations) if len(violations) > 0 else 0

        has_issue = (
            violation_count > self.max_violations
            or violation_ratio > self.max_violation_ratio
        )

        if has_issue and violation_count > 0:
            # Get sample violations
            violation_indices = np.where(violations)[0]
            sample_indices = violation_indices[:5]

            samples = []
            for i in sample_indices:
                samples.append({
                    "index": int(i),
                    "timestamp": str(timestamps[i]),
                    "value": float(values[i]),
                    "next_value": float(values[i + 1]),
                    "diff": float(values[i + 1] - values[i]),
                })

            issues.append(
                ValidationIssue(
                    column=self.value_column,
                    issue_type="monotonicity_violation",
                    count=violation_count,
                    severity=self._calculate_severity(violation_ratio),
                    details=(
                        f"Found {violation_count} violations ({violation_ratio:.2%}). "
                        f"Expected {desc}. Sample violations: {samples}"
                    ),
                    expected=f"Values should be {desc}",
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
        return self.validate_series(lf)


@register_validator
class TimeSeriesOrderValidator(ValueTimeSeriesValidator):
    """Validates that timestamps are in chronological order.

    Ensures the time series is properly ordered, detecting:
    - Out-of-order timestamps
    - Time reversals
    - Future timestamps appearing before past ones

    Example:
        validator = TimeSeriesOrderValidator(
            timestamp_column="event_time",
        )
    """

    name = "timeseries_order"

    def __init__(
        self,
        timestamp_column: str,
        value_column: str | None = None,
        allow_equal: bool = True,
        max_violations: int = 0,
        **kwargs: Any,
    ):
        """Initialize order validator.

        Args:
            timestamp_column: Column containing timestamps
            value_column: Optional value column (not used for ordering)
            allow_equal: Whether to allow equal consecutive timestamps
            max_violations: Maximum allowed out-of-order pairs
            **kwargs: Additional config
        """
        super().__init__(
            timestamp_column=timestamp_column,
            value_column=value_column or timestamp_column,
            **kwargs,
        )
        self.allow_equal = allow_equal
        self.max_violations = max_violations

    def validate_series(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate timestamp ordering.

        Args:
            lf: Input LazyFrame

        Returns:
            List of validation issues
        """
        issues: list[ValidationIssue] = []

        # Get timestamps in original order (not sorted)
        df = lf.select(pl.col(self.timestamp_column)).collect()
        if len(df) < 2:
            return issues

        timestamps = df[self.timestamp_column].to_numpy()
        diffs = np.diff(timestamps.astype("datetime64[ns]").astype(np.int64))

        if self.allow_equal:
            violations = diffs < 0
        else:
            violations = diffs <= 0

        violation_count = int(np.sum(violations))

        if violation_count > self.max_violations:
            violation_indices = np.where(violations)[0]
            sample_indices = violation_indices[:5]

            samples = []
            for i in sample_indices:
                samples.append({
                    "index": int(i),
                    "timestamp": str(timestamps[i]),
                    "next_timestamp": str(timestamps[i + 1]),
                })

            issues.append(
                ValidationIssue(
                    column=self.timestamp_column,
                    issue_type="timestamp_order_violation",
                    count=violation_count,
                    severity=(
                        Severity.CRITICAL if violation_count > len(timestamps) * 0.1
                        else Severity.HIGH
                    ),
                    details=(
                        f"Found {violation_count} out-of-order timestamp pairs. "
                        f"Samples: {samples}"
                    ),
                    expected="Timestamps should be in chronological order",
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
        return self.validate_series(lf)
