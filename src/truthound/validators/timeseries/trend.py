"""Time series trend detection validators.

This module provides validators for detecting and validating
trends in time series data.
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


class TrendDirection(str, Enum):
    """Trend direction types."""

    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    ANY = "any"


@register_validator
class TrendValidator(ValueTimeSeriesValidator):
    """Validates trend characteristics in time series data.

    Detects:
    - Presence or absence of expected trends
    - Trend direction changes
    - Trend strength and significance

    Uses linear regression to estimate trend.

    Example:
        validator = TrendValidator(
            timestamp_column="date",
            value_column="revenue",
            expected_direction=TrendDirection.INCREASING,
            min_trend_strength=0.01,
        )
    """

    name = "trend"

    def __init__(
        self,
        timestamp_column: str,
        value_column: str,
        expected_direction: TrendDirection | str = TrendDirection.ANY,
        min_trend_strength: float | None = None,
        max_trend_strength: float | None = None,
        min_r_squared: float | None = None,
        detect_trend_change: bool = False,
        window_size: int | None = None,
        frequency: TimeFrequency | str | None = None,
        **kwargs: Any,
    ):
        """Initialize trend validator.

        Args:
            timestamp_column: Column containing timestamps
            value_column: Column containing values
            expected_direction: Expected trend direction
            min_trend_strength: Minimum absolute slope (per data point)
            max_trend_strength: Maximum absolute slope
            min_r_squared: Minimum R² for trend fit
            detect_trend_change: Whether to detect trend direction changes
            window_size: Window size for trend change detection
            frequency: Expected data frequency
            **kwargs: Additional config
        """
        super().__init__(
            timestamp_column=timestamp_column,
            value_column=value_column,
            frequency=frequency,
            **kwargs,
        )

        if isinstance(expected_direction, str):
            self.expected_direction = TrendDirection(expected_direction)
        else:
            self.expected_direction = expected_direction

        self.min_trend_strength = min_trend_strength
        self.max_trend_strength = max_trend_strength
        self.min_r_squared = min_r_squared
        self.detect_trend_change = detect_trend_change
        self.window_size = window_size

    def _compute_trend(
        self, values: np.ndarray
    ) -> tuple[float, float, float]:
        """Compute linear trend statistics.

        Args:
            values: Array of values

        Returns:
            Tuple of (slope, intercept, r_squared)
        """
        n = len(values)
        x = np.arange(n)

        # Linear regression
        x_mean = np.mean(x)
        y_mean = np.mean(values)

        numerator = np.sum((x - x_mean) * (values - y_mean))
        denominator = np.sum((x - x_mean) ** 2)

        if denominator == 0:
            return 0.0, y_mean, 0.0

        slope = numerator / denominator
        intercept = y_mean - slope * x_mean

        # R-squared
        y_pred = slope * x + intercept
        ss_res = np.sum((values - y_pred) ** 2)
        ss_tot = np.sum((values - y_mean) ** 2)

        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return float(slope), float(intercept), float(r_squared)

    def _detect_direction_changes(
        self, values: np.ndarray, window: int
    ) -> list[tuple[int, str, str]]:
        """Detect trend direction changes.

        Args:
            values: Array of values
            window: Window size for local trend

        Returns:
            List of (index, from_direction, to_direction) tuples
        """
        changes = []
        n = len(values)

        if n < window * 2:
            return changes

        prev_direction = None

        for i in range(0, n - window, window // 2):
            window_values = values[i : i + window]
            slope, _, _ = self._compute_trend(window_values)

            if slope > 0.001:
                direction = "increasing"
            elif slope < -0.001:
                direction = "decreasing"
            else:
                direction = "stable"

            if prev_direction is not None and direction != prev_direction:
                changes.append((i, prev_direction, direction))

            prev_direction = direction

        return changes

    def validate_series(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate trend in time series.

        Args:
            lf: Input LazyFrame

        Returns:
            List of validation issues
        """
        issues: list[ValidationIssue] = []

        df = self._get_sorted_series(lf)
        if len(df) < 3:
            return issues

        values = df[self.value_column].to_numpy()

        # Handle NaN values
        valid_mask = ~np.isnan(values)
        if not np.all(valid_mask):
            values = values[valid_mask]

        if len(values) < 3:
            return issues

        # Compute overall trend
        slope, intercept, r_squared = self._compute_trend(values)

        # Determine actual direction
        if slope > 0.001:
            actual_direction = TrendDirection.INCREASING
        elif slope < -0.001:
            actual_direction = TrendDirection.DECREASING
        else:
            actual_direction = TrendDirection.STABLE

        # Check expected direction
        if self.expected_direction != TrendDirection.ANY:
            if actual_direction != self.expected_direction:
                issues.append(
                    ValidationIssue(
                        column=self.value_column,
                        issue_type="unexpected_trend_direction",
                        count=1,
                        severity=Severity.MEDIUM,
                        details=(
                            f"Expected {self.expected_direction.value} trend, "
                            f"found {actual_direction.value}. "
                            f"Slope: {slope:.6f}, R²: {r_squared:.3f}"
                        ),
                        expected=f"Trend direction: {self.expected_direction.value}",
                    )
                )

        # Check trend strength
        abs_slope = abs(slope)

        if self.min_trend_strength is not None:
            if abs_slope < self.min_trend_strength:
                issues.append(
                    ValidationIssue(
                        column=self.value_column,
                        issue_type="weak_trend",
                        count=1,
                        severity=Severity.LOW,
                        details=(
                            f"Trend strength ({abs_slope:.6f}) below minimum "
                            f"({self.min_trend_strength:.6f}). R²: {r_squared:.3f}"
                        ),
                        expected=f"Trend strength >= {self.min_trend_strength}",
                    )
                )

        if self.max_trend_strength is not None:
            if abs_slope > self.max_trend_strength:
                issues.append(
                    ValidationIssue(
                        column=self.value_column,
                        issue_type="excessive_trend",
                        count=1,
                        severity=Severity.MEDIUM,
                        details=(
                            f"Trend strength ({abs_slope:.6f}) exceeds maximum "
                            f"({self.max_trend_strength:.6f}). R²: {r_squared:.3f}"
                        ),
                        expected=f"Trend strength <= {self.max_trend_strength}",
                    )
                )

        # Check R-squared
        if self.min_r_squared is not None:
            if r_squared < self.min_r_squared:
                issues.append(
                    ValidationIssue(
                        column=self.value_column,
                        issue_type="poor_trend_fit",
                        count=1,
                        severity=Severity.LOW,
                        details=(
                            f"Trend R² ({r_squared:.3f}) below minimum "
                            f"({self.min_r_squared:.3f}). Data may not follow "
                            f"a linear trend."
                        ),
                        expected=f"Trend R² >= {self.min_r_squared}",
                    )
                )

        # Detect trend changes
        if self.detect_trend_change:
            window = self.window_size or max(10, len(values) // 10)
            changes = self._detect_direction_changes(values, window)

            if changes:
                change_details = [
                    f"[{idx}]: {from_dir} -> {to_dir}"
                    for idx, from_dir, to_dir in changes[:5]
                ]

                issues.append(
                    ValidationIssue(
                        column=self.value_column,
                        issue_type="trend_direction_change",
                        count=len(changes),
                        severity=Severity.LOW,
                        details=(
                            f"Detected {len(changes)} trend direction change(s). "
                            f"Changes: {'; '.join(change_details)}"
                        ),
                        expected="Stable trend direction",
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
class TrendBreakValidator(ValueTimeSeriesValidator):
    """Validates for structural breaks in time series trends.

    Detects sudden changes in trend that may indicate:
    - Regime changes
    - Data collection issues
    - External events

    Example:
        validator = TrendBreakValidator(
            timestamp_column="date",
            value_column="price",
            min_break_magnitude=0.1,
        )
    """

    name = "trend_break"

    def __init__(
        self,
        timestamp_column: str,
        value_column: str,
        min_break_magnitude: float = 0.05,
        window_size: int = 10,
        max_breaks: int | None = None,
        frequency: TimeFrequency | str | None = None,
        **kwargs: Any,
    ):
        """Initialize trend break validator.

        Args:
            timestamp_column: Column containing timestamps
            value_column: Column containing values
            min_break_magnitude: Minimum relative change to consider a break
            window_size: Window for computing local statistics
            max_breaks: Maximum allowed structural breaks
            frequency: Expected data frequency
            **kwargs: Additional config
        """
        super().__init__(
            timestamp_column=timestamp_column,
            value_column=value_column,
            frequency=frequency,
            **kwargs,
        )
        self.min_break_magnitude = min_break_magnitude
        self.window_size = window_size
        self.max_breaks = max_breaks

    def _detect_breaks(
        self, values: np.ndarray
    ) -> list[tuple[int, float, float]]:
        """Detect structural breaks.

        Args:
            values: Array of values

        Returns:
            List of (index, before_mean, after_mean) tuples
        """
        breaks = []
        n = len(values)
        window = self.window_size

        if n < window * 2:
            return breaks

        for i in range(window, n - window):
            before = values[i - window : i]
            after = values[i : i + window]

            before_mean = np.mean(before)
            after_mean = np.mean(after)

            if before_mean == 0:
                continue

            relative_change = abs(after_mean - before_mean) / abs(before_mean)

            if relative_change >= self.min_break_magnitude:
                # Check if this is a local maximum of change
                is_peak = True
                for j in range(max(window, i - 3), min(n - window, i + 4)):
                    if j == i:
                        continue
                    b = values[j - window : j]
                    a = values[j : j + window]
                    b_mean = np.mean(b)
                    a_mean = np.mean(a)
                    if b_mean != 0:
                        other_change = abs(a_mean - b_mean) / abs(b_mean)
                        if other_change > relative_change:
                            is_peak = False
                            break

                if is_peak:
                    breaks.append((i, float(before_mean), float(after_mean)))

        return breaks

    def validate_series(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate for trend breaks.

        Args:
            lf: Input LazyFrame

        Returns:
            List of validation issues
        """
        issues: list[ValidationIssue] = []

        df = self._get_sorted_series(lf)
        if len(df) < self.window_size * 2:
            return issues

        values = df[self.value_column].to_numpy()
        timestamps = df[self.timestamp_column].to_numpy()

        # Handle NaN values
        valid_mask = ~np.isnan(values)
        if not np.all(valid_mask):
            values = values[valid_mask]
            timestamps = timestamps[valid_mask]

        if len(values) < self.window_size * 2:
            return issues

        # Detect breaks
        breaks = self._detect_breaks(values)

        if self.max_breaks is not None and len(breaks) > self.max_breaks:
            break_details = []
            for idx, before, after in breaks[:5]:
                change_pct = (after - before) / before * 100 if before != 0 else 0
                break_details.append(
                    f"[{timestamps[idx]}]: {before:.2f} -> {after:.2f} "
                    f"({change_pct:+.1f}%)"
                )

            issues.append(
                ValidationIssue(
                    column=self.value_column,
                    issue_type="structural_breaks_detected",
                    count=len(breaks),
                    severity=(
                        Severity.HIGH if len(breaks) > self.max_breaks * 2
                        else Severity.MEDIUM
                    ),
                    details=(
                        f"Detected {len(breaks)} structural breaks "
                        f"(max allowed: {self.max_breaks}). "
                        f"Breaks: {'; '.join(break_details)}"
                    ),
                    expected=f"<= {self.max_breaks} structural breaks",
                )
            )
        elif len(breaks) > 0 and self.max_breaks is None:
            # Report breaks even without limit (informational)
            break_details = []
            for idx, before, after in breaks[:5]:
                change_pct = (after - before) / before * 100 if before != 0 else 0
                break_details.append(
                    f"[{timestamps[idx]}]: {before:.2f} -> {after:.2f} "
                    f"({change_pct:+.1f}%)"
                )

            issues.append(
                ValidationIssue(
                    column=self.value_column,
                    issue_type="structural_breaks_detected",
                    count=len(breaks),
                    severity=Severity.LOW,
                    details=(
                        f"Detected {len(breaks)} potential structural break(s). "
                        f"Breaks: {'; '.join(break_details)}"
                    ),
                    expected="Smooth trend transitions",
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
