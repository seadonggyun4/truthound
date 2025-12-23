"""Time series seasonality validators.

This module provides validators for detecting and validating
seasonal patterns in time series data.
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


class SeasonalPeriod(str, Enum):
    """Common seasonal periods."""

    HOURLY = "hourly"  # 24 periods per day
    DAILY = "daily"  # 7 periods per week
    WEEKLY = "weekly"  # 4-5 periods per month
    MONTHLY = "monthly"  # 12 periods per year
    QUARTERLY = "quarterly"  # 4 periods per year


@register_validator
class SeasonalityValidator(ValueTimeSeriesValidator):
    """Validates seasonal patterns in time series data.

    Detects:
    - Presence or absence of expected seasonality
    - Strength of seasonal patterns
    - Irregular seasonal variations

    Uses autocorrelation analysis to detect periodicity.

    Example:
        validator = SeasonalityValidator(
            timestamp_column="date",
            value_column="sales",
            expected_period=12,  # Monthly data with yearly seasonality
            min_seasonality_strength=0.3,
        )
    """

    name = "seasonality"

    def __init__(
        self,
        timestamp_column: str,
        value_column: str,
        expected_period: int | None = None,
        min_seasonality_strength: float = 0.3,
        max_seasonality_strength: float | None = None,
        detect_period: bool = True,
        max_period_search: int = 100,
        frequency: TimeFrequency | str | None = None,
        **kwargs: Any,
    ):
        """Initialize seasonality validator.

        Args:
            timestamp_column: Column containing timestamps
            value_column: Column containing values
            expected_period: Expected seasonal period in data points
            min_seasonality_strength: Minimum autocorrelation at period
            max_seasonality_strength: Maximum autocorrelation (for weak seasonality check)
            detect_period: Whether to auto-detect the period
            max_period_search: Maximum lag to search for period
            frequency: Expected data frequency
            **kwargs: Additional config
        """
        super().__init__(
            timestamp_column=timestamp_column,
            value_column=value_column,
            frequency=frequency,
            **kwargs,
        )
        self.expected_period = expected_period
        self.min_seasonality_strength = min_seasonality_strength
        self.max_seasonality_strength = max_seasonality_strength
        self.detect_period = detect_period
        self.max_period_search = max_period_search

    def _find_dominant_period(
        self, acf: np.ndarray, min_lag: int = 2
    ) -> tuple[int | None, float]:
        """Find the dominant period from autocorrelation.

        Args:
            acf: Autocorrelation function values
            min_lag: Minimum lag to consider

        Returns:
            Tuple of (period, strength)
        """
        if len(acf) < min_lag + 1:
            return None, 0.0

        # Find peaks in ACF (local maxima)
        peaks = []
        for i in range(min_lag, len(acf) - 1):
            if acf[i] > acf[i - 1] and acf[i] > acf[i + 1]:
                peaks.append((i, acf[i]))

        if not peaks:
            return None, 0.0

        # Return the strongest peak
        peaks.sort(key=lambda x: x[1], reverse=True)
        return peaks[0]

    def _compute_period_strength(
        self, acf: np.ndarray, period: int
    ) -> float:
        """Compute seasonality strength at a specific period.

        Args:
            acf: Autocorrelation function values
            period: Period to check

        Returns:
            Autocorrelation value at the period
        """
        if period >= len(acf):
            return 0.0
        return float(acf[period])

    def validate_series(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate seasonality in time series.

        Args:
            lf: Input LazyFrame

        Returns:
            List of validation issues
        """
        issues: list[ValidationIssue] = []

        df = self._get_sorted_series(lf)
        if len(df) < 10:  # Need enough data for seasonality
            return issues

        values = df[self.value_column].to_numpy()

        # Handle NaN values
        valid_mask = ~np.isnan(values)
        if not np.all(valid_mask):
            values = values[valid_mask]

        if len(values) < 10:
            return issues

        # Detrend the data
        detrended = self._detrend_linear(values)

        # Compute autocorrelation
        max_lag = min(self.max_period_search, len(values) // 2)
        acf = self._compute_autocorrelation(detrended, max_lag)

        if self.expected_period is not None:
            # Validate expected seasonality
            strength = self._compute_period_strength(acf, self.expected_period)

            if strength < self.min_seasonality_strength:
                issues.append(
                    ValidationIssue(
                        column=self.value_column,
                        issue_type="weak_seasonality",
                        count=1,
                        severity=Severity.MEDIUM,
                        details=(
                            f"Expected seasonality at period {self.expected_period} "
                            f"not found. Strength: {strength:.3f} "
                            f"(minimum: {self.min_seasonality_strength:.3f})"
                        ),
                        expected=(
                            f"Seasonality strength >= {self.min_seasonality_strength} "
                            f"at period {self.expected_period}"
                        ),
                    )
                )

            if (
                self.max_seasonality_strength is not None
                and strength > self.max_seasonality_strength
            ):
                issues.append(
                    ValidationIssue(
                        column=self.value_column,
                        issue_type="excessive_seasonality",
                        count=1,
                        severity=Severity.LOW,
                        details=(
                            f"Seasonality at period {self.expected_period} is "
                            f"stronger than expected. Strength: {strength:.3f} "
                            f"(maximum: {self.max_seasonality_strength:.3f})"
                        ),
                        expected=(
                            f"Seasonality strength <= {self.max_seasonality_strength}"
                        ),
                    )
                )

        elif self.detect_period:
            # Auto-detect and report dominant period
            detected_period, strength = self._find_dominant_period(acf)

            if detected_period is not None and strength >= self.min_seasonality_strength:
                # This is informational, not necessarily an issue
                # But if strength is very high, it might indicate problems
                if strength > 0.8:
                    issues.append(
                        ValidationIssue(
                            column=self.value_column,
                            issue_type="strong_periodicity_detected",
                            count=1,
                            severity=Severity.LOW,
                            details=(
                                f"Strong periodicity detected at period {detected_period} "
                                f"(strength: {strength:.3f}). This may indicate "
                                f"repetitive patterns or data quality issues."
                            ),
                            expected="Moderate seasonality patterns",
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
class SeasonalDecompositionValidator(ValueTimeSeriesValidator):
    """Validates seasonal decomposition components.

    Performs additive decomposition (value = trend + seasonal + residual)
    and validates each component against expected bounds.

    Example:
        validator = SeasonalDecompositionValidator(
            timestamp_column="date",
            value_column="revenue",
            period=12,
            max_residual_ratio=0.2,
        )
    """

    name = "seasonal_decomposition"

    def __init__(
        self,
        timestamp_column: str,
        value_column: str,
        period: int,
        max_residual_ratio: float = 0.3,
        max_seasonal_ratio: float | None = None,
        frequency: TimeFrequency | str | None = None,
        **kwargs: Any,
    ):
        """Initialize decomposition validator.

        Args:
            timestamp_column: Column containing timestamps
            value_column: Column containing values
            period: Seasonal period for decomposition
            max_residual_ratio: Maximum ratio of residual variance to total
            max_seasonal_ratio: Maximum ratio of seasonal variance to total
            frequency: Expected data frequency
            **kwargs: Additional config
        """
        super().__init__(
            timestamp_column=timestamp_column,
            value_column=value_column,
            frequency=frequency,
            **kwargs,
        )
        self.period = period
        self.max_residual_ratio = max_residual_ratio
        self.max_seasonal_ratio = max_seasonal_ratio

    def _decompose(
        self, values: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform simple additive decomposition.

        Args:
            values: Time series values

        Returns:
            Tuple of (trend, seasonal, residual)
        """
        n = len(values)

        # Compute trend using moving average
        if n < self.period * 2:
            trend = np.full(n, np.mean(values))
        else:
            # Centered moving average
            kernel = np.ones(self.period) / self.period
            trend = np.convolve(values, kernel, mode="same")

            # Fix edges
            half = self.period // 2
            trend[:half] = trend[half]
            trend[-half:] = trend[-half - 1]

        # Compute seasonal component
        detrended = values - trend
        seasonal = np.zeros(n)

        for i in range(self.period):
            indices = np.arange(i, n, self.period)
            seasonal[indices] = np.mean(detrended[indices])

        # Compute residual
        residual = values - trend - seasonal

        return trend, seasonal, residual

    def validate_series(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate seasonal decomposition.

        Args:
            lf: Input LazyFrame

        Returns:
            List of validation issues
        """
        issues: list[ValidationIssue] = []

        df = self._get_sorted_series(lf)
        if len(df) < self.period * 2:
            return issues

        values = df[self.value_column].to_numpy()

        # Handle NaN values
        valid_mask = ~np.isnan(values)
        if not np.all(valid_mask):
            values = values[valid_mask]

        if len(values) < self.period * 2:
            return issues

        # Decompose
        trend, seasonal, residual = self._decompose(values)

        total_var = np.var(values)
        if total_var == 0:
            return issues

        residual_var = np.var(residual)
        seasonal_var = np.var(seasonal)

        residual_ratio = residual_var / total_var
        seasonal_ratio = seasonal_var / total_var

        # Check residual variance
        if residual_ratio > self.max_residual_ratio:
            issues.append(
                ValidationIssue(
                    column=self.value_column,
                    issue_type="high_residual_variance",
                    count=1,
                    severity=Severity.MEDIUM,
                    details=(
                        f"Residual variance ratio ({residual_ratio:.3f}) exceeds "
                        f"maximum ({self.max_residual_ratio:.3f}). This may indicate "
                        f"irregular patterns or noise in the data."
                    ),
                    expected=f"Residual variance ratio <= {self.max_residual_ratio}",
                )
            )

        # Check seasonal variance
        if (
            self.max_seasonal_ratio is not None
            and seasonal_ratio > self.max_seasonal_ratio
        ):
            issues.append(
                ValidationIssue(
                    column=self.value_column,
                    issue_type="excessive_seasonal_variance",
                    count=1,
                    severity=Severity.LOW,
                    details=(
                        f"Seasonal variance ratio ({seasonal_ratio:.3f}) exceeds "
                        f"maximum ({self.max_seasonal_ratio:.3f}). Seasonality may be "
                        f"dominating the signal."
                    ),
                    expected=f"Seasonal variance ratio <= {self.max_seasonal_ratio}",
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
