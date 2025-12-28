"""Base trend analyzer implementation.

Provides abstract base class with common analysis functionality.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from truthound.checkpoint.analytics.protocols import (
    TrendAnalyzerProtocol,
    TimeSeriesPoint,
    TrendResult,
    TrendDirection,
    AnomalyResult,
    AnomalyType,
    ForecastResult,
    AnalyzerError,
)

logger = logging.getLogger(__name__)


class BaseTrendAnalyzer(ABC):
    """Abstract base class for trend analyzers.

    Provides common functionality including:
    - Statistical calculations
    - Linear regression
    - Percentile computation

    Subclasses must implement the analysis methods.
    """

    def __init__(
        self,
        name: str | None = None,
        min_samples: int = 3,
    ) -> None:
        """Initialize analyzer.

        Args:
            name: Analyzer name.
            min_samples: Minimum samples required for analysis.
        """
        self._name = name or self.__class__.__name__
        self._min_samples = min_samples

    @property
    def name(self) -> str:
        """Get analyzer name."""
        return self._name

    @abstractmethod
    def analyze(self, points: list[TimeSeriesPoint]) -> TrendResult:
        """Analyze trend in time series."""
        pass

    @abstractmethod
    def detect_anomalies(
        self,
        points: list[TimeSeriesPoint],
        threshold: float = 2.0,
    ) -> list[AnomalyResult]:
        """Detect anomalies in time series."""
        pass

    @abstractmethod
    def forecast(
        self,
        points: list[TimeSeriesPoint],
        periods: int,
    ) -> ForecastResult:
        """Forecast future values."""
        pass

    # Statistical helper methods

    def _compute_mean(self, values: list[float]) -> float:
        """Compute mean of values."""
        if not values:
            return 0.0
        return sum(values) / len(values)

    def _compute_std_dev(self, values: list[float], mean: float | None = None) -> float:
        """Compute standard deviation."""
        if len(values) < 2:
            return 0.0

        if mean is None:
            mean = self._compute_mean(values)

        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return math.sqrt(variance)

    def _compute_z_score(self, value: float, mean: float, std_dev: float) -> float:
        """Compute Z-score for a value."""
        if std_dev == 0:
            return 0.0
        return (value - mean) / std_dev

    def _compute_percentile(self, values: list[float], percentile: float) -> float:
        """Compute percentile of values.

        Args:
            values: Values to analyze.
            percentile: Percentile (0-100).

        Returns:
            Percentile value.
        """
        if not values:
            return 0.0

        sorted_values = sorted(values)
        n = len(sorted_values)
        k = (n - 1) * (percentile / 100)
        f = math.floor(k)
        c = math.ceil(k)

        if f == c:
            return sorted_values[int(k)]

        return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)

    def _compute_percentiles(
        self,
        values: list[float],
        percentiles: list[float] = [50, 90, 95, 99],
    ) -> dict[str, float]:
        """Compute multiple percentiles."""
        return {
            f"p{int(p)}": self._compute_percentile(values, p)
            for p in percentiles
        }

    def _linear_regression(
        self,
        x_values: list[float],
        y_values: list[float],
    ) -> tuple[float, float, float]:
        """Perform simple linear regression.

        Args:
            x_values: Independent variable values.
            y_values: Dependent variable values.

        Returns:
            Tuple of (slope, intercept, r_squared).
        """
        n = len(x_values)
        if n < 2:
            return 0.0, 0.0, 0.0

        x_mean = sum(x_values) / n
        y_mean = sum(y_values) / n

        # Calculate slope
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)

        if denominator == 0:
            return 0.0, y_mean, 0.0

        slope = numerator / denominator
        intercept = y_mean - slope * x_mean

        # Calculate R-squared
        y_pred = [slope * x + intercept for x in x_values]
        ss_res = sum((y - yp) ** 2 for y, yp in zip(y_values, y_pred))
        ss_tot = sum((y - y_mean) ** 2 for y in y_values)

        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        return slope, intercept, r_squared

    def _points_to_xy(
        self,
        points: list[TimeSeriesPoint],
    ) -> tuple[list[float], list[float]]:
        """Convert points to x, y arrays for regression.

        Uses timestamps as x values (seconds since first point).

        Args:
            points: Time series points.

        Returns:
            Tuple of (x_values, y_values).
        """
        if not points:
            return [], []

        sorted_points = sorted(points, key=lambda p: p.timestamp)
        base_time = sorted_points[0].timestamp

        x_values = [
            (p.timestamp - base_time).total_seconds()
            for p in sorted_points
        ]
        y_values = [p.value for p in sorted_points]

        return x_values, y_values

    def _determine_direction(
        self,
        slope: float,
        std_dev: float,
        threshold: float = 0.01,
    ) -> TrendDirection:
        """Determine trend direction from slope.

        Args:
            slope: Linear regression slope.
            std_dev: Standard deviation of values.
            threshold: Slope threshold relative to std_dev.

        Returns:
            Trend direction.
        """
        if std_dev == 0:
            return TrendDirection.STABLE

        # Normalize slope by standard deviation
        normalized_slope = abs(slope) / std_dev if std_dev > 0 else 0

        if normalized_slope < threshold:
            return TrendDirection.STABLE
        elif slope > 0:
            return TrendDirection.INCREASING
        else:
            return TrendDirection.DECREASING

    def _compute_change_percent(
        self,
        first_value: float,
        last_value: float,
    ) -> float:
        """Compute percentage change between first and last value."""
        if first_value == 0:
            return 0.0 if last_value == 0 else float("inf")
        return ((last_value - first_value) / abs(first_value)) * 100

    def _compute_confidence(
        self,
        r_squared: float,
        sample_count: int,
        min_samples: int = 10,
    ) -> float:
        """Compute confidence score for trend analysis.

        Args:
            r_squared: R-squared value from regression.
            sample_count: Number of samples.
            min_samples: Minimum samples for full confidence.

        Returns:
            Confidence score (0.0 to 1.0).
        """
        # Penalize for too few samples
        sample_factor = min(sample_count / min_samples, 1.0)

        # Combine R-squared with sample factor
        confidence = r_squared * sample_factor

        return max(0.0, min(1.0, confidence))
