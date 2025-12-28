"""Simple trend analyzer implementation.

Provides basic trend analysis using linear regression.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from truthound.checkpoint.analytics.protocols import (
    TimeSeriesPoint,
    TrendResult,
    TrendDirection,
    AnomalyResult,
    AnomalyType,
    ForecastResult,
    AnalyzerError,
)
from truthound.checkpoint.analytics.analyzers.base import BaseTrendAnalyzer

logger = logging.getLogger(__name__)


class SimpleTrendAnalyzer(BaseTrendAnalyzer):
    """Simple trend analyzer using linear regression.

    Analyzes time series data to detect trends and changes.
    Uses linear regression for trend detection.

    Example:
        >>> analyzer = SimpleTrendAnalyzer()
        >>>
        >>> # Analyze trend
        >>> trend = analyzer.analyze(points)
        >>> print(f"Direction: {trend.direction}")
        >>> print(f"Slope: {trend.slope}")
        >>> print(f"Confidence: {trend.confidence}")
    """

    def __init__(
        self,
        name: str = "simple_trend",
        min_samples: int = 3,
        significance_threshold: float = 0.01,
    ) -> None:
        """Initialize simple trend analyzer.

        Args:
            name: Analyzer name.
            min_samples: Minimum samples for analysis.
            significance_threshold: Threshold for significant trend.
        """
        super().__init__(name=name, min_samples=min_samples)
        self._significance_threshold = significance_threshold

    def analyze(self, points: list[TimeSeriesPoint]) -> TrendResult:
        """Analyze trend in time series.

        Args:
            points: Time series points.

        Returns:
            Trend analysis result.

        Raises:
            AnalyzerError: If analysis fails.
        """
        if len(points) < self._min_samples:
            return TrendResult(
                direction=TrendDirection.STABLE,
                slope=0.0,
                confidence=0.0,
                change_percent=0.0,
                period_start=points[0].timestamp if points else datetime.now(),
                period_end=points[-1].timestamp if points else datetime.now(),
                r_squared=0.0,
                sample_count=len(points),
            )

        try:
            # Sort by timestamp
            sorted_points = sorted(points, key=lambda p: p.timestamp)

            # Convert to x, y values
            x_values, y_values = self._points_to_xy(sorted_points)

            # Perform linear regression
            slope, intercept, r_squared = self._linear_regression(x_values, y_values)

            # Compute statistics
            mean = self._compute_mean(y_values)
            std_dev = self._compute_std_dev(y_values, mean)

            # Determine direction
            direction = self._determine_direction(
                slope, std_dev, self._significance_threshold
            )

            # Handle volatile data
            if std_dev > 0:
                cv = std_dev / mean if mean != 0 else 0
                if cv > 0.5 and r_squared < 0.3:
                    direction = TrendDirection.VOLATILE

            # Compute change
            change_percent = self._compute_change_percent(y_values[0], y_values[-1])

            # Compute confidence
            confidence = self._compute_confidence(
                r_squared, len(points), self._min_samples * 3
            )

            return TrendResult(
                direction=direction,
                slope=slope,
                confidence=confidence,
                change_percent=change_percent,
                period_start=sorted_points[0].timestamp,
                period_end=sorted_points[-1].timestamp,
                r_squared=r_squared,
                sample_count=len(points),
            )

        except Exception as e:
            raise AnalyzerError(
                f"Failed to analyze trend: {e}",
                self.name,
                str(e),
            )

    def detect_anomalies(
        self,
        points: list[TimeSeriesPoint],
        threshold: float = 2.0,
    ) -> list[AnomalyResult]:
        """Detect anomalies using Z-score.

        Args:
            points: Time series points.
            threshold: Z-score threshold for anomaly detection.

        Returns:
            List of detected anomalies.
        """
        if len(points) < self._min_samples:
            return []

        try:
            values = [p.value for p in points]
            mean = self._compute_mean(values)
            std_dev = self._compute_std_dev(values, mean)

            if std_dev == 0:
                return []

            anomalies = []
            for point in points:
                z_score = abs(self._compute_z_score(point.value, mean, std_dev))

                if z_score > threshold:
                    # Determine anomaly type
                    if point.value > mean:
                        anomaly_type = AnomalyType.SPIKE
                    else:
                        anomaly_type = AnomalyType.DROP

                    # Calculate severity based on z-score
                    severity = min(z_score / (threshold * 2), 1.0)

                    anomalies.append(AnomalyResult(
                        timestamp=point.timestamp,
                        anomaly_type=anomaly_type,
                        value=point.value,
                        expected_value=mean,
                        deviation=z_score * std_dev,
                        severity=severity,
                        labels=point.labels,
                    ))

            return anomalies

        except Exception as e:
            logger.warning(f"Anomaly detection failed: {e}")
            return []

    def forecast(
        self,
        points: list[TimeSeriesPoint],
        periods: int,
    ) -> ForecastResult:
        """Forecast future values using linear extrapolation.

        Args:
            points: Historical time series points.
            periods: Number of periods to forecast.

        Returns:
            Forecast result.
        """
        if len(points) < self._min_samples:
            return ForecastResult(
                predictions=[],
                confidence_lower=[],
                confidence_upper=[],
                method="linear",
                horizon=periods,
            )

        try:
            sorted_points = sorted(points, key=lambda p: p.timestamp)
            x_values, y_values = self._points_to_xy(sorted_points)

            # Fit linear model
            slope, intercept, r_squared = self._linear_regression(x_values, y_values)

            # Estimate interval between points
            if len(sorted_points) > 1:
                total_time = (sorted_points[-1].timestamp - sorted_points[0].timestamp).total_seconds()
                avg_interval = total_time / (len(sorted_points) - 1)
            else:
                avg_interval = 3600  # Default to 1 hour

            # Compute prediction standard error
            y_pred = [slope * x + intercept for x in x_values]
            residuals = [y - yp for y, yp in zip(y_values, y_pred)]
            std_error = self._compute_std_dev(residuals)

            # Generate predictions
            last_x = x_values[-1]
            last_time = sorted_points[-1].timestamp

            predictions = []
            confidence_lower = []
            confidence_upper = []

            for i in range(1, periods + 1):
                future_x = last_x + avg_interval * i
                future_time = last_time + timedelta(seconds=avg_interval * i)
                predicted_value = slope * future_x + intercept

                # Confidence interval widens with distance
                interval_width = std_error * 1.96 * (1 + 0.1 * i)

                predictions.append(TimeSeriesPoint(
                    timestamp=future_time,
                    value=predicted_value,
                    metadata={"forecast": True, "period": i},
                ))
                confidence_lower.append(predicted_value - interval_width)
                confidence_upper.append(predicted_value + interval_width)

            return ForecastResult(
                predictions=predictions,
                confidence_lower=confidence_lower,
                confidence_upper=confidence_upper,
                method="linear",
                horizon=periods,
            )

        except Exception as e:
            logger.warning(f"Forecasting failed: {e}")
            return ForecastResult(
                predictions=[],
                confidence_lower=[],
                confidence_upper=[],
                method="linear",
                horizon=periods,
            )

    def compare_periods(
        self,
        current: list[TimeSeriesPoint],
        previous: list[TimeSeriesPoint],
    ) -> dict[str, Any]:
        """Compare two time periods.

        Args:
            current: Current period points.
            previous: Previous period points.

        Returns:
            Comparison results.
        """
        current_values = [p.value for p in current]
        previous_values = [p.value for p in previous]

        current_mean = self._compute_mean(current_values)
        previous_mean = self._compute_mean(previous_values)

        current_std = self._compute_std_dev(current_values)
        previous_std = self._compute_std_dev(previous_values)

        mean_change = current_mean - previous_mean
        mean_change_percent = (
            (mean_change / previous_mean * 100) if previous_mean != 0 else 0
        )

        return {
            "current_mean": current_mean,
            "previous_mean": previous_mean,
            "mean_change": mean_change,
            "mean_change_percent": mean_change_percent,
            "current_std_dev": current_std,
            "previous_std_dev": previous_std,
            "current_count": len(current),
            "previous_count": len(previous),
            "direction": (
                "improved" if mean_change < 0 else
                "degraded" if mean_change > 0 else
                "stable"
            ),
        }
