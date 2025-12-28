"""Simple forecaster implementation.

Provides basic time series forecasting using moving averages
and exponential smoothing.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta
from typing import Any

from truthound.checkpoint.analytics.protocols import (
    TimeSeriesPoint,
    TrendResult,
    TrendDirection,
    AnomalyResult,
    ForecastResult,
    AnalyzerError,
)
from truthound.checkpoint.analytics.analyzers.base import BaseTrendAnalyzer

logger = logging.getLogger(__name__)


class SimpleForecaster(BaseTrendAnalyzer):
    """Simple time series forecaster.

    Supports multiple forecasting methods:
    - Linear extrapolation
    - Moving average
    - Simple exponential smoothing
    - Double exponential smoothing (Holt's method)

    Example:
        >>> forecaster = SimpleForecaster(method="exponential")
        >>>
        >>> # Forecast next 5 periods
        >>> forecast = forecaster.forecast(points, periods=5)
        >>> for p in forecast.predictions:
        ...     print(f"{p.timestamp}: {p.value}")
    """

    METHODS = ["linear", "moving_avg", "exponential", "holt"]

    def __init__(
        self,
        name: str = "simple_forecaster",
        method: str = "exponential",
        min_samples: int = 5,
        alpha: float = 0.3,
        beta: float = 0.1,
        window_size: int = 5,
    ) -> None:
        """Initialize forecaster.

        Args:
            name: Forecaster name.
            method: Forecasting method.
            min_samples: Minimum samples required.
            alpha: Smoothing parameter for exponential methods.
            beta: Trend smoothing parameter for Holt's method.
            window_size: Window size for moving average.
        """
        super().__init__(name=name, min_samples=min_samples)
        self._method = method.lower()
        self._alpha = alpha
        self._beta = beta
        self._window_size = window_size

        if self._method not in self.METHODS:
            raise ValueError(f"Unknown method: {method}. Valid methods: {self.METHODS}")

    def analyze(self, points: list[TimeSeriesPoint]) -> TrendResult:
        """Analyze trend using linear regression."""
        if len(points) < self._min_samples:
            return TrendResult(
                direction=TrendDirection.STABLE,
                slope=0.0,
                confidence=0.0,
                change_percent=0.0,
                period_start=points[0].timestamp if points else datetime.now(),
                period_end=points[-1].timestamp if points else datetime.now(),
                sample_count=len(points),
            )

        sorted_points = sorted(points, key=lambda p: p.timestamp)
        x_values, y_values = self._points_to_xy(sorted_points)
        slope, intercept, r_squared = self._linear_regression(x_values, y_values)

        mean = self._compute_mean(y_values)
        std_dev = self._compute_std_dev(y_values, mean)
        direction = self._determine_direction(slope, std_dev)
        change_percent = self._compute_change_percent(y_values[0], y_values[-1])
        confidence = self._compute_confidence(r_squared, len(points))

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

    def detect_anomalies(
        self,
        points: list[TimeSeriesPoint],
        threshold: float = 2.0,
    ) -> list[AnomalyResult]:
        """Detect anomalies based on forecast error."""
        # Use one-step-ahead forecast errors for anomaly detection
        if len(points) < self._min_samples + 1:
            return []

        sorted_points = sorted(points, key=lambda p: p.timestamp)
        anomalies = []

        # Compute forecast errors
        errors = []
        for i in range(self._min_samples, len(sorted_points)):
            history = sorted_points[:i]
            actual = sorted_points[i]

            forecast = self.forecast(history, periods=1)
            if forecast.predictions:
                predicted = forecast.predictions[0].value
                error = actual.value - predicted
                errors.append((actual, predicted, error))

        if not errors:
            return []

        # Compute error statistics
        error_values = [e[2] for e in errors]
        mean_error = self._compute_mean(error_values)
        std_error = self._compute_std_dev(error_values, mean_error)

        if std_error == 0:
            return []

        from truthound.checkpoint.analytics.protocols import AnomalyType

        for actual, predicted, error in errors:
            z_score = (error - mean_error) / std_error

            if abs(z_score) > threshold:
                anomaly_type = AnomalyType.SPIKE if z_score > 0 else AnomalyType.DROP
                severity = min(abs(z_score) / (threshold * 2), 1.0)

                anomalies.append(AnomalyResult(
                    timestamp=actual.timestamp,
                    anomaly_type=anomaly_type,
                    value=actual.value,
                    expected_value=predicted,
                    deviation=z_score,
                    severity=severity,
                    labels=actual.labels,
                ))

        return anomalies

    def forecast(
        self,
        points: list[TimeSeriesPoint],
        periods: int,
    ) -> ForecastResult:
        """Generate forecast using configured method.

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
                method=self._method,
                horizon=periods,
            )

        if self._method == "linear":
            return self._forecast_linear(points, periods)
        elif self._method == "moving_avg":
            return self._forecast_moving_avg(points, periods)
        elif self._method == "exponential":
            return self._forecast_exponential(points, periods)
        elif self._method == "holt":
            return self._forecast_holt(points, periods)
        else:
            return self._forecast_linear(points, periods)

    def _forecast_linear(
        self,
        points: list[TimeSeriesPoint],
        periods: int,
    ) -> ForecastResult:
        """Forecast using linear extrapolation."""
        sorted_points = sorted(points, key=lambda p: p.timestamp)
        x_values, y_values = self._points_to_xy(sorted_points)
        slope, intercept, _ = self._linear_regression(x_values, y_values)

        avg_interval = self._estimate_interval(sorted_points)
        std_dev = self._compute_std_dev(y_values)

        last_x = x_values[-1]
        last_time = sorted_points[-1].timestamp

        predictions, lower, upper = self._generate_predictions(
            last_x, last_time, periods, avg_interval,
            lambda x: slope * x + intercept,
            std_dev,
        )

        return ForecastResult(
            predictions=predictions,
            confidence_lower=lower,
            confidence_upper=upper,
            method="linear",
            horizon=periods,
        )

    def _forecast_moving_avg(
        self,
        points: list[TimeSeriesPoint],
        periods: int,
    ) -> ForecastResult:
        """Forecast using simple moving average."""
        sorted_points = sorted(points, key=lambda p: p.timestamp)
        values = [p.value for p in sorted_points]

        # Use last window_size values for forecast
        window = values[-self._window_size:]
        forecast_value = sum(window) / len(window)
        std_dev = self._compute_std_dev(window)

        avg_interval = self._estimate_interval(sorted_points)
        last_time = sorted_points[-1].timestamp

        predictions = []
        lower = []
        upper = []

        for i in range(1, periods + 1):
            future_time = last_time + timedelta(seconds=avg_interval * i)
            interval = std_dev * 1.96

            predictions.append(TimeSeriesPoint(
                timestamp=future_time,
                value=forecast_value,
                metadata={"forecast": True, "method": "moving_avg"},
            ))
            lower.append(forecast_value - interval)
            upper.append(forecast_value + interval)

        return ForecastResult(
            predictions=predictions,
            confidence_lower=lower,
            confidence_upper=upper,
            method="moving_avg",
            horizon=periods,
        )

    def _forecast_exponential(
        self,
        points: list[TimeSeriesPoint],
        periods: int,
    ) -> ForecastResult:
        """Forecast using simple exponential smoothing."""
        sorted_points = sorted(points, key=lambda p: p.timestamp)
        values = [p.value for p in sorted_points]

        # Apply exponential smoothing
        level = values[0]
        for value in values[1:]:
            level = self._alpha * value + (1 - self._alpha) * level

        # Estimate forecast error
        errors = []
        temp_level = values[0]
        for value in values[1:]:
            errors.append(value - temp_level)
            temp_level = self._alpha * value + (1 - self._alpha) * temp_level

        std_error = self._compute_std_dev(errors) if errors else 0

        avg_interval = self._estimate_interval(sorted_points)
        last_time = sorted_points[-1].timestamp

        predictions = []
        lower = []
        upper = []

        for i in range(1, periods + 1):
            future_time = last_time + timedelta(seconds=avg_interval * i)
            # Error grows with forecast horizon
            interval = std_error * 1.96 * math.sqrt(i)

            predictions.append(TimeSeriesPoint(
                timestamp=future_time,
                value=level,
                metadata={"forecast": True, "method": "exponential"},
            ))
            lower.append(level - interval)
            upper.append(level + interval)

        return ForecastResult(
            predictions=predictions,
            confidence_lower=lower,
            confidence_upper=upper,
            method="exponential",
            horizon=periods,
        )

    def _forecast_holt(
        self,
        points: list[TimeSeriesPoint],
        periods: int,
    ) -> ForecastResult:
        """Forecast using Holt's linear method (double exponential smoothing)."""
        sorted_points = sorted(points, key=lambda p: p.timestamp)
        values = [p.value for p in sorted_points]

        if len(values) < 2:
            return self._forecast_exponential(points, periods)

        # Initialize level and trend
        level = values[0]
        trend = values[1] - values[0]

        # Apply double exponential smoothing
        for value in values[1:]:
            last_level = level
            level = self._alpha * value + (1 - self._alpha) * (level + trend)
            trend = self._beta * (level - last_level) + (1 - self._beta) * trend

        # Estimate forecast error
        errors = []
        temp_level = values[0]
        temp_trend = values[1] - values[0]
        for i, value in enumerate(values[1:]):
            forecast = temp_level + temp_trend
            errors.append(value - forecast)
            last_level = temp_level
            temp_level = self._alpha * value + (1 - self._alpha) * (temp_level + temp_trend)
            temp_trend = self._beta * (temp_level - last_level) + (1 - self._beta) * temp_trend

        std_error = self._compute_std_dev(errors) if errors else 0

        avg_interval = self._estimate_interval(sorted_points)
        last_time = sorted_points[-1].timestamp

        predictions = []
        lower = []
        upper = []

        for i in range(1, periods + 1):
            future_time = last_time + timedelta(seconds=avg_interval * i)
            predicted_value = level + trend * i
            interval = std_error * 1.96 * math.sqrt(i)

            predictions.append(TimeSeriesPoint(
                timestamp=future_time,
                value=predicted_value,
                metadata={"forecast": True, "method": "holt"},
            ))
            lower.append(predicted_value - interval)
            upper.append(predicted_value + interval)

        return ForecastResult(
            predictions=predictions,
            confidence_lower=lower,
            confidence_upper=upper,
            method="holt",
            horizon=periods,
        )

    def _estimate_interval(self, points: list[TimeSeriesPoint]) -> float:
        """Estimate average interval between points."""
        if len(points) < 2:
            return 3600.0  # Default 1 hour

        total_time = (points[-1].timestamp - points[0].timestamp).total_seconds()
        return total_time / (len(points) - 1)

    def _generate_predictions(
        self,
        last_x: float,
        last_time: datetime,
        periods: int,
        interval: float,
        forecast_fn,
        std_dev: float,
    ) -> tuple[list[TimeSeriesPoint], list[float], list[float]]:
        """Generate prediction points with confidence intervals."""
        predictions = []
        lower = []
        upper = []

        for i in range(1, periods + 1):
            future_x = last_x + interval * i
            future_time = last_time + timedelta(seconds=interval * i)
            predicted = forecast_fn(future_x)
            conf_interval = std_dev * 1.96 * (1 + 0.1 * i)

            predictions.append(TimeSeriesPoint(
                timestamp=future_time,
                value=predicted,
                metadata={"forecast": True},
            ))
            lower.append(predicted - conf_interval)
            upper.append(predicted + conf_interval)

        return predictions, lower, upper
