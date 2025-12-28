"""Anomaly detection analyzer.

Provides multiple anomaly detection methods including Z-score,
IQR, and moving average deviation.
"""

from __future__ import annotations

import logging
from collections import deque
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


class AnomalyDetector(BaseTrendAnalyzer):
    """Anomaly detector with multiple detection methods.

    Supports:
    - Z-score based detection
    - IQR (Interquartile Range) based detection
    - Moving average deviation detection
    - Seasonal adjustment

    Example:
        >>> detector = AnomalyDetector(method="zscore")
        >>>
        >>> # Detect anomalies
        >>> anomalies = detector.detect_anomalies(points, threshold=2.5)
        >>> for a in anomalies:
        ...     print(f"{a.timestamp}: {a.anomaly_type} (severity: {a.severity})")
    """

    METHODS = ["zscore", "iqr", "moving_avg", "mad"]

    def __init__(
        self,
        name: str = "anomaly_detector",
        method: str = "zscore",
        min_samples: int = 10,
        window_size: int = 20,
    ) -> None:
        """Initialize anomaly detector.

        Args:
            name: Analyzer name.
            method: Detection method ("zscore", "iqr", "moving_avg", "mad").
            min_samples: Minimum samples for detection.
            window_size: Window size for moving average method.
        """
        super().__init__(name=name, min_samples=min_samples)
        self._method = method.lower()
        self._window_size = window_size

        if self._method not in self.METHODS:
            raise ValueError(f"Unknown method: {method}. Valid methods: {self.METHODS}")

    def analyze(self, points: list[TimeSeriesPoint]) -> TrendResult:
        """Analyze trend (delegates to parent)."""
        # Use simple linear regression for trend analysis
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
        """Detect anomalies using configured method.

        Args:
            points: Time series points.
            threshold: Detection threshold.

        Returns:
            List of detected anomalies.
        """
        if len(points) < self._min_samples:
            return []

        if self._method == "zscore":
            return self._detect_zscore(points, threshold)
        elif self._method == "iqr":
            return self._detect_iqr(points, threshold)
        elif self._method == "moving_avg":
            return self._detect_moving_avg(points, threshold)
        elif self._method == "mad":
            return self._detect_mad(points, threshold)
        else:
            return self._detect_zscore(points, threshold)

    def _detect_zscore(
        self,
        points: list[TimeSeriesPoint],
        threshold: float,
    ) -> list[AnomalyResult]:
        """Detect anomalies using Z-score method."""
        values = [p.value for p in points]
        mean = self._compute_mean(values)
        std_dev = self._compute_std_dev(values, mean)

        if std_dev == 0:
            return []

        anomalies = []
        for point in points:
            z_score = self._compute_z_score(point.value, mean, std_dev)

            if abs(z_score) > threshold:
                anomaly_type = AnomalyType.SPIKE if z_score > 0 else AnomalyType.DROP
                severity = min(abs(z_score) / (threshold * 2), 1.0)

                anomalies.append(AnomalyResult(
                    timestamp=point.timestamp,
                    anomaly_type=anomaly_type,
                    value=point.value,
                    expected_value=mean,
                    deviation=z_score,
                    severity=severity,
                    labels=point.labels,
                ))

        return anomalies

    def _detect_iqr(
        self,
        points: list[TimeSeriesPoint],
        threshold: float = 1.5,
    ) -> list[AnomalyResult]:
        """Detect anomalies using IQR method."""
        values = [p.value for p in points]
        q1 = self._compute_percentile(values, 25)
        q3 = self._compute_percentile(values, 75)
        iqr = q3 - q1

        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        median = self._compute_percentile(values, 50)

        anomalies = []
        for point in points:
            if point.value < lower_bound or point.value > upper_bound:
                if point.value > upper_bound:
                    anomaly_type = AnomalyType.SPIKE
                    deviation = (point.value - upper_bound) / iqr if iqr > 0 else 0
                else:
                    anomaly_type = AnomalyType.DROP
                    deviation = (lower_bound - point.value) / iqr if iqr > 0 else 0

                severity = min(abs(deviation) / threshold, 1.0)

                anomalies.append(AnomalyResult(
                    timestamp=point.timestamp,
                    anomaly_type=anomaly_type,
                    value=point.value,
                    expected_value=median,
                    deviation=deviation,
                    severity=severity,
                    labels=point.labels,
                ))

        return anomalies

    def _detect_moving_avg(
        self,
        points: list[TimeSeriesPoint],
        threshold: float,
    ) -> list[AnomalyResult]:
        """Detect anomalies using moving average deviation."""
        sorted_points = sorted(points, key=lambda p: p.timestamp)

        if len(sorted_points) < self._window_size:
            return self._detect_zscore(sorted_points, threshold)

        anomalies = []
        window: deque[float] = deque(maxlen=self._window_size)

        for i, point in enumerate(sorted_points):
            if len(window) >= self._min_samples:
                window_mean = sum(window) / len(window)
                window_std = self._compute_std_dev(list(window), window_mean)

                if window_std > 0:
                    deviation = (point.value - window_mean) / window_std

                    if abs(deviation) > threshold:
                        anomaly_type = AnomalyType.SPIKE if deviation > 0 else AnomalyType.DROP
                        severity = min(abs(deviation) / (threshold * 2), 1.0)

                        anomalies.append(AnomalyResult(
                            timestamp=point.timestamp,
                            anomaly_type=anomaly_type,
                            value=point.value,
                            expected_value=window_mean,
                            deviation=deviation,
                            severity=severity,
                            labels=point.labels,
                        ))

            window.append(point.value)

        return anomalies

    def _detect_mad(
        self,
        points: list[TimeSeriesPoint],
        threshold: float = 3.0,
    ) -> list[AnomalyResult]:
        """Detect anomalies using Median Absolute Deviation (MAD)."""
        values = [p.value for p in points]
        median = self._compute_percentile(values, 50)

        # Calculate MAD
        absolute_deviations = [abs(v - median) for v in values]
        mad = self._compute_percentile(absolute_deviations, 50)

        if mad == 0:
            return []

        # Modified Z-score using MAD
        k = 1.4826  # Constant for normal distribution

        anomalies = []
        for point in points:
            modified_z = (point.value - median) / (k * mad)

            if abs(modified_z) > threshold:
                anomaly_type = AnomalyType.SPIKE if modified_z > 0 else AnomalyType.DROP
                severity = min(abs(modified_z) / (threshold * 2), 1.0)

                anomalies.append(AnomalyResult(
                    timestamp=point.timestamp,
                    anomaly_type=anomaly_type,
                    value=point.value,
                    expected_value=median,
                    deviation=modified_z,
                    severity=severity,
                    labels=point.labels,
                ))

        return anomalies

    def detect_level_shift(
        self,
        points: list[TimeSeriesPoint],
        window_size: int | None = None,
    ) -> list[AnomalyResult]:
        """Detect level shifts (sudden persistent changes).

        Args:
            points: Time series points.
            window_size: Window size for comparison.

        Returns:
            List of level shift anomalies.
        """
        window_size = window_size or self._window_size
        sorted_points = sorted(points, key=lambda p: p.timestamp)

        if len(sorted_points) < window_size * 2:
            return []

        anomalies = []

        for i in range(window_size, len(sorted_points) - window_size):
            before = [p.value for p in sorted_points[i - window_size:i]]
            after = [p.value for p in sorted_points[i:i + window_size]]

            before_mean = self._compute_mean(before)
            after_mean = self._compute_mean(after)
            before_std = self._compute_std_dev(before)

            # Detect significant shift
            if before_std > 0:
                shift_magnitude = abs(after_mean - before_mean) / before_std

                if shift_magnitude > 2.0:
                    anomalies.append(AnomalyResult(
                        timestamp=sorted_points[i].timestamp,
                        anomaly_type=AnomalyType.SHIFT,
                        value=sorted_points[i].value,
                        expected_value=before_mean,
                        deviation=shift_magnitude,
                        severity=min(shift_magnitude / 4.0, 1.0),
                        labels={"before_mean": str(before_mean), "after_mean": str(after_mean)},
                    ))

        return anomalies

    def forecast(
        self,
        points: list[TimeSeriesPoint],
        periods: int,
    ) -> ForecastResult:
        """Basic forecast (uses linear extrapolation)."""
        # Simplified forecast - primarily an anomaly detector
        if len(points) < self._min_samples:
            return ForecastResult(
                predictions=[],
                confidence_lower=[],
                confidence_upper=[],
                method="linear",
                horizon=periods,
            )

        sorted_points = sorted(points, key=lambda p: p.timestamp)
        x_values, y_values = self._points_to_xy(sorted_points)
        slope, intercept, _ = self._linear_regression(x_values, y_values)

        if len(sorted_points) > 1:
            total_time = (sorted_points[-1].timestamp - sorted_points[0].timestamp).total_seconds()
            avg_interval = total_time / (len(sorted_points) - 1)
        else:
            avg_interval = 3600

        std_dev = self._compute_std_dev(y_values)
        last_x = x_values[-1]
        last_time = sorted_points[-1].timestamp

        predictions = []
        confidence_lower = []
        confidence_upper = []

        for i in range(1, periods + 1):
            future_x = last_x + avg_interval * i
            future_time = last_time + timedelta(seconds=avg_interval * i)
            predicted = slope * future_x + intercept

            interval = std_dev * 1.96 * (1 + 0.1 * i)

            predictions.append(TimeSeriesPoint(
                timestamp=future_time,
                value=predicted,
                metadata={"forecast": True},
            ))
            confidence_lower.append(predicted - interval)
            confidence_upper.append(predicted + interval)

        return ForecastResult(
            predictions=predictions,
            confidence_lower=confidence_lower,
            confidence_upper=confidence_upper,
            method="linear",
            horizon=periods,
        )
