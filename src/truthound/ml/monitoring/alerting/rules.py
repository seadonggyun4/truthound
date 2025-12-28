"""Alert rules for model monitoring.

Provides various rule types for detecting issues:
- Threshold: Simple metric thresholds
- Anomaly: Statistical anomaly detection
- Trend: Trend-based alerting
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable
import uuid

from truthound.ml.monitoring.protocols import (
    IAlertRule,
    AlertSeverity,
    ModelMetrics,
    Alert,
)


class AlertRule(ABC, IAlertRule):
    """Base class for alert rules."""

    def __init__(
        self,
        name: str,
        severity: AlertSeverity = AlertSeverity.WARNING,
        description: str = "",
    ):
        self._name = name
        self._severity = severity
        self._description = description

    @property
    def name(self) -> str:
        return self._name

    @property
    def severity(self) -> AlertSeverity:
        return self._severity

    @property
    def description(self) -> str:
        return self._description

    @abstractmethod
    def evaluate(self, metrics: ModelMetrics) -> bool:
        """Evaluate rule against metrics."""
        ...

    @abstractmethod
    def get_message(self, metrics: ModelMetrics) -> str:
        """Get alert message."""
        ...

    def create_alert(self, metrics: ModelMetrics) -> Alert:
        """Create an alert from metrics.

        Args:
            metrics: Metrics that triggered the rule

        Returns:
            Alert object
        """
        return Alert(
            alert_id=str(uuid.uuid4()),
            model_id=metrics.model_id,
            rule_name=self._name,
            severity=self._severity,
            message=self.get_message(metrics),
            metrics=metrics,
        )


@dataclass
class ThresholdConfig:
    """Configuration for threshold rule.

    Attributes:
        metric_name: Name of metric to check
        threshold: Threshold value
        comparison: Comparison operator (gt, lt, gte, lte, eq)
        for_duration_seconds: Must be above threshold for this long
    """

    metric_name: str
    threshold: float
    comparison: str = "gt"  # gt, lt, gte, lte, eq
    for_duration_seconds: int = 0


class ThresholdRule(AlertRule):
    """Simple threshold-based alert rule.

    Triggers when a metric crosses a threshold.

    Example:
        >>> rule = ThresholdRule(
        ...     name="high_latency",
        ...     config=ThresholdConfig(
        ...         metric_name="latency_ms",
        ...         threshold=100,
        ...         comparison="gt",
        ...     ),
        ...     severity=AlertSeverity.WARNING,
        ... )
    """

    _comparisons: dict[str, Callable[[float, float], bool]] = {
        "gt": lambda a, b: a > b,
        "lt": lambda a, b: a < b,
        "gte": lambda a, b: a >= b,
        "lte": lambda a, b: a <= b,
        "eq": lambda a, b: a == b,
    }

    def __init__(
        self,
        name: str,
        config: ThresholdConfig,
        severity: AlertSeverity = AlertSeverity.WARNING,
    ):
        super().__init__(name, severity)
        self._config = config
        self._first_exceeded: dict[str, datetime] = {}

    def evaluate(self, metrics: ModelMetrics) -> bool:
        """Check if metric exceeds threshold.

        Args:
            metrics: Metrics to check

        Returns:
            True if threshold exceeded
        """
        value = self._get_metric_value(metrics)
        if value is None:
            return False

        comparator = self._comparisons.get(self._config.comparison)
        if not comparator:
            return False

        is_exceeded = comparator(value, self._config.threshold)

        # Handle duration requirement
        if self._config.for_duration_seconds > 0:
            model_id = metrics.model_id
            now = datetime.now(timezone.utc)

            if is_exceeded:
                if model_id not in self._first_exceeded:
                    self._first_exceeded[model_id] = now
                    return False

                elapsed = (now - self._first_exceeded[model_id]).total_seconds()
                return elapsed >= self._config.for_duration_seconds
            else:
                self._first_exceeded.pop(model_id, None)
                return False

        return is_exceeded

    def _get_metric_value(self, metrics: ModelMetrics) -> float | None:
        """Get metric value by name."""
        name = self._config.metric_name

        # Check standard metrics
        if name == "latency_ms":
            return metrics.latency_ms
        elif name == "throughput_rps":
            return metrics.throughput_rps
        elif name == "prediction_drift":
            return metrics.prediction_drift
        elif name == "accuracy":
            return metrics.accuracy
        elif name == "precision":
            return metrics.precision
        elif name == "recall":
            return metrics.recall
        elif name == "f1_score":
            return metrics.f1_score

        # Check feature drift
        if name.startswith("drift_"):
            feature = name[6:]
            return metrics.feature_drift.get(feature)

        # Check custom metrics
        return metrics.custom_metrics.get(name)

    def get_message(self, metrics: ModelMetrics) -> str:
        """Get alert message."""
        value = self._get_metric_value(metrics)
        return (
            f"Model {metrics.model_id}: {self._config.metric_name} "
            f"is {value:.2f}, which is {self._config.comparison} "
            f"threshold {self._config.threshold}"
        )


class AnomalyRule(AlertRule):
    """Statistical anomaly detection rule.

    Uses z-score to detect anomalies in metrics.

    Example:
        >>> rule = AnomalyRule(
        ...     name="latency_anomaly",
        ...     metric_name="latency_ms",
        ...     z_threshold=3.0,
        ...     window_size=100,
        ... )
    """

    def __init__(
        self,
        name: str,
        metric_name: str,
        z_threshold: float = 3.0,
        window_size: int = 100,
        severity: AlertSeverity = AlertSeverity.WARNING,
    ):
        super().__init__(name, severity)
        self._metric_name = metric_name
        self._z_threshold = z_threshold
        self._window_size = window_size
        self._history: dict[str, list[float]] = {}

    def evaluate(self, metrics: ModelMetrics) -> bool:
        """Check if metric is anomalous.

        Args:
            metrics: Metrics to check

        Returns:
            True if anomaly detected
        """
        value = self._get_metric_value(metrics)
        if value is None:
            return False

        model_id = metrics.model_id

        # Update history
        if model_id not in self._history:
            self._history[model_id] = []

        self._history[model_id].append(value)

        # Keep window size
        if len(self._history[model_id]) > self._window_size:
            self._history[model_id].pop(0)

        # Need enough data
        if len(self._history[model_id]) < 10:
            return False

        # Calculate z-score
        history = self._history[model_id]
        mean = sum(history) / len(history)
        variance = sum((x - mean) ** 2 for x in history) / len(history)
        std = variance ** 0.5 if variance > 0 else 1.0

        z_score = abs(value - mean) / std if std > 0 else 0

        return z_score > self._z_threshold

    def _get_metric_value(self, metrics: ModelMetrics) -> float | None:
        """Get metric value by name."""
        # Same logic as ThresholdRule
        name = self._metric_name

        if name == "latency_ms":
            return metrics.latency_ms
        elif name == "throughput_rps":
            return metrics.throughput_rps
        elif name == "prediction_drift":
            return metrics.prediction_drift
        elif name == "accuracy":
            return metrics.accuracy
        elif name == "precision":
            return metrics.precision
        elif name == "recall":
            return metrics.recall
        elif name == "f1_score":
            return metrics.f1_score

        return metrics.custom_metrics.get(name)

    def get_message(self, metrics: ModelMetrics) -> str:
        """Get alert message."""
        value = self._get_metric_value(metrics)
        history = self._history.get(metrics.model_id, [])
        mean = sum(history) / len(history) if history else 0

        return (
            f"Model {metrics.model_id}: Anomaly detected in {self._metric_name}. "
            f"Current value: {value:.2f}, Historical mean: {mean:.2f}"
        )


class TrendRule(AlertRule):
    """Trend-based alert rule.

    Detects sustained increases or decreases in metrics.

    Example:
        >>> rule = TrendRule(
        ...     name="accuracy_declining",
        ...     metric_name="accuracy",
        ...     trend="decreasing",
        ...     window_size=10,
        ...     min_change_percent=5.0,
        ... )
    """

    def __init__(
        self,
        name: str,
        metric_name: str,
        trend: str = "increasing",  # increasing, decreasing
        window_size: int = 10,
        min_change_percent: float = 10.0,
        severity: AlertSeverity = AlertSeverity.WARNING,
    ):
        super().__init__(name, severity)
        self._metric_name = metric_name
        self._trend = trend
        self._window_size = window_size
        self._min_change_percent = min_change_percent
        self._history: dict[str, list[float]] = {}

    def evaluate(self, metrics: ModelMetrics) -> bool:
        """Check if metric shows trend.

        Args:
            metrics: Metrics to check

        Returns:
            True if trend detected
        """
        value = self._get_metric_value(metrics)
        if value is None:
            return False

        model_id = metrics.model_id

        # Update history
        if model_id not in self._history:
            self._history[model_id] = []

        self._history[model_id].append(value)

        # Keep window size
        if len(self._history[model_id]) > self._window_size:
            self._history[model_id].pop(0)

        # Need full window
        if len(self._history[model_id]) < self._window_size:
            return False

        # Calculate trend
        history = self._history[model_id]
        first_half = history[:len(history)//2]
        second_half = history[len(history)//2:]

        first_mean = sum(first_half) / len(first_half)
        second_mean = sum(second_half) / len(second_half)

        if first_mean == 0:
            return False

        change_percent = ((second_mean - first_mean) / abs(first_mean)) * 100

        if self._trend == "increasing":
            return change_percent >= self._min_change_percent
        elif self._trend == "decreasing":
            return change_percent <= -self._min_change_percent

        return False

    def _get_metric_value(self, metrics: ModelMetrics) -> float | None:
        """Get metric value by name."""
        name = self._metric_name

        if name == "latency_ms":
            return metrics.latency_ms
        elif name == "throughput_rps":
            return metrics.throughput_rps
        elif name == "prediction_drift":
            return metrics.prediction_drift
        elif name == "accuracy":
            return metrics.accuracy
        elif name == "precision":
            return metrics.precision
        elif name == "recall":
            return metrics.recall
        elif name == "f1_score":
            return metrics.f1_score

        return metrics.custom_metrics.get(name)

    def get_message(self, metrics: ModelMetrics) -> str:
        """Get alert message."""
        history = self._history.get(metrics.model_id, [])
        if len(history) < 2:
            return f"Model {metrics.model_id}: {self._metric_name} trend detected"

        first_half = history[:len(history)//2]
        second_half = history[len(history)//2:]
        first_mean = sum(first_half) / len(first_half)
        second_mean = sum(second_half) / len(second_half)

        return (
            f"Model {metrics.model_id}: {self._metric_name} is {self._trend}. "
            f"Changed from {first_mean:.2f} to {second_mean:.2f}"
        )


class RuleEngine:
    """Engine for evaluating multiple alert rules.

    Example:
        >>> engine = RuleEngine()
        >>> engine.add_rule(ThresholdRule(...))
        >>> engine.add_rule(AnomalyRule(...))
        >>> alerts = engine.evaluate(metrics)
    """

    def __init__(self):
        self._rules: list[AlertRule] = []
        self._active_alerts: dict[str, Alert] = {}  # rule_name:model_id -> Alert

    def add_rule(self, rule: AlertRule) -> None:
        """Add a rule to the engine.

        Args:
            rule: Rule to add
        """
        self._rules.append(rule)

    def remove_rule(self, name: str) -> bool:
        """Remove a rule by name.

        Args:
            name: Rule name

        Returns:
            True if removed
        """
        for i, rule in enumerate(self._rules):
            if rule.name == name:
                self._rules.pop(i)
                return True
        return False

    def evaluate(self, metrics: ModelMetrics) -> list[Alert]:
        """Evaluate all rules against metrics.

        Args:
            metrics: Metrics to evaluate

        Returns:
            List of triggered alerts
        """
        alerts = []

        for rule in self._rules:
            alert_key = f"{rule.name}:{metrics.model_id}"

            if rule.evaluate(metrics):
                # Check if already alerted
                if alert_key not in self._active_alerts:
                    alert = rule.create_alert(metrics)
                    self._active_alerts[alert_key] = alert
                    alerts.append(alert)
            else:
                # Clear if resolved
                if alert_key in self._active_alerts:
                    resolved_alert = self._active_alerts.pop(alert_key)
                    resolved_alert.resolved_at = datetime.now(timezone.utc)
                    # Could emit resolved notification here

        return alerts

    def get_active_alerts(self, model_id: str | None = None) -> list[Alert]:
        """Get currently active alerts.

        Args:
            model_id: Optional model filter

        Returns:
            List of active alerts
        """
        alerts = list(self._active_alerts.values())
        if model_id:
            alerts = [a for a in alerts if a.model_id == model_id]
        return alerts

    def clear(self) -> None:
        """Clear all rules and alerts."""
        self._rules.clear()
        self._active_alerts.clear()
