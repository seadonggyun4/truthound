"""Protocols for ML model monitoring.

Defines interfaces for metric collection, storage, alerting,
and dashboard components.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass(frozen=True)
class ModelMetrics:
    """Comprehensive model metrics snapshot.

    Attributes:
        model_id: Unique model identifier
        timestamp: Metrics timestamp
        latency_ms: Prediction latency in milliseconds
        throughput_rps: Throughput in requests per second
        prediction_drift: Prediction distribution drift score
        feature_drift: Per-feature drift scores
        accuracy: Model accuracy (if labels available)
        precision: Model precision
        recall: Model recall
        f1_score: F1 score
        custom_metrics: User-defined metrics
    """

    model_id: str
    timestamp: datetime
    latency_ms: float = 0.0
    throughput_rps: float = 0.0
    prediction_drift: float | None = None
    feature_drift: dict[str, float] = field(default_factory=dict)
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1_score: float | None = None
    custom_metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "timestamp": self.timestamp.isoformat(),
            "latency_ms": self.latency_ms,
            "throughput_rps": self.throughput_rps,
            "prediction_drift": self.prediction_drift,
            "feature_drift": dict(self.feature_drift),
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "custom_metrics": dict(self.custom_metrics),
        }

    def merge(self, other: "ModelMetrics") -> "ModelMetrics":
        """Merge with another metrics snapshot."""
        merged_feature_drift = {**self.feature_drift, **other.feature_drift}
        merged_custom = {**self.custom_metrics, **other.custom_metrics}

        return ModelMetrics(
            model_id=self.model_id,
            timestamp=max(self.timestamp, other.timestamp),
            latency_ms=other.latency_ms if other.latency_ms else self.latency_ms,
            throughput_rps=other.throughput_rps if other.throughput_rps else self.throughput_rps,
            prediction_drift=other.prediction_drift if other.prediction_drift is not None else self.prediction_drift,
            feature_drift=merged_feature_drift,
            accuracy=other.accuracy if other.accuracy is not None else self.accuracy,
            precision=other.precision if other.precision is not None else self.precision,
            recall=other.recall if other.recall is not None else self.recall,
            f1_score=other.f1_score if other.f1_score is not None else self.f1_score,
            custom_metrics=merged_custom,
        )


@dataclass
class PredictionRecord:
    """Record of a single prediction for monitoring.

    Attributes:
        model_id: Model identifier
        prediction_id: Unique prediction identifier
        timestamp: Prediction timestamp
        features: Input features
        prediction: Model prediction
        actual: Actual value (for delayed feedback)
        latency_ms: Prediction latency
        metadata: Additional metadata
    """

    model_id: str
    prediction_id: str
    timestamp: datetime
    features: dict[str, Any]
    prediction: Any
    actual: Any | None = None
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """Model monitoring alert.

    Attributes:
        alert_id: Unique alert identifier
        model_id: Model that triggered the alert
        rule_name: Alert rule that fired
        severity: Alert severity
        message: Alert message
        metrics: Metrics that triggered the alert
        triggered_at: When the alert was triggered
        resolved_at: When the alert was resolved (if applicable)
        metadata: Additional alert metadata
    """

    alert_id: str
    model_id: str
    rule_name: str
    severity: AlertSeverity
    message: str
    metrics: ModelMetrics
    triggered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_resolved(self) -> bool:
        """Check if alert is resolved."""
        return self.resolved_at is not None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "model_id": self.model_id,
            "rule_name": self.rule_name,
            "severity": self.severity.value,
            "message": self.message,
            "metrics": self.metrics.to_dict(),
            "triggered_at": self.triggered_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "metadata": self.metadata,
        }


@dataclass
class DashboardData:
    """Data for model monitoring dashboard.

    Attributes:
        model_id: Model identifier
        current_metrics: Latest metrics
        metrics_history: Historical metrics
        active_alerts: Currently active alerts
        alert_history: Recent alert history
        health_score: Overall model health score (0-100)
        last_updated: Last update timestamp
    """

    model_id: str
    current_metrics: ModelMetrics | None
    metrics_history: list[ModelMetrics]
    active_alerts: list[Alert]
    alert_history: list[Alert]
    health_score: float
    last_updated: datetime

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "current_metrics": self.current_metrics.to_dict() if self.current_metrics else None,
            "metrics_history": [m.to_dict() for m in self.metrics_history],
            "active_alerts": [a.to_dict() for a in self.active_alerts],
            "alert_history": [a.to_dict() for a in self.alert_history],
            "health_score": self.health_score,
            "last_updated": self.last_updated.isoformat(),
        }


# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class IMetricCollector(Protocol):
    """Protocol for metric collectors.

    Collectors gather specific types of metrics from predictions.
    """

    @property
    def name(self) -> str:
        """Collector name."""
        ...

    def collect(
        self,
        model_id: str,
        predictions: list[PredictionRecord],
    ) -> ModelMetrics:
        """Collect metrics from predictions.

        Args:
            model_id: Model identifier
            predictions: List of prediction records

        Returns:
            Collected metrics
        """
        ...

    def reset(self) -> None:
        """Reset collector state."""
        ...


@runtime_checkable
class IMetricStore(Protocol):
    """Protocol for metric storage.

    Stores provide persistence for collected metrics.
    """

    async def store(self, metrics: ModelMetrics) -> None:
        """Store metrics.

        Args:
            metrics: Metrics to store
        """
        ...

    async def query(
        self,
        model_id: str,
        start_time: datetime,
        end_time: datetime,
        aggregation: str = "mean",
    ) -> list[ModelMetrics]:
        """Query historical metrics.

        Args:
            model_id: Model identifier
            start_time: Query start time
            end_time: Query end time
            aggregation: Aggregation function (mean, max, min, sum)

        Returns:
            List of metrics matching the query
        """
        ...

    async def get_latest(self, model_id: str) -> ModelMetrics | None:
        """Get latest metrics for model.

        Args:
            model_id: Model identifier

        Returns:
            Latest metrics or None
        """
        ...

    async def delete(
        self,
        model_id: str,
        before: datetime | None = None,
    ) -> int:
        """Delete metrics.

        Args:
            model_id: Model identifier
            before: Delete metrics before this time

        Returns:
            Number of deleted records
        """
        ...


@runtime_checkable
class IAlertRule(Protocol):
    """Protocol for alert rules.

    Rules evaluate metrics and determine if alerts should fire.
    """

    @property
    def name(self) -> str:
        """Rule name."""
        ...

    @property
    def severity(self) -> AlertSeverity:
        """Alert severity."""
        ...

    def evaluate(self, metrics: ModelMetrics) -> bool:
        """Evaluate rule against metrics.

        Args:
            metrics: Metrics to evaluate

        Returns:
            True if rule condition is met
        """
        ...

    def get_message(self, metrics: ModelMetrics) -> str:
        """Get alert message.

        Args:
            metrics: Metrics that triggered the rule

        Returns:
            Alert message
        """
        ...


@runtime_checkable
class IAlertHandler(Protocol):
    """Protocol for alert handlers.

    Handlers send alerts to external systems.
    """

    @property
    def name(self) -> str:
        """Handler name."""
        ...

    async def handle(self, alert: Alert) -> bool:
        """Handle an alert.

        Args:
            alert: Alert to handle

        Returns:
            True if handled successfully
        """
        ...

    async def resolve(self, alert: Alert) -> bool:
        """Handle alert resolution.

        Args:
            alert: Resolved alert

        Returns:
            True if handled successfully
        """
        ...


@runtime_checkable
class IModelMonitor(Protocol):
    """Protocol for model monitors.

    Monitors coordinate metric collection, storage, and alerting.
    """

    def register_model(self, model_id: str, config: Any) -> None:
        """Register a model for monitoring.

        Args:
            model_id: Model identifier
            config: Monitor configuration
        """
        ...

    def unregister_model(self, model_id: str) -> None:
        """Unregister a model.

        Args:
            model_id: Model identifier
        """
        ...

    async def record_prediction(
        self,
        model_id: str,
        features: dict[str, Any],
        prediction: Any,
        actual: Any | None = None,
        latency_ms: float = 0.0,
    ) -> None:
        """Record a prediction for monitoring.

        Args:
            model_id: Model identifier
            features: Input features
            prediction: Model prediction
            actual: Actual value (for delayed feedback)
            latency_ms: Prediction latency
        """
        ...

    async def record_actual(
        self,
        model_id: str,
        prediction_id: str,
        actual: Any,
    ) -> None:
        """Record actual value for delayed feedback.

        Args:
            model_id: Model identifier
            prediction_id: Prediction identifier
            actual: Actual value
        """
        ...

    async def get_metrics(self, model_id: str) -> ModelMetrics | None:
        """Get current metrics for model.

        Args:
            model_id: Model identifier

        Returns:
            Current metrics or None
        """
        ...

    async def get_dashboard_data(self, model_id: str) -> DashboardData:
        """Get dashboard data for model.

        Args:
            model_id: Model identifier

        Returns:
            Dashboard data
        """
        ...
