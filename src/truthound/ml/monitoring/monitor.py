"""Model monitor - unified monitoring interface.

Provides a high-level interface for model monitoring that
coordinates collectors, stores, rules, and handlers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any
import asyncio
import uuid
import logging

from truthound.ml.monitoring.protocols import (
    IMetricCollector,
    IMetricStore,
    IAlertHandler,
    IModelMonitor,
    ModelMetrics,
    PredictionRecord,
    DashboardData,
    Alert,
)
from truthound.ml.monitoring.collectors import (
    PerformanceCollector,
    DriftCollector,
    QualityCollector,
    CompositeCollector,
)
from truthound.ml.monitoring.stores import InMemoryMetricStore
from truthound.ml.monitoring.alerting.rules import RuleEngine, AlertRule


logger = logging.getLogger(__name__)


@dataclass
class MonitorConfig:
    """Configuration for model monitor.

    Attributes:
        batch_size: Number of predictions to batch before collecting
        collect_interval_seconds: Interval for metric collection
        alert_evaluation_interval_seconds: Interval for alert evaluation
        retention_hours: Hours to retain metrics
        enable_drift_detection: Enable drift detection
        enable_quality_metrics: Enable quality metrics
    """

    batch_size: int = 100
    collect_interval_seconds: int = 60
    alert_evaluation_interval_seconds: int = 30
    retention_hours: int = 24
    enable_drift_detection: bool = True
    enable_quality_metrics: bool = True


class ModelMonitor(IModelMonitor):
    """Unified model monitoring interface.

    Coordinates metric collection, storage, alerting, and dashboards.

    Example:
        >>> monitor = ModelMonitor()
        >>> monitor.register_model("my-model", config)
        >>>
        >>> # Record predictions
        >>> await monitor.record_prediction(
        ...     "my-model",
        ...     features={"age": 25, "income": 50000},
        ...     prediction=0.8,
        ...     latency_ms=5.2,
        ... )
        >>>
        >>> # Get dashboard data
        >>> dashboard = await monitor.get_dashboard_data("my-model")
    """

    def __init__(
        self,
        store: IMetricStore | None = None,
        handlers: list[IAlertHandler] | None = None,
    ):
        """Initialize model monitor.

        Args:
            store: Metric store (defaults to in-memory)
            handlers: Alert handlers
        """
        self._store = store or InMemoryMetricStore()
        self._handlers = handlers or []
        self._rule_engine = RuleEngine()

        # Per-model state
        self._configs: dict[str, MonitorConfig] = {}
        self._collectors: dict[str, CompositeCollector] = {}
        self._prediction_buffers: dict[str, list[PredictionRecord]] = {}
        self._reference_set: set[str] = set()  # Models with reference data

        # Background tasks
        self._running = False
        self._tasks: list[asyncio.Task] = []

    def register_model(
        self,
        model_id: str,
        config: MonitorConfig | None = None,
    ) -> None:
        """Register a model for monitoring.

        Args:
            model_id: Model identifier
            config: Monitor configuration
        """
        config = config or MonitorConfig()
        self._configs[model_id] = config
        self._prediction_buffers[model_id] = []

        # Create collectors
        collectors: list[IMetricCollector] = [PerformanceCollector()]

        if config.enable_drift_detection:
            collectors.append(DriftCollector())

        if config.enable_quality_metrics:
            collectors.append(QualityCollector())

        self._collectors[model_id] = CompositeCollector(collectors)

        logger.info(f"Registered model for monitoring: {model_id}")

    def unregister_model(self, model_id: str) -> None:
        """Unregister a model.

        Args:
            model_id: Model identifier
        """
        self._configs.pop(model_id, None)
        self._collectors.pop(model_id, None)
        self._prediction_buffers.pop(model_id, None)
        self._reference_set.discard(model_id)

        logger.info(f"Unregistered model: {model_id}")

    def add_rule(self, rule: AlertRule) -> None:
        """Add an alert rule.

        Args:
            rule: Alert rule
        """
        self._rule_engine.add_rule(rule)

    def add_handler(self, handler: IAlertHandler) -> None:
        """Add an alert handler.

        Args:
            handler: Alert handler
        """
        self._handlers.append(handler)

    async def start(self) -> None:
        """Start background monitoring tasks."""
        if self._running:
            return

        self._running = True

        # Start metric store if needed
        if hasattr(self._store, "start"):
            await self._store.start()

        logger.info("Model monitor started")

    async def stop(self) -> None:
        """Stop background monitoring tasks."""
        self._running = False

        # Cancel tasks
        for task in self._tasks:
            task.cancel()

        # Wait for cancellation
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        self._tasks.clear()

        # Stop metric store if needed
        if hasattr(self._store, "stop"):
            await self._store.stop()

        logger.info("Model monitor stopped")

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
        if model_id not in self._configs:
            logger.warning(f"Model not registered: {model_id}")
            return

        record = PredictionRecord(
            model_id=model_id,
            prediction_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            features=features,
            prediction=prediction,
            actual=actual,
            latency_ms=latency_ms,
        )

        self._prediction_buffers[model_id].append(record)

        # Check if batch is full
        config = self._configs[model_id]
        if len(self._prediction_buffers[model_id]) >= config.batch_size:
            await self._collect_and_store(model_id)

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
        # Find prediction in buffer and update
        buffer = self._prediction_buffers.get(model_id, [])
        for record in buffer:
            if record.prediction_id == prediction_id:
                record.actual = actual
                return

        # If not in buffer, log warning
        logger.debug(f"Prediction not found in buffer: {prediction_id}")

    async def set_reference_data(
        self,
        model_id: str,
        predictions: list[PredictionRecord],
    ) -> None:
        """Set reference data for drift detection.

        Args:
            model_id: Model identifier
            predictions: Reference predictions
        """
        if model_id not in self._collectors:
            return

        collector = self._collectors[model_id]
        for c in collector.collectors:
            if isinstance(c, DriftCollector):
                c.set_reference(model_id, predictions)

        self._reference_set.add(model_id)
        logger.info(f"Set reference data for {model_id}: {len(predictions)} records")

    async def get_metrics(self, model_id: str) -> ModelMetrics | None:
        """Get current metrics for model.

        Args:
            model_id: Model identifier

        Returns:
            Current metrics or None
        """
        return await self._store.get_latest(model_id)

    async def get_dashboard_data(self, model_id: str) -> DashboardData:
        """Get dashboard data for model.

        Args:
            model_id: Model identifier

        Returns:
            Dashboard data
        """
        now = datetime.now(timezone.utc)
        config = self._configs.get(model_id, MonitorConfig())

        # Get metrics history
        start_time = now - timedelta(hours=config.retention_hours)
        metrics_history = await self._store.query(model_id, start_time, now)

        # Get current metrics
        current_metrics = await self._store.get_latest(model_id)

        # Get alerts
        active_alerts = self._rule_engine.get_active_alerts(model_id)

        # Calculate health score
        health_score = self._calculate_health_score(current_metrics, active_alerts)

        return DashboardData(
            model_id=model_id,
            current_metrics=current_metrics,
            metrics_history=metrics_history,
            active_alerts=active_alerts,
            alert_history=[],  # Would need alert history store
            health_score=health_score,
            last_updated=now,
        )

    async def _collect_and_store(self, model_id: str) -> None:
        """Collect metrics from buffer and store.

        Args:
            model_id: Model identifier
        """
        predictions = self._prediction_buffers.get(model_id, [])
        if not predictions:
            return

        collector = self._collectors.get(model_id)
        if not collector:
            return

        # Collect metrics
        metrics = collector.collect(model_id, predictions)

        # Store metrics
        await self._store.store(metrics)

        # Evaluate alerts
        alerts = self._rule_engine.evaluate(metrics)

        # Send alerts
        for alert in alerts:
            await self._send_alert(alert)

        # Clear buffer
        self._prediction_buffers[model_id] = []

    async def _send_alert(self, alert: Alert) -> None:
        """Send alert to all handlers.

        Args:
            alert: Alert to send
        """
        for handler in self._handlers:
            try:
                await handler.handle(alert)
            except Exception as e:
                logger.error(f"Handler {handler.name} failed: {e}")

    def _calculate_health_score(
        self,
        metrics: ModelMetrics | None,
        alerts: list[Alert],
    ) -> float:
        """Calculate model health score (0-100).

        Args:
            metrics: Current metrics
            alerts: Active alerts

        Returns:
            Health score
        """
        if metrics is None:
            return 100.0

        score = 100.0

        # Deduct for drift
        if metrics.prediction_drift is not None:
            if metrics.prediction_drift > 0.2:
                score -= 20
            elif metrics.prediction_drift > 0.1:
                score -= 10

        # Deduct for feature drift
        for drift in metrics.feature_drift.values():
            if drift > 0.2:
                score -= 5
            elif drift > 0.1:
                score -= 2

        # Deduct for low accuracy
        if metrics.accuracy is not None:
            if metrics.accuracy < 0.8:
                score -= 15
            elif metrics.accuracy < 0.9:
                score -= 5

        # Deduct for high latency
        if metrics.latency_ms > 1000:
            score -= 15
        elif metrics.latency_ms > 500:
            score -= 5

        # Deduct for active alerts
        for alert in alerts:
            if alert.severity.value == "critical":
                score -= 20
            elif alert.severity.value == "error":
                score -= 10
            elif alert.severity.value == "warning":
                score -= 5

        return max(0.0, min(100.0, score))

    async def __aenter__(self) -> "ModelMonitor":
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.stop()


class MonitoringPipeline:
    """Composable monitoring pipeline.

    Example:
        >>> pipeline = MonitoringPipeline()
        >>> pipeline.add_collector(PerformanceCollector())
        >>> pipeline.add_collector(DriftCollector())
        >>> pipeline.set_store(PrometheusMetricStore())
        >>> pipeline.add_rule(ThresholdRule(...))
        >>> pipeline.add_handler(SlackAlertHandler(...))
        >>>
        >>> await pipeline.process(model_id, predictions)
    """

    def __init__(self):
        self._collectors: list[IMetricCollector] = []
        self._store: IMetricStore | None = None
        self._rules: list[AlertRule] = []
        self._handlers: list[IAlertHandler] = []

    def add_collector(self, collector: IMetricCollector) -> "MonitoringPipeline":
        """Add a collector.

        Args:
            collector: Metric collector

        Returns:
            Self for chaining
        """
        self._collectors.append(collector)
        return self

    def set_store(self, store: IMetricStore) -> "MonitoringPipeline":
        """Set metric store.

        Args:
            store: Metric store

        Returns:
            Self for chaining
        """
        self._store = store
        return self

    def add_rule(self, rule: AlertRule) -> "MonitoringPipeline":
        """Add alert rule.

        Args:
            rule: Alert rule

        Returns:
            Self for chaining
        """
        self._rules.append(rule)
        return self

    def add_handler(self, handler: IAlertHandler) -> "MonitoringPipeline":
        """Add alert handler.

        Args:
            handler: Alert handler

        Returns:
            Self for chaining
        """
        self._handlers.append(handler)
        return self

    async def process(
        self,
        model_id: str,
        predictions: list[PredictionRecord],
    ) -> tuple[ModelMetrics, list[Alert]]:
        """Process predictions through pipeline.

        Args:
            model_id: Model identifier
            predictions: Predictions to process

        Returns:
            Tuple of (metrics, alerts)
        """
        # Collect metrics
        composite = CompositeCollector(self._collectors)
        metrics = composite.collect(model_id, predictions)

        # Store metrics
        if self._store:
            await self._store.store(metrics)

        # Evaluate rules
        alerts = []
        for rule in self._rules:
            if rule.evaluate(metrics):
                alert = rule.create_alert(metrics)
                alerts.append(alert)

                # Send to handlers
                for handler in self._handlers:
                    try:
                        await handler.handle(alert)
                    except Exception as e:
                        logger.error(f"Handler failed: {e}")

        return metrics, alerts
