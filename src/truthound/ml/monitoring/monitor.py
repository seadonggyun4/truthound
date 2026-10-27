"""Model monitor - unified monitoring interface.

Provides a high-level interface for model monitoring that
coordinates collectors, stores, rules, and handlers.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import math
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from numbers import Real
from typing import Any

from truthound.ml.monitoring.alerting.rules import AlertRule, RuleEngine
from truthound.ml.monitoring.collectors import (
    CompositeCollector,
    DriftCollector,
    PerformanceCollector,
    QualityCollector,
)
from truthound.ml.monitoring.collectors.drift import DriftConfig
from truthound.ml.monitoring.collectors.performance import PerformanceConfig
from truthound.ml.monitoring.collectors.quality import QualityConfig
from truthound.ml.monitoring.protocols import (
    Alert,
    DashboardData,
    IAlertHandler,
    IMetricCollector,
    IMetricStore,
    IModelMonitor,
    ModelMetrics,
    PredictionRecord,
)
from truthound.ml.monitoring.stores import InMemoryMetricStore

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
    collect_interval_seconds: float = 60
    alert_evaluation_interval_seconds: float = 30
    retention_hours: float = 24
    enable_drift_detection: bool = True
    enable_quality_metrics: bool = True
    performance: PerformanceConfig | None = None
    drift: DriftConfig | None = None
    quality: QualityConfig | None = None

    def __post_init__(self):
        if type(self.batch_size) is not int or self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        for name in (
            "collect_interval_seconds",
            "alert_evaluation_interval_seconds",
            "retention_hours",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        for name in ("enable_drift_detection", "enable_quality_metrics"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        for name, kind in (
            ("performance", PerformanceConfig),
            ("drift", DriftConfig),
            ("quality", QualityConfig),
        ):
            value = getattr(self, name)
            if value is not None:
                if not isinstance(value, kind):
                    raise TypeError("A native collector configuration is required")
                setattr(self, name, replace(copy.deepcopy(value)))


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
        self._store = store if store is not None else InMemoryMetricStore()
        self._handlers = list(handlers) if handlers is not None else []
        self._rule_engine = RuleEngine()

        # Per-model state
        self._configs: dict[str, MonitorConfig] = {}
        self._collectors: dict[str, CompositeCollector] = {}
        self._prediction_buffers: dict[str, list[PredictionRecord]] = {}
        self._reference_set: set[str] = set()  # Models with reference data

        # Background tasks
        self._running = False
        self._tasks: list[asyncio.Task] = []
        self._locks: dict[str, asyncio.Lock] = {}
        self._uncertain: dict[str, str] = {}
        self._model_epochs = {}
        self._model_tasks = {}
        self._store_active = False
        self._execution_guard = None
        self._evaluated = {}
        self._stopping = False
        self._lifecycle_lock = asyncio.Lock()
        self._inflight = {}

    def set_execution_guard(self, guard):
        if guard is not None and not callable(guard):
            raise TypeError("Execution guard must be callable")
        self._execution_guard = guard
        self._rule_engine.set_execution_guard(guard)

    def _guard_execution(self):
        if self._execution_guard is not None:
            self._execution_guard()

    def _require_model(self, model_id):
        if self._stopping:
            raise RuntimeError("Native monitor cleanup is in progress")
        if model_id not in self._configs:
            raise KeyError("Native monitor model is not registered")
        if model_id in self._uncertain:
            raise RuntimeError("Native model collection outcome is uncertain")

    def get_status(self):
        self._guard_execution()
        return {
            "running": self._running,
            "store_active": self._store_active,
            "stopping": self._stopping,
            "models": {
                model_id: {
                    "buffered": len(self._prediction_buffers[model_id]),
                    "uncertain": model_id in self._uncertain,
                    "error_type": self._uncertain.get(model_id),
                    "reference_set": model_id in self._reference_set,
                }
                for model_id in self._configs
            },
        }

    def get_pending_prediction_ids(self, model_id: str) -> list[str]:
        """Return pending identities without exposing prediction contents."""
        self._guard_execution()
        if model_id not in self._configs:
            raise KeyError("Native monitor model is not registered")
        return [record.prediction_id for record in self._prediction_buffers[model_id]]

    def get_components(self):
        """List native component names without backend or handler configuration."""
        self._guard_execution()
        return {
            "models": list(self._configs),
            "rules": self._rule_engine.get_rule_names(),
            "handlers": [handler.name for handler in self._handlers],
        }

    @asynccontextmanager
    async def _locked_model(self, model_id, *, allow_uncertain=False):
        lock = self._locks[model_id]
        async with lock:
            if self._stopping:
                raise RuntimeError("Native monitor cleanup is in progress")
            if not allow_uncertain:
                self._require_model(model_id)
            task = asyncio.current_task()
            self._inflight[task] = lock
            try:
                yield
            finally:
                self._inflight.pop(task, None)

    async def collect_now(self, model_id):
        self._guard_execution()
        self._require_model(model_id)
        async with self._locked_model(model_id):
            self._require_model(model_id)
            return await self._collect_and_store(model_id)

    async def discard_uncertain(self, model_id):
        """Explicitly discard buffered records; never retry uncertain effects."""
        if model_id not in self._uncertain:
            raise ValueError("Native model has no uncertain collection")
        async with self._locked_model(model_id, allow_uncertain=True):
            self._guard_execution()
            discarded = len(self._prediction_buffers[model_id])
            metrics = await self._store.get_latest(model_id)
            self._guard_execution()
            if self._stopping:
                raise RuntimeError("Native monitor cleanup is in progress")
            if metrics is not None:
                self._evaluated[model_id] = self._metric_key(metrics)
            self._prediction_buffers[model_id].clear()
            self._collectors[model_id].reset()
            self._reference_set.discard(model_id)
            del self._uncertain[model_id]
            return discarded

    def _spawn_collection(self, model_id):
        for operation in (self._collection_loop, self._alert_loop):
            self._spawn_task(model_id, operation)

    def _spawn_task(self, model_id, operation):
        epoch = self._model_epochs[model_id]
        task = asyncio.create_task(operation(model_id, epoch))
        self._tasks.append(task)
        self._model_tasks.setdefault(model_id, []).append(task)

        def finished(done):
            if done in self._tasks:
                self._tasks.remove(done)
            if not done.cancelled():
                error = done.exception()
                if error is not None and self._model_epochs.get(model_id) is epoch:
                    self._uncertain.setdefault(model_id, type(error).__name__)

        task.add_done_callback(finished)

    async def _collection_loop(self, model_id, epoch):
        while self._running and self._model_epochs.get(model_id) is epoch:
            await asyncio.sleep(self._configs[model_id].collect_interval_seconds)
            if not self._running or self._model_epochs.get(model_id) is not epoch:
                return
            if model_id in self._uncertain:
                continue
            try:
                await self.collect_now(model_id)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self._uncertain.setdefault(model_id, type(error).__name__)

    async def _alert_loop(self, model_id, epoch):
        while self._running and self._model_epochs.get(model_id) is epoch:
            await asyncio.sleep(self._configs[model_id].alert_evaluation_interval_seconds)
            if not self._running or self._model_epochs.get(model_id) is not epoch:
                return
            if model_id in self._uncertain:
                continue
            try:
                await self.evaluate_now(model_id)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self._uncertain.setdefault(model_id, type(error).__name__)

    async def evaluate_now(self, model_id):
        """Evaluate a newly observed snapshot once, then apply model retention."""
        self._guard_execution()
        self._require_model(model_id)
        async with self._locked_model(model_id):
            self._require_model(model_id)
            epoch = self._model_epochs[model_id]
            try:
                metrics = await self._store.get_latest(model_id)
                self._guard_execution()
                self._require_model(model_id)
                if self._model_epochs.get(model_id) is not epoch:
                    raise RuntimeError("Native model registration changed during evaluation")
                if metrics is not None:
                    await self._evaluate_snapshot(model_id, metrics)
                self._guard_execution()
                self._require_model(model_id)
                cutoff = datetime.now(UTC) - timedelta(
                    hours=self._configs[model_id].retention_hours
                )
                deleted = await self._store.delete(model_id, before=cutoff)
                self._guard_execution()
                if type(deleted) is not int or deleted < 0:
                    raise RuntimeError("Native metric retention outcome is uncertain")
                return deleted
            except BaseException as error:
                if self._model_epochs.get(model_id) is epoch:
                    self._uncertain[model_id] = type(error).__name__
                raise

    @staticmethod
    def _metric_key(metrics):
        if not isinstance(metrics, ModelMetrics):
            raise TypeError("A native model metrics snapshot is required")
        encoded = json.dumps(
            metrics.to_dict(), sort_keys=True, allow_nan=False, separators=(",", ":")
        )
        return hashlib.sha256(encoded.encode()).hexdigest()

    async def _evaluate_snapshot(self, model_id, metrics):
        if metrics.model_id != model_id:
            raise ValueError("Native metric snapshot belongs to another model")
        key = self._metric_key(metrics)
        if self._evaluated.get(model_id) == key:
            return
        alerts = self._rule_engine.evaluate(metrics)
        for alert in alerts:
            await self._send_alert(alert)
        for alert in self._rule_engine.get_resolved_alerts():
            await self._send_alert(alert, resolved=True)
        self._guard_execution()
        self._evaluated[model_id] = key

    def register_model(self, model_id: str, config: MonitorConfig | None = None) -> None:
        """Register a unique model without replacing pending prediction state."""
        self._guard_execution()
        if not isinstance(model_id, str) or not model_id:
            raise ValueError("A native monitor model identifier is required")
        if model_id in self._configs:
            raise ValueError("Native monitor model is already registered")
        config = replace(config) if config is not None else MonitorConfig()
        collectors: list[IMetricCollector] = [PerformanceCollector(config.performance)]
        if config.enable_drift_detection:
            collectors.append(DriftCollector(config.drift))
        if config.enable_quality_metrics:
            collectors.append(QualityCollector(config.quality))
        self._configs[model_id] = config
        self._prediction_buffers[model_id] = []
        self._locks[model_id] = asyncio.Lock()
        self._model_epochs[model_id] = object()
        self._collectors[model_id] = CompositeCollector(collectors)
        self._collectors[model_id].set_execution_guard(self._guard_execution)
        if self._running:
            self._spawn_collection(model_id)
        logger.info("Registered native model for monitoring")

    def unregister_model(self, model_id: str) -> None:
        """Explicitly remove a model and its in-memory pending predictions."""
        self._guard_execution()
        if model_id not in self._configs:
            raise KeyError("Native monitor model is not registered")
        if self._locks[model_id].locked():
            raise RuntimeError("Native monitor model operation is in progress")
        for task in self._model_tasks.pop(model_id, []):
            task.cancel()
        self._model_epochs.pop(model_id, None)
        self._configs.pop(model_id, None)
        self._collectors.pop(model_id, None)
        self._prediction_buffers.pop(model_id, None)
        self._reference_set.discard(model_id)
        self._uncertain.pop(model_id, None)
        self._locks.pop(model_id, None)
        self._evaluated.pop(model_id, None)
        logger.info("Unregistered native model from monitoring")

    def add_rule(self, rule: AlertRule) -> None:
        """Add an alert rule.

        Args:
            rule: Alert rule
        """
        self._guard_execution()
        self._rule_engine.add_rule(rule)
        self._evaluated.clear()

    def remove_rule(self, name: str) -> bool:
        self._guard_execution()
        removed = self._rule_engine.remove_rule(name)
        if removed:
            self._evaluated.clear()
        return removed

    def add_handler(self, handler: IAlertHandler) -> None:
        """Add an alert handler.

        Args:
            handler: Alert handler
        """
        self._guard_execution()
        if not isinstance(handler, IAlertHandler):
            raise TypeError("A native alert handler is required")
        if any(value.name == handler.name for value in self._handlers):
            raise ValueError("Native alert handler name is already registered")
        self._handlers.append(handler)

    def remove_handler(self, name: str) -> bool:
        self._guard_execution()
        for index, handler in enumerate(self._handlers):
            if handler.name == name:
                self._handlers.pop(index)
                return True
        return False

    async def start(self) -> None:
        async with self._lifecycle_lock:
            await self._start_owned()

    async def _start_owned(self) -> None:
        """Start the owned store and configured per-model collection tasks."""
        if self._running:
            return
        if self._store_active or self._stopping:
            raise RuntimeError("Native monitor store cleanup is incomplete")
        self._guard_execution()
        if hasattr(self._store, "start"):
            self._store_active = True
            try:
                result = await self._store.start()
                if result is False:
                    raise RuntimeError("Native metric store did not start")
            except BaseException as original:
                try:
                    await self._stop_owned()
                except BaseException as cleanup:
                    raise BaseExceptionGroup(
                        "Native monitor startup and cleanup failed", [original, cleanup]
                    ) from None
                raise
        self._running = True
        for model_id in self._configs:
            self._spawn_collection(model_id)
        logger.info("Model monitor started")

    async def stop(self) -> None:
        """Stop owned tasks/store without flushing or replaying pending records."""
        if asyncio.current_task() in self._inflight:
            raise RuntimeError("Cannot stop a native monitor from its in-flight model operation")
        async with self._lifecycle_lock:
            await self._stop_owned()

    async def _stop_owned(self):
        self._stopping = True
        self._running = False
        current = asyncio.current_task()
        tasks = [task for task in self._tasks if task is not current]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks = [task for task in self._tasks if not task.done()]
        self._model_tasks.clear()
        # Foreground callers own their tasks; wait for their model operation
        # instead of cancelling unrelated caller work or closing its store early.
        locks = tuple(dict.fromkeys((*self._locks.values(), *self._inflight.values())))
        for lock in locks:
            async with lock:
                pass
        if hasattr(self._store, "stop"):
            result = await self._store.stop()
            if result is False:
                raise RuntimeError("Native metric store cleanup is incomplete")
        self._store_active = False
        self._stopping = False
        logger.info("Model monitor stopped")

    async def record_prediction(
        self,
        model_id: str,
        features: dict[str, Any],
        prediction: Any,
        actual: Any | None = None,
        latency_ms: float = 0.0,
    ) -> str:
        """Record a prediction and return its identity for pending feedback."""
        self._guard_execution()
        self._require_model(model_id)
        if not isinstance(features, dict) or any(not isinstance(key, str) for key in features):
            raise TypeError("Prediction features must be a mapping with string keys")
        if (
            isinstance(latency_ms, bool)
            or not isinstance(latency_ms, Real)
            or not math.isfinite(latency_ms)
            or latency_ms < 0
        ):
            raise ValueError("Prediction latency must be finite and non-negative")
        async with self._locked_model(model_id):
            self._require_model(model_id)
            self._guard_execution()
            record = PredictionRecord(
                model_id=model_id,
                prediction_id=str(uuid.uuid4()),
                timestamp=datetime.now(UTC),
                features=copy.deepcopy(features),
                prediction=copy.deepcopy(prediction),
                actual=copy.deepcopy(actual),
                latency_ms=float(latency_ms),
            )
            self._prediction_buffers[model_id].append(record)
            if len(self._prediction_buffers[model_id]) >= self._configs[model_id].batch_size:
                await self._collect_and_store(model_id)
            return record.prediction_id

    async def record_actual(self, model_id: str, prediction_id: str, actual: Any) -> None:
        """Update a still-pending prediction; missing identities are failures."""
        self._guard_execution()
        self._require_model(model_id)
        async with self._locked_model(model_id):
            self._require_model(model_id)
            for record in self._prediction_buffers[model_id]:
                if record.prediction_id == prediction_id:
                    record.actual = copy.deepcopy(actual)
                    return
            raise KeyError("Native prediction is no longer pending")

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
        self._guard_execution()
        self._require_model(model_id)
        if not predictions or any(
            not isinstance(item, PredictionRecord) or item.model_id != model_id
            for item in predictions
        ):
            raise ValueError("Reference predictions must belong to the registered model")
        async with self._locked_model(model_id):
            self._require_model(model_id)
            self._guard_execution()
            collectors = [
                item
                for item in self._collectors[model_id].collectors
                if isinstance(item, DriftCollector)
            ]
            if not collectors:
                raise ValueError("Native drift collection is disabled")
            for collector in collectors:
                collector.set_reference(model_id, copy.deepcopy(predictions))
                self._guard_execution()
            self._reference_set.add(model_id)
        logger.info("Native model reference predictions set")

    async def get_metrics(self, model_id: str) -> ModelMetrics | None:
        """Get current metrics for model.

        Args:
            model_id: Model identifier

        Returns:
            Current metrics or None
        """
        self._guard_execution()
        if model_id not in self._configs:
            raise KeyError("Native monitor model is not registered")
        result = await self._store.get_latest(model_id)
        self._guard_execution()
        return result

    async def get_dashboard_data(self, model_id: str) -> DashboardData:
        """Get dashboard data for model.

        Args:
            model_id: Model identifier

        Returns:
            Dashboard data
        """
        self._guard_execution()
        if model_id not in self._configs:
            raise KeyError("Native monitor model is not registered")
        now = datetime.now(UTC)
        config = self._configs[model_id]

        # Get metrics history
        start_time = now - timedelta(hours=config.retention_hours)
        metrics_history = await self._store.query(model_id, start_time, now)

        # Get current metrics
        current_metrics = await self._store.get_latest(model_id)
        self._guard_execution()

        # Get alerts
        active_alerts = self._rule_engine.get_active_alerts(model_id)

        # Calculate health score
        health_score = self._calculate_health_score(current_metrics, active_alerts)

        return DashboardData(
            model_id=model_id,
            current_metrics=current_metrics,
            metrics_history=metrics_history,
            active_alerts=active_alerts,
            alert_history=self._rule_engine.get_alert_history(model_id),
            health_score=health_score,
            last_updated=now,
        )

    async def _collect_and_store(self, model_id: str) -> int:
        """Collect one locked batch; uncertain effects are latched, never replayed."""
        self._require_model(model_id)
        epoch = self._model_epochs[model_id]
        predictions = list(self._prediction_buffers[model_id])
        if not predictions:
            return 0
        collector = self._collectors[model_id]
        try:
            self._guard_execution()
            metrics = collector.collect(model_id, predictions)
            self._guard_execution()
            self._metric_key(metrics)
            if metrics.model_id != model_id:
                raise ValueError("Native metric snapshot belongs to another model")
            receipt = await self._store.store(metrics)
            if receipt is not None and receipt is not True:
                raise RuntimeError("Native metric storage outcome is uncertain")
            self._guard_execution()
            self._require_model(model_id)
            if self._model_epochs.get(model_id) is not epoch:
                raise RuntimeError("Native model registration changed during collection")
            await self._evaluate_snapshot(model_id, metrics)
            self._guard_execution()
        except BaseException as error:
            if self._model_epochs.get(model_id) is epoch:
                self._uncertain[model_id] = type(error).__name__
            raise
        del self._prediction_buffers[model_id][: len(predictions)]
        return len(predictions)

    async def _send_alert(self, alert: Alert, *, resolved: bool = False) -> None:
        """Require every selected handler's actual positive delivery receipt."""
        for handler in tuple(self._handlers):
            self._guard_execution()
            self._require_model(alert.model_id)
            result = await (handler.resolve(alert) if resolved else handler.handle(alert))
            self._guard_execution()
            self._require_model(alert.model_id)
            if result is not True:
                raise RuntimeError("Native alert delivery outcome is uncertain")

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

    async def __aenter__(self) -> ModelMonitor:
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
        self._locks: dict[str, asyncio.Lock] = {}
        self._uncertain: set[str] = set()

    def discard_uncertain(self, model_id: str) -> None:
        """Explicitly acknowledge unknown effects without retrying them."""
        if self._locks.get(model_id) is not None and self._locks[model_id].locked():
            raise RuntimeError("Native pipeline model is busy")
        if model_id not in self._uncertain:
            raise ValueError("Native pipeline model has no uncertain outcome")
        self._uncertain.remove(model_id)

    def add_collector(self, collector: IMetricCollector) -> MonitoringPipeline:
        """Add a collector.

        Args:
            collector: Metric collector

        Returns:
            Self for chaining
        """
        self._collectors.append(collector)
        return self

    def set_store(self, store: IMetricStore) -> MonitoringPipeline:
        """Set metric store.

        Args:
            store: Metric store

        Returns:
            Self for chaining
        """
        self._store = store
        return self

    def add_rule(self, rule: AlertRule) -> MonitoringPipeline:
        """Add alert rule.

        Args:
            rule: Alert rule

        Returns:
            Self for chaining
        """
        self._rules.append(rule)
        return self

    def add_handler(self, handler: IAlertHandler) -> MonitoringPipeline:
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
        async with self._locks.setdefault(model_id, asyncio.Lock()):
            if model_id in self._uncertain:
                raise RuntimeError("Native pipeline model outcome is uncertain")
            if any(
                not isinstance(item, PredictionRecord) or item.model_id != model_id
                for item in predictions
            ):
                raise ValueError("Predictions must belong to the native pipeline model")
            try:
                composite = CompositeCollector(self._collectors)
                metrics = composite.collect(model_id, copy.deepcopy(predictions))
                ModelMonitor._metric_key(metrics)
                if metrics.model_id != model_id:
                    raise ValueError("Native metric snapshot belongs to another model")
                if self._store is not None:
                    receipt = await self._store.store(metrics)
                    if receipt is not None and receipt is not True:
                        raise RuntimeError("Native metric storage outcome is uncertain")
                alerts = []
                for rule in self._rules:
                    verdict = rule.evaluate(metrics)
                    if type(verdict) is not bool:
                        raise TypeError("Native alert rule must return a boolean verdict")
                    if verdict:
                        alert = rule.create_alert(metrics)
                        alerts.append(alert)
                        for handler in self._handlers:
                            if await handler.handle(alert) is not True:
                                raise RuntimeError("Native alert delivery outcome is uncertain")
                return metrics, alerts
            except BaseException:
                self._uncertain.add(model_id)
                raise
