"""Analytics service facade.

Provides a unified interface for the historical trend analysis system,
coordinating stores, analyzers, and aggregations.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from truthound.checkpoint.analytics.protocols import (
    TimeSeriesStoreProtocol,
    TrendAnalyzerProtocol,
    TimeSeriesPoint,
    TimeGranularity,
    AggregationFunction,
    TrendResult,
    TrendDirection,
    AnomalyResult,
    ForecastResult,
    AnalyzerError,
)
from truthound.checkpoint.analytics.models import (
    CheckpointExecution,
    ExecutionMetrics,
    SuccessRateMetrics,
    DurationMetrics,
)
from truthound.checkpoint.analytics.stores import (
    InMemoryTimeSeriesStore,
    SQLiteTimeSeriesStore,
)
from truthound.checkpoint.analytics.analyzers import (
    SimpleTrendAnalyzer,
    AnomalyDetector,
    SimpleForecaster,
)
from truthound.checkpoint.analytics.aggregations import (
    TimeBucketAggregation,
    RollupAggregation,
    RollupLevel,
)

logger = logging.getLogger(__name__)


@dataclass
class AnalyticsConfig:
    """Configuration for the analytics service.

    Attributes:
        enabled: Whether analytics is enabled.
        store_type: Type of time series store to use.
        sqlite_path: Path for SQLite store.
        retention_days: How long to keep data.
        default_granularity: Default time bucket granularity.
        anomaly_threshold: Z-score threshold for anomaly detection.
        auto_rollup: Whether to automatically compute rollups.
    """

    enabled: bool = True
    store_type: str = "memory"  # "memory", "sqlite", "timescale"
    sqlite_path: str = "analytics.db"
    timescale_dsn: str = ""
    retention_days: int = 90
    default_granularity: TimeGranularity = TimeGranularity.HOUR
    anomaly_threshold: float = 2.0
    auto_rollup: bool = True
    forecast_periods: int = 7


class AnalyticsService:
    """Facade for the analytics system.

    Provides a unified interface for storing checkpoint execution data,
    analyzing trends, detecting anomalies, and forecasting.

    Example:
        >>> service = AnalyticsService()
        >>> await service.start()
        >>>
        >>> # Record execution
        >>> await service.record_execution(execution)
        >>>
        >>> # Analyze trends
        >>> trend = await service.analyze_checkpoint_trend("my_checkpoint")
        >>> print(f"Trend: {trend.direction}, Slope: {trend.slope:.2f}")
        >>>
        >>> # Get anomalies
        >>> anomalies = await service.detect_anomalies("my_checkpoint")
        >>> for a in anomalies:
        ...     print(f"Anomaly at {a.timestamp}: {a.anomaly_type}")
        >>>
        >>> # Forecast
        >>> forecast = await service.forecast_checkpoint("my_checkpoint")
        >>> for p in forecast.predictions:
        ...     print(f"{p.timestamp}: {p.value:.2f}")
    """

    # Metric name patterns
    METRIC_SUCCESS_RATE = "checkpoint.{name}.success_rate"
    METRIC_DURATION = "checkpoint.{name}.duration_ms"
    METRIC_TASK_COUNT = "checkpoint.{name}.task_count"
    METRIC_FAILURE_COUNT = "checkpoint.{name}.failure_count"

    def __init__(
        self,
        config: AnalyticsConfig | None = None,
    ) -> None:
        """Initialize analytics service.

        Args:
            config: Service configuration.
        """
        self._config = config or AnalyticsConfig()

        # Components (lazy initialized)
        self._store: TimeSeriesStoreProtocol | None = None
        self._trend_analyzer: TrendAnalyzerProtocol | None = None
        self._anomaly_detector: AnomalyDetector | None = None
        self._forecaster: SimpleForecaster | None = None
        self._rollup: RollupAggregation | None = None

        # State
        self._started = False
        self._cleanup_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the analytics service."""
        if self._started:
            return

        if not self._config.enabled:
            logger.info("Analytics is disabled")
            return

        # Initialize store
        self._store = await self._create_store()
        await self._store.connect()

        # Initialize analyzers
        self._trend_analyzer = SimpleTrendAnalyzer()
        self._anomaly_detector = AnomalyDetector(
            threshold=self._config.anomaly_threshold,
        )
        self._forecaster = SimpleForecaster(
            method="holt",
        )

        # Initialize rollup
        if self._config.auto_rollup:
            self._rollup = RollupAggregation(use_defaults=True)

        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        self._started = True
        logger.info("Analytics service started")

    async def stop(self) -> None:
        """Stop the analytics service."""
        if not self._started:
            return

        # Stop cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

        # Disconnect store
        if self._store:
            await self._store.disconnect()

        self._started = False
        logger.info("Analytics service stopped")

    async def _create_store(self) -> TimeSeriesStoreProtocol:
        """Create the time series store based on config."""
        store_type = self._config.store_type.lower()

        if store_type == "sqlite":
            return SQLiteTimeSeriesStore(
                db_path=self._config.sqlite_path,
            )
        else:
            return InMemoryTimeSeriesStore()

    async def _cleanup_loop(self) -> None:
        """Background loop for cleaning up old data."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour

                cutoff = datetime.now() - timedelta(days=self._config.retention_days)

                # Clean up store data (implementation-specific)
                if hasattr(self._store, "cleanup"):
                    await self._store.cleanup(before=cutoff)

                # Clean up rollup data
                if self._rollup:
                    removed = self._rollup.cleanup_expired()
                    logger.debug(f"Rollup cleanup: {removed}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")

    async def record_execution(
        self,
        execution: CheckpointExecution,
    ) -> None:
        """Record a checkpoint execution.

        Args:
            execution: Execution to record.
        """
        if not self._store:
            return

        timestamp = execution.end_time or execution.start_time
        labels = {
            "checkpoint": execution.checkpoint_name,
            "status": execution.status,
        }

        # Record success rate (1.0 for success, 0.0 for failure)
        success_value = 1.0 if execution.status == "completed" else 0.0
        await self._store.write(
            self.METRIC_SUCCESS_RATE.format(name=execution.checkpoint_name),
            TimeSeriesPoint(
                timestamp=timestamp,
                value=success_value,
                labels=labels,
            ),
        )

        # Record duration
        if execution.duration_ms:
            await self._store.write(
                self.METRIC_DURATION.format(name=execution.checkpoint_name),
                TimeSeriesPoint(
                    timestamp=timestamp,
                    value=execution.duration_ms,
                    labels=labels,
                ),
            )

        # Record task counts
        if execution.metrics:
            await self._store.write(
                self.METRIC_TASK_COUNT.format(name=execution.checkpoint_name),
                TimeSeriesPoint(
                    timestamp=timestamp,
                    value=float(execution.metrics.total_tasks),
                    labels=labels,
                ),
            )

            await self._store.write(
                self.METRIC_FAILURE_COUNT.format(name=execution.checkpoint_name),
                TimeSeriesPoint(
                    timestamp=timestamp,
                    value=float(execution.metrics.failed_tasks),
                    labels=labels,
                ),
            )

        # Add to rollup
        if self._rollup:
            self._rollup.add_points(
                [TimeSeriesPoint(
                    timestamp=timestamp,
                    value=execution.duration_ms or 0,
                    labels=labels,
                )],
                level=RollupLevel.REALTIME,
            )

    async def get_success_rate(
        self,
        checkpoint_name: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> SuccessRateMetrics | None:
        """Get success rate metrics for a checkpoint.

        Args:
            checkpoint_name: Name of the checkpoint.
            start: Start time (default: 24 hours ago).
            end: End time (default: now).

        Returns:
            Success rate metrics or None if no data.
        """
        if not self._store:
            return None

        end = end or datetime.now()
        start = start or (end - timedelta(days=1))

        metric_name = self.METRIC_SUCCESS_RATE.format(name=checkpoint_name)
        points = await self._store.query(metric_name, start, end)

        if not points:
            return None

        values = [p.value for p in points]
        success_count = sum(1 for v in values if v == 1.0)
        total_count = len(values)

        return SuccessRateMetrics(
            checkpoint_name=checkpoint_name,
            period_start=start,
            period_end=end,
            total_executions=total_count,
            successful_executions=success_count,
            failed_executions=total_count - success_count,
            success_rate=success_count / total_count if total_count > 0 else 0.0,
        )

    async def get_duration_stats(
        self,
        checkpoint_name: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> DurationMetrics | None:
        """Get duration statistics for a checkpoint.

        Args:
            checkpoint_name: Name of the checkpoint.
            start: Start time (default: 24 hours ago).
            end: End time (default: now).

        Returns:
            Duration metrics or None if no data.
        """
        if not self._store:
            return None

        end = end or datetime.now()
        start = start or (end - timedelta(days=1))

        metric_name = self.METRIC_DURATION.format(name=checkpoint_name)
        points = await self._store.query(metric_name, start, end)

        if not points:
            return None

        values = sorted([p.value for p in points])

        return DurationMetrics(
            checkpoint_name=checkpoint_name,
            period_start=start,
            period_end=end,
            sample_count=len(values),
            min_duration_ms=min(values),
            max_duration_ms=max(values),
            avg_duration_ms=sum(values) / len(values),
            p50_duration_ms=self._percentile(values, 50),
            p95_duration_ms=self._percentile(values, 95),
            p99_duration_ms=self._percentile(values, 99),
        )

    def _percentile(self, values: list[float], p: float) -> float:
        """Compute percentile of sorted values."""
        if not values:
            return 0.0
        index = (p / 100) * (len(values) - 1)
        lower = int(index)
        upper = min(lower + 1, len(values) - 1)
        weight = index - lower
        return values[lower] * (1 - weight) + values[upper] * weight

    async def analyze_checkpoint_trend(
        self,
        checkpoint_name: str,
        metric_type: str = "duration",
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> TrendResult | None:
        """Analyze trend for a checkpoint.

        Args:
            checkpoint_name: Name of the checkpoint.
            metric_type: "duration" or "success_rate".
            start: Start time (default: 7 days ago).
            end: End time (default: now).

        Returns:
            Trend analysis result or None if no data.
        """
        if not self._store or not self._trend_analyzer:
            return None

        end = end or datetime.now()
        start = start or (end - timedelta(days=7))

        if metric_type == "success_rate":
            metric_name = self.METRIC_SUCCESS_RATE.format(name=checkpoint_name)
        else:
            metric_name = self.METRIC_DURATION.format(name=checkpoint_name)

        points = await self._store.query(metric_name, start, end)

        if not points:
            return None

        return self._trend_analyzer.analyze(points)

    async def detect_anomalies(
        self,
        checkpoint_name: str,
        metric_type: str = "duration",
        start: datetime | None = None,
        end: datetime | None = None,
        threshold: float | None = None,
    ) -> list[AnomalyResult]:
        """Detect anomalies in checkpoint execution.

        Args:
            checkpoint_name: Name of the checkpoint.
            metric_type: "duration" or "success_rate".
            start: Start time (default: 7 days ago).
            end: End time (default: now).
            threshold: Optional Z-score threshold override.

        Returns:
            List of detected anomalies.
        """
        if not self._store or not self._anomaly_detector:
            return []

        end = end or datetime.now()
        start = start or (end - timedelta(days=7))

        if metric_type == "success_rate":
            metric_name = self.METRIC_SUCCESS_RATE.format(name=checkpoint_name)
        else:
            metric_name = self.METRIC_DURATION.format(name=checkpoint_name)

        points = await self._store.query(metric_name, start, end)

        if not points:
            return []

        detector = self._anomaly_detector
        if threshold:
            detector = AnomalyDetector(threshold=threshold)

        return detector.detect_anomalies(points)

    async def forecast_checkpoint(
        self,
        checkpoint_name: str,
        metric_type: str = "duration",
        periods: int | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> ForecastResult | None:
        """Forecast future checkpoint metrics.

        Args:
            checkpoint_name: Name of the checkpoint.
            metric_type: "duration" or "success_rate".
            periods: Number of periods to forecast.
            start: Start time for historical data.
            end: End time for historical data.

        Returns:
            Forecast result or None if insufficient data.
        """
        if not self._store or not self._forecaster:
            return None

        end = end or datetime.now()
        start = start or (end - timedelta(days=30))
        periods = periods or self._config.forecast_periods

        if metric_type == "success_rate":
            metric_name = self.METRIC_SUCCESS_RATE.format(name=checkpoint_name)
        else:
            metric_name = self.METRIC_DURATION.format(name=checkpoint_name)

        points = await self._store.query(metric_name, start, end)

        if not points:
            return None

        return self._forecaster.forecast(points, periods=periods)

    async def get_aggregated_metrics(
        self,
        checkpoint_name: str,
        metric_type: str = "duration",
        start: datetime | None = None,
        end: datetime | None = None,
        granularity: TimeGranularity | None = None,
        aggregation: AggregationFunction = AggregationFunction.AVG,
    ) -> list[TimeSeriesPoint]:
        """Get aggregated metrics for a checkpoint.

        Args:
            checkpoint_name: Name of the checkpoint.
            metric_type: "duration" or "success_rate".
            start: Start time.
            end: End time.
            granularity: Time bucket granularity.
            aggregation: Aggregation function.

        Returns:
            List of aggregated time series points.
        """
        if not self._store:
            return []

        end = end or datetime.now()
        start = start or (end - timedelta(days=7))
        granularity = granularity or self._config.default_granularity

        if metric_type == "success_rate":
            metric_name = self.METRIC_SUCCESS_RATE.format(name=checkpoint_name)
        else:
            metric_name = self.METRIC_DURATION.format(name=checkpoint_name)

        return await self._store.aggregate(
            metric_name,
            start,
            end,
            aggregation=aggregation.value,
            granularity=granularity,
        )

    async def get_dashboard_summary(
        self,
        checkpoint_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Get summary data suitable for a dashboard.

        Args:
            checkpoint_names: List of checkpoint names (all if None).

        Returns:
            Dashboard summary dictionary.
        """
        if not self._store:
            return {}

        end = datetime.now()
        start_24h = end - timedelta(hours=24)
        start_7d = end - timedelta(days=7)

        # Get all unique checkpoints if not specified
        if checkpoint_names is None:
            checkpoint_names = await self._get_checkpoint_names()

        summary = {
            "timestamp": end.isoformat(),
            "checkpoints": {},
        }

        for name in checkpoint_names:
            checkpoint_summary = {}

            # Success rate (24h)
            success_rate = await self.get_success_rate(name, start_24h, end)
            if success_rate:
                checkpoint_summary["success_rate_24h"] = {
                    "rate": success_rate.success_rate,
                    "total": success_rate.total_executions,
                    "failed": success_rate.failed_executions,
                }

            # Duration stats (24h)
            duration = await self.get_duration_stats(name, start_24h, end)
            if duration:
                checkpoint_summary["duration_24h"] = {
                    "avg_ms": duration.avg_duration_ms,
                    "p95_ms": duration.p95_duration_ms,
                    "min_ms": duration.min_duration_ms,
                    "max_ms": duration.max_duration_ms,
                }

            # Trend (7d)
            trend = await self.analyze_checkpoint_trend(name, "duration", start_7d, end)
            if trend:
                checkpoint_summary["trend_7d"] = {
                    "direction": trend.direction.value,
                    "slope": trend.slope,
                    "confidence": trend.confidence,
                    "change_percent": trend.change_percent,
                }

            # Anomalies (7d)
            anomalies = await self.detect_anomalies(name, "duration", start_7d, end)
            checkpoint_summary["anomalies_7d"] = {
                "count": len(anomalies),
                "latest": [
                    {
                        "timestamp": a.timestamp.isoformat(),
                        "type": a.anomaly_type.value,
                        "severity": a.severity,
                    }
                    for a in anomalies[-5:]
                ],
            }

            summary["checkpoints"][name] = checkpoint_summary

        return summary

    async def _get_checkpoint_names(self) -> list[str]:
        """Get all unique checkpoint names from stored data."""
        if not self._store:
            return []

        # Query a broad time range
        end = datetime.now()
        start = end - timedelta(days=30)

        # This is a simplified implementation
        # A real implementation would query the store for unique label values
        names: set[str] = set()

        # Try to get names from success rate metrics
        # Note: This requires the store to support label queries
        if hasattr(self._store, "get_metric_names"):
            metric_names = await self._store.get_metric_names()
            for name in metric_names:
                if name.startswith("checkpoint.") and name.endswith(".success_rate"):
                    checkpoint_name = name[11:-13]  # Extract checkpoint name
                    names.add(checkpoint_name)

        return list(names)

    async def health_check(self) -> dict[str, Any]:
        """Check health of analytics service.

        Returns:
            Health check results.
        """
        results = {
            "status": "healthy",
            "started": self._started,
            "timestamp": datetime.now().isoformat(),
        }

        if not self._started:
            results["status"] = "not_started"
            return results

        # Check store connection
        if self._store:
            try:
                if hasattr(self._store, "health_check"):
                    store_healthy = await self._store.health_check()
                    results["store"] = {
                        "connected": self._store.is_connected,
                        "healthy": store_healthy,
                    }
                    if not store_healthy:
                        results["status"] = "degraded"
            except Exception as e:
                results["store"] = {
                    "connected": False,
                    "error": str(e),
                }
                results["status"] = "degraded"

        # Rollup stats
        if self._rollup:
            results["rollup"] = self._rollup.get_stats()

        return results

    @property
    def is_started(self) -> bool:
        """Check if service is started."""
        return self._started


# Global service instance
_analytics_service: AnalyticsService | None = None


def get_analytics_service() -> AnalyticsService:
    """Get the global analytics service instance.

    Returns:
        Global AnalyticsService instance.
    """
    global _analytics_service
    if _analytics_service is None:
        _analytics_service = AnalyticsService()
    return _analytics_service


def configure_analytics(config: AnalyticsConfig) -> AnalyticsService:
    """Configure and return the global analytics service.

    Args:
        config: Service configuration.

    Returns:
        Configured AnalyticsService instance.
    """
    global _analytics_service
    _analytics_service = AnalyticsService(config)
    return _analytics_service
