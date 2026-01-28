"""Configuration classes for distributed monitoring.

This module provides dataclass-based configuration for all monitoring components.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskTrackerConfig:
    """Configuration for task tracking.

    Attributes:
        max_history_size: Maximum number of completed tasks to keep in history
        timeout_seconds: Task timeout in seconds (0 = no timeout)
        max_retries: Maximum retry attempts per task
        retry_delay_seconds: Delay between retries
        enable_progress_tracking: Enable per-task progress tracking
        progress_update_interval_seconds: Minimum interval between progress updates
    """

    max_history_size: int = 1000
    timeout_seconds: float = 0.0
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    enable_progress_tracking: bool = True
    progress_update_interval_seconds: float = 0.5


@dataclass
class HealthCheckConfig:
    """Configuration for health monitoring.

    Attributes:
        heartbeat_interval_seconds: Expected heartbeat interval
        heartbeat_timeout_seconds: Time before worker marked unhealthy
        stall_threshold_seconds: Time before worker marked stalled
        memory_warning_percent: Memory usage warning threshold
        memory_critical_percent: Memory usage critical threshold
        cpu_warning_percent: CPU usage warning threshold
        error_rate_warning: Error rate warning threshold
        error_rate_critical: Error rate critical threshold
        min_healthy_workers_percent: Minimum healthy workers for system health
    """

    heartbeat_interval_seconds: float = 10.0
    heartbeat_timeout_seconds: float = 30.0
    stall_threshold_seconds: float = 60.0
    memory_warning_percent: float = 80.0
    memory_critical_percent: float = 95.0
    cpu_warning_percent: float = 90.0
    error_rate_warning: float = 0.1
    error_rate_critical: float = 0.25
    min_healthy_workers_percent: float = 50.0


@dataclass
class MetricsConfig:
    """Configuration for metrics collection.

    Attributes:
        enable_metrics: Enable metrics collection
        collection_interval_seconds: Metrics collection interval
        histogram_buckets: Buckets for duration histograms
        enable_percentiles: Enable percentile calculations
        percentiles: Percentiles to calculate
        enable_rate_limiting: Enable throughput rate limiting
        max_events_per_second: Max events to process per second
        enable_prometheus: Enable Prometheus export
        prometheus_port: Prometheus metrics port
    """

    enable_metrics: bool = True
    collection_interval_seconds: float = 5.0
    histogram_buckets: list[float] = field(
        default_factory=lambda: [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]
    )
    enable_percentiles: bool = True
    percentiles: list[float] = field(default_factory=lambda: [0.5, 0.75, 0.9, 0.95, 0.99])
    enable_rate_limiting: bool = False
    max_events_per_second: int = 1000
    enable_prometheus: bool = False
    prometheus_port: int = 9090


@dataclass
class CallbackConfig:
    """Configuration for monitoring callbacks.

    Attributes:
        enable_callbacks: Enable callback system
        async_dispatch: Dispatch events asynchronously
        buffer_size: Event buffer size for async dispatch
        max_callbacks: Maximum number of callbacks
        error_handling: How to handle callback errors (ignore, log, raise)
        throttle_interval_ms: Minimum ms between callback invocations
        batch_events: Batch events before dispatch
        batch_size: Events per batch
        batch_timeout_ms: Max wait time for batching
    """

    enable_callbacks: bool = True
    async_dispatch: bool = False
    buffer_size: int = 1000
    max_callbacks: int = 10
    error_handling: str = "log"  # ignore, log, raise
    throttle_interval_ms: int = 100
    batch_events: bool = False
    batch_size: int = 10
    batch_timeout_ms: int = 1000


@dataclass
class MonitorConfig:
    """Main configuration for DistributedMonitor.

    Attributes:
        enable_monitoring: Master switch for monitoring
        enable_task_tracking: Enable task lifecycle tracking
        enable_progress_aggregation: Enable progress aggregation
        enable_health_monitoring: Enable worker health monitoring
        enable_metrics_collection: Enable metrics collection
        task_tracker: Task tracker configuration
        health_check: Health check configuration
        metrics: Metrics configuration
        callbacks: Callback configuration
        log_level: Logging level for monitor (DEBUG, INFO, WARNING, ERROR)
        metadata: Additional configuration metadata
    """

    enable_monitoring: bool = True
    enable_task_tracking: bool = True
    enable_progress_aggregation: bool = True
    enable_health_monitoring: bool = True
    enable_metrics_collection: bool = True
    task_tracker: TaskTrackerConfig = field(default_factory=TaskTrackerConfig)
    health_check: HealthCheckConfig = field(default_factory=HealthCheckConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    callbacks: CallbackConfig = field(default_factory=CallbackConfig)
    log_level: str = "INFO"
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> list[str]:
        """Validate configuration.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        if self.task_tracker.max_history_size < 0:
            errors.append("task_tracker.max_history_size must be non-negative")

        if self.task_tracker.timeout_seconds < 0:
            errors.append("task_tracker.timeout_seconds must be non-negative")

        if self.task_tracker.max_retries < 0:
            errors.append("task_tracker.max_retries must be non-negative")

        if self.health_check.heartbeat_timeout_seconds <= 0:
            errors.append("health_check.heartbeat_timeout_seconds must be positive")

        if not 0 <= self.health_check.memory_warning_percent <= 100:
            errors.append("health_check.memory_warning_percent must be 0-100")

        if not 0 <= self.health_check.memory_critical_percent <= 100:
            errors.append("health_check.memory_critical_percent must be 0-100")

        if self.metrics.collection_interval_seconds <= 0:
            errors.append("metrics.collection_interval_seconds must be positive")

        if self.callbacks.error_handling not in ("ignore", "log", "raise"):
            errors.append("callbacks.error_handling must be ignore, log, or raise")

        return errors

    @classmethod
    def minimal(cls) -> "MonitorConfig":
        """Create minimal configuration for low overhead.

        Returns:
            MonitorConfig with minimal features enabled
        """
        return cls(
            enable_monitoring=True,
            enable_task_tracking=True,
            enable_progress_aggregation=True,
            enable_health_monitoring=False,
            enable_metrics_collection=False,
            task_tracker=TaskTrackerConfig(
                max_history_size=100,
                enable_progress_tracking=False,
            ),
            callbacks=CallbackConfig(
                enable_callbacks=False,
            ),
        )

    @classmethod
    def standard(cls) -> "MonitorConfig":
        """Create standard configuration for general use.

        Returns:
            MonitorConfig with standard features
        """
        return cls()

    @classmethod
    def full(cls) -> "MonitorConfig":
        """Create full configuration with all features.

        Returns:
            MonitorConfig with all features enabled
        """
        return cls(
            enable_monitoring=True,
            enable_task_tracking=True,
            enable_progress_aggregation=True,
            enable_health_monitoring=True,
            enable_metrics_collection=True,
            task_tracker=TaskTrackerConfig(
                max_history_size=10000,
                enable_progress_tracking=True,
            ),
            health_check=HealthCheckConfig(
                heartbeat_interval_seconds=5.0,
            ),
            metrics=MetricsConfig(
                enable_metrics=True,
                enable_percentiles=True,
                enable_prometheus=True,
            ),
            callbacks=CallbackConfig(
                enable_callbacks=True,
                async_dispatch=True,
            ),
        )

    @classmethod
    def production(cls) -> "MonitorConfig":
        """Create production configuration.

        Returns:
            MonitorConfig optimized for production use
        """
        return cls(
            enable_monitoring=True,
            enable_task_tracking=True,
            enable_progress_aggregation=True,
            enable_health_monitoring=True,
            enable_metrics_collection=True,
            task_tracker=TaskTrackerConfig(
                max_history_size=5000,
                timeout_seconds=3600.0,  # 1 hour
                max_retries=3,
                enable_progress_tracking=True,
                progress_update_interval_seconds=1.0,
            ),
            health_check=HealthCheckConfig(
                heartbeat_interval_seconds=10.0,
                heartbeat_timeout_seconds=30.0,
                memory_warning_percent=80.0,
                memory_critical_percent=95.0,
            ),
            metrics=MetricsConfig(
                enable_metrics=True,
                collection_interval_seconds=10.0,
                enable_percentiles=True,
                enable_prometheus=True,
            ),
            callbacks=CallbackConfig(
                enable_callbacks=True,
                async_dispatch=True,
                error_handling="log",
                throttle_interval_ms=500,
            ),
            log_level="INFO",
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "enable_monitoring": self.enable_monitoring,
            "enable_task_tracking": self.enable_task_tracking,
            "enable_progress_aggregation": self.enable_progress_aggregation,
            "enable_health_monitoring": self.enable_health_monitoring,
            "enable_metrics_collection": self.enable_metrics_collection,
            "task_tracker": {
                "max_history_size": self.task_tracker.max_history_size,
                "timeout_seconds": self.task_tracker.timeout_seconds,
                "max_retries": self.task_tracker.max_retries,
                "retry_delay_seconds": self.task_tracker.retry_delay_seconds,
                "enable_progress_tracking": self.task_tracker.enable_progress_tracking,
                "progress_update_interval_seconds": self.task_tracker.progress_update_interval_seconds,
            },
            "health_check": {
                "heartbeat_interval_seconds": self.health_check.heartbeat_interval_seconds,
                "heartbeat_timeout_seconds": self.health_check.heartbeat_timeout_seconds,
                "stall_threshold_seconds": self.health_check.stall_threshold_seconds,
                "memory_warning_percent": self.health_check.memory_warning_percent,
                "memory_critical_percent": self.health_check.memory_critical_percent,
                "cpu_warning_percent": self.health_check.cpu_warning_percent,
                "error_rate_warning": self.health_check.error_rate_warning,
                "error_rate_critical": self.health_check.error_rate_critical,
            },
            "metrics": {
                "enable_metrics": self.metrics.enable_metrics,
                "collection_interval_seconds": self.metrics.collection_interval_seconds,
                "enable_percentiles": self.metrics.enable_percentiles,
                "enable_prometheus": self.metrics.enable_prometheus,
            },
            "callbacks": {
                "enable_callbacks": self.callbacks.enable_callbacks,
                "async_dispatch": self.callbacks.async_dispatch,
                "error_handling": self.callbacks.error_handling,
                "throttle_interval_ms": self.callbacks.throttle_interval_ms,
            },
            "log_level": self.log_level,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MonitorConfig":
        """Create from dictionary.

        Args:
            data: Configuration dictionary

        Returns:
            MonitorConfig instance
        """
        task_tracker_data = data.get("task_tracker", {})
        health_check_data = data.get("health_check", {})
        metrics_data = data.get("metrics", {})
        callbacks_data = data.get("callbacks", {})

        return cls(
            enable_monitoring=data.get("enable_monitoring", True),
            enable_task_tracking=data.get("enable_task_tracking", True),
            enable_progress_aggregation=data.get("enable_progress_aggregation", True),
            enable_health_monitoring=data.get("enable_health_monitoring", True),
            enable_metrics_collection=data.get("enable_metrics_collection", True),
            task_tracker=TaskTrackerConfig(**task_tracker_data) if task_tracker_data else TaskTrackerConfig(),
            health_check=HealthCheckConfig(**health_check_data) if health_check_data else HealthCheckConfig(),
            metrics=MetricsConfig(**metrics_data) if metrics_data else MetricsConfig(),
            callbacks=CallbackConfig(**callbacks_data) if callbacks_data else CallbackConfig(),
            log_level=data.get("log_level", "INFO"),
            metadata=data.get("metadata", {}),
        )
