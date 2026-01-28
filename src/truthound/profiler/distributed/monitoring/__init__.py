"""Distributed Profiler Monitoring System.

This module provides comprehensive monitoring for distributed profiling operations,
including task tracking, progress aggregation, health monitoring, and metrics collection.

Key Components:
- DistributedMonitor: Main orchestrator for distributed monitoring
- TaskTracker: Tracks individual partition/task execution
- ProgressAggregator: Aggregates progress from multiple workers
- MetricsCollector: Collects performance and resource metrics
- HealthMonitor: Monitors worker health and availability

Example:
    from truthound.profiler.distributed.monitoring import (
        DistributedMonitor,
        MonitorConfig,
    )

    # Create monitor with configuration
    monitor = DistributedMonitor(
        config=MonitorConfig(
            enable_health_check=True,
            metrics_interval_seconds=5.0,
        )
    )

    # Attach to distributed profiler
    profiler = DistributedProfiler.create(backend="dask")
    profiler.set_monitor(monitor)

    # Profile with monitoring
    result = profiler.profile(data)

    # Access monitoring data
    print(monitor.get_summary())
"""

from __future__ import annotations

# Core protocols and types
from truthound.profiler.distributed.monitoring.protocols import (
    IMonitorCallback,
    ITaskTracker,
    IProgressAggregator,
    IHealthMonitor,
    IMetricsCollector,
    MonitorEvent,
    MonitorEventType,
    TaskState,
    TaskInfo,
    WorkerHealth,
    HealthStatus,
    MonitorMetrics,
    AggregatedProgress,
)

# Configuration
from truthound.profiler.distributed.monitoring.config import (
    MonitorConfig,
    TaskTrackerConfig,
    HealthCheckConfig,
    MetricsConfig,
    CallbackConfig,
)

# Core implementations
from truthound.profiler.distributed.monitoring.monitor import DistributedMonitor
from truthound.profiler.distributed.monitoring.task_tracker import TaskTracker
from truthound.profiler.distributed.monitoring.progress_aggregator import (
    DistributedProgressAggregator,
)
from truthound.profiler.distributed.monitoring.health_monitor import (
    WorkerHealthMonitor,
)
from truthound.profiler.distributed.monitoring.metrics_collector import (
    DistributedMetricsCollector,
)

# Callback adapters
from truthound.profiler.distributed.monitoring.callbacks import (
    MonitorCallbackAdapter,
    ConsoleMonitorCallback,
    LoggingMonitorCallback,
    WebhookMonitorCallback,
    CallbackChain as MonitorCallbackChain,
)

# Factory and registry
from truthound.profiler.distributed.monitoring.factory import (
    MonitorFactory,
    MonitorRegistry,
)

__all__ = [
    # Protocols
    "IMonitorCallback",
    "ITaskTracker",
    "IProgressAggregator",
    "IHealthMonitor",
    "IMetricsCollector",
    # Types
    "MonitorEvent",
    "MonitorEventType",
    "TaskState",
    "TaskInfo",
    "WorkerHealth",
    "HealthStatus",
    "MonitorMetrics",
    "AggregatedProgress",
    # Configuration
    "MonitorConfig",
    "TaskTrackerConfig",
    "HealthCheckConfig",
    "MetricsConfig",
    "CallbackConfig",
    # Core implementations
    "DistributedMonitor",
    "TaskTracker",
    "DistributedProgressAggregator",
    "WorkerHealthMonitor",
    "DistributedMetricsCollector",
    # Callbacks
    "MonitorCallbackAdapter",
    "ConsoleMonitorCallback",
    "LoggingMonitorCallback",
    "WebhookMonitorCallback",
    "MonitorCallbackChain",
    # Factory
    "MonitorFactory",
    "MonitorRegistry",
]
