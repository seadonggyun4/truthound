"""Job Queue Monitoring System for Truthound Checkpoints.

This module provides real-time monitoring of checkpoint job queues,
workers, and task execution with pluggable backends.

Architecture:
    MonitoringService (Facade)
         |
         +---> MetricCollector(Protocol)
         |     +---> InMemoryCollector
         |     +---> RedisCollector
         |     +---> PrometheusCollector
         |
         +---> MetricAggregator(Protocol)
         |     +---> RealtimeAggregator
         |     +---> SlidingWindowAggregator
         |
         +---> MonitoringView(Protocol)
               +---> QueueStatusView
               +---> WorkerStatusView
               +---> TaskDetailView

Usage:
    >>> from truthound.checkpoint.monitoring import (
    ...     MonitoringService,
    ...     InMemoryCollector,
    ...     RealtimeAggregator,
    ... )
    >>>
    >>> # Create monitoring service
    >>> service = MonitoringService()
    >>> service.add_collector(InMemoryCollector())
    >>>
    >>> # Get queue metrics
    >>> metrics = await service.get_queue_metrics()
    >>> for m in metrics:
    ...     print(f"{m.queue_name}: {m.pending_count} pending")
    >>>
    >>> # Subscribe to real-time updates
    >>> async for update in service.subscribe():
    ...     print(f"Update: {update}")
"""

from truthound.checkpoint.monitoring.protocols import (
    # Core protocols
    MetricCollectorProtocol,
    MetricAggregatorProtocol,
    MonitoringViewProtocol,
    # Data classes
    MetricType,
    QueueMetrics,
    WorkerMetrics,
    TaskMetrics,
    MonitoringEvent,
    MonitoringEventType,
    # Exceptions
    MonitoringError,
    CollectorError,
    AggregatorError,
)

from truthound.checkpoint.monitoring.events import (
    EventEmitter,
    EventBus,
    event_bus,
)

from truthound.checkpoint.monitoring.collectors import (
    BaseCollector,
    InMemoryCollector,
    RedisCollector,
    PrometheusCollector,
)

from truthound.checkpoint.monitoring.aggregators import (
    BaseAggregator,
    RealtimeAggregator,
    SlidingWindowAggregator,
)

from truthound.checkpoint.monitoring.views import (
    BaseView,
    QueueStatusView,
    WorkerStatusView,
    TaskDetailView,
)

from truthound.checkpoint.monitoring.service import (
    MonitoringService,
    MonitoringConfig,
    get_monitoring_service,
    configure_monitoring,
)

__all__ = [
    # Protocols
    "MetricCollectorProtocol",
    "MetricAggregatorProtocol",
    "MonitoringViewProtocol",
    # Data classes
    "MetricType",
    "QueueMetrics",
    "WorkerMetrics",
    "TaskMetrics",
    "MonitoringEvent",
    "MonitoringEventType",
    # Exceptions
    "MonitoringError",
    "CollectorError",
    "AggregatorError",
    # Events
    "EventEmitter",
    "EventBus",
    "event_bus",
    # Collectors
    "BaseCollector",
    "InMemoryCollector",
    "RedisCollector",
    "PrometheusCollector",
    # Aggregators
    "BaseAggregator",
    "RealtimeAggregator",
    "SlidingWindowAggregator",
    # Views
    "BaseView",
    "QueueStatusView",
    "WorkerStatusView",
    "TaskDetailView",
    # Service
    "MonitoringService",
    "MonitoringConfig",
    "get_monitoring_service",
    "configure_monitoring",
]
