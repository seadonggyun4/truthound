"""ML Model Monitoring and Observability.

Provides comprehensive monitoring for ML models:
- Performance metrics (latency, throughput)
- Drift detection (feature, prediction drift)
- Quality metrics (accuracy, F1, etc.)
- Alerting and dashboards
"""

from truthound.ml.monitoring.protocols import (
    # Protocols
    IMetricCollector,
    IMetricStore,
    IAlertRule,
    IAlertHandler,
    IModelMonitor,
    # Data classes
    ModelMetrics,
    PredictionRecord,
    DashboardData,
    Alert,
    AlertSeverity,
)

from truthound.ml.monitoring.collectors import (
    PerformanceCollector,
    DriftCollector,
    QualityCollector,
    CompositeCollector,
)

from truthound.ml.monitoring.stores import (
    InMemoryMetricStore,
    PrometheusMetricStore,
)

from truthound.ml.monitoring.alerting import (
    AlertRule,
    ThresholdRule,
    AnomalyRule,
    TrendRule,
    RuleEngine,
    SlackAlertHandler,
    PagerDutyAlertHandler,
    WebhookAlertHandler,
)

from truthound.ml.monitoring.monitor import (
    ModelMonitor,
    MonitoringPipeline,
    MonitorConfig,
)

__all__ = [
    # Protocols
    "IMetricCollector",
    "IMetricStore",
    "IAlertRule",
    "IAlertHandler",
    "IModelMonitor",
    # Data classes
    "ModelMetrics",
    "PredictionRecord",
    "DashboardData",
    "Alert",
    "AlertSeverity",
    # Collectors
    "PerformanceCollector",
    "DriftCollector",
    "QualityCollector",
    "CompositeCollector",
    # Stores
    "InMemoryMetricStore",
    "PrometheusMetricStore",
    # Alerting
    "AlertRule",
    "ThresholdRule",
    "AnomalyRule",
    "TrendRule",
    "RuleEngine",
    "SlackAlertHandler",
    "PagerDutyAlertHandler",
    "WebhookAlertHandler",
    # Monitor
    "ModelMonitor",
    "MonitoringPipeline",
    "MonitorConfig",
]
