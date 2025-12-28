"""Historical Trend Analysis System for Truthound Checkpoints.

This module provides time series analysis and trend detection
for checkpoint execution history.

Architecture:
    AnalyticsService (Facade)
         |
         +---> TimeSeriesStore(Protocol)
         |     +---> InMemoryTimeSeriesStore
         |     +---> SQLiteTimeSeriesStore
         |     +---> TimescaleDBStore
         |
         +---> TrendAnalyzer(Protocol)
         |     +---> SimpleTrendAnalyzer
         |     +---> AnomalyDetector
         |     +---> SimpleForecaster
         |
         +---> TimeAggregation
               +---> TimeBucketAggregation
               +---> RollupAggregation

Usage:
    >>> from truthound.checkpoint.analytics import (
    ...     AnalyticsService,
    ...     InMemoryTimeSeriesStore,
    ...     TrendAnalyzer,
    ... )
    >>>
    >>> # Create analytics service
    >>> service = AnalyticsService()
    >>> await service.start()
    >>>
    >>> # Record checkpoint execution
    >>> service.record_execution(
    ...     checkpoint_name="daily_check",
    ...     success=True,
    ...     duration_ms=1234.5,
    ...     issues=5,
    ... )
    >>>
    >>> # Analyze trends
    >>> trend = service.analyze_trend("daily_check", period_days=7)
    >>> print(f"Direction: {trend.direction}, Slope: {trend.slope}")
"""

from truthound.checkpoint.analytics.protocols import (
    # Core protocols
    TimeSeriesStoreProtocol,
    TrendAnalyzerProtocol,
    # Data classes
    TimeGranularity,
    AggregationFunction,
    TimeSeriesPoint,
    TrendResult,
    TrendDirection,
    AnomalyType,
    AnalysisSummary,
    AnomalyResult,
    ForecastResult,
    # Exceptions
    AnalyticsError,
    StoreError,
    AnalyzerError,
)

from truthound.checkpoint.analytics.models import (
    CheckpointExecution,
    ExecutionMetrics,
    SuccessRateMetrics,
    DurationMetrics,
)

from truthound.checkpoint.analytics.stores import (
    BaseTimeSeriesStore,
    InMemoryTimeSeriesStore,
    SQLiteTimeSeriesStore,
    TimescaleDBStore,
)

from truthound.checkpoint.analytics.analyzers import (
    BaseTrendAnalyzer,
    SimpleTrendAnalyzer,
    AnomalyDetector,
    SimpleForecaster,
)

from truthound.checkpoint.analytics.aggregations import (
    TimeBucketAggregation,
    BucketResult,
    RollupAggregation,
    RollupConfig,
    RollupLevel,
)

from truthound.checkpoint.analytics.service import (
    AnalyticsService,
    AnalyticsConfig,
    get_analytics_service,
    configure_analytics,
)

__all__ = [
    # Protocols
    "TimeSeriesStoreProtocol",
    "TrendAnalyzerProtocol",
    # Data classes
    "TimeGranularity",
    "AggregationFunction",
    "TimeSeriesPoint",
    "TrendResult",
    "TrendDirection",
    "AnomalyType",
    "AnalysisSummary",
    "AnomalyResult",
    "ForecastResult",
    # Exceptions
    "AnalyticsError",
    "StoreError",
    "AnalyzerError",
    # Models
    "CheckpointExecution",
    "ExecutionMetrics",
    "SuccessRateMetrics",
    "DurationMetrics",
    # Stores
    "BaseTimeSeriesStore",
    "InMemoryTimeSeriesStore",
    "SQLiteTimeSeriesStore",
    "TimescaleDBStore",
    # Analyzers
    "BaseTrendAnalyzer",
    "SimpleTrendAnalyzer",
    "AnomalyDetector",
    "SimpleForecaster",
    # Aggregations
    "TimeBucketAggregation",
    "BucketResult",
    "RollupAggregation",
    "RollupConfig",
    "RollupLevel",
    # Service
    "AnalyticsService",
    "AnalyticsConfig",
    "get_analytics_service",
    "configure_analytics",
]
