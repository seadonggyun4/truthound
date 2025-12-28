"""Advanced Timeout Management for Enterprise Validation.

This package provides comprehensive timeout management capabilities:
- OpenTelemetry integration for distributed tracing
- Performance prediction models for execution time estimation
- Adaptive sampling based on data characteristics
- Priority-based execution for critical validators
- Retry/rollback policies with exponential backoff
- SLA monitoring with alerting
- Redis integration for distributed coordination
- Circuit breaker pattern for resilience

Architecture:
    +------------------------------------------------------------------------+
    |                    Advanced Timeout Framework                           |
    +------------------------------------------------------------------------+
                                        |
    +----------+---------+---------+---------+---------+---------+---------+
    |          |         |         |         |         |         |         |
    v          v         v         v         v         v         v         v
+-------+ +-------+ +-------+ +-------+ +-------+ +-------+ +-------+ +--------+
|OpenTel| |Predict| |Sample | |Prior- | |Retry  | | SLA   | |Redis  | |Circuit |
|emetry | |  ion  | |  er   | |  ity  | |Policy | |Monitor| |Backend| |Breaker |
+-------+ +-------+ +-------+ +-------+ +-------+ +-------+ +-------+ +--------+

Usage:
    from truthound.validators.timeout.advanced import (
        # OpenTelemetry integration
        TelemetryProvider,
        TracingConfig,
        create_tracer,

        # Performance prediction
        PerformancePredictor,
        ExecutionHistory,
        predict_execution_time,

        # Adaptive sampling
        AdaptiveSampler,
        SamplingStrategy,
        calculate_sample_size,

        # Priority execution
        PriorityExecutor,
        ValidationPriority,
        execute_by_priority,

        # Retry/rollback
        RetryPolicy,
        RollbackManager,
        with_retry,

        # SLA monitoring
        SLAMonitor,
        SLADefinition,
        check_sla_compliance,

        # Redis integration
        RedisCoordinator,
        RedisConfig,
        create_redis_coordinator,

        # Circuit breaker
        CircuitBreaker,
        CircuitState,
        with_circuit_breaker,
    )
"""

from __future__ import annotations

# OpenTelemetry integration
from truthound.validators.timeout.advanced.telemetry import (
    TelemetryProvider,
    TracingConfig,
    MetricsConfig,
    SpanContext,
    TracingSpan,
    MetricsCollector,
    TelemetryExporter,
    ConsoleExporter,
    OTLPExporter,
    create_tracer,
    create_metrics_collector,
    trace_operation,
)

# Performance prediction
from truthound.validators.timeout.advanced.prediction import (
    PerformancePredictor,
    ExecutionHistory,
    PredictionResult,
    PredictionModel,
    MovingAverageModel,
    ExponentialSmoothingModel,
    QuantileRegressionModel,
    predict_execution_time,
    record_execution,
)

# Adaptive sampling
from truthound.validators.timeout.advanced.sampling import (
    AdaptiveSampler,
    SamplingStrategy,
    SamplingResult,
    DataCharacteristics,
    UniformSampling,
    StratifiedSampling,
    ReservoirSampling,
    calculate_sample_size,
    auto_sample,
)

# Priority execution
from truthound.validators.timeout.advanced.priority import (
    PriorityExecutor,
    ValidationPriority,
    PriorityQueue,
    ExecutionResult,
    PriorityConfig,
    execute_by_priority,
    create_priority_executor,
)

# Retry/rollback policies
from truthound.validators.timeout.advanced.retry import (
    RetryPolicy,
    RollbackManager,
    RetryResult,
    BackoffStrategy,
    ExponentialBackoff,
    LinearBackoff,
    DecorrelatedJitter,
    with_retry,
    create_retry_policy,
)

# SLA monitoring
from truthound.validators.timeout.advanced.sla import (
    SLAMonitor,
    SLADefinition,
    SLAViolation,
    SLAMetrics,
    SLAAlert,
    AlertChannel,
    check_sla_compliance,
    create_sla_monitor,
)

# Redis integration
from truthound.validators.timeout.advanced.redis_backend import (
    RedisCoordinator,
    RedisConfig,
    RedisLock,
    RedisDeadline,
    create_redis_coordinator,
    is_redis_available,
)

# Circuit breaker
from truthound.validators.timeout.advanced.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    CircuitConfig,
    CircuitMetrics,
    HalfOpenPolicy,
    with_circuit_breaker,
    create_circuit_breaker,
)

__all__ = [
    # Telemetry
    "TelemetryProvider",
    "TracingConfig",
    "MetricsConfig",
    "SpanContext",
    "TracingSpan",
    "MetricsCollector",
    "TelemetryExporter",
    "ConsoleExporter",
    "OTLPExporter",
    "create_tracer",
    "create_metrics_collector",
    "trace_operation",
    # Prediction
    "PerformancePredictor",
    "ExecutionHistory",
    "PredictionResult",
    "PredictionModel",
    "MovingAverageModel",
    "ExponentialSmoothingModel",
    "QuantileRegressionModel",
    "predict_execution_time",
    "record_execution",
    # Sampling
    "AdaptiveSampler",
    "SamplingStrategy",
    "SamplingResult",
    "DataCharacteristics",
    "UniformSampling",
    "StratifiedSampling",
    "ReservoirSampling",
    "calculate_sample_size",
    "auto_sample",
    # Priority
    "PriorityExecutor",
    "ValidationPriority",
    "PriorityQueue",
    "ExecutionResult",
    "PriorityConfig",
    "execute_by_priority",
    "create_priority_executor",
    # Retry
    "RetryPolicy",
    "RollbackManager",
    "RetryResult",
    "BackoffStrategy",
    "ExponentialBackoff",
    "LinearBackoff",
    "DecorrelatedJitter",
    "with_retry",
    "create_retry_policy",
    # SLA
    "SLAMonitor",
    "SLADefinition",
    "SLAViolation",
    "SLAMetrics",
    "SLAAlert",
    "AlertChannel",
    "check_sla_compliance",
    "create_sla_monitor",
    # Redis
    "RedisCoordinator",
    "RedisConfig",
    "RedisLock",
    "RedisDeadline",
    "create_redis_coordinator",
    "is_redis_available",
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitState",
    "CircuitConfig",
    "CircuitMetrics",
    "HalfOpenPolicy",
    "with_circuit_breaker",
    "create_circuit_breaker",
]
