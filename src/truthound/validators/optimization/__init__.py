"""Optimization abstractions for validator performance.

This module provides reusable mixins for optimizing common validation patterns:

Performance Improvements:
    - GraphTraversalMixin: Tarjan's SCC O(V+E), iterative DFS
    - BatchCovarianceMixin: Incremental covariance O(n), Woodbury updates
    - VectorizedGeoMixin: SIMD-friendly Haversine, batch processing
    - LazyAggregationMixin: Polars lazy evaluation, predicate pushdown

DAG-Based Execution:
    - ValidatorDAG: Dependency graph for validators
    - ExecutionPlan: Optimized execution order with parallel levels
    - ExecutionStrategy: Sequential, Parallel, or Adaptive execution

Performance Profiling:
    - ValidatorProfiler: Main profiling interface
    - ProfilerConfig: Configuration for profiling modes
    - profile_validator: Context manager for validator profiling
    - profiled: Decorator for automatic profiling

Usage:
    from truthound.validators.optimization import (
        GraphTraversalMixin,
        BatchCovarianceMixin,
        VectorizedGeoMixin,
        LazyAggregationMixin,
        # DAG execution
        ValidatorDAG,
        ExecutionPlan,
        ParallelExecutionStrategy,
        # Profiling
        ValidatorProfiler,
        profile_validator,
    )
"""

from truthound.validators.optimization.graph import (
    GraphTraversalMixin,
    TarjanSCC,
    IterativeDFS,
    TopologicalSort,
    CycleInfo,
)
from truthound.validators.optimization.covariance import (
    BatchCovarianceMixin,
    IncrementalCovariance,
    WoodburyCovariance,
    RobustCovarianceEstimator,
)
from truthound.validators.optimization.geo import (
    VectorizedGeoMixin,
    SpatialIndexMixin,
    BoundingBox,
    DistanceUnit,
    EARTH_RADIUS,
)
from truthound.validators.optimization.aggregation import (
    LazyAggregationMixin,
    AggregationResult,
    AggregationExpressionBuilder,
    JoinStrategy,
)
from truthound.validators.optimization.orchestrator import (
    ValidatorDAG,
    ValidatorNode,
    ValidatorPhase,
    ExecutionPlan,
    ExecutionLevel,
    ExecutionResult,
    ExecutionContext,
    ExecutionStrategy,
    SequentialExecutionStrategy,
    ParallelExecutionStrategy,
    AdaptiveExecutionStrategy,
    NodeExecutionResult,
    LevelExecutionResult,
    create_execution_plan,
    execute_validators,
)
from truthound.validators.optimization.profiling import (
    # Enums
    MetricType,
    ProfilerMode,
    # Data classes
    TimingMetrics,
    MemoryMetrics,
    ThroughputMetrics,
    ValidatorMetrics,
    ExecutionSnapshot,
    ProfilingSession,
    ProfilerConfig,
    ProfileContext,
    # Main classes
    ValidatorProfiler,
    MemoryTracker,
    ProfilingReport,
    # Convenience functions
    get_default_profiler,
    set_default_profiler,
    reset_default_profiler,
    profile_validator,
    profiled,
)

__all__ = [
    # Graph algorithms
    "GraphTraversalMixin",
    "TarjanSCC",
    "IterativeDFS",
    "TopologicalSort",
    "CycleInfo",
    # Covariance optimization
    "BatchCovarianceMixin",
    "IncrementalCovariance",
    "WoodburyCovariance",
    "RobustCovarianceEstimator",
    # Geo optimization
    "VectorizedGeoMixin",
    "SpatialIndexMixin",
    "BoundingBox",
    "DistanceUnit",
    "EARTH_RADIUS",
    # Aggregation optimization
    "LazyAggregationMixin",
    "AggregationResult",
    "AggregationExpressionBuilder",
    "JoinStrategy",
    # DAG-based execution
    "ValidatorDAG",
    "ValidatorNode",
    "ValidatorPhase",
    "ExecutionPlan",
    "ExecutionLevel",
    "ExecutionResult",
    "ExecutionContext",
    "ExecutionStrategy",
    "SequentialExecutionStrategy",
    "ParallelExecutionStrategy",
    "AdaptiveExecutionStrategy",
    "NodeExecutionResult",
    "LevelExecutionResult",
    "create_execution_plan",
    "execute_validators",
    # Performance profiling
    "MetricType",
    "ProfilerMode",
    "TimingMetrics",
    "MemoryMetrics",
    "ThroughputMetrics",
    "ValidatorMetrics",
    "ExecutionSnapshot",
    "ProfilingSession",
    "ProfilerConfig",
    "ProfileContext",
    "ValidatorProfiler",
    "MemoryTracker",
    "ProfilingReport",
    "get_default_profiler",
    "set_default_profiler",
    "reset_default_profiler",
    "profile_validator",
    "profiled",
]
