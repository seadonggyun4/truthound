"""Distributed execution framework for large-scale data validation.

This module provides a unified abstraction layer for distributed computing backends
(Spark, Dask, Ray) with:
- Native operations without Polars conversion overhead
- Arrow-based zero-copy data transfer
- Distributed aggregation patterns
- Fault-tolerant execution with checkpointing

Architecture:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                     DistributedExecutionEngine                       │
    │                    (Unified Abstraction Layer)                       │
    ├─────────────────────────────────────────────────────────────────────┤
    │    SparkEngine     │     DaskEngine     │      RayEngine            │
    │  (Native Spark)    │   (Native Dask)    │    (Native Ray)           │
    ├─────────────────────────────────────────────────────────────────────┤
    │                    DistributedBackendProtocol                        │
    │           (Interface for custom backend implementations)             │
    └─────────────────────────────────────────────────────────────────────┘

Example:
    >>> from truthound.execution.distributed import SparkExecutionEngine
    >>>
    >>> # Create Spark-native execution engine
    >>> engine = SparkExecutionEngine.from_dataframe(spark_df)
    >>>
    >>> # Run distributed aggregations
    >>> null_counts = engine.count_nulls_distributed()
    >>> stats = engine.get_stats_distributed("price")
    >>>
    >>> # Convert to Polars only when needed
    >>> lf = engine.to_polars_lazyframe()  # Arrow-based zero-copy
"""

from truthound.execution.distributed.aggregations import (
    AggregationExecutor,
    AggregationPlan,
    DistributedAggregator,
    aggregate_distributed,
    create_null_count_plan,
    create_stats_plan,
)
from truthound.execution.distributed.arrow_bridge import (
    ArrowBridge,
    ArrowConversionStrategy,
)
from truthound.execution.distributed.base import (
    BaseDistributedEngine,
    DistributedEngineConfig,
    ExecutionMetrics,
)
from truthound.execution.distributed.dask_engine import (
    DaskEngineConfig,
    DaskExecutionEngine,
)
from truthound.execution.distributed.mixins import (
    DataQualityMixin,
    DataQualityReport,
    FullFeaturedMixin,
    IOOperationsMixin,
    PartitioningMixin,
    StatisticalMixin,
    ValidationMixin,
    ValidationResult,
)
from truthound.execution.distributed.policies import (
    DistributedOperationTimeout,
    RetryableExecutionError,
)
from truthound.execution.distributed.protocols import (
    AggregationScope,
    AggregationSpec,
    BaseAggregator,
    ComputeBackend,
    CountAggregator,
    DistinctCountAggregator,
    DistributedAggregation,
    DistributedBackendProtocol,
    DistributedResult,
    ExecutionMode,
    MeanAggregator,
    MinMaxAggregator,
    NullCountAggregator,
    PartitionInfo,
    PartitionStrategy,
    StdAggregator,
    SumAggregator,
    get_aggregator,
    register_aggregator,
)
from truthound.execution.distributed.ray_engine import (
    RayEngineConfig,
    RayExecutionEngine,
)
from truthound.execution.distributed.registry import (
    DistributedEngineRegistry,
    get_distributed_engine,
    get_engine_registry,
    list_available_engines,
    list_distributed_engines,
    register_distributed_engine,
)
from truthound.execution.distributed.spark_engine import (
    SparkEngineConfig,
    SparkExecutionEngine,
)
from truthound.execution.distributed.validator_adapter import (
    AdapterConfig,
    DistributedValidatorAdapter,
    ExecutionStrategy,
    create_distributed_adapter,
    validate_distributed,
)

__all__ = [
    # Protocols & Enums
    "DistributedBackendProtocol",
    "DistributedAggregation",
    "PartitionStrategy",
    "ExecutionMode",
    "DistributedResult",
    "PartitionInfo",
    "ComputeBackend",
    "AggregationScope",
    "AggregationSpec",
    # Built-in Aggregators
    "BaseAggregator",
    "CountAggregator",
    "SumAggregator",
    "MeanAggregator",
    "StdAggregator",
    "MinMaxAggregator",
    "NullCountAggregator",
    "DistinctCountAggregator",
    "get_aggregator",
    "register_aggregator",
    # Base classes
    "BaseDistributedEngine",
    "DistributedEngineConfig",
    "ExecutionMetrics",
    "DistributedOperationTimeout",
    "RetryableExecutionError",
    # Spark engine
    "SparkExecutionEngine",
    "SparkEngineConfig",
    # Dask engine
    "DaskExecutionEngine",
    "DaskEngineConfig",
    # Ray engine
    "RayExecutionEngine",
    "RayEngineConfig",
    # Aggregations
    "AggregationPlan",
    "AggregationExecutor",
    "DistributedAggregator",
    "aggregate_distributed",
    "create_stats_plan",
    "create_null_count_plan",
    # Arrow bridge
    "ArrowBridge",
    "ArrowConversionStrategy",
    # Registry
    "DistributedEngineRegistry",
    "get_distributed_engine",
    "register_distributed_engine",
    "list_distributed_engines",
    "list_available_engines",
    "get_engine_registry",
    # Validator Adapter
    "DistributedValidatorAdapter",
    "AdapterConfig",
    "ExecutionStrategy",
    "validate_distributed",
    "create_distributed_adapter",
    # Mixins
    "StatisticalMixin",
    "DataQualityMixin",
    "PartitioningMixin",
    "IOOperationsMixin",
    "ValidationMixin",
    "FullFeaturedMixin",
    "DataQualityReport",
    "ValidationResult",
]
