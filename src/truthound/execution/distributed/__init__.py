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

from truthound.execution.distributed.protocols import (
    DistributedBackendProtocol,
    DistributedAggregation,
    PartitionStrategy,
    ExecutionMode,
    DistributedResult,
    PartitionInfo,
    ComputeBackend,
    AggregationScope,
    AggregationSpec,
    BaseAggregator,
    CountAggregator,
    SumAggregator,
    MeanAggregator,
    StdAggregator,
    MinMaxAggregator,
    NullCountAggregator,
    DistinctCountAggregator,
    get_aggregator,
    register_aggregator,
)
from truthound.execution.distributed.base import (
    BaseDistributedEngine,
    DistributedEngineConfig,
    ExecutionMetrics,
)
from truthound.execution.distributed.spark_engine import (
    SparkExecutionEngine,
    SparkEngineConfig,
)
from truthound.execution.distributed.dask_engine import (
    DaskExecutionEngine,
    DaskEngineConfig,
)
from truthound.execution.distributed.ray_engine import (
    RayExecutionEngine,
    RayEngineConfig,
)
from truthound.execution.distributed.aggregations import (
    AggregationPlan,
    AggregationExecutor,
    DistributedAggregator,
    aggregate_distributed,
    create_stats_plan,
    create_null_count_plan,
)
from truthound.execution.distributed.arrow_bridge import (
    ArrowBridge,
    ArrowConversionStrategy,
)
from truthound.execution.distributed.registry import (
    DistributedEngineRegistry,
    get_distributed_engine,
    register_distributed_engine,
    list_distributed_engines,
    list_available_engines,
    get_engine_registry,
)
from truthound.execution.distributed.validator_adapter import (
    DistributedValidatorAdapter,
    AdapterConfig,
    ExecutionStrategy,
    validate_distributed,
    create_distributed_adapter,
)
from truthound.execution.distributed.mixins import (
    StatisticalMixin,
    DataQualityMixin,
    PartitioningMixin,
    IOOperationsMixin,
    ValidationMixin,
    FullFeaturedMixin,
    DataQualityReport,
    ValidationResult,
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
