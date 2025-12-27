"""Base classes for distributed execution engines.

This module provides the abstract base class that all distributed
execution engines must inherit from. It defines the common interface
and provides default implementations where possible.
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from truthound.execution._protocols import AggregationType
from truthound.execution.base import BaseExecutionEngine, ExecutionConfig
from truthound.execution.distributed.protocols import (
    AggregationScope,
    AggregationSpec,
    ComputeBackend,
    DistributedAggregation,
    DistributedBackendProtocol,
    DistributedResult,
    ExecutionMode,
    PartitionInfo,
    PartitionStrategy,
    get_aggregator,
)

if TYPE_CHECKING:
    import polars as pl
    import pyarrow as pa

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class DistributedEngineConfig(ExecutionConfig):
    """Configuration for distributed execution engines.

    Attributes:
        num_partitions: Default number of partitions (0 = auto).
        partition_strategy: Default partitioning strategy.
        execution_mode: Default execution mode.
        timeout_seconds: Operation timeout in seconds.
        retry_count: Number of retries for failed operations.
        checkpoint_enabled: Enable checkpointing for fault tolerance.
        checkpoint_dir: Directory for checkpoints.
        arrow_batch_size: Batch size for Arrow conversions.
        memory_fraction: Fraction of memory to use per partition.
        collect_metrics: Whether to collect execution metrics.
    """

    num_partitions: int = 0
    partition_strategy: PartitionStrategy = PartitionStrategy.ROW_HASH
    execution_mode: ExecutionMode = ExecutionMode.LAZY
    timeout_seconds: int = 3600
    retry_count: int = 3
    checkpoint_enabled: bool = False
    checkpoint_dir: str = ""
    arrow_batch_size: int = 65536
    memory_fraction: float = 0.6
    collect_metrics: bool = True


ConfigT = TypeVar("ConfigT", bound=DistributedEngineConfig)


# =============================================================================
# Metrics
# =============================================================================


@dataclass
class ExecutionMetrics:
    """Metrics collected during distributed execution.

    Attributes:
        operation: Operation name.
        start_time: Start timestamp.
        end_time: End timestamp.
        partitions_processed: Number of partitions processed.
        rows_processed: Total rows processed.
        bytes_processed: Total bytes processed.
        errors: List of errors encountered.
        warnings: List of warnings.
    """

    operation: str
    start_time: float = 0.0
    end_time: float = 0.0
    partitions_processed: int = 0
    rows_processed: int = 0
    bytes_processed: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        """Get duration in milliseconds."""
        return (self.end_time - self.start_time) * 1000

    @property
    def success(self) -> bool:
        """Check if operation succeeded."""
        return len(self.errors) == 0


# =============================================================================
# Abstract Base Distributed Engine
# =============================================================================


class BaseDistributedEngine(BaseExecutionEngine[ConfigT], ABC, Generic[ConfigT]):
    """Abstract base class for distributed execution engines.

    This class extends BaseExecutionEngine with distributed-specific
    functionality and abstractions for:
    - Partition-aware operations
    - Distributed aggregations
    - Arrow-based data conversion
    - Fault-tolerant execution

    Subclasses must implement:
    - _get_backend(): Return the underlying distributed backend
    - _create_from_partition(): Create engine from single partition
    - _execute_distributed(): Execute operation across partitions
    """

    engine_type: str = "distributed"

    def __init__(
        self,
        config: ConfigT | None = None,
    ) -> None:
        """Initialize the distributed engine.

        Args:
            config: Optional configuration.
        """
        super().__init__(config)
        self._lock = threading.RLock()
        self._metrics: list[ExecutionMetrics] = []
        self._initialized = False

    @classmethod
    def _default_config(cls) -> ConfigT:
        """Create default configuration."""
        return DistributedEngineConfig()  # type: ignore

    # -------------------------------------------------------------------------
    # Abstract Methods - Backend Specific
    # -------------------------------------------------------------------------

    @property
    @abstractmethod
    def backend_type(self) -> ComputeBackend:
        """Get the compute backend type."""
        pass

    @abstractmethod
    def _get_partition_count(self) -> int:
        """Get the number of data partitions."""
        pass

    @abstractmethod
    def _get_partition_info(self) -> list[PartitionInfo]:
        """Get information about all partitions."""
        pass

    @abstractmethod
    def _execute_on_partitions(
        self,
        operation: str,
        func: Any,
        columns: list[str] | None = None,
    ) -> list[DistributedResult]:
        """Execute a function on all partitions.

        Args:
            operation: Operation name for logging/metrics.
            func: Function to execute on each partition.
            columns: Columns to include (None = all).

        Returns:
            Results from all partitions.
        """
        pass

    @abstractmethod
    def _aggregate_distributed(
        self,
        spec: AggregationSpec,
    ) -> dict[str, Any]:
        """Perform distributed aggregation.

        This should use native distributed operations when possible.

        Args:
            spec: Aggregation specification.

        Returns:
            Aggregated results.
        """
        pass

    @abstractmethod
    def _to_arrow_batches(
        self,
        batch_size: int | None = None,
    ) -> list["pa.RecordBatch"]:
        """Convert distributed data to Arrow record batches.

        This should use native Arrow conversion when available
        to minimize serialization overhead.

        Args:
            batch_size: Batch size for conversion.

        Returns:
            List of Arrow record batches.
        """
        pass

    @abstractmethod
    def _repartition(self, num_partitions: int) -> "BaseDistributedEngine":
        """Repartition the underlying data.

        Args:
            num_partitions: New number of partitions.

        Returns:
            New engine with repartitioned data.
        """
        pass

    # -------------------------------------------------------------------------
    # Implemented Methods - Core Operations (Distributed)
    # -------------------------------------------------------------------------

    def count_rows(self) -> int:
        """Count total rows using distributed aggregation."""
        cache_key = self._cache_key("count_rows")
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        spec = AggregationSpec(scope=AggregationScope.GLOBAL)
        spec.add("*", "count", alias="total_count")

        result = self._aggregate_distributed(spec)
        count = result.get("total_count", 0)

        self._set_cached(cache_key, count)
        return count

    def get_columns(self) -> list[str]:
        """Get column names from partition info."""
        partitions = self._get_partition_info()
        if partitions:
            return list(partitions[0].columns)
        return []

    def count_nulls(self, column: str) -> int:
        """Count nulls in a column using distributed aggregation."""
        cache_key = self._cache_key("count_nulls", column)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        spec = AggregationSpec(scope=AggregationScope.COLUMN)
        spec.add(column, "null_count", alias="null_count")

        result = self._aggregate_distributed(spec)
        count = result.get("null_count", {}).get("null_count", 0)

        self._set_cached(cache_key, count)
        return count

    def count_nulls_all(self) -> dict[str, int]:
        """Count nulls in all columns using distributed aggregation."""
        cache_key = self._cache_key("count_nulls_all")
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        columns = self.get_columns()
        spec = AggregationSpec(scope=AggregationScope.COLUMN)
        for col in columns:
            spec.add(col, "null_count", alias=f"{col}_nulls")

        result = self._aggregate_distributed(spec)

        null_counts = {}
        for col in columns:
            key = f"{col}_nulls"
            if key in result:
                null_counts[col] = result[key].get("null_count", 0)
            else:
                null_counts[col] = 0

        self._set_cached(cache_key, null_counts)
        return null_counts

    def count_distinct(self, column: str) -> int:
        """Count distinct values using distributed aggregation."""
        cache_key = self._cache_key("count_distinct", column)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        spec = AggregationSpec(scope=AggregationScope.COLUMN)
        spec.add(column, "distinct_count", alias="distinct")

        result = self._aggregate_distributed(spec)
        count = result.get("distinct", 0)

        self._set_cached(cache_key, count)
        return count

    def get_stats(self, column: str) -> dict[str, Any]:
        """Get comprehensive column statistics using distributed aggregation."""
        cache_key = self._cache_key("get_stats", column)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        spec = AggregationSpec(scope=AggregationScope.COLUMN)
        spec.add(column, "count", alias="count")
        spec.add(column, "null_count", alias="null_info")
        spec.add(column, "mean", alias="mean")
        spec.add(column, "std", alias="std")
        spec.add(column, "minmax", alias="minmax")

        result = self._aggregate_distributed(spec)

        stats = {
            "count": result.get("count", 0),
            "null_count": result.get("null_info", {}).get("null_count", 0),
            "mean": result.get("mean", 0.0),
            "std": result.get("std", 0.0),
            "min": result.get("minmax", {}).get("min"),
            "max": result.get("minmax", {}).get("max"),
        }

        self._set_cached(cache_key, stats)
        return stats

    # -------------------------------------------------------------------------
    # Arrow Conversion - Zero-Copy Path
    # -------------------------------------------------------------------------

    def to_polars_lazyframe(self) -> "pl.LazyFrame":
        """Convert to Polars LazyFrame via Arrow zero-copy.

        This method uses Arrow as the intermediate format to
        minimize serialization overhead.

        Returns:
            Polars LazyFrame.
        """
        import polars as pl
        import pyarrow as pa

        # Get Arrow batches from distributed data
        batches = self._to_arrow_batches(
            batch_size=self._config.arrow_batch_size
        )

        if not batches:
            # Return empty LazyFrame with correct schema
            columns = self.get_columns()
            return pl.DataFrame({col: [] for col in columns}).lazy()

        # Combine batches into table
        table = pa.Table.from_batches(batches)

        # Convert to Polars (zero-copy when possible)
        df = pl.from_arrow(table)

        return df.lazy()

    def to_arrow_table(self) -> "pa.Table":
        """Convert to PyArrow Table.

        Returns:
            PyArrow Table.
        """
        import pyarrow as pa

        batches = self._to_arrow_batches(
            batch_size=self._config.arrow_batch_size
        )

        if not batches:
            return pa.table({})

        return pa.Table.from_batches(batches)

    # -------------------------------------------------------------------------
    # Aggregation Interface
    # -------------------------------------------------------------------------

    def aggregate(
        self,
        aggregations: dict[str, AggregationType],
    ) -> dict[str, Any]:
        """Perform multiple aggregations using distributed execution.

        This overrides the base class implementation to use
        distributed aggregation.

        Args:
            aggregations: Mapping of column to aggregation type.

        Returns:
            Aggregated results.
        """
        # Convert to distributed aggregation spec
        agg_type_to_op = {
            AggregationType.COUNT: "count",
            AggregationType.SUM: "sum",
            AggregationType.MEAN: "mean",
            AggregationType.MEDIAN: "median",  # Note: Expensive distributed op
            AggregationType.MIN: "minmax",
            AggregationType.MAX: "minmax",
            AggregationType.STD: "std",
            AggregationType.VAR: "var",
            AggregationType.COUNT_DISTINCT: "distinct_count",
            AggregationType.NULL_COUNT: "null_count",
        }

        spec = AggregationSpec(scope=AggregationScope.GLOBAL)
        for col, agg_type in aggregations.items():
            op = agg_type_to_op.get(agg_type, "count")
            spec.add(col, op, alias=f"{col}_{agg_type.value}")

        results = self._aggregate_distributed(spec)

        # Post-process results
        final_results = {}
        for col, agg_type in aggregations.items():
            key = f"{col}_{agg_type.value}"
            value = results.get(key)

            # Handle minmax splitting
            if agg_type == AggregationType.MIN:
                value = value.get("min") if isinstance(value, dict) else value
            elif agg_type == AggregationType.MAX:
                value = value.get("max") if isinstance(value, dict) else value
            elif isinstance(value, dict):
                # Extract relevant value from dict results
                if "null_count" in value:
                    value = value["null_count"]
                elif "count" in value:
                    value = value["count"]

            final_results[key] = value

        return final_results

    # -------------------------------------------------------------------------
    # Metrics Collection
    # -------------------------------------------------------------------------

    def _start_metrics(self, operation: str) -> ExecutionMetrics:
        """Start collecting metrics for an operation."""
        metrics = ExecutionMetrics(operation=operation, start_time=time.time())
        return metrics

    def _end_metrics(self, metrics: ExecutionMetrics) -> None:
        """Finish collecting metrics and store them."""
        metrics.end_time = time.time()
        if self._config.collect_metrics:
            with self._lock:
                self._metrics.append(metrics)

    def get_metrics(self) -> list[ExecutionMetrics]:
        """Get collected execution metrics."""
        with self._lock:
            return list(self._metrics)

    def clear_metrics(self) -> None:
        """Clear collected metrics."""
        with self._lock:
            self._metrics.clear()

    # -------------------------------------------------------------------------
    # Partitioning
    # -------------------------------------------------------------------------

    @property
    def num_partitions(self) -> int:
        """Get current number of partitions."""
        return self._get_partition_count()

    def repartition(self, num_partitions: int) -> "BaseDistributedEngine":
        """Repartition the data.

        Args:
            num_partitions: New number of partitions.

        Returns:
            New engine with repartitioned data.
        """
        return self._repartition(num_partitions)

    def coalesce(self, num_partitions: int) -> "BaseDistributedEngine":
        """Coalesce partitions (no shuffle).

        This is more efficient than repartition when reducing
        the number of partitions.

        Args:
            num_partitions: New number of partitions.

        Returns:
            New engine with coalesced data.
        """
        # Default implementation uses repartition
        # Subclasses should override with native coalesce
        return self._repartition(num_partitions)

    # -------------------------------------------------------------------------
    # Context Manager
    # -------------------------------------------------------------------------

    def __enter__(self) -> "BaseDistributedEngine":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.clear_cache()
        if self._config.collect_metrics:
            self.clear_metrics()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"backend={self.backend_type.value}, "
            f"partitions={self.num_partitions})"
        )
