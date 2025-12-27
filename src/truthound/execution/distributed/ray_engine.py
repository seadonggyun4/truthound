"""Ray-native execution engine for distributed data validation.

This module provides a Ray-native execution engine that:
- Executes validation operations directly on Ray Datasets
- Avoids Polars conversion overhead for distributed operations
- Uses Arrow for efficient data transfer when conversion is needed
- Supports distributed aggregations with proper reduce semantics

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    RayExecutionEngine                           │
    │                                                                  │
    │   ┌──────────────────────────────────────────────────────────┐  │
    │   │              Native Ray Operations                       │  │
    │   │  (count, aggregate, filter - no conversion overhead)      │  │
    │   └──────────────────────────────────────────────────────────┘  │
    │                              │                                   │
    │                              ▼                                   │
    │   ┌──────────────────────────────────────────────────────────┐  │
    │   │               Arrow Bridge (when needed)                  │  │
    │   │    (zero-copy conversion to Polars for ML validators)     │  │
    │   └──────────────────────────────────────────────────────────┘  │
    │                              │                                   │
    │                              ▼                                   │
    │   ┌──────────────────────────────────────────────────────────┐  │
    │   │            Polars LazyFrame (fallback)                    │  │
    │   │  (only for validators that require Polars operations)     │  │
    │   └──────────────────────────────────────────────────────────┘  │
    │                                                                  │
    └─────────────────────────────────────────────────────────────────┘

Example:
    >>> import ray
    >>> from truthound.execution.distributed import RayExecutionEngine
    >>>
    >>> ray.init()
    >>> ds = ray.data.read_parquet("large_data.parquet")
    >>>
    >>> # Create native Ray engine
    >>> engine = RayExecutionEngine.from_dataset(ds)
    >>>
    >>> # Native Ray operations (no conversion overhead)
    >>> row_count = engine.count_rows()
    >>> null_counts = engine.count_nulls_all()
    >>> stats = engine.get_stats("price")
    >>>
    >>> # Convert to Polars only when needed (via Arrow)
    >>> lf = engine.to_polars_lazyframe()
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from functools import reduce
from typing import TYPE_CHECKING, Any, Callable, Iterator

from truthound.execution.distributed.base import (
    BaseDistributedEngine,
    DistributedEngineConfig,
    ExecutionMetrics,
)
from truthound.execution.distributed.protocols import (
    AggregationScope,
    AggregationSpec,
    ComputeBackend,
    DistributedResult,
    PartitionInfo,
    PartitionStrategy,
    get_aggregator,
)

if TYPE_CHECKING:
    import pyarrow as pa
    import ray
    from ray.data import Dataset

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class RayEngineConfig(DistributedEngineConfig):
    """Configuration for Ray execution engine.

    Attributes:
        ray_address: Ray cluster address (None = local).
        num_cpus: Number of CPUs to use.
        num_gpus: Number of GPUs to use.
        object_store_memory: Object store memory in bytes.
        batch_size: Batch size for iterating over data.
        prefetch_batches: Number of batches to prefetch.
        concurrency: Number of concurrent tasks for map operations.
        use_actors: Use actor pool for better resource utilization.
        actor_pool_size: Size of actor pool.
        target_max_block_size: Target max block size in bytes.
    """

    ray_address: str | None = None
    num_cpus: int | None = None
    num_gpus: int | None = None
    object_store_memory: int | None = None
    batch_size: int = 4096
    prefetch_batches: int = 2
    concurrency: int | None = None
    use_actors: bool = False
    actor_pool_size: int = 4
    target_max_block_size: int = 128 * 1024 * 1024  # 128MB


def _check_ray_available() -> None:
    """Check if Ray is available."""
    try:
        import ray  # noqa: F401
        import ray.data  # noqa: F401
    except ImportError:
        raise ImportError(
            "ray is required for RayExecutionEngine. "
            "Install with: pip install 'ray[data]'"
        )


def _ensure_ray_initialized(config: RayEngineConfig) -> None:
    """Ensure Ray is initialized."""
    import ray

    if not ray.is_initialized():
        init_kwargs = {}
        if config.ray_address:
            init_kwargs["address"] = config.ray_address
        if config.num_cpus:
            init_kwargs["num_cpus"] = config.num_cpus
        if config.num_gpus:
            init_kwargs["num_gpus"] = config.num_gpus
        if config.object_store_memory:
            init_kwargs["object_store_memory"] = config.object_store_memory

        ray.init(**init_kwargs)


# =============================================================================
# Ray Execution Engine
# =============================================================================


class RayExecutionEngine(BaseDistributedEngine[RayEngineConfig]):
    """Ray-native execution engine for distributed validation.

    This engine executes validation operations directly on Ray Datasets,
    avoiding the overhead of converting to Polars for operations that can
    be performed natively in Ray.

    Key Features:
    - Native Ray aggregations (count, sum, mean, min, max, etc.)
    - Distributed null/duplicate checking
    - Arrow-based zero-copy conversion to Polars when needed
    - Block-aware operations
    - Automatic scaling and fault tolerance

    Example:
        >>> engine = RayExecutionEngine.from_dataset(ray_dataset)
        >>> null_counts = engine.count_nulls_all()  # Native Ray
        >>> lf = engine.to_polars_lazyframe()  # Arrow-based conversion
    """

    engine_type = "ray"

    def __init__(
        self,
        dataset: "Dataset",
        config: RayEngineConfig | None = None,
    ) -> None:
        """Initialize Ray execution engine.

        Args:
            dataset: Ray Dataset.
            config: Optional configuration.
        """
        _check_ray_available()
        super().__init__(config)

        _ensure_ray_initialized(self._config)

        self._ds = dataset
        self._schema = dataset.schema()
        self._columns = list(self._schema.names) if self._schema else []
        self._cached_row_count: int | None = None

    @classmethod
    def _default_config(cls) -> RayEngineConfig:
        """Create default configuration."""
        return RayEngineConfig()

    # -------------------------------------------------------------------------
    # Factory Methods
    # -------------------------------------------------------------------------

    @classmethod
    def from_dataset(
        cls,
        dataset: "Dataset",
        config: RayEngineConfig | None = None,
    ) -> "RayExecutionEngine":
        """Create engine from existing Ray Dataset.

        Args:
            dataset: Ray Dataset.
            config: Optional configuration.

        Returns:
            RayExecutionEngine instance.
        """
        return cls(dataset, config)

    @classmethod
    def from_parquet(
        cls,
        path: str,
        config: RayEngineConfig | None = None,
        **read_kwargs: Any,
    ) -> "RayExecutionEngine":
        """Create engine from Parquet files.

        Args:
            path: Path to Parquet files (can use glob patterns).
            config: Optional configuration.
            **read_kwargs: Additional arguments for read_parquet.

        Returns:
            RayExecutionEngine instance.
        """
        _check_ray_available()
        import ray.data

        cfg = config or RayEngineConfig()
        _ensure_ray_initialized(cfg)

        ds = ray.data.read_parquet(path, **read_kwargs)

        return cls(ds, config)

    @classmethod
    def from_csv(
        cls,
        path: str,
        config: RayEngineConfig | None = None,
        **read_kwargs: Any,
    ) -> "RayExecutionEngine":
        """Create engine from CSV files.

        Args:
            path: Path to CSV files (can use glob patterns).
            config: Optional configuration.
            **read_kwargs: Additional arguments for read_csv.

        Returns:
            RayExecutionEngine instance.
        """
        _check_ray_available()
        import ray.data

        cfg = config or RayEngineConfig()
        _ensure_ray_initialized(cfg)

        ds = ray.data.read_csv(path, **read_kwargs)

        return cls(ds, config)

    @classmethod
    def from_pandas(
        cls,
        df: Any,
        config: RayEngineConfig | None = None,
    ) -> "RayExecutionEngine":
        """Create engine from Pandas DataFrame.

        Args:
            df: Pandas DataFrame.
            config: Optional configuration.

        Returns:
            RayExecutionEngine instance.
        """
        _check_ray_available()
        import ray.data

        cfg = config or RayEngineConfig()
        _ensure_ray_initialized(cfg)

        ds = ray.data.from_pandas(df)

        return cls(ds, config)

    @classmethod
    def from_arrow(
        cls,
        table: "pa.Table",
        config: RayEngineConfig | None = None,
    ) -> "RayExecutionEngine":
        """Create engine from Arrow Table.

        Args:
            table: PyArrow Table.
            config: Optional configuration.

        Returns:
            RayExecutionEngine instance.
        """
        _check_ray_available()
        import ray.data

        cfg = config or RayEngineConfig()
        _ensure_ray_initialized(cfg)

        ds = ray.data.from_arrow(table)

        return cls(ds, config)

    @classmethod
    def from_items(
        cls,
        items: list[dict[str, Any]],
        config: RayEngineConfig | None = None,
    ) -> "RayExecutionEngine":
        """Create engine from list of dictionaries.

        Args:
            items: List of row dictionaries.
            config: Optional configuration.

        Returns:
            RayExecutionEngine instance.
        """
        _check_ray_available()
        import ray.data

        cfg = config or RayEngineConfig()
        _ensure_ray_initialized(cfg)

        ds = ray.data.from_items(items)

        return cls(ds, config)

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def backend_type(self) -> ComputeBackend:
        """Get the compute backend type."""
        return ComputeBackend.RAY

    @property
    def dataset(self) -> "Dataset":
        """Get the underlying Ray Dataset."""
        return self._ds

    @property
    def schema(self) -> Any:
        """Get the dataset schema."""
        return self._schema

    @property
    def supports_sql_pushdown(self) -> bool:
        """Ray doesn't have native SQL pushdown."""
        return False

    # -------------------------------------------------------------------------
    # Abstract Method Implementations
    # -------------------------------------------------------------------------

    def _get_partition_count(self) -> int:
        """Get number of data blocks (partitions)."""
        return self._ds.num_blocks()

    def _get_partition_info(self) -> list[PartitionInfo]:
        """Get information about all partitions (blocks)."""
        num_blocks = self._get_partition_count()
        columns = tuple(self._columns)

        return [
            PartitionInfo(
                partition_id=i,
                total_partitions=num_blocks,
                columns=columns,
            )
            for i in range(num_blocks)
        ]

    def _execute_on_partitions(
        self,
        operation: str,
        func: Callable[[Any], dict[str, Any]],
        columns: list[str] | None = None,
    ) -> list[DistributedResult]:
        """Execute function on all blocks using map_batches.

        Args:
            operation: Operation name for metrics.
            func: Function to apply to each batch.
            columns: Columns to include (None = all).

        Returns:
            Results from all blocks.
        """
        import ray

        metrics = self._start_metrics(operation)

        try:
            ds = self._ds
            if columns:
                ds = ds.select_columns(columns)

            # Map batches - func receives batch dict
            def wrapped_func(batch: dict[str, Any]) -> dict[str, Any]:
                start_time = time.time()
                result = func(batch)
                duration_ms = (time.time() - start_time) * 1000

                # Get row count from batch
                row_count = len(next(iter(batch.values()))) if batch else 0

                return {
                    "value": [result.get("value")],
                    "row_count": [row_count],
                    "duration_ms": [duration_ms],
                    "errors": [result.get("errors", [])],
                    "metadata": [result.get("metadata", {})],
                }

            results_ds = ds.map_batches(
                wrapped_func,
                batch_format="pydict",
                batch_size=self._config.batch_size,
            )

            # Collect results
            collected = []
            for i, batch in enumerate(results_ds.iter_batches(batch_format="pydict")):
                for j in range(len(batch["value"])):
                    collected.append(
                        DistributedResult(
                            partition_id=i,
                            operation=operation,
                            value=batch["value"][j],
                            row_count=batch["row_count"][j],
                            duration_ms=batch["duration_ms"][j],
                            errors=batch["errors"][j] if batch["errors"][j] else [],
                            metadata=batch["metadata"][j] if batch["metadata"][j] else {},
                        )
                    )

            total_rows = sum(r.row_count for r in collected)
            metrics.partitions_processed = len(collected)
            metrics.rows_processed = total_rows

            return collected

        except Exception as e:
            metrics.errors.append(str(e))
            raise
        finally:
            self._end_metrics(metrics)

    def _aggregate_distributed(
        self,
        spec: AggregationSpec,
    ) -> dict[str, Any]:
        """Perform distributed aggregation using native Ray operations.

        Args:
            spec: Aggregation specification.

        Returns:
            Aggregated results.
        """
        import ray

        metrics = self._start_metrics("aggregate")

        try:
            results = {}

            for agg in spec.aggregations:
                column = agg.column
                operation = agg.operation
                alias = agg.alias
                params = agg.params

                if operation == "count":
                    if column == "*":
                        value = self._ds.count()
                    else:
                        # Count non-null values
                        value = self._count_non_null(column)
                    results[alias] = value

                elif operation == "sum":
                    value = self._ds.sum(column)
                    results[alias] = value

                elif operation == "mean":
                    value = self._ds.mean(column)
                    results[alias] = value

                elif operation == "min":
                    value = self._ds.min(column)
                    results[alias] = value

                elif operation == "max":
                    value = self._ds.max(column)
                    results[alias] = value

                elif operation == "std":
                    value = self._ds.std(column)
                    results[alias] = value

                elif operation == "var":
                    # Ray doesn't have built-in var, compute from std
                    std = self._ds.std(column)
                    value = std ** 2 if std is not None else None
                    results[alias] = value

                elif operation == "minmax":
                    min_val = self._ds.min(column)
                    max_val = self._ds.max(column)
                    results[alias] = {"min": min_val, "max": max_val}

                elif operation == "null_count":
                    null_count = self._count_nulls_column(column)
                    total_count = self._ds.count()
                    results[alias] = {
                        "null_count": null_count,
                        "total_count": total_count,
                    }

                elif operation == "distinct_count":
                    value = self._count_distinct_column(column)
                    results[alias] = value

                else:
                    # Use custom aggregator via map-reduce
                    result = self._aggregate_with_aggregator(agg)
                    results[alias] = result

            return results

        except Exception as e:
            metrics.errors.append(str(e))
            raise
        finally:
            self._end_metrics(metrics)

    def _count_non_null(self, column: str) -> int:
        """Count non-null values in a column."""
        total = self._ds.count()
        null_count = self._count_nulls_column(column)
        return total - null_count

    def _count_nulls_column(self, column: str) -> int:
        """Count null values in a column."""
        import ray

        @ray.remote
        def count_nulls_batch(batch: dict) -> int:
            values = batch.get(column, [])
            return sum(1 for v in values if v is None)

        null_counts = []
        for batch in self._ds.iter_batches(
            batch_format="pydict",
            batch_size=self._config.batch_size,
        ):
            ref = count_nulls_batch.remote(batch)
            null_counts.append(ref)

        return sum(ray.get(null_counts))

    def _count_distinct_column(self, column: str) -> int:
        """Count distinct values in a column."""
        # Use unique() which returns a dataset with unique values
        unique_ds = self._ds.unique(column)
        return unique_ds.count()

    def _aggregate_with_aggregator(
        self,
        agg: Any,
    ) -> Any:
        """Perform aggregation using custom aggregator via map-reduce.

        Args:
            agg: Aggregation specification.

        Returns:
            Aggregated result.
        """
        import ray

        aggregator = get_aggregator(agg.operation, **agg.params)
        column = agg.column

        @ray.remote
        def map_batch(batch: dict) -> Any:
            state = aggregator.initialize()
            values = batch.get(column, [])
            for value in values:
                state = aggregator.accumulate(state, value)
            return state

        # Map phase: compute partial aggregates per batch
        batch_refs = []
        for batch in self._ds.iter_batches(
            batch_format="pydict",
            batch_size=self._config.batch_size,
        ):
            ref = map_batch.remote(batch)
            batch_refs.append(ref)

        partial_states = ray.get(batch_refs)

        # Reduce phase: merge all partial states
        if not partial_states:
            return aggregator.finalize(aggregator.initialize())

        final_state = reduce(aggregator.merge, partial_states)
        return aggregator.finalize(final_state)

    def _to_arrow_batches(
        self,
        batch_size: int | None = None,
    ) -> list["pa.RecordBatch"]:
        """Convert Ray Dataset to Arrow batches.

        Ray has native Arrow support, making this efficient.

        Args:
            batch_size: Batch size for conversion.

        Returns:
            List of Arrow record batches.
        """
        import pyarrow as pa

        batch_size = batch_size or self._config.arrow_batch_size

        # Ray Dataset has native Arrow support
        batches = []
        for batch in self._ds.iter_batches(
            batch_format="pyarrow",
            batch_size=batch_size,
        ):
            if isinstance(batch, pa.RecordBatch):
                batches.append(batch)
            elif isinstance(batch, pa.Table):
                batches.extend(batch.to_batches(max_chunksize=batch_size))

        return batches

    def _repartition(self, num_partitions: int) -> "RayExecutionEngine":
        """Repartition the underlying Dataset.

        Args:
            num_partitions: New number of partitions (blocks).

        Returns:
            New engine with repartitioned data.
        """
        repartitioned = self._ds.repartition(num_partitions)
        return RayExecutionEngine(repartitioned, self._config)

    def coalesce(self, num_partitions: int) -> "RayExecutionEngine":
        """Coalesce partitions (blocks).

        Args:
            num_partitions: New number of partitions.

        Returns:
            New engine with coalesced data.
        """
        # Ray's repartition can reduce partitions without full shuffle
        coalesced = self._ds.repartition(num_partitions)
        return RayExecutionEngine(coalesced, self._config)

    # -------------------------------------------------------------------------
    # Core Operation Overrides (Native Ray)
    # -------------------------------------------------------------------------

    def count_rows(self) -> int:
        """Count rows using native Ray count."""
        if self._cached_row_count is not None:
            return self._cached_row_count

        cache_key = self._cache_key("count_rows")
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        count = self._ds.count()
        self._cached_row_count = count
        self._set_cached(cache_key, count)
        return count

    def get_columns(self) -> list[str]:
        """Get column names."""
        return self._columns

    def count_nulls(self, column: str) -> int:
        """Count nulls using distributed computation."""
        cache_key = self._cache_key("count_nulls", column)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        count = self._count_nulls_column(column)
        self._set_cached(cache_key, count)
        return count

    def count_nulls_all(self) -> dict[str, int]:
        """Count nulls in all columns."""
        import ray

        cache_key = self._cache_key("count_nulls_all")
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        # Compute null counts for all columns in parallel
        @ray.remote
        def count_batch_nulls(batch: dict, columns: list) -> dict[str, int]:
            result = {}
            for col in columns:
                values = batch.get(col, [])
                result[col] = sum(1 for v in values if v is None)
            return result

        batch_results = []
        for batch in self._ds.iter_batches(
            batch_format="pydict",
            batch_size=self._config.batch_size,
        ):
            ref = count_batch_nulls.remote(batch, self._columns)
            batch_results.append(ref)

        all_counts = ray.get(batch_results)

        # Merge results
        result = {col: 0 for col in self._columns}
        for counts in all_counts:
            for col, count in counts.items():
                result[col] += count

        self._set_cached(cache_key, result)
        return result

    def count_distinct(self, column: str) -> int:
        """Count distinct values using native Ray."""
        cache_key = self._cache_key("count_distinct", column)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        count = self._count_distinct_column(column)
        self._set_cached(cache_key, count)
        return count

    def get_stats(self, column: str) -> dict[str, Any]:
        """Get column statistics using native Ray aggregations."""
        cache_key = self._cache_key("get_stats", column)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        # Compute stats
        stats = {
            "count": self._ds.count(),
            "null_count": self._count_nulls_column(column),
            "mean": self._ds.mean(column),
            "std": self._ds.std(column),
            "min": self._ds.min(column),
            "max": self._ds.max(column),
        }

        self._set_cached(cache_key, stats)
        return stats

    def get_value_counts(
        self,
        column: str,
        limit: int | None = None,
    ) -> dict[Any, int]:
        """Get value counts."""
        import ray

        cache_key = self._cache_key("get_value_counts", column, limit)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        # Use groupby with count
        grouped = self._ds.groupby(column).count()

        # Collect and sort
        counts = {}
        for batch in grouped.iter_batches(batch_format="pydict"):
            for i in range(len(batch[column])):
                value = batch[column][i]
                count = batch["count()"][i]
                counts[value] = count

        # Sort by count descending
        sorted_counts = dict(
            sorted(counts.items(), key=lambda x: x[1], reverse=True)
        )

        if limit:
            sorted_counts = dict(list(sorted_counts.items())[:limit])

        self._set_cached(cache_key, sorted_counts)
        return sorted_counts

    def count_duplicates(self, columns: list[str]) -> int:
        """Count duplicates."""
        cache_key = self._cache_key("count_duplicates", tuple(columns))
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        total = self.count_rows()

        # Get unique count
        if len(columns) == 1:
            unique_ds = self._ds.unique(columns[0])
        else:
            # For multiple columns, use groupby
            grouped = self._ds.groupby(columns).count()
            unique_count = grouped.count()
            duplicates = total - unique_count
            self._set_cached(cache_key, duplicates)
            return duplicates

        unique_count = unique_ds.count()
        duplicates = total - unique_count

        self._set_cached(cache_key, duplicates)
        return duplicates

    def count_matching_regex(self, column: str, pattern: str) -> int:
        """Count values matching regex."""
        import ray
        import re

        cache_key = self._cache_key("count_matching_regex", column, pattern)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        compiled = re.compile(pattern)

        @ray.remote
        def count_matches_batch(batch: dict) -> int:
            values = batch.get(column, [])
            return sum(
                1 for v in values
                if v is not None and compiled.match(str(v))
            )

        batch_refs = []
        for batch in self._ds.iter_batches(
            batch_format="pydict",
            batch_size=self._config.batch_size,
        ):
            ref = count_matches_batch.remote(batch)
            batch_refs.append(ref)

        count = sum(ray.get(batch_refs))

        self._set_cached(cache_key, count)
        return count

    def count_in_range(
        self,
        column: str,
        min_value: Any | None = None,
        max_value: Any | None = None,
        inclusive: bool = True,
    ) -> int:
        """Count values in range."""
        import ray

        cache_key = self._cache_key(
            "count_in_range", column, min_value, max_value, inclusive
        )
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        @ray.remote
        def count_range_batch(batch: dict) -> int:
            values = batch.get(column, [])
            count = 0
            for v in values:
                if v is None:
                    continue
                in_range = True
                if min_value is not None:
                    in_range = v >= min_value if inclusive else v > min_value
                if in_range and max_value is not None:
                    in_range = v <= max_value if inclusive else v < max_value
                if in_range:
                    count += 1
            return count

        batch_refs = []
        for batch in self._ds.iter_batches(
            batch_format="pydict",
            batch_size=self._config.batch_size,
        ):
            ref = count_range_batch.remote(batch)
            batch_refs.append(ref)

        count = sum(ray.get(batch_refs))

        self._set_cached(cache_key, count)
        return count

    def count_in_set(self, column: str, values: set[Any]) -> int:
        """Count values in set."""
        import ray

        cache_key = self._cache_key("count_in_set", column, frozenset(values))
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        values_set = set(values)

        @ray.remote
        def count_in_set_batch(batch: dict) -> int:
            col_values = batch.get(column, [])
            return sum(1 for v in col_values if v in values_set)

        batch_refs = []
        for batch in self._ds.iter_batches(
            batch_format="pydict",
            batch_size=self._config.batch_size,
        ):
            ref = count_in_set_batch.remote(batch)
            batch_refs.append(ref)

        count = sum(ray.get(batch_refs))

        self._set_cached(cache_key, count)
        return count

    # -------------------------------------------------------------------------
    # Sampling
    # -------------------------------------------------------------------------

    def sample(
        self,
        n: int = 1000,
        seed: int | None = None,
    ) -> "RayExecutionEngine":
        """Create sampled engine using Ray's native sampling.

        Args:
            n: Target number of rows.
            seed: Random seed.

        Returns:
            New engine with sampled data.
        """
        row_count = self.count_rows()

        if row_count <= n:
            return self

        fraction = min((n * 1.1) / row_count, 1.0)

        # Ray's random_sample method
        sampled = self._ds.random_sample(fraction, seed=seed)

        # Limit to exact n rows
        sampled = sampled.limit(n)

        return RayExecutionEngine(sampled, self._config)

    # -------------------------------------------------------------------------
    # Ray-Specific Methods
    # -------------------------------------------------------------------------

    def materialize(self) -> "RayExecutionEngine":
        """Materialize the dataset (trigger execution and cache).

        Returns:
            Self after materializing.
        """
        self._ds = self._ds.materialize()
        return self

    def filter(
        self,
        fn: Callable[[dict[str, Any]], bool],
    ) -> "RayExecutionEngine":
        """Filter the dataset using a function.

        Args:
            fn: Filter function that takes a row dict and returns bool.

        Returns:
            New engine with filtered data.
        """
        filtered = self._ds.filter(fn)
        return RayExecutionEngine(filtered, self._config)

    def select_columns(self, columns: list[str]) -> "RayExecutionEngine":
        """Select specific columns.

        Args:
            columns: Columns to select.

        Returns:
            New engine with selected columns.
        """
        selected = self._ds.select_columns(columns)
        return RayExecutionEngine(selected, self._config)

    def take(self, n: int = 5) -> list[dict[str, Any]]:
        """Get first n rows as list of dicts.

        Args:
            n: Number of rows.

        Returns:
            List of row dictionaries.
        """
        return self._ds.take(n)

    def take_all(self) -> list[dict[str, Any]]:
        """Get all rows as list of dicts.

        Returns:
            List of row dictionaries.
        """
        return self._ds.take_all()

    def show(self, n: int = 20) -> None:
        """Print the first n rows.

        Args:
            n: Number of rows to show.
        """
        self._ds.show(n)

    def to_pandas(self) -> Any:
        """Convert to Pandas DataFrame.

        Returns:
            Pandas DataFrame.
        """
        return self._ds.to_pandas()

    def to_arrow(self) -> "pa.Table":
        """Convert to Arrow Table.

        Returns:
            PyArrow Table.
        """
        return self._ds.to_arrow()

    def write_parquet(self, path: str, **kwargs: Any) -> None:
        """Write to Parquet files.

        Args:
            path: Output path.
            **kwargs: Additional arguments for write_parquet.
        """
        self._ds.write_parquet(path, **kwargs)

    def write_csv(self, path: str, **kwargs: Any) -> None:
        """Write to CSV files.

        Args:
            path: Output path.
            **kwargs: Additional arguments for write_csv.
        """
        self._ds.write_csv(path, **kwargs)

    def stats(self) -> str:
        """Get dataset statistics.

        Returns:
            Statistics string.
        """
        return self._ds.stats()

    def schema_str(self) -> str:
        """Get schema as string.

        Returns:
            Schema string.
        """
        return str(self._schema)

    # -------------------------------------------------------------------------
    # Context Manager
    # -------------------------------------------------------------------------

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        super().__exit__(exc_type, exc_val, exc_tb)
        # Note: We don't shutdown Ray here as it might be shared
        # Users should manage Ray lifecycle separately

    @staticmethod
    def shutdown() -> None:
        """Shutdown Ray."""
        import ray

        if ray.is_initialized():
            ray.shutdown()
