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
from dataclasses import dataclass, replace
from functools import reduce
from typing import TYPE_CHECKING, Any

from truthound.execution.distributed.base import (
    BaseDistributedEngine,
    DistributedEngineConfig,
)
from truthound.execution.distributed.policies import operation
from truthound.execution.distributed.protocols import (
    AggregationScope,
    AggregationSpec,
    ComputeBackend,
    DistributedResult,
    ExecutionMode,
    PartitionInfo,
    PartitionStrategy,
    get_aggregator,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import pyarrow as pa
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

    def __post_init__(self):
        super().__post_init__()
        for name in ("batch_size", "actor_pool_size", "target_max_block_size"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.prefetch_batches) is not int or self.prefetch_batches < 0:
            raise ValueError("prefetch_batches must be a non-negative integer")
        if self.concurrency is not None and (
            type(self.concurrency) is not int or self.concurrency <= 0
        ):
            raise ValueError("concurrency must be a positive integer")
        if type(self.use_actors) is not bool:
            raise ValueError("use_actors must be boolean")
        if self.use_actors and self.concurrency is not None:
            raise ValueError("Actor mode uses actor_pool_size; task concurrency cannot also be set")


def _check_ray_available() -> None:
    """Check if Ray is available."""
    try:
        import ray  # noqa: F401
        import ray.data  # noqa: F401
    except ImportError:
        raise ImportError(
            "ray is required for RayExecutionEngine. Install with: pip install 'ray[data]'"
        ) from None


def _ensure_ray_initialized(config: RayEngineConfig) -> None:
    """Ensure Ray is initialized."""
    import ray

    if not ray.is_initialized():
        init_kwargs = {}
        if config.ray_address:
            init_kwargs["address"] = config.ray_address
        if config.num_cpus is not None:
            init_kwargs["num_cpus"] = config.num_cpus
        if config.num_gpus is not None:
            init_kwargs["num_gpus"] = config.num_gpus
        if config.object_store_memory is not None:
            init_kwargs["object_store_memory"] = config.object_store_memory

        from truthound.execution.distributed._ray_startup import initialize_ray

        initialize_ray(**init_kwargs)


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
        dataset: Dataset,
        config: RayEngineConfig | None = None,
        *,
        _apply_defaults: bool = True,
    ) -> None:
        """Initialize Ray execution engine.

        Args:
            dataset: Ray Dataset.
            config: Optional configuration.
        """
        _check_ray_available()
        super().__init__(replace(config) if config is not None else None)

        _ensure_ray_initialized(self._config)

        self._ds = dataset
        self._before_materialize = None
        # DataContext is bound to this selected Dataset's logical plan. Do not
        # mutate Ray's process-global context or initialize a second cluster.
        self._ds.context.target_max_block_size = self._config.target_max_block_size
        self._schema = dataset.schema()
        self._columns = list(self._schema.names) if self._schema else []
        if _apply_defaults and (
            self._config.num_partitions
            or self._config.partition_strategy != PartitionStrategy.ROW_HASH
        ):
            self._ds = self._partition_native(
                self._ds, self._config.num_partitions or self._get_partition_count()
            )
        if _apply_defaults and self._config.checkpoint_enabled:
            restored = self.checkpoint()
            self._ds = restored._ds
            self._checkpoint_path = restored.checkpoint_path
        if self._config.execution_mode == ExecutionMode.EAGER:
            self.materialize()

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
        dataset: Dataset,
        config: RayEngineConfig | None = None,
    ) -> RayExecutionEngine:
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
    ) -> RayExecutionEngine:
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
    ) -> RayExecutionEngine:
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
    ) -> RayExecutionEngine:
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
        table: pa.Table,
        config: RayEngineConfig | None = None,
    ) -> RayExecutionEngine:
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
    ) -> RayExecutionEngine:
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
    def dataset(self) -> Dataset:
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
        """Observe actual blocks; lazy Ray plans have no trustworthy count."""
        try:
            return self._ds.num_blocks()
        except NotImplementedError:
            self.materialize()
            return self._ds.num_blocks()

    @operation
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

    @operation
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

        # Ray 2.48 defines this class but does not export it from ray.data.
        from ray.data._internal.compute import TaskPoolStrategy

        metrics = self._start_metrics(operation)

        try:
            ds = self._ds
            if columns:
                ds = ds.select_columns(columns)

            # Map batches - func receives batch dict
            def wrapped_func(table):
                import pyarrow as pa

                batch = table.to_pydict()
                start_time = time.time()
                result = func(batch)
                duration_ms = (time.time() - start_time) * 1000

                # Get row count from batch
                row_count = len(next(iter(batch.values()))) if batch else 0

                return pa.Table.from_pydict(
                    {
                        "value": [result.get("value")],
                        "row_count": [row_count],
                        "duration_ms": [duration_ms],
                        "errors": [result.get("errors", [])],
                        "metadata": [result.get("metadata", {})],
                    }
                )

            mapper = wrapped_func
            compute = TaskPoolStrategy(size=self._config.concurrency)
            if self._config.use_actors:

                class BatchMapper:
                    def __call__(self, table):
                        return wrapped_func(table)

                mapper = BatchMapper
                compute = ray.data.ActorPoolStrategy(size=self._config.actor_pool_size)
            results_ds = ds.map_batches(
                mapper,
                batch_format="pyarrow",
                batch_size=self._config.batch_size,
                compute=compute,
            )

            # Collect results
            collected = []
            for i, batch in enumerate(self._iter_dictionary_batches(results_ds)):
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

    @operation
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

        metrics = self._start_metrics("aggregate")

        try:
            if (
                spec.group_by
                or spec.filter_condition
                or spec.scope in {AggregationScope.GROUPED, AggregationScope.PARTITION}
            ):
                from truthound.execution.distributed.aggregations import execute_spec

                return execute_spec(self._ds, spec)
            results = {}

            for agg in spec.aggregations:
                column = agg.column
                operation = agg.operation
                alias = agg.alias
                params = agg.params
                cache_key = None
                if (
                    spec.scope == AggregationScope.GLOBAL
                    and type(params) is dict
                    and not params
                    and operation in {"count", "sum", "mean", "min", "max", "minmax", "std", "var", "null_count"}
                ):
                    cache_key = self._cache_key("ray_native_aggregate", operation, column)
                    cached = self._get_cached(cache_key)
                    if cached is None:
                        cached = self._cached_aggregation_stat(column, operation)
                    if cached is not None:
                        results[alias] = cached["value"]
                        self._set_cached(cache_key, cached)
                        continue

                if operation == "count":
                    value = self._ds.count() if column == "*" else self._count_non_null(column)
                    results[alias] = value

                elif operation == "first":
                    values = self._ds.take(1)
                    results[alias] = values[0][column] if values else None

                elif operation == "last":
                    values = self._ds.map_batches(
                        lambda batch: batch.tail(1), batch_format="pandas", batch_size=None
                    ).take_all()
                    results[alias] = values[-1][column] if values else None

                elif operation == "sum":
                    value = self._ds.sum(column)
                    results[alias] = value

                elif operation == "mean":
                    value = self._ds.mean(column)
                    results[alias] = value

                elif operation == "median":
                    from truthound.execution._protocols import AggregationType
                    from truthound.execution.polars_engine import PolarsExecutionEngine

                    # Ray has no exact median reduction. Reuse Core's bounded
                    # Polars bridge instead of averaging partition medians.
                    with PolarsExecutionEngine(self.to_polars_lazyframe()) as local:
                        results[alias] = local.aggregate_column(column, AggregationType.MEDIAN)

                elif operation == "min":
                    value = self._ds.min(column)
                    results[alias] = value

                elif operation == "max":
                    value = self._ds.max(column)
                    results[alias] = value

                elif operation == "std":
                    value = self._ds.std(
                        column,
                        ddof=params.get("ddof", 1),
                        ignore_nulls=params.get("ignore_nulls", True),
                    )
                    results[alias] = value

                elif operation == "var":
                    # Ray doesn't have built-in var, compute from std
                    std = self._ds.std(
                        column,
                        ddof=params.get("ddof", 1),
                        ignore_nulls=params.get("ignore_nulls", True),
                    )
                    value = std**2 if std is not None else None
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

                if cache_key is not None:
                    self._set_cached(cache_key, {"value": results[alias]})

            return results

        except Exception as e:
            metrics.errors.append(str(e))
            raise
        finally:
            self._end_metrics(metrics)

    def _cached_aggregation_stat(self, column: str, operation: str) -> dict[str, Any] | None:
        """Reuse exact default statistics without changing custom/grouped reductions."""
        stats = self._get_cached(self._cache_key("get_stats", column))
        if stats is None:
            return None
        fields = {"count": "count", "mean": "mean", "min": "min", "max": "max", "std": "std"}
        if operation in fields and not (operation == "count" and column == "*"):
            field = fields[operation]
            if field in stats:
                return {"value": stats[field]}
        if operation == "minmax" and {"min", "max"} <= stats.keys():
            return {"value": {"min": stats["min"], "max": stats["max"]}}
        if operation == "var" and "std" in stats:
            return {"value": None if stats["std"] is None else stats["std"] ** 2}
        if operation == "null_count" and {"count", "null_count"} <= stats.keys():
            return {"value": {"null_count": stats["null_count"], "total_count": stats["count"] + stats["null_count"]}}
        return None

    def _iter_dictionary_batches(self, dataset=None):
        source = self._ds if dataset is None else dataset
        for batch in source.iter_batches(
            batch_format="pyarrow",
            batch_size=self._config.batch_size,
            prefetch_batches=self._config.prefetch_batches,
        ):
            yield batch.to_pydict()

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
        for batch in self._iter_dictionary_batches():
            ref = count_nulls_batch.remote(batch)
            null_counts.append(ref)

        return sum(ray.get(null_counts))

    def _count_distinct_column(self, column: str) -> int:
        """Count distinct values in a column."""
        # Dataset.unique returns a materialized list, including null by default.
        return len(self._ds.unique(column))

    def _grouped_counts(self, columns):
        """Use Ray's hash aggregation without mutating another plan's context.

        Ray 2.48's sort-shuffle splitter cannot compare nullable composite keys.
        Hash aggregation preserves native typed key equality, including nulls.
        """
        from ray.data import Dataset
        from ray.data.context import ShuffleStrategy

        partitions = self._config.num_partitions or self._get_partition_count()
        source = Dataset.copy(self._ds, _deep_copy=True)
        source.context.shuffle_strategy = ShuffleStrategy.HASH_SHUFFLE
        return source.groupby(columns, num_partitions=max(1, partitions)).count()

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
        for batch in self._iter_dictionary_batches():
            ref = map_batch.remote(batch)
            batch_refs.append(ref)

        partial_states = ray.get(batch_refs)

        # Reduce phase: merge all partial states
        if not partial_states:
            return aggregator.finalize(aggregator.initialize())

        final_state = reduce(aggregator.merge, partial_states)
        return aggregator.finalize(final_state)

    def iter_arrow_batches(self, batch_size=None):
        """Yield native Arrow batches and eagerly release this iterator on close."""
        import pyarrow as pa

        from truthound.execution.distributed.policies import batch_size as selected_size
        from truthound.execution.distributed.policies import checked_batches

        size = selected_size(self, batch_size)

        def batches():
            source = iter(
                self._ds.iter_batches(
                    batch_format="pyarrow",
                    batch_size=size,
                    prefetch_batches=self._config.prefetch_batches,
                )
            )
            observed = False
            try:
                for table in source:
                    schema = self._schema.base_schema if self._schema is not None else None
                    if isinstance(schema, pa.Schema) and table.schema != schema:
                        table = table.cast(schema)
                    if isinstance(table, pa.RecordBatch):
                        observed = True
                        yield table
                    else:
                        for batch in table.to_batches(max_chunksize=size):
                            observed = True
                            yield batch
                if not observed:
                    schema = self._schema.base_schema if self._schema is not None else pa.schema([])
                    if not isinstance(schema, pa.Schema):
                        schema = pa.Schema.from_pandas(
                            self._ds.to_pandas(limit=0), preserve_index=False
                        )
                    yield pa.RecordBatch.from_pylist([], schema=schema)
            finally:
                close = getattr(source, "close", None)
                if close is not None:
                    close()

        return checked_batches(self, batches())

    @operation
    def _to_arrow_batches(self, batch_size=None):
        return list(self.iter_arrow_batches(batch_size))

    def _repartition(self, num_partitions: int) -> RayExecutionEngine:
        """Repartition the underlying Dataset.

        Args:
            num_partitions: New number of partitions (blocks).

        Returns:
            New engine with repartitioned data.
        """
        repartitioned = self._partition_native(self._ds, num_partitions)
        return RayExecutionEngine(repartitioned, self._config, _apply_defaults=False)

    def _partition_native(self, data, count):
        from ray.data import Dataset
        from ray.data.context import ShuffleStrategy

        strategy = self._config.partition_strategy
        columns = self._config.selected_partition_columns(self._columns)
        if strategy == PartitionStrategy.CUSTOM:
            result = self._config.partitioner(data, count)
            if not isinstance(result, Dataset) or result.schema() != data.schema():
                raise ValueError("Custom partitioner changed the dataset type or schema")
            materialized = result.materialize()
            if materialized.num_blocks() != count:
                raise ValueError("Custom partitioner returned a different partition count")
            return materialized
        if strategy in {PartitionStrategy.ROW_HASH, PartitionStrategy.COLUMN}:
            selected = Dataset.copy(data, _deep_copy=True)
            selected.context.shuffle_strategy = ShuffleStrategy.HASH_SHUFFLE
            return selected.repartition(count, shuffle=True, keys=columns)
        if strategy == PartitionStrategy.ROW_RANGE:
            return data.sort(columns).repartition(count)
        from truthound.execution.distributed.partitioning import round_robin_ray

        return round_robin_ray(data, count)

    def coalesce(self, num_partitions: int) -> RayExecutionEngine:
        """Coalesce partitions (blocks).

        Args:
            num_partitions: New number of partitions.

        Returns:
            New engine with coalesced data.
        """
        self._config.validate_partition_count(num_partitions)
        # Ray's repartition can reduce partitions without full shuffle
        coalesced = self._ds.repartition(num_partitions)
        return RayExecutionEngine(coalesced, self._config, _apply_defaults=False)

    # -------------------------------------------------------------------------
    # Core Operation Overrides (Native Ray)
    # -------------------------------------------------------------------------

    @operation
    def count_rows(self) -> int:
        """Count rows using native Ray count."""
        cache_key = self._cache_key("count_rows")
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        count = self._ds.count()
        self._set_cached(cache_key, count)
        return count

    def get_columns(self) -> list[str]:
        """Get column names."""
        return self._columns

    @operation
    def count_nulls(self, column: str) -> int:
        """Count nulls using distributed computation."""
        cache_key = self._cache_key("count_nulls", column)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        count = self._count_nulls_column(column)
        self._set_cached(cache_key, count)
        return count

    @operation
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
        for batch in self._iter_dictionary_batches():
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

    @operation
    def count_distinct(self, column: str) -> int:
        """Count distinct values using native Ray."""
        cache_key = self._cache_key("count_distinct", column)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        count = self._count_distinct_column(column)
        self._set_cached(cache_key, count)
        return count

    @operation
    def get_stats(self, column: str) -> dict[str, Any]:
        """Get column statistics using native Ray aggregations."""
        cache_key = self._cache_key("get_stats", column)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        from ray.data.aggregate import Count, Max, Mean, Min, Std

        # One native reduction keeps all fields on the same dataset execution.
        # Count() includes rows while the column count excludes native nulls.
        values = self._ds.aggregate(
            Count(alias_name="_rows"),
            Count(column, ignore_nulls=True, alias_name="count"),
            Mean(column, alias_name="mean"),
            Std(column, ddof=1, alias_name="std"),
            Min(column, alias_name="min"),
            Max(column, alias_name="max"),
        )
        if values is None:
            if self._ds.count() != 0:
                raise RuntimeError("Native Ray statistics missing for a non-empty dataset")
            values = {"_rows": 0, "count": 0, "mean": None, "std": None, "min": None, "max": None}
        count = 0 if values["count"] is None else values["count"]
        stats = {
            "count": count,
            "null_count": values["_rows"] - count,
            **{field: values[field] for field in ("mean", "std", "min", "max")},
        }

        self._set_cached(cache_key, stats)
        return stats

    @operation
    def get_value_counts(
        self,
        column: str,
        limit: int | None = None,
    ) -> dict[Any, int]:
        """Get value counts."""

        cache_key = self._cache_key("get_value_counts", column, limit)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        # Use groupby with count
        grouped = self._grouped_counts(column)

        # Collect and sort
        counts = {}
        for batch in self._iter_dictionary_batches(grouped):
            for i in range(len(batch[column])):
                value = batch[column][i]
                count = batch["count()"][i]
                counts[value] = count

        # Sort by count descending
        sorted_counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))

        if limit:
            sorted_counts = dict(list(sorted_counts.items())[:limit])

        self._set_cached(cache_key, sorted_counts)
        return sorted_counts

    @operation
    def count_duplicates(self, columns: list[str]) -> int:
        """Count duplicates."""
        cache_key = self._cache_key("count_duplicates", tuple(columns))
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        total = self.count_rows()

        # Get unique count
        if len(columns) == 1:
            unique_count = len(self._ds.unique(columns[0]))
        else:
            # For multiple columns, use groupby
            grouped = self._grouped_counts(columns)
            unique_count = grouped.count()
            duplicates = total - unique_count
            self._set_cached(cache_key, duplicates)
            return duplicates

        duplicates = total - unique_count

        self._set_cached(cache_key, duplicates)
        return duplicates

    @operation
    def count_matching_regex(self, column: str, pattern: str) -> int:
        """Count values matching regex."""
        import re

        import ray

        cache_key = self._cache_key("count_matching_regex", column, pattern)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        compiled = re.compile(pattern)

        @ray.remote
        def count_matches_batch(batch: dict) -> int:
            values = batch.get(column, [])
            return sum(1 for v in values if v is not None and compiled.match(str(v)))

        batch_refs = []
        for batch in self._iter_dictionary_batches():
            ref = count_matches_batch.remote(batch)
            batch_refs.append(ref)

        count = sum(ray.get(batch_refs))

        self._set_cached(cache_key, count)
        return count

    @operation
    def count_in_range(
        self,
        column: str,
        min_value: Any | None = None,
        max_value: Any | None = None,
        inclusive: bool = True,
    ) -> int:
        """Count values in range."""
        import ray

        cache_key = self._cache_key("count_in_range", column, min_value, max_value, inclusive)
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
        for batch in self._iter_dictionary_batches():
            ref = count_range_batch.remote(batch)
            batch_refs.append(ref)

        count = sum(ray.get(batch_refs))

        self._set_cached(cache_key, count)
        return count

    @operation
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
        for batch in self._iter_dictionary_batches():
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
    ) -> RayExecutionEngine:
        """Create an exact-size sample using Ray's native shuffle.

        Args:
            n: Target number of rows.
            seed: Random seed.

        Returns:
            New engine with sampled data.
        """
        row_count = self.count_rows()

        if row_count <= n:
            return self

        # Bernoulli sampling followed by limit can return fewer than n rows.
        sampled = self._ds.random_shuffle(seed=seed).limit(n)

        return RayExecutionEngine(sampled, self._config, _apply_defaults=False)

    # -------------------------------------------------------------------------
    # Ray-Specific Methods
    # -------------------------------------------------------------------------

    def checkpoint(self) -> RayExecutionEngine:
        """Publish native Parquet in checkpoint_dir and return its independent plan."""
        from pyarrow.fs import LocalFileSystem

        from truthound.execution.distributed.checkpoints import publish

        def write(path):
            # An existing local directory prevents URI inference for a Windows
            # drive path in Ray/PyArrow's native destination resolver.
            path.mkdir()
            self._ds.write_parquet(path.as_posix(), filesystem=LocalFileSystem())
            if not any(path.rglob("*.parquet")):
                # Ray legitimately emits no files for zero rows. Preserve the
                # known Arrow schema before committing an empty snapshot.
                import pyarrow as pa
                import pyarrow.parquet as parquet

                if self._ds.count() != 0:
                    raise ValueError("Native checkpoint write omitted nonempty data")
                schema = self._schema.base_schema if self._schema is not None else pa.schema([])
                if not isinstance(schema, pa.Schema):
                    raise TypeError("Empty native checkpoint requires an Arrow schema")
                path.mkdir(parents=True, exist_ok=True)
                parquet.write_table(
                    pa.Table.from_batches([], schema=schema), path / "empty.parquet"
                )

        location = publish(self._config.checkpoint_dir, "ray", write)
        return self.from_checkpoint(location, self._config)

    @classmethod
    def from_checkpoint(cls, location, config=None) -> RayExecutionEngine:
        """Restore a checked local snapshot through Ray's native Parquet reader."""
        from pathlib import Path

        import ray.data
        from pyarrow.fs import LocalFileSystem

        from truthound.execution.distributed.checkpoints import validate

        data = validate(location, "ray")
        restored = cls(
            ray.data.read_parquet(Path(data).as_posix(), filesystem=LocalFileSystem()),
            config, _apply_defaults=False,
        )
        restored._checkpoint_path = str(Path(location).resolve())
        return restored

    def materialize(self) -> RayExecutionEngine:
        """Materialize the dataset (trigger execution and cache).

        Returns:
            Self after materializing.
        """
        from ray.data.dataset import MaterializedDataset

        if not isinstance(self._ds, MaterializedDataset):
            previous = self._ds
            materialized = previous.materialize()
            self._before_materialize = previous
            self._ds = materialized
        return self

    def filter(
        self,
        fn: Callable[[dict[str, Any]], bool],
    ) -> RayExecutionEngine:
        """Filter the dataset using a function.

        Args:
            fn: Filter function that takes a row dict and returns bool.

        Returns:
            New engine with filtered data.
        """
        filtered = self._ds.filter(fn)
        return RayExecutionEngine(filtered, self._config, _apply_defaults=False)

    def select_columns(self, columns: list[str]) -> RayExecutionEngine:
        """Select specific columns.

        Args:
            columns: Columns to select.

        Returns:
            New engine with selected columns.
        """
        selected = self._ds.select_columns(columns)
        return RayExecutionEngine(selected, self._config, _apply_defaults=False)

    @operation
    def take(self, n: int = 5) -> list[dict[str, Any]]:
        """Get first n rows as list of dicts.

        Args:
            n: Number of rows.

        Returns:
            List of row dictionaries.
        """
        return self._ds.take(n)

    @operation
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

    @operation
    def to_pandas(self) -> Any:
        """Convert to Pandas DataFrame.

        Returns:
            Pandas DataFrame.
        """
        return self._ds.to_pandas()

    @operation
    def to_arrow(self) -> pa.Table:
        """Convert to Arrow Table.

        Returns:
            PyArrow Table.
        """
        return self.to_arrow_table()

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
        if self._before_materialize is not None:
            # Drop only this engine's pinned result. Borrowed datasets and the
            # caller's Ray lifetime remain owned by their original contexts.
            self._ds = self._before_materialize
            self._before_materialize = None
        # Note: We don't shutdown Ray here as it might be shared
        # Users should manage Ray lifecycle separately

    @staticmethod
    def shutdown() -> None:
        """Shutdown Ray."""
        import ray

        if ray.is_initialized():
            ray.shutdown()
