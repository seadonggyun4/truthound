"""Dask-native execution engine for distributed data validation.

This module provides a Dask-native execution engine that:
- Executes validation operations directly on Dask DataFrames
- Avoids Polars conversion overhead for distributed operations
- Uses Arrow for efficient data transfer when conversion is needed
- Supports distributed aggregations with proper reduce semantics

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    DaskExecutionEngine                          │
    │                                                                  │
    │   ┌──────────────────────────────────────────────────────────┐  │
    │   │              Native Dask Operations                      │  │
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
    >>> import dask.dataframe as dd
    >>> from truthound.execution.distributed import DaskExecutionEngine
    >>>
    >>> ddf = dd.read_parquet("large_data.parquet")
    >>>
    >>> # Create native Dask engine
    >>> engine = DaskExecutionEngine.from_dataframe(ddf)
    >>>
    >>> # Native Dask operations (no conversion overhead)
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

    import dask.dataframe as dd
    import pandas as pd
    import pyarrow as pa
    from distributed import Client

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class DaskEngineConfig(DistributedEngineConfig):
    """Configuration for Dask execution engine.

    Attributes:
        scheduler: Dask scheduler to use ('distributed', 'threads', 'synchronous').
        client_address: Address of distributed scheduler (for distributed mode).
        n_workers: Number of workers (for local cluster).
        threads_per_worker: Threads per worker.
        memory_per_worker: Memory limit per worker.
        processes: Use processes instead of threads.
        dashboard_address: Dashboard address (for distributed mode).
        blocksize: Block size for reading files.
        persist_intermediate: Persist intermediate results.
    """

    scheduler: str = "threads"  # 'distributed', 'threads', 'synchronous'
    client_address: str | None = None
    n_workers: int | None = None
    threads_per_worker: int = 2
    memory_per_worker: str = "2GB"
    processes: bool = False
    dashboard_address: str = ":8787"
    blocksize: str = "128MB"
    persist_intermediate: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.scheduler not in {"distributed", "threads", "synchronous"}:
            raise ValueError("Unsupported Dask scheduler")
        for name in ("n_workers", "threads_per_worker"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be a positive integer")
        for name in ("processes", "persist_intermediate", "parallel", "cache_results"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")


def _check_dask_available() -> None:
    """Check if Dask is available."""
    try:
        import dask.dataframe  # noqa: F401
    except ImportError as error:
        raise ImportError(
            "dask is required for DaskExecutionEngine. "
            "Install with: pip install dask[dataframe] distributed"
        ) from error


# =============================================================================
# Dask Execution Engine
# =============================================================================


class DaskExecutionEngine(BaseDistributedEngine[DaskEngineConfig]):
    """Dask-native execution engine for distributed validation.

    This engine executes validation operations directly on Dask DataFrames,
    avoiding the overhead of converting to Polars for operations that can
    be performed natively in Dask.

    Key Features:
    - Native Dask aggregations (count, sum, mean, min, max, etc.)
    - Distributed null/duplicate checking
    - Arrow-based zero-copy conversion to Polars when needed
    - Partition-aware operations
    - Lazy evaluation with optimized task graphs

    Example:
        >>> engine = DaskExecutionEngine.from_dataframe(dask_df)
        >>> null_counts = engine.count_nulls_all()  # Native Dask
        >>> lf = engine.to_polars_lazyframe()  # Arrow-based conversion
    """

    engine_type = "dask"

    def __init__(
        self,
        dask_df: dd.DataFrame,
        config: DaskEngineConfig | None = None,
        client: Client | None = None,
        *,
        _apply_defaults: bool = True,
    ) -> None:
        """Initialize Dask execution engine.

        Args:
            dask_df: Dask DataFrame.
            config: Optional configuration.
            client: Optional Dask distributed client.
        """
        _check_dask_available()
        super().__init__(replace(config) if config is not None else None)

        self._ddf = dask_df
        self._client = client
        self._owns_client = False
        self._columns = list(dask_df.columns)
        self._cached_row_count: int | None = None
        self._dtypes = dict(dask_df.dtypes)

        # Initialize distributed client if configured
        self._setup_client()
        try:
            if _apply_defaults and (
                self._config.num_partitions
                or self._config.partition_strategy != PartitionStrategy.ROW_HASH
            ):
                self._ddf = self._partition_native(
                    self._ddf, self._config.num_partitions or self._ddf.npartitions
                )
            if _apply_defaults and self._config.checkpoint_enabled:
                restored = self.checkpoint()
                self._ddf = restored._ddf
                self._checkpoint_path = restored.checkpoint_path
            if (
                self._config.persist_intermediate
                or self._config.execution_mode == ExecutionMode.EAGER
            ):
                self.persist()
            if self._config.execution_mode == ExecutionMode.EAGER:
                self._wait_eager()
        except BaseException as original:
            try:
                self.close()
            except BaseException as cleanup:
                raise BaseExceptionGroup(
                    "Dask initialization and cleanup failed", [original, cleanup]
                ) from None
            raise

    def _wait_eager(self):
        if self._client is None:
            return
        from distributed import futures_of, wait

        futures = futures_of(self._ddf, client=self._client)
        if not futures:
            return
        # persist() submits asynchronously to a distributed scheduler. Do not
        # gather full pandas partitions into the driver just to observe them.
        with self._client.as_current():
            wait(futures, timeout=self._config.timeout_seconds)
        for future in futures:
            error = future.exception()
            if error is not None:
                raise error

    @classmethod
    def _default_config(cls) -> DaskEngineConfig:
        """Create default configuration."""
        return DaskEngineConfig()

    def _setup_client(self) -> None:
        """Set up Dask distributed client if needed."""
        if self._config.scheduler == "distributed" and self._client is None:
            try:
                from distributed import Client

                if self._config.client_address:
                    self._client = Client(self._config.client_address, set_as_default=False)
                else:
                    self._client = Client(
                        set_as_default=False,
                        n_workers=self._config.n_workers,
                        threads_per_worker=self._config.threads_per_worker,
                        memory_limit=self._config.memory_per_worker,
                        processes=self._config.processes,
                        dashboard_address=self._config.dashboard_address,
                    )
                self._owns_client = True
            except ImportError as error:
                raise ImportError(
                    "The selected distributed scheduler requires distributed"
                ) from error

    def _scheduler(self):
        if not self._config.parallel:
            return "synchronous"
        if self._client is not None:
            return self._client.get
        return self._config.scheduler

    # -------------------------------------------------------------------------
    # Factory Methods
    # -------------------------------------------------------------------------

    @classmethod
    def from_dataframe(
        cls,
        ddf: dd.DataFrame,
        config: DaskEngineConfig | None = None,
        client: Client | None = None,
    ) -> DaskExecutionEngine:
        """Create engine from existing Dask DataFrame.

        Args:
            ddf: Dask DataFrame.
            config: Optional configuration.
            client: Optional distributed client.

        Returns:
            DaskExecutionEngine instance.
        """
        return cls(ddf, config, client)

    @classmethod
    def from_parquet(
        cls,
        path: str,
        config: DaskEngineConfig | None = None,
        client: Client | None = None,
        **read_kwargs: Any,
    ) -> DaskExecutionEngine:
        """Create engine from Parquet files.

        Args:
            path: Path to Parquet files (can use glob patterns).
            config: Optional configuration.
            client: Optional distributed client.
            **read_kwargs: Additional arguments for read_parquet.

        Returns:
            DaskExecutionEngine instance.
        """
        _check_dask_available()
        import dask.dataframe as dd

        cfg = config or DaskEngineConfig()
        ddf = dd.read_parquet(path, blocksize=cfg.blocksize, **read_kwargs)

        return cls(ddf, config, client)

    @classmethod
    def from_csv(
        cls,
        path: str,
        config: DaskEngineConfig | None = None,
        client: Client | None = None,
        **read_kwargs: Any,
    ) -> DaskExecutionEngine:
        """Create engine from CSV files.

        Args:
            path: Path to CSV files (can use glob patterns).
            config: Optional configuration.
            client: Optional distributed client.
            **read_kwargs: Additional arguments for read_csv.

        Returns:
            DaskExecutionEngine instance.
        """
        _check_dask_available()
        import dask.dataframe as dd

        cfg = config or DaskEngineConfig()
        ddf = dd.read_csv(path, blocksize=cfg.blocksize, **read_kwargs)

        return cls(ddf, config, client)

    @classmethod
    def from_pandas(
        cls,
        pdf: pd.DataFrame,
        npartitions: int = 4,
        config: DaskEngineConfig | None = None,
        client: Client | None = None,
    ) -> DaskExecutionEngine:
        """Create engine from Pandas DataFrame.

        Args:
            pdf: Pandas DataFrame.
            npartitions: Number of partitions.
            config: Optional configuration.
            client: Optional distributed client.

        Returns:
            DaskExecutionEngine instance.
        """
        _check_dask_available()
        import dask.dataframe as dd

        ddf = dd.from_pandas(pdf, npartitions=npartitions)

        return cls(ddf, config, client)

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def backend_type(self) -> ComputeBackend:
        """Get the compute backend type."""
        return ComputeBackend.DASK

    @property
    def dask_dataframe(self) -> dd.DataFrame:
        """Get the underlying Dask DataFrame."""
        return self._ddf

    @property
    def client(self) -> Client | None:
        """Get the distributed client."""
        return self._client

    @property
    def supports_sql_pushdown(self) -> bool:
        """Dask has limited SQL pushdown support via dask-sql."""
        return False

    # -------------------------------------------------------------------------
    # Abstract Method Implementations
    # -------------------------------------------------------------------------

    def _get_partition_count(self) -> int:
        """Get number of data partitions."""
        return self._ddf.npartitions

    @operation
    def _get_partition_info(self) -> list[PartitionInfo]:
        """Get information about all partitions.

        Note: Dask doesn't expose partition boundaries easily,
        so we return estimated information.
        """
        num_partitions = self._get_partition_count()
        columns = tuple(self._columns)

        return [
            PartitionInfo(
                partition_id=i,
                total_partitions=num_partitions,
                columns=columns,
            )
            for i in range(num_partitions)
        ]

    @operation
    def _execute_on_partitions(
        self,
        operation: str,
        func: Callable[[Any], dict[str, Any]],
        columns: list[str] | None = None,
    ) -> list[DistributedResult]:
        """Execute function on all partitions using map_partitions.

        Args:
            operation: Operation name for metrics.
            func: Function to apply to each partition (receives pandas DataFrame).
            columns: Columns to include (None = all).

        Returns:
            Results from all partitions.
        """
        import pandas as pd

        metrics = self._start_metrics(operation)

        try:
            ddf = self._ddf
            if columns:
                ddf = ddf[columns]

            # Map partitions - func receives pandas DataFrame
            def wrapped_func(pdf: pd.DataFrame, partition_info: dict | None = None) -> pd.DataFrame:
                start_time = time.time()
                result = func(pdf)
                duration_ms = (time.time() - start_time) * 1000

                partition_id = 0
                if partition_info:
                    partition_id = partition_info.get("number", 0)

                return pd.DataFrame(
                    [
                        {
                            "partition_id": partition_id,
                            "value": result.get("value"),
                            "row_count": len(pdf),
                            "duration_ms": duration_ms,
                            "errors": result.get("errors", []),
                            "metadata": result.get("metadata", {}),
                        }
                    ]
                )

            results_ddf = ddf.map_partitions(
                wrapped_func,
                meta=pd.DataFrame(
                    {
                        "partition_id": pd.Series(dtype=int),
                        "value": pd.Series(dtype=object),
                        "row_count": pd.Series(dtype=int),
                        "duration_ms": pd.Series(dtype=float),
                        "errors": pd.Series(dtype=object),
                        "metadata": pd.Series(dtype=object),
                    }
                ),
            )

            results_pdf = results_ddf.compute(scheduler=self._scheduler())

            results = []
            total_rows = 0
            for _, row in results_pdf.iterrows():
                row_count = row["row_count"]
                total_rows += row_count
                results.append(
                    DistributedResult(
                        partition_id=row["partition_id"],
                        operation=operation,
                        value=row["value"],
                        row_count=row_count,
                        duration_ms=row["duration_ms"],
                        errors=row["errors"] if row["errors"] else [],
                        metadata=row["metadata"] if row["metadata"] else {},
                    )
                )

            metrics.partitions_processed = len(results)
            metrics.rows_processed = total_rows

            return results

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
        """Perform distributed aggregation using native Dask operations.

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

                return execute_spec(self._ddf, spec, dask_scheduler=self._scheduler())
            results = {}

            for agg in spec.aggregations:
                column = agg.column
                operation = agg.operation
                alias = agg.alias
                params = agg.params

                if operation == "count":
                    if column == "*":
                        value = int(self._ddf.shape[0].compute(scheduler=self._scheduler()))
                    else:
                        value = self._ddf[column].count().compute(scheduler=self._scheduler())
                    results[alias] = value

                elif operation in {"first", "last"}:
                    if operation == "first":
                        values = (
                            self._ddf[column]
                            .head(1, npartitions=-1, compute=False)
                            .compute(scheduler=self._scheduler())
                        )
                        results[alias] = values.iloc[0] if len(values) else None
                    else:
                        values = (
                            self._ddf[column]
                            .map_partitions(
                                lambda partition: partition.tail(1), meta=self._ddf._meta[column]
                            )
                            .compute(scheduler=self._scheduler())
                        )
                        results[alias] = values.iloc[-1] if len(values) else None

                elif operation == "sum":
                    value = self._ddf[column].sum().compute(scheduler=self._scheduler())
                    results[alias] = value

                elif operation == "mean":
                    value = self._ddf[column].mean().compute(scheduler=self._scheduler())
                    results[alias] = value

                elif operation == "min":
                    value = self._ddf[column].min().compute(scheduler=self._scheduler())
                    results[alias] = value

                elif operation == "max":
                    value = self._ddf[column].max().compute(scheduler=self._scheduler())
                    results[alias] = value

                elif operation == "std":
                    ddof = params.get("ddof", 1)
                    value = self._ddf[column].std(ddof=ddof).compute(scheduler=self._scheduler())
                    results[alias] = value

                elif operation == "var":
                    ddof = params.get("ddof", 1)
                    value = self._ddf[column].var(ddof=ddof).compute(scheduler=self._scheduler())
                    results[alias] = value

                elif operation == "minmax":
                    min_val = self._ddf[column].min().compute(scheduler=self._scheduler())
                    max_val = self._ddf[column].max().compute(scheduler=self._scheduler())
                    results[alias] = {"min": min_val, "max": max_val}

                elif operation == "null_count":
                    null_count = self._ddf[column].isna().sum().compute(scheduler=self._scheduler())
                    total_count = int(self._ddf.shape[0].compute(scheduler=self._scheduler()))
                    results[alias] = {
                        "null_count": int(null_count),
                        "total_count": total_count,
                    }

                elif operation == "distinct_count":
                    value = self._ddf[column].nunique().compute(scheduler=self._scheduler())
                    results[alias] = int(value)

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
        import pandas as pd

        aggregator = get_aggregator(agg.operation, **agg.params)
        column = agg.column

        # Map phase: compute partial aggregates per partition
        def map_partition(pdf: pd.DataFrame) -> pd.DataFrame:
            state = aggregator.initialize()
            for value in pdf[column]:
                state = aggregator.accumulate(state, value)
            return pd.DataFrame([{"state": state}])

        partial_results = self._ddf.map_partitions(
            map_partition,
            meta=pd.DataFrame({"state": pd.Series(dtype=object)}),
        ).compute(scheduler=self._scheduler())

        states = partial_results["state"].tolist()

        # Reduce phase: merge all partial results
        if not states:
            return aggregator.finalize(aggregator.initialize())

        final_state = reduce(aggregator.merge, states)
        return aggregator.finalize(final_state)

    def iter_arrow_batches(self, batch_size=None):
        """Read one native partition at a time and close without consuming the next."""
        import pyarrow as pa

        from truthound.execution.distributed.policies import batch_size as selected_size
        from truthound.execution.distributed.policies import checked_batches

        size = selected_size(self, batch_size)

        def batches():
            observed = False
            for task in self._ddf.to_delayed():
                frame = task.compute(scheduler=self._scheduler())
                table = pa.Table.from_pandas(frame, preserve_index=False)
                for batch in table.to_batches(max_chunksize=size):
                    observed = True
                    yield batch
            if not observed:
                schema = pa.Schema.from_pandas(self._ddf._meta, preserve_index=False)
                yield pa.RecordBatch.from_pylist([], schema=schema)

        return checked_batches(self, batches())

    @operation
    def _to_arrow_batches(
        self,
        batch_size: int | None = None,
    ) -> list[pa.RecordBatch]:
        """Convert Dask DataFrame to Arrow batches.

        Args:
            batch_size: Batch size for conversion.

        Returns:
            List of Arrow record batches.
        """
        import pyarrow as pa

        from truthound.execution.distributed.policies import batch_size as selected_size

        batch_size = selected_size(self, batch_size)

        if self._config.execution_mode == ExecutionMode.STREAMING:
            return list(self.iter_arrow_batches(batch_size))
        pdf = self._ddf.compute(scheduler=self._scheduler())
        table = pa.Table.from_pandas(pdf, preserve_index=False)
        from truthound.execution.distributed.policies import checked_batches

        batches = table.to_batches(max_chunksize=batch_size) or [
            pa.RecordBatch.from_pylist([], schema=table.schema)
        ]
        return list(checked_batches(self, batches))

    def _repartition(self, num_partitions: int) -> DaskExecutionEngine:
        """Repartition the underlying DataFrame.

        Args:
            num_partitions: New number of partitions.

        Returns:
            New engine with repartitioned data.
        """
        repartitioned = self._partition_native(self._ddf, num_partitions)
        return DaskExecutionEngine(repartitioned, self._config, self._client, _apply_defaults=False)

    def _partition_native(self, data, count):
        import dask.dataframe as dd

        strategy = self._config.partition_strategy
        columns = self._config.selected_partition_columns(list(data.columns))
        if strategy == PartitionStrategy.CUSTOM:
            result = self._config.partitioner(data, count)
            if (
                not isinstance(result, dd.DataFrame)
                or list(result.columns) != list(data.columns)
                or not result.dtypes.equals(data.dtypes)
            ):
                raise ValueError("Custom partitioner changed the dataframe type or schema")
            if result.npartitions != count:
                raise ValueError("Custom partitioner returned a different partition count")
            return result
        if strategy in {PartitionStrategy.ROW_HASH, PartitionStrategy.COLUMN}:
            return data.shuffle(on=columns, npartitions=count, shuffle_method="tasks", force=True)
        if strategy == PartitionStrategy.ROW_RANGE:
            if len(columns) != 1:
                raise ValueError("Dask range partitioning requires exactly one column")
            from truthound.execution.distributed.partitioning import range_partition_dask

            return range_partition_dask(data, count, columns[0], self._scheduler())
        from truthound.execution.distributed.partitioning import round_robin_dask

        return round_robin_dask(data, count)

    def coalesce(self, num_partitions: int) -> DaskExecutionEngine:
        """Coalesce partitions (no shuffle when reducing).

        Args:
            num_partitions: New number of partitions.

        Returns:
            New engine with coalesced data.
        """
        self._config.validate_partition_count(num_partitions)
        # Dask's repartition with fewer partitions is similar to coalesce
        coalesced = self._ddf.repartition(npartitions=num_partitions)
        return DaskExecutionEngine(coalesced, self._config, self._client, _apply_defaults=False)

    # -------------------------------------------------------------------------
    # Core Operation Overrides (Native Dask)
    # -------------------------------------------------------------------------

    @operation
    def count_rows(self) -> int:
        """Count rows using native Dask len."""
        cache_key = self._cache_key("count_rows")
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        count = int(self._ddf.shape[0].compute(scheduler=self._scheduler()))
        self._set_cached(cache_key, count)
        return count

    def get_columns(self) -> list[str]:
        """Get column names."""
        return self._columns

    @operation
    def count_nulls(self, column: str) -> int:
        """Count nulls using native Dask isna."""
        cache_key = self._cache_key("count_nulls", column)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        count = int(self._ddf[column].isna().sum().compute(scheduler=self._scheduler()))
        self._set_cached(cache_key, count)
        return count

    @operation
    def count_nulls_all(self) -> dict[str, int]:
        """Count nulls in all columns using batch aggregation."""
        cache_key = self._cache_key("count_nulls_all")
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        # Compute all null counts in parallel
        results = {}
        for col in self._columns:
            results[col] = self._ddf[col].isna().sum()

        # Compute all at once
        import dask

        computed = dask.compute(results, scheduler=self._scheduler())[0]
        result = {col: int(val) for col, val in computed.items()}

        self._set_cached(cache_key, result)
        return result

    @operation
    def count_distinct(self, column: str) -> int:
        """Count distinct values using native Dask nunique."""
        cache_key = self._cache_key("count_distinct", column)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        count = int(self._ddf[column].nunique().compute(scheduler=self._scheduler()))
        self._set_cached(cache_key, count)
        return count

    @operation
    def get_stats(self, column: str) -> dict[str, Any]:
        """Get column statistics using native Dask aggregations."""
        cache_key = self._cache_key("get_stats", column)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        # Compute all stats in parallel
        col = self._ddf[column]
        computations = {
            "count": col.count(),
            "null_count": col.isna().sum(),
            "mean": col.mean(),
            "std": col.std(),
            "min": col.min(),
            "max": col.max(),
        }

        import dask

        computed = dask.compute(computations, scheduler=self._scheduler())[0]

        stats = {
            "count": int(computed["count"]),
            "null_count": int(computed["null_count"]),
            "mean": float(computed["mean"]) if computed["mean"] is not None else None,
            "std": float(computed["std"]) if computed["std"] is not None else None,
            "min": computed["min"],
            "max": computed["max"],
        }

        self._set_cached(cache_key, stats)
        return stats

    @operation
    def get_quantiles(
        self,
        column: str,
        quantiles: list[float],
    ) -> list[float]:
        """Get quantiles using Dask's quantile method."""
        cache_key = self._cache_key("get_quantiles", column, tuple(quantiles))
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        result = self._ddf[column].quantile(quantiles).compute(scheduler=self._scheduler())
        result_list = list(result)

        self._set_cached(cache_key, result_list)
        return result_list

    @operation
    def get_value_counts(
        self,
        column: str,
        limit: int | None = None,
    ) -> dict[Any, int]:
        """Get value counts using native Dask value_counts."""
        cache_key = self._cache_key("get_value_counts", column, limit)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        counts = self._ddf[column].value_counts()

        if limit:
            counts = counts.head(limit, npartitions=-1, compute=False)

        result_series = counts.compute(scheduler=self._scheduler())
        result = dict(result_series)

        self._set_cached(cache_key, result)
        return result

    @operation
    def count_duplicates(self, columns: list[str]) -> int:
        """Count duplicates using native Dask operations."""
        cache_key = self._cache_key("count_duplicates", tuple(columns))
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        total = self.count_rows()
        unique = int(
            self._ddf[columns].drop_duplicates().shape[0].compute(scheduler=self._scheduler())
        )
        duplicates = total - unique

        self._set_cached(cache_key, duplicates)
        return duplicates

    @operation
    def count_matching_regex(self, column: str, pattern: str) -> int:
        """Count values matching regex."""
        cache_key = self._cache_key("count_matching_regex", column, pattern)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        count = int(
            self._ddf[column]
            .str.match(pattern, na=False)
            .sum()
            .compute(scheduler=self._scheduler())
        )

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
        """Count values in range using native Dask filter."""
        cache_key = self._cache_key("count_in_range", column, min_value, max_value, inclusive)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        series = self._ddf[column]
        mask = None

        if min_value is not None:
            mask = series >= min_value if inclusive else series > min_value

        if max_value is not None:
            max_mask = series <= max_value if inclusive else series < max_value
            mask = mask & max_mask if mask is not None else max_mask

        if mask is None:
            count = self.count_rows()
        else:
            count = int(mask.sum().compute(scheduler=self._scheduler()))

        self._set_cached(cache_key, count)
        return count

    @operation
    def count_in_set(self, column: str, values: set[Any]) -> int:
        """Count values in set using Dask isin."""
        cache_key = self._cache_key("count_in_set", column, frozenset(values))
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        count = int(self._ddf[column].isin(list(values)).sum().compute(scheduler=self._scheduler()))
        self._set_cached(cache_key, count)
        return count

    # -------------------------------------------------------------------------
    # Sampling
    # -------------------------------------------------------------------------

    def sample(
        self,
        n: int = 1000,
        seed: int | None = None,
    ) -> DaskExecutionEngine:
        """Select an exact-size sample by native distributed random-rank sort.

        Args:
            n: Target number of rows.
            seed: Random seed.

        Returns:
            New engine with sampled data.
        """
        from uuid import uuid4

        import numpy as np

        if type(n) is not int or n < 0:
            raise ValueError("Sample size must be a non-negative integer")
        if seed is not None and (type(seed) is not int or seed < 0):
            raise ValueError("Sample seed must be a non-negative integer")
        row_count = self.count_rows()

        if row_count <= n:
            return self

        if n == 0:
            return DaskExecutionEngine(
                self._ddf.head(0, compute=False), self._config, self._client, _apply_defaults=False
            )
        rank = "__truthound_sample_" + uuid4().hex
        while rank in self._ddf.columns:
            rank = "__truthound_sample_" + uuid4().hex
        resolved_seed = seed if seed is not None else np.random.SeedSequence().entropy

        def ranked_partition(frame, partition_info=None):
            number = partition_info["number"] if partition_info is not None else 0
            rng = np.random.default_rng(np.random.SeedSequence([resolved_seed, number]))
            return frame.assign(**{rank: rng.random(len(frame))})

        metadata = self._ddf._meta.assign(**{rank: np.empty(0, dtype=float)})
        ranked = self._ddf.map_partitions(ranked_partition, meta=metadata)
        # Bernoulli sampling can return no rows when each tiny partition rounds
        # its fractional request to zero. Sorting ranks always supplies n rows.
        selected = ranked.sort_values(rank).head(n, npartitions=-1, compute=False)
        # Keep the rank available through Dask's sort/head optimization. A
        # projected drop can otherwise remove the sort key before the reduction.
        sampled = selected.map_partitions(
            lambda frame: frame.drop(columns=[rank]), meta=self._ddf._meta
        )

        return DaskExecutionEngine(sampled, self._config, self._client, _apply_defaults=False)

    # -------------------------------------------------------------------------
    # Dask-Specific Methods
    # -------------------------------------------------------------------------

    def persist(self) -> DaskExecutionEngine:
        """Persist the DataFrame in distributed memory.

        Returns:
            Self after persisting.
        """
        self._ddf = self._ddf.persist(scheduler=self._scheduler())
        return self

    def checkpoint(self) -> DaskExecutionEngine:
        """Publish native Parquet in checkpoint_dir and return its independent graph."""
        from truthound.execution.distributed.checkpoints import publish

        def write(path):
            self._ddf.to_parquet(
                str(path),
                engine="pyarrow",
                write_index=True,
                compute_kwargs={"scheduler": self._scheduler()},
            )

        location = publish(self._config.checkpoint_dir, "dask", write)
        return self.from_checkpoint(location, self._config, client=self._client)

    @classmethod
    def from_checkpoint(cls, location, config=None, client=None) -> DaskExecutionEngine:
        """Restore a committed native checkpoint without executing the source graph."""
        from pathlib import Path

        import dask.dataframe as dd

        from truthound.execution.distributed.checkpoints import validate

        data = validate(location, "dask")
        restored = cls(
            dd.read_parquet(data, engine="pyarrow"), config, client, _apply_defaults=False
        )
        restored._checkpoint_path = str(Path(location).resolve())
        return restored

    @operation
    def compute(self) -> pd.DataFrame:
        """Compute and return as Pandas DataFrame.

        Returns:
            Pandas DataFrame.
        """
        return self._ddf.compute(scheduler=self._scheduler())

    def visualize(
        self,
        filename: str = "dask_graph",
        format: str = "png",
    ) -> str:
        """Visualize the task graph.

        Args:
            filename: Output filename (without extension).
            format: Output format (png, svg, pdf).

        Returns:
            Path to the generated file.
        """
        return self._ddf.visualize(filename=filename, format=format)

    def filter(self, condition: str) -> DaskExecutionEngine:
        """Filter the DataFrame using a query string.

        Args:
            condition: Query condition string.

        Returns:
            New engine with filtered data.
        """
        filtered = self._ddf.query(condition)
        return DaskExecutionEngine(filtered, self._config, self._client, _apply_defaults=False)

    def select(self, columns: list[str]) -> DaskExecutionEngine:
        """Select specific columns.

        Args:
            columns: Columns to select.

        Returns:
            New engine with selected columns.
        """
        selected = self._ddf[columns]
        return DaskExecutionEngine(selected, self._config, self._client, _apply_defaults=False)

    @operation
    def head(self, n: int = 5) -> pd.DataFrame:
        """Get first n rows as Pandas DataFrame.

        Args:
            n: Number of rows.

        Returns:
            Pandas DataFrame.
        """
        return self._ddf.head(n, npartitions=-1, compute=False).compute(scheduler=self._scheduler())

    @operation
    def tail(self, n: int = 5) -> pd.DataFrame:
        """Get last n rows as Pandas DataFrame.

        Args:
            n: Number of rows.

        Returns:
            Pandas DataFrame.
        """
        return (
            self._ddf.map_partitions(lambda frame: frame.tail(n), meta=self._ddf._meta)
            .compute(scheduler=self._scheduler())
            .tail(n)
        )

    @operation
    def describe(self) -> pd.DataFrame:
        """Get descriptive statistics.

        Returns:
            Pandas DataFrame with statistics.
        """
        return self._ddf.describe().compute(scheduler=self._scheduler())

    # -------------------------------------------------------------------------
    # Context Manager
    # -------------------------------------------------------------------------

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Clear local state and close only the client created by this engine."""
        try:
            super().__exit__(exc_type, exc_val, exc_tb)
        finally:
            self.close()

    def close(self) -> None:
        """Close the owned client; borrowed clients remain caller-owned."""
        if self._client is not None and self._owns_client:
            self._client.close()
            self._client = None
            self._owns_client = False
