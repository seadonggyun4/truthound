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


def _check_dask_available() -> None:
    """Check if Dask is available."""
    try:
        import dask.dataframe  # noqa: F401
    except ImportError:
        raise ImportError(
            "dask is required for DaskExecutionEngine. "
            "Install with: pip install dask[dataframe] distributed"
        )


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
        dask_df: "dd.DataFrame",
        config: DaskEngineConfig | None = None,
        client: "Client | None" = None,
    ) -> None:
        """Initialize Dask execution engine.

        Args:
            dask_df: Dask DataFrame.
            config: Optional configuration.
            client: Optional Dask distributed client.
        """
        _check_dask_available()
        super().__init__(config)

        self._ddf = dask_df
        self._client = client
        self._columns = list(dask_df.columns)
        self._cached_row_count: int | None = None
        self._dtypes = dict(dask_df.dtypes)

        # Initialize distributed client if configured
        self._setup_client()

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
                    self._client = Client(self._config.client_address)
                else:
                    self._client = Client(
                        n_workers=self._config.n_workers,
                        threads_per_worker=self._config.threads_per_worker,
                        memory_limit=self._config.memory_per_worker,
                        processes=self._config.processes,
                        dashboard_address=self._config.dashboard_address,
                    )
            except ImportError:
                logger.warning(
                    "distributed not installed. Using default scheduler."
                )

    # -------------------------------------------------------------------------
    # Factory Methods
    # -------------------------------------------------------------------------

    @classmethod
    def from_dataframe(
        cls,
        ddf: "dd.DataFrame",
        config: DaskEngineConfig | None = None,
        client: "Client | None" = None,
    ) -> "DaskExecutionEngine":
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
        client: "Client | None" = None,
        **read_kwargs: Any,
    ) -> "DaskExecutionEngine":
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
        client: "Client | None" = None,
        **read_kwargs: Any,
    ) -> "DaskExecutionEngine":
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
        pdf: "pd.DataFrame",
        npartitions: int = 4,
        config: DaskEngineConfig | None = None,
        client: "Client | None" = None,
    ) -> "DaskExecutionEngine":
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
    def dask_dataframe(self) -> "dd.DataFrame":
        """Get the underlying Dask DataFrame."""
        return self._ddf

    @property
    def client(self) -> "Client | None":
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

                return pd.DataFrame([{
                    "partition_id": partition_id,
                    "value": result.get("value"),
                    "row_count": len(pdf),
                    "duration_ms": duration_ms,
                    "errors": result.get("errors", []),
                    "metadata": result.get("metadata", {}),
                }])

            results_ddf = ddf.map_partitions(
                wrapped_func,
                meta=pd.DataFrame({
                    "partition_id": pd.Series(dtype=int),
                    "value": pd.Series(dtype=object),
                    "row_count": pd.Series(dtype=int),
                    "duration_ms": pd.Series(dtype=float),
                    "errors": pd.Series(dtype=object),
                    "metadata": pd.Series(dtype=object),
                }),
            )

            results_pdf = results_ddf.compute()

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
            results = {}

            for agg in spec.aggregations:
                column = agg.column
                operation = agg.operation
                alias = agg.alias
                params = agg.params

                if operation == "count":
                    if column == "*":
                        value = len(self._ddf)
                    else:
                        value = self._ddf[column].count().compute()
                    results[alias] = value

                elif operation == "sum":
                    value = self._ddf[column].sum().compute()
                    results[alias] = value

                elif operation == "mean":
                    value = self._ddf[column].mean().compute()
                    results[alias] = value

                elif operation == "min":
                    value = self._ddf[column].min().compute()
                    results[alias] = value

                elif operation == "max":
                    value = self._ddf[column].max().compute()
                    results[alias] = value

                elif operation == "std":
                    ddof = params.get("ddof", 1)
                    value = self._ddf[column].std(ddof=ddof).compute()
                    results[alias] = value

                elif operation == "var":
                    ddof = params.get("ddof", 1)
                    value = self._ddf[column].var(ddof=ddof).compute()
                    results[alias] = value

                elif operation == "minmax":
                    min_val = self._ddf[column].min().compute()
                    max_val = self._ddf[column].max().compute()
                    results[alias] = {"min": min_val, "max": max_val}

                elif operation == "null_count":
                    null_count = self._ddf[column].isna().sum().compute()
                    total_count = len(self._ddf)
                    results[alias] = {
                        "null_count": int(null_count),
                        "total_count": total_count,
                    }

                elif operation == "distinct_count":
                    value = self._ddf[column].nunique().compute()
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
        ).compute()

        states = partial_results["state"].tolist()

        # Reduce phase: merge all partial results
        if not states:
            return aggregator.finalize(aggregator.initialize())

        final_state = reduce(aggregator.merge, states)
        return aggregator.finalize(final_state)

    def _to_arrow_batches(
        self,
        batch_size: int | None = None,
    ) -> list["pa.RecordBatch"]:
        """Convert Dask DataFrame to Arrow batches.

        Args:
            batch_size: Batch size for conversion.

        Returns:
            List of Arrow record batches.
        """
        import pyarrow as pa

        batch_size = batch_size or self._config.arrow_batch_size

        try:
            # Dask has native Arrow support via to_arrow
            # This works when pyarrow is installed
            table = self._ddf.compute().to_arrow()
            return table.to_batches(max_chunksize=batch_size)
        except AttributeError:
            # Fallback: Convert via Pandas
            logger.debug("Falling back to Pandas-based Arrow conversion")
            pdf = self._ddf.compute()
            table = pa.Table.from_pandas(pdf)
            return table.to_batches(max_chunksize=batch_size)

    def _repartition(self, num_partitions: int) -> "DaskExecutionEngine":
        """Repartition the underlying DataFrame.

        Args:
            num_partitions: New number of partitions.

        Returns:
            New engine with repartitioned data.
        """
        repartitioned = self._ddf.repartition(npartitions=num_partitions)
        return DaskExecutionEngine(repartitioned, self._config, self._client)

    def coalesce(self, num_partitions: int) -> "DaskExecutionEngine":
        """Coalesce partitions (no shuffle when reducing).

        Args:
            num_partitions: New number of partitions.

        Returns:
            New engine with coalesced data.
        """
        # Dask's repartition with fewer partitions is similar to coalesce
        coalesced = self._ddf.repartition(npartitions=num_partitions)
        return DaskExecutionEngine(coalesced, self._config, self._client)

    # -------------------------------------------------------------------------
    # Core Operation Overrides (Native Dask)
    # -------------------------------------------------------------------------

    def count_rows(self) -> int:
        """Count rows using native Dask len."""
        if self._cached_row_count is not None:
            return self._cached_row_count

        cache_key = self._cache_key("count_rows")
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        count = len(self._ddf)
        self._cached_row_count = count
        self._set_cached(cache_key, count)
        return count

    def get_columns(self) -> list[str]:
        """Get column names."""
        return self._columns

    def count_nulls(self, column: str) -> int:
        """Count nulls using native Dask isna."""
        cache_key = self._cache_key("count_nulls", column)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        count = int(self._ddf[column].isna().sum().compute())
        self._set_cached(cache_key, count)
        return count

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

        computed = dask.compute(results)[0]
        result = {col: int(val) for col, val in computed.items()}

        self._set_cached(cache_key, result)
        return result

    def count_distinct(self, column: str) -> int:
        """Count distinct values using native Dask nunique."""
        cache_key = self._cache_key("count_distinct", column)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        count = int(self._ddf[column].nunique().compute())
        self._set_cached(cache_key, count)
        return count

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

        computed = dask.compute(computations)[0]

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

        result = self._ddf[column].quantile(quantiles).compute()
        result_list = list(result)

        self._set_cached(cache_key, result_list)
        return result_list

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

        result_series = counts.compute()
        result = dict(result_series)

        self._set_cached(cache_key, result)
        return result

    def count_duplicates(self, columns: list[str]) -> int:
        """Count duplicates using native Dask operations."""
        cache_key = self._cache_key("count_duplicates", tuple(columns))
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        total = self.count_rows()
        unique = len(self._ddf[columns].drop_duplicates())
        duplicates = total - unique

        self._set_cached(cache_key, duplicates)
        return duplicates

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
            .compute()
        )

        self._set_cached(cache_key, count)
        return count

    def count_in_range(
        self,
        column: str,
        min_value: Any | None = None,
        max_value: Any | None = None,
        inclusive: bool = True,
    ) -> int:
        """Count values in range using native Dask filter."""
        cache_key = self._cache_key(
            "count_in_range", column, min_value, max_value, inclusive
        )
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        series = self._ddf[column]
        mask = None

        if min_value is not None:
            if inclusive:
                mask = series >= min_value
            else:
                mask = series > min_value

        if max_value is not None:
            max_mask = series <= max_value if inclusive else series < max_value
            mask = mask & max_mask if mask is not None else max_mask

        if mask is None:
            count = self.count_rows()
        else:
            count = int(mask.sum().compute())

        self._set_cached(cache_key, count)
        return count

    def count_in_set(self, column: str, values: set[Any]) -> int:
        """Count values in set using Dask isin."""
        cache_key = self._cache_key("count_in_set", column, frozenset(values))
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        count = int(self._ddf[column].isin(list(values)).sum().compute())
        self._set_cached(cache_key, count)
        return count

    # -------------------------------------------------------------------------
    # Sampling
    # -------------------------------------------------------------------------

    def sample(
        self,
        n: int = 1000,
        seed: int | None = None,
    ) -> "DaskExecutionEngine":
        """Create sampled engine using Dask's native sampling.

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

        sampled = self._ddf.sample(
            frac=fraction,
            random_state=seed,
        )

        # Limit to exact n rows
        sampled = sampled.head(n, npartitions=-1, compute=False)

        return DaskExecutionEngine(sampled, self._config, self._client)

    # -------------------------------------------------------------------------
    # Dask-Specific Methods
    # -------------------------------------------------------------------------

    def persist(self) -> "DaskExecutionEngine":
        """Persist the DataFrame in distributed memory.

        Returns:
            Self after persisting.
        """
        self._ddf = self._ddf.persist()
        return self

    def compute(self) -> "pd.DataFrame":
        """Compute and return as Pandas DataFrame.

        Returns:
            Pandas DataFrame.
        """
        return self._ddf.compute()

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

    def filter(self, condition: str) -> "DaskExecutionEngine":
        """Filter the DataFrame using a query string.

        Args:
            condition: Query condition string.

        Returns:
            New engine with filtered data.
        """
        filtered = self._ddf.query(condition)
        return DaskExecutionEngine(filtered, self._config, self._client)

    def select(self, columns: list[str]) -> "DaskExecutionEngine":
        """Select specific columns.

        Args:
            columns: Columns to select.

        Returns:
            New engine with selected columns.
        """
        selected = self._ddf[columns]
        return DaskExecutionEngine(selected, self._config, self._client)

    def head(self, n: int = 5) -> "pd.DataFrame":
        """Get first n rows as Pandas DataFrame.

        Args:
            n: Number of rows.

        Returns:
            Pandas DataFrame.
        """
        return self._ddf.head(n)

    def tail(self, n: int = 5) -> "pd.DataFrame":
        """Get last n rows as Pandas DataFrame.

        Args:
            n: Number of rows.

        Returns:
            Pandas DataFrame.
        """
        return self._ddf.tail(n)

    def describe(self) -> "pd.DataFrame":
        """Get descriptive statistics.

        Returns:
            Pandas DataFrame with statistics.
        """
        return self._ddf.describe().compute()

    # -------------------------------------------------------------------------
    # Context Manager
    # -------------------------------------------------------------------------

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - cleanup client if we created it."""
        super().__exit__(exc_type, exc_val, exc_tb)
        # Note: We don't close the client here as it might be shared
        # Users should manage client lifecycle separately

    def close(self) -> None:
        """Close the distributed client if it exists."""
        if self._client is not None:
            self._client.close()
            self._client = None
