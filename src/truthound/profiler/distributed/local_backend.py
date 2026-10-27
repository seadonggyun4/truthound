"""Local multi-threaded backend as fallback.

Uses ThreadPoolExecutor for parallel column processing.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

import polars as pl

from truthound.profiler.distributed.base import (
    BackendConfig,
    DistributedBackend,
    PartitionInfo,
    PartitionStrategy,
    WorkerResult,
    backend_registry,
)

if TYPE_CHECKING:
    from collections.abc import Callable


logger = logging.getLogger(__name__)


class LocalBackend(DistributedBackend):
    """Local multi-threaded backend as fallback.

    Uses ThreadPoolExecutor for parallel column processing.
    Always available as it has no external dependencies.
    """

    name = "local"
    available = True

    def __init__(self, config: BackendConfig | None = None):
        super().__init__(config or BackendConfig())
        self._executor: ThreadPoolExecutor | None = None
        self._current_data: pl.DataFrame | None = None

    def initialize(self) -> None:
        if self._initialized:
            return
        num_workers = self.config.num_workers
        if num_workers <= 0:
            num_workers = min(os.cpu_count() or 4, 8)

        self._executor = ThreadPoolExecutor(max_workers=num_workers)
        self._initialized = True
        logger.info(f"LocalBackend initialized with {num_workers} workers")

    def shutdown(self) -> None:
        if self._executor:
            self._executor.shutdown(wait=True)
        self._executor = None
        self._current_data = None
        self._initialized = False

    def is_available(self) -> bool:
        return True

    def distribute_data(
        self,
        data: pl.DataFrame | pl.LazyFrame | str,
        num_partitions: int | None = None,
        strategy: PartitionStrategy = PartitionStrategy.COLUMN_BASED,
        sample_size: int | None = None,
    ) -> list[PartitionInfo]:
        if num_partitions is not None and (isinstance(num_partitions, bool) or num_partitions < 1):
            raise ValueError("num_partitions must be positive")
        # Load data if path
        if isinstance(data, str):
            if data.endswith(".parquet"):
                df = pl.read_parquet(data)
            elif data.endswith(".csv"):
                df = pl.read_csv(data)
            else:
                df = pl.scan_parquet(data).collect()
        elif isinstance(data, pl.LazyFrame):
            df = data.collect()
        else:
            df = data

        if sample_size is not None:
            df = df.head(sample_size)
        self._checkpoint_polars(df)
        self._current_data = df
        columns = df.columns
        row_count = len(df)

        from truthound.profiler.distributed._partitions import polars_plan
        return polars_plan(row_count, columns, num_partitions or self.config.parallelism or (os.cpu_count() or 4), strategy)

    def map_partitions(
        self,
        func: Callable[[PartitionInfo, Any], WorkerResult],
        partitions: list[PartitionInfo],
        data: Any,
    ) -> list[WorkerResult]:
        if not self._executor:
            raise RuntimeError("Backend not initialized")

        from truthound.profiler.distributed.base import execute_partition
        policy = self.config
        frame = self._current_data
        if frame is None:
            raise RuntimeError("No data distributed")
        def profile_partition(partition):
            from truthound.profiler.distributed._partitions import select_polars
            selected = select_polars(frame, partition)
            return execute_partition(func, partition, selected, policy)

        futures = {
            self._executor.submit(profile_partition, partition): partition
            for partition in partitions
        }

        results = []
        for future in as_completed(futures):
            partition = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error("Native profiling partition failed")
                results.append(WorkerResult(
                    partition_id=partition.partition_id,
                    column_stats={},
                    row_count=0,
                    processing_time_ms=0,
                    errors=[type(e).__name__],
                ))

        return sorted(results, key=lambda r: r.partition_id)

    def _profile_column(self, col: pl.Series) -> dict[str, Any]:
        """Profile a single column."""
        row_count = len(col)
        null_count = col.null_count()
        distinct_count = col.n_unique()

        stats = {
            "row_count": row_count,
            "null_count": null_count,
            "distinct_count": distinct_count,
            "dtype": str(col.dtype),
        }

        # Numeric statistics
        if col.dtype.is_numeric():
            non_null = col.drop_nulls()
            if len(non_null) > 0:
                stats.update({
                    "min_value": float(non_null.min()),
                    "max_value": float(non_null.max()),
                    "sum_value": float(non_null.sum()),
                    "mean": float(non_null.mean()),
                    "std": float(non_null.std()) if len(non_null) > 1 else 0.0,
                    "sum_squared": float((non_null ** 2).sum()),
                })

        # String statistics
        elif col.dtype == pl.Utf8:
            non_null = col.drop_nulls()
            if len(non_null) > 0:
                lengths = non_null.str.len_chars()
                stats.update({
                    "min_length": int(lengths.min()),
                    "max_length": int(lengths.max()),
                    "avg_length": float(lengths.mean()),
                })

        return stats

    def sample_column(self, name, limit):
        return self._current_data.select(name).drop_nulls().head(limit)

    def aggregate_results(
        self,
        results: list[WorkerResult],
    ) -> dict[str, dict[str, Any]]:
        if any(result.errors for result in results):
            raise RuntimeError("Native profiling partition failed")
        if self._current_data is None:
            raise RuntimeError("No active native profiling input")
        # Global native reductions preserve cross-partition duplicates and null
        # denominators without shipping unbounded distinct-value sets in workers.
        columns = {name for result in results for name in result.column_stats}
        return {name: self._profile_column(self._current_data[name])
                for name in self._current_data.columns if name in columns}


# Register with global registry

backend_registry.register("local", LocalBackend)
