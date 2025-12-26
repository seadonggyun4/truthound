"""Local multi-threaded backend as fallback.

Uses ThreadPoolExecutor for parallel column processing.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

import polars as pl

from truthound.profiler.distributed.base import (
    BackendConfig,
    DistributedBackend,
    PartitionInfo,
    PartitionStrategy,
    WorkerResult,
)


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
    ) -> list[PartitionInfo]:
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

        self._current_data = df
        columns = df.columns
        row_count = len(df)

        if strategy == PartitionStrategy.COLUMN_BASED:
            # One partition per column or group of columns
            num_parts = len(columns) if num_partitions is None else min(num_partitions, len(columns))
            partitions = []

            cols_per_part = max(1, len(columns) // num_parts)
            for i in range(num_parts):
                start = i * cols_per_part
                end = start + cols_per_part if i < num_parts - 1 else len(columns)
                partitions.append(PartitionInfo(
                    partition_id=i,
                    total_partitions=num_parts,
                    start_row=0,
                    end_row=row_count,
                    columns=columns[start:end],
                ))

            return partitions

        else:  # ROW_BASED or HYBRID
            num_parts = num_partitions or (os.cpu_count() or 4)
            rows_per_part = max(1, row_count // num_parts)

            partitions = []
            for i in range(num_parts):
                start = i * rows_per_part
                end = start + rows_per_part if i < num_parts - 1 else row_count
                partitions.append(PartitionInfo(
                    partition_id=i,
                    total_partitions=num_parts,
                    start_row=start,
                    end_row=end,
                    columns=columns,
                ))

            return partitions

    def map_partitions(
        self,
        func: Callable[[PartitionInfo, Any], WorkerResult],
        partitions: list[PartitionInfo],
        data: Any,
    ) -> list[WorkerResult]:
        if not self._executor:
            raise RuntimeError("Backend not initialized")

        # Use internal profile function for actual profiling
        def profile_partition(partition: PartitionInfo) -> WorkerResult:
            start_time = time.time()
            df = self._current_data

            if df is None:
                return WorkerResult(
                    partition_id=partition.partition_id,
                    column_stats={},
                    row_count=0,
                    processing_time_ms=0,
                    errors=["No data available"],
                )

            # Slice data if row-based
            if partition.start_row > 0 or partition.end_row < len(df):
                df = df.slice(partition.start_row, partition.end_row - partition.start_row)

            # Profile columns
            stats = {}
            for col_name in partition.columns:
                if col_name not in df.columns:
                    continue

                col = df.get_column(col_name)
                col_stats = self._profile_column(col)
                stats[col_name] = col_stats

            elapsed_ms = (time.time() - start_time) * 1000

            return WorkerResult(
                partition_id=partition.partition_id,
                column_stats=stats,
                row_count=len(df),
                processing_time_ms=elapsed_ms,
            )

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
                logger.error(f"Partition {partition.partition_id} failed: {e}")
                results.append(WorkerResult(
                    partition_id=partition.partition_id,
                    column_stats={},
                    row_count=0,
                    processing_time_ms=0,
                    errors=[str(e)],
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

    def aggregate_results(
        self,
        results: list[WorkerResult],
    ) -> dict[str, dict[str, Any]]:
        aggregated: dict[str, dict[str, Any]] = {}

        for result in results:
            for col_name, stats in result.column_stats.items():
                if col_name not in aggregated:
                    aggregated[col_name] = {
                        "row_count": 0,
                        "null_count": 0,
                        "distinct_values": set(),
                        "min_value": None,
                        "max_value": None,
                        "sum_value": 0,
                        "sum_squared": 0,
                        "dtype": stats.get("dtype", "unknown"),
                    }

                agg = aggregated[col_name]
                agg["row_count"] += stats.get("row_count", 0)
                agg["null_count"] += stats.get("null_count", 0)

                # Min/Max
                if stats.get("min_value") is not None:
                    if agg["min_value"] is None or stats["min_value"] < agg["min_value"]:
                        agg["min_value"] = stats["min_value"]
                if stats.get("max_value") is not None:
                    if agg["max_value"] is None or stats["max_value"] > agg["max_value"]:
                        agg["max_value"] = stats["max_value"]

                # For computing variance
                agg["sum_value"] += stats.get("sum_value", 0)
                agg["sum_squared"] += stats.get("sum_squared", 0)

                # Length stats (take from first partition for strings)
                if "min_length" in stats and "min_length" not in agg:
                    agg["min_length"] = stats["min_length"]
                    agg["max_length"] = stats["max_length"]
                    agg["avg_length"] = stats["avg_length"]

                # Distinct count from first partition (approximate)
                if "distinct_count" in stats:
                    agg["distinct_count"] = stats["distinct_count"]

        # Finalize aggregations
        for col_name, agg in aggregated.items():
            if isinstance(agg.get("distinct_values"), set):
                agg["distinct_count"] = len(agg.pop("distinct_values"))

            n = agg["row_count"]
            if n > 0 and agg.get("sum_value"):
                mean = agg["sum_value"] / n
                agg["mean"] = mean
                variance = (agg["sum_squared"] / n) - (mean ** 2)
                agg["std"] = variance ** 0.5 if variance > 0 else 0

        return aggregated


# Register with global registry
from truthound.profiler.distributed.base import backend_registry
backend_registry.register("local", LocalBackend)
