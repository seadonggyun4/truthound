"""Apache Spark Backend for Distributed Processing.

This module provides the Spark integration for large-scale distributed
data profiling. Requires PySpark to be installed.

Example:
    from truthound.profiler.distributed import SparkBackend, SparkConfig

    config = SparkConfig(
        master="spark://cluster:7077",
        app_name="TruthoundProfiler",
        executor_memory="4g",
    )

    with SparkBackend(config) as backend:
        profile = backend.profile(data)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import polars as pl

from truthound.profiler.distributed.base import (
    BackendConfig,
    BackendType,
    DistributedBackend,
    PartitionInfo,
    PartitionStrategy,
    WorkerResult,
    backend_registry,
)


logger = logging.getLogger(__name__)


@dataclass
class SparkConfig(BackendConfig):
    """Configuration for Apache Spark backend.

    Attributes:
        master: Spark master URL (local[*], spark://host:port, yarn, etc.)
        app_name: Application name
        executor_memory: Memory per executor
        executor_cores: CPU cores per executor
        num_executors: Number of executors
        spark_conf: Additional Spark configuration
    """

    backend_type: BackendType = field(default=BackendType.SPARK)
    master: str = "local[*]"
    app_name: str = "TruthoundProfiler"
    executor_memory: str = "4g"
    executor_cores: int = 2
    num_executors: int = 2
    spark_conf: dict[str, str] = field(default_factory=dict)

    def to_spark_conf(self) -> dict[str, str]:
        """Convert to Spark configuration dictionary."""
        conf = {
            "spark.app.name": self.app_name,
            "spark.master": self.master,
            "spark.executor.memory": self.executor_memory,
            "spark.executor.cores": str(self.executor_cores),
        }
        if self.num_executors > 0:
            conf["spark.executor.instances"] = str(self.num_executors)
        conf.update(self.spark_conf)
        return conf


class SparkBackend(DistributedBackend):
    """Apache Spark distributed backend.

    Provides distributed data profiling using Apache Spark.
    This is ideal for very large datasets on cluster environments.

    Features:
    - Native integration with Spark DataFrames
    - Partition-level parallelism
    - Support for YARN, Mesos, Kubernetes
    - Fault tolerance and speculative execution

    Example:
        config = SparkConfig(master="yarn", num_executors=10)
        backend = SparkBackend(config)

        with backend:
            partitions = backend.distribute_data(df, num_partitions=100)
            results = backend.map_partitions(profile_func, partitions, df)
            stats = backend.aggregate_results(results)
    """

    name = "spark"
    available = False

    def __init__(self, config: SparkConfig | None = None):
        """Initialize Spark backend.

        Args:
            config: Spark configuration
        """
        super().__init__(config or SparkConfig())
        self._spark = None
        self._sc = None

    @property
    def spark_config(self) -> SparkConfig:
        """Get typed configuration."""
        return self.config  # type: ignore

    def is_available(self) -> bool:
        """Check if PySpark is available."""
        try:
            import pyspark
            return True
        except ImportError:
            return False

    def initialize(self) -> None:
        """Initialize Spark session.

        Creates a SparkSession with the configured parameters.
        """
        if self._initialized:
            return

        if not self.is_available():
            raise ImportError(
                "PySpark is required for SparkBackend. "
                "Install with: pip install pyspark"
            )

        with self._lock:
            if self._initialized:
                return

            from pyspark.sql import SparkSession

            # Build Spark session
            builder = SparkSession.builder

            for key, value in self.spark_config.to_spark_conf().items():
                builder = builder.config(key, value)

            self._spark = builder.getOrCreate()
            self._sc = self._spark.sparkContext

            logger.info(
                f"Spark session initialized: {self.spark_config.app_name} "
                f"({self.spark_config.master})"
            )

            self._initialized = True

    def shutdown(self) -> None:
        """Stop Spark session."""
        if self._spark is not None:
            self._spark.stop()
            self._spark = None
            self._sc = None
            self._initialized = False
            logger.info("Spark session stopped")

    def distribute_data(
        self,
        data: pl.DataFrame | pl.LazyFrame | str,
        num_partitions: int | None = None,
        strategy: PartitionStrategy = PartitionStrategy.ROW_BASED,
    ) -> list[PartitionInfo]:
        """Distribute data across Spark executors.

        Args:
            data: Polars DataFrame, LazyFrame, or path to data file
            num_partitions: Number of partitions (None = auto based on data size)
            strategy: Partitioning strategy

        Returns:
            List of partition information
        """
        if not self._initialized:
            self.initialize()

        # Convert to Spark DataFrame if needed
        if isinstance(data, str):
            # Load from path
            spark_df = self._spark.read.parquet(data)
        elif isinstance(data, pl.LazyFrame):
            # Collect and convert
            pdf = data.collect().to_pandas()
            spark_df = self._spark.createDataFrame(pdf)
        elif isinstance(data, pl.DataFrame):
            pdf = data.to_pandas()
            spark_df = self._spark.createDataFrame(pdf)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        # Determine partitions
        row_count = spark_df.count()
        if num_partitions is None:
            # Aim for ~1M rows per partition
            num_partitions = max(1, row_count // 1_000_000)
            # But at least as many as default parallelism
            num_partitions = max(num_partitions, self._spark.sparkContext.defaultParallelism)

        # Repartition data
        spark_df = spark_df.repartition(num_partitions)

        # Create partition info
        rows_per_partition = row_count // num_partitions
        partitions = []

        for i in range(num_partitions):
            start_row = i * rows_per_partition
            end_row = (i + 1) * rows_per_partition if i < num_partitions - 1 else row_count

            partitions.append(PartitionInfo(
                partition_id=i,
                total_partitions=num_partitions,
                start_row=start_row,
                end_row=end_row,
                columns=spark_df.columns,
                metadata={"spark_df_id": id(spark_df)},
            ))

        # Store reference to Spark DataFrame
        self._current_df = spark_df

        return partitions

    def map_partitions(
        self,
        func: Callable[[PartitionInfo, Any], WorkerResult],
        partitions: list[PartitionInfo],
        data: Any,
    ) -> list[WorkerResult]:
        """Execute profiling function on each partition using Spark.

        Args:
            func: Profiling function
            partitions: Partition info list
            data: Reference to data (Spark DataFrame)

        Returns:
            List of worker results
        """
        if not self._initialized:
            self.initialize()

        import time

        # Use the stored Spark DataFrame
        spark_df = getattr(self, "_current_df", None)
        if spark_df is None:
            raise ValueError("No distributed data available. Call distribute_data first.")

        results = []

        # Process each partition
        def process_partition(partition_data):
            """Process a single Spark partition."""
            import polars as pl

            # Convert to Polars for processing
            rows = list(partition_data)
            if not rows:
                return []

            # Convert to Polars DataFrame
            df = pl.DataFrame(rows)

            # Calculate basic stats
            stats = {}
            for col in df.columns:
                series = df.get_column(col)
                stats[col] = {
                    "count": len(series),
                    "null_count": series.null_count(),
                    "distinct_count": series.n_unique(),
                }

                if series.dtype.is_numeric():
                    non_null = series.drop_nulls()
                    if len(non_null) > 0:
                        stats[col].update({
                            "min": float(non_null.min()),
                            "max": float(non_null.max()),
                            "mean": float(non_null.mean()),
                        })

            return [stats]

        # Map over partitions
        rdd = spark_df.rdd.mapPartitions(process_partition)
        partition_results = rdd.collect()

        # Convert to WorkerResult format
        for i, stats_list in enumerate(partition_results):
            if stats_list:
                stats = stats_list[0] if isinstance(stats_list, list) else stats_list
                results.append(WorkerResult(
                    partition_id=i,
                    column_stats=stats,
                    row_count=sum(s.get("count", 0) for s in stats.values()),
                    processing_time_ms=0,  # Not tracked per partition
                ))

        return results

    def aggregate_results(
        self,
        results: list[WorkerResult],
    ) -> dict[str, dict[str, Any]]:
        """Aggregate statistics from all partitions.

        Args:
            results: Worker results from each partition

        Returns:
            Aggregated column statistics
        """
        if not results:
            return {}

        # Gather all columns
        all_columns = set()
        for result in results:
            all_columns.update(result.column_stats.keys())

        aggregated = {}

        for col in all_columns:
            col_stats = [r.column_stats.get(col, {}) for r in results if col in r.column_stats]

            if not col_stats:
                continue

            # Aggregate counts
            total_count = sum(s.get("count", 0) for s in col_stats)
            total_null = sum(s.get("null_count", 0) for s in col_stats)

            # For distinct count, we can only estimate (could use HLL)
            max_distinct = max((s.get("distinct_count", 0) for s in col_stats), default=0)

            aggregated[col] = {
                "count": total_count,
                "null_count": total_null,
                "null_ratio": total_null / total_count if total_count > 0 else 0,
                "distinct_count_estimate": max_distinct,  # Lower bound
            }

            # Aggregate numeric stats
            mins = [s.get("min") for s in col_stats if s.get("min") is not None]
            maxs = [s.get("max") for s in col_stats if s.get("max") is not None]
            means = [s.get("mean") for s in col_stats if s.get("mean") is not None]
            counts = [s.get("count", 0) for s in col_stats]

            if mins:
                aggregated[col]["min"] = min(mins)
            if maxs:
                aggregated[col]["max"] = max(maxs)
            if means and sum(counts) > 0:
                # Weighted average
                weighted_mean = sum(m * c for m, c in zip(means, counts)) / sum(counts)
                aggregated[col]["mean"] = weighted_mean

        return aggregated


# Register backend
backend_registry.register("spark", SparkBackend)
