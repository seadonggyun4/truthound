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

import importlib
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:
    from collections.abc import Callable


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
        self._owns_spark = False
        self._spark = None
        self._sc = None

    @property
    def spark_config(self) -> SparkConfig:
        """Get typed configuration."""
        return self.config  # type: ignore

    def is_available(self) -> bool:
        """Check if PySpark is available."""
        try:
            importlib.import_module("pyspark")
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

            from pyspark import SparkContext
            self._owns_spark = SparkContext._active_spark_context is None
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
            if self._owns_spark:
                self._spark.stop()
            self._owns_spark = False
            self._current_df = None
            self._spark = None
            self._sc = None
            self._initialized = False
            logger.info("Spark session stopped")

    def distribute_data(
        self,
        data: pl.DataFrame | pl.LazyFrame | str,
        num_partitions: int | None = None,
        strategy: PartitionStrategy = PartitionStrategy.ROW_BASED,
        sample_size: int | None = None,
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
        elif isinstance(data, (pl.LazyFrame, pl.DataFrame)):
            from pyspark.sql.pandas.types import from_arrow_schema
            frame = data.collect() if isinstance(data, pl.LazyFrame) else data
            schema = from_arrow_schema(frame.to_arrow().schema)
            spark_df = self._spark.createDataFrame(frame.rows(), schema=schema)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        if sample_size is not None:
            spark_df = spark_df.limit(sample_size)
        # Determine partitions
        row_count = spark_df.count()
        if num_partitions is None:
            # Aim for ~1M rows per partition
            num_partitions = max(1, row_count // 1_000_000)
            # But at least as many as default parallelism
            num_partitions = self.config.parallelism or max(num_partitions, self._spark.sparkContext.defaultParallelism)

        from pyspark.sql import functions as F

        from truthound.profiler.distributed._partitions import column_groups, expand
        if not isinstance(strategy, PartitionStrategy):
            raise ValueError("Unknown partition strategy")
        requested = num_partitions
        groups = column_groups(spark_df.columns, requested, strategy)
        physical = 1 if strategy is PartitionStrategy.COLUMN_BASED else max(1, (requested + len(groups) - 1) // len(groups))
        if strategy is PartitionStrategy.HASH:
            spark_df = spark_df.repartition(physical, *[F.col("`" + name.replace("`", "``") + "`") for name in spark_df.columns])
        else:
            spark_df = spark_df.repartition(physical)
        self._publish_checkpoint(lambda directory: spark_df.write.mode("error").parquet(str(directory)))
        self._current_df = spark_df
        return expand(physical, spark_df.columns, requested, strategy)

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


        # Use the stored Spark DataFrame
        spark_df = getattr(self, "_current_df", None)
        if spark_df is None:
            raise ValueError("No distributed data available. Call distribute_data first.")

        policy = self.config
        columns = spark_df.columns

        def process_partition(index, rows):
            import polars as pl

            from truthound.profiler.distributed.base import execute_partition
            records = [row.asDict(recursive=True) for row in rows]
            frame = pl.from_dicts(records, infer_schema_length=None) if records else pl.DataFrame(schema={name: pl.Null for name in columns})
            return [execute_partition(func, partition, frame.select(partition.columns), policy)
                    for partition in partitions if partition.metadata["source_partition"] == index]

        return spark_df.rdd.mapPartitionsWithIndex(process_partition).collect()

    def sample_column(self, name, limit):
        from pyspark.sql import functions as F
        column = F.col("`" + name.replace("`", "``") + "`")
        rows = self._current_df.select(column).where(column.isNotNull()).limit(limit).collect()
        return pl.DataFrame({name: [row[0] for row in rows]}, schema={name: pl.String})

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
        from pyspark.sql import functions as F
        from pyspark.sql.types import NumericType, StringType
        if any(result.errors for result in results):
            raise RuntimeError("Native profiling partition failed")
        frame = getattr(self, "_current_df", None)
        if frame is None:
            raise RuntimeError("No active native profiling input")
        aggregated = {}
        for schema_field in frame.schema:
            column = F.col("`" + schema_field.name.replace("`", "``") + "`")
            expressions = [F.count(F.lit(1)).alias("row_count"),
                           F.sum(column.isNull().cast("long")).alias("null_count"),
                           (F.countDistinct(column) + F.max(column.isNull().cast("long"))).alias("distinct_count")]
            if isinstance(schema_field.dataType, NumericType):
                expressions += [F.avg(column).alias("mean"), F.stddev_samp(column).alias("std"),
                                F.min(column).alias("min_value"), F.max(column).alias("max_value")]
            elif isinstance(schema_field.dataType, StringType):
                length = F.length(column)
                expressions += [F.min(length).alias("min_length"), F.max(length).alias("max_length"), F.avg(length).alias("avg_length")]
            values = frame.agg(*expressions).first().asDict()
            values["null_count"] = values["null_count"] or 0
            values["distinct_count"] = values["distinct_count"] or 0
            values["dtype"] = schema_field.dataType.typeName()
            aggregated[schema_field.name] = values
        return aggregated


# Register backend
backend_registry.register("spark", SparkBackend)
