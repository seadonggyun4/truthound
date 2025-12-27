"""Spark-native execution engine for distributed data validation.

This module provides a Spark-native execution engine that:
- Executes validation operations directly on Spark DataFrames
- Avoids Polars conversion overhead for distributed operations
- Uses Arrow for efficient data transfer when conversion is needed
- Supports distributed aggregations with proper reduce semantics

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    SparkExecutionEngine                          │
    │                                                                  │
    │   ┌──────────────────────────────────────────────────────────┐  │
    │   │              Native Spark Operations                      │  │
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
    >>> from pyspark.sql import SparkSession
    >>> from truthound.execution.distributed import SparkExecutionEngine
    >>>
    >>> spark = SparkSession.builder.getOrCreate()
    >>> df = spark.read.parquet("large_data.parquet")
    >>>
    >>> # Create native Spark engine
    >>> engine = SparkExecutionEngine.from_dataframe(df)
    >>>
    >>> # Native Spark operations (no conversion overhead)
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
    from pyspark.sql import DataFrame as SparkDataFrame
    from pyspark.sql import SparkSession

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class SparkEngineConfig(DistributedEngineConfig):
    """Configuration for Spark execution engine.

    Attributes:
        app_name: Spark application name.
        master: Spark master URL.
        executor_memory: Memory per executor.
        driver_memory: Driver memory.
        executor_cores: Cores per executor.
        arrow_enabled: Enable Arrow optimization.
        adaptive_enabled: Enable adaptive query execution.
        broadcast_threshold: Broadcast join threshold in bytes.
        shuffle_partitions: Number of shuffle partitions.
    """

    app_name: str = "truthound-spark"
    master: str = ""  # Empty = use existing session
    executor_memory: str = "4g"
    driver_memory: str = "2g"
    executor_cores: int = 2
    arrow_enabled: bool = True
    adaptive_enabled: bool = True
    broadcast_threshold: int = 10 * 1024 * 1024  # 10MB
    shuffle_partitions: int = 200
    extra_spark_conf: dict[str, str] = field(default_factory=dict)


def _check_pyspark_available() -> None:
    """Check if PySpark is available."""
    try:
        import pyspark  # noqa: F401
    except ImportError:
        raise ImportError(
            "pyspark is required for SparkExecutionEngine. "
            "Install with: pip install pyspark"
        )


# =============================================================================
# Spark Execution Engine
# =============================================================================


class SparkExecutionEngine(BaseDistributedEngine[SparkEngineConfig]):
    """Spark-native execution engine for distributed validation.

    This engine executes validation operations directly on Spark DataFrames,
    avoiding the overhead of converting to Polars for operations that can
    be performed natively in Spark.

    Key Features:
    - Native Spark aggregations (count, sum, avg, min, max, etc.)
    - Distributed null/duplicate checking
    - Arrow-based zero-copy conversion to Polars when needed
    - Partition-aware operations
    - Checkpoint support for fault tolerance

    Example:
        >>> engine = SparkExecutionEngine.from_dataframe(spark_df)
        >>> null_counts = engine.count_nulls_all()  # Native Spark
        >>> lf = engine.to_polars_lazyframe()  # Arrow-based conversion
    """

    engine_type = "spark"

    def __init__(
        self,
        spark_df: "SparkDataFrame",
        config: SparkEngineConfig | None = None,
        spark_session: "SparkSession | None" = None,
    ) -> None:
        """Initialize Spark execution engine.

        Args:
            spark_df: PySpark DataFrame.
            config: Optional configuration.
            spark_session: Optional SparkSession (defaults to df's session).
        """
        _check_pyspark_available()
        super().__init__(config)

        self._df = spark_df
        self._spark = spark_session or spark_df.sparkSession
        self._schema = spark_df.schema
        self._columns = spark_df.columns
        self._cached_row_count: int | None = None

        # Configure Spark for optimal performance
        self._configure_spark()

    @classmethod
    def _default_config(cls) -> SparkEngineConfig:
        """Create default configuration."""
        return SparkEngineConfig()

    def _configure_spark(self) -> None:
        """Configure Spark session for optimal performance."""
        if self._config.arrow_enabled:
            self._spark.conf.set(
                "spark.sql.execution.arrow.pyspark.enabled",
                "true",
            )
            self._spark.conf.set(
                "spark.sql.execution.arrow.pyspark.fallback.enabled",
                "true",
            )

        if self._config.adaptive_enabled:
            self._spark.conf.set(
                "spark.sql.adaptive.enabled",
                "true",
            )

        self._spark.conf.set(
            "spark.sql.autoBroadcastJoinThreshold",
            str(self._config.broadcast_threshold),
        )

        self._spark.conf.set(
            "spark.sql.shuffle.partitions",
            str(self._config.shuffle_partitions),
        )

        # Apply extra configurations
        for key, value in self._config.extra_spark_conf.items():
            self._spark.conf.set(key, value)

    # -------------------------------------------------------------------------
    # Factory Methods
    # -------------------------------------------------------------------------

    @classmethod
    def from_dataframe(
        cls,
        df: "SparkDataFrame",
        config: SparkEngineConfig | None = None,
    ) -> "SparkExecutionEngine":
        """Create engine from existing Spark DataFrame.

        Args:
            df: PySpark DataFrame.
            config: Optional configuration.

        Returns:
            SparkExecutionEngine instance.
        """
        return cls(df, config)

    @classmethod
    def from_table(
        cls,
        spark: "SparkSession",
        table_name: str,
        database: str | None = None,
        config: SparkEngineConfig | None = None,
    ) -> "SparkExecutionEngine":
        """Create engine from Spark table.

        Args:
            spark: SparkSession.
            table_name: Table name.
            database: Optional database name.
            config: Optional configuration.

        Returns:
            SparkExecutionEngine instance.
        """
        _check_pyspark_available()

        full_name = f"{database}.{table_name}" if database else table_name
        df = spark.table(full_name)

        return cls(df, config, spark)

    @classmethod
    def from_parquet(
        cls,
        spark: "SparkSession",
        path: str,
        config: SparkEngineConfig | None = None,
    ) -> "SparkExecutionEngine":
        """Create engine from Parquet files.

        Args:
            spark: SparkSession.
            path: Path to Parquet files.
            config: Optional configuration.

        Returns:
            SparkExecutionEngine instance.
        """
        _check_pyspark_available()

        df = spark.read.parquet(path)
        return cls(df, config, spark)

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def backend_type(self) -> ComputeBackend:
        """Get the compute backend type."""
        return ComputeBackend.SPARK

    @property
    def spark_dataframe(self) -> "SparkDataFrame":
        """Get the underlying Spark DataFrame."""
        return self._df

    @property
    def spark_session(self) -> "SparkSession":
        """Get the Spark session."""
        return self._spark

    @property
    def supports_sql_pushdown(self) -> bool:
        """Spark supports SQL pushdown."""
        return True

    # -------------------------------------------------------------------------
    # Abstract Method Implementations
    # -------------------------------------------------------------------------

    def _get_partition_count(self) -> int:
        """Get number of data partitions."""
        return self._df.rdd.getNumPartitions()

    def _get_partition_info(self) -> list[PartitionInfo]:
        """Get information about all partitions."""
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
        func: Callable[[Iterator[Any]], Iterator[dict[str, Any]]],
        columns: list[str] | None = None,
    ) -> list[DistributedResult]:
        """Execute function on all partitions using mapPartitions.

        Args:
            operation: Operation name for metrics.
            func: Function to apply to each partition.
            columns: Columns to include (None = all).

        Returns:
            Results from all partitions.
        """
        import time

        metrics = self._start_metrics(operation)

        try:
            df = self._df
            if columns:
                df = df.select(*columns)

            # Execute on partitions
            results_rdd = df.rdd.mapPartitions(func)
            raw_results = results_rdd.collect()

            results = []
            total_rows = 0
            for i, result_dict in enumerate(raw_results):
                row_count = result_dict.get("row_count", 0)
                total_rows += row_count
                results.append(
                    DistributedResult(
                        partition_id=i,
                        operation=operation,
                        value=result_dict.get("value"),
                        row_count=row_count,
                        duration_ms=result_dict.get("duration_ms", 0),
                        errors=result_dict.get("errors", []),
                        metadata=result_dict.get("metadata", {}),
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
        """Perform distributed aggregation using native Spark operations.

        This method uses Spark's built-in aggregation functions for
        optimal performance, falling back to map-reduce style
        aggregation for custom aggregators.

        Args:
            spec: Aggregation specification.

        Returns:
            Aggregated results.
        """
        from pyspark.sql import functions as F

        metrics = self._start_metrics("aggregate")

        try:
            results = {}

            # Group aggregations by type for batching
            spark_aggs = []
            custom_aggs = []

            spark_agg_funcs = {
                "count": lambda c: F.count(F.lit(1)) if c == "*" else F.count(c),
                "sum": F.sum,
                "mean": F.avg,
                "min": F.min,
                "max": F.max,
                "std": F.stddev,
                "var": F.variance,
            }

            for agg in spec.aggregations:
                if agg.operation in spark_agg_funcs:
                    spark_aggs.append(agg)
                else:
                    custom_aggs.append(agg)

            # Execute native Spark aggregations in batch
            if spark_aggs:
                exprs = []
                for agg in spark_aggs:
                    func = spark_agg_funcs[agg.operation]
                    expr = func(agg.column).alias(agg.alias)
                    exprs.append(expr)

                if spec.group_by:
                    agg_df = self._df.groupBy(*spec.group_by).agg(*exprs)
                else:
                    agg_df = self._df.agg(*exprs)

                # Collect results
                row = agg_df.collect()[0]
                for agg in spark_aggs:
                    results[agg.alias] = row[agg.alias]

            # Handle minmax specially (returns dict)
            minmax_aggs = [a for a in spec.aggregations if a.operation == "minmax"]
            for agg in minmax_aggs:
                min_val = self._df.agg(F.min(agg.column)).collect()[0][0]
                max_val = self._df.agg(F.max(agg.column)).collect()[0][0]
                results[agg.alias] = {"min": min_val, "max": max_val}

            # Execute custom aggregations using map-reduce
            for agg in custom_aggs:
                if agg.operation == "null_count":
                    # Native Spark null count
                    null_count = self._df.filter(F.col(agg.column).isNull()).count()
                    total_count = self._df.count()
                    results[agg.alias] = {
                        "null_count": null_count,
                        "total_count": total_count,
                    }
                elif agg.operation == "distinct_count":
                    # Native Spark distinct count
                    distinct_count = self._df.select(agg.column).distinct().count()
                    results[agg.alias] = distinct_count
                else:
                    # Use custom aggregator via map-reduce
                    result = self._aggregate_with_aggregator(agg)
                    results[agg.alias] = result

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
        aggregator = get_aggregator(agg.operation, **agg.params)
        column = agg.column

        # Map phase: compute partial aggregates per partition
        def map_partition(iterator: Iterator) -> Iterator:
            state = aggregator.initialize()
            for row in iterator:
                value = row[column] if column in row.asDict() else None
                state = aggregator.accumulate(state, value)
            yield state

        partial_results = self._df.rdd.mapPartitions(map_partition).collect()

        # Reduce phase: merge all partial results
        if not partial_results:
            return aggregator.finalize(aggregator.initialize())

        final_state = partial_results[0]
        for state in partial_results[1:]:
            final_state = aggregator.merge(final_state, state)

        return aggregator.finalize(final_state)

    def _to_arrow_batches(
        self,
        batch_size: int | None = None,
    ) -> list["pa.RecordBatch"]:
        """Convert Spark DataFrame to Arrow batches.

        Uses Spark's native Arrow support when available for
        optimal performance and zero-copy conversion.

        Args:
            batch_size: Batch size for conversion.

        Returns:
            List of Arrow record batches.
        """
        import pyarrow as pa

        batch_size = batch_size or self._config.arrow_batch_size

        try:
            # Try native Arrow conversion (Spark 3.0+)
            # This is the most efficient path
            arrow_batches = self._df._collect_as_arrow()
            return arrow_batches
        except AttributeError:
            # Fallback: Convert via Pandas with Arrow
            logger.debug("Falling back to Pandas-based Arrow conversion")

            try:
                # Use toPandas with Arrow enabled
                pandas_df = self._df.toPandas()
                table = pa.Table.from_pandas(pandas_df)
                return table.to_batches(max_chunksize=batch_size)
            except Exception as e:
                logger.warning(f"Arrow conversion failed: {e}")
                # Last resort: manual conversion
                return self._manual_arrow_conversion(batch_size)

    def _manual_arrow_conversion(
        self,
        batch_size: int,
    ) -> list["pa.RecordBatch"]:
        """Manual Arrow conversion for older Spark versions.

        Args:
            batch_size: Batch size.

        Returns:
            List of Arrow record batches.
        """
        import pyarrow as pa

        # Collect data in batches
        batches = []
        schema = self._infer_arrow_schema()

        for partition in self._df.rdd.mapPartitions(
            lambda it: [list(it)]
        ).collect():
            if not partition:
                continue

            # Convert partition to dict of arrays
            data = {col: [] for col in self._columns}
            for row in partition:
                row_dict = row.asDict()
                for col in self._columns:
                    data[col].append(row_dict.get(col))

            # Create record batch
            batch = pa.RecordBatch.from_pydict(data, schema=schema)
            batches.append(batch)

        return batches

    def _infer_arrow_schema(self) -> "pa.Schema":
        """Infer Arrow schema from Spark schema."""
        import pyarrow as pa
        from pyspark.sql.types import (
            BooleanType,
            ByteType,
            DateType,
            DecimalType,
            DoubleType,
            FloatType,
            IntegerType,
            LongType,
            ShortType,
            StringType,
            TimestampType,
        )

        type_mapping = {
            ByteType: pa.int8(),
            ShortType: pa.int16(),
            IntegerType: pa.int32(),
            LongType: pa.int64(),
            FloatType: pa.float32(),
            DoubleType: pa.float64(),
            StringType: pa.string(),
            BooleanType: pa.bool_(),
            DateType: pa.date32(),
            TimestampType: pa.timestamp("us"),
        }

        fields = []
        for field in self._schema.fields:
            arrow_type = type_mapping.get(type(field.dataType), pa.string())
            if isinstance(field.dataType, DecimalType):
                arrow_type = pa.decimal128(
                    field.dataType.precision,
                    field.dataType.scale,
                )
            fields.append(pa.field(field.name, arrow_type, nullable=field.nullable))

        return pa.schema(fields)

    def _repartition(self, num_partitions: int) -> "SparkExecutionEngine":
        """Repartition the underlying DataFrame.

        Args:
            num_partitions: New number of partitions.

        Returns:
            New engine with repartitioned data.
        """
        repartitioned = self._df.repartition(num_partitions)
        return SparkExecutionEngine(repartitioned, self._config, self._spark)

    def coalesce(self, num_partitions: int) -> "SparkExecutionEngine":
        """Coalesce partitions (no shuffle).

        Args:
            num_partitions: New number of partitions.

        Returns:
            New engine with coalesced data.
        """
        coalesced = self._df.coalesce(num_partitions)
        return SparkExecutionEngine(coalesced, self._config, self._spark)

    # -------------------------------------------------------------------------
    # Core Operation Overrides (Native Spark)
    # -------------------------------------------------------------------------

    def count_rows(self) -> int:
        """Count rows using native Spark count."""
        if self._cached_row_count is not None:
            return self._cached_row_count

        cache_key = self._cache_key("count_rows")
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        count = self._df.count()
        self._cached_row_count = count
        self._set_cached(cache_key, count)
        return count

    def get_columns(self) -> list[str]:
        """Get column names."""
        return self._columns

    def count_nulls(self, column: str) -> int:
        """Count nulls using native Spark filter."""
        from pyspark.sql import functions as F

        cache_key = self._cache_key("count_nulls", column)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        count = self._df.filter(F.col(column).isNull()).count()
        self._set_cached(cache_key, count)
        return count

    def count_nulls_all(self) -> dict[str, int]:
        """Count nulls in all columns using batch aggregation."""
        from pyspark.sql import functions as F

        cache_key = self._cache_key("count_nulls_all")
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        # Single pass aggregation for all columns
        exprs = [
            F.sum(F.when(F.col(col).isNull(), 1).otherwise(0)).alias(f"{col}_nulls")
            for col in self._columns
        ]

        row = self._df.agg(*exprs).collect()[0]

        result = {
            col: row[f"{col}_nulls"] or 0
            for col in self._columns
        }

        self._set_cached(cache_key, result)
        return result

    def count_distinct(self, column: str) -> int:
        """Count distinct values using native Spark."""
        from pyspark.sql import functions as F

        cache_key = self._cache_key("count_distinct", column)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        count = self._df.select(F.countDistinct(column)).collect()[0][0]
        self._set_cached(cache_key, count)
        return count

    def get_stats(self, column: str) -> dict[str, Any]:
        """Get column statistics using native Spark aggregations."""
        from pyspark.sql import functions as F

        cache_key = self._cache_key("get_stats", column)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        # Single-pass aggregation for all stats
        row = self._df.agg(
            F.count(column).alias("count"),
            F.sum(F.when(F.col(column).isNull(), 1).otherwise(0)).alias("null_count"),
            F.avg(column).alias("mean"),
            F.stddev(column).alias("std"),
            F.min(column).alias("min"),
            F.max(column).alias("max"),
        ).collect()[0]

        stats = {
            "count": row["count"],
            "null_count": row["null_count"] or 0,
            "mean": row["mean"],
            "std": row["std"],
            "min": row["min"],
            "max": row["max"],
        }

        self._set_cached(cache_key, stats)
        return stats

    def get_quantiles(
        self,
        column: str,
        quantiles: list[float],
    ) -> list[float]:
        """Get quantiles using Spark's approxQuantile."""
        cache_key = self._cache_key("get_quantiles", column, tuple(quantiles))
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        # approxQuantile with 0.01 relative error
        result = self._df.approxQuantile(column, quantiles, 0.01)
        self._set_cached(cache_key, result)
        return result

    def get_value_counts(
        self,
        column: str,
        limit: int | None = None,
    ) -> dict[Any, int]:
        """Get value counts using native Spark groupBy."""
        from pyspark.sql import functions as F

        cache_key = self._cache_key("get_value_counts", column, limit)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        counts = (
            self._df.groupBy(column)
            .agg(F.count("*").alias("count"))
            .orderBy(F.desc("count"))
        )

        if limit:
            counts = counts.limit(limit)

        rows = counts.collect()
        result = {row[column]: row["count"] for row in rows}

        self._set_cached(cache_key, result)
        return result

    def count_duplicates(self, columns: list[str]) -> int:
        """Count duplicates using native Spark operations."""
        from pyspark.sql import functions as F

        cache_key = self._cache_key("count_duplicates", tuple(columns))
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        total = self.count_rows()
        unique = self._df.select(columns).distinct().count()
        duplicates = total - unique

        self._set_cached(cache_key, duplicates)
        return duplicates

    def count_matching_regex(self, column: str, pattern: str) -> int:
        """Count values matching regex using Spark rlike."""
        from pyspark.sql import functions as F

        cache_key = self._cache_key("count_matching_regex", column, pattern)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        count = self._df.filter(F.col(column).rlike(pattern)).count()
        self._set_cached(cache_key, count)
        return count

    def count_in_range(
        self,
        column: str,
        min_value: Any | None = None,
        max_value: Any | None = None,
        inclusive: bool = True,
    ) -> int:
        """Count values in range using native Spark filter."""
        from pyspark.sql import functions as F

        cache_key = self._cache_key(
            "count_in_range", column, min_value, max_value, inclusive
        )
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        condition = None

        if min_value is not None:
            if inclusive:
                condition = F.col(column) >= min_value
            else:
                condition = F.col(column) > min_value

        if max_value is not None:
            max_cond = (
                F.col(column) <= max_value
                if inclusive
                else F.col(column) < max_value
            )
            condition = condition & max_cond if condition is not None else max_cond

        if condition is None:
            count = self.count_rows()
        else:
            count = self._df.filter(condition).count()

        self._set_cached(cache_key, count)
        return count

    def count_in_set(self, column: str, values: set[Any]) -> int:
        """Count values in set using Spark isin."""
        from pyspark.sql import functions as F

        cache_key = self._cache_key("count_in_set", column, frozenset(values))
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        count = self._df.filter(F.col(column).isin(list(values))).count()
        self._set_cached(cache_key, count)
        return count

    # -------------------------------------------------------------------------
    # Sampling
    # -------------------------------------------------------------------------

    def sample(
        self,
        n: int = 1000,
        seed: int | None = None,
    ) -> "SparkExecutionEngine":
        """Create sampled engine using Spark's native sampling.

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

        if seed is not None:
            sampled = self._df.sample(
                withReplacement=False,
                fraction=fraction,
                seed=seed,
            )
        else:
            sampled = self._df.sample(withReplacement=False, fraction=fraction)

        sampled = sampled.limit(n)

        return SparkExecutionEngine(sampled, self._config, self._spark)

    # -------------------------------------------------------------------------
    # Spark-Specific Methods
    # -------------------------------------------------------------------------

    def persist(self, storage_level: str = "MEMORY_AND_DISK") -> "SparkExecutionEngine":
        """Persist the DataFrame.

        Args:
            storage_level: Spark storage level.

        Returns:
            Self after persisting.
        """
        from pyspark import StorageLevel

        levels = {
            "MEMORY_ONLY": StorageLevel.MEMORY_ONLY,
            "MEMORY_AND_DISK": StorageLevel.MEMORY_AND_DISK,
            "DISK_ONLY": StorageLevel.DISK_ONLY,
            "MEMORY_ONLY_SER": StorageLevel.MEMORY_ONLY_SER,
        }

        level = levels.get(storage_level, StorageLevel.MEMORY_AND_DISK)
        self._df.persist(level)
        return self

    def unpersist(self) -> "SparkExecutionEngine":
        """Unpersist the DataFrame.

        Returns:
            Self after unpersisting.
        """
        self._df.unpersist()
        return self

    def checkpoint(self) -> "SparkExecutionEngine":
        """Checkpoint the DataFrame for fault tolerance.

        Returns:
            New engine with checkpointed data.
        """
        if self._config.checkpoint_dir:
            self._spark.sparkContext.setCheckpointDir(self._config.checkpoint_dir)

        checkpointed = self._df.checkpoint()
        return SparkExecutionEngine(checkpointed, self._config, self._spark)

    def explain(self, extended: bool = False) -> str:
        """Get the execution plan.

        Args:
            extended: Show extended plan.

        Returns:
            Execution plan as string.
        """
        import io
        import sys

        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        try:
            self._df.explain(extended=extended)
            return buffer.getvalue()
        finally:
            sys.stdout = old_stdout

    def sql(self, query: str) -> "SparkExecutionEngine":
        """Execute SQL query on this DataFrame.

        Args:
            query: SQL query with {table} placeholder.

        Returns:
            New engine with query results.
        """
        # Register temp view
        view_name = f"truthound_temp_{id(self._df)}"
        self._df.createOrReplaceTempView(view_name)

        try:
            result_df = self._spark.sql(query.format(table=view_name))
            return SparkExecutionEngine(result_df, self._config, self._spark)
        finally:
            self._spark.catalog.dropTempView(view_name)
