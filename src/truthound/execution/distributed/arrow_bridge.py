"""Arrow-based zero-copy bridge between distributed backends and Polars.

This module provides efficient data transfer between distributed
computing frameworks (Spark, Dask, Ray) and Polars using Apache Arrow
as the intermediate format.

Key Features:
- Zero-copy conversion when possible (same memory buffer)
- Chunked transfer for memory efficiency
- Automatic type mapping
- Fallback paths for unsupported types

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                        ArrowBridge                               │
    │                                                                  │
    │   ┌───────────────┐         ┌───────────────┐                   │
    │   │ Spark DataFrame│   →    │  Arrow Table  │   →  Polars       │
    │   │ (Native Arrow) │         │ (Zero-copy)   │      LazyFrame   │
    │   └───────────────┘         └───────────────┘                   │
    │                                                                  │
    │   ┌───────────────┐         ┌───────────────┐                   │
    │   │ Dask DataFrame │   →    │ Arrow Batches │   →  Polars       │
    │   │  (Pandas-based)│         │ (Chunked)     │      LazyFrame   │
    │   └───────────────┘         └───────────────┘                   │
    │                                                                  │
    │   ┌───────────────┐         ┌───────────────┐                   │
    │   │  Ray Dataset   │   →    │ Arrow Batches │   →  Polars       │
    │   │ (Native Arrow) │         │ (Streaming)   │      LazyFrame   │
    │   └───────────────┘         └───────────────┘                   │
    │                                                                  │
    └─────────────────────────────────────────────────────────────────┘

Example:
    >>> from truthound.execution.distributed.arrow_bridge import ArrowBridge
    >>>
    >>> # Convert Spark DataFrame to Polars
    >>> bridge = ArrowBridge()
    >>> polars_lf = bridge.spark_to_polars(spark_df)
    >>>
    >>> # Convert with chunking for large data
    >>> for chunk_lf in bridge.spark_to_polars_chunked(spark_df, chunk_size=100_000):
    ...     process(chunk_lf)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    import polars as pl
    import pyarrow as pa
    from pyspark.sql import DataFrame as SparkDataFrame

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


class ArrowConversionStrategy(str, Enum):
    """Strategies for Arrow conversion."""

    NATIVE = "native"  # Use native Arrow support (fastest)
    PANDAS = "pandas"  # Convert via Pandas (most compatible)
    MANUAL = "manual"  # Manual row-by-row conversion (slowest)
    AUTO = "auto"  # Auto-detect best strategy


@dataclass
class ArrowBridgeConfig:
    """Configuration for Arrow bridge.

    Attributes:
        strategy: Conversion strategy to use.
        batch_size: Batch size for chunked conversion.
        max_memory_bytes: Maximum memory to use for conversion.
        coerce_temporal: Coerce temporal types to standard formats.
        preserve_index: Preserve DataFrame index (Pandas).
        null_handling: How to handle nulls ("mask", "sentinel", "error").
    """

    strategy: ArrowConversionStrategy = ArrowConversionStrategy.AUTO
    batch_size: int = 65536
    max_memory_bytes: int = 1024 * 1024 * 1024  # 1GB
    coerce_temporal: bool = True
    preserve_index: bool = False
    null_handling: str = "mask"


# =============================================================================
# Type Mapping
# =============================================================================


def _spark_type_to_arrow(spark_type: Any) -> "pa.DataType":
    """Convert Spark data type to Arrow data type.

    Args:
        spark_type: Spark data type.

    Returns:
        Corresponding Arrow data type.
    """
    import pyarrow as pa
    from pyspark.sql.types import (
        ArrayType,
        BinaryType,
        BooleanType,
        ByteType,
        DateType,
        DecimalType,
        DoubleType,
        FloatType,
        IntegerType,
        LongType,
        MapType,
        ShortType,
        StringType,
        StructField,
        StructType,
        TimestampType,
    )

    type_map = {
        BooleanType: pa.bool_,
        ByteType: pa.int8,
        ShortType: pa.int16,
        IntegerType: pa.int32,
        LongType: pa.int64,
        FloatType: pa.float32,
        DoubleType: pa.float64,
        StringType: pa.string,
        BinaryType: pa.binary,
        DateType: pa.date32,
    }

    spark_type_class = type(spark_type)

    if spark_type_class in type_map:
        return type_map[spark_type_class]()

    if spark_type_class == TimestampType:
        return pa.timestamp("us")

    if spark_type_class == DecimalType:
        return pa.decimal128(spark_type.precision, spark_type.scale)

    if spark_type_class == ArrayType:
        element_type = _spark_type_to_arrow(spark_type.elementType)
        return pa.list_(element_type)

    if spark_type_class == MapType:
        key_type = _spark_type_to_arrow(spark_type.keyType)
        value_type = _spark_type_to_arrow(spark_type.valueType)
        return pa.map_(key_type, value_type)

    if spark_type_class == StructType:
        fields = []
        for field in spark_type.fields:
            arrow_type = _spark_type_to_arrow(field.dataType)
            fields.append(pa.field(field.name, arrow_type, nullable=field.nullable))
        return pa.struct(fields)

    # Default to string for unknown types
    logger.warning(f"Unknown Spark type {spark_type}, defaulting to string")
    return pa.string()


def _polars_type_to_arrow(polars_type: Any) -> "pa.DataType":
    """Convert Polars data type to Arrow data type.

    Args:
        polars_type: Polars data type.

    Returns:
        Corresponding Arrow data type.
    """
    import pyarrow as pa
    import polars as pl

    type_map = {
        pl.Boolean: pa.bool_,
        pl.Int8: pa.int8,
        pl.Int16: pa.int16,
        pl.Int32: pa.int32,
        pl.Int64: pa.int64,
        pl.UInt8: pa.uint8,
        pl.UInt16: pa.uint16,
        pl.UInt32: pa.uint32,
        pl.UInt64: pa.uint64,
        pl.Float32: pa.float32,
        pl.Float64: pa.float64,
        pl.String: pa.string,
        pl.Utf8: pa.string,
        pl.Binary: pa.binary,
        pl.Date: pa.date32,
        pl.Time: lambda: pa.time64("us"),
        pl.Datetime: lambda: pa.timestamp("us"),
        pl.Duration: lambda: pa.duration("us"),
    }

    polars_type_class = type(polars_type)

    if polars_type_class in type_map:
        result = type_map[polars_type_class]
        return result() if callable(result) else result

    if polars_type_class == pl.List:
        inner_type = _polars_type_to_arrow(polars_type.inner)
        return pa.list_(inner_type)

    if polars_type_class == pl.Struct:
        fields = []
        for field in polars_type.fields:
            arrow_type = _polars_type_to_arrow(field.dtype)
            fields.append(pa.field(field.name, arrow_type))
        return pa.struct(fields)

    # Default to string
    return pa.string()


# =============================================================================
# Arrow Bridge
# =============================================================================


class ArrowBridge:
    """Bridge for converting between distributed data and Polars via Arrow.

    This class provides efficient conversion between Spark, Dask, Ray,
    and Polars using Apache Arrow as the intermediate format.

    Features:
    - Zero-copy when possible (native Arrow support)
    - Chunked conversion for memory efficiency
    - Automatic type mapping
    - Multiple fallback strategies

    Example:
        >>> bridge = ArrowBridge()
        >>>
        >>> # Convert Spark to Polars
        >>> polars_lf = bridge.spark_to_polars(spark_df)
        >>>
        >>> # Convert Polars to Spark
        >>> spark_df = bridge.polars_to_spark(polars_df, spark)
    """

    def __init__(self, config: ArrowBridgeConfig | None = None) -> None:
        """Initialize Arrow bridge.

        Args:
            config: Optional configuration.
        """
        self._config = config or ArrowBridgeConfig()

    @property
    def config(self) -> ArrowBridgeConfig:
        """Get bridge configuration."""
        return self._config

    # -------------------------------------------------------------------------
    # Spark <-> Polars
    # -------------------------------------------------------------------------

    def spark_to_polars(
        self,
        spark_df: "SparkDataFrame",
        collect: bool = True,
    ) -> "pl.LazyFrame | pl.DataFrame":
        """Convert Spark DataFrame to Polars.

        Args:
            spark_df: Spark DataFrame.
            collect: If True, return DataFrame; if False, return LazyFrame.

        Returns:
            Polars DataFrame or LazyFrame.
        """
        import polars as pl

        strategy = self._determine_strategy(spark_df)

        if strategy == ArrowConversionStrategy.NATIVE:
            table = self._spark_to_arrow_native(spark_df)
        elif strategy == ArrowConversionStrategy.PANDAS:
            table = self._spark_to_arrow_pandas(spark_df)
        else:
            table = self._spark_to_arrow_manual(spark_df)

        df = pl.from_arrow(table)

        if collect:
            return df
        return df.lazy()

    def spark_to_polars_chunked(
        self,
        spark_df: "SparkDataFrame",
        chunk_size: int | None = None,
    ) -> Iterator["pl.DataFrame"]:
        """Convert Spark DataFrame to Polars in chunks.

        This is useful for very large DataFrames that don't fit in memory.

        Args:
            spark_df: Spark DataFrame.
            chunk_size: Rows per chunk.

        Yields:
            Polars DataFrames for each chunk.
        """
        import polars as pl

        chunk_size = chunk_size or self._config.batch_size

        # Get Arrow batches
        batches = self._spark_to_arrow_batches(spark_df, chunk_size)

        for batch in batches:
            import pyarrow as pa

            table = pa.Table.from_batches([batch])
            yield pl.from_arrow(table)

    def polars_to_spark(
        self,
        polars_df: "pl.DataFrame | pl.LazyFrame",
        spark: Any,
    ) -> "SparkDataFrame":
        """Convert Polars DataFrame to Spark.

        Args:
            polars_df: Polars DataFrame or LazyFrame.
            spark: SparkSession.

        Returns:
            Spark DataFrame.
        """
        import polars as pl
        import pyarrow as pa

        if isinstance(polars_df, pl.LazyFrame):
            polars_df = polars_df.collect()

        # Convert to Arrow
        arrow_table = polars_df.to_arrow()

        # Try native Arrow conversion (Spark 3.0+)
        try:
            return spark.createDataFrame(arrow_table.to_pandas())
        except Exception:
            # Fallback to Pandas
            return spark.createDataFrame(polars_df.to_pandas())

    def _spark_to_arrow_native(
        self,
        spark_df: "SparkDataFrame",
    ) -> "pa.Table":
        """Convert Spark to Arrow using native support.

        Args:
            spark_df: Spark DataFrame.

        Returns:
            Arrow Table.
        """
        import pyarrow as pa

        try:
            # Spark 3.0+ supports _collect_as_arrow()
            batches = spark_df._collect_as_arrow()
            return pa.Table.from_batches(batches)
        except AttributeError:
            # Fallback to Pandas path
            return self._spark_to_arrow_pandas(spark_df)

    def _spark_to_arrow_pandas(
        self,
        spark_df: "SparkDataFrame",
    ) -> "pa.Table":
        """Convert Spark to Arrow via Pandas.

        Args:
            spark_df: Spark DataFrame.

        Returns:
            Arrow Table.
        """
        import pyarrow as pa

        # Enable Arrow optimization in Spark
        spark_df.sparkSession.conf.set(
            "spark.sql.execution.arrow.pyspark.enabled",
            "true",
        )

        pandas_df = spark_df.toPandas()
        return pa.Table.from_pandas(
            pandas_df,
            preserve_index=self._config.preserve_index,
        )

    def _spark_to_arrow_manual(
        self,
        spark_df: "SparkDataFrame",
    ) -> "pa.Table":
        """Convert Spark to Arrow manually.

        This is the slowest but most compatible path.

        Args:
            spark_df: Spark DataFrame.

        Returns:
            Arrow Table.
        """
        import pyarrow as pa

        # Infer Arrow schema
        arrow_fields = []
        for field in spark_df.schema.fields:
            arrow_type = _spark_type_to_arrow(field.dataType)
            arrow_fields.append(pa.field(field.name, arrow_type, nullable=field.nullable))
        arrow_schema = pa.schema(arrow_fields)

        # Collect data
        columns = spark_df.columns
        data = {col: [] for col in columns}

        for row in spark_df.collect():
            row_dict = row.asDict()
            for col in columns:
                data[col].append(row_dict.get(col))

        return pa.Table.from_pydict(data, schema=arrow_schema)

    def _spark_to_arrow_batches(
        self,
        spark_df: "SparkDataFrame",
        batch_size: int,
    ) -> Iterator["pa.RecordBatch"]:
        """Convert Spark to Arrow batches.

        Args:
            spark_df: Spark DataFrame.
            batch_size: Batch size.

        Yields:
            Arrow RecordBatches.
        """
        import pyarrow as pa

        try:
            # Try native Arrow batches
            batches = spark_df._collect_as_arrow()
            for batch in batches:
                yield batch
        except AttributeError:
            # Fallback: collect partitions separately
            for partition_rows in spark_df.rdd.mapPartitions(
                lambda it: [list(it)]
            ).collect():
                if not partition_rows:
                    continue

                columns = spark_df.columns
                data = {col: [] for col in columns}

                for row in partition_rows:
                    row_dict = row.asDict()
                    for col in columns:
                        data[col].append(row_dict.get(col))

                # Create batch
                batch = pa.RecordBatch.from_pydict(data)
                yield batch

    def _determine_strategy(
        self,
        spark_df: "SparkDataFrame",
    ) -> ArrowConversionStrategy:
        """Determine best conversion strategy.

        Args:
            spark_df: Spark DataFrame.

        Returns:
            Best strategy to use.
        """
        if self._config.strategy != ArrowConversionStrategy.AUTO:
            return self._config.strategy

        # Check for native Arrow support
        try:
            # Spark 3.0+ has native Arrow support
            version = spark_df.sparkSession.version
            major_version = int(version.split(".")[0])
            if major_version >= 3:
                return ArrowConversionStrategy.NATIVE
        except Exception:
            pass

        # Check if Arrow is enabled
        try:
            arrow_enabled = spark_df.sparkSession.conf.get(
                "spark.sql.execution.arrow.pyspark.enabled",
                "false",
            )
            if arrow_enabled.lower() == "true":
                return ArrowConversionStrategy.PANDAS
        except Exception:
            pass

        # Default to Pandas
        return ArrowConversionStrategy.PANDAS

    # -------------------------------------------------------------------------
    # Dask <-> Polars
    # -------------------------------------------------------------------------

    def dask_to_polars(
        self,
        dask_df: Any,
        collect: bool = True,
    ) -> "pl.LazyFrame | pl.DataFrame":
        """Convert Dask DataFrame to Polars.

        Args:
            dask_df: Dask DataFrame.
            collect: If True, return DataFrame; if False, return LazyFrame.

        Returns:
            Polars DataFrame or LazyFrame.
        """
        import polars as pl
        import pyarrow as pa

        # Convert via Pandas with Arrow
        pandas_df = dask_df.compute()
        arrow_table = pa.Table.from_pandas(
            pandas_df,
            preserve_index=self._config.preserve_index,
        )

        df = pl.from_arrow(arrow_table)

        if collect:
            return df
        return df.lazy()

    def dask_to_polars_chunked(
        self,
        dask_df: Any,
        chunk_size: int | None = None,
    ) -> Iterator["pl.DataFrame"]:
        """Convert Dask DataFrame to Polars in chunks.

        Args:
            dask_df: Dask DataFrame.
            chunk_size: Rows per chunk (ignored, uses partitions).

        Yields:
            Polars DataFrames for each partition.
        """
        import polars as pl
        import pyarrow as pa

        for i in range(dask_df.npartitions):
            pandas_df = dask_df.get_partition(i).compute()
            arrow_table = pa.Table.from_pandas(pandas_df)
            yield pl.from_arrow(arrow_table)

    def polars_to_dask(
        self,
        polars_df: "pl.DataFrame | pl.LazyFrame",
        npartitions: int | None = None,
    ) -> Any:
        """Convert Polars DataFrame to Dask.

        Args:
            polars_df: Polars DataFrame or LazyFrame.
            npartitions: Number of partitions.

        Returns:
            Dask DataFrame.
        """
        import dask.dataframe as dd
        import polars as pl

        if isinstance(polars_df, pl.LazyFrame):
            polars_df = polars_df.collect()

        pandas_df = polars_df.to_pandas()

        if npartitions:
            return dd.from_pandas(pandas_df, npartitions=npartitions)
        return dd.from_pandas(pandas_df, npartitions=4)

    # -------------------------------------------------------------------------
    # Ray <-> Polars
    # -------------------------------------------------------------------------

    def ray_to_polars(
        self,
        ray_dataset: Any,
        collect: bool = True,
    ) -> "pl.LazyFrame | pl.DataFrame":
        """Convert Ray Dataset to Polars.

        Args:
            ray_dataset: Ray Dataset.
            collect: If True, return DataFrame; if False, return LazyFrame.

        Returns:
            Polars DataFrame or LazyFrame.
        """
        import polars as pl

        # Ray datasets support Arrow natively
        arrow_table = ray_dataset.to_arrow()
        df = pl.from_arrow(arrow_table)

        if collect:
            return df
        return df.lazy()

    def ray_to_polars_chunked(
        self,
        ray_dataset: Any,
        chunk_size: int | None = None,
    ) -> Iterator["pl.DataFrame"]:
        """Convert Ray Dataset to Polars in chunks.

        Args:
            ray_dataset: Ray Dataset.
            chunk_size: Rows per chunk.

        Yields:
            Polars DataFrames for each chunk.
        """
        import polars as pl
        import pyarrow as pa

        batch_size = chunk_size or self._config.batch_size

        for batch in ray_dataset.iter_batches(
            batch_size=batch_size,
            batch_format="pyarrow",
        ):
            if isinstance(batch, pa.Table):
                yield pl.from_arrow(batch)
            else:
                # RecordBatch
                table = pa.Table.from_batches([batch])
                yield pl.from_arrow(table)

    def polars_to_ray(
        self,
        polars_df: "pl.DataFrame | pl.LazyFrame",
    ) -> Any:
        """Convert Polars DataFrame to Ray Dataset.

        Args:
            polars_df: Polars DataFrame or LazyFrame.

        Returns:
            Ray Dataset.
        """
        import ray
        import polars as pl

        if isinstance(polars_df, pl.LazyFrame):
            polars_df = polars_df.collect()

        arrow_table = polars_df.to_arrow()
        return ray.data.from_arrow(arrow_table)

    # -------------------------------------------------------------------------
    # Generic Methods
    # -------------------------------------------------------------------------

    def to_arrow(self, data: Any) -> "pa.Table":
        """Convert any supported data type to Arrow Table.

        Args:
            data: Data to convert.

        Returns:
            Arrow Table.
        """
        import pyarrow as pa
        import polars as pl

        if isinstance(data, pa.Table):
            return data

        if isinstance(data, (pl.DataFrame, pl.LazyFrame)):
            if isinstance(data, pl.LazyFrame):
                data = data.collect()
            return data.to_arrow()

        # Check for Spark
        if hasattr(data, "sparkSession"):
            return self._spark_to_arrow_native(data)

        # Check for Dask
        if hasattr(data, "compute") and hasattr(data, "npartitions"):
            pandas_df = data.compute()
            return pa.Table.from_pandas(pandas_df)

        # Check for Ray Dataset
        if hasattr(data, "to_arrow"):
            return data.to_arrow()

        # Check for Pandas
        try:
            import pandas as pd
            if isinstance(data, pd.DataFrame):
                return pa.Table.from_pandas(data)
        except ImportError:
            pass

        raise ValueError(f"Cannot convert {type(data)} to Arrow")

    def to_polars(
        self,
        data: Any,
        collect: bool = True,
    ) -> "pl.LazyFrame | pl.DataFrame":
        """Convert any supported data type to Polars.

        Args:
            data: Data to convert.
            collect: If True, return DataFrame; if False, return LazyFrame.

        Returns:
            Polars DataFrame or LazyFrame.
        """
        import polars as pl

        if isinstance(data, pl.DataFrame):
            return data if collect else data.lazy()

        if isinstance(data, pl.LazyFrame):
            return data.collect() if collect else data

        # Convert via Arrow
        arrow_table = self.to_arrow(data)
        df = pl.from_arrow(arrow_table)

        if collect:
            return df
        return df.lazy()


# =============================================================================
# Convenience Functions
# =============================================================================


def spark_to_polars(
    spark_df: "SparkDataFrame",
    lazy: bool = True,
) -> "pl.LazyFrame | pl.DataFrame":
    """Convert Spark DataFrame to Polars.

    Args:
        spark_df: Spark DataFrame.
        lazy: If True, return LazyFrame; if False, return DataFrame.

    Returns:
        Polars DataFrame or LazyFrame.
    """
    bridge = ArrowBridge()
    return bridge.spark_to_polars(spark_df, collect=not lazy)


def polars_to_spark(
    polars_df: "pl.DataFrame | pl.LazyFrame",
    spark: Any,
) -> "SparkDataFrame":
    """Convert Polars DataFrame to Spark.

    Args:
        polars_df: Polars DataFrame or LazyFrame.
        spark: SparkSession.

    Returns:
        Spark DataFrame.
    """
    bridge = ArrowBridge()
    return bridge.polars_to_spark(polars_df, spark)


def convert_to_polars(
    data: Any,
    lazy: bool = True,
) -> "pl.LazyFrame | pl.DataFrame":
    """Convert any supported data type to Polars.

    Args:
        data: Data to convert.
        lazy: If True, return LazyFrame; if False, return DataFrame.

    Returns:
        Polars DataFrame or LazyFrame.
    """
    bridge = ArrowBridge()
    return bridge.to_polars(data, collect=not lazy)
