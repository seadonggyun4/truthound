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
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

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
    sentinel_values: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.strategy = ArrowConversionStrategy(self.strategy)
        for value in (self.batch_size, self.max_memory_bytes):
            if type(value) is not int or value <= 0:
                raise ValueError("Arrow conversion limits must be positive integers")
        if self.null_handling not in {"mask", "sentinel", "error"}:
            raise ValueError("Unknown Arrow null policy")
        if type(self.preserve_index) is not bool or type(self.coerce_temporal) is not bool:
            raise ValueError("Arrow conversion flags must be boolean")
        if not isinstance(self.sentinel_values, dict):
            raise ValueError("Sentinel values must be a column mapping")


# =============================================================================
# Type Mapping
# =============================================================================


def _spark_type_to_arrow(spark_type: Any) -> pa.DataType:
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


def _polars_type_to_arrow(polars_type: Any) -> pa.DataType:
    """Convert Polars data type to Arrow data type.

    Args:
        polars_type: Polars data type.

    Returns:
        Corresponding Arrow data type.
    """
    import polars as pl
    import pyarrow as pa

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
        self._config = deepcopy(config) if config is not None else ArrowBridgeConfig()
        self._config.__post_init__()

    @property
    def config(self) -> ArrowBridgeConfig:
        """Get bridge configuration."""
        return deepcopy(self._config)

    # -------------------------------------------------------------------------
    # Spark <-> Polars
    # -------------------------------------------------------------------------

    def spark_to_polars(self, spark_df, collect=True):
        return self.to_polars(spark_df, collect=collect)

    def spark_to_polars_chunked(self, spark_df, chunk_size=None):
        """Collect with selected Spark strategy, then yield bounded row slices."""
        import polars as pl

        size = self._chunk_size(chunk_size)
        table = self.to_arrow(spark_df)
        for batch in table.to_batches(max_chunksize=size):
            yield pl.from_arrow(batch)

    def polars_to_spark(
        self,
        polars_df: pl.DataFrame | pl.LazyFrame,
        spark: Any,
    ) -> SparkDataFrame:
        """Convert Polars DataFrame to Spark.

        Args:
            polars_df: Polars DataFrame or LazyFrame.
            spark: SparkSession.

        Returns:
            Spark DataFrame.
        """
        import polars as pl

        if isinstance(polars_df, pl.LazyFrame):
            polars_df = polars_df.collect()

        # Convert to Arrow
        arrow_table = polars_df.to_arrow()

        table = self._normalize_table(arrow_table)
        return spark.createDataFrame(table.to_pandas())

    def _spark_to_arrow_native(
        self,
        spark_df: SparkDataFrame,
    ) -> pa.Table:
        """Convert Spark to Arrow using native support.

        Args:
            spark_df: Spark DataFrame.

        Returns:
            Arrow Table.
        """
        import pyarrow as pa

        collector = getattr(spark_df, "_collect_as_arrow", None)
        if collector is None:
            if self._config.strategy == ArrowConversionStrategy.NATIVE:
                raise NotImplementedError("Selected native Spark Arrow collector is unavailable")
            return self._spark_to_arrow_pandas(spark_df)
        batches = collector()
        schema = pa.schema(
            [
                pa.field(field.name, _spark_type_to_arrow(field.dataType), nullable=field.nullable)
                for field in spark_df.schema.fields
            ]
        )
        return pa.Table.from_batches(batches, schema=None if batches else schema)

    def _spark_to_arrow_pandas(
        self,
        spark_df: SparkDataFrame,
    ) -> pa.Table:
        """Convert Spark to Arrow via Pandas.

        Args:
            spark_df: Spark DataFrame.

        Returns:
            Arrow Table.
        """
        import pyarrow as pa

        pandas_df = spark_df.toPandas()
        return pa.Table.from_pandas(
            pandas_df,
            preserve_index=self._config.preserve_index,
        )

    def _spark_to_arrow_manual(
        self,
        spark_df: SparkDataFrame,
    ) -> pa.Table:
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
        for spark_field in spark_df.schema.fields:
            arrow_type = _spark_type_to_arrow(spark_field.dataType)
            arrow_fields.append(
                pa.field(spark_field.name, arrow_type, nullable=spark_field.nullable)
            )
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
        spark_df: SparkDataFrame,
        batch_size: int,
    ) -> Iterator[pa.RecordBatch]:
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
            for partition_rows in spark_df.rdd.mapPartitions(lambda it: [list(it)]).collect():
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
        spark_df: SparkDataFrame,
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

    def dask_to_polars(self, dask_df, collect=True):
        return self.to_polars(dask_df, collect=collect)

    def dask_to_polars_chunked(self, dask_df, chunk_size=None):
        """Materialize one Dask partition, then yield bounded row slices."""
        import polars as pl

        size = self._chunk_size(chunk_size)
        for index in range(dask_df.npartitions):
            table = self.to_arrow(dask_df.get_partition(index).compute())
            for batch in table.to_batches(max_chunksize=size):
                yield pl.from_arrow(batch)

    def polars_to_dask(
        self,
        polars_df: pl.DataFrame | pl.LazyFrame,
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

        selected = 4 if npartitions is None else npartitions
        if type(selected) is not int or selected <= 0:
            raise ValueError("Dask partition count must be a positive integer")
        pandas_df = self.to_arrow(polars_df).to_pandas()
        return dd.from_pandas(pandas_df, npartitions=selected)

    # -------------------------------------------------------------------------
    # Ray <-> Polars
    # -------------------------------------------------------------------------

    def ray_to_polars(self, ray_dataset, collect=True):
        return self.to_polars(ray_dataset, collect=collect)

    def ray_to_polars_chunked(self, ray_dataset, chunk_size=None):
        import polars as pl

        size = self._chunk_size(chunk_size)
        for table in self._ray_tables(ray_dataset, size):
            for part in table.to_batches(max_chunksize=size):
                yield pl.from_arrow(part)

    def polars_to_ray(
        self,
        polars_df: pl.DataFrame | pl.LazyFrame,
    ) -> Any:
        """Convert Polars DataFrame to Ray Dataset.

        Args:
            polars_df: Polars DataFrame or LazyFrame.

        Returns:
            Ray Dataset.
        """
        import polars as pl
        import ray

        if isinstance(polars_df, pl.LazyFrame):
            polars_df = polars_df.collect()

        arrow_table = self.to_arrow(polars_df)
        return ray.data.from_arrow(arrow_table)

    # -------------------------------------------------------------------------
    # Generic Methods
    # -------------------------------------------------------------------------

    def to_arrow(self, data):
        """Materialize supported input and enforce the captured Arrow policy."""
        pd = sys.modules.get("pandas")
        import polars as pl
        import pyarrow as pa

        if isinstance(data, pa.RecordBatch):
            table = pa.Table.from_batches([data])
        elif isinstance(data, pa.Table):
            table = data
        elif isinstance(data, (pl.DataFrame, pl.LazyFrame)):
            table = (data.collect() if isinstance(data, pl.LazyFrame) else data).to_arrow()
        elif pd is not None and isinstance(data, pd.DataFrame):
            table = pa.Table.from_pandas(data, preserve_index=self._config.preserve_index)
        elif hasattr(data, "sparkSession"):
            strategy = self._determine_strategy(data)
            conversion = {
                ArrowConversionStrategy.NATIVE: self._spark_to_arrow_native,
                ArrowConversionStrategy.PANDAS: self._spark_to_arrow_pandas,
                ArrowConversionStrategy.MANUAL: self._spark_to_arrow_manual,
            }[strategy]
            table = conversion(data)
        elif hasattr(data, "compute") and hasattr(data, "npartitions"):
            table = pa.Table.from_pandas(data.compute(), preserve_index=self._config.preserve_index)
        elif hasattr(data, "iter_batches"):
            tables = []
            size = 0
            for part in self._ray_tables(data, self._config.batch_size):
                size += part.nbytes
                if size > self._config.max_memory_bytes:
                    raise MemoryError("Arrow conversion exceeds selected byte budget")
                tables.append(part)
            if tables:
                table = pa.concat_tables(tables)
            else:
                schema = data.schema()
                schema = getattr(schema, "base_schema", schema)
                table = pa.Table.from_batches([], schema=schema)
        else:
            raise ValueError("Unsupported Arrow conversion input")
        return self._normalize_table(table)

    def _ray_tables(self, data, batch_size):
        import pyarrow as pa

        selected = data.schema()
        schema = getattr(selected, "base_schema", selected)
        if not isinstance(schema, pa.Schema):
            raise TypeError("Ray dataset did not expose an Arrow schema")
        for batch in data.iter_batches(batch_size=batch_size, batch_format="pyarrow"):
            table = pa.Table.from_batches([batch]) if isinstance(batch, pa.RecordBatch) else batch
            # A batch containing only nulls may carry Arrow null type even
            # when the dataset's resolved column type is numeric.
            table = table.cast(schema, safe=True)
            yield self._normalize_table(table)

    def _normalize_table(self, table):
        import pyarrow as pa
        import pyarrow.compute as pc

        if isinstance(table, pa.RecordBatch):
            table = pa.Table.from_batches([table])
        if not isinstance(table, pa.Table):
            raise TypeError("Conversion did not return an Arrow table")
        if table.nbytes > self._config.max_memory_bytes:
            raise MemoryError("Arrow conversion exceeds selected byte budget")
        columns = []
        for name, column in zip(table.column_names, table.columns, strict=True):
            if column.null_count and self._config.null_handling == "error":
                raise ValueError("Arrow null policy rejects null values")
            if column.null_count and self._config.null_handling == "sentinel":
                if name not in self._config.sentinel_values:
                    raise ValueError("Missing explicit sentinel for nullable column")
                scalar = pa.scalar(self._config.sentinel_values[name], type=column.type)
                if not scalar.is_valid:
                    raise ValueError("Sentinel must be non-null")
                column = pc.fill_null(column, scalar)
            if self._config.coerce_temporal:
                dtype = column.type
                target = None
                if pa.types.is_timestamp(dtype):
                    target = pa.timestamp("us", tz=dtype.tz)
                elif pa.types.is_duration(dtype):
                    target = pa.duration("us")
                elif pa.types.is_time(dtype):
                    target = pa.time64("us")
                elif pa.types.is_date64(dtype):
                    target = pa.date32()
                if target is not None:
                    column = column.cast(target, safe=True)
            columns.append(column)
        schema = pa.schema(
            [
                original.with_type(column.type)
                for original, column in zip(table.schema, columns, strict=True)
            ],
            metadata=table.schema.metadata,
        )
        table = pa.Table.from_arrays(columns, schema=schema)
        if table.nbytes > self._config.max_memory_bytes:
            raise MemoryError("Arrow conversion exceeds selected byte budget")
        return table

    def _chunk_size(self, selected):
        value = self._config.batch_size if selected is None else selected
        if type(value) is not int or value <= 0:
            raise ValueError("Arrow chunk size must be a positive integer")
        return value

    def to_polars(self, data, collect=True):
        """Apply conversion policy even when the input is already Polars."""
        import polars as pl

        frame = pl.from_arrow(self.to_arrow(data))
        return frame if collect else frame.lazy()


# =============================================================================
# Convenience Functions
# =============================================================================


def spark_to_polars(
    spark_df: SparkDataFrame,
    lazy: bool = True,
) -> pl.LazyFrame | pl.DataFrame:
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
    polars_df: pl.DataFrame | pl.LazyFrame,
    spark: Any,
) -> SparkDataFrame:
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
) -> pl.LazyFrame | pl.DataFrame:
    """Convert any supported data type to Polars.

    Args:
        data: Data to convert.
        lazy: If True, return LazyFrame; if False, return DataFrame.

    Returns:
        Polars DataFrame or LazyFrame.
    """
    bridge = ArrowBridge()
    return bridge.to_polars(data, collect=not lazy)
