"""Spark data source implementation.

This module provides a data source for PySpark DataFrames,
with automatic sampling for large datasets to prevent memory issues.

Requires: pip install pyspark
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from truthound.datasources._protocols import (
    ColumnType,
    DataSourceCapability,
)
from truthound.datasources.base import (
    BaseDataSource,
    DataSourceConfig,
    DataSourceError,
    DataSourceSizeError,
)

if TYPE_CHECKING:
    import polars as pl
    from pyspark.sql import DataFrame as SparkDataFrame
    from truthound.execution.base import BaseExecutionEngine


def _check_pyspark_available() -> None:
    """Check if PySpark is available."""
    try:
        import pyspark  # noqa: F401
    except ImportError:
        raise ImportError(
            "pyspark is required for SparkDataSource. "
            "Install with: pip install pyspark"
        )


def _spark_type_to_column_type(spark_type: Any) -> ColumnType:
    """Convert Spark data type to unified ColumnType."""
    from pyspark.sql.types import (
        ByteType, ShortType, IntegerType, LongType,
        FloatType, DoubleType, DecimalType,
        StringType, BinaryType, BooleanType,
        DateType, TimestampType, TimestampNTZType,
        ArrayType, MapType, StructType,
    )

    if isinstance(spark_type, (ByteType, ShortType, IntegerType, LongType)):
        return ColumnType.INTEGER
    if isinstance(spark_type, (FloatType, DoubleType)):
        return ColumnType.FLOAT
    if isinstance(spark_type, DecimalType):
        return ColumnType.DECIMAL
    if isinstance(spark_type, StringType):
        return ColumnType.STRING
    if isinstance(spark_type, BinaryType):
        return ColumnType.BINARY
    if isinstance(spark_type, BooleanType):
        return ColumnType.BOOLEAN
    if isinstance(spark_type, DateType):
        return ColumnType.DATE
    if isinstance(spark_type, (TimestampType, TimestampNTZType)):
        return ColumnType.DATETIME
    if isinstance(spark_type, ArrayType):
        return ColumnType.LIST
    if isinstance(spark_type, (MapType, StructType)):
        return ColumnType.STRUCT

    return ColumnType.UNKNOWN


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class SparkDataSourceConfig(DataSourceConfig):
    """Configuration for Spark data sources.

    Attributes:
        max_rows_for_local: Maximum rows to collect to driver for local operations.
        sampling_fraction: Fraction of data to sample when exceeding limits.
        persist_sampled: Whether to persist sampled DataFrame in memory.
        force_sampling: Always sample regardless of size.
        repartition_for_sampling: Repartition before sampling for better distribution.
    """

    max_rows_for_local: int = 100_000  # More conservative for Spark
    sampling_fraction: float | None = None  # Auto-calculate based on size
    persist_sampled: bool = True
    force_sampling: bool = False
    repartition_for_sampling: int | None = None


# =============================================================================
# Spark Data Source
# =============================================================================


class SparkDataSource(BaseDataSource[SparkDataSourceConfig]):
    """Data source for PySpark DataFrames.

    This data source handles large-scale data in Spark, automatically
    sampling when necessary to prevent memory issues during validation.

    WARNING: Many validation operations require collecting data to the
    driver node. For very large datasets, always use sampling.

    Example:
        >>> from pyspark.sql import SparkSession
        >>> spark = SparkSession.builder.getOrCreate()
        >>> df = spark.read.parquet("large_data.parquet")
        >>>
        >>> # With automatic sampling
        >>> source = SparkDataSource(df)
        >>> if source.needs_sampling():
        ...     source = source.sample(n=100_000)
        >>>
        >>> engine = source.get_execution_engine()
        >>> print(engine.count_rows())

        >>> # Force sampling for safety
        >>> source = SparkDataSource(df, force_sampling=True)
    """

    source_type = "spark"

    def __init__(
        self,
        data: "SparkDataFrame",
        config: SparkDataSourceConfig | None = None,
        force_sampling: bool = False,
    ) -> None:
        """Initialize Spark data source.

        Args:
            data: PySpark DataFrame.
            config: Optional configuration.
            force_sampling: Force sampling even for smaller datasets.
        """
        _check_pyspark_available()
        super().__init__(config)

        self._df = data
        self._spark_schema = data.schema

        if force_sampling:
            self._config.force_sampling = True

    @classmethod
    def _default_config(cls) -> SparkDataSourceConfig:
        return SparkDataSourceConfig()

    @property
    def spark_dataframe(self) -> "SparkDataFrame":
        """Get the underlying Spark DataFrame."""
        return self._df

    @property
    def spark_schema(self) -> Any:
        """Get the native Spark schema."""
        return self._spark_schema

    @property
    def schema(self) -> dict[str, ColumnType]:
        """Get the schema as column name to type mapping."""
        if self._cached_schema is None:
            self._cached_schema = {
                field.name: _spark_type_to_column_type(field.dataType)
                for field in self._spark_schema.fields
            }
        return self._cached_schema

    @property
    def columns(self) -> list[str]:
        """Get list of column names."""
        return self._df.columns

    @property
    def row_count(self) -> int | None:
        """Get row count.

        Note: This triggers a Spark action which may be expensive.
        """
        if self._cached_row_count is None:
            self._cached_row_count = self._df.count()
        return self._cached_row_count

    @property
    def capabilities(self) -> set[DataSourceCapability]:
        """Get data source capabilities."""
        return {
            DataSourceCapability.LAZY_EVALUATION,
            DataSourceCapability.SAMPLING,
            DataSourceCapability.SCHEMA_INFERENCE,
            DataSourceCapability.STREAMING,
        }

    def needs_sampling(self) -> bool:
        """Check if sampling is needed due to size.

        For Spark, we use a more conservative threshold.
        """
        if self._config.force_sampling:
            return True

        row_count = self.row_count
        if row_count is None:
            return True  # Unknown size, sample to be safe
        return row_count > self._config.max_rows_for_local

    def get_execution_engine(self) -> "BaseExecutionEngine":
        """Get an execution engine for this data source.

        Returns a Polars engine after converting sampled data.
        Direct Spark execution is not yet supported.
        """
        from truthound.execution.polars_engine import PolarsExecutionEngine

        # Convert to Polars (with sampling if needed)
        lf = self.to_polars_lazyframe()
        return PolarsExecutionEngine(lf, self._config)

    def sample(
        self,
        n: int = 1000,
        seed: int | None = None,
    ) -> "SparkDataSource":
        """Create a new data source with sampled data.

        Uses Spark's native sampling for efficiency.

        Args:
            n: Target number of rows.
            seed: Random seed for reproducibility.

        Returns:
            New SparkDataSource with sampled data.
        """
        row_count = self.row_count or 0

        if row_count <= n and not self._config.force_sampling:
            return self

        # Calculate sampling fraction
        if row_count > 0:
            # Over-sample slightly to account for sampling variance
            fraction = min((n * 1.1) / row_count, 1.0)
        else:
            fraction = 0.1  # Default if row count unknown

        # Apply sampling
        if seed is not None:
            sampled_df = self._df.sample(withReplacement=False, fraction=fraction, seed=seed)
        else:
            sampled_df = self._df.sample(withReplacement=False, fraction=fraction)

        # Limit to exact n if we over-sampled
        sampled_df = sampled_df.limit(n)

        # Optionally persist for performance
        if self._config.persist_sampled:
            sampled_df = sampled_df.persist()

        config = SparkDataSourceConfig(
            name=f"{self.name}_sample",
            max_rows=self._config.max_rows,
            max_rows_for_local=self._config.max_rows_for_local,
            sample_size=n,
            force_sampling=False,
        )

        return SparkDataSource(sampled_df, config)

    def to_polars_lazyframe(self) -> "pl.LazyFrame":
        """Convert to Polars LazyFrame.

        WARNING: This collects data to the driver node!
        For large datasets, sample first.

        Returns:
            Polars LazyFrame.

        Raises:
            DataSourceSizeError: If data exceeds size limits.
        """
        import polars as pl

        row_count = self.row_count or 0

        # Check size limits
        if row_count > self._config.max_rows_for_local:
            raise DataSourceSizeError(
                current_size=row_count,
                max_size=self._config.max_rows_for_local,
                unit="rows (Spark to local conversion limit)",
            )

        # Convert via Pandas (most reliable method)
        pandas_df = self._df.toPandas()
        return pl.from_pandas(pandas_df).lazy()

    def to_pandas(self) -> Any:
        """Convert to Pandas DataFrame.

        WARNING: This collects all data to the driver!
        """
        row_count = self.row_count or 0

        if row_count > self._config.max_rows_for_local:
            raise DataSourceSizeError(
                current_size=row_count,
                max_size=self._config.max_rows_for_local,
                unit="rows",
            )

        return self._df.toPandas()

    def validate_connection(self) -> bool:
        """Validate by checking if DataFrame is accessible."""
        try:
            # Check if we can access schema
            _ = self._df.schema
            return True
        except Exception:
            return False

    # -------------------------------------------------------------------------
    # Spark-specific Methods
    # -------------------------------------------------------------------------

    def repartition(self, num_partitions: int) -> "SparkDataSource":
        """Create a new data source with repartitioned data.

        Args:
            num_partitions: Number of partitions.

        Returns:
            New SparkDataSource with repartitioned data.
        """
        repartitioned = self._df.repartition(num_partitions)
        return SparkDataSource(repartitioned, self._config)

    def coalesce(self, num_partitions: int) -> "SparkDataSource":
        """Create a new data source with coalesced partitions.

        Args:
            num_partitions: Number of partitions.

        Returns:
            New SparkDataSource with coalesced data.
        """
        coalesced = self._df.coalesce(num_partitions)
        return SparkDataSource(coalesced, self._config)

    def persist(self) -> "SparkDataSource":
        """Persist the DataFrame in memory.

        Returns:
            Self after persisting.
        """
        self._df.persist()
        return self

    def unpersist(self) -> "SparkDataSource":
        """Unpersist the DataFrame from memory.

        Returns:
            Self after unpersisting.
        """
        self._df.unpersist()
        return self

    def cache(self) -> "SparkDataSource":
        """Cache the DataFrame (alias for persist).

        Returns:
            Self after caching.
        """
        self._df.cache()
        return self

    def explain(self, extended: bool = False) -> str:
        """Get the execution plan.

        Args:
            extended: If True, show extended plan.

        Returns:
            Execution plan as string.
        """
        import io
        import sys

        # Capture explain output
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        try:
            self._df.explain(extended=extended)
            return buffer.getvalue()
        finally:
            sys.stdout = old_stdout

    def get_num_partitions(self) -> int:
        """Get the number of partitions."""
        return self._df.rdd.getNumPartitions()

    def get_storage_level(self) -> str:
        """Get the storage level (if persisted)."""
        return str(self._df.storageLevel)

    @classmethod
    def from_table(
        cls,
        spark: Any,
        table_name: str,
        database: str | None = None,
        config: SparkDataSourceConfig | None = None,
    ) -> "SparkDataSource":
        """Create data source from a Spark table.

        Args:
            spark: SparkSession instance.
            table_name: Name of the table.
            database: Optional database name.
            config: Optional configuration.

        Returns:
            SparkDataSource for the table.

        Example:
            >>> source = SparkDataSource.from_table(spark, "users", database="mydb")
        """
        _check_pyspark_available()

        if database:
            full_name = f"{database}.{table_name}"
        else:
            full_name = table_name

        df = spark.table(full_name)

        if config is None:
            config = SparkDataSourceConfig(name=full_name)
        else:
            config.name = config.name or full_name

        return cls(df, config)

    @classmethod
    def from_parquet(
        cls,
        spark: Any,
        path: str,
        config: SparkDataSourceConfig | None = None,
    ) -> "SparkDataSource":
        """Create data source from Parquet files.

        Args:
            spark: SparkSession instance.
            path: Path to Parquet file(s).
            config: Optional configuration.

        Returns:
            SparkDataSource for the Parquet data.
        """
        _check_pyspark_available()

        df = spark.read.parquet(path)

        if config is None:
            config = SparkDataSourceConfig(name=path)

        return cls(df, config)
