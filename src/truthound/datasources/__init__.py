"""Data source implementations for Truthound.

This package provides data sources for various data backends,
enabling validation on different data formats and storage systems.

Supported data sources:
- Polars DataFrame/LazyFrame (primary)
- Pandas DataFrame (with optimized conversion)
- PySpark DataFrame (with automatic sampling)
- File-based (CSV, JSON, Parquet)
- SQL databases (PostgreSQL, MySQL, SQLite, etc.)
- NoSQL databases (MongoDB, Elasticsearch)
- Streaming platforms (Apache Kafka)

Sync Example:
    >>> from truthound.datasources import get_datasource
    >>>
    >>> # Auto-detect data source type
    >>> source = get_datasource("data.csv")
    >>> source = get_datasource(pandas_df)
    >>> source = get_datasource(polars_df)
    >>>
    >>> # Get execution engine for validation
    >>> engine = source.get_execution_engine()

Async Example:
    >>> from truthound.datasources import get_async_datasource
    >>>
    >>> # MongoDB async data source
    >>> source = await get_async_datasource(
    ...     "mongodb://localhost:27017",
    ...     database="mydb",
    ...     collection="users",
    ... )
    >>>
    >>> async with source:
    ...     schema = await source.get_schema_async()
    ...     lf = await source.to_polars_lazyframe_async()
"""

# Sync protocols
from truthound.datasources._protocols import (
    ColumnType,
    DataSourceCapability,
    DataSourceProtocol,
    ConnectableProtocol,
    SQLDataSourceProtocol,
)

# Async protocols
from truthound.datasources._async_protocols import (
    AsyncDataSourceProtocol,
    AsyncConnectableProtocol,
    AsyncStreamableProtocol,
    AsyncQueryableProtocol,
)

# Base classes
from truthound.datasources.base import (
    BaseDataSource,
    DataSourceConfig,
    DataSourceError,
    DataSourceConnectionError,
    DataSourceSizeError,
    DataSourceSchemaError,
)

# Async base classes
from truthound.datasources.async_base import (
    AsyncBaseDataSource,
    AsyncDataSourceConfig,
    AsyncConnectionPool,
    AsyncDataSourceError,
    AsyncConnectionPoolError,
    AsyncTimeoutError,
)

# Adapters
from truthound.datasources.adapters import (
    SyncToAsyncAdapter,
    AsyncToSyncAdapter,
    adapt_to_async,
    adapt_to_sync,
    is_async_source,
    is_sync_source,
)

# Polars sources
from truthound.datasources.polars_source import (
    PolarsDataSource,
    PolarsDataSourceConfig,
    FileDataSource,
    FileDataSourceConfig,
    DictDataSource,
)

# Pandas sources
from truthound.datasources.pandas_source import (
    PandasDataSource,
    PandasDataSourceConfig,
)

# Optimized Pandas sources
from truthound.datasources.pandas_optimized import (
    OptimizedPandasDataSource,
    OptimizedPandasConfig,
    DataFrameOptimizer,
    optimize_pandas_to_polars,
    estimate_polars_memory,
    get_optimal_chunk_size,
)

# Spark sources
from truthound.datasources.spark_source import (
    SparkDataSource,
    SparkDataSourceConfig,
)

# Sync factory functions
from truthound.datasources.factory import (
    get_datasource,
    get_sql_datasource,
    detect_datasource_type,
    from_polars,
    from_pandas,
    from_spark,
    from_file,
    from_dict,
)

# Async factory functions
from truthound.datasources.async_factory import (
    get_async_datasource,
    from_mongodb,
    from_elasticsearch,
    from_kafka,
    from_confluent,
    from_atlas,
    detect_async_datasource_type,
    is_native_async_source,
)

# Subpackages
from truthound.datasources import sql
from truthound.datasources import nosql
from truthound.datasources import streaming

__all__ = [
    # Sync Protocols
    "ColumnType",
    "DataSourceCapability",
    "DataSourceProtocol",
    "ConnectableProtocol",
    "SQLDataSourceProtocol",
    # Async Protocols
    "AsyncDataSourceProtocol",
    "AsyncConnectableProtocol",
    "AsyncStreamableProtocol",
    "AsyncQueryableProtocol",
    # Base classes
    "BaseDataSource",
    "DataSourceConfig",
    # Async base classes
    "AsyncBaseDataSource",
    "AsyncDataSourceConfig",
    "AsyncConnectionPool",
    # Exceptions
    "DataSourceError",
    "DataSourceConnectionError",
    "DataSourceSizeError",
    "DataSourceSchemaError",
    "AsyncDataSourceError",
    "AsyncConnectionPoolError",
    "AsyncTimeoutError",
    # Adapters
    "SyncToAsyncAdapter",
    "AsyncToSyncAdapter",
    "adapt_to_async",
    "adapt_to_sync",
    "is_async_source",
    "is_sync_source",
    # Polars sources
    "PolarsDataSource",
    "PolarsDataSourceConfig",
    "FileDataSource",
    "FileDataSourceConfig",
    "DictDataSource",
    # Pandas sources
    "PandasDataSource",
    "PandasDataSourceConfig",
    # Optimized Pandas sources
    "OptimizedPandasDataSource",
    "OptimizedPandasConfig",
    "DataFrameOptimizer",
    "optimize_pandas_to_polars",
    "estimate_polars_memory",
    "get_optimal_chunk_size",
    # Spark source
    "SparkDataSource",
    "SparkDataSourceConfig",
    # Sync factory functions
    "get_datasource",
    "get_sql_datasource",
    "detect_datasource_type",
    "from_polars",
    "from_pandas",
    "from_spark",
    "from_file",
    "from_dict",
    # Async factory functions
    "get_async_datasource",
    "from_mongodb",
    "from_elasticsearch",
    "from_kafka",
    "from_confluent",
    "from_atlas",
    "detect_async_datasource_type",
    "is_native_async_source",
    # Subpackages
    "sql",
    "nosql",
    "streaming",
]
