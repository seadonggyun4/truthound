"""Data source implementations for Truthound.

This package provides data sources for various data backends,
enabling validation on different data formats and storage systems.

Supported data sources:
- Polars DataFrame/LazyFrame (primary)
- Pandas DataFrame
- PySpark DataFrame (with automatic sampling)
- File-based (CSV, JSON, Parquet)
- SQL databases (PostgreSQL, MySQL, SQLite)

Example:
    >>> from truthound.datasources import get_datasource
    >>>
    >>> # Auto-detect data source type
    >>> source = get_datasource("data.csv")
    >>> source = get_datasource(pandas_df)
    >>> source = get_datasource(polars_df)
    >>>
    >>> # Get execution engine for validation
    >>> engine = source.get_execution_engine()
"""

from truthound.datasources._protocols import (
    ColumnType,
    DataSourceCapability,
    DataSourceProtocol,
    ConnectableProtocol,
    SQLDataSourceProtocol,
)

from truthound.datasources.base import (
    BaseDataSource,
    DataSourceConfig,
    DataSourceError,
    DataSourceConnectionError,
    DataSourceSizeError,
    DataSourceSchemaError,
)

from truthound.datasources.polars_source import (
    PolarsDataSource,
    PolarsDataSourceConfig,
    FileDataSource,
    FileDataSourceConfig,
    DictDataSource,
)

from truthound.datasources.pandas_source import (
    PandasDataSource,
    PandasDataSourceConfig,
)

from truthound.datasources.spark_source import (
    SparkDataSource,
    SparkDataSourceConfig,
)

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

# SQL data sources are in subpackage
from truthound.datasources import sql

__all__ = [
    # Protocols
    "ColumnType",
    "DataSourceCapability",
    "DataSourceProtocol",
    "ConnectableProtocol",
    "SQLDataSourceProtocol",
    # Base classes
    "BaseDataSource",
    "DataSourceConfig",
    # Exceptions
    "DataSourceError",
    "DataSourceConnectionError",
    "DataSourceSizeError",
    "DataSourceSchemaError",
    # Polars sources
    "PolarsDataSource",
    "PolarsDataSourceConfig",
    "FileDataSource",
    "FileDataSourceConfig",
    "DictDataSource",
    # Pandas source
    "PandasDataSource",
    "PandasDataSourceConfig",
    # Spark source
    "SparkDataSource",
    "SparkDataSourceConfig",
    # Factory functions
    "get_datasource",
    "get_sql_datasource",
    "detect_datasource_type",
    "from_polars",
    "from_pandas",
    "from_spark",
    "from_file",
    "from_dict",
    # SQL subpackage
    "sql",
]
