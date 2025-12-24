"""Factory functions for creating data sources.

This module provides factory functions that automatically detect the
appropriate data source type based on input data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TYPE_CHECKING

from truthound.datasources._protocols import DataSourceProtocol
from truthound.datasources.base import DataSourceConfig, DataSourceError

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl
    from pyspark.sql import DataFrame as SparkDataFrame


# =============================================================================
# Type Detection
# =============================================================================


def _is_polars_dataframe(obj: Any) -> bool:
    """Check if object is a Polars DataFrame."""
    return type(obj).__name__ == "DataFrame" and type(obj).__module__.startswith("polars")


def _is_polars_lazyframe(obj: Any) -> bool:
    """Check if object is a Polars LazyFrame."""
    return type(obj).__name__ == "LazyFrame" and type(obj).__module__.startswith("polars")


def _is_pandas_dataframe(obj: Any) -> bool:
    """Check if object is a Pandas DataFrame."""
    return type(obj).__name__ == "DataFrame" and type(obj).__module__.startswith("pandas")


def _is_spark_dataframe(obj: Any) -> bool:
    """Check if object is a PySpark DataFrame."""
    type_name = type(obj).__name__
    module = type(obj).__module__
    return type_name == "DataFrame" and "pyspark" in module


def _is_dict(obj: Any) -> bool:
    """Check if object is a dictionary suitable for DataFrame conversion."""
    if not isinstance(obj, dict):
        return False
    # Check if it looks like columnar data
    if not obj:
        return True
    first_value = next(iter(obj.values()))
    return isinstance(first_value, (list, tuple))


def _detect_file_type(path: str | Path) -> str | None:
    """Detect file type from path.

    Returns:
        File type string or None if unsupported.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    supported = {
        ".csv": "csv",
        ".json": "json",
        ".parquet": "parquet",
        ".pq": "parquet",
        ".ndjson": "ndjson",
        ".jsonl": "ndjson",
    }

    return supported.get(suffix)


# =============================================================================
# Factory Functions
# =============================================================================


def get_datasource(
    data: Any,
    name: str | None = None,
    **kwargs: Any,
) -> DataSourceProtocol:
    """Create an appropriate data source for the given data.

    This function auto-detects the type of input data and returns
    the appropriate data source implementation.

    Supported input types:
    - Polars DataFrame or LazyFrame
    - Pandas DataFrame
    - PySpark DataFrame
    - Python dictionary (columnar format)
    - File path (CSV, JSON, Parquet)
    - SQL connection string (with table parameter)

    Args:
        data: Input data in any supported format.
        name: Optional name for the data source.
        **kwargs: Additional arguments passed to the data source constructor.

    Returns:
        Appropriate DataSource implementation.

    Raises:
        DataSourceError: If the input type is not supported.

    Example:
        >>> import polars as pl
        >>> df = pl.DataFrame({"a": [1, 2, 3]})
        >>> source = get_datasource(df)

        >>> source = get_datasource("data.csv")

        >>> source = get_datasource({"a": [1, 2], "b": [3, 4]})
    """
    # Polars LazyFrame
    if _is_polars_lazyframe(data):
        from truthound.datasources.polars_source import (
            PolarsDataSource,
            PolarsDataSourceConfig,
        )
        config = PolarsDataSourceConfig(name=name, **kwargs) if name or kwargs else None
        return PolarsDataSource(data, config)

    # Polars DataFrame
    if _is_polars_dataframe(data):
        from truthound.datasources.polars_source import (
            PolarsDataSource,
            PolarsDataSourceConfig,
        )
        config = PolarsDataSourceConfig(name=name, **kwargs) if name or kwargs else None
        return PolarsDataSource(data, config)

    # Pandas DataFrame
    if _is_pandas_dataframe(data):
        from truthound.datasources.pandas_source import (
            PandasDataSource,
            PandasDataSourceConfig,
        )
        config = PandasDataSourceConfig(name=name, **kwargs) if name or kwargs else None
        return PandasDataSource(data, config)

    # PySpark DataFrame
    if _is_spark_dataframe(data):
        from truthound.datasources.spark_source import (
            SparkDataSource,
            SparkDataSourceConfig,
        )
        config = SparkDataSourceConfig(name=name, **kwargs) if name or kwargs else None
        return SparkDataSource(data, config)

    # Dictionary (columnar data)
    if _is_dict(data):
        from truthound.datasources.polars_source import (
            DictDataSource,
            PolarsDataSourceConfig,
        )
        config = PolarsDataSourceConfig(name=name, **kwargs) if name or kwargs else None
        return DictDataSource(data, config)

    # File path (string or Path)
    if isinstance(data, (str, Path)):
        path = Path(data)

        # Check if it's a file
        if path.exists() and path.is_file():
            file_type = _detect_file_type(path)
            if file_type:
                from truthound.datasources.polars_source import (
                    FileDataSource,
                    FileDataSourceConfig,
                )
                config = FileDataSourceConfig(name=name, **kwargs) if name or kwargs else None
                return FileDataSource(path, config)
            else:
                raise DataSourceError(
                    f"Unsupported file type: {path.suffix}. "
                    "Supported: .csv, .json, .parquet, .ndjson, .jsonl"
                )

        # Check if it might be a SQL connection string
        if isinstance(data, str):
            if data.startswith(("postgresql://", "postgres://")):
                table = kwargs.pop("table", None)
                if not table:
                    raise DataSourceError(
                        "SQL connection string requires 'table' parameter"
                    )
                from truthound.datasources.sql import PostgreSQLDataSource
                return PostgreSQLDataSource.from_connection_string(
                    data, table=table, **kwargs
                )

            if data.startswith("mysql://"):
                table = kwargs.pop("table", None)
                if not table:
                    raise DataSourceError(
                        "SQL connection string requires 'table' parameter"
                    )
                from truthound.datasources.sql import MySQLDataSource
                return MySQLDataSource.from_connection_string(
                    data, table=table, **kwargs
                )

        # File doesn't exist
        if not path.exists():
            raise DataSourceError(f"File not found: {path}")

    raise DataSourceError(
        f"Unsupported data type: {type(data).__name__}. "
        "Supported types: Polars DataFrame/LazyFrame, Pandas DataFrame, "
        "PySpark DataFrame, dict, file path (CSV/JSON/Parquet), "
        "SQL connection string"
    )


def get_sql_datasource(
    connection_string: str,
    table: str,
    **kwargs: Any,
) -> DataSourceProtocol:
    """Create a SQL data source from connection string.

    Args:
        connection_string: Database connection string.
        table: Table name to validate.
        **kwargs: Additional arguments for the data source.

    Returns:
        Appropriate SQL DataSource implementation.

    Raises:
        DataSourceError: If the connection string format is not supported.

    Example:
        >>> source = get_sql_datasource(
        ...     "postgresql://user:pass@host:5432/db",
        ...     table="users",
        ... )
    """
    if connection_string.startswith(("postgresql://", "postgres://")):
        from truthound.datasources.sql import PostgreSQLDataSource
        return PostgreSQLDataSource.from_connection_string(
            connection_string, table=table, **kwargs
        )

    if connection_string.startswith("mysql://"):
        from truthound.datasources.sql import MySQLDataSource
        return MySQLDataSource.from_connection_string(
            connection_string, table=table, **kwargs
        )

    if connection_string.endswith(".db") or connection_string == ":memory:":
        from truthound.datasources.sql import SQLiteDataSource
        return SQLiteDataSource(table=table, database=connection_string, **kwargs)

    # Oracle
    if connection_string.startswith("oracle://") or "oracle" in connection_string.lower():
        from truthound.datasources.sql import OracleDataSource
        if OracleDataSource is None:
            raise DataSourceError(
                "Oracle support requires oracledb. "
                "Install with: pip install oracledb"
            )
        # Parse oracle://user:pass@host:port/service
        # For now, require explicit parameters
        raise DataSourceError(
            "Oracle connection string parsing not implemented. "
            "Use OracleDataSource directly with explicit parameters."
        )

    # SQL Server
    if connection_string.startswith(("mssql://", "sqlserver://")):
        from truthound.datasources.sql import SQLServerDataSource
        if SQLServerDataSource is None:
            raise DataSourceError(
                "SQL Server support requires pyodbc or pymssql. "
                "Install with: pip install pyodbc"
            )
        return SQLServerDataSource.from_connection_string(
            connection_string, table=table, **kwargs
        )

    # Redshift (uses postgresql:// but with redshift host)
    if "redshift.amazonaws.com" in connection_string:
        from truthound.datasources.sql import RedshiftDataSource
        if RedshiftDataSource is None:
            raise DataSourceError(
                "Redshift support requires redshift-connector. "
                "Install with: pip install redshift-connector"
            )
        # Parse postgresql://user:pass@cluster.region.redshift.amazonaws.com:5439/db
        # For now, require explicit parameters
        raise DataSourceError(
            "Redshift connection string parsing not implemented. "
            "Use RedshiftDataSource directly with explicit parameters."
        )

    raise DataSourceError(
        f"Unsupported SQL connection string format: {connection_string}. "
        "Supported: postgresql://, mysql://, mssql://, SQLite file path. "
        "For BigQuery, Snowflake, Redshift, Databricks, use their specific classes."
    )


def detect_datasource_type(data: Any) -> str:
    """Detect the type of data source for given data.

    Args:
        data: Input data.

    Returns:
        Data source type string.

    Example:
        >>> import polars as pl
        >>> detect_datasource_type(pl.DataFrame({"a": [1]}))
        'polars'
    """
    if _is_polars_lazyframe(data):
        return "polars_lazy"
    if _is_polars_dataframe(data):
        return "polars"
    if _is_pandas_dataframe(data):
        return "pandas"
    if _is_spark_dataframe(data):
        return "spark"
    if _is_dict(data):
        return "dict"
    if isinstance(data, (str, Path)):
        path = Path(data) if isinstance(data, str) else data
        if path.exists() and path.is_file():
            return f"file:{_detect_file_type(path) or 'unknown'}"
        if isinstance(data, str):
            if "postgresql" in data or "postgres" in data:
                return "postgresql"
            if "mysql" in data:
                return "mysql"
            if data.endswith(".db") or data == ":memory:":
                return "sqlite"
    return "unknown"


# =============================================================================
# Convenience Functions
# =============================================================================


def from_polars(
    data: "pl.DataFrame | pl.LazyFrame",
    name: str | None = None,
) -> DataSourceProtocol:
    """Create a data source from Polars DataFrame or LazyFrame.

    Args:
        data: Polars DataFrame or LazyFrame.
        name: Optional name.

    Returns:
        PolarsDataSource.
    """
    from truthound.datasources.polars_source import PolarsDataSource
    config = DataSourceConfig(name=name) if name else None
    return PolarsDataSource(data, config)  # type: ignore


def from_pandas(data: "pd.DataFrame", name: str | None = None) -> DataSourceProtocol:
    """Create a data source from Pandas DataFrame.

    Args:
        data: Pandas DataFrame.
        name: Optional name.

    Returns:
        PandasDataSource.
    """
    from truthound.datasources.pandas_source import PandasDataSource
    return PandasDataSource(data)


def from_spark(
    data: "SparkDataFrame",
    name: str | None = None,
    force_sampling: bool = False,
) -> DataSourceProtocol:
    """Create a data source from PySpark DataFrame.

    Args:
        data: PySpark DataFrame.
        name: Optional name.
        force_sampling: Force sampling for large datasets.

    Returns:
        SparkDataSource.
    """
    from truthound.datasources.spark_source import SparkDataSource, SparkDataSourceConfig
    config = SparkDataSourceConfig(name=name) if name else None
    return SparkDataSource(data, config, force_sampling=force_sampling)


def from_file(path: str | Path, name: str | None = None) -> DataSourceProtocol:
    """Create a data source from a file.

    Args:
        path: Path to file.
        name: Optional name.

    Returns:
        FileDataSource.
    """
    from truthound.datasources.polars_source import FileDataSource
    return FileDataSource(path)


def from_dict(data: dict[str, list], name: str | None = None) -> DataSourceProtocol:
    """Create a data source from a dictionary.

    Args:
        data: Dictionary with column names as keys.
        name: Optional name.

    Returns:
        DictDataSource.
    """
    from truthound.datasources.polars_source import DictDataSource
    return DictDataSource(data)
