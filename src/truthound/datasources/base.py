"""Base classes for data sources.

This module provides the abstract base classes that all data source
implementations should extend. It includes common functionality for
configuration, connection management, and size limits.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from truthound.datasources._protocols import (
    ColumnType,
    DataSourceCapability,
    DataSourceProtocol,
)

if TYPE_CHECKING:
    import polars as pl
    from truthound.execution.base import BaseExecutionEngine


# =============================================================================
# Exceptions
# =============================================================================


class DataSourceError(Exception):
    """Base exception for data source errors."""

    pass


class DataSourceConnectionError(DataSourceError):
    """Raised when connection to data source fails."""

    def __init__(self, source_type: str, message: str) -> None:
        self.source_type = source_type
        super().__init__(f"Failed to connect to {source_type}: {message}")


class DataSourceSizeError(DataSourceError):
    """Raised when data source exceeds size limits."""

    def __init__(
        self,
        current_size: int,
        max_size: int,
        unit: str = "rows",
    ) -> None:
        self.current_size = current_size
        self.max_size = max_size
        super().__init__(
            f"Data source size ({current_size:,} {unit}) exceeds maximum "
            f"({max_size:,} {unit}). Use sample() to reduce size."
        )


class DataSourceSchemaError(DataSourceError):
    """Raised when there's a schema-related error."""

    pass


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class DataSourceConfig:
    """Base configuration for all data sources.

    Attributes:
        name: Optional custom name for the data source.
        max_rows: Maximum rows allowed before requiring sampling.
        max_memory_mb: Maximum memory in MB for in-memory operations.
        sample_size: Default sample size when sampling is needed.
        sample_seed: Random seed for reproducible sampling.
        cache_schema: Whether to cache schema information.
        strict_types: Whether to enforce strict type checking.
        metadata: Additional custom metadata.
    """

    name: str | None = None
    max_rows: int = 10_000_000  # 10M rows default limit
    max_memory_mb: int = 4096  # 4GB default memory limit
    sample_size: int = 100_000  # Default sample size
    sample_seed: int | None = 42  # Default seed for reproducibility
    cache_schema: bool = True
    strict_types: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


ConfigT = TypeVar("ConfigT", bound=DataSourceConfig)


# =============================================================================
# Type Mapping Utilities
# =============================================================================


def polars_to_column_type(polars_dtype: Any) -> ColumnType:
    """Convert Polars dtype to unified ColumnType.

    Args:
        polars_dtype: Polars data type.

    Returns:
        Corresponding ColumnType.
    """
    import polars as pl

    dtype_class = type(polars_dtype)
    dtype_name = dtype_class.__name__

    # Integer types
    if dtype_name in ("Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64"):
        return ColumnType.INTEGER

    # Float types
    if dtype_name in ("Float32", "Float64"):
        return ColumnType.FLOAT

    # Decimal
    if dtype_name == "Decimal":
        return ColumnType.DECIMAL

    # String types
    if dtype_name in ("String", "Utf8"):
        return ColumnType.STRING

    # Date/time types
    if dtype_name == "Date":
        return ColumnType.DATE
    if dtype_name == "Datetime":
        return ColumnType.DATETIME
    if dtype_name == "Time":
        return ColumnType.TIME
    if dtype_name == "Duration":
        return ColumnType.DURATION

    # Boolean
    if dtype_name == "Boolean":
        return ColumnType.BOOLEAN

    # Binary
    if dtype_name == "Binary":
        return ColumnType.BINARY

    # Complex types
    if dtype_name == "List":
        return ColumnType.LIST
    if dtype_name == "Struct":
        return ColumnType.STRUCT

    # Null
    if dtype_name == "Null":
        return ColumnType.NULL

    return ColumnType.UNKNOWN


def pandas_dtype_to_column_type(pandas_dtype: Any) -> ColumnType:
    """Convert Pandas dtype to unified ColumnType.

    Args:
        pandas_dtype: Pandas/numpy data type.

    Returns:
        Corresponding ColumnType.
    """
    dtype_str = str(pandas_dtype).lower()

    # Integer types
    if "int" in dtype_str:
        return ColumnType.INTEGER

    # Float types
    if "float" in dtype_str:
        return ColumnType.FLOAT

    # String/object
    if dtype_str in ("object", "string", "str"):
        return ColumnType.STRING

    # Datetime
    if "datetime" in dtype_str:
        return ColumnType.DATETIME

    # Timedelta
    if "timedelta" in dtype_str:
        return ColumnType.DURATION

    # Boolean
    if "bool" in dtype_str:
        return ColumnType.BOOLEAN

    # Category (treat as string)
    if "category" in dtype_str:
        return ColumnType.STRING

    return ColumnType.UNKNOWN


def sql_type_to_column_type(sql_type: str) -> ColumnType:
    """Convert SQL type string to unified ColumnType.

    Args:
        sql_type: SQL data type string.

    Returns:
        Corresponding ColumnType.
    """
    sql_upper = sql_type.upper()

    # Integer types
    if any(t in sql_upper for t in ("INT", "SERIAL", "SMALLINT", "BIGINT", "TINYINT")):
        return ColumnType.INTEGER

    # Float types
    if any(t in sql_upper for t in ("FLOAT", "DOUBLE", "REAL")):
        return ColumnType.FLOAT

    # Decimal types
    if any(t in sql_upper for t in ("DECIMAL", "NUMERIC", "MONEY")):
        return ColumnType.DECIMAL

    # String types (including TEXT for consistency)
    if any(t in sql_upper for t in ("CHAR", "VARCHAR", "NCHAR", "NVARCHAR", "TEXT", "CLOB", "NTEXT")):
        return ColumnType.STRING

    # Date/time types
    if sql_upper == "DATE":
        return ColumnType.DATE
    if any(t in sql_upper for t in ("TIMESTAMP", "DATETIME")):
        return ColumnType.DATETIME
    if sql_upper == "TIME":
        return ColumnType.TIME
    if "INTERVAL" in sql_upper:
        return ColumnType.DURATION

    # Boolean
    if any(t in sql_upper for t in ("BOOL", "BOOLEAN", "BIT")):
        return ColumnType.BOOLEAN

    # Binary
    if any(t in sql_upper for t in ("BINARY", "BLOB", "BYTEA", "VARBINARY")):
        return ColumnType.BINARY

    # JSON
    if "JSON" in sql_upper:
        return ColumnType.JSON

    return ColumnType.UNKNOWN


# =============================================================================
# Abstract Base Data Source
# =============================================================================


class BaseDataSource(ABC, Generic[ConfigT]):
    """Abstract base class for all data sources.

    This class provides common functionality and defines the interface
    that all data source implementations must follow.

    Type Parameters:
        ConfigT: The configuration type for this data source.

    Example:
        >>> class MyDataSource(BaseDataSource[MyConfig]):
        ...     source_type = "my_source"
        ...
        ...     def get_execution_engine(self) -> BaseExecutionEngine:
        ...         return MyExecutionEngine(self)
    """

    source_type: str = "base"

    def __init__(self, config: ConfigT | None = None) -> None:
        """Initialize the data source.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self._config = config or self._default_config()
        self._cached_schema: dict[str, ColumnType] | None = None
        self._cached_row_count: int | None = None
        self._is_connected: bool = False

    @classmethod
    def _default_config(cls) -> ConfigT:
        """Create default configuration.

        Override in subclasses for custom default configurations.
        """
        return DataSourceConfig()  # type: ignore

    @property
    def config(self) -> ConfigT:
        """Get the data source configuration."""
        return self._config

    @property
    def name(self) -> str:
        """Get the data source name."""
        if self._config.name:
            return self._config.name
        return f"{self.source_type}_source"

    # -------------------------------------------------------------------------
    # Abstract Properties
    # -------------------------------------------------------------------------

    @property
    @abstractmethod
    def schema(self) -> dict[str, ColumnType]:
        """Get the schema as column name to type mapping."""
        pass

    @property
    def columns(self) -> list[str]:
        """Get list of column names."""
        return list(self.schema.keys())

    @property
    def row_count(self) -> int | None:
        """Get row count if efficiently available.

        Returns:
            Row count or None if not efficiently computable.
        """
        return self._cached_row_count

    @property
    def capabilities(self) -> set[DataSourceCapability]:
        """Get the capabilities this data source supports.

        Override in subclasses to declare specific capabilities.
        """
        return {DataSourceCapability.SCHEMA_INFERENCE}

    # -------------------------------------------------------------------------
    # Abstract Methods
    # -------------------------------------------------------------------------

    @abstractmethod
    def get_execution_engine(self) -> "BaseExecutionEngine":
        """Get an execution engine for this data source.

        Returns:
            An execution engine appropriate for this data source.
        """
        pass

    @abstractmethod
    def sample(self, n: int = 1000, seed: int | None = None) -> "BaseDataSource":
        """Create a new data source with sampled data.

        Args:
            n: Number of rows to sample.
            seed: Random seed for reproducibility.

        Returns:
            A new data source containing the sampled data.
        """
        pass

    @abstractmethod
    def to_polars_lazyframe(self) -> "pl.LazyFrame":
        """Convert the data source to a Polars LazyFrame.

        This is the fallback for validators that require Polars.
        Implementations should respect size limits.

        Returns:
            Polars LazyFrame.

        Raises:
            DataSourceSizeError: If data exceeds size limits.
        """
        pass

    # -------------------------------------------------------------------------
    # Size Limit Checking
    # -------------------------------------------------------------------------

    def check_size_limits(self) -> None:
        """Check if data source exceeds configured size limits.

        Raises:
            DataSourceSizeError: If size limits are exceeded.
        """
        row_count = self.row_count
        if row_count is not None and row_count > self._config.max_rows:
            raise DataSourceSizeError(
                current_size=row_count,
                max_size=self._config.max_rows,
                unit="rows",
            )

    def needs_sampling(self) -> bool:
        """Check if data source needs sampling due to size.

        Returns:
            True if sampling is recommended.
        """
        row_count = self.row_count
        if row_count is None:
            return False
        return row_count > self._config.max_rows

    def get_safe_sample(self) -> "BaseDataSource":
        """Get a safely-sized sample of the data source.

        If the data source is within size limits, returns self.
        Otherwise, returns a sampled version.

        Returns:
            A data source within size limits.
        """
        if not self.needs_sampling():
            return self
        return self.sample(
            n=self._config.sample_size,
            seed=self._config.sample_seed,
        )

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def validate_connection(self) -> bool:
        """Validate that the data source is accessible.

        Returns:
            True if connection is valid.
        """
        try:
            # Try to access schema as basic validation
            _ = self.schema
            return True
        except Exception:
            return False

    def validate_columns(self, required_columns: list[str]) -> list[str]:
        """Validate that required columns exist.

        Args:
            required_columns: List of column names that must exist.

        Returns:
            List of missing column names (empty if all exist).
        """
        existing = set(self.columns)
        return [col for col in required_columns if col not in existing]

    # -------------------------------------------------------------------------
    # Context Manager
    # -------------------------------------------------------------------------

    def __enter__(self) -> "BaseDataSource":
        """Context manager entry."""
        self._connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self._disconnect()

    def _connect(self) -> None:
        """Establish connection. Override in subclasses that need connections."""
        self._is_connected = True

    def _disconnect(self) -> None:
        """Close connection. Override in subclasses that need cleanup."""
        self._is_connected = False

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def get_column_type(self, column: str) -> ColumnType | None:
        """Get the type of a specific column.

        Args:
            column: Column name.

        Returns:
            Column type or None if column doesn't exist.
        """
        return self.schema.get(column)

    def get_numeric_columns(self) -> list[str]:
        """Get list of numeric columns."""
        numeric_types = {ColumnType.INTEGER, ColumnType.FLOAT, ColumnType.DECIMAL}
        return [col for col, dtype in self.schema.items() if dtype in numeric_types]

    def get_string_columns(self) -> list[str]:
        """Get list of string columns."""
        string_types = {ColumnType.STRING, ColumnType.TEXT}
        return [col for col, dtype in self.schema.items() if dtype in string_types]

    def get_datetime_columns(self) -> list[str]:
        """Get list of datetime columns."""
        dt_types = {ColumnType.DATE, ColumnType.DATETIME, ColumnType.TIME}
        return [col for col, dtype in self.schema.items() if dtype in dt_types]

    def __repr__(self) -> str:
        """Get string representation."""
        row_info = f", rows={self.row_count}" if self.row_count else ""
        return f"{self.__class__.__name__}(name='{self.name}', columns={len(self.columns)}{row_info})"
