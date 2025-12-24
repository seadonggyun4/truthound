"""Protocol definitions for data sources.

This module defines the structural typing protocols that all data source
implementations should follow. Using protocols enables duck typing while
maintaining type safety.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import polars as pl
    from truthound.execution._protocols import ExecutionEngineProtocol


class DataSourceCapability(Enum):
    """Capabilities that a data source may support."""

    LAZY_EVALUATION = "lazy_evaluation"  # Supports lazy/deferred execution
    SQL_PUSHDOWN = "sql_pushdown"  # Can push operations to database
    SAMPLING = "sampling"  # Supports efficient sampling
    STREAMING = "streaming"  # Supports streaming/chunked reads
    SCHEMA_INFERENCE = "schema_inference"  # Can infer schema without full scan
    ROW_COUNT = "row_count"  # Can get row count efficiently


class ColumnType(Enum):
    """Unified column type representation across different backends."""

    # Numeric types
    INTEGER = "integer"
    FLOAT = "float"
    DECIMAL = "decimal"

    # String types
    STRING = "string"
    TEXT = "text"

    # Date/Time types
    DATE = "date"
    DATETIME = "datetime"
    TIME = "time"
    DURATION = "duration"

    # Boolean
    BOOLEAN = "boolean"

    # Binary
    BINARY = "binary"

    # Complex types
    LIST = "list"
    STRUCT = "struct"
    JSON = "json"

    # Other
    NULL = "null"
    UNKNOWN = "unknown"


@runtime_checkable
class DataSourceProtocol(Protocol):
    """Protocol defining the interface for all data sources.

    Data sources are responsible for:
    - Providing access to data from various backends
    - Exposing schema and metadata information
    - Creating appropriate execution engines
    - Managing connections and resources
    """

    @property
    def name(self) -> str:
        """Get the data source identifier/name."""
        ...

    @property
    def source_type(self) -> str:
        """Get the type of data source (e.g., 'polars', 'pandas', 'postgresql')."""
        ...

    @property
    def schema(self) -> dict[str, ColumnType]:
        """Get column name to type mapping."""
        ...

    @property
    def columns(self) -> list[str]:
        """Get list of column names."""
        ...

    @property
    def row_count(self) -> int | None:
        """Get row count if efficiently available, None otherwise."""
        ...

    @property
    def capabilities(self) -> set[DataSourceCapability]:
        """Get the set of capabilities this data source supports."""
        ...

    def get_execution_engine(self) -> "ExecutionEngineProtocol":
        """Get an execution engine for this data source."""
        ...

    def sample(self, n: int = 1000, seed: int | None = None) -> "DataSourceProtocol":
        """Create a new data source with sampled data."""
        ...

    def validate_connection(self) -> bool:
        """Validate that the data source connection is working."""
        ...


@runtime_checkable
class ConnectableProtocol(Protocol):
    """Protocol for data sources that require explicit connection management."""

    def connect(self) -> None:
        """Establish connection to the data source."""
        ...

    def disconnect(self) -> None:
        """Close connection to the data source."""
        ...

    def is_connected(self) -> bool:
        """Check if connection is active."""
        ...


@runtime_checkable
class SQLDataSourceProtocol(DataSourceProtocol, Protocol):
    """Protocol for SQL-based data sources with additional SQL capabilities."""

    @property
    def table_name(self) -> str:
        """Get the table/view name."""
        ...

    @property
    def database(self) -> str | None:
        """Get the database name if applicable."""
        ...

    @property
    def schema_name(self) -> str | None:
        """Get the schema name if applicable."""
        ...

    def execute_query(self, query: str) -> list[dict[str, Any]]:
        """Execute a raw SQL query and return results."""
        ...

    def get_table_size_bytes(self) -> int | None:
        """Get the table size in bytes if available."""
        ...
