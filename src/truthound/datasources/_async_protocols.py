"""Async protocol definitions for data sources.

This module defines the structural typing protocols for async-capable data source
implementations. These protocols extend the sync protocols to support async I/O
operations while maintaining backward compatibility.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncIterator, Protocol, runtime_checkable

if TYPE_CHECKING:
    import polars as pl

    from truthound.datasources._protocols import (
        ColumnType,
        DataSourceCapability,
    )


@runtime_checkable
class AsyncDataSourceProtocol(Protocol):
    """Protocol for async-capable data sources.

    Extends the sync DataSourceProtocol with async methods for I/O operations.
    Sync properties return cached values for compatibility with existing code.

    The design philosophy:
    - Properties (name, source_type, schema, etc.) are sync and return cached values
    - Methods that perform I/O (_async suffix) are async
    - This allows the same datasource to be used in both sync and async contexts
    """

    # -------------------------------------------------------------------------
    # Sync Properties (cached values, existing compatibility)
    # -------------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Get the data source identifier/name."""
        ...

    @property
    def source_type(self) -> str:
        """Get the type of data source (e.g., 'mongodb', 'elasticsearch')."""
        ...

    @property
    def schema(self) -> dict[str, "ColumnType"]:
        """Get column name to type mapping (cached)."""
        ...

    @property
    def columns(self) -> list[str]:
        """Get list of column names."""
        ...

    @property
    def row_count(self) -> int | None:
        """Get row count if efficiently available (cached), None otherwise."""
        ...

    @property
    def capabilities(self) -> set["DataSourceCapability"]:
        """Get the set of capabilities this data source supports."""
        ...

    # -------------------------------------------------------------------------
    # Async Methods (I/O operations)
    # -------------------------------------------------------------------------

    async def get_schema_async(self) -> dict[str, "ColumnType"]:
        """Asynchronously fetch and return the schema.

        This method performs the actual I/O operation to retrieve schema
        information. The result should be cached for subsequent sync access.

        Returns:
            Column name to type mapping.
        """
        ...

    async def get_row_count_async(self) -> int | None:
        """Asynchronously get the row count.

        Returns:
            Row count or None if not efficiently computable.
        """
        ...

    async def validate_connection_async(self) -> bool:
        """Asynchronously validate the data source connection.

        Returns:
            True if connection is valid.
        """
        ...

    async def sample_async(
        self, n: int = 1000, seed: int | None = None
    ) -> "AsyncDataSourceProtocol":
        """Asynchronously create a new data source with sampled data.

        Args:
            n: Number of rows to sample.
            seed: Random seed for reproducibility.

        Returns:
            A new async data source containing the sampled data.
        """
        ...

    async def to_polars_lazyframe_async(self) -> "pl.LazyFrame":
        """Asynchronously convert the data source to a Polars LazyFrame.

        Returns:
            Polars LazyFrame.
        """
        ...


@runtime_checkable
class AsyncConnectableProtocol(Protocol):
    """Protocol for async data sources that require explicit connection management.

    This extends ConnectableProtocol with async connection methods for
    non-blocking connection establishment and cleanup.
    """

    async def connect_async(self) -> None:
        """Asynchronously establish connection to the data source."""
        ...

    async def disconnect_async(self) -> None:
        """Asynchronously close connection to the data source."""
        ...

    async def is_connected_async(self) -> bool:
        """Asynchronously check if connection is active.

        This may perform a health check to validate the connection.
        """
        ...

    async def __aenter__(self) -> "AsyncConnectableProtocol":
        """Async context manager entry."""
        ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Async context manager exit."""
        ...


@runtime_checkable
class AsyncStreamableProtocol(Protocol):
    """Protocol for data sources that support async streaming/iteration.

    Use this for sources that can provide data in chunks without
    loading everything into memory at once.
    """

    async def iter_batches_async(
        self, batch_size: int = 1000
    ) -> AsyncIterator[list[dict[str, Any]]]:
        """Asynchronously iterate over data in batches.

        Args:
            batch_size: Number of rows per batch.

        Yields:
            Batches of records as dictionaries.
        """
        ...

    async def iter_records_async(self) -> AsyncIterator[dict[str, Any]]:
        """Asynchronously iterate over individual records.

        Yields:
            Individual records as dictionaries.
        """
        ...


@runtime_checkable
class AsyncQueryableProtocol(Protocol):
    """Protocol for async data sources that support querying.

    This is typically used for database sources that support
    executing queries asynchronously.
    """

    async def execute_query_async(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Asynchronously execute a query and return results.

        Args:
            query: Query string (SQL, MQL, DSL, etc.)
            params: Optional query parameters.

        Returns:
            List of result records.
        """
        ...

    async def execute_aggregation_async(
        self, pipeline: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Asynchronously execute an aggregation pipeline.

        Args:
            pipeline: Aggregation pipeline stages.

        Returns:
            Aggregation results.
        """
        ...
