"""Async base classes for data sources.

This module provides the abstract base classes for async data source
implementations, including connection pooling and async context management.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar

from truthound.datasources._protocols import (
    ColumnType,
    DataSourceCapability,
)
from truthound.datasources.base import (
    DataSourceConfig,
    DataSourceConnectionError,
    DataSourceError,
    polars_to_column_type,
)

if TYPE_CHECKING:
    import polars as pl

    from truthound.execution.base import BaseExecutionEngine


# =============================================================================
# Exceptions
# =============================================================================


class AsyncDataSourceError(DataSourceError):
    """Base exception for async data source errors."""

    pass


class AsyncConnectionPoolError(AsyncDataSourceError):
    """Error related to connection pool operations."""

    def __init__(self, message: str, pool_size: int | None = None) -> None:
        self.pool_size = pool_size
        super().__init__(message)


class AsyncTimeoutError(AsyncDataSourceError):
    """Timeout during async operation."""

    def __init__(self, operation: str, timeout: float) -> None:
        self.operation = operation
        self.timeout = timeout
        super().__init__(f"Timeout after {timeout}s during {operation}")


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class AsyncDataSourceConfig(DataSourceConfig):
    """Configuration for async data sources.

    Extends base DataSourceConfig with async-specific settings.

    Attributes:
        max_concurrent_requests: Maximum number of concurrent requests.
        connection_timeout: Timeout for establishing connections (seconds).
        query_timeout: Timeout for query operations (seconds).
        pool_size: Number of connections in the pool.
        retry_attempts: Number of retry attempts for failed operations.
        retry_delay: Initial delay between retries (seconds).
        retry_backoff: Backoff multiplier for retry delays.
    """

    max_concurrent_requests: int = 10
    connection_timeout: float = 30.0
    query_timeout: float = 300.0
    pool_size: int = 5
    retry_attempts: int = 3
    retry_delay: float = 1.0
    retry_backoff: float = 2.0


AsyncConfigT = TypeVar("AsyncConfigT", bound=AsyncDataSourceConfig)


# =============================================================================
# Connection Pool
# =============================================================================


class AsyncConnectionPool(Generic[TypeVar("ConnT")]):
    """Async connection pool with semaphore-based concurrency control.

    Manages a pool of connections for async data sources, providing
    automatic connection lifecycle management and concurrency limiting.

    Example:
        >>> async def create_connection():
        ...     return await SomeClient.connect(uri)
        >>>
        >>> pool = AsyncConnectionPool(
        ...     factory=create_connection,
        ...     size=5,
        ...     timeout=30.0,
        ... )
        >>> await pool.initialize()
        >>>
        >>> async with pool.acquire() as conn:
        ...     result = await conn.query("...")
        >>>
        >>> await pool.close()
    """

    def __init__(
        self,
        factory: Callable[[], Any],
        size: int = 5,
        timeout: float = 30.0,
        validator: Callable[[Any], bool] | None = None,
    ) -> None:
        """Initialize the connection pool.

        Args:
            factory: Async callable that creates a new connection.
            size: Maximum number of connections in the pool.
            timeout: Timeout for acquiring a connection (seconds).
            validator: Optional callable to validate connection health.
        """
        self._factory = factory
        self._size = size
        self._timeout = timeout
        self._validator = validator

        self._semaphore = asyncio.Semaphore(size)
        self._connections: asyncio.Queue[Any] = asyncio.Queue(maxsize=size)
        self._lock = asyncio.Lock()
        self._initialized = False
        self._closed = False
        self._active_count = 0

    @property
    def size(self) -> int:
        """Get pool size."""
        return self._size

    @property
    def available(self) -> int:
        """Get number of available connections."""
        return self._connections.qsize()

    @property
    def active(self) -> int:
        """Get number of active (in-use) connections."""
        return self._active_count

    @property
    def is_initialized(self) -> bool:
        """Check if pool is initialized."""
        return self._initialized

    @property
    def is_closed(self) -> bool:
        """Check if pool is closed."""
        return self._closed

    async def initialize(self) -> None:
        """Initialize the connection pool by creating initial connections."""
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            # Pre-create connections
            for _ in range(self._size):
                try:
                    if asyncio.iscoroutinefunction(self._factory):
                        conn = await asyncio.wait_for(
                            self._factory(), timeout=self._timeout
                        )
                    else:
                        conn = self._factory()
                    await self._connections.put(conn)
                except asyncio.TimeoutError:
                    raise AsyncConnectionPoolError(
                        f"Timeout creating connection after {self._timeout}s",
                        pool_size=self._size,
                    )
                except Exception as e:
                    raise AsyncConnectionPoolError(
                        f"Failed to create connection: {e}",
                        pool_size=self._size,
                    )

            self._initialized = True

    @asynccontextmanager
    async def acquire(self):
        """Acquire a connection from the pool.

        Yields:
            A connection from the pool.

        Raises:
            AsyncConnectionPoolError: If pool is closed or timeout occurs.
        """
        if self._closed:
            raise AsyncConnectionPoolError("Connection pool is closed")

        if not self._initialized:
            await self.initialize()

        # Acquire semaphore to limit concurrency
        try:
            await asyncio.wait_for(
                self._semaphore.acquire(), timeout=self._timeout
            )
        except asyncio.TimeoutError:
            raise AsyncTimeoutError("acquire_connection", self._timeout)

        conn = None
        try:
            # Get connection from queue
            try:
                conn = self._connections.get_nowait()
            except asyncio.QueueEmpty:
                # Create new connection if pool exhausted
                if asyncio.iscoroutinefunction(self._factory):
                    conn = await self._factory()
                else:
                    conn = self._factory()

            # Validate connection if validator provided
            if self._validator and not self._validator(conn):
                # Connection is stale, create new one
                if asyncio.iscoroutinefunction(self._factory):
                    conn = await self._factory()
                else:
                    conn = self._factory()

            self._active_count += 1
            yield conn

        finally:
            self._active_count -= 1

            # Return connection to pool
            if conn is not None and not self._closed:
                try:
                    self._connections.put_nowait(conn)
                except asyncio.QueueFull:
                    # Pool is full, close this connection
                    if hasattr(conn, "close"):
                        if asyncio.iscoroutinefunction(conn.close):
                            await conn.close()
                        else:
                            conn.close()

            self._semaphore.release()

    async def close(self) -> None:
        """Close all connections in the pool."""
        if self._closed:
            return

        async with self._lock:
            if self._closed:
                return

            self._closed = True

            # Close all pooled connections
            while not self._connections.empty():
                try:
                    conn = self._connections.get_nowait()
                    if hasattr(conn, "close"):
                        if asyncio.iscoroutinefunction(conn.close):
                            await conn.close()
                        else:
                            conn.close()
                except asyncio.QueueEmpty:
                    break
                except Exception:
                    pass  # Best effort cleanup

    async def __aenter__(self) -> "AsyncConnectionPool":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Async context manager exit."""
        await self.close()


# =============================================================================
# Abstract Async Base Data Source
# =============================================================================


class AsyncBaseDataSource(ABC, Generic[AsyncConfigT]):
    """Abstract base class for async data sources.

    This class provides common functionality for async data source
    implementations, including connection pooling, caching, and
    async context management.

    Type Parameters:
        AsyncConfigT: The configuration type for this data source.

    Example:
        >>> class MyAsyncDataSource(AsyncBaseDataSource[MyConfig]):
        ...     source_type = "my_async_source"
        ...
        ...     async def get_schema_async(self) -> dict[str, ColumnType]:
        ...         async with self._pool.acquire() as conn:
        ...             return await conn.get_schema()
    """

    source_type: str = "async_base"

    def __init__(self, config: AsyncConfigT | None = None) -> None:
        """Initialize the async data source.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self._config = config or self._default_config()
        self._cached_schema: dict[str, ColumnType] | None = None
        self._cached_row_count: int | None = None
        self._pool: AsyncConnectionPool | None = None
        self._is_connected: bool = False
        self._lock = asyncio.Lock()

    @classmethod
    def _default_config(cls) -> AsyncConfigT:
        """Create default configuration.

        Override in subclasses for custom default configurations.
        """
        return AsyncDataSourceConfig()  # type: ignore

    @property
    def config(self) -> AsyncConfigT:
        """Get the data source configuration."""
        return self._config

    @property
    def name(self) -> str:
        """Get the data source name."""
        if self._config.name:
            return self._config.name
        return f"{self.source_type}_source"

    # -------------------------------------------------------------------------
    # Sync Properties (cached values)
    # -------------------------------------------------------------------------

    @property
    def schema(self) -> dict[str, ColumnType]:
        """Get the schema as column name to type mapping (cached).

        Note: This returns cached schema. Call get_schema_async() first
        to populate the cache, or use the async context manager.

        Returns:
            Column name to type mapping.

        Raises:
            RuntimeError: If schema not yet cached.
        """
        if self._cached_schema is None:
            raise RuntimeError(
                "Schema not yet loaded. Call get_schema_async() or use "
                "async context manager to initialize the data source."
            )
        return self._cached_schema

    @property
    def columns(self) -> list[str]:
        """Get list of column names."""
        return list(self.schema.keys())

    @property
    def row_count(self) -> int | None:
        """Get row count if efficiently available (cached).

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
    # Abstract Async Methods
    # -------------------------------------------------------------------------

    @abstractmethod
    async def get_schema_async(self) -> dict[str, ColumnType]:
        """Asynchronously fetch and return the schema.

        This method performs the actual I/O operation to retrieve schema
        information. Implementations should cache the result.

        Returns:
            Column name to type mapping.
        """
        pass

    @abstractmethod
    async def to_polars_lazyframe_async(self) -> "pl.LazyFrame":
        """Asynchronously convert the data source to a Polars LazyFrame.

        Returns:
            Polars LazyFrame.
        """
        pass

    @abstractmethod
    async def sample_async(
        self, n: int = 1000, seed: int | None = None
    ) -> "AsyncBaseDataSource":
        """Asynchronously create a new data source with sampled data.

        Args:
            n: Number of rows to sample.
            seed: Random seed for reproducibility.

        Returns:
            A new async data source containing the sampled data.
        """
        pass

    # -------------------------------------------------------------------------
    # Connection Management
    # -------------------------------------------------------------------------

    @abstractmethod
    async def _create_connection_factory(self) -> Callable:
        """Create a connection factory for the pool.

        Returns:
            Async callable that creates a new connection.
        """
        pass

    async def connect_async(self) -> None:
        """Establish connection to the data source."""
        if self._is_connected:
            return

        async with self._lock:
            if self._is_connected:
                return

            # Create connection pool
            factory = await self._create_connection_factory()
            self._pool = AsyncConnectionPool(
                factory=factory,
                size=self._config.pool_size,
                timeout=self._config.connection_timeout,
            )
            await self._pool.initialize()

            # Pre-fetch schema
            self._cached_schema = await self.get_schema_async()

            self._is_connected = True

    async def disconnect_async(self) -> None:
        """Close connection to the data source."""
        if not self._is_connected:
            return

        async with self._lock:
            if not self._is_connected:
                return

            if self._pool:
                await self._pool.close()
                self._pool = None

            self._is_connected = False

    async def validate_connection_async(self) -> bool:
        """Validate that the data source connection is working.

        Returns:
            True if connection is valid.
        """
        try:
            if not self._is_connected:
                await self.connect_async()
            # Try to get schema as basic validation
            await self.get_schema_async()
            return True
        except Exception:
            return False

    async def get_row_count_async(self) -> int | None:
        """Asynchronously get the row count.

        Override in subclasses to provide efficient implementation.

        Returns:
            Row count or None if not efficiently computable.
        """
        return self._cached_row_count

    # -------------------------------------------------------------------------
    # Sync Fallback (for compatibility)
    # -------------------------------------------------------------------------

    def get_execution_engine(self) -> "BaseExecutionEngine":
        """Get an execution engine for this data source.

        Note: This is a sync method for compatibility. It creates
        a wrapped sync execution engine from the async source.

        Returns:
            An execution engine appropriate for this data source.
        """
        from truthound.datasources.adapters import AsyncToSyncAdapter

        # Wrap self in sync adapter and get its engine
        sync_adapter = AsyncToSyncAdapter(self)
        return sync_adapter.get_execution_engine()

    # -------------------------------------------------------------------------
    # Async Context Manager
    # -------------------------------------------------------------------------

    async def __aenter__(self) -> "AsyncBaseDataSource":
        """Async context manager entry."""
        await self.connect_async()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Async context manager exit."""
        await self.disconnect_async()

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

    def __repr__(self) -> str:
        """Get string representation."""
        connected = "connected" if self._is_connected else "disconnected"
        row_info = f", rows={self.row_count}" if self.row_count else ""
        return (
            f"{self.__class__.__name__}(name='{self.name}', "
            f"status={connected}{row_info})"
        )
