"""Sync-Async adapters for data sources.

This module provides adapters to convert between sync and async data sources,
enabling gradual migration and interoperability between the two paradigms.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Callable

from truthound.datasources._protocols import (
    ColumnType,
    DataSourceCapability,
    DataSourceProtocol,
)
from truthound.datasources._async_protocols import (
    AsyncDataSourceProtocol,
)
from truthound.datasources.base import (
    DataSourceConfig,
    polars_to_column_type,
)

if TYPE_CHECKING:
    import polars as pl

    from truthound.execution.base import BaseExecutionEngine


# =============================================================================
# Sync to Async Adapter
# =============================================================================


class SyncToAsyncAdapter:
    """Wrap a sync data source for async context.

    This adapter enables sync data sources to be used in async code
    by running sync operations in a thread pool executor.

    Example:
        >>> sync_source = PolarsDataSource(df)
        >>> async_source = SyncToAsyncAdapter(sync_source)
        >>>
        >>> async with async_source:
        ...     schema = await async_source.get_schema_async()
        ...     lf = await async_source.to_polars_lazyframe_async()
    """

    def __init__(
        self,
        wrapped: DataSourceProtocol,
        executor: ThreadPoolExecutor | None = None,
        max_workers: int = 4,
    ) -> None:
        """Initialize the adapter.

        Args:
            wrapped: The sync data source to wrap.
            executor: Optional executor for running sync operations.
            max_workers: Number of workers if creating new executor.
        """
        self._wrapped = wrapped
        self._own_executor = executor is None
        self._executor = executor or ThreadPoolExecutor(max_workers=max_workers)
        self._cached_schema: dict[str, ColumnType] | None = None

    # -------------------------------------------------------------------------
    # Sync Properties (delegate to wrapped)
    # -------------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Get the data source name."""
        return self._wrapped.name

    @property
    def source_type(self) -> str:
        """Get the source type."""
        return f"async_{self._wrapped.source_type}"

    @property
    def schema(self) -> dict[str, ColumnType]:
        """Get cached schema (sync)."""
        if self._cached_schema is None:
            self._cached_schema = self._wrapped.schema
        return self._cached_schema

    @property
    def columns(self) -> list[str]:
        """Get column names."""
        return self._wrapped.columns

    @property
    def row_count(self) -> int | None:
        """Get row count if available."""
        return self._wrapped.row_count

    @property
    def capabilities(self) -> set[DataSourceCapability]:
        """Get capabilities."""
        return self._wrapped.capabilities

    # -------------------------------------------------------------------------
    # Async Methods
    # -------------------------------------------------------------------------

    async def _run_in_executor(self, func: Callable, *args: Any) -> Any:
        """Run a sync function in the thread pool executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, func, *args)

    async def get_schema_async(self) -> dict[str, ColumnType]:
        """Asynchronously get schema."""
        if self._cached_schema is None:
            self._cached_schema = await self._run_in_executor(
                lambda: self._wrapped.schema
            )
        return self._cached_schema

    async def get_row_count_async(self) -> int | None:
        """Asynchronously get row count."""
        return await self._run_in_executor(lambda: self._wrapped.row_count)

    async def validate_connection_async(self) -> bool:
        """Asynchronously validate connection."""
        return await self._run_in_executor(self._wrapped.validate_connection)

    async def sample_async(
        self, n: int = 1000, seed: int | None = None
    ) -> "SyncToAsyncAdapter":
        """Asynchronously create a sampled data source."""
        sampled = await self._run_in_executor(
            lambda: self._wrapped.sample(n, seed)
        )
        return SyncToAsyncAdapter(sampled, self._executor)

    async def to_polars_lazyframe_async(self) -> "pl.LazyFrame":
        """Asynchronously convert to Polars LazyFrame."""
        # to_polars_lazyframe is defined in base class but may not be in protocol
        if hasattr(self._wrapped, "to_polars_lazyframe"):
            return await self._run_in_executor(self._wrapped.to_polars_lazyframe)
        else:
            # Fallback: get execution engine and use it
            engine = await self._run_in_executor(self._wrapped.get_execution_engine)
            if hasattr(engine, "_lf"):
                return engine._lf
            raise NotImplementedError(
                f"{type(self._wrapped).__name__} does not support to_polars_lazyframe"
            )

    # -------------------------------------------------------------------------
    # Context Manager
    # -------------------------------------------------------------------------

    async def __aenter__(self) -> "SyncToAsyncAdapter":
        """Async context manager entry."""
        # Pre-cache schema
        await self.get_schema_async()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Async context manager exit."""
        if self._own_executor:
            self._executor.shutdown(wait=False)

    def close(self) -> None:
        """Close the adapter and cleanup resources."""
        if self._own_executor:
            self._executor.shutdown(wait=True)

    def __repr__(self) -> str:
        """Get string representation."""
        return f"SyncToAsyncAdapter(wrapped={self._wrapped!r})"


# =============================================================================
# Async to Sync Adapter
# =============================================================================


class AsyncToSyncAdapter:
    """Wrap an async data source for sync context.

    This adapter enables async data sources to be used in sync code
    by running async operations in an event loop.

    Example:
        >>> async_source = MongoDBDataSource(config)
        >>> sync_source = AsyncToSyncAdapter(async_source)
        >>>
        >>> with sync_source:
        ...     schema = sync_source.schema
        ...     engine = sync_source.get_execution_engine()
    """

    def __init__(self, wrapped: AsyncDataSourceProtocol) -> None:
        """Initialize the adapter.

        Args:
            wrapped: The async data source to wrap.
        """
        self._wrapped = wrapped
        self._loop: asyncio.AbstractEventLoop | None = None
        self._cached_schema: dict[str, ColumnType] | None = None

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create an event loop for running async operations."""
        if self._loop is None or self._loop.is_closed():
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
        return self._loop

    def _run_async(self, coro: Any) -> Any:
        """Run an async coroutine synchronously."""
        loop = self._get_loop()
        try:
            # If we're in an async context, we can't use run_until_complete
            asyncio.get_running_loop()
            # We're in an async context, use nest_asyncio or raise
            raise RuntimeError(
                "Cannot use AsyncToSyncAdapter from within an async context. "
                "Use the async datasource directly."
            )
        except RuntimeError:
            # No running loop, safe to use run_until_complete
            return loop.run_until_complete(coro)

    # -------------------------------------------------------------------------
    # Sync Properties
    # -------------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Get the data source name."""
        return self._wrapped.name

    @property
    def source_type(self) -> str:
        """Get the source type."""
        return self._wrapped.source_type.replace("async_", "")

    @property
    def schema(self) -> dict[str, ColumnType]:
        """Get schema (runs async operation synchronously)."""
        if self._cached_schema is None:
            self._cached_schema = self._run_async(self._wrapped.get_schema_async())
        return self._cached_schema

    @property
    def columns(self) -> list[str]:
        """Get column names."""
        return list(self.schema.keys())

    @property
    def row_count(self) -> int | None:
        """Get row count."""
        return self._wrapped.row_count

    @property
    def capabilities(self) -> set[DataSourceCapability]:
        """Get capabilities."""
        return self._wrapped.capabilities

    # -------------------------------------------------------------------------
    # Sync Methods
    # -------------------------------------------------------------------------

    def get_execution_engine(self) -> "BaseExecutionEngine":
        """Get an execution engine for this data source."""
        from truthound.execution.polars_engine import PolarsExecutionEngine

        lf = self.to_polars_lazyframe()
        return PolarsExecutionEngine(lf)

    def sample(self, n: int = 1000, seed: int | None = None) -> "AsyncToSyncAdapter":
        """Create a sampled data source."""
        sampled = self._run_async(self._wrapped.sample_async(n, seed))
        return AsyncToSyncAdapter(sampled)

    def validate_connection(self) -> bool:
        """Validate connection."""
        return self._run_async(self._wrapped.validate_connection_async())

    def to_polars_lazyframe(self) -> "pl.LazyFrame":
        """Convert to Polars LazyFrame."""
        return self._run_async(self._wrapped.to_polars_lazyframe_async())

    # -------------------------------------------------------------------------
    # Context Manager
    # -------------------------------------------------------------------------

    def __enter__(self) -> "AsyncToSyncAdapter":
        """Context manager entry."""
        if hasattr(self._wrapped, "connect_async"):
            self._run_async(self._wrapped.connect_async())
        # Pre-cache schema
        _ = self.schema
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Context manager exit."""
        if hasattr(self._wrapped, "disconnect_async"):
            self._run_async(self._wrapped.disconnect_async())

    def close(self) -> None:
        """Close the adapter and cleanup resources."""
        if hasattr(self._wrapped, "disconnect_async"):
            self._run_async(self._wrapped.disconnect_async())
        if self._loop and not self._loop.is_running():
            self._loop.close()

    def __repr__(self) -> str:
        """Get string representation."""
        return f"AsyncToSyncAdapter(wrapped={self._wrapped!r})"


# =============================================================================
# Utility Functions
# =============================================================================


def adapt_to_async(
    source: DataSourceProtocol | AsyncDataSourceProtocol,
    executor: ThreadPoolExecutor | None = None,
) -> AsyncDataSourceProtocol | SyncToAsyncAdapter:
    """Convert any data source to async-compatible.

    If the source is already async-capable, returns it unchanged.
    Otherwise, wraps it in a SyncToAsyncAdapter.

    Args:
        source: Any data source.
        executor: Optional executor for sync operations.

    Returns:
        An async-compatible data source.

    Example:
        >>> async_source = adapt_to_async(polars_source)
        >>> async with async_source:
        ...     schema = await async_source.get_schema_async()
    """
    # Check if already async-capable
    if isinstance(source, AsyncDataSourceProtocol):
        return source

    # Check if it's already an adapter
    if isinstance(source, SyncToAsyncAdapter):
        return source

    # Wrap sync source
    return SyncToAsyncAdapter(source, executor)


def adapt_to_sync(
    source: DataSourceProtocol | AsyncDataSourceProtocol,
) -> DataSourceProtocol | AsyncToSyncAdapter:
    """Convert any data source to sync-compatible.

    If the source is already sync-capable, returns it unchanged.
    Otherwise, wraps it in an AsyncToSyncAdapter.

    Args:
        source: Any data source.

    Returns:
        A sync-compatible data source.

    Example:
        >>> sync_source = adapt_to_sync(mongo_source)
        >>> with sync_source:
        ...     schema = sync_source.schema
    """
    # Check if already sync-capable (has all required sync properties)
    if isinstance(source, DataSourceProtocol):
        # Verify it's not just an async source implementing the protocol partially
        if not isinstance(source, AsyncDataSourceProtocol):
            return source

    # Check if it's already an adapter
    if isinstance(source, AsyncToSyncAdapter):
        return source

    # Wrap async source
    if isinstance(source, AsyncDataSourceProtocol):
        return AsyncToSyncAdapter(source)

    # Default: wrap in async adapter then sync adapter
    return source  # type: ignore


def is_async_source(source: Any) -> bool:
    """Check if a data source is async-capable.

    Args:
        source: Any data source.

    Returns:
        True if the source supports async operations.
    """
    return isinstance(source, (AsyncDataSourceProtocol, SyncToAsyncAdapter))


def is_sync_source(source: Any) -> bool:
    """Check if a data source is sync-capable.

    Args:
        source: Any data source.

    Returns:
        True if the source supports sync operations.
    """
    return isinstance(source, (DataSourceProtocol, AsyncToSyncAdapter))
