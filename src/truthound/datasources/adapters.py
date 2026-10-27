"""Sync-Async adapters for data sources.

This module provides adapters to convert between sync and async data sources,
enabling gradual migration and interoperability between the two paradigms.
"""

from __future__ import annotations

import asyncio
import inspect
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from typing import TYPE_CHECKING, Any

from truthound.datasources._async_protocols import (
    AsyncDataSourceProtocol,
)
from truthound.datasources._protocols import (
    ColumnType,
    DataSourceCapability,
    DataSourceProtocol,
)
from truthound.datasources.async_base import AsyncBaseDataSource
from truthound.datasources.base import (
    BaseDataSource,
    DataSourceConfig,
)

if TYPE_CHECKING:
    from collections.abc import Callable

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
            self._cached_schema = await self._run_in_executor(lambda: self._wrapped.schema)
        return self._cached_schema

    async def get_row_count_async(self) -> int | None:
        """Asynchronously get row count."""
        return await self._run_in_executor(lambda: self._wrapped.row_count)

    async def validate_connection_async(self) -> bool:
        """Asynchronously validate connection."""
        return await self._run_in_executor(self._wrapped.validate_connection)

    async def sample_async(self, n: int = 1000, seed: int | None = None) -> SyncToAsyncAdapter:
        """Asynchronously create a sampled data source."""
        sampled = await self._run_in_executor(lambda: self._wrapped.sample(n, seed))
        return SyncToAsyncAdapter(sampled, self._executor)

    async def to_polars_lazyframe_async(self) -> pl.LazyFrame:
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

    async def __aenter__(self) -> SyncToAsyncAdapter:
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


class AsyncToSyncAdapter(BaseDataSource[DataSourceConfig]):
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
        config = getattr(wrapped, "config", None)
        super().__init__(config if isinstance(config, DataSourceConfig) else DataSourceConfig())
        self._wrapped = wrapped
        self._loop: asyncio.AbstractEventLoop | None = None
        self._owns_loop = True
        self._entered = False
        self._closed = False

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create an event loop for running async operations."""
        if self._closed:
            raise RuntimeError("AsyncToSyncAdapter is closed")
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
        return self._loop

    def _run_async(self, coro: Any) -> Any:
        """Run an async coroutine synchronously."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            if inspect.iscoroutine(coro):
                coro.close()
            raise RuntimeError(
                "Cannot use AsyncToSyncAdapter from within an async context. "
                "Use the async datasource directly."
            )
        try:
            loop = self._get_loop()
            return loop.run_until_complete(coro)
        except BaseException:
            if inspect.iscoroutine(coro):
                coro.close()
            raise

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
        return self._wrapped.source_type.removeprefix("async_")

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
        count = self._wrapped.row_count
        return count if count is not None else self._run_async(self._wrapped.get_row_count_async())

    @property
    def capabilities(self) -> set[DataSourceCapability]:
        """Get capabilities."""
        return self._wrapped.capabilities

    # -------------------------------------------------------------------------
    # Sync Methods
    # -------------------------------------------------------------------------

    def get_execution_engine(self) -> BaseExecutionEngine:
        """Get an execution engine for this data source."""
        from truthound.execution.polars_engine import PolarsExecutionEngine

        lf = self.to_polars_lazyframe()
        return PolarsExecutionEngine(lf)

    def sample(self, n: int = 1000, seed: int | None = None) -> AsyncToSyncAdapter:
        """Create a sampled data source."""
        sampled = self._run_async(self._wrapped.sample_async(n, seed))
        if sampled is self._wrapped:
            return self
        adapter = AsyncToSyncAdapter(sampled)
        adapter._loop = self._get_loop()
        adapter._owns_loop = False
        return adapter.__enter__()

    def validate_connection(self) -> bool:
        """Validate connection."""
        return self._run_async(self._wrapped.validate_connection_async())

    def to_polars_lazyframe(self) -> pl.LazyFrame:
        """Convert to Polars LazyFrame."""
        return self._run_async(self._wrapped.to_polars_lazyframe_async())

    # -------------------------------------------------------------------------
    # Context Manager
    # -------------------------------------------------------------------------

    def __enter__(self) -> AsyncToSyncAdapter:
        """Context manager entry."""
        if self._closed:
            raise RuntimeError("AsyncToSyncAdapter is closed")
        if self._entered:
            return self
        self._entered = True
        try:
            connect = getattr(self._wrapped, "connect_async", None)
            if inspect.iscoroutinefunction(connect):
                self._run_async(connect())
            _ = self.schema
        except BaseException:
            with suppress(Exception):
                self.close()
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Context manager exit."""
        self.close()

    def close(self) -> None:
        """Close the adapter and cleanup resources."""
        if self._closed:
            return
        try:
            disconnect = getattr(self._wrapped, "disconnect_async", None)
            if self._entered and inspect.iscoroutinefunction(disconnect):
                self._run_async(disconnect())
        finally:
            self._entered = False
            self._closed = True
            if self._owns_loop and self._loop and not self._loop.is_running():
                self._loop.run_until_complete(self._loop.shutdown_asyncgens())
                self._loop.run_until_complete(self._loop.shutdown_default_executor())
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
    if isinstance(source, (AsyncBaseDataSource, AsyncDataSourceProtocol)):
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
    # Native async schema properties deliberately raise before connection; do
    # not evaluate structural protocol properties to identify a native source.
    if isinstance(source, AsyncBaseDataSource):
        return AsyncToSyncAdapter(source)
    if isinstance(source, BaseDataSource):
        return source
    # Check if already sync-capable (has all required sync properties)
    if isinstance(source, DataSourceProtocol) and not isinstance(source, AsyncDataSourceProtocol):
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
    return isinstance(source, (AsyncBaseDataSource, SyncToAsyncAdapter, AsyncDataSourceProtocol))


def is_sync_source(source: Any) -> bool:
    """Check if a data source is sync-capable.

    Args:
        source: Any data source.

    Returns:
        True if the source supports sync operations.
    """
    if isinstance(source, AsyncBaseDataSource):
        return False
    return isinstance(source, (BaseDataSource, DataSourceProtocol, AsyncToSyncAdapter))
