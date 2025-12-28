"""Tests for async base classes."""

from __future__ import annotations

import asyncio
import pytest
from typing import Any, Callable
from unittest.mock import AsyncMock, MagicMock, patch

from truthound.datasources._protocols import ColumnType, DataSourceCapability
from truthound.datasources.async_base import (
    AsyncBaseDataSource,
    AsyncDataSourceConfig,
    AsyncConnectionPool,
    AsyncDataSourceError,
    AsyncConnectionPoolError,
    AsyncTimeoutError,
)


class TestAsyncDataSourceConfig:
    """Tests for AsyncDataSourceConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = AsyncDataSourceConfig()

        assert config.max_concurrent_requests == 10
        assert config.connection_timeout == 30.0
        assert config.query_timeout == 300.0
        assert config.pool_size == 5
        assert config.retry_attempts == 3
        assert config.retry_delay == 1.0
        assert config.retry_backoff == 2.0

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = AsyncDataSourceConfig(
            name="test_source",
            max_concurrent_requests=20,
            connection_timeout=60.0,
            pool_size=10,
        )

        assert config.name == "test_source"
        assert config.max_concurrent_requests == 20
        assert config.connection_timeout == 60.0
        assert config.pool_size == 10


class TestAsyncConnectionPool:
    """Tests for AsyncConnectionPool."""

    @pytest.fixture
    def mock_factory(self) -> Callable:
        """Create a mock connection factory."""

        async def factory():
            return MagicMock()

        return factory

    @pytest.mark.asyncio
    async def test_initialize_creates_connections(self, mock_factory: Callable) -> None:
        """Test that initialize creates the specified number of connections."""
        pool = AsyncConnectionPool(factory=mock_factory, size=3)

        await pool.initialize()

        assert pool.is_initialized
        assert pool.available == 3
        assert pool.active == 0

    @pytest.mark.asyncio
    async def test_acquire_returns_connection(self, mock_factory: Callable) -> None:
        """Test that acquire returns a connection."""
        pool = AsyncConnectionPool(factory=mock_factory, size=2)
        await pool.initialize()

        async with pool.acquire() as conn:
            assert conn is not None
            assert pool.active == 1

        assert pool.active == 0

    @pytest.mark.asyncio
    async def test_acquire_respects_semaphore(self, mock_factory: Callable) -> None:
        """Test that acquire respects the semaphore limit."""
        pool = AsyncConnectionPool(factory=mock_factory, size=2)
        await pool.initialize()

        acquired = []

        async def acquire_conn(index: int):
            async with pool.acquire() as conn:
                acquired.append(index)
                await asyncio.sleep(0.1)

        # Start 3 acquisitions, but pool size is 2
        tasks = [asyncio.create_task(acquire_conn(i)) for i in range(3)]

        # Wait a bit for first two to acquire
        await asyncio.sleep(0.05)
        assert len(acquired) <= 2

        await asyncio.gather(*tasks)
        assert len(acquired) == 3

    @pytest.mark.asyncio
    async def test_close_cleans_up(self, mock_factory: Callable) -> None:
        """Test that close properly cleans up."""
        pool = AsyncConnectionPool(factory=mock_factory, size=2)
        await pool.initialize()

        await pool.close()

        assert pool.is_closed

    @pytest.mark.asyncio
    async def test_acquire_fails_when_closed(self, mock_factory: Callable) -> None:
        """Test that acquire fails when pool is closed."""
        pool = AsyncConnectionPool(factory=mock_factory, size=2)
        await pool.initialize()
        await pool.close()

        with pytest.raises(AsyncConnectionPoolError):
            async with pool.acquire():
                pass

    @pytest.mark.asyncio
    async def test_context_manager(self, mock_factory: Callable) -> None:
        """Test pool as context manager."""
        async with AsyncConnectionPool(factory=mock_factory, size=2) as pool:
            assert pool.is_initialized
            async with pool.acquire() as conn:
                assert conn is not None

        assert pool.is_closed


class TestAsyncBaseDataSource:
    """Tests for AsyncBaseDataSource."""

    @pytest.fixture
    def concrete_source_class(self):
        """Create a concrete implementation for testing."""

        class ConcreteAsyncDataSource(AsyncBaseDataSource[AsyncDataSourceConfig]):
            source_type = "test_async"

            def __init__(self, config=None):
                super().__init__(config)
                self._test_schema = {"col1": ColumnType.STRING, "col2": ColumnType.INTEGER}

            async def get_schema_async(self):
                self._cached_schema = self._test_schema
                return self._test_schema

            async def to_polars_lazyframe_async(self):
                import polars as pl

                return pl.DataFrame({"col1": ["a", "b"], "col2": [1, 2]}).lazy()

            async def sample_async(self, n=1000, seed=None):
                return self

            async def _create_connection_factory(self):
                async def factory():
                    return MagicMock()

                return factory

        return ConcreteAsyncDataSource

    def test_default_config(self, concrete_source_class) -> None:
        """Test default configuration."""
        source = concrete_source_class()
        assert isinstance(source.config, AsyncDataSourceConfig)

    def test_custom_config(self, concrete_source_class) -> None:
        """Test custom configuration."""
        config = AsyncDataSourceConfig(name="custom_source", pool_size=10)
        source = concrete_source_class(config)
        assert source.name == "custom_source"
        assert source.config.pool_size == 10

    def test_name_property(self, concrete_source_class) -> None:
        """Test name property defaults to source_type."""
        source = concrete_source_class()
        assert source.name == "test_async_source"

    def test_schema_raises_before_load(self, concrete_source_class) -> None:
        """Test schema raises error if not loaded."""
        source = concrete_source_class()
        with pytest.raises(RuntimeError, match="Schema not yet loaded"):
            _ = source.schema

    @pytest.mark.asyncio
    async def test_get_schema_async(self, concrete_source_class) -> None:
        """Test async schema loading."""
        source = concrete_source_class()
        schema = await source.get_schema_async()

        assert schema == {"col1": ColumnType.STRING, "col2": ColumnType.INTEGER}
        assert source.schema == schema  # Now cached

    @pytest.mark.asyncio
    async def test_to_polars_lazyframe_async(self, concrete_source_class) -> None:
        """Test async Polars conversion."""
        source = concrete_source_class()
        lf = await source.to_polars_lazyframe_async()

        df = lf.collect()
        assert len(df) == 2
        assert "col1" in df.columns
        assert "col2" in df.columns

    @pytest.mark.asyncio
    async def test_context_manager(self, concrete_source_class) -> None:
        """Test async context manager."""
        async with concrete_source_class() as source:
            assert source._is_connected
            schema = source.schema  # Should be cached now
            assert "col1" in schema

    def test_repr(self, concrete_source_class) -> None:
        """Test string representation."""
        source = concrete_source_class()
        repr_str = repr(source)
        assert "ConcreteAsyncDataSource" in repr_str
        assert "test_async_source" in repr_str


class TestAsyncExceptions:
    """Tests for async-specific exceptions."""

    def test_async_datasource_error(self) -> None:
        """Test AsyncDataSourceError."""
        error = AsyncDataSourceError("Test error")
        assert str(error) == "Test error"

    def test_async_connection_pool_error(self) -> None:
        """Test AsyncConnectionPoolError with pool size."""
        error = AsyncConnectionPoolError("Pool error", pool_size=5)
        assert error.pool_size == 5
        assert "Pool error" in str(error)

    def test_async_timeout_error(self) -> None:
        """Test AsyncTimeoutError."""
        error = AsyncTimeoutError("connect", 30.0)
        assert error.operation == "connect"
        assert error.timeout == 30.0
        assert "30.0s" in str(error)
        assert "connect" in str(error)
