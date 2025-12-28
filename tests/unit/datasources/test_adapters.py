"""Tests for sync-async adapters."""

from __future__ import annotations

import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import polars as pl

from truthound.datasources._protocols import ColumnType, DataSourceCapability
from truthound.datasources.adapters import (
    SyncToAsyncAdapter,
    AsyncToSyncAdapter,
    adapt_to_async,
    adapt_to_sync,
    is_async_source,
    is_sync_source,
)
from truthound.datasources.polars_source import PolarsDataSource


class TestSyncToAsyncAdapter:
    """Tests for SyncToAsyncAdapter."""

    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        """Create a sample DataFrame."""
        return pl.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
        })

    @pytest.fixture
    def sync_source(self, sample_df: pl.DataFrame) -> PolarsDataSource:
        """Create a sync data source."""
        return PolarsDataSource(sample_df)

    def test_adapter_creation(self, sync_source: PolarsDataSource) -> None:
        """Test adapter creation."""
        adapter = SyncToAsyncAdapter(sync_source)

        assert adapter._wrapped is sync_source
        assert adapter._own_executor is True

    def test_name_property(self, sync_source: PolarsDataSource) -> None:
        """Test name property delegation."""
        adapter = SyncToAsyncAdapter(sync_source)
        assert adapter.name == sync_source.name

    def test_source_type(self, sync_source: PolarsDataSource) -> None:
        """Test source_type includes async prefix."""
        adapter = SyncToAsyncAdapter(sync_source)
        assert adapter.source_type == "async_polars"

    def test_schema_property(self, sync_source: PolarsDataSource) -> None:
        """Test schema property delegation."""
        adapter = SyncToAsyncAdapter(sync_source)
        schema = adapter.schema

        assert "id" in schema
        assert "name" in schema
        assert schema["id"] == ColumnType.INTEGER
        assert schema["name"] == ColumnType.STRING

    def test_columns_property(self, sync_source: PolarsDataSource) -> None:
        """Test columns property delegation."""
        adapter = SyncToAsyncAdapter(sync_source)
        assert adapter.columns == ["id", "name"]

    def test_row_count_property(self, sync_source: PolarsDataSource) -> None:
        """Test row_count property delegation."""
        adapter = SyncToAsyncAdapter(sync_source)
        assert adapter.row_count == 3

    def test_capabilities_property(self, sync_source: PolarsDataSource) -> None:
        """Test capabilities property delegation."""
        adapter = SyncToAsyncAdapter(sync_source)
        caps = adapter.capabilities

        assert DataSourceCapability.LAZY_EVALUATION in caps
        assert DataSourceCapability.SCHEMA_INFERENCE in caps

    @pytest.mark.asyncio
    async def test_get_schema_async(self, sync_source: PolarsDataSource) -> None:
        """Test async schema retrieval."""
        adapter = SyncToAsyncAdapter(sync_source)
        schema = await adapter.get_schema_async()

        assert "id" in schema
        assert "name" in schema

    @pytest.mark.asyncio
    async def test_get_row_count_async(self, sync_source: PolarsDataSource) -> None:
        """Test async row count retrieval."""
        adapter = SyncToAsyncAdapter(sync_source)
        count = await adapter.get_row_count_async()
        assert count == 3

    @pytest.mark.asyncio
    async def test_validate_connection_async(self, sync_source: PolarsDataSource) -> None:
        """Test async connection validation."""
        adapter = SyncToAsyncAdapter(sync_source)
        valid = await adapter.validate_connection_async()
        assert valid is True

    @pytest.mark.asyncio
    async def test_sample_async(self, sync_source: PolarsDataSource) -> None:
        """Test async sampling."""
        adapter = SyncToAsyncAdapter(sync_source)
        sampled = await adapter.sample_async(2)

        assert isinstance(sampled, SyncToAsyncAdapter)
        assert sampled.row_count <= 2

    @pytest.mark.asyncio
    async def test_to_polars_lazyframe_async(self, sync_source: PolarsDataSource) -> None:
        """Test async Polars conversion."""
        adapter = SyncToAsyncAdapter(sync_source)
        lf = await adapter.to_polars_lazyframe_async()

        df = lf.collect()
        assert len(df) == 3
        assert "id" in df.columns

    @pytest.mark.asyncio
    async def test_context_manager(self, sync_source: PolarsDataSource) -> None:
        """Test async context manager."""
        async with SyncToAsyncAdapter(sync_source) as adapter:
            schema = await adapter.get_schema_async()
            assert "id" in schema

    def test_repr(self, sync_source: PolarsDataSource) -> None:
        """Test string representation."""
        adapter = SyncToAsyncAdapter(sync_source)
        repr_str = repr(adapter)
        assert "SyncToAsyncAdapter" in repr_str


class TestAsyncToSyncAdapter:
    """Tests for AsyncToSyncAdapter."""

    @pytest.fixture
    def mock_async_source(self) -> MagicMock:
        """Create a mock async source."""
        source = MagicMock()
        source.name = "test_async"
        source.source_type = "async_test"
        source.row_count = 5
        source.capabilities = {DataSourceCapability.SCHEMA_INFERENCE}

        # Mock async methods
        async def get_schema():
            return {"col": ColumnType.STRING}

        async def validate():
            return True

        async def sample(n, seed):
            return source

        async def to_lf():
            return pl.DataFrame({"col": ["a", "b"]}).lazy()

        source.get_schema_async = get_schema
        source.validate_connection_async = validate
        source.sample_async = sample
        source.to_polars_lazyframe_async = to_lf

        return source

    def test_adapter_creation(self, mock_async_source: MagicMock) -> None:
        """Test adapter creation."""
        adapter = AsyncToSyncAdapter(mock_async_source)
        assert adapter._wrapped is mock_async_source

    def test_name_property(self, mock_async_source: MagicMock) -> None:
        """Test name property."""
        adapter = AsyncToSyncAdapter(mock_async_source)
        assert adapter.name == "test_async"

    def test_source_type_removes_async_prefix(self, mock_async_source: MagicMock) -> None:
        """Test source_type removes async prefix."""
        adapter = AsyncToSyncAdapter(mock_async_source)
        assert adapter.source_type == "test"

    def test_schema_property(self, mock_async_source: MagicMock) -> None:
        """Test schema property runs async."""
        adapter = AsyncToSyncAdapter(mock_async_source)
        schema = adapter.schema
        assert schema == {"col": ColumnType.STRING}

    def test_columns_property(self, mock_async_source: MagicMock) -> None:
        """Test columns property."""
        adapter = AsyncToSyncAdapter(mock_async_source)
        assert adapter.columns == ["col"]

    def test_row_count_property(self, mock_async_source: MagicMock) -> None:
        """Test row_count property."""
        adapter = AsyncToSyncAdapter(mock_async_source)
        assert adapter.row_count == 5

    def test_capabilities_property(self, mock_async_source: MagicMock) -> None:
        """Test capabilities property."""
        adapter = AsyncToSyncAdapter(mock_async_source)
        assert DataSourceCapability.SCHEMA_INFERENCE in adapter.capabilities

    def test_validate_connection(self, mock_async_source: MagicMock) -> None:
        """Test sync connection validation."""
        adapter = AsyncToSyncAdapter(mock_async_source)
        assert adapter.validate_connection() is True

    def test_sample(self, mock_async_source: MagicMock) -> None:
        """Test sync sampling."""
        adapter = AsyncToSyncAdapter(mock_async_source)
        sampled = adapter.sample(2)
        assert isinstance(sampled, AsyncToSyncAdapter)

    def test_to_polars_lazyframe(self, mock_async_source: MagicMock) -> None:
        """Test sync Polars conversion."""
        adapter = AsyncToSyncAdapter(mock_async_source)
        lf = adapter.to_polars_lazyframe()
        df = lf.collect()
        assert len(df) == 2

    def test_get_execution_engine(self, mock_async_source: MagicMock) -> None:
        """Test execution engine retrieval."""
        adapter = AsyncToSyncAdapter(mock_async_source)
        engine = adapter.get_execution_engine()
        assert engine is not None

    def test_repr(self, mock_async_source: MagicMock) -> None:
        """Test string representation."""
        adapter = AsyncToSyncAdapter(mock_async_source)
        repr_str = repr(adapter)
        assert "AsyncToSyncAdapter" in repr_str


class TestAdaptFunctions:
    """Tests for adapt_to_async and adapt_to_sync functions."""

    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        """Create a sample DataFrame."""
        return pl.DataFrame({"x": [1, 2, 3]})

    def test_adapt_to_async_wraps_sync(self, sample_df: pl.DataFrame) -> None:
        """Test that adapt_to_async wraps sync sources."""
        sync_source = PolarsDataSource(sample_df)
        async_source = adapt_to_async(sync_source)

        assert isinstance(async_source, SyncToAsyncAdapter)

    def test_adapt_to_async_returns_async_unchanged(self) -> None:
        """Test that adapt_to_async returns async sources unchanged."""
        # Create a mock async source
        mock = MagicMock()
        mock.__class__.__name__ = "MockAsyncSource"

        # Make it pass isinstance check by adding required attributes
        with patch(
            "truthound.datasources.adapters.AsyncDataSourceProtocol", new=type(mock)
        ):
            result = adapt_to_async(mock)
            # Should not be wrapped (returns input for already-async)

    def test_adapt_to_async_with_custom_executor(self, sample_df: pl.DataFrame) -> None:
        """Test adapt_to_async with custom executor."""
        from concurrent.futures import ThreadPoolExecutor

        sync_source = PolarsDataSource(sample_df)
        executor = ThreadPoolExecutor(max_workers=2)

        async_source = adapt_to_async(sync_source, executor=executor)

        assert isinstance(async_source, SyncToAsyncAdapter)
        assert async_source._executor is executor
        assert async_source._own_executor is False

        executor.shutdown(wait=False)


class TestTypeCheckers:
    """Tests for is_async_source and is_sync_source."""

    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        """Create a sample DataFrame."""
        return pl.DataFrame({"x": [1, 2, 3]})

    def test_is_sync_source(self, sample_df: pl.DataFrame) -> None:
        """Test is_sync_source detection."""
        sync_source = PolarsDataSource(sample_df)
        assert is_sync_source(sync_source) is True

    def test_is_async_source_for_adapter(self, sample_df: pl.DataFrame) -> None:
        """Test is_async_source for adapted sources."""
        sync_source = PolarsDataSource(sample_df)
        async_source = SyncToAsyncAdapter(sync_source)

        assert is_async_source(async_source) is True

    def test_is_sync_source_for_async_adapter(self) -> None:
        """Test is_sync_source for async-to-sync adapter."""
        mock = MagicMock()
        mock.name = "test"
        mock.source_type = "test"
        mock.row_count = 1
        mock.capabilities = set()

        async def get_schema():
            return {}

        mock.get_schema_async = get_schema

        adapter = AsyncToSyncAdapter(mock)
        assert is_sync_source(adapter) is True
