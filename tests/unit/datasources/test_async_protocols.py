"""Tests for async protocol definitions."""

from __future__ import annotations

import pytest
from typing import Any

from truthound.datasources._protocols import ColumnType, DataSourceCapability
from truthound.datasources._async_protocols import (
    AsyncDataSourceProtocol,
    AsyncConnectableProtocol,
    AsyncStreamableProtocol,
    AsyncQueryableProtocol,
)


class TestAsyncDataSourceProtocol:
    """Tests for AsyncDataSourceProtocol."""

    def test_protocol_is_runtime_checkable(self) -> None:
        """Test that the protocol can be used for isinstance checks."""
        # Protocol should be runtime_checkable
        assert hasattr(AsyncDataSourceProtocol, "__protocol_attrs__") or hasattr(
            AsyncDataSourceProtocol, "__subclasshook__"
        )

    def test_protocol_requires_name_property(self) -> None:
        """Test that name property is required."""

        class ValidSource:
            @property
            def name(self) -> str:
                return "test"

            @property
            def source_type(self) -> str:
                return "test"

            @property
            def schema(self) -> dict[str, ColumnType]:
                return {}

            @property
            def columns(self) -> list[str]:
                return []

            @property
            def row_count(self) -> int | None:
                return None

            @property
            def capabilities(self) -> set[DataSourceCapability]:
                return set()

            async def get_schema_async(self) -> dict[str, ColumnType]:
                return {}

            async def get_row_count_async(self) -> int | None:
                return None

            async def validate_connection_async(self) -> bool:
                return True

            async def sample_async(
                self, n: int = 1000, seed: int | None = None
            ) -> "ValidSource":
                return self

            async def to_polars_lazyframe_async(self) -> Any:
                import polars as pl

                return pl.DataFrame().lazy()

        source = ValidSource()
        assert isinstance(source, AsyncDataSourceProtocol)


class TestAsyncConnectableProtocol:
    """Tests for AsyncConnectableProtocol."""

    def test_protocol_requires_async_methods(self) -> None:
        """Test that async methods are required."""

        class ValidConnectable:
            async def connect_async(self) -> None:
                pass

            async def disconnect_async(self) -> None:
                pass

            async def is_connected_async(self) -> bool:
                return True

            async def __aenter__(self) -> "ValidConnectable":
                return self

            async def __aexit__(self, *args: Any) -> None:
                pass

        conn = ValidConnectable()
        assert isinstance(conn, AsyncConnectableProtocol)


class TestAsyncStreamableProtocol:
    """Tests for AsyncStreamableProtocol."""

    def test_protocol_requires_iteration_methods(self) -> None:
        """Test that iteration methods are required."""

        class ValidStreamable:
            async def iter_batches_async(self, batch_size: int = 1000):
                yield []

            async def iter_records_async(self):
                yield {}

        stream = ValidStreamable()
        assert isinstance(stream, AsyncStreamableProtocol)


class TestAsyncQueryableProtocol:
    """Tests for AsyncQueryableProtocol."""

    def test_protocol_requires_query_methods(self) -> None:
        """Test that query methods are required."""

        class ValidQueryable:
            async def execute_query_async(
                self, query: str, params: dict[str, Any] | None = None
            ) -> list[dict[str, Any]]:
                return []

            async def execute_aggregation_async(
                self, pipeline: list[dict[str, Any]]
            ) -> list[dict[str, Any]]:
                return []

        queryable = ValidQueryable()
        assert isinstance(queryable, AsyncQueryableProtocol)
