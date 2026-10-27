"""Native async bridging must preserve ownership, limits and validation semantics."""

import asyncio
import inspect

import polars as pl
import pytest

from truthound.datasources._protocols import ColumnType
from truthound.datasources.adapters import (
    AsyncToSyncAdapter,
    adapt_to_async,
    adapt_to_sync,
    is_async_source,
)
from truthound.datasources.async_base import (
    AsyncBaseDataSource,
    AsyncConnectionPoolError,
    AsyncDataSourceConfig,
)
from truthound.datasources.base import BaseDataSource, DataSourceSizeError

pytestmark = pytest.mark.contract


class LocalAsyncSource(AsyncBaseDataSource):
    def __init__(self, values=None, *, events=None, fail_schema=False):
        super().__init__(AsyncDataSourceConfig(name="approved-source", max_rows=2))
        self.values = [1, None, 3] if values is None else values
        self.events = [] if events is None else events
        self.fail_schema = fail_schema
        self.loop = None

    async def connect_async(self):
        self.loop = asyncio.get_running_loop()
        self.events.append("connect")
        self._is_connected = True

    async def disconnect_async(self):
        assert self.loop is asyncio.get_running_loop()
        self.events.append("disconnect")
        self._is_connected = False

    async def _create_connection_factory(self):
        raise NotImplementedError

    async def get_schema_async(self):
        if self.fail_schema:
            raise ValueError("schema failure")
        self._cached_schema = {"x": ColumnType.INTEGER}
        return self._cached_schema

    async def get_row_count_async(self):
        return len(self.values)

    async def validate_connection_async(self):
        return self._is_connected and self.loop is asyncio.get_running_loop()

    async def to_polars_lazyframe_async(self):
        assert await self.validate_connection_async()
        return pl.DataFrame({"x": pl.Series(self.values, dtype=pl.Int64)}).lazy()

    async def sample_async(self, n=1000, seed=None):
        return LocalAsyncSource(self.values[:n], events=self.events)


def test_native_async_adapter_keeps_limits_sample_connections_and_core_validation(tmp_path):
    import truthound as th
    from truthound.mcp.operations import _context

    source = LocalAsyncSource()
    with AsyncToSyncAdapter(source) as adapter:
        assert isinstance(adapter, BaseDataSource)
        assert adapter.validate_connection()
        assert adapter.needs_sampling()
        with pytest.raises(DataSourceSizeError):
            adapter.check_size_limits()
        with adapter.sample(2) as sample:
            assert sample.validate_connection()
            assert sample.to_polars_lazyframe().collect()["x"].to_list() == [1, None]
        assert adapter.validate_connection()
        assert not adapter._loop.is_closed()
        result = th.check(
            source=adapter, validators=["null"], context=_context(str(tmp_path / "run.ipc"))
        )
        assert result.row_count == 3 and len(result.issues) == 1
        loop = adapter._loop
    assert loop.is_closed()
    adapter.close()
    assert source.events == ["connect", "connect", "disconnect", "disconnect"]


def test_schema_failure_disconnects_and_closes_owned_loop():
    source = LocalAsyncSource(fail_schema=True)
    adapter = AsyncToSyncAdapter(source)
    with pytest.raises(ValueError, match="schema failure"):
        adapter.__enter__()
    assert source.events == ["connect", "disconnect"]
    assert adapter._loop.is_closed()
    adapter.close()


def test_uncached_native_source_adaptation_does_not_read_sync_schema():
    source = LocalAsyncSource()
    assert is_async_source(source)
    assert adapt_to_async(source) is source
    with adapt_to_sync(source) as adapted:
        assert adapted.validate_connection()


@pytest.mark.asyncio
async def test_async_context_is_rejected_without_borrowing_or_leaking_coroutine():
    adapter = AsyncToSyncAdapter(LocalAsyncSource())
    coroutine = adapter._wrapped.get_schema_async()
    loop = asyncio.get_running_loop()
    with pytest.raises(RuntimeError, match="Use the async datasource directly"):
        adapter._run_async(coroutine)
    assert inspect.getcoroutinestate(coroutine) == inspect.CORO_CLOSED
    assert adapter._loop is None
    adapter.close()
    assert not loop.is_closed()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["schema", "partial_pool", "cancelled"])
async def test_native_async_connect_failure_releases_partial_pool_and_can_retry(failure):
    closed = []

    class Connection:
        async def close(self):
            closed.append(self)

    class PooledSource(LocalAsyncSource):
        connect_async = AsyncBaseDataSource.connect_async
        disconnect_async = AsyncBaseDataSource.disconnect_async

        async def _create_connection_factory(self):
            async def factory():
                if self.created == 1 and failure == "partial_pool" and self.fail_schema:
                    raise ValueError("pool failure")
                self.created += 1
                return Connection()

            return factory

        async def get_schema_async(self):
            if failure == "cancelled" and self.fail_schema:
                raise asyncio.CancelledError
            return await super().get_schema_async()

    source = PooledSource(fail_schema=True)
    source.created = 0
    error = {
        "schema": ValueError,
        "partial_pool": AsyncConnectionPoolError,
        "cancelled": asyncio.CancelledError,
    }[failure]
    with pytest.raises(error):
        await source.connect_async()
    assert len(closed) == source.created > 0
    assert source._pool is None and not source._is_connected
    source.fail_schema = False
    await source.connect_async()
    assert source._is_connected
    await source.disconnect_async()
    assert len(closed) == source.created
