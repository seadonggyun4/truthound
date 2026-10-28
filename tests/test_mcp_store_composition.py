"""Native cache/batch compositions retain records through approved MCP owners."""

import asyncio
from datetime import datetime

import pytest

from tests import test_mcp_plugin_bindings as plugins
from tests.test_mcp_brokers import approved
from tests.test_mcp_execution_stores import prepare, command

runtime = plugins.runtime
pytestmark = pytest.mark.contract


@pytest.mark.parametrize('mode', ['nested-cache-batch', 'nested-batch-cache'])
def test_native_store_composition_persists_and_reopens(runtime, monkeypatch, tmp_path, mode):
    from truthound.stores.backends.filesystem import FileSystemStore
    from truthound.stores.batching.store import BatchedStore
    from truthound.stores.caching.store import CachedStore
    from truthound.stores.results import ValidationResult, ResultStatus

    native = FileSystemStore(base_path=str(tmp_path / 'direct'))
    cache = CachedStore(native if mode == 'nested-batch-cache' else BatchedStore(native))
    composed = BatchedStore(cache) if mode == 'nested-batch-cache' else cache
    record = ValidationResult('private-native-run', datetime(2026, 1, 1), 'private-native-asset', ResultStatus.FAILURE)
    try:
        if isinstance(composed, BatchedStore):
            async def native_batch():
                await composed.start()
                try:
                    assert await composed.add(record)
                    assert not composed.exists(record.id)
                    await composed.flush()
                    return composed.get(record.id)
                finally:
                    await composed.stop()
            expected = asyncio.run(native_batch())
        else:
            composed.save(record)
            expected = composed.get(record.id)
        assert expected.to_dict() == record.to_dict()
    finally:
        if isinstance(composed, BatchedStore): asyncio.run(composed.stop())
        cache.close()
        if mode == 'nested-cache-batch': asyncio.run(cache.store.stop())
        native.close()

    group = 'store_services'
    identity = prepare(runtime, monkeypatch, group, mode)
    if mode == 'nested-batch-cache':
        command(runtime, group, identity, 'batch_start')
        assert command(runtime, group, identity, 'batch_add', record_id='seed')['added']
        assert not command(runtime, group, identity, 'exists', record_id='seed')['exists']
        assert command(runtime, group, identity, 'batch_flush')['batches'] == 1
    else:
        assert command(runtime, group, identity, 'save', record_id='seed')['record_id'] == 'seed'
    actual = command(runtime, group, identity, 'get', record_id='seed')['record']
    assert actual['status'] == expected.status.value
    assert actual['statistics'] == expected.statistics.to_dict()
    assert actual['run_time'] == expected.run_time.isoformat()
    assert actual['data_asset_id'] == 'local'
    assert 'synthetic-private-record' not in str(actual)
    assert command(runtime, group, identity, 'shutdown')['shutdown'] is True
    reopened = approved(runtime, group + '_open', binding_id='local', idempotency_key='composition-reopen')['data']['resource_id']
    assert command(runtime, group, reopened, 'get', key='composition-read', record_id='seed')['record'] == actual
    assert command(runtime, group, reopened, 'shutdown', key='composition-close')['shutdown'] is True
    assert runtime[2].read_text().splitlines().count('store-close') == 2


@pytest.mark.fault
def test_cyclic_store_composition_is_rejected_without_native_read():
    from truthound.mcp.store_services import StoreRuntime
    from truthound.mcp.store_worker import StoreOwner
    from truthound.mcp.policy import MCPError
    from truthound.stores.backends.memory import MemoryStore
    from truthound.stores.caching.store import CachedStore

    base = MemoryStore()
    cache = CachedStore(base)
    cache._store = cache
    try:
        with pytest.raises(MCPError, match='invalid_binding'):
            StoreOwner(StoreRuntime(cache), {'native_group':'store_services'})
    finally:
        cache.close()
        base.close()
