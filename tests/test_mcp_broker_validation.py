"""Native retained-batch validation preserves effects, metrics and failures."""
from datetime import datetime
import asyncio

import polars as pl
import pytest

from truthound.mcp.broker_worker import BrokerOwner
from truthound.mcp.brokers import BrokerRuntime, BrokerCommand
from truthound.mcp.policy import MCPError
from truthound.realtime.adapters.mock import MockAdapter, MockAdapterConfig
from truthound.realtime.base import StreamingValidator, StreamingConfig
from truthound.realtime.protocols import StreamMessage

pytestmark = pytest.mark.contract


def test_native_batch_validation_stats_reset_and_no_implicit_ack(tmp_path, monkeypatch):
    async def scenario():
        from truthound.realtime.protocols import MessageBatch
        monkeypatch.chdir(tmp_path)
        validator = StreamingValidator(validators=['null'], config=StreamingConfig(error_handling='fail'))
        owner = BrokerOwner(BrokerRuntime(MockAdapter(MockAdapterConfig()), validator=validator))
        messages = [StreamMessage(value={'x': value}, key=None, partition=0, offset=index,
                                  timestamp=datetime.now(), topic='synthetic')
                    for index, value in enumerate([1, None])]
        batch = MessageBatch(messages=messages)
        identity = 'batch_' + 'a'*32
        owner.batches[identity] = (batch, 1)
        result = await owner.execute({'command':'validate_batch', 'batch_id':identity}, 100000)
        direct = StreamingValidator(validators=['null'], config=StreamingConfig(error_handling='fail')).validate_batch(batch.to_dataframe(), batch_id=identity)
        assert result['result']['issues'] == direct.to_dict()['issues']
        assert result['result']['record_count'] == 2
        assert identity in owner.batches and identity not in owner.processed
        stats = await owner.execute({'command':'validation_stats'}, 100000)
        assert stats['stats']['batch_count'] == 1 and stats['stats']['total_records'] == 2
        reset = await owner.execute({'command':'validation_reset'}, 100000)
        assert reset['stats']['batch_count'] == 0
        with pytest.raises(MCPError, match='invalid_handle'):
            await owner.execute({'command':'validate_batch','batch_id':'missing'},100000)
        await owner.close()
    asyncio.run(scenario())


def test_batch_validation_rejects_unbound_validator():
    async def scenario():
        owner = BrokerOwner(BrokerRuntime(MockAdapter(MockAdapterConfig())))
        with pytest.raises(MCPError, match='invalid_binding'):
            await owner.execute({'command':'validation_stats'},10000)
        await owner.close()
    asyncio.run(scenario())


def test_validation_command_requires_retained_batch_and_closed_limits():
    from pydantic import ValidationError
    base = dict(binding_id='broker', service_id='monitor_'+'a'*32, idempotency_key='native-validation')
    with pytest.raises(ValidationError):
        BrokerCommand.model_validate(dict(base,command='validate_batch'))
    with pytest.raises(ValidationError):
        BrokerCommand.model_validate(dict(base,command='validation_reset',batch_id='batch_'+'a'*32))


from tests.test_mcp_brokers import runtime


def test_installed_broker_native_validation_is_approved_and_replayed_once(runtime, monkeypatch):
    from tests.test_mcp_brokers import prepare, command, invoke, approved
    from tests import test_mcp_plugin_bindings as plugins
    identity = prepare(runtime, monkeypatch)
    command(runtime, identity, 'connect')
    batch = command(runtime, identity, 'consume', limit=3)
    request = dict(binding_id='local-broker',service_id=identity,command='validate_batch',
                   batch_id=batch['batch_id'],idempotency_key='native-batch-validation')
    pending = invoke(runtime[0],'brokers_execute',**request)
    assert pending['issues'] == [{'code':'approval_required'}]
    stats = command(runtime,identity,'validation_stats')
    assert stats['stats']['batch_count'] == 0
    result = approved(runtime,'brokers_execute',**request)
    payload = plugins.source_artifact(runtime[0],result['data']['resource_id'])
    from truthound.realtime.protocols import MessageBatch
    native_messages = [StreamMessage(**{**message, 'timestamp': datetime.fromisoformat(message['timestamp']) if isinstance(message['timestamp'], str) else message['timestamp']}) for message in batch['messages']]
    direct = StreamingValidator(validators=['null'],config=StreamingConfig(error_handling='fail')).validate_batch(
        MessageBatch(messages=native_messages).to_dataframe(), batch_id=batch['batch_id'])
    assert payload['result']['issues'] == direct.to_dict()['issues']
    assert invoke(runtime[0],'brokers_execute',**request) == result
    assert command(runtime,identity,'validation_stats',key='after-validation')['stats']['batch_count'] == 1
    assert command(runtime,identity,'validation_reset')['stats']['batch_count'] == 0
