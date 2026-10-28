"""Native streams, pipelines and real loopback Arrow Flight, including failures."""
import asyncio
from contextlib import contextmanager
import threading

import polars as pl
import pytest

from truthound.mcp.brokers import BrokerRuntime
from truthound.mcp.stream_validation import StreamValidationOwner
from truthound.realtime.base import StreamingValidator, StreamingConfig
from truthound.validators.streaming.base import StreamingValidationPipeline
from truthound.validators.streaming.completeness import StreamingNullValidator
from truthound.validators.streaming.sources import ArrowFlightStreamingSource, LazyFrameStreamingSource

pytestmark = pytest.mark.contract


@contextmanager
def flight_source(live=False):
    import pyarrow as pa
    import pyarrow.flight as flight
    cancelled = threading.Event()
    table = pa.table({'x': [1, None, 3, None, 5, 6, 7], 'label': list('abcdefg')})
    class Server(flight.FlightServerBase):
        def do_get(self, context, ticket):
            if ticket.ticket != b'approved':
                raise flight.FlightUnavailableError('synthetic denial')
            if live:
                import time
                def batches():
                    try:
                        while not context.is_cancelled():
                            yield table.to_batches()[0]
                            time.sleep(0.01)
                    finally:
                        cancelled.set()
                return flight.GeneratorStream(table.schema, batches())
            return flight.RecordBatchStream(table.to_reader(max_chunksize=3))
    server = Server('grpc://127.0.0.1:0')
    thread = threading.Thread(target=server.serve, daemon=True)
    thread.start()
    source = ArrowFlightStreamingSource(host='127.0.0.1', port=server.port,
                                       ticket=b'approved', chunk_size=2)
    source._test_cancelled = cancelled
    try:
        yield source, table
    finally:
        try:
            source.close()
        finally:
            server.shutdown()
            thread.join(timeout=5)
            assert not thread.is_alive()


def test_real_flight_chunks_preserve_all_rows_projection_and_close():
    with flight_source() as (source, table):
        with source:
            frames = list(source)
            actual = pl.concat(frames)
            assert actual.to_dicts() == pl.from_arrow(table).to_dicts()
            assert [len(frame) for frame in frames] == [2,2,2,1]
            assert source.rows_read == 7
        assert not source.is_open and source._client is None and source._reader is None


@pytest.mark.parametrize('mode', ['sync','async'])
def test_native_stream_sessions_continue_cancel_and_never_restart_implicitly(tmp_path, monkeypatch, mode):
    monkeypatch.chdir(tmp_path)
    source = LazyFrameStreamingSource(pl.DataFrame({'x':[1,None,3,None,5]}).lazy(), chunk_size=2)
    validator = StreamingValidator(validators=['null'],config=StreamingConfig(error_handling='fail'))
    owner = StreamValidationOwner(BrokerRuntime(None, validator=validator, file_source=source))
    async def scenario():
        await owner.execute({'command':'source_connect'},100000,{})
        await owner.execute({'command':'stream_start','stream_mode':mode,'max_batches':10},100000,{})
        first = await owner.execute({'command':'stream_next','limit':1},100000,{})
        assert len(first['results']) == 1 and not first['exhausted']
        assert validator.get_stats()['total_records'] == 2
        rest = await owner.execute({'command':'stream_next','limit':10},100000,{})
        assert rest['exhausted'] and sum(r['record_count'] for r in rest['results']) == 3
        assert validator.get_stats()['total_records'] == 5
        await owner.close()
        assert not source.is_open
    asyncio.run(scenario())


def test_native_pipeline_flight_matches_lazy_pipeline_and_releases_reader():
    with flight_source() as (source, table):
        pipeline = StreamingValidationPipeline([StreamingNullValidator()],chunk_size=2)
        expected = StreamingValidationPipeline([StreamingNullValidator()],chunk_size=2).validate(pl.from_arrow(table).lazy())
        actual = pipeline.validate_source(source)
        assert [i.to_dict() for i in actual] == [i.to_dict() for i in expected]
        assert not source.is_open


from tests.test_mcp_brokers import runtime


@pytest.mark.parametrize('source_kind,mode', [('realtime','sync'),('realtime','async'),('local','sync'),('flight','sync'),('flight','async')])
def test_installed_native_stream_approval_cursor_replay_and_cancel(runtime, monkeypatch, tmp_path, source_kind, mode):
    import json
    from contextlib import nullcontext
    from tests.test_mcp_brokers import prepare, command, approved, invoke
    from tests import test_mcp_plugin_bindings as plugins
    with (flight_source() if source_kind == 'flight' else nullcontext((None,None))) as (flight, _):
        options = {'mode':source_kind}
        if flight is not None:
            options['port'] = flight.config.port
        (tmp_path/'stream-settings.json').write_text(json.dumps(options))
        identity = prepare(runtime,monkeypatch)
        command(runtime,identity,'source_connect')
        command(runtime,identity,'stream_start',stream_mode=mode,max_batches=10)
        request = dict(binding_id='local-broker',service_id=identity,command='stream_next',limit=1,idempotency_key='stream-page-one')
        pending = invoke(runtime[0],'brokers_execute',**request)
        assert pending['issues'] == [{'code':'approval_required'}]
        first = approved(runtime,'brokers_execute',**request)
        page = plugins.source_artifact(runtime[0],first['data']['resource_id'])
        assert len(page['results']) == 1
        assert invoke(runtime[0],'brokers_execute',**request) == first
        assert command(runtime,identity,'validation_stats')['stats']['total_records'] == 2
        command(runtime,identity,'stream_cancel')
        status = command(runtime,identity,'source_status')
        assert not status['validation_active'] and status['connected']
        # Cancelling a generator does not rewind an external source.
        command(runtime,identity,'stream_start',stream_mode=mode,max_batches=10,key='restart-at-cursor')
        remaining = command(runtime,identity,'stream_next',limit=10,key='stream-remaining')
        expected = 6 if source_kind == 'realtime' else 7 if source_kind == 'flight' else 5
        assert remaining['exhausted']
        assert sum(item['record_count'] for item in remaining['results']) == expected - 2
        command(runtime,identity,'source_disconnect')
        assert not command(runtime,identity,'source_status',key='disconnected-status')['connected']


def test_installed_flight_pipeline_many_and_raw_cursor(runtime, monkeypatch, tmp_path):
    import json
    from tests.test_mcp_brokers import prepare, command
    from truthound.validators import NullValidator
    from truthound.validators.streaming.mixin import stream_validate_many
    with flight_source() as (flight, table):
        (tmp_path/'stream-settings.json').write_text(json.dumps({'mode':'flight','port':flight.config.port}))
        identity = prepare(runtime,monkeypatch)
        command(runtime,identity,'source_connect')
        result = command(runtime,identity,'pipeline_stream')
        expected = StreamingValidationPipeline([StreamingNullValidator()],chunk_size=2).validate(pl.from_arrow(table).lazy())
        assert result['issues'] == [i.to_dict() for i in expected]
        command(runtime,identity,'source_disconnect')
        command(runtime,identity,'source_connect',key='reopen-for-many')
        result = command(runtime,identity,'stream_validate_many')
        native = LazyFrameStreamingSource(pl.from_arrow(table).lazy(),chunk_size=2)
        with native:
            expected = stream_validate_many([NullValidator()],native)
        assert result['results'] == {key:[i.to_dict() for i in issues] for key,issues in expected.items()}
        command(runtime,identity,'source_disconnect',key='disconnect-many')
        command(runtime,identity,'source_connect',key='reopen-for-read')
        rows = command(runtime,identity,'source_next',limit=10)
        assert rows['exhausted']
        assert [row for batch in rows['batches'] for row in batch['rows']] == pl.from_arrow(table).to_dicts()
        command(runtime,identity,'source_disconnect',key='disconnect-read')


def test_source_failure_and_output_limit_are_not_success(tmp_path,monkeypatch):
    from truthound.mcp.policy import MCPError
    from truthound.realtime.streaming import MockStreamingSource, MockStreamingConfig
    monkeypatch.chdir(tmp_path)
    class FailingSource(MockStreamingSource):
        def read_batch(self,max_records=None):
            raise ValueError('synthetic-private-source-failure')
    source = FailingSource(MockStreamingConfig())
    owner = StreamValidationOwner(BrokerRuntime(None,stream_source=source,validator=StreamingValidator()))
    async def scenario():
        await owner.execute({'command':'source_connect'},100000,{})
        await owner.execute({'command':'stream_start'},100000,{})
        with pytest.raises(ValueError,match='synthetic-private-source-failure'):
            await owner.execute({'command':'stream_next','limit':1},100000,{})
        await owner.close()
        assert not source.is_connected
        native = LazyFrameStreamingSource(pl.DataFrame({'x':[1,None]}).lazy(),chunk_size=1)
        other = StreamValidationOwner(BrokerRuntime(None,file_source=native,validator=StreamingValidator(validators=['null'])))
        try:
            await other.execute({'command':'source_connect'},100000,{})
            await other.execute({'command':'stream_start'},100000,{})
            with pytest.raises(MCPError,match='resource_exhausted'):
                await other.execute({'command':'stream_next','limit':1},1,{})
        finally:
            await other.close()
        assert not native.is_open
    asyncio.run(scenario())


def test_flight_projection_row_limit_unknown_length_and_failed_open_cleanup():
    import pyarrow.flight as flight
    with flight_source() as (source, table):
        source.config.columns = ['x']
        source.config.max_rows = 5
        with pytest.raises(TypeError,match='unknown'):
            len(source)
        with source:
            actual = pl.concat(list(source))
            assert actual.to_dicts() == pl.from_arrow(table).select('x').head(5).to_dicts()
            assert len(source) == 5 and source.rows_read == 5
        source.config.ticket = b'unapproved'
        with pytest.raises(flight.FlightUnavailableError):
            source.open()
        assert not source.is_open and source._reader is None and source._client is None


def test_pipeline_failure_closes_only_owned_source():
    class BrokenValidator(StreamingNullValidator):
        def process_chunk(self,*args):
            raise ValueError('synthetic pipeline failure')
    pipeline = StreamingValidationPipeline([BrokenValidator()],chunk_size=2)
    source = LazyFrameStreamingSource(pl.DataFrame({'x':[1,None]}).lazy(),chunk_size=1)
    with pytest.raises(ValueError,match='synthetic pipeline failure'):
        pipeline.validate_source(source)
    assert not source.is_open
    with source:
        with pytest.raises(ValueError,match='synthetic pipeline failure'):
            pipeline.validate_source(source)
        assert source.is_open


def test_stream_wire_rejects_transport_credentials_and_cross_command_fields():
    from pydantic import ValidationError
    from truthound.mcp.brokers import BrokerCommand
    base=dict(binding_id='source',service_id='monitor_'+'a'*32,idempotency_key='source-command')
    for extra in ({'host':'127.0.0.1'},{'ticket':'secret'},{'token':'secret'},
                  {'max_batches':0},{'max_batches':10001},{'stream_mode':'python'}):
        with pytest.raises(ValidationError):
            BrokerCommand.model_validate(dict(base,command='stream_start',**extra))
    with pytest.raises(ValidationError):
        BrokerCommand.model_validate(dict(base,command='source_disconnect',stream_mode='async'))


@pytest.mark.parametrize('end', ['release','revoke','parent-close'])
def test_installed_active_flight_connection_is_cancelled_on_owner_end(runtime,monkeypatch,tmp_path,end):
    import json
    from tests.test_mcp_brokers import prepare,command,invoke
    with flight_source(live=True) as (source,_):
        (tmp_path/'stream-settings.json').write_text(json.dumps({'mode':'flight','port':source.config.port}))
        identity=prepare(runtime,monkeypatch)
        command(runtime,identity,'source_connect')
        command(runtime,identity,'stream_start',stream_mode='async',max_batches=100)
        command(runtime,identity,'stream_next',limit=1)
        if end == 'release':
            assert invoke(runtime[0],'resources_release',resource_id=identity)['execution_status'] == 'succeeded'
        elif end == 'revoke':
            (runtime[1]/'bindings.json').write_text('{}')
        else:
            runtime[0].close()
        assert source._test_cancelled.wait(10), 'owned Flight connection was not cancelled'


def test_sliced_arrow_conversion_preserves_rich_types_without_value_reconstruction():
    import pyarrow as pa
    from datetime import date
    from decimal import Decimal
    from truthound.validators.streaming.sources import _stream_frame
    table=pa.table({'n':pa.array([Decimal('1.1'),None,Decimal('3.3')],type=pa.decimal128(10,1)),
                    'date':[date(2025,1,1),None,date(2025,1,3)],
                    'binary':[b'a',None,b'c']}).slice(1)
    assert _stream_frame(table).to_dicts() == table.to_pylist()


def test_arrow_ipc_sliced_multibatch_pipeline_source_preserves_values(tmp_path):
    import pyarrow as pa
    from truthound.validators.streaming.sources import ArrowIPCStreamingSource
    table=pa.table({'x':[1,None,3,None,5,6,7], 's':list('abcdefg')})
    path=tmp_path/'synthetic.arrow'
    with pa.OSFile(str(path),'wb') as sink:
        with pa.ipc.new_file(sink,table.schema) as writer:
            writer.write_table(table,max_chunksize=3)
    with ArrowIPCStreamingSource(str(path),chunk_size=2) as source:
        frames=list(source)
        assert pl.concat(frames).to_dicts() == table.to_pylist()
        assert [len(frame) for frame in frames] == [2,2,2,1]


def test_installed_retained_batch_pipeline_preserves_ack_ownership(runtime,monkeypatch):
    from tests.test_mcp_brokers import prepare,command,approved,invoke
    from tests import test_mcp_plugin_bindings as plugins
    from truthound.realtime.protocols import MessageBatch,StreamMessage
    from datetime import datetime
    identity=prepare(runtime,monkeypatch)
    command(runtime,identity,'connect')
    batch=command(runtime,identity,'consume',limit=3)
    request=dict(binding_id='local-broker',service_id=identity,command='pipeline_validate',
                 batch_id=batch['batch_id'],idempotency_key='retained-pipeline')
    result=approved(runtime,'brokers_execute',**request)
    values=plugins.source_artifact(runtime[0],result['data']['resource_id'])
    messages=[StreamMessage(**{**m,'timestamp':datetime.fromisoformat(m['timestamp']) if isinstance(m['timestamp'],str) else m['timestamp']}) for m in batch['messages']]
    expected=StreamingValidationPipeline([StreamingNullValidator()],chunk_size=2).validate(MessageBatch(messages=messages).to_dataframe().lazy())
    assert values['issues'] == [i.to_dict() for i in expected]
    assert invoke(runtime[0],'brokers_execute',**request) == result
    assert command(runtime,identity,'status')['retained_batches'] == 1
    assert command(runtime,identity,'ack',batch_id=batch['batch_id'])['acknowledged'] == 3
