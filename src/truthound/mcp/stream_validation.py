"""Owned native sync/async validators and operator-pinned streaming transports."""
import json

from truthound.mcp.encoding import encode
from truthound.mcp.policy import MCPError


class StreamValidationOwner:
    commands = {'source_status', 'source_connect', 'source_disconnect', 'source_next', 'stream_start',
                'stream_next', 'stream_cancel', 'pipeline_validate', 'pipeline_stream', 'stream_validate_many'}

    def __init__(self, runtime):
        from truthound.realtime.base import StreamingSource
        from truthound.validators.base import Validator
        from truthound.validators.streaming.sources import StreamingSource as FileSource
        from truthound.validators.streaming.base import StreamingValidationPipeline

        if runtime.stream_source is not None and not isinstance(runtime.stream_source, StreamingSource):
            raise MCPError('invalid_binding')
        if runtime.file_source is not None and not isinstance(runtime.file_source, FileSource):
            raise MCPError('invalid_binding')
        if runtime.stream_source is not None and runtime.file_source is not None:
            raise MCPError('invalid_binding')
        if runtime.pipeline is not None and not isinstance(runtime.pipeline, StreamingValidationPipeline):
            raise MCPError('invalid_binding')
        if runtime.streaming_validators is not None and (
            not isinstance(runtime.streaming_validators, list)
            or not runtime.streaming_validators
            or any(not isinstance(v, Validator) for v in runtime.streaming_validators)
            or len({v.name for v in runtime.streaming_validators}) != len(runtime.streaming_validators)
        ):
            raise MCPError('invalid_binding')
        self.runtime = runtime
        self.iterator = None
        self.mode = None
        self.connected = False
        self.raw_iterator = None
        self.source = runtime.stream_source
        if runtime.file_source is not None:
            self.source = file_bridge(runtime.file_source)
        if self.source is not None and self.source.is_connected:
            raise MCPError('invalid_binding')
        if runtime.file_source is not None and runtime.file_source.is_open:
            raise MCPError('invalid_binding')

    async def cancel(self):
        iterator, self.iterator = self.iterator, None
        if iterator is not None:
            if self.mode == 'async':
                await iterator.aclose()
            else:
                iterator.close()
        self.mode = None

    async def close(self):
        try:
            await self.cancel()
        finally:
            raw, self.raw_iterator = self.raw_iterator, None
            try:
                close = getattr(raw, 'close', None)
                if close:
                    close()
            finally:
                try:
                    if self.source is not None:
                        self.source.disconnect()
                finally:
                    self.connected = False

    def bounded(self, value, budget):
        if len(json.dumps(encode(value), allow_nan=False).encode()) > budget:
            raise MCPError('resource_exhausted')
        return value

    async def execute(self, args, budget, batches):
        command = args['command']
        if command == 'pipeline_validate':
            pipeline = self.runtime.pipeline
            if pipeline is None:
                raise MCPError('invalid_binding')
            if args['batch_id'] not in batches:
                raise MCPError('invalid_handle')
            frame = batches[args['batch_id']][0].to_dataframe()
            return self.bounded({'issues': [i.to_dict() for i in pipeline.validate(frame.lazy())]}, budget)
        if self.source is None:
            raise MCPError('invalid_binding')
        if command == 'source_status':
            return {'connected': self.connected, 'validation_active': self.iterator is not None,
                    'mode': self.mode, 'raw_cursor_active': self.raw_iterator is not None}
        if command == 'source_connect':
            if self.connected:
                raise MCPError('invalid_arguments')
            try:
                self.source.connect()
                self.connected = True
            except BaseException:
                self.source.disconnect()
                raise
            return {'connected': True}
        if command == 'source_disconnect':
            await self.close()
            return {'connected': False}
        if command == 'stream_cancel':
            await self.cancel()
            return {'cancelled': True}
        if not self.connected:
            raise MCPError('invalid_handle')
        if command == 'stream_start':
            if self.iterator is not None or self.raw_iterator is not None:
                raise MCPError('invalid_arguments')
            if self.runtime.validator is None:
                raise MCPError('invalid_binding')
            self.mode = args.get('stream_mode', 'sync')
            method = (self.runtime.validator.validate_stream_async if self.mode == 'async'
                      else self.runtime.validator.validate_stream)
            self.iterator = method(self.source, max_batches=args.get('max_batches', 100))
            return {'started': True, 'mode': self.mode}
        if command == 'stream_next':
            if self.iterator is None:
                raise MCPError('invalid_handle')
            values, exhausted = [], False
            for _ in range(args.get('limit', 100)):
                try:
                    value = (await anext(self.iterator) if self.mode == 'async' else next(self.iterator))
                except (StopIteration, StopAsyncIteration):
                    exhausted = True
                    await self.cancel()
                    break
                values.append(value.to_dict())
                self.bounded({'results': values, 'exhausted': exhausted}, budget)
            return {'results': values, 'exhausted': exhausted}
        if self.iterator is not None:
            raise MCPError('invalid_arguments')
        if command == 'source_next':
            if self.raw_iterator is None:
                self.raw_iterator = self.source.read_batches()
            values, exhausted = [], False
            for _ in range(args.get('limit', 100)):
                frame = next(self.raw_iterator, None)
                if frame is None:
                    exhausted = True
                    break
                values.append({'columns': frame.columns, 'rows': frame.to_dicts(), 'row_count': frame.height})
                self.bounded({'batches': values, 'exhausted': exhausted}, budget)
            return {'batches': values, 'exhausted': exhausted}
        if command == 'pipeline_stream':
            if self.runtime.file_source is None or self.runtime.pipeline is None or self.raw_iterator is not None:
                raise MCPError('invalid_binding')
            issues = self.runtime.pipeline.validate_source(self.runtime.file_source)
            return self.bounded({'issues': [i.to_dict() for i in issues]}, budget)
        if command == 'stream_validate_many':
            if self.runtime.file_source is None or not self.runtime.streaming_validators or self.raw_iterator is not None:
                raise MCPError('invalid_binding')
            from truthound.validators.streaming.mixin import stream_validate_many
            result = stream_validate_many(self.runtime.streaming_validators, self.runtime.file_source)
            return self.bounded({'results': {key: [i.to_dict() for i in issues]
                                           for key, issues in result.items()}}, budget)
        raise MCPError('invalid_arguments')


def file_bridge(native):
    """Adapt native file/Flight chunks to Core's realtime-source protocol."""
    from truthound.realtime.base import StreamingSource

    class NativeChunkSource(StreamingSource):
        def __init__(self):
            super().__init__()
            self.chunks = None
        def connect(self):
            try:
                native.open()
                self.chunks = iter(native)
                self._connected = True
            except BaseException:
                native.close()
                raise
        def disconnect(self):
            try:
                if self.chunks is not None:
                    close = getattr(self.chunks, 'close', None)
                    if close:
                        close()
            finally:
                self.chunks = None
                try:
                    native.close()
                finally:
                    self._connected = False
        def read_batch(self, max_records=None):
            if not self._connected:
                raise MCPError('invalid_handle')
            return next(self.chunks, None)

    return NativeChunkSource()
