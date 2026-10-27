"""Owner-local Arrow cursors over selected Core iterators; no query callbacks."""

from dataclasses import dataclass
from uuid import uuid4

from truthound.mcp.checkpoint_worker import binding_guard
from truthound.mcp.encoding import encode
from truthound.mcp.policy import MCPError


@dataclass
class ArrowCursor:
    engine_id: str
    iterator: object
    batch: object = None
    offset: int = 0
    complete: bool = False
    failed: bool = False

    def close(self):
        self.failed = True
        self.iterator.close()
        self.batch = None
        self.complete = True


def execute(owner, engine, args, budget):
    import json

    from truthound.execution.distributed.base import BaseDistributedEngine

    command = args["command"]
    engine_id = args.get("engine_id", "root")
    guard = binding_guard(owner.request)
    guard()
    if command == "arrow_stream_open":
        if not isinstance(engine, BaseDistributedEngine) or not hasattr(
            engine, "iter_arrow_batches"
        ):
            raise MCPError("unsupported_operation")
        if len(owner.arrow_streams) >= 16:
            raise MCPError("resource_exhausted")
        iterator = engine.iter_arrow_batches(args.get("batch_size", 10_000))
        identity = "stream_" + uuid4().hex
        owner.arrow_streams[identity] = ArrowCursor(engine_id, iterator)
        return {"stream_id": identity, "engine_id": engine_id}
    identity = args["stream_id"]
    cursor = owner.arrow_streams.get(identity)
    if cursor is None or cursor.engine_id != engine_id:
        raise MCPError("invalid_handle")
    if command == "arrow_stream_close":
        cursor.close()
        del owner.arrow_streams[identity]
        return {"closed": True}
    if cursor.failed:
        raise MCPError("invalid_handle")
    rows = []
    size = 128 + len(identity)
    try:
        while len(rows) < args.get("limit", 100) and not cursor.complete:
            guard()
            if cursor.batch is None or cursor.offset == cursor.batch.num_rows:
                cursor.batch = None
                cursor.offset = 0
                try:
                    cursor.batch = next(cursor.iterator)
                except StopIteration:
                    cursor.complete = True
                    break
                guard()
                if cursor.batch.num_rows == 0:
                    continue
            row = cursor.batch.slice(cursor.offset, 1).to_pylist()[0]
            size += len(json.dumps(encode(row), allow_nan=False).encode()) + 2
            if size > budget:
                raise MCPError("resource_exhausted")
            rows.append(row)
            cursor.offset += 1
        guard()
        return {"stream_id": identity, "rows": rows, "complete": cursor.complete}
    except BaseException:
        # A consumed iterator cannot be replayed after an uncertain reply.
        cursor.failed = True
        raise
