"""Owned native local-file streams with explicit cursor and retry semantics."""

import json
from contextlib import suppress
from typing import Annotated, Literal

from pydantic import Field

from truthound.mcp.contracts import FileInput, Name, WireModel
from truthound.mcp.ingestion import capture
from truthound.mcp.operations import envelope
from truthound.mcp.policy import MCPError
from truthound.mcp.session_contracts import Handle, ResourceInfo
from truthound.mcp.sources import SourceProcess


class StreamConfig(WireModel):
    chunk_size: Annotated[int, Field(ge=1, le=10_000_000)] = 100_000
    columns: list[Name] | None = None
    skip_rows: Annotated[int, Field(ge=0)] = 0
    max_rows: Annotated[int, Field(ge=1)] | None = None


class CSVConfig(StreamConfig):
    separator: Annotated[str, Field(min_length=1, max_length=1)] = ","
    has_header: bool = True


class MappedConfig(StreamConfig):
    use_mmap: bool = True


class CSVSpec(WireModel):
    format: Literal["csv"]
    config: CSVConfig = Field(default_factory=CSVConfig)


class JSONLSpec(WireModel):
    format: Literal["jsonl"]
    config: StreamConfig = Field(default_factory=StreamConfig)


class MappedSpec(WireModel):
    format: Literal["parquet", "arrow"]
    config: MappedConfig = Field(default_factory=MappedConfig)


class StreamOpen(FileInput):
    source: Annotated[CSVSpec | JSONLSpec | MappedSpec, Field(discriminator="format")]


class StreamInput(WireModel):
    stream_id: Handle


class StreamNext(StreamInput):
    idempotency_key: Name


class StreamBatch(WireModel):
    artifact: ResourceInfo | None
    exhausted: bool


class StreamProcess(SourceProcess):
    worker_module = "truthound.mcp.stream_worker"


class StreamSession:
    def __init__(self, session):
        self.session = session
        self.workers = {}
        self.receipts = {}

    def close(self):
        for worker in self.workers.values():
            worker.close()
        self.workers.clear()
        self.receipts.clear()

    def reap(self):
        for identity, worker in list(self.workers.items()):
            try:
                self.session.resources.get(identity, "source")
                if worker.closed:
                    raise MCPError("invalid_handle")
            except MCPError:
                worker.close()
                self.workers.pop(identity, None)
                self.receipts = {
                    key: value for key, value in self.receipts.items() if key[0] != identity
                }
                with suppress(MCPError):
                    self.session.resources.release_owned(identity)

    async def call(self, operation, arguments):
        session = self.session
        resources, policy = session.resources, session.policy
        pending = []
        worker = snapshot = None
        try:
            if operation == "streams_open":
                config = arguments.source.config.model_dump()
                source = policy.resolve(
                    arguments.path,
                    suffixes={".csv", ".jsonl", ".ndjson", ".parquet", ".arrow", ".ipc"},
                )
                identity, snapshot_path = resources.reserve("artifact")
                pending.append(identity)
                await capture(policy, source, snapshot_path)
                snapshot = resources.commit(
                    identity,
                    kind="artifact",
                    media_type="application/vnd.truthound.stream-input",
                    origins=(source,),
                )
                identity, path = resources.reserve("source")
                pending.append(identity)
                path.write_text(
                    json.dumps({"format": arguments.source.format, "config": config}),
                    encoding="utf-8",
                )
                worker = StreamProcess(max(0, snapshot.expires - resources.clock()))
                await worker.call(
                    {
                        "operation": operation,
                        "path": str(snapshot_path),
                        "format": arguments.source.format,
                        "config": config,
                    },
                    policy.timeout_seconds,
                )
                entry = resources.commit(
                    identity,
                    kind="source",
                    parents=(snapshot.id,),
                    media_type="application/vnd.truthound.stream+json",
                    metadata={"owned_input": snapshot.id},
                )
                self.workers[entry.id] = worker
                return envelope(
                    operation,
                    data=session._info(entry),
                    provenance=session._provenance(snapshot.sha256),
                )
            entry = resources.get(arguments.stream_id, "source")
            worker = self.workers.get(entry.id)
            if worker is None or worker.closed:
                raise MCPError("invalid_handle")
            key = (entry.id, arguments.idempotency_key) if operation == "streams_next" else None
            if key in self.receipts:
                result = self.receipts[key]
                if result["artifact"] is not None:
                    result = {
                        **result,
                        "artifact": session._info(
                            resources.get(result["artifact"]["resource_id"], "asset")
                        ),
                    }
                return envelope(operation, data=result)
            if key and len(self.receipts) >= policy.max_items:
                raise MCPError("resource_exhausted")
            identity, path = resources.reserve("asset" if key else "artifact")
            pending.append(identity)
            try:
                result = await worker.call(
                    {
                        "operation": operation,
                        "output_path": str(path),
                        "output_budget": resources.available_bytes,
                    },
                    policy.timeout_seconds,
                )
                resources.get(entry.id, "source")
                if key:
                    if result == {"exhausted": True}:
                        data = {"artifact": None, "exhausted": True}
                    elif result == {"exhausted": False}:
                        asset = resources.commit(
                            identity,
                            kind="asset",
                            parents=(entry.id,),
                            media_type="application/vnd.apache.arrow.file",
                        )
                        data = {"artifact": session._info(asset), "exhausted": False}
                    else:
                        raise MCPError("worker_error")
                    self.receipts[key] = data
                    return envelope(operation, data=data)
                info = resources.commit(
                    identity,
                    kind="artifact",
                    parents=(entry.id,),
                    media_type="application/vnd.truthound.stream-info+json",
                )
                return envelope(operation, data=session._info(info))
            except BaseException:
                if key:
                    # Cursor might have advanced even when no artifact was
                    # published. Never retry from an uncertain position.
                    worker.close()
                raise
        except BaseException:
            if operation == "streams_open":
                if worker is not None:
                    worker.close()
                if snapshot is not None:
                    with suppress(MCPError):
                        resources.release(snapshot.id)
            raise
        finally:
            for identity in pending:
                resources.discard(identity)
