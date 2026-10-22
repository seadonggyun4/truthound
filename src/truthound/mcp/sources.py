"""Owned native datasource sessions, not materialized-dataframe stand-ins."""

import json
from contextlib import suppress
from typing import Annotated

from pydantic import Field, JsonValue

from truthound.mcp.contracts import FileInput, Name, WireModel
from truthound.mcp.model_process import OwnedModelProcess
from truthound.mcp.operations import envelope
from truthound.mcp.policy import MCPError
from truthound.mcp.session_contracts import AssetValidation, Handle


class BaseConfig(WireModel):
    name: Name | None = None
    max_rows: Annotated[int, Field(ge=1)] = 10_000_000
    max_memory_mb: Annotated[int, Field(ge=1)] = 4096
    sample_size: Annotated[int, Field(ge=1)] = 100_000
    sample_seed: int | None = 42
    cache_schema: bool = True
    strict_types: bool = False
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class PolarsConfig(BaseConfig):
    rechunk: bool = False
    streaming: bool = False


class SQLiteConfig(BaseConfig):
    pool_size: Annotated[int, Field(ge=1)] = 5
    pool_timeout: Annotated[float, Field(gt=0)] = 30.0
    query_timeout: Annotated[float, Field(gt=0)] = 300.0
    fetch_size: Annotated[int, Field(ge=1)] = 10000
    materialization_row_limit: Annotated[int, Field(ge=1)] = 100_000
    use_server_side_cursor: bool = False
    schema_name: Name | None = None
    timeout: Annotated[float, Field(gt=0)] = 5.0
    detect_types: Annotated[int, Field(ge=0, le=3)] = 3
    isolation_level: str | None = None


class SQLiteSourceOpen(FileInput):
    table: Name
    # database is bound exclusively to the private quota-accounted snapshot.
    config: SQLiteConfig = Field(default_factory=SQLiteConfig)


class DuckDBConfig(BaseConfig):
    pool_size: Annotated[int, Field(ge=1)] = 5
    pool_timeout: Annotated[float, Field(gt=0)] = 30.0
    query_timeout: Annotated[float, Field(gt=0)] = 300.0
    fetch_size: Annotated[int, Field(ge=1)] = 10000
    materialization_row_limit: Annotated[int, Field(ge=1)] = 100_000
    use_server_side_cursor: bool = False
    schema_name: Name | None = None
    threads: Annotated[int, Field(ge=1, le=16)] | None = None
    memory_limit: Annotated[str, Field(min_length=1, max_length=100)] | None = None


class DuckDBSourceOpen(FileInput):
    table: Name
    config: DuckDBConfig = Field(default_factory=DuckDBConfig)


class PolarsSourceOpen(WireModel):
    asset_id: Handle
    # The Polars constructor shares these base options. File parsing fields
    # belong to assets_capture and cannot be reapplied to an Arrow snapshot.
    config: PolarsConfig = Field(default_factory=PolarsConfig)


class SourceInput(WireModel):
    source_id: Handle


class SourceSample(SourceInput):
    n: Annotated[int, Field(ge=1, le=10_000_000)] = 1000
    seed: int | None = None


class SourceColumns(SourceInput):
    required_columns: Annotated[list[str], Field(max_length=100)]


class SourceValidation(AssetValidation):
    asset_id: None = None
    source_id: Handle


def effective_validation(arguments, asset_id):
    return AssetValidation.model_validate(
        {
            **arguments.model_dump(exclude={"source_id", "asset_id"}),
            "asset_id": asset_id,
        }
    )


class SourceProcess(OwnedModelProcess):
    worker_module = "truthound.mcp.source_worker"
    # Inline validation summaries are bounded again by the session policy.
    # Allow that policy's 1 MiB maximum plus the private protocol envelope.
    max_reply_bytes = 1024**2 + 4096

    def validate_reply(self, result):
        if result.get("completed") is not True:
            raise MCPError("worker_error")
        if set(result) == {"ok", "completed", "result"} and isinstance(result["result"], dict):
            return result["result"]
        if set(result) != {"ok", "completed"}:
            raise MCPError("worker_error")
        return result

    def close(self):
        # EOF lets the worker run the native context-manager cleanup first.
        # Bound cleanup; only this owned child is killed if it does not exit.
        if hasattr(self, "process") and not self.closed:
            with suppress(OSError, ValueError):
                self.process.stdin.close()
            with suppress(Exception):
                self.process.wait(timeout=1)
        super().close()


class SourceSession:
    def __init__(self, session):
        self.session = session
        self.workers = {}

    def release(self, identity):
        worker = self.workers.get(identity)
        if worker is not None:
            worker.close()
            self.workers.pop(identity, None)

    def close(self):
        for worker in self.workers.values():
            worker.close()
        self.workers.clear()

    def reap(self):
        for identity, worker in list(self.workers.items()):
            try:
                self.session.resources.get(identity, "source")
                if worker.closed:
                    raise MCPError("invalid_handle")
            except MCPError:
                self.release(identity)
                with suppress(MCPError):
                    self.session.resources.release_owned(identity)

    async def open_sqlite(self, arguments):
        session = self.session
        resources, policy = session.resources, session.policy
        original = policy.resolve(arguments.path, suffixes={".db", ".sqlite", ".sqlite3"})
        snapshot_id = source_id = snapshot = worker = None
        try:
            snapshot_id, snapshot_path = resources.reserve("artifact")
            source_id, source_path = resources.reserve("source")
            config = arguments.config.model_dump()
            source_path.write_text(
                json.dumps({"provider": "sqlite", "table": arguments.table, "config": config}),
                encoding="utf-8",
            )
            worker = SourceProcess(policy.handle_ttl_seconds)
            await worker.call(
                {
                    "operation": "sources_open_sqlite",
                    "source_path": str(original),
                    "snapshot_path": str(snapshot_path),
                    "snapshot_budget": min(
                        policy.max_file_bytes,
                        resources.available_bytes - source_path.stat().st_size,
                    ),
                    "table": arguments.table,
                    "config": config,
                },
                policy.timeout_seconds,
            )
            policy.authorize_origin(original)
            snapshot = resources.commit(
                snapshot_id,
                kind="artifact",
                media_type="application/vnd.sqlite3",
                origins=(original,),
            )
            entry = resources.commit(
                source_id,
                kind="source",
                parents=(snapshot.id,),
                media_type="application/vnd.truthound.source+json",
                metadata={
                    "asset_id": snapshot.id,
                    "input_kind": "artifact",
                    "owned_input": snapshot.id,
                },
            )
            worker.expire_after(max(0, entry.expires - resources.clock()))
            self.workers[entry.id] = worker
            return envelope(
                "sources_open_sqlite",
                data=session._info(entry),
                provenance=session._provenance(snapshot.sha256),
            )
        except BaseException:
            if worker is not None:
                worker.close()
            if snapshot is not None:
                with suppress(MCPError):
                    resources.release(snapshot.id)
            raise
        finally:
            for identity in (source_id, snapshot_id):
                if identity:
                    resources.discard(identity)

    async def open_duckdb(self, arguments):
        from dataclasses import replace
        from pathlib import Path

        from truthound.mcp.ingestion import capture

        session = self.session
        resources, policy = session.resources, session.policy
        original = policy.resolve(arguments.path, suffixes={".db", ".duckdb", ".ddb"})
        wal = Path(str(original) + ".wal")
        if wal.exists():
            raise MCPError("unsupported_database_mode")
        pending, worker, snapshot = [], None, None
        try:
            snapshot_id, snapshot_path = resources.reserve("artifact")
            pending.append(snapshot_id)
            source_id, source_path = resources.reserve("source")
            pending.append(source_id)
            config = arguments.config.model_dump()
            source_path.write_text(
                json.dumps({"provider": "duckdb", "table": arguments.table, "config": config}),
                encoding="utf-8",
            )
            budget = min(
                policy.max_file_bytes, resources.available_bytes - source_path.stat().st_size
            )
            if budget < 1:
                raise MCPError("resource_exhausted")
            await capture(replace(policy, max_file_bytes=budget), original, snapshot_path)
            if wal.exists():
                raise MCPError("source_changed")
            worker = SourceProcess(policy.handle_ttl_seconds)
            await worker.call(
                {
                    "operation": "sources_open_duckdb",
                    "snapshot_path": str(snapshot_path),
                    "table": arguments.table,
                    "config": config,
                },
                policy.timeout_seconds,
            )
            snapshot = resources.commit(
                snapshot_id,
                kind="artifact",
                media_type="application/vnd.duckdb",
                origins=(original,),
            )
            entry = resources.commit(
                source_id,
                kind="source",
                parents=(snapshot.id,),
                media_type="application/vnd.truthound.source+json",
                metadata={
                    "asset_id": snapshot.id,
                    "input_kind": "artifact",
                    "owned_input": snapshot.id,
                },
            )
            worker.expire_after(max(0, entry.expires - resources.clock()))
            self.workers[entry.id] = worker
            return envelope(
                "sources_open_duckdb",
                data=session._info(entry),
                provenance=session._provenance(snapshot.sha256),
            )
        except BaseException:
            if worker is not None:
                worker.close()
            if snapshot is not None:
                with suppress(MCPError):
                    resources.release(snapshot.id)
            raise
        finally:
            for identity in pending:
                resources.discard(identity)

    async def call(self, operation, arguments):
        if operation == "sources_open_sqlite":
            return await self.open_sqlite(arguments)
        if operation == "sources_open_duckdb":
            return await self.open_duckdb(arguments)
        session = self.session
        resources, policy = session.resources, session.policy
        worker = None
        pending = None
        try:
            if operation == "sources_open_polars":
                config = arguments.config.model_dump()
                asset = resources.get(arguments.asset_id, "asset")
                identity, path = resources.reserve("source")
                pending = identity
                path.write_text(
                    json.dumps({"provider": "polars", "config": config}), encoding="utf-8"
                )
                worker = SourceProcess(
                    min(policy.handle_ttl_seconds, asset.expires - resources.clock())
                )
                await worker.call(
                    {"operation": operation, "asset_path": str(asset.path), "config": config},
                    policy.timeout_seconds,
                )
                resources.get(asset.id, "asset")
                entry = resources.commit(
                    identity,
                    kind="source",
                    parents=(asset.id,),
                    media_type="application/vnd.truthound.source+json",
                    metadata={"asset_id": asset.id},
                )
                self.workers[entry.id] = worker
                return envelope(operation, data=session._info(entry))
            entry = resources.get(arguments.source_id, "source")
            worker = self.workers.get(entry.id)
            if worker is None or worker.closed:
                raise MCPError("invalid_handle")
            if operation == "sources_validate":
                return await session._validate(
                    effective_validation(arguments, entry.metadata["asset_id"]),
                    operation=operation,
                    public_arguments=arguments,
                    source_entry=entry,
                    source_worker=worker,
                    source_input_kind=entry.metadata.get("input_kind", "asset"),
                )
            identity, output = resources.reserve(
                "asset" if operation == "sources_sample" else "artifact"
            )
            pending = identity
            await worker.call(
                {
                    "operation": operation,
                    "arguments": arguments.model_dump(),
                    "output_path": str(output),
                    "output_budget": resources.available_bytes,
                },
                policy.timeout_seconds,
            )
            resources.get(entry.id, "source")
            result = resources.commit(
                identity,
                kind="asset" if operation == "sources_sample" else "artifact",
                parents=(entry.id,),
                media_type="application/vnd.apache.arrow.file"
                if operation == "sources_sample"
                else "application/vnd.truthound.source-info+json",
            )
            return envelope(operation, data=session._info(result))
        except BaseException:
            if worker is not None and (operation == "sources_open_polars" or worker.closed):
                worker.close()
            raise
        finally:
            if pending:
                resources.discard(pending)
