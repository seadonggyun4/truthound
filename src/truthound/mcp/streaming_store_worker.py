"""One approved streaming store with native child ownership and closed commands."""

import json
from pathlib import Path
from uuid import uuid4

from truthound.mcp.checkpoint_worker import binding_guard
from truthound.mcp.governance_worker import selected
from truthound.mcp.policy import MCPError
from truthound.mcp.streaming_store_services import StreamingRun, StreamingStoreRuntime
from truthound.mcp.tables import decode


def _metrics_view(metrics):
    value = metrics.to_dict()
    errors = value.pop("errors", None)
    value["error_count"] = len(errors) if isinstance(errors, list) else None
    return value


class StreamingStoreOwner:
    def __init__(self, runtime, request):
        from truthound.stores.streaming.base import StreamingValidationStore

        if (
            request["native_group"] != "streaming_store_services"
            or not isinstance(runtime, StreamingStoreRuntime)
            or not isinstance(runtime.store, StreamingValidationStore)
            or not runtime.runs
            or len(runtime.runs) > 4096
            or any(not isinstance(value, StreamingRun) for value in runtime.runs.values())
            or len({value.run_id for value in runtime.runs.values()}) != len(runtime.runs)
        ):
            raise MCPError("invalid_binding")
        self.runtime = runtime
        if not isinstance(runtime.multipart_receipts, dict) or any(
            not isinstance(alias, str)
            or not isinstance(receipts, dict)
            or any(
                not isinstance(key, str) or not isinstance(value, str) or not value
                for key, value in receipts.items()
            )
            for alias, receipts in runtime.multipart_receipts.items()
        ):
            raise MCPError("invalid_binding")
        self.multipart_receipts = {
            alias: dict(receipts) for alias, receipts in runtime.multipart_receipts.items()
        }
        self.guard = binding_guard(request)
        runtime.store.set_io_guard(self.guard)
        self.sessions, self.writers, self.readers, self.chunks = {}, {}, {}, {}
        self.recoveries = {}

    def retain(self, rows, value, prefix, maximum=16):
        for identity, previous in rows.items():
            if previous is value:
                return identity
        if len(rows) >= maximum:
            raise MCPError("resource_exhausted")
        identity = prefix + "_" + uuid4().hex
        rows[identity] = value
        return identity

    def session_view(self, identity, session):
        return {
            "session_id": identity,
            "status": session.status.value,
            "chunks": len(session.chunks),
            "checkpoint_offset": session.checkpoint_offset,
            "metrics": _metrics_view(session.metrics),
            "config": session.config.to_dict(),
        }

    def assert_idle(self, run):
        from truthound.stores.streaming.writer import AsyncStreamWriter

        for writer in self.writers.values():
            native = writer._writer if isinstance(writer, AsyncStreamWriter) else writer
            if native.session.run_id == run.run_id and not native._closed:
                raise MCPError("invalid_arguments")

    async def execute(self, args, budget, *, inputs=None):
        from truthound.core.results import ValidationRunResult
        from truthound.stores.results import ValidationResult, ValidatorResult
        from truthound.stores.streaming.reader import AsyncStreamReader
        from truthound.stores.streaming.writer import AsyncStreamWriter

        store, command = self.runtime.store, args["command"]
        self.guard()
        if command in {"chunk_release", "recovery_release"}:
            rows, key = (
                (self.chunks, "chunk_id")
                if command == "chunk_release"
                else (self.recoveries, "recovery_id")
            )
            selected(rows, args[key])
            del rows[args[key]]
            return {"released": True}
        if command == "inspect":
            offset, limit = args.get("offset", 0), args.get("limit", 100)
            return {
                key: list(rows)[offset : offset + limit]
                for key, rows in {
                    "runs": self.runtime.runs,
                    "sessions": self.sessions,
                    "writers": self.writers,
                    "readers": self.readers,
                    "chunks": self.chunks,
                    "recoveries": self.recoveries,
                    "multipart_receipts": self.multipart_receipts,
                }.items()
            }
        if command == "multipart_reconcile":
            from truthound.stores.streaming.s3 import StreamingS3Store

            if not isinstance(store, StreamingS3Store):
                raise MCPError("invalid_arguments")
            alias, fingerprint = selected(self.recoveries, args["recovery_id"])
            run = selected(self.runtime.runs, alias)
            self.assert_idle(run)
            if self.readers:
                raise MCPError("invalid_arguments")
            receipts = (
                selected(self.multipart_receipts, args["receipt_id"])
                if args.get("receipt_id") is not None
                else None
            )
            result = store.reconcile_multipart(
                run.run_id, expected_fingerprint=fingerprint, receipts=receipts
            )
            current = result.pop("fingerprint")
            del self.recoveries[args["recovery_id"]]
            self.sessions = {
                key: item for key, item in self.sessions.items() if item.run_id != run.run_id
            }
            self.chunks.clear()
            identity = self.retain(self.recoveries, (alias, current), "recovery")
            return {"recovery_id": identity, **result}
        if command == "recovery_reconcile":
            from truthound.stores.streaming.database import StreamingDatabaseStore
            from truthound.stores.streaming.filesystem import StreamingFileSystemStore
            from truthound.stores.streaming.s3 import StreamingS3Store

            if not isinstance(
                store, (StreamingFileSystemStore, StreamingDatabaseStore, StreamingS3Store)
            ):
                raise MCPError("invalid_arguments")
            alias, fingerprint = selected(self.recoveries, args["recovery_id"])
            run = selected(self.runtime.runs, alias)
            self.assert_idle(run)
            if self.readers:
                raise MCPError("invalid_arguments")
            result = store.reconcile_stream(
                run.run_id, resolution=args["resolution"], expected_fingerprint=fingerprint
            )
            self.sessions = {
                key: item for key, item in self.sessions.items() if item.run_id != run.run_id
            }
            self.chunks.clear()
            del self.recoveries[args["recovery_id"]]
            return result
        if command in {"session_create", "session_open"}:
            if len(self.sessions) >= 16:
                raise MCPError("resource_exhausted")
            run = selected(self.runtime.runs, args["run_id"])
            self.assert_idle(run)
            if command == "session_create":
                session = store.create_session(run.run_id, run.data_asset, metadata=run.metadata)
            else:
                session = store.get_session_for_run(run.run_id)
                if session is None:
                    raise MCPError("invalid_handle")
            # Keep a single native session object for each durable identity.
            for identity, previous in self.sessions.items():
                if previous.session_id == session.session_id:
                    return self.session_view(identity, previous)
            identity = self.retain(self.sessions, session, "session")
            return self.session_view(identity, session)
        if command in {
            "session_resume",
            "session_status",
            "session_release",
            "writer_open",
            "finalize",
        }:
            session = selected(self.sessions, args["session_id"])
            if command == "session_release":
                run = next(
                    value for value in self.runtime.runs.values() if value.run_id == session.run_id
                )
                self.assert_idle(run)
                if self.readers:
                    raise MCPError("invalid_arguments")
                del self.sessions[args["session_id"]]
                store._active_sessions.pop(session.session_id, None)
                return {"released": True}
            if command == "session_resume":
                run = next(
                    value for value in self.runtime.runs.values() if value.run_id == session.run_id
                )
                self.assert_idle(run)
                session = store.resume_session(session.session_id)
                self.sessions[args["session_id"]] = session
            elif command == "writer_open":
                if len(self.writers) >= 16:
                    raise MCPError("resource_exhausted")
                writer = (
                    await store.create_async_writer(session)
                    if args.get("asynchronous")
                    else store.create_writer(session)
                )
                return {"writer_id": self.retain(self.writers, writer, "writer")}
            elif command == "finalize":
                result = store.finalize_result(session)
                return {
                    "status": result.status.value,
                    "statistics": result.statistics.to_dict(),
                    "results": [item.to_dict() for item in result.results],
                }
            return self.session_view(args["session_id"], session)
        if command.startswith("writer_"):
            writer = selected(self.writers, args["writer_id"])
            asynchronous = isinstance(writer, AsyncStreamWriter)
            native = writer._writer if asynchronous else writer
            if command == "writer_status":
                with native._lock:
                    return {
                        "closed": native._closed,
                        "buffered": len(native.buffer.records),
                        "status": native.session.status.value,
                        "metrics": _metrics_view(native.metrics),
                    }
            if command in {"writer_write", "writer_write_run"}:
                if command == "writer_write_run":
                    if not inputs or set(inputs) != {"run_path"}:
                        raise MCPError("invalid_handle")
                    run = ValidationRunResult.from_dict(
                        decode(json.loads(Path(inputs["run_path"]).read_text(encoding="utf-8")))
                    )
                    stored = ValidationResult.from_validation_run_result(run)
                    if run.execution_issues:
                        # A list of validator verdicts cannot encode a failed run's
                        # execution issues; do not silently turn it into success.
                        raise MCPError("invalid_arguments")
                    records = stored.results
                else:
                    records = [ValidatorResult.from_dict(item) for item in args["records"]]
                if (
                    len(json.dumps([item.to_dict() for item in records], allow_nan=False).encode())
                    > budget
                ):
                    raise MCPError("resource_exhausted")
                if asynchronous:
                    await writer.write_results(records)
                else:
                    writer.write_results(records)
            elif command == "writer_flush":
                await writer.flush() if asynchronous else writer.flush()
            elif command in {"writer_close", "writer_abort"}:
                if command == "writer_abort":
                    await writer.abort() if asynchronous else writer.abort()
                else:
                    await writer.close() if asynchronous else writer.close()
                del self.writers[args["writer_id"]]
            else:
                raise MCPError("invalid_arguments")
            return {
                "closed": native._closed,
                "buffered": len(native.buffer.records),
                "status": native.session.status.value,
                "metrics": _metrics_view(native.metrics),
            }
        if command == "reader_open":
            if len(self.readers) >= 16:
                raise MCPError("resource_exhausted")
            run = selected(self.runtime.runs, args["run_id"])
            reader = (
                await store.create_async_reader(run.run_id)
                if args.get("asynchronous")
                else store.create_reader(run.run_id)
            )
            return {"reader_id": self.retain(self.readers, reader, "reader")}
        if command.startswith("reader_"):
            reader = selected(self.readers, args["reader_id"])
            asynchronous = isinstance(reader, AsyncStreamReader)
            native = reader._reader if asynchronous else reader
            if command == "reader_status":
                return {"closed": native._closed, "metrics": _metrics_view(native.metrics)}
            if command == "reader_read":
                records = (
                    await reader.read_batch(args.get("limit", 100))
                    if asynchronous
                    else reader.read_batch(args.get("limit", 100))
                )
                return {"records": records, "count": len(records)}
            if command == "reader_reset":
                native.reset()
            elif command == "reader_close":
                await reader.close() if asynchronous else reader.close()
                del self.readers[args["reader_id"]]
            else:
                raise MCPError("invalid_arguments")
            return {"closed": native._closed}
        if command == "chunk_get":
            chunk = selected(self.chunks, args["chunk_id"])
            return {"records": [item.to_dict() for item in store.get_chunk(chunk)]}
        run = selected(self.runtime.runs, args["run_id"])
        if command == "recovery_status":
            from truthound.stores.streaming.database import StreamingDatabaseStore
            from truthound.stores.streaming.filesystem import StreamingFileSystemStore
            from truthound.stores.streaming.s3 import StreamingS3Store

            if not isinstance(
                store, (StreamingFileSystemStore, StreamingDatabaseStore, StreamingS3Store)
            ):
                raise MCPError("invalid_arguments")
            self.assert_idle(run)
            if len(self.recoveries) >= 16:
                raise MCPError("resource_exhausted")
            result = store.recovery_status(run.run_id)
            fingerprint = result.pop("fingerprint")
            identity = self.retain(self.recoveries, (args["run_id"], fingerprint), "recovery")
            return {"recovery_id": identity, **result}
        if command == "chunks_list":
            chunks = store.list_chunks(run.run_id)
            offset, limit = args.get("offset", 0), args.get("limit", 100)
            result = []
            for chunk in chunks[offset : offset + limit]:
                identity = next(
                    (
                        key
                        for key, item in self.chunks.items()
                        if item.chunk_id == chunk.chunk_id and item.path == chunk.path
                    ),
                    None,
                )
                if identity is None:
                    identity = self.retain(self.chunks, chunk, "chunk", maximum=4096)
                result.append(
                    {
                        "chunk_id": identity,
                        "index": chunk.chunk_index,
                        "records": chunk.record_count,
                        "bytes": chunk.byte_size,
                    }
                )
            return {"chunks": result, "total": len(chunks)}
        if command == "chunks_delete":
            self.assert_idle(run)
            for reader in self.readers.values():
                native = reader._reader if isinstance(reader, AsyncStreamReader) else reader
                if (
                    getattr(native, "_run_id", None) == run.run_id
                    or getattr(native, "session", None) is not None
                    and native.session.run_id == run.run_id
                ):
                    raise MCPError("invalid_arguments")
            deleted = store.delete_chunks(run.run_id)
            self.sessions = {
                key: item for key, item in self.sessions.items() if item.run_id != run.run_id
            }
            self.chunks.clear()
            return {"deleted": deleted}
        if command == "stats":
            session = store.get_session_for_run(run.run_id)
            return {
                "exists": session is not None,
                "status": session.status.value if session else None,
                "metrics": _metrics_view(session.metrics) if session else None,
            }
        if command == "storage_size":
            from truthound.stores.streaming.filesystem import StreamingFileSystemStore

            if not isinstance(store, StreamingFileSystemStore):
                raise MCPError("invalid_arguments")
            return {"bytes": store.get_storage_size(run.run_id)}
        raise MCPError("invalid_arguments")

    async def close(self):
        from truthound.stores.streaming.writer import AsyncStreamWriter

        failures = []
        for writer in self.writers.values():
            native = writer._writer if isinstance(writer, AsyncStreamWriter) else writer
            if not native._closed:
                try:
                    native.abort()
                except Exception as error:
                    failures.append(error)
        try:
            self.runtime.store.close()
        except Exception as error:
            failures.append(error)
        if failures:
            raise MCPError("worker_error")


if __name__ == "__main__":
    from truthound.mcp.broker_worker import main

    main(StreamingStoreOwner)
