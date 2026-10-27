"""Fingerprint-bound reconciliation in a real file SQLite transaction."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime

from truthound.stores.streaming.base import ChunkInfo, StreamStatus


def _lease(store, run_id):
    from truthound.stores.streaming.database import _run_lease

    store.initialize()
    store._guard_io()
    lease = _run_lease(store._engine, run_id)
    if lease is None:
        raise NotImplementedError("Native database stream reconciliation requires file SQLite ownership")
    return lease


def _capture(store, connection, run_id):
    from sqlalchemy import select

    from truthound.stores.streaming.database import _decode_session

    store._guard_io()
    row = connection.execute(select(store._sessions_table).where(
        store._sessions_table.c.run_id == run_id
    )).one_or_none()
    if row is None:
        raise RuntimeError("Native database stream recovery requires its original session")
    session = _decode_session(row)
    digest = hashlib.sha256(json.dumps(dict(row._mapping), sort_keys=True, default=str).encode())
    groups = {}
    size = 0
    query = select(store._results_table).where(store._results_table.c.run_id == run_id).order_by(
        store._results_table.c.chunk_index, store._results_table.c.record_index, store._results_table.c.id,
    )
    for record in connection.execute(query):
        store._guard_io()
        content = json.dumps(dict(record._mapping), sort_keys=True, default=str).encode()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
        size += len(content)
        if size > session.config.max_memory_mb * 1024 ** 2:
            raise RuntimeError("Native database stream recovery exceeds its configured memory bound")
        groups.setdefault(record.chunk_id, []).append(record)
    return session, groups, digest.hexdigest()


def _validate_prefix(session, groups):
    from truthound.stores.streaming.database import _record_checksum

    offset = 0
    for index, chunk in enumerate(session.chunks):
        records = groups.get(chunk.chunk_id, [])
        if (
            chunk.chunk_index != index or chunk.start_offset != offset
            or chunk.end_offset != offset + chunk.record_count
            or len(records) != chunk.record_count
            or any(row.chunk_index != index or row.record_index != position for position, row in enumerate(records))
        ):
            raise RuntimeError("Confirmed native database stream prefix requires reconciliation")
        if chunk.checksum and chunk.checksum.startswith("db-json-sha256:") and _record_checksum([json.loads(row.data_json) for row in records]) != chunk.checksum:
            raise RuntimeError("Confirmed native database stream checksum changed")
        offset = chunk.end_offset


def status(store, run_id):
    with _lease(store, run_id), store._engine.connect() as connection:
        connection.exec_driver_sql("BEGIN")
        session, groups, fingerprint = _capture(store, connection, run_id)
        expected = {chunk.chunk_id for chunk in session.chunks}
        return {"fingerprint": fingerprint, "status": session.status.value,
                "confirmed_chunks": len(expected), "unconfirmed_chunks": len(set(groups) - expected),
                "missing_chunks": len(expected - set(groups))}


def reconcile(store, run_id, *, resolution, expected_fingerprint):
    from truthound.stores.streaming.database import _record_checksum, _settings_json
    from truthound.stores.streaming.reader import _validator_result
    from truthound.stores.streaming.writer import get_compressor, get_serializer

    if resolution not in {"adopt", "discard"} or not re.fullmatch("[0-9a-f]{64}", expected_fingerprint):
        raise ValueError("Native stream reconciliation requires an explicit resolution and fingerprint")
    with _lease(store, run_id):
        with store._engine.connect() as connection:
            connection.exec_driver_sql("BEGIN IMMEDIATE")
            session, groups, fingerprint = _capture(store, connection, run_id)
            if fingerprint != expected_fingerprint:
                raise RuntimeError("Native database stream recovery evidence changed")
            if session.status in {StreamStatus.COMPLETED, StreamStatus.ABORTED}:
                raise RuntimeError("A terminal native database stream cannot adopt or discard chunks")
            _validate_prefix(session, groups)
            expected = {chunk.chunk_id for chunk in session.chunks}
            extra = set(groups) - expected
            adopted = []
            if resolution == "adopt":
                indexed = []
                pattern = re.compile(re.escape(run_id) + r"_chunk_([0-9]{6,})")
                for identity in extra:
                    match = pattern.fullmatch(identity)
                    if match is None:
                        raise RuntimeError("Unconfirmed native database chunk identity is ambiguous")
                    indexed.append((int(match[1]), identity))
                offset = sum(chunk.record_count for chunk in session.chunks)
                for index, identity in sorted(indexed):
                    store._guard_io()
                    rows = groups[identity]
                    if index != len(session.chunks) + len(adopted) or any(
                        row.chunk_index != index or row.record_index != position for position, row in enumerate(rows)
                    ):
                        raise RuntimeError("Unconfirmed native database stream sequence is invalid")
                    records = [json.loads(row.data_json) for row in rows]
                    for record in records:
                        _validator_result(record)
                    serialized = get_serializer(session.config.format).serialize_batch(records)
                    compressed = get_compressor(session.config.compression).compress(serialized)
                    adopted.append(ChunkInfo(
                        identity, index, len(records), len(compressed), offset, offset + len(records),
                        checksum=_record_checksum(records), path=f"db://{run_id}/{identity}",
                    ))
                    offset += len(records)
            store._guard_io()
            if resolution == "discard" and extra:
                connection.execute(store._results_table.delete().where(
                    store._results_table.c.run_id == run_id, store._results_table.c.chunk_id.in_(extra),
                ))
            session.chunks.extend(adopted)
            session.metrics.records_written = sum(chunk.record_count for chunk in session.chunks)
            session.metrics.bytes_written = sum(chunk.byte_size for chunk in session.chunks)
            session.metrics.chunks_written = len(session.chunks)
            session.metrics.flush_count = max(session.metrics.flush_count, len(session.chunks))
            session.checkpoint_offset = session.metrics.records_written
            session.status = StreamStatus.PAUSED
            session.updated_at = datetime.now()
            store._guard_io()
            connection.execute(store._sessions_table.update().where(
                store._sessions_table.c.session_id == session.session_id,
            ).values(
                status=session.status.value, metadata_json=_settings_json(session),
                chunks_json=json.dumps([chunk.to_dict() for chunk in session.chunks]),
                metrics_json=json.dumps(session.metrics.to_dict()), updated_at=session.updated_at,
                checkpoint_offset=session.checkpoint_offset,
            ))
            store._guard_io()
            connection.commit()
        store._active_sessions.pop(session.session_id, None)
        return {"resolution": resolution, "adopted_chunks": len(adopted),
                "discarded_chunks": len(extra) if resolution == "discard" else 0,
                "confirmed_chunks": len(session.chunks), "records": session.metrics.records_written,
                "status": session.status.value}
