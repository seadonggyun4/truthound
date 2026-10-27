"""Explicit reconciliation of unconfirmed filesystem chunks under native ownership."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime

from truthound.stores.streaming._filesystem_io import (
    StreamFileLease,
    contained_path,
    run_lease_path,
)
from truthound.stores.streaming.base import ChunkInfo, StreamStatus


def _capture(store, run_id):
    from truthound.stores.streaming.filesystem import StreamingRecoveryRequired

    root = store._run_path(run_id)
    if not root.is_dir():
        raise ValueError("Native stream run does not exist")
    manifest = store._manifest_path(run_id)
    rows = []
    for path in sorted(root.iterdir()):
        store._guard_io()
        if path != manifest and "_chunk_" not in path.name:
            continue
        path = contained_path(root, path.absolute())
        if not path.is_file():
            raise StreamingRecoveryRequired("Native stream recovery entry is not a regular file")
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                store._guard_io()
                digest.update(block)
                size += len(block)
        rows.append({"name": path.name, "bytes": size, "sha256": digest.hexdigest()})
    fingerprint = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    session = store.get_session_for_run(run_id)
    if session is None:
        raise StreamingRecoveryRequired("Native stream recovery requires its original valid manifest")
    expected = {contained_path(root, item.path).name for item in session.chunks}
    actual = {row["name"] for row in rows if row["name"] != manifest.name}
    return root, session, rows, expected, actual, fingerprint


def status(store, run_id):
    root = store._run_path(run_id)
    if not root.is_dir():
        raise ValueError("Native stream run does not exist")
    store._guard_io()
    with StreamFileLease(run_lease_path(store._base_path, run_id)):
        _, session, _, expected, actual, fingerprint = _capture(store, run_id)
        return {"fingerprint": fingerprint, "status": session.status.value,
                "confirmed_chunks": len(expected), "unconfirmed_chunks": len(actual - expected),
                "missing_chunks": len(expected - actual)}


def reconcile(store, run_id, *, resolution, expected_fingerprint):
    from truthound.stores.streaming.filesystem import StreamingRecoveryRequired
    from truthound.stores.streaming.reader import (
        _validator_result,
        get_decompressor,
        get_deserializer,
    )
    from truthound.stores.streaming.writer import get_compressor

    if resolution not in {"adopt", "discard"} or not re.fullmatch("[0-9a-f]{64}", expected_fingerprint):
        raise ValueError("Native stream reconciliation requires an explicit resolution and fingerprint")
    root = store._run_path(run_id)
    if not root.is_dir():
        raise ValueError("Native stream run does not exist")
    store._guard_io()
    with StreamFileLease(run_lease_path(store._base_path, run_id)):
        _, session, rows, expected, actual, fingerprint = _capture(store, run_id)
        if fingerprint != expected_fingerprint:
            raise StreamingRecoveryRequired("Native stream recovery evidence changed")
        if expected - actual:
            raise StreamingRecoveryRequired("Confirmed native stream chunks are missing")
        if session.status in {StreamStatus.COMPLETED, StreamStatus.ABORTED}:
            raise StreamingRecoveryRequired("A terminal native stream cannot adopt or discard chunks")
        # Validate every confirmed chunk before any mutation. Do not let orphan
        # discovery hide corruption of an earlier acknowledged prefix.
        extra = actual - expected
        confirmed = list(session.chunks)
        session.chunks = confirmed
        for item in confirmed:
            store._guard_io()
            path = contained_path(root, item.path)
            compressed = path.read_bytes()
            if len(compressed) != item.byte_size:
                raise StreamingRecoveryRequired("Confirmed native stream chunk size changed")
            data = get_decompressor(session.config.compression).decompress(compressed)
            if item.checksum and hashlib.md5(data).hexdigest() != item.checksum:
                raise StreamingRecoveryRequired("Confirmed native stream chunk checksum changed")
            if sum(1 for _ in get_deserializer(session.config.format).deserialize(data)) != item.record_count:
                raise StreamingRecoveryRequired("Confirmed native stream chunk count changed")
        adopted = []
        if resolution == "adopt":
            suffix = "." + session.config.format.value + get_compressor(session.config.compression).get_extension()
            pattern = re.compile(re.escape(run_id) + r"_chunk_([0-9]{6,})" + re.escape(suffix))
            indexed = []
            for name in extra:
                match = pattern.fullmatch(name)
                if match is None:
                    raise StreamingRecoveryRequired("Unconfirmed native stream chunk name is ambiguous")
                indexed.append((int(match[1]), name))
            offset = sum(item.record_count for item in confirmed)
            sizes = {row["name"]: row["bytes"] for row in rows}
            for index, name in sorted(indexed):
                store._guard_io()
                if index != len(confirmed) + len(adopted) or sizes[name] > session.config.max_memory_mb * 1024 ** 2:
                    raise StreamingRecoveryRequired("Unconfirmed native stream chunk sequence or size is invalid")
                path = contained_path(root, name)
                data = get_decompressor(session.config.compression).decompress(path.read_bytes())
                records = list(get_deserializer(session.config.format).deserialize(data))
                for record in records:
                    _validator_result(record)
                adopted.append(ChunkInfo(
                    chunk_id=name[:-len(suffix)], chunk_index=index, record_count=len(records),
                    byte_size=sizes[name], start_offset=offset, end_offset=offset + len(records),
                    checksum=hashlib.md5(data).hexdigest(), path=str(path),
                ))
                offset += len(records)
        # Recheck after decoding and before all effects, including callbacks.
        store._guard_io()
        if _capture(store, run_id)[-1] != fingerprint:
            raise StreamingRecoveryRequired("Native stream recovery evidence changed during inspection")
        if resolution == "discard":
            for name in sorted(extra):
                store._guard_io()
                contained_path(root, name).unlink()
        session.chunks.extend(adopted)
        session.metrics.records_written = sum(item.record_count for item in session.chunks)
        session.metrics.bytes_written = sum(item.byte_size for item in session.chunks)
        session.metrics.chunks_written = len(session.chunks)
        session.metrics.flush_count = max(session.metrics.flush_count, len(session.chunks))
        session.checkpoint_offset = session.metrics.records_written
        session.status = StreamStatus.PAUSED
        session.updated_at = datetime.now()
        store._guard_io()
        store._write_manifest(session)
        store._active_sessions.pop(session.session_id, None)
        return {"resolution": resolution, "adopted_chunks": len(adopted),
                "discarded_chunks": len(extra) if resolution == "discard" else 0,
                "confirmed_chunks": len(session.chunks), "records": session.metrics.records_written,
                "status": session.status.value}
