"""Explicit S3 reconciliation using object receipts and manifest compare-and-swap.

This is an optimistic multi-object protocol, not an exclusive remote lease or
an atomic transaction. A failed discard leaves a blocking manifest journal and
requires a fresh inspection and explicit reconciliation before normal use.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime

from truthound.stores.streaming.base import ChunkInfo, StreamStatus
from truthound.stores.streaming.reader import _validator_result, get_decompressor, get_deserializer
from truthound.stores.streaming.s3 import _etag_receipt, _publish_manifest, _read_manifest
from truthound.stores.streaming.writer import get_compressor

JOURNAL = "_truthound_recovery_v1"


def _idle(store, run_id):
    for writer in store._writers.values():
        if writer.session.run_id == run_id and not writer._closed:
            raise RuntimeError("Native S3 recovery requires all owned writers to be closed")
    for reader in store._readers:
        if getattr(reader, "_run_id", None) == run_id and not reader._closed:
            raise RuntimeError("Native S3 recovery requires all owned readers to be closed")


def _read_object(store, key, limit):
    client, config = store._get_s3_client(), store.config
    store._guard_io()
    head = client.head_object(Bucket=config.bucket, Key=key)
    etag = _etag_receipt(head)
    if head["ContentLength"] > limit:
        raise RuntimeError("Native S3 recovery object exceeds its inspection bound")
    store._guard_io()
    response = client.get_object(Bucket=config.bucket, Key=key, IfMatch=etag)
    body = response["Body"]
    try:
        data = body.read(limit + 1)
    finally:
        body.close()
    if len(data) > limit or len(data) != head["ContentLength"] or _etag_receipt(response) != etag:
        raise RuntimeError("Native S3 recovery object changed during inspection")
    return data, {
        "key": key,
        "etag": etag,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _key(store, run_id):
    return f"{store.config.prefix.rstrip('/')}/{run_id}/_manifest.json"


def _capture(store, run_id):
    _idle(store, run_id)
    client, config = store._get_s3_client(), store.config
    session, document, raw = _read_manifest(
        client, config, run_id, store._guard_io, allow_recovery=True
    )
    prefix = _key(store, run_id).removesuffix("_manifest.json")
    journal = document.get(JOURNAL)
    if JOURNAL in document and (
        not isinstance(journal, dict)
        or set(journal) != {"resolution", "targets"}
        or journal["resolution"] not in {"discard", "delete"}
        or not isinstance(journal["targets"], list)
    ):
        raise RuntimeError("Native S3 recovery journal is invalid")
    rows = []
    pages = iter(
        client.get_paginator("list_objects_v2").paginate(Bucket=config.bucket, Prefix=prefix)
    )
    while True:
        store._guard_io()
        page = next(pages, None)
        if page is None:
            break
        for entry in page.get("Contents", []):
            key = entry["Key"]
            if key == _key(store, run_id):
                continue
            if not key.startswith(prefix) or "/" in key[len(prefix) :]:
                raise RuntimeError("Native S3 recovery object scope is ambiguous")
            _, row = _read_object(store, key, session.config.max_memory_mb * 1024**2)
            if row["bytes"] != entry["Size"]:
                raise RuntimeError("Native S3 recovery inventory changed")
            rows.append(row)
    rows.sort(key=lambda row: row["key"])
    if len({row["key"] for row in rows}) != len(rows):
        raise RuntimeError("Native S3 recovery inventory is ambiguous")
    evidence = {
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "manifest_etag": session._manifest_etag,
        "objects": rows,
    }
    fingerprint = hashlib.sha256(
        json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    store._guard_io()
    client.head_object(
        Bucket=config.bucket, Key=_key(store, run_id), IfMatch=session._manifest_etag
    )
    return session, document, rows, fingerprint


def _confirmed(store, session):
    expected = set()
    offset = 0
    prefix = (
        f"s3://{store.config.bucket}/{_key(store, session.run_id).removesuffix('_manifest.json')}"
    )
    for index, chunk in enumerate(session.chunks):
        numbers = (
            chunk.chunk_index,
            chunk.record_count,
            chunk.byte_size,
            chunk.start_offset,
            chunk.end_offset,
        )
        if (
            any(type(value) is not int for value in numbers)
            or chunk.record_count < 1
            or chunk.byte_size < 0
        ):
            raise RuntimeError("Confirmed native S3 chunk metadata is invalid")
        if (
            chunk.chunk_index != index
            or chunk.start_offset != offset
            or chunk.end_offset != offset + chunk.record_count
        ):
            raise RuntimeError("Confirmed native S3 chunk sequence changed")
        if not chunk.path.startswith(prefix) or "/" in chunk.path[len(prefix) :]:
            raise RuntimeError("Confirmed native S3 chunk scope changed")
        key = chunk.path[len(f"s3://{store.config.bucket}/") :]
        if key in expected:
            raise RuntimeError("Confirmed native S3 chunk identity is ambiguous")
        expected.add(key)
        offset = chunk.end_offset
    return expected, offset


def _decode(session, compressed):
    data = get_decompressor(session.config.compression).decompress(compressed)
    if len(data) > session.config.max_memory_mb * 1024**2:
        raise RuntimeError("Native S3 recovery decoded chunk exceeds its inspection bound")
    return data, list(get_deserializer(session.config.format).deserialize(data))


def status(store, run_id):
    session, document, rows, fingerprint = _capture(store, run_id)
    expected, _ = _confirmed(store, session)
    actual = {row["key"] for row in rows}
    from truthound.stores.streaming.s3_multipart import pending

    uploads = pending(session)
    return {
        "fingerprint": fingerprint,
        "status": session.status.value,
        "multipart_uploads": len(uploads),
        "unresolved_upload_intents": sum(entry["upload_id"] is None for entry in uploads.values()),
        "confirmed_chunks": len(expected),
        "unconfirmed_chunks": len(actual - expected),
        "missing_chunks": len(expected - actual),
        "pending_resolution": document[JOURNAL]["resolution"] if JOURNAL in document else None,
    }


def reconcile(store, run_id, *, resolution, expected_fingerprint):
    if resolution not in {"adopt", "discard"} or not re.fullmatch(
        "[0-9a-f]{64}", expected_fingerprint
    ):
        raise ValueError("Native S3 reconciliation requires an explicit resolution and fingerprint")
    session, document, rows, fingerprint = _capture(store, run_id)
    from truthound.stores.streaming.s3_multipart import require_clear

    require_clear(session)
    if fingerprint != expected_fingerprint:
        raise RuntimeError("Native S3 recovery evidence changed")
    if session.status in {StreamStatus.COMPLETED, StreamStatus.ABORTED}:
        raise RuntimeError("A terminal native S3 stream cannot reconcile chunks")
    if JOURNAL in document and resolution != document[JOURNAL]["resolution"]:
        raise RuntimeError("Pending native S3 recovery resolution cannot be changed")
    expected, offset = _confirmed(store, session)
    actual = {row["key"]: row for row in rows}
    if expected - actual.keys():
        raise RuntimeError("Confirmed native S3 chunks are missing")
    for chunk in session.chunks:
        key = chunk.path[len(f"s3://{store.config.bucket}/") :]
        compressed, row = _read_object(store, key, session.config.max_memory_mb * 1024**2)
        data, records = _decode(session, compressed)
        if (
            row != actual[key]
            or len(compressed) != chunk.byte_size
            or len(records) != chunk.record_count
        ):
            raise RuntimeError("Confirmed native S3 chunk changed")
        if chunk.checksum and hashlib.md5(data).hexdigest() != chunk.checksum:
            raise RuntimeError("Confirmed native S3 chunk checksum changed")
    extra = sorted(actual.keys() - expected)
    adopted = []
    if resolution == "adopt":
        suffix = (
            "."
            + session.config.format.value
            + get_compressor(session.config.compression).get_extension()
        )
        prefix = _key(store, run_id).removesuffix("_manifest.json")
        pattern = re.compile(re.escape(prefix + run_id) + r"_chunk_([0-9]{6,})" + re.escape(suffix))
        indexed = []
        for key in extra:
            match = pattern.fullmatch(key)
            if match is None:
                raise RuntimeError("Unconfirmed native S3 chunk identity is ambiguous")
            indexed.append((int(match[1]), key))
        for index, key in sorted(indexed):
            if index != len(session.chunks) + len(adopted):
                raise RuntimeError("Unconfirmed native S3 chunk sequence is invalid")
            compressed, row = _read_object(store, key, session.config.max_memory_mb * 1024**2)
            if row != actual[key]:
                raise RuntimeError("Native S3 recovery chunk changed")
            data, records = _decode(session, compressed)
            if not records:
                raise RuntimeError("Unconfirmed native S3 chunk is empty")
            for record in records:
                _validator_result(record)
            adopted.append(
                ChunkInfo(
                    chunk_id=key[len(prefix) : -len(suffix)],
                    chunk_index=index,
                    record_count=len(records),
                    byte_size=len(compressed),
                    start_offset=offset,
                    end_offset=offset + len(records),
                    checksum=hashlib.md5(data).hexdigest(),
                    path=f"s3://{store.config.bucket}/{key}",
                )
            )
            offset += len(records)
    if _capture(store, run_id)[-1] != fingerprint:
        raise RuntimeError("Native S3 recovery evidence changed during inspection")
    client, config = store._get_s3_client(), store.config
    session.status = StreamStatus.PAUSED
    session.updated_at = datetime.now()
    if resolution == "discard":
        journal = session.to_dict()
        journal[JOURNAL] = {"resolution": "discard", "targets": [actual[key] for key in extra]}
        store._guard_io()
        response = client.put_object(
            Bucket=config.bucket,
            Key=_key(store, run_id),
            Body=json.dumps(journal, sort_keys=True).encode(),
            ContentType="application/json",
            IfMatch=session._manifest_etag,
        )
        session._manifest_etag = _etag_receipt(response)
        for key in extra:
            store._guard_io()
            client.head_object(
                Bucket=config.bucket, Key=_key(store, run_id), IfMatch=session._manifest_etag
            )
            store._guard_io()
            response = client.delete_object(
                Bucket=config.bucket, Key=key, IfMatch=actual[key]["etag"]
            )
            if response.get("ResponseMetadata", {}).get("HTTPStatusCode") != 204:
                raise RuntimeError("Native S3 deletion has no confirmed response receipt")
    session.chunks.extend(adopted)
    session.metrics.records_written = sum(chunk.record_count for chunk in session.chunks)
    session.metrics.bytes_written = sum(chunk.byte_size for chunk in session.chunks)
    session.metrics.chunks_written = len(session.chunks)
    session.checkpoint_offset = session.metrics.records_written
    # Check all surviving object bytes before removing the blocking journal or
    # publishing adoption. Concurrent changes remain explicit failures.
    _, _, after, _ = _capture(store, run_id)
    expected_after = [actual[key] for key in sorted(expected)]
    if resolution == "adopt":
        expected_after = rows
    if after != expected_after:
        raise RuntimeError("Native S3 recovery inventory changed during mutation")
    store._guard_io()
    _publish_manifest(client, config, session, store._guard_io)
    store._active_sessions = {
        key: value for key, value in store._active_sessions.items() if value.run_id != run_id
    }
    return {
        "resolution": resolution,
        "adopted_chunks": len(adopted),
        "discarded_chunks": len(extra) if resolution == "discard" else 0,
        "confirmed_chunks": len(session.chunks),
        "records": session.metrics.records_written,
        "status": session.status.value,
    }


def delete(store, run_id):
    """Delete only inspected current objects; retain a journal on partial loss."""
    client, config = store._get_s3_client(), store.config
    try:
        session, document, rows, fingerprint = _capture(store, run_id)
    except client.exceptions.NoSuchKey:
        prefix = _key(store, run_id).removesuffix("_manifest.json")
        pages = iter(
            client.get_paginator("list_objects_v2").paginate(Bucket=config.bucket, Prefix=prefix)
        )
        while True:
            store._guard_io()
            page = next(pages, None)
            if page is None:
                return 0
            if page.get("Contents"):
                raise RuntimeError("Native S3 deletion requires its original manifest") from None
    from truthound.stores.streaming.s3_multipart import require_clear

    require_clear(session)
    if JOURNAL in document and document[JOURNAL]["resolution"] != "delete":
        raise RuntimeError("Pending native S3 reconciliation must be resolved before deletion")
    if _capture(store, run_id)[-1] != fingerprint:
        raise RuntimeError("Native S3 deletion evidence changed during inspection")
    journal = session.to_dict()
    journal["status"] = StreamStatus.PAUSED.value
    journal[JOURNAL] = {"resolution": "delete", "targets": rows}
    store._guard_io()
    response = client.put_object(
        Bucket=config.bucket,
        Key=_key(store, run_id),
        Body=json.dumps(journal, sort_keys=True).encode(),
        ContentType="application/json",
        IfMatch=session._manifest_etag,
    )
    etag = _etag_receipt(response)
    for row in rows:
        store._guard_io()
        client.head_object(Bucket=config.bucket, Key=_key(store, run_id), IfMatch=etag)
        store._guard_io()
        response = client.delete_object(Bucket=config.bucket, Key=row["key"], IfMatch=row["etag"])
        if response.get("ResponseMetadata", {}).get("HTTPStatusCode") != 204:
            raise RuntimeError("Native S3 deletion has no confirmed response receipt")
    if _capture(store, run_id)[2]:
        raise RuntimeError("Native S3 deletion inventory changed during mutation")
    store._guard_io()
    response = client.delete_object(Bucket=config.bucket, Key=_key(store, run_id), IfMatch=etag)
    if response.get("ResponseMetadata", {}).get("HTTPStatusCode") != 204:
        raise RuntimeError("Native S3 manifest deletion has no confirmed response receipt")
    store._active_sessions = {
        key: value for key, value in store._active_sessions.items() if value.run_id != run_id
    }
    store._writers = {
        key: value for key, value in store._writers.items() if value.session.run_id != run_id
    }
    return len(rows) + 1
