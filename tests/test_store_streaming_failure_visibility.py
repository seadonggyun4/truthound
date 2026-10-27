"""Streaming writes and restored sessions must retain failed/uncertain work."""

import asyncio

import pytest

from tests.synthetic_s3 import MemoryS3
from truthound.stores.streaming.base import (
    CompressionType,
    StreamingConfig,
    StreamingFormat,
    StreamSession,
    StreamStatus,
)
from truthound.stores.streaming.writer import (
    AsyncStreamWriter,
    BufferedStreamWriter,
    StreamFlushError,
)

pytestmark = pytest.mark.fault


def session(config):
    return StreamSession("session", "run", "synthetic", config=config)


@pytest.mark.parametrize("failure", ["serialize", "write"])
def test_failed_flush_with_no_retries_never_commits_phantom_chunk_or_discards_records(failure):
    config = StreamingConfig(buffer_size=10, flush_interval_seconds=0, max_retries=0)
    native = BufferedStreamWriter(session(config), config)
    native.write({"x": 1})

    def failed(*args):
        raise OSError("synthetic-private-stream-write")

    if failure == "serialize":
        native.serializer.serialize_batch = failed
    else:
        native._write_chunk = failed
    with pytest.raises(StreamFlushError):
        native.flush()
    assert native.session.status == StreamStatus.FAILED
    assert native.session.chunks == [] and native.metrics.records_written == 0
    assert native.buffer.records == [{"x": 1}]
    assert "synthetic-private-stream-write" not in str(native.metrics.to_dict())


@pytest.mark.parametrize("asynchronous", [False, True])
def test_failed_context_exit_never_becomes_completed(asynchronous):
    config = StreamingConfig(flush_interval_seconds=0)
    native = BufferedStreamWriter(session(config), config)

    async def use():
        async with AsyncStreamWriter(native):
            raise RuntimeError("synthetic-private-stream-body")

    with pytest.raises(RuntimeError):
        if asynchronous:
            asyncio.run(use())
        else:
            with native:
                raise RuntimeError("synthetic-private-stream-body")
    assert native.session.status == StreamStatus.FAILED
    assert "synthetic-private-stream-body" not in str(native.metrics.to_dict())


def test_restored_session_preserves_config_and_datetime_metric_types():
    config = StreamingConfig(format=StreamingFormat.CSV, compression=CompressionType.GZIP,
                             buffer_size=7, flush_interval_seconds=0, max_retries=0)
    previous = session(config)
    previous.metrics.start()
    restored = StreamSession.from_dict(previous.to_dict())
    assert restored.config == config
    restored.metrics.finish()
    assert restored.metrics.average_throughput == 0


def test_background_flush_failure_is_retained_and_not_retried_on_close():
    config = StreamingConfig(flush_interval_seconds=0, max_retries=0)
    native = BufferedStreamWriter(session(config), config)
    native.write({"x": 1})
    calls = []

    def failed(*args):
        calls.append("write")
        raise OSError("synthetic-private-autoflush")

    native._write_chunk = failed
    native._auto_flush()
    assert native.session.status == StreamStatus.FAILED
    with pytest.raises(StreamFlushError):
        native.close()
    assert calls == ["write"] and not native.session.chunks


def test_native_write_cannot_acknowledge_a_chunk_larger_than_its_reader_memory_bound(tmp_path):
    from contextlib import suppress

    from truthound.stores.streaming.filesystem import StreamingFileSystemStore

    store = StreamingFileSystemStore(str(tmp_path), max_memory_mb=1, buffer_size=1, flush_interval_seconds=0)
    current = store.create_session("oversized", "synthetic")
    writer = store.create_writer(current)
    try:
        with pytest.raises(StreamFlushError):
            writer.write({"validator_name": "unicode", "success": False, "details": {"value": "한" * 200_000}})
        assert current.metrics.records_written == 0 and not current.chunks
        assert len(writer.buffer.records) == 1 and not list((tmp_path / "oversized").glob("*_chunk_*"))
    finally:
        writer.abort()
        with suppress(RuntimeError):
            store.close()


def test_untyped_legacy_false_verdict_cannot_become_a_successful_final_result(tmp_path):
    from truthound.stores.streaming.filesystem import StreamingFileSystemStore
    from truthound.stores.streaming.reader import StreamReadError

    native = StreamingFileSystemStore(str(tmp_path), flush_interval_seconds=0)
    current = native.create_session("legacy-verdict", "synthetic")
    with native.create_writer(current) as writer:
        writer.write({"validator_name": "legacy", "success": "False", "count": "3"})
    try:
        with pytest.raises(StreamReadError):
            native.finalize_result(current)
        with native.create_reader("legacy-verdict") as reader:
            assert reader.read()["success"] == "False"
    finally:
        native.close()


def test_untyped_restored_checkpoint_flag_is_rejected_before_any_native_callback():
    config = StreamingConfig(buffer_size=1, checkpoint_interval=1, flush_interval_seconds=0)
    data = session(config).to_dict()
    data["config"]["enable_checkpoints"] = "False"
    called = []
    with pytest.raises(ValueError):
        restored = StreamSession.from_dict(data)
        writer = BufferedStreamWriter(restored, restored.config)
        writer._write_session_state = lambda: called.append("unexpected-checkpoint")
        writer.write({"x": 1})
    assert not called


def test_store_close_drains_direct_writer_and_all_other_sessions_after_failure(tmp_path):
    from truthound.stores.streaming.filesystem import StreamingFileSystemStore

    native = StreamingFileSystemStore(str(tmp_path), buffer_size=10, flush_interval_seconds=0, max_retries=0)
    failed_session = native.create_session("failed", "synthetic")
    failed = native.create_writer(failed_session)
    good_session = native.create_session("good", "synthetic")
    good = native.create_writer(good_session)
    failed.write({"validator_name": "failed", "success": True})
    good.write({"validator_name": "good", "success": True})

    def no_write(*args):
        raise OSError("synthetic-private-store-close")

    failed._write_chunk = no_write
    with pytest.raises(RuntimeError, match="cleanup failed"):
        native.close()
    assert failed_session.status == StreamStatus.FAILED
    assert good._closed and good_session.status == StreamStatus.COMPLETED
    assert good_session.metrics.records_written == 1
    assert len(list(native.iter_results("good"))) == 1


def test_second_writer_for_active_session_is_rejected_before_any_effect(tmp_path):
    from truthound.stores.streaming.filesystem import StreamingFileSystemStore

    native = StreamingFileSystemStore(str(tmp_path), flush_interval_seconds=0)
    current = native.create_session("shared", "synthetic")
    writer = native.create_writer(current)
    try:
        with pytest.raises(ValueError):
            native.create_writer(current)
    finally:
        writer.close()
        native.close()


def test_csv_chunks_preserve_headers_and_original_validator_types(tmp_path):
    from truthound.stores.results import ValidatorResult
    from truthound.stores.streaming.filesystem import StreamingFileSystemStore

    native = StreamingFileSystemStore(str(tmp_path), format=StreamingFormat.CSV,
                                     buffer_size=1, flush_interval_seconds=0)
    current = native.create_session("csv", "synthetic")
    records = [ValidatorResult("first", False, count=2, details={"flag": False}),
               ValidatorResult("second", True, count=0, details={"flag": True})]
    with native.create_writer(current) as writer:
        writer.write_results(records)
    restored = list(native.iter_results("csv"))
    assert [record.to_dict() for record in restored] == [record.to_dict() for record in records]
    native.close()


def test_sqlite_writer_checkpoint_and_close_restore_real_session_state(tmp_path):
    from truthound.stores.results import ValidatorResult
    from truthound.stores.streaming.database import StreamingDatabaseStore

    url = "sqlite:///" + str(tmp_path / "native.db")
    native = StreamingDatabaseStore(url, format=StreamingFormat.CSV, compression=CompressionType.GZIP,
                                    buffer_size=1, checkpoint_interval=1, flush_interval_seconds=0)
    current = native.create_session("sqlite", "synthetic", {"original": {"flag": False}})
    writer = native.create_writer(current)
    writer.write_result(ValidatorResult("native", False, count=2))
    fresh = StreamingDatabaseStore(url, flush_interval_seconds=0)
    try:
        restored = fresh.get_session(current.session_id)
        assert restored.metrics.records_written == 1 and len(restored.chunks) == 1
        assert restored.config.format == StreamingFormat.CSV
        assert restored.config.compression == CompressionType.GZIP
        assert restored.metadata == current.metadata
        restored.metrics.finish()
        native.close()
        assert writer._closed
        assert fresh.get_session(current.session_id).status == StreamStatus.COMPLETED
    finally:
        native.close()
        fresh.close()


def test_sqlite_close_preserves_failed_status_and_releases_native_session(tmp_path):
    from truthound.stores.streaming.database import StreamingDatabaseStore

    url = "sqlite:///" + str(tmp_path / "failed.db")
    native = StreamingDatabaseStore(url, flush_interval_seconds=0)
    current = native.create_session("failed", "synthetic")
    writer = native.create_writer(current)
    writer.write({"validator_name": "native", "success": False})
    writer._get_db_session()

    def failed(*args):
        raise OSError("synthetic-private-db-failure")

    writer._write_chunk = failed
    with pytest.raises(RuntimeError):
        native.close()
    assert writer._closed and writer._db_session is None
    fresh = StreamingDatabaseStore(url, flush_interval_seconds=0)
    try:
        assert fresh.get_session(current.session_id).status == StreamStatus.FAILED
    finally:
        fresh.close()


def test_sqlite_same_count_record_corruption_is_detected_before_native_read(tmp_path):
    import json

    from sqlalchemy import update

    from truthound.stores.streaming.database import StreamingDatabaseStore

    native = StreamingDatabaseStore("sqlite:///" + str(tmp_path / "integrity.db"), flush_interval_seconds=0)
    current = native.create_session("integrity", "synthetic")
    with native.create_writer(current) as writer:
        writer.write({"validator_name": "original", "success": False, "count": 3})
    with native._engine.begin() as connection:
        connection.execute(update(native._results_table).values(data_json=json.dumps({"validator_name": "changed", "success": True})))
    try:
        with pytest.raises(RuntimeError, match="checksum"):
            list(native.iter_results("integrity"))
    finally:
        native.close()


def test_resume_rejects_unmanifested_committed_chunk_instead_of_overwriting_it(tmp_path):
    import json
    import subprocess
    import sys

    from truthound.stores.streaming.filesystem import StreamingFileSystemStore

    code = """
import os,sys
from truthound.stores.streaming.filesystem import StreamingFileSystemStore
store=StreamingFileSystemStore(sys.argv[1],buffer_size=1,flush_interval_seconds=0,checkpoint_interval=100)
session=store.create_session('recovery','synthetic')
writer=store.create_writer(session)
writer.write({'validator_name':'first','success':False})
print(session.session_id,flush=True)
os._exit(0)
"""
    completed = subprocess.run([sys.executable, "-c", code, str(tmp_path)], capture_output=True, text=True, timeout=10, check=True)
    identity = completed.stdout.strip()
    chunk = next((tmp_path / "recovery").glob("*.jsonl"))
    before = chunk.read_bytes()
    manifest = json.loads((tmp_path / "recovery" / "_manifest.json").read_text())
    assert manifest["chunks"] == []
    fresh = StreamingFileSystemStore(str(tmp_path), flush_interval_seconds=0)
    try:
        with pytest.raises(RuntimeError, match="reconciliation"):
            fresh.resume_session(identity)
        assert chunk.read_bytes() == before
    finally:
        fresh.close()


def test_restored_gzip_reader_uses_persisted_config_with_custom_manifest(tmp_path):
    from truthound.stores.results import ValidatorResult
    from truthound.stores.streaming.filesystem import StreamingFileSystemStore

    native = StreamingFileSystemStore(str(tmp_path), compression=CompressionType.GZIP,
                                     manifest_filename="state.json", flush_interval_seconds=0)
    current = native.create_session("compressed", "synthetic")
    original = ValidatorResult("failure", False, count=3)
    with native.create_writer(current) as writer:
        writer.write_result(original)
    native.close()
    fresh = StreamingFileSystemStore(str(tmp_path), manifest_filename="state.json", flush_interval_seconds=0)
    with fresh.create_reader("compressed") as reader:
        assert list(reader.iter_results())[0].to_dict() == original.to_dict()
    fresh.close()


@pytest.mark.parametrize("run_id", ["../outside", "..\\outside", "absolute", ".", ".."])
def test_native_stream_run_identifiers_cannot_escape_selected_store_root(tmp_path, run_id):
    from truthound.stores.streaming.filesystem import StreamingFileSystemStore

    if run_id == "absolute":
        run_id = str(tmp_path / "absolute-run")
    root = tmp_path / "selected"
    native = StreamingFileSystemStore(str(root), flush_interval_seconds=0)
    with pytest.raises(ValueError):
        native.create_session(run_id, "synthetic")
    assert not root.exists()


def test_finalize_drains_direct_writer_and_refuses_failed_session(tmp_path):
    from truthound.stores.results import ValidatorResult
    from truthound.stores.streaming.filesystem import StreamingFileSystemStore

    native = StreamingFileSystemStore(str(tmp_path), flush_interval_seconds=0)
    current = native.create_session("finalize", "synthetic")
    writer = native.create_writer(current)
    writer.write_result(ValidatorResult("pending", False, count=3, severity="high"))
    result = native.finalize_result(current)
    assert writer._closed and result.statistics.total_validators == 1
    assert result.statistics.total_issues == 3
    assert result.statistics.execution_time_ms >= 0
    current.status = StreamStatus.FAILED
    with pytest.raises(RuntimeError):
        native.finalize_result(current)
    native.close()


def test_manifest_publication_failure_preserves_previous_bytes(tmp_path, monkeypatch):
    import os

    from truthound.stores.streaming.filesystem import StreamingFileSystemStore

    native = StreamingFileSystemStore(str(tmp_path), flush_interval_seconds=0)
    current = native.create_session("atomic", "synthetic")
    path = tmp_path / "atomic" / "_manifest.json"
    previous = path.read_bytes()
    writer = native.create_writer(current)

    def failed(*args):
        raise OSError("synthetic-private-publish")

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", failed)
        with pytest.raises(StreamFlushError):
            writer.close()
    assert path.read_bytes() == previous
    assert not list(path.parent.glob("*.tmp"))
    with pytest.raises(RuntimeError):
        native.close()


def test_reader_rejects_tampered_chunk_path_before_reading_outside_run(tmp_path):
    import json

    from truthound.stores.streaming.filesystem import StreamingFileSystemStore

    root = tmp_path / "selected"
    native = StreamingFileSystemStore(str(root), buffer_size=1, flush_interval_seconds=0)
    current = native.create_session("tampered", "synthetic")
    with native.create_writer(current) as writer:
        writer.write({"validator_name": "inside", "success": True})
    outside = tmp_path / "outside.jsonl"
    outside.write_text('{"validator_name":"outside","success":true}\n')
    manifest = root / "tampered" / "_manifest.json"
    data = json.loads(manifest.read_text())
    data["chunks"][0]["path"] = str(outside)
    manifest.write_text(json.dumps(data))
    with pytest.raises((ValueError, RuntimeError)), native.create_reader("tampered") as reader:
        reader.read()
    native.close()


class MultipartMemoryS3(MemoryS3):
    def __init__(self, failure):
        super().__init__()
        self.failure = failure
        self.pending = {}
        self.abort_failure = failure == "part_and_abort"
        self.exceptions.NoSuchUpload = KeyError

    def create_multipart_upload(self, *, Key, **kwargs):
        identity = "owned-upload-" + str(len(self.pending))
        self.pending[identity] = (Key, {})
        return {"UploadId": identity}

    def upload_part(self, *, UploadId, PartNumber, Body, **kwargs):
        if self.failure == "part_and_abort":
            raise OSError("synthetic-private-part")
        self.pending[UploadId][1][PartNumber] = bytes(Body)
        return {"ETag": "part-" + str(PartNumber)}

    def complete_multipart_upload(self, *, UploadId, Key, IfNoneMatch=None, **kwargs):
        _, parts = self.pending[UploadId]
        reply = self.put_object(Key=Key, Body=b"".join(parts[index] for index in sorted(parts)), IfNoneMatch=IfNoneMatch)
        del self.pending[UploadId]
        return {} if self.failure == "lost_complete" else reply

    def abort_multipart_upload(self, *, UploadId, **kwargs):
        if self.abort_failure:
            raise PermissionError("synthetic-private-abort")
        del self.pending[UploadId]
        return {"ResponseMetadata": {"HTTPStatusCode": 204}}


def test_s3_multipart_uses_configured_concurrency_and_orders_receipts():
    import threading

    from truthound.stores.streaming.s3 import StreamingS3Store

    class ConcurrentParts(MultipartMemoryS3):
        def __init__(self):
            super().__init__(None)
            self.lock = threading.Lock()
            self.overlap = threading.Event()
            self.active = self.maximum = 0
            self.receipts = []

        def upload_part(self, **arguments):
            with self.lock:
                self.active += 1
                self.maximum = max(self.maximum, self.active)
                if self.active == 2:
                    self.overlap.set()
            try:
                assert self.overlap.wait(1), "Native multipart did not overlap upload workers"
                return super().upload_part(**arguments)
            finally:
                with self.lock:
                    self.active -= 1

        def complete_multipart_upload(self, *, MultipartUpload, **arguments):
            self.receipts = MultipartUpload["Parts"]
            return super().complete_multipart_upload(**arguments)

    client = ConcurrentParts()
    store = StreamingS3Store("synthetic", max_concurrency=2, multipart_part_size=5 * 1024 ** 2,
                             flush_interval_seconds=0)
    store._s3_client = client
    session = store.create_session("parallel", "synthetic")
    writer = store.create_writer(session)
    try:
        writer._multipart_upload("truthound/streaming/parallel/probe", b"x" * (10 * 1024 ** 2 + 1), {})
        assert client.maximum == 2
        assert [part["PartNumber"] for part in client.receipts] == [1, 2, 3]
        assert not client.pending and not writer._multipart_uploads
    finally:
        writer.abort()
        store.close()


@pytest.mark.parametrize("multipart", [False, True])
def test_actual_botocore_models_accept_native_conditional_writer_contract(multipart):
    import json
    from io import BytesIO

    import boto3
    from botocore.config import Config
    from botocore.stub import ANY, Stubber

    from truthound.stores.streaming.s3 import StreamingS3Store

    # Explicit synthetic credentials prevent credential-provider discovery.
    # Stubber intercepts every request before HTTP; there is no live S3 test.
    client = boto3.client("s3", region_name="us-east-1", aws_access_key_id="synthetic",
                          aws_secret_access_key="synthetic", endpoint_url="http://127.0.0.1:9",
                          config=Config(retries={"max_attempts": 0}))
    store = StreamingS3Store("synthetic", compression=CompressionType.NONE, buffer_size=1,
                             multipart_threshold=1 if multipart else 8 * 1024 ** 2,
                             flush_interval_seconds=0)
    store._s3_client = client
    manifest = "truthound/streaming/sdk/_manifest.json"
    chunk_key = "truthound/streaming/sdk/sdk_chunk_000000.jsonl"
    with Stubber(client) as stub:
        stub.add_response("head_bucket", {}, {"Bucket": "synthetic"})
        stub.add_client_error("get_object", "NoSuchKey", http_status_code=404,
                              expected_params={"Bucket": "synthetic", "Key": manifest})
        stub.add_response("put_object", {"ETag": '"manifest-one"'}, {
            "Bucket": "synthetic", "Key": manifest, "Body": ANY,
            "ContentType": "application/json", "IfNoneMatch": "*",
        })
        current = store.create_session("sdk", "synthetic")
        raw = json.dumps(current.to_dict()).encode()
        stub.add_response("get_object", {"ETag": '"manifest-one"', "Body": BytesIO(raw)},
                          {"Bucket": "synthetic", "Key": manifest})
        inventory = {"Bucket": "synthetic", "Prefix": "truthound/streaming/sdk/"}
        stub.add_response("list_objects_v2", {"Contents": [{"Key": manifest, "Size": len(raw)}]}, inventory)
        writer = store.create_writer(current)
        head = {"Bucket": "synthetic", "Key": manifest, "IfMatch": '"manifest-one"'}
        stub.add_response("head_object", {"ETag": '"manifest-one"'}, head)
        if multipart:
            stub.add_response("put_object", {"ETag": '"intent-one"'}, {
                "Bucket": "synthetic", "Key": manifest, "Body": ANY,
                "ContentType": "application/json", "IfMatch": '"manifest-one"',
            })
            stub.add_response("create_multipart_upload", {"UploadId": "synthetic-owned-upload"}, {
                "Bucket": "synthetic", "Key": chunk_key, "ContentType": "application/x-ndjson",
                "StorageClass": "STANDARD",
            })
            stub.add_response("put_object", {"ETag": '"receipt-one"'}, {
                "Bucket": "synthetic", "Key": manifest, "Body": ANY,
                "ContentType": "application/json", "IfMatch": '"intent-one"',
            })
            head = {"Bucket": "synthetic", "Key": manifest, "IfMatch": '"receipt-one"'}
            stub.add_response("head_object", {"ETag": '"receipt-one"'}, head)
            stub.add_response("upload_part", {"ETag": '"part-one"'}, {
                "Bucket": "synthetic", "Key": chunk_key, "UploadId": "synthetic-owned-upload",
                "PartNumber": 1, "Body": ANY,
            })
            stub.add_response("head_object", {"ETag": '"receipt-one"'}, head)
            stub.add_response("complete_multipart_upload", {"ETag": '"chunk-one"'}, {
                "Bucket": "synthetic", "Key": chunk_key, "UploadId": "synthetic-owned-upload",
                "IfNoneMatch": "*", "MultipartUpload": {"Parts": [{"PartNumber": 1, "ETag": '"part-one"'}]},
            })
            stub.add_response("put_object", {"ETag": '"cleared-one"'}, {
                "Bucket": "synthetic", "Key": manifest, "Body": ANY,
                "ContentType": "application/json", "IfMatch": '"receipt-one"',
            })
        else:
            stub.add_response("put_object", {"ETag": '"chunk-one"'}, {
                "Bucket": "synthetic", "Key": chunk_key, "Body": ANY,
                "ContentType": "application/x-ndjson", "IfNoneMatch": "*", "StorageClass": "STANDARD",
            })
        writer.write({"validator_name": "synthetic", "success": False})
        stub.add_response("list_objects_v2", {"Contents": [
            {"Key": manifest, "Size": len(raw)}, {"Key": chunk_key, "Size": current.chunks[0].byte_size},
        ]}, inventory)
        stub.add_response("put_object", {"ETag": '"manifest-two"'}, {
            "Bucket": "synthetic", "Key": manifest, "Body": ANY,
            "ContentType": "application/json", "IfMatch": '"cleared-one"' if multipart else '"manifest-one"',
        })
        writer.close()
        assert current.metrics.records_written == 1 and current.status == StreamStatus.COMPLETED
        stub.assert_no_pending_responses()
    store.close()


def test_s3_protocol_close_drains_writer_and_restores_persisted_codec():
    from truthound.stores.results import ValidatorResult
    from truthound.stores.streaming.s3 import StreamingS3Store

    client = MemoryS3()
    native = StreamingS3Store("synthetic", format=StreamingFormat.CSV,
                              compression=CompressionType.GZIP, flush_interval_seconds=0)
    native._s3_client = client
    current = native.create_session("codec", "synthetic")
    writer = native.create_writer(current)
    writer.write_result(ValidatorResult("native", False, count=2))
    native.close()
    assert writer._closed and current.status == StreamStatus.COMPLETED
    assert client.closed
    fresh = StreamingS3Store("synthetic", format=StreamingFormat.PARQUET,
                             compression=CompressionType.NONE, flush_interval_seconds=0)
    fresh._s3_client = client
    try:
        with fresh.create_reader("codec") as reader:
            assert list(reader.iter_results())[0].success is False
        assert fresh.get_chunk(fresh.list_chunks("codec")[0])[0].count == 2
    finally:
        fresh.close()


def test_s3_protocol_close_does_not_overwrite_failed_session():
    import json

    from truthound.stores.streaming.s3 import StreamingS3Store

    client = MemoryS3()
    native = StreamingS3Store("synthetic", flush_interval_seconds=0)
    native._s3_client = client
    current = native.create_session("failed", "synthetic")
    writer = native.create_writer(current)
    writer.write({"validator_name": "native", "success": False})

    def failed(*args):
        raise OSError("synthetic-private-s3")

    writer._write_chunk = failed
    with pytest.raises(RuntimeError):
        native.close()
    manifest = json.loads(client.objects["truthound/streaming/failed/_manifest.json"])
    assert manifest["status"] == "failed"
    assert client.closed


def test_s3_protocol_reader_detects_same_size_chunk_corruption():
    from truthound.stores.streaming.reader import StreamReadError
    from truthound.stores.streaming.s3 import StreamingS3Store

    client = MemoryS3()
    native = StreamingS3Store("synthetic", compression=CompressionType.NONE, flush_interval_seconds=0)
    native._s3_client = client
    current = native.create_session("integrity", "synthetic")
    with native.create_writer(current) as writer:
        writer.write({"validator_name": "native", "success": False, "count": 3})
    key = next(key for key in client.objects if "_chunk_" in key)
    previous = client.objects[key]
    client.objects[key] = previous.replace(b'"count": 3', b'"count": 9')
    assert client.objects[key] != previous and len(client.objects[key]) == len(previous)
    try:
        with pytest.raises(StreamReadError), native.create_reader("integrity") as reader:
            reader.read()
    finally:
        native.close()


def test_s3_protocol_second_writer_never_overwrites_an_unconfirmed_chunk():
    from contextlib import suppress

    from truthound.stores.streaming.s3 import StreamingS3Store

    client = MemoryS3()
    first = StreamingS3Store("synthetic", buffer_size=1, checkpoint_interval=100, flush_interval_seconds=0)
    first._s3_client = client
    current = first.create_session("collision", "synthetic")
    writer = first.create_writer(current)
    writer.write({"validator_name": "original", "success": False})
    key = next(key for key in client.objects if "_chunk_" in key)
    previous = client.objects[key]
    second = StreamingS3Store("synthetic", buffer_size=1, checkpoint_interval=100, flush_interval_seconds=0)
    second._s3_client = client
    other = None
    try:
        with pytest.raises((RuntimeError, StreamFlushError)):
            other = second.create_writer(second.get_session_for_run("collision"))
            other.write({"validator_name": "replacement", "success": True})
        assert client.objects[key] == previous
    finally:
        with suppress(StreamFlushError):
            writer.abort()
        if other is not None:
            with suppress(StreamFlushError):
                other.abort()
        with suppress(RuntimeError):
            first.close()
        with suppress(RuntimeError):
            second.close()


def test_s3_protocol_stale_manifest_is_rejected_before_next_chunk_and_missing_receipt_is_not_success():
    from contextlib import suppress

    from truthound.stores.streaming.s3 import StreamingS3Store

    client = MemoryS3()
    native = StreamingS3Store("synthetic", buffer_size=1, flush_interval_seconds=0)
    native._s3_client = client
    current = native.create_session("stale", "synthetic")
    writer = native.create_writer(current)
    key = "truthound/streaming/stale/_manifest.json"
    client.objects[key] += b" "
    try:
        with pytest.raises(StreamFlushError):
            writer.write({"validator_name": "stale", "success": True})
        assert not any("_chunk_" in name for name in client.objects)
    finally:
        with suppress(StreamFlushError):
            writer.abort()
        with suppress(RuntimeError):
            native.close()

    other = MemoryS3()
    store = StreamingS3Store("synthetic", flush_interval_seconds=0)
    store._s3_client = other
    original = other.put_object

    def lose_receipt(**arguments):
        original(**arguments)
        return {}

    other.put_object = lose_receipt
    try:
        with pytest.raises(RuntimeError, match="ETag"):
            store.create_session("receipt", "synthetic")
        assert "truthound/streaming/receipt/_manifest.json" in other.objects
        assert not store._active_sessions
    finally:
        store.close()


@pytest.mark.parametrize("failure", ["part_and_abort", "lost_complete"])
def test_s3_protocol_multipart_uncertainty_never_acknowledges_records_and_cleans_owned_upload(failure):
    from contextlib import suppress

    from truthound.stores.streaming.s3 import StreamingS3Store

    client = MultipartMemoryS3(failure)
    native = StreamingS3Store("synthetic", buffer_size=1, multipart_threshold=1, flush_interval_seconds=0)
    native._s3_client = client
    current = native.create_session("multipart", "synthetic")
    writer = native.create_writer(current)
    try:
        with pytest.raises(StreamFlushError):
            writer.write({"validator_name": "uncertain", "success": False})
        assert current.metrics.records_written == 0 and not current.chunks
        assert writer.buffer.records
        if failure == "part_and_abort":
            assert len(writer._multipart_uploads) == 1
            assert "multipart_abort:PermissionError" in current.metrics.errors
        else:
            assert any("_chunk_" in name for name in client.objects)
        client.abort_failure = False
        writer.abort()
        assert not client.pending and not writer._multipart_uploads
        assert current.status == StreamStatus.FAILED
        assert "synthetic-private" not in str(current.metrics.to_dict())
    finally:
        client.abort_failure = False
        with suppress(StreamFlushError):
            writer.abort()
        with suppress(RuntimeError):
            native.close()


def test_filesystem_second_owner_cannot_open_writer_or_publish_stale_manifest(tmp_path):
    from truthound.stores.streaming.filesystem import StreamingFileSystemStore

    first = StreamingFileSystemStore(str(tmp_path), flush_interval_seconds=0)
    current = first.create_session("exclusive", "synthetic")
    writer = first.create_writer(current)
    second = StreamingFileSystemStore(str(tmp_path), flush_interval_seconds=0)
    stale = second.get_session(current.session_id)
    try:
        with pytest.raises(RuntimeError):
            second.create_writer(stale)
        writer.write({"validator_name": "native", "success": False})
        writer.close()
        with pytest.raises((ValueError, RuntimeError)):
            second.create_writer(stale)
        assert second.get_session(current.session_id).status == StreamStatus.COMPLETED
    finally:
        first.close()
        second.close()


@pytest.mark.parametrize("backend", ["filesystem", "sqlite"])
def test_other_filesystem_owner_cannot_delete_live_writer_files(tmp_path, backend):
    from truthound.stores.streaming.database import StreamingDatabaseStore
    from truthound.stores.streaming.filesystem import StreamingFileSystemStore

    def factory():
        if backend == "sqlite":
            return StreamingDatabaseStore("sqlite:///" + str(tmp_path / "live.db"), flush_interval_seconds=0)
        return StreamingFileSystemStore(str(tmp_path), flush_interval_seconds=0)

    first = factory()
    current = first.create_session("live", "synthetic")
    writer = first.create_writer(current)
    second = factory()
    try:
        with pytest.raises(RuntimeError):
            second.delete_chunks("live")
        assert second.get_session_for_run("live") is not None
    finally:
        writer.abort()
        first.close()
        second.close()


def test_memory_sqlite_native_async_writer_and_reader_share_owned_database():
    from truthound.stores.streaming.database import StreamingDatabaseStore

    async def execute():
        native = StreamingDatabaseStore("sqlite:///:memory:", flush_interval_seconds=0)
        try:
            current = native.create_session("memory", "synthetic")
            async with await native.create_async_writer(current) as writer:
                await writer.write({"validator_name": "memory", "success": False})
            async with await native.create_async_reader("memory") as reader:
                assert (await reader.read())["validator_name"] == "memory"
        finally:
            native.close()

    asyncio.run(execute())


def test_memory_sqlite_failed_transaction_cannot_rollback_another_acknowledged_writer():
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from truthound.stores.streaming.database import StreamingDatabaseStore

    native = StreamingDatabaseStore("sqlite:///:memory:", buffer_size=1, checkpoint_interval=1, flush_interval_seconds=0)
    failed_session = native.create_session("failed", "synthetic")
    good_session = native.create_session("good", "synthetic")
    failed, good = native.create_writer(failed_session), native.create_writer(good_session)
    failed_database, good_database = failed._get_db_session(), good._get_db_session()
    inserted, failed_done, first_inserted = threading.Event(), threading.Event(), threading.Event()
    original_execute, original_commit = good_database.execute, good_database.commit

    def pause_execute(*args, **kwargs):
        value = original_execute(*args, **kwargs)
        inserted.set()
        return value

    def pause_commit():
        assert failed_done.wait(2)
        return original_commit()

    def fail_commit():
        first_inserted.set()
        inserted.wait(0.2)
        raise OSError("synthetic-private-transaction")

    good_database.execute, good_database.commit = pause_execute, pause_commit
    failed_database.commit = fail_commit

    def fail_write():
        try:
            with pytest.raises(StreamFlushError):
                failed.write({"validator_name": "failed", "success": False})
        finally:
            failed_done.set()

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            failing = executor.submit(fail_write)
            assert first_inserted.wait(2)
            successful = executor.submit(good.write, {"validator_name": "good", "success": True})
            failing.result(timeout=5)
            successful.result(timeout=5)
        good.close()
        assert [result.validator_name for result in native.iter_results("good")] == ["good"]
        assert failed_session.chunks == [] and failed.buffer.records
    finally:
        failed.abort()
        with pytest.raises(RuntimeError):
            native.close()


def test_sqlite_resume_refuses_uncheckpointed_records_and_duplicate_run(tmp_path):
    from truthound.stores.streaming.database import StreamingDatabaseStore

    url = "sqlite:///" + str(tmp_path / "ambiguous.db")
    first = StreamingDatabaseStore(url, buffer_size=1, checkpoint_interval=100, flush_interval_seconds=0)
    current = first.create_session("exclusive", "synthetic")
    writer = first.create_writer(current)
    writer.write({"validator_name": "native", "success": False})
    second = StreamingDatabaseStore(url, flush_interval_seconds=0)
    try:
        from sqlalchemy.exc import IntegrityError

        with pytest.raises(IntegrityError):
            second.create_session("exclusive", "synthetic")
        with pytest.raises(RuntimeError, match="owns this run"):
            second.resume_session(current.session_id)
    finally:
        writer.close()
        first.close()
        second.close()


@pytest.mark.parametrize("compression", list(CompressionType))
def test_parquet_chunks_preserve_empty_nested_and_varying_validator_details(tmp_path, compression):
    from truthound.stores.results import ValidatorResult
    from truthound.stores.streaming.filesystem import StreamingFileSystemStore

    store = StreamingFileSystemStore(str(tmp_path), format=StreamingFormat.PARQUET,
                                    compression=compression, buffer_size=1, flush_interval_seconds=0)
    current = store.create_session("parquet", "synthetic")
    records = [ValidatorResult("empty", True), ValidatorResult("nested", False, details={"items": [{"flag": False}, None]})]
    try:
        with store.create_writer(current) as writer:
            writer.write_results(records)
        assert [value.to_dict() for value in store.iter_results("parquet")] == [value.to_dict() for value in records]
        from pathlib import Path

        from truthound.stores.streaming.reader import get_decompressor

        assert all(get_decompressor(compression).decompress(Path(chunk.path).read_bytes())[:4] == b"PAR1" for chunk in current.chunks)
    finally:
        store.close()


@pytest.mark.parametrize("format", [StreamingFormat.CSV, StreamingFormat.PARQUET, StreamingFormat.JSONL])
@pytest.mark.parametrize("compression", [CompressionType.NONE, CompressionType.SNAPPY, CompressionType.ZSTD])
def test_buffered_output_is_one_readable_document_after_multiple_flushes(format, compression):
    from truthound.stores.streaming.reader import get_decompressor, get_deserializer

    config = StreamingConfig(format=format, compression=compression, buffer_size=1, flush_interval_seconds=0)
    records = [{"flag": False, "details": {}}, {"flag": True, "details": {"items": [None, 2]}}]
    with BufferedStreamWriter(session(config), config) as writer:
        for record in records:
            writer.write(record)
        payload = get_decompressor(compression).decompress(writer.get_output())
        assert list(get_deserializer(format).deserialize(payload)) == records


@pytest.mark.parametrize("backend", ["filesystem", "sqlite"])
@pytest.mark.parametrize("checkpoint", [1, 100])
def test_actual_native_process_loss_releases_lease_and_preserves_checkpoint_boundary(tmp_path, backend, checkpoint):
    import subprocess
    import sys

    from truthound.stores.streaming.database import StreamingDatabaseStore
    from truthound.stores.streaming.filesystem import StreamingFileSystemStore

    code = """
import os,sys
from truthound.stores.streaming.database import StreamingDatabaseStore
from truthound.stores.streaming.filesystem import StreamingFileSystemStore
from truthound.stores.streaming.base import StreamingFormat,CompressionType
backend,path,checkpoint=sys.argv[1:]
options=dict(buffer_size=1,flush_interval_seconds=0,checkpoint_interval=int(checkpoint),format=StreamingFormat.CSV,compression=CompressionType.GZIP)
store=StreamingDatabaseStore('sqlite:///'+path,**options) if backend=='sqlite' else StreamingFileSystemStore(path,**options)
session=store.create_session('crash','synthetic')
writer=store.create_writer(session)
writer.write({'validator_name':'before','success':False,'count':2})
print(session.session_id,flush=True)
os._exit(0)
"""
    location = tmp_path / ("native.db" if backend == "sqlite" else "native")
    completed = subprocess.run([sys.executable, "-c", code, backend, str(location), str(checkpoint)],
                               capture_output=True, text=True, check=True, timeout=10)
    native = StreamingDatabaseStore("sqlite:///" + str(location), flush_interval_seconds=0) if backend == "sqlite" else StreamingFileSystemStore(str(location), flush_interval_seconds=0)
    try:
        if checkpoint == 100:
            with pytest.raises(RuntimeError, match="reconciliation"):
                native.resume_session(completed.stdout.strip())
            with pytest.raises(RuntimeError, match="reconciliation"):
                list(native.iter_results("crash"))
            if backend == "filesystem":
                assert len(list(location.joinpath("crash").glob("*.csv.gz"))) == 1
            else:
                from sqlalchemy import func, select

                with native._engine.connect() as connection:
                    assert connection.execute(select(func.count()).select_from(native._results_table)).scalar() == 1
        else:
            current = native.resume_session(completed.stdout.strip())
            assert current.metrics.records_written == 1
            with native.create_writer(current) as writer:
                writer.write({'validator_name':'after','success':True,'count':0})
            assert [value.validator_name for value in native.iter_results("crash")] == ["before", "after"]
            assert native.get_session_for_run("crash").status == StreamStatus.COMPLETED
    finally:
        native.close()


def test_native_botocore_multipart_abort_requires_receipt_and_distinguishes_absence():
    import boto3
    from botocore.stub import Stubber

    from truthound.stores.streaming.s3 import StreamingS3Config
    from truthound.stores.streaming.s3_multipart import abort

    client = boto3.client('s3',region_name='us-east-1',aws_access_key_id='synthetic',aws_secret_access_key='synthetic')
    config = StreamingS3Config(bucket='synthetic')
    request = {'Bucket':'synthetic','Key':'truthound/streaming/run/chunk','UploadId':'synthetic-upload'}
    try:
        with Stubber(client) as stub:
            stub.add_response('abort_multipart_upload',{'ResponseMetadata':{'HTTPStatusCode':204}},request)
            assert abort(client,config,request['Key'],request['UploadId'],lambda:None) is True
            stub.add_response('abort_multipart_upload',{},request)
            with pytest.raises(RuntimeError,match='confirmed receipt'):
                abort(client,config,request['Key'],request['UploadId'],lambda:None)
            stub.add_client_error('abort_multipart_upload','NoSuchUpload',http_status_code=404,expected_params=request)
            assert abort(client,config,request['Key'],request['UploadId'],lambda:None) is False
            stub.assert_no_pending_responses()
    finally:
        client.close()
