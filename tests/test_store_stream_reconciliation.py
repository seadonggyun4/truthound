"""Actual crashed native stream evidence requires explicit, fingerprinted recovery."""

import json
import subprocess
import sys

import pytest

from truthound.stores.streaming.filesystem import StreamingFileSystemStore

pytestmark = pytest.mark.fault


def s3_unconfirmed():
    from tests.test_store_streaming_failure_visibility import MemoryS3
    from truthound.stores.streaming.base import CompressionType
    from truthound.stores.streaming.s3 import StreamingS3Store

    client = MemoryS3()
    store = StreamingS3Store(
        "synthetic", compression=CompressionType.NONE, buffer_size=1, flush_interval_seconds=0
    )
    store._s3_client = client
    session = store.create_session("recovery", "synthetic")
    with store.create_writer(session) as writer:
        writer.write({"validator_name": "confirmed", "success": True})
    # Model an acknowledged prefix plus a chunk whose manifest publication was
    # lost. This is a synthetic object protocol, not a live cloud crash test.
    import json

    manifest = "truthound/streaming/recovery/_manifest.json"
    document = json.loads(client.objects[manifest])
    document["status"] = "failed"
    client.objects[manifest] = json.dumps(document).encode()
    extra = "truthound/streaming/recovery/recovery_chunk_000001.jsonl"
    client.objects[extra] = b'{"validator_name":"unconfirmed","success":false,"count":2}\n'
    return store, client, session.session_id, manifest, extra


@pytest.mark.parametrize("resolution", ["adopt", "discard"])
def test_s3_explicit_reconciliation_preserves_confirmed_prefix(resolution):
    store, client, identity, manifest, extra = s3_unconfirmed()
    before = dict(client.objects)
    try:
        state = store.recovery_status("recovery")
        assert (state["confirmed_chunks"], state["unconfirmed_chunks"]) == (1, 1)
        result = store.reconcile_stream(
            "recovery", resolution=resolution, expected_fingerprint=state["fingerprint"]
        )
        assert result["status"] == "paused"
        assert result["confirmed_chunks"] == (2 if resolution == "adopt" else 1)
        store._active_sessions.clear()
        current = store.resume_session(identity)
        with store.create_writer(current) as writer:
            writer.write({"validator_name": "after", "success": True})
        assert [row.validator_name for row in store.iter_results("recovery")] == (
            ["confirmed", "unconfirmed", "after"]
            if resolution == "adopt"
            else ["confirmed", "after"]
        )
        prefix = next(key for key in before if "chunk_000000" in key)
        assert client.objects[prefix] == before[prefix]
    finally:
        store.close()


def test_s3_recovery_stale_fingerprint_never_deletes_changed_chunk():
    store, client, _, manifest, extra = s3_unconfirmed()
    try:
        state = store.recovery_status("recovery")
        client.objects[extra] += b" "
        before = dict(client.objects)
        with pytest.raises(RuntimeError, match="changed"):
            store.reconcile_stream(
                "recovery", resolution="discard", expected_fingerprint=state["fingerprint"]
            )
        assert client.objects == before
    finally:
        store.close()


def test_s3_partial_discard_retains_blocking_journal_until_fresh_explicit_recovery():
    import json

    store, client, _, manifest, extra = s3_unconfirmed()
    next_key = extra.replace("000001", "000002")
    client.objects[next_key] = client.objects[extra]
    native_delete = client.delete_object

    def partial_failure(*, Key, **arguments):
        if Key == next_key:
            raise OSError("synthetic-private-delete-failure")
        return native_delete(Key=Key, **arguments)

    client.delete_object = partial_failure
    try:
        state = store.recovery_status("recovery")
        with pytest.raises(OSError):
            store.reconcile_stream(
                "recovery", resolution="discard", expected_fingerprint=state["fingerprint"]
            )
        assert extra not in client.objects and next_key in client.objects
        assert "_truthound_recovery_v1" in json.loads(client.objects[manifest])
        with pytest.raises(RuntimeError, match="pending"):
            store.create_reader("recovery")
        fresh = store.recovery_status("recovery")
        assert fresh["pending_resolution"] == "discard" and fresh["unconfirmed_chunks"] == 1
        client.delete_object = native_delete
        with pytest.raises(RuntimeError, match="changed"):
            store.reconcile_stream(
                "recovery", resolution="discard", expected_fingerprint=state["fingerprint"]
            )
        store.reconcile_stream(
            "recovery", resolution="discard", expected_fingerprint=fresh["fingerprint"]
        )
        assert "_truthound_recovery_v1" not in json.loads(client.objects[manifest])
        assert [row.validator_name for row in store.iter_results("recovery")] == ["confirmed"]
    finally:
        store.close()


def test_s3_discard_conditionally_rejects_object_changed_after_inspection():
    store, client, _, manifest, extra = s3_unconfirmed()
    native_delete = client.delete_object

    def replaced_object(*, Key, **arguments):
        client.objects[Key] = b"synthetic-replacement"
        return native_delete(Key=Key, **arguments)

    client.delete_object = replaced_object
    try:
        state = store.recovery_status("recovery")
        with pytest.raises(RuntimeError, match="precondition"):
            store.reconcile_stream(
                "recovery", resolution="discard", expected_fingerprint=state["fingerprint"]
            )
        assert client.objects[extra] == b"synthetic-replacement"
        with pytest.raises(RuntimeError, match="pending"):
            store.create_reader("recovery")
    finally:
        store.close()


def test_s3_delete_missing_receipt_cannot_report_success_and_requires_explicit_retry():
    import json

    store, client, _, manifest, extra = s3_unconfirmed()
    # The old batch path ignored Errors and counted every requested object.
    client.delete_objects = lambda **kwargs: {"Errors": [{"Key": extra, "Code": "AccessDenied"}]}
    native_delete = client.delete_object

    def lost_receipt(**arguments):
        native_delete(**arguments)
        return {}

    client.delete_object = lost_receipt
    try:
        with pytest.raises(RuntimeError, match="receipt"):
            store.delete_chunks("recovery")
        assert (
            json.loads(client.objects[manifest])["_truthound_recovery_v1"]["resolution"] == "delete"
        )
        with pytest.raises(RuntimeError, match="pending"):
            store.create_reader("recovery")
        client.delete_object = native_delete
        assert store.delete_chunks("recovery") == 2
        assert not client.objects
        assert store.get_session_for_run("recovery") is None
    finally:
        store.close()


def test_s3_live_writer_prevents_recovery_and_deletion_before_mutation():
    store, client, identity, manifest, extra = s3_unconfirmed()
    state = store.recovery_status("recovery")
    store.reconcile_stream(
        "recovery", resolution="discard", expected_fingerprint=state["fingerprint"]
    )
    current = store.resume_session(identity)
    writer = store.create_writer(current)
    try:
        before = dict(client.objects)
        with pytest.raises(RuntimeError, match="writers"):
            store.delete_chunks("recovery")
        with pytest.raises(RuntimeError, match="writers"):
            store.recovery_status("recovery")
        assert client.objects == before
    finally:
        writer.abort()
        store.close()


def test_s3_reader_does_not_silently_cross_a_changed_manifest_receipt():
    store, client, _, manifest, extra = s3_unconfirmed()
    del client.objects[extra]
    reader = store.create_reader("recovery")
    try:
        client.objects[manifest] += b" "
        with pytest.raises(RuntimeError, match="precondition"):
            reader.read()
    finally:
        reader.close()
        store.close()


def crashed(tmp_path, backend="filesystem"):
    code = """
import os,sys
from truthound.stores.streaming.filesystem import StreamingFileSystemStore
from truthound.stores.streaming.database import StreamingDatabaseStore
store=StreamingDatabaseStore('sqlite:///'+sys.argv[1]+'/native.db',buffer_size=1,checkpoint_interval=100,flush_interval_seconds=0) if sys.argv[2]=='sqlite' else StreamingFileSystemStore(sys.argv[1],buffer_size=1,checkpoint_interval=100,flush_interval_seconds=0)
session=store.create_session('crashed','synthetic')
writer=store.create_writer(session)
writer.write({'validator_name':'unknown','success':False,'count':3})
print(session.session_id,flush=True)
os._exit(0)
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path), backend],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    return result.stdout.strip()


@pytest.mark.parametrize("resolution", ["adopt", "discard"])
def test_reconcile_actual_unconfirmed_chunk_only_after_explicit_selection(tmp_path, resolution):
    identity = crashed(tmp_path)
    store = StreamingFileSystemStore(str(tmp_path), flush_interval_seconds=0)
    try:
        state = store.recovery_status("crashed")
        assert state["unconfirmed_chunks"] == 1 and state["confirmed_chunks"] == 0
        result = store.reconcile_stream(
            "crashed", resolution=resolution, expected_fingerprint=state["fingerprint"]
        )
        assert result["adopted_chunks"] == (resolution == "adopt")
        assert result["discarded_chunks"] == (resolution == "discard")
        current = store.resume_session(identity)
        with store.create_writer(current) as writer:
            writer.write({"validator_name": "after", "success": True})
        records = list(store.iter_results("crashed"))
        assert [item.validator_name for item in records] == (
            ["unknown", "after"] if resolution == "adopt" else ["after"]
        )
        if resolution == "adopt":
            assert records[0].count == 3
    finally:
        store.close()


def test_stale_fingerprint_or_corrupt_chunk_never_mutates_manifest_or_data(tmp_path):
    crashed(tmp_path)
    store = StreamingFileSystemStore(str(tmp_path), flush_interval_seconds=0)
    state = store.recovery_status("crashed")
    chunk = next((tmp_path / "crashed").glob("*.jsonl"))
    manifest = tmp_path / "crashed" / "_manifest.json"
    previous = manifest.read_bytes()
    chunk.write_bytes(b"synthetic-invalid-record")
    try:
        with pytest.raises(RuntimeError, match="changed"):
            store.reconcile_stream(
                "crashed", resolution="discard", expected_fingerprint=state["fingerprint"]
            )
        state = store.recovery_status("crashed")
        with pytest.raises(ValueError):
            store.reconcile_stream(
                "crashed", resolution="adopt", expected_fingerprint=state["fingerprint"]
            )
        assert (
            chunk.read_bytes() == b"synthetic-invalid-record" and manifest.read_bytes() == previous
        )
    finally:
        store.close()


@pytest.mark.parametrize("resolution", ["adopt", "discard"])
def test_sqlite_crash_reconciliation_updates_records_and_session_together(tmp_path, resolution):
    from truthound.stores.streaming.database import StreamingDatabaseStore

    identity = crashed(tmp_path, "sqlite")
    store = StreamingDatabaseStore(
        "sqlite:///" + str(tmp_path / "native.db"), flush_interval_seconds=0
    )
    try:
        state = store.recovery_status("crashed")
        assert state["unconfirmed_chunks"] == 1 and state["confirmed_chunks"] == 0
        result = store.reconcile_stream(
            "crashed", resolution=resolution, expected_fingerprint=state["fingerprint"]
        )
        assert result["confirmed_chunks"] == (1 if resolution == "adopt" else 0)
        current = store.resume_session(identity)
        with store.create_writer(current) as writer:
            writer.write({"validator_name": "after", "success": True})
        assert [item.validator_name for item in store.iter_results("crashed")] == (
            ["unknown", "after"] if resolution == "adopt" else ["after"]
        )
    finally:
        store.close()


def test_sqlite_recovery_failure_after_delete_rolls_back_all_mutations(tmp_path):
    from sqlalchemy import event, func, select

    from truthound.stores.streaming.database import StreamingDatabaseStore

    identity = crashed(tmp_path, "sqlite")
    store = StreamingDatabaseStore(
        "sqlite:///" + str(tmp_path / "native.db"), flush_interval_seconds=0
    )
    try:
        state = store.recovery_status("crashed")
        previous = store.get_session(identity).to_dict()

        def fail_update(conn, cursor, statement, parameters, context, executemany):
            if statement.startswith("UPDATE"):
                raise OSError("synthetic-private-session-update")

        event.listen(store._engine, "before_cursor_execute", fail_update)
        try:
            with pytest.raises(OSError):
                store.reconcile_stream(
                    "crashed", resolution="discard", expected_fingerprint=state["fingerprint"]
                )
        finally:
            event.remove(store._engine, "before_cursor_execute", fail_update)
        with store._engine.connect() as connection:
            assert (
                connection.execute(select(func.count()).select_from(store._results_table)).scalar()
                == 1
            )
        assert store.get_session(identity).to_dict() == previous
        assert store.recovery_status("crashed")["fingerprint"] == state["fingerprint"]
    finally:
        store.close()


def test_sqlite_recovery_process_loss_before_session_update_rolls_back_transaction(tmp_path):
    from sqlalchemy import func, select

    from truthound.stores.streaming.database import StreamingDatabaseStore

    crashed(tmp_path, "sqlite")
    store = StreamingDatabaseStore(
        "sqlite:///" + str(tmp_path / "native.db"), flush_interval_seconds=0
    )
    try:
        previous = store.recovery_status("crashed")
        code = """
import os,sys
from sqlalchemy import event
from truthound.stores.streaming.database import StreamingDatabaseStore
store=StreamingDatabaseStore('sqlite:///'+sys.argv[1],flush_interval_seconds=0)
state=store.recovery_status('crashed')
def stop_before_session(conn,cursor,statement,parameters,context,executemany):
    if statement.startswith('UPDATE'):
        print('after-uncommitted-delete',flush=True)
        os._exit(0)
event.listen(store._engine,'before_cursor_execute',stop_before_session)
store.reconcile_stream('crashed',resolution='discard',expected_fingerprint=state['fingerprint'])
raise AssertionError('crash point was not reached')
"""
        result = subprocess.run(
            [sys.executable, "-c", code, str(tmp_path / "native.db")],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        assert result.stdout.strip() == "after-uncommitted-delete"
        with store._engine.connect() as connection:
            assert (
                connection.execute(select(func.count()).select_from(store._results_table)).scalar()
                == 1
            )
        assert store.recovery_status("crashed")["fingerprint"] == previous["fingerprint"]
    finally:
        store.close()


@pytest.mark.parametrize("phase", ["create", "part"])
def test_s3_multipart_ownership_survives_actual_process_loss_and_blocks_resume(tmp_path, phase):
    from tests.synthetic_s3 import PersistentMultipartS3
    from truthound.stores.streaming.s3 import StreamingS3Store

    code = """
import sys
from tests.synthetic_s3 import PersistentMultipartS3
from truthound.stores.streaming.s3 import StreamingS3Store
from truthound.stores.streaming.base import CompressionType
client=PersistentMultipartS3(sys.argv[1], crash_at=sys.argv[2])
store=StreamingS3Store('synthetic',compression=CompressionType.NONE,buffer_size=1,multipart_threshold=1,flush_interval_seconds=0,max_retries=0)
store._s3_client=client
current=store.create_session('multipart-crash','synthetic')
print(current.session_id,flush=True)
writer=store.create_writer(current)
writer.write({'validator_name':'synthetic','success':False})
"""
    path = tmp_path / "objects.json"
    exited = subprocess.run(
        [sys.executable, "-c", code, str(path), phase], capture_output=True, text=True, timeout=10
    )
    assert exited.returncode == 74, exited.stderr
    identity = exited.stdout.strip()
    client = PersistentMultipartS3(path)
    assert len(client.pending) == 1
    store = StreamingS3Store("synthetic", flush_interval_seconds=0)
    store._s3_client = client
    try:
        state = store.recovery_status("multipart-crash")
        with pytest.raises(RuntimeError, match="multipart"):
            store.resume_session(identity)
        assert state["multipart_uploads"] == 1
        assert state["unresolved_upload_intents"] == (1 if phase == "create" else 0)
        before = dict(client.objects)
        with pytest.raises(RuntimeError, match="multipart"):
            store.delete_chunks("multipart-crash")
        assert client.objects == before
        assert len(client.pending) == 1
        receipts = None
        if phase == "create":
            with pytest.raises(RuntimeError, match="operator-selected receipt"):
                store.reconcile_multipart(
                    "multipart-crash", expected_fingerprint=state["fingerprint"]
                )
            receipts = {key: upload_id for upload_id, (key, _) in client.pending.items()}
        recovered = store.reconcile_multipart(
            "multipart-crash", expected_fingerprint=state["fingerprint"], receipts=receipts
        )
        assert recovered["multipart_uploads"] == recovered["unresolved_upload_intents"] == 0
        assert recovered["aborted_uploads"] == 1 and not client.pending
        with pytest.raises(RuntimeError, match="changed"):
            store.reconcile_multipart(
                "multipart-crash", expected_fingerprint=state["fingerprint"], receipts=receipts
            )
        resumed = store.resume_session(identity)
        with store.create_writer(resumed) as writer:
            writer.write({"validator_name": "after-recovery", "success": True})
        assert [row.validator_name for row in store.iter_results("multipart-crash")] == [
            "after-recovery"
        ]
    finally:
        store.close()


@pytest.mark.parametrize(
    "fault",
    [
        "assert_receipt",
        "after_abort_publish",
        "after_abort_receipt",
        "second_abort",
        "final_receipt",
    ],
)
def test_s3_multipart_partial_cleanup_keeps_native_journal_until_fresh_inspection(
    tmp_path, monkeypatch, fault
):
    from tests.synthetic_s3 import PersistentMultipartS3
    from truthound.stores.streaming.s3 import StreamingS3Store, _publish_manifest
    from truthound.stores.streaming.s3_multipart import FIELD

    client = PersistentMultipartS3(tmp_path / "multipart.json")
    store = StreamingS3Store("synthetic", flush_interval_seconds=0)
    store._s3_client = client
    session = store.create_session("partial-multipart", "synthetic")
    entries = {}
    for index in range(2):
        key = f"{store.config.prefix.rstrip('/')}/{session.run_id}/part-{index}.json"
        receipt = client.create_multipart_upload(Key=key)["UploadId"]
        entries[str(index)] = {"key": key, "upload_id": receipt}
    session._multipart_intents = entries
    _publish_manifest(client, store.config, session, store._guard_io)
    initial = store.recovery_status(session.run_id)
    native_put, native_abort = client.put_object, client.abort_multipart_upload
    writes, aborts = [], []

    def failed_put(**arguments):
        writes.append(arguments["Key"])
        if (
            fault == "assert_receipt"
            and len(writes) == 1
            or fault == "after_abort_publish"
            and len(writes) == 2
        ):
            raise OSError("synthetic manifest write failure")
        response = native_put(**arguments)
        if (
            fault == "after_abort_receipt"
            and len(writes) == 2
            or fault == "final_receipt"
            and len(writes) == 3
        ):
            return {}
        return response

    def failed_abort(**arguments):
        aborts.append(arguments["UploadId"])
        if fault == "second_abort" and len(aborts) == 2:
            raise OSError("synthetic abort failure")
        return native_abort(**arguments)

    monkeypatch.setattr(client, "put_object", failed_put)
    monkeypatch.setattr(client, "abort_multipart_upload", failed_abort)
    try:
        with pytest.raises((OSError, RuntimeError)):
            store.reconcile_multipart(session.run_id, expected_fingerprint=initial["fingerprint"])
        expected_pending = {
            "assert_receipt": 2,
            "after_abort_publish": 1,
            "after_abort_receipt": 1,
            "second_abort": 1,
            "final_receipt": 0,
        }[fault]
        assert len(client.pending) == expected_pending
        if fault == "assert_receipt":
            assert aborts == []
        fresh = store.recovery_status(session.run_id)
        expected_journal = (
            2 if fault in {"assert_receipt", "after_abort_publish"} else expected_pending
        )
        assert fresh["multipart_uploads"] == expected_journal
        if expected_journal:
            # The original owner's receipt can already be stale after partial
            # publication; a fresh owner must still reject the durable journal.
            with pytest.raises(RuntimeError):
                store.resume_session(session.session_id)
            reopened = StreamingS3Store("synthetic", flush_interval_seconds=0)
            reopened._s3_client = client
            try:
                with pytest.raises(RuntimeError, match="multipart"):
                    reopened.resume_session(session.session_id)
            finally:
                reopened.close()
        monkeypatch.setattr(client, "put_object", native_put)
        monkeypatch.setattr(client, "abort_multipart_upload", native_abort)
        recovered = store.reconcile_multipart(
            session.run_id, expected_fingerprint=fresh["fingerprint"]
        )
        assert recovered["multipart_uploads"] == 0
        assert recovered["resolved_uploads"] == expected_journal
        assert recovered["aborted_uploads"] == expected_pending
        assert not client.pending
        manifest_key = f"{store.config.prefix.rstrip('/')}/{session.run_id}/_manifest.json"
        assert FIELD not in json.loads(client.objects[manifest_key])
    finally:
        store.close()
