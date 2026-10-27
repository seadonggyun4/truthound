"""Native filesystem batches roll back before index commit and retain uncertain effects."""

from datetime import datetime

import pytest

from truthound.stores.backends.concurrent_filesystem import ConcurrentFileSystemStore
from truthound.stores.base import StoreWriteError
from truthound.stores.results import ResultStatus, ValidationResult

pytestmark = pytest.mark.fault


def result(identity, status=ResultStatus.SUCCESS):
    return ValidationResult(identity, datetime(2026, 1, 1), "local-asset", status)


def test_documented_native_batch_context_rolls_back_uncommitted_buffers(tmp_path):
    store = ConcurrentFileSystemStore(base_path=tmp_path)
    with store.batch() as batch:
        batch.save(result("pending"))
    assert store.list_ids() == []
    with store.batch() as batch:
        batch.save(result("committed"))
        assert batch.commit() == 1
    assert store.get("committed").status == ResultStatus.SUCCESS


def test_native_failed_batch_restores_deleted_and_overwritten_files_and_stops_later_work(
    tmp_path, monkeypatch
):
    store = ConcurrentFileSystemStore(base_path=tmp_path)
    store.save(result("delete-me"))
    store.save(result("overwrite-me"))
    original = store._do_save

    def fail(item, txn=None):
        if item.id == "fail":
            raise StoreWriteError("synthetic-local-write-failure")
        return original(item, txn)

    monkeypatch.setattr(store, "_do_save", fail)
    batch = store.batch()
    batch.delete("delete-me")
    batch.save(result("overwrite-me", ResultStatus.FAILURE))
    batch.save(result("fail"))
    batch.save(result("must-not-run"))
    with pytest.raises(StoreWriteError):
        batch.commit()
    reopened = ConcurrentFileSystemStore(base_path=tmp_path)
    assert set(reopened.list_ids()) == {"delete-me", "overwrite-me"}
    assert reopened.get("overwrite-me").status == ResultStatus.SUCCESS
    assert reopened.get("delete-me").status == ResultStatus.SUCCESS
    assert not reopened.exists("must-not-run")


def test_native_uncertain_index_commit_retains_manifest_and_original_bytes(tmp_path, monkeypatch):
    import hashlib
    import json

    store = ConcurrentFileSystemStore(base_path=tmp_path)
    store.save(result("original"))
    original = store._get_file_path("original").read_bytes()
    batch = store.batch()
    batch.save(result("original", ResultStatus.FAILURE))

    def fail():
        raise OSError("synthetic-private-index-failure")

    monkeypatch.setattr(batch._index_txn, "commit", fail)
    with pytest.raises(StoreWriteError, match="uncertain"):
        batch.commit()
    recovery = list(store.config.get_full_path().glob(".batch-recovery-*"))
    assert len(recovery) == 1
    manifest = json.loads((recovery[0] / "manifest.json").read_text())
    assert manifest["state"] == "index_attempted"
    entry = manifest["files"][0]
    assert entry["name"] == store._get_file_path("original").name
    assert entry["sha256"] == hashlib.sha256(original).hexdigest()
    assert (recovery[0] / entry["backup"]).read_bytes() == original


def test_native_batch_revocation_before_second_write_restores_first(tmp_path, monkeypatch):
    store = ConcurrentFileSystemStore(base_path=tmp_path)
    store.initialize()
    revoked = False
    native = store._do_save

    def write(item, txn):
        nonlocal revoked
        answer = native(item, txn)
        revoked = True
        return answer

    def guard():
        if revoked:
            raise PermissionError("local-revocation")

    batch = store.batch()
    batch.save(result("first"))
    batch.save(result("second"))
    store.set_io_guard(guard)
    monkeypatch.setattr(store, "_do_save", write)
    with pytest.raises(PermissionError):
        batch.commit()
    reopened = ConcurrentFileSystemStore(base_path=tmp_path)
    assert reopened.list_ids() == []
    assert not reopened.exists("first") and not reopened.exists("second")


def test_native_uncertain_batch_quarantines_instance_and_reopen(tmp_path, monkeypatch):
    from truthound.stores.base import StoreReadError
    store = ConcurrentFileSystemStore(base_path=tmp_path)
    batch = store.batch()
    batch.save(result('pending'))
    def fail():
        raise OSError('local-index-failure')
    monkeypatch.setattr(batch._index_txn, 'commit', fail)
    with pytest.raises(StoreWriteError):
        batch.commit()
    with pytest.raises(StoreReadError, match='reconciliation'):
        store.list_ids()
    reopened = ConcurrentFileSystemStore(base_path=tmp_path)
    with pytest.raises(StoreReadError, match='reconciliation'):
        reopened.initialize()
