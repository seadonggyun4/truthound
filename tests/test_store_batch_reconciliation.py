"""Retained native batches need explicit, pinned reconciliation after actual crashes."""

import json
import subprocess
import sys
from datetime import datetime

import pytest

from truthound.stores.backends.concurrent_filesystem import ConcurrentFileSystemStore
from truthound.stores.base import StoreReadError, StoreWriteError
from truthound.stores.results import ResultStatus, ValidationResult

pytestmark = pytest.mark.fault


def retained(tmp_path, monkeypatch):
    store = ConcurrentFileSystemStore(base_path=tmp_path)
    store.save(ValidationResult("existing", datetime(2026, 1, 1), "local", ResultStatus.SUCCESS))
    original = store._get_file_path("existing").read_bytes()
    batch = store.batch()
    batch.save(ValidationResult("existing", datetime(2026, 1, 1), "local", ResultStatus.FAILURE))
    batch.save(ValidationResult("new", datetime(2026, 1, 1), "local", ResultStatus.FAILURE))

    def fail():
        raise OSError("local-index-failure")

    monkeypatch.setattr(batch._index_txn, "commit", fail)
    with pytest.raises(StoreWriteError):
        batch.commit()
    return store, original


@pytest.mark.parametrize("resolution", ["rollback", "keep_current"])
def test_native_explicit_reconciliation_pins_files_and_survives_reopen(
    tmp_path, monkeypatch, resolution
):
    store, original = retained(tmp_path, monkeypatch)
    descriptor = store.list_batch_recovery()[0]
    assert descriptor["state"] == "index_attempted" and descriptor["file_count"] == 2
    result = store.reconcile_batch(
        descriptor["recovery_id"],
        resolution=resolution,
        expected_fingerprint=descriptor["fingerprint"],
    )
    assert result["resolved"] and not result["quarantined"]
    reopened = ConcurrentFileSystemStore(base_path=tmp_path)
    try:
        if resolution == "rollback":
            assert reopened.list_ids() == ["existing"]
            assert reopened.get("existing").status == ResultStatus.SUCCESS
            assert reopened._get_file_path("existing").read_bytes() == original
        else:
            assert set(reopened.list_ids()) == {"existing", "new"}
            assert reopened.get("existing").status == ResultStatus.FAILURE
        assert reopened.list_batch_recovery() == []
    finally:
        reopened.close()
        store.close()


def test_native_stale_recovery_snapshot_rejects_before_overwriting_current_data(
    tmp_path, monkeypatch
):
    store, _ = retained(tmp_path, monkeypatch)
    descriptor = store.list_batch_recovery()[0]
    path = store._get_file_path("existing")
    path.write_bytes(path.read_bytes() + b" ")
    current = path.read_bytes()
    with pytest.raises(StoreWriteError, match="changed"):
        store.reconcile_batch(
            descriptor["recovery_id"],
            resolution="rollback",
            expected_fingerprint=descriptor["fingerprint"],
        )
    assert path.read_bytes() == current
    with pytest.raises(StoreReadError):
        store.list_ids()


def test_native_corrupt_backup_and_traversing_manifest_cannot_be_applied(tmp_path, monkeypatch):
    store, _ = retained(tmp_path, monkeypatch)
    descriptor = store.list_batch_recovery()[0]
    recovery = store.config.get_full_path() / descriptor["recovery_id"]
    manifest_path = recovery / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    backup = recovery / next(row["backup"] for row in manifest["files"] if row["backup"])
    backup.write_bytes(b"corrupt-backup")
    with pytest.raises(StoreReadError):
        store.list_batch_recovery()
    manifest["files"][0]["name"] = "../outside.json"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(StoreReadError):
        store.reconcile_batch(
            descriptor["recovery_id"],
            resolution="rollback",
            expected_fingerprint=descriptor["fingerprint"],
        )
    assert not (store.config.get_full_path().parent / "outside.json").exists()


@pytest.mark.parametrize("point", ["file", "index", "receipt"])
def test_actual_process_crash_requires_explicit_rollback_and_preserves_originals(tmp_path, point):
    program = """
import os,sys
from datetime import datetime
from truthound.stores.backends.concurrent_filesystem import ConcurrentFileSystemStore
from truthound.stores.results import ValidationResult,ResultStatus
store=ConcurrentFileSystemStore(base_path=sys.argv[1])
def record(name,status):return ValidationResult(name,datetime(2026,1,1),'local',status)
for name in ('first','second'):store.save(record(name,ResultStatus.SUCCESS))
batch=store.batch()
for name in ('first','second'):batch.save(record(name,ResultStatus.FAILURE))
if sys.argv[2]=='file':
 native=store._do_save
 def interrupted(item,txn=None):
  result=native(item,txn)
  if item.id=='first':os._exit(71)
  return result
 store._do_save=interrupted
elif sys.argv[2]=='index':
 native=batch._index_txn.commit
 def interrupted():
  native();os._exit(72)
 batch._index_txn.commit=interrupted
else:
 import truthound.stores.concurrency.batch_recovery as recovery
 def interrupted(*args):os._exit(76)
 recovery.os.rename=interrupted
batch.commit()
"""
    store = ConcurrentFileSystemStore(base_path=tmp_path)
    store.initialize()  # An already initialized owner must also detect retained state.
    completed = subprocess.run(
        [sys.executable, "-I", "-c", program, str(tmp_path), point], capture_output=True, timeout=15
    )
    assert completed.returncode == {"file": 71, "index": 72, "receipt": 76}[point], completed.stderr
    with pytest.raises(StoreReadError):
        store.initialize()
    descriptor = store.list_batch_recovery()[0]
    outcome = store.reconcile_batch(
        descriptor["recovery_id"],
        resolution="rollback",
        expected_fingerprint=descriptor["fingerprint"],
    )
    assert outcome["resolved"]
    try:
        assert set(store.list_ids()) == {"first", "second"}
        assert all(store.get(name).status == ResultStatus.SUCCESS for name in ("first", "second"))
    finally:
        store.close()


def test_preparing_crash_can_be_discarded_only_after_live_owner_releases(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    from truthound.stores.backends.concurrent_filesystem import ConcurrencyConfig
    from truthound.stores.concurrency.manager import LockTimeout

    program = """
import os,sys
from datetime import datetime
from truthound.stores.backends.concurrent_filesystem import ConcurrentFileSystemStore
from truthound.stores.results import ValidationResult,ResultStatus
store=ConcurrentFileSystemStore(base_path=sys.argv[1])
store.save(ValidationResult('original',datetime(2026,1,1),'local',ResultStatus.SUCCESS))
batch=store.batch();batch.save(ValidationResult('original',datetime(2026,1,1),'local',ResultStatus.FAILURE))
native=batch._manifest
def preparing(directory,payload):
 native(directory,payload)
 if payload['state']=='preparing':
  print('preparing',flush=True);sys.stdin.readline();os._exit(74)
batch._manifest=preparing;batch.commit()
"""
    owner = subprocess.Popen(
        [sys.executable, "-I", "-c", program, str(tmp_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(owner.stdout.readline).result(timeout=10).strip() == "preparing"
        store = ConcurrentFileSystemStore(
            base_path=tmp_path, concurrency=ConcurrencyConfig(lock_timeout=0.1)
        )
        descriptor = store.list_batch_recovery()[0]
        assert descriptor["state"] == "preparing"
        with pytest.raises(LockTimeout):
            store.reconcile_batch(
                descriptor["recovery_id"],
                resolution="rollback",
                expected_fingerprint=descriptor["fingerprint"],
            )
        assert owner.poll() is None
        owner.stdin.write("stop\n")
        owner.stdin.flush()
        assert owner.wait(timeout=5) == 74
        result = store.reconcile_batch(
            descriptor["recovery_id"],
            resolution="rollback",
            expected_fingerprint=descriptor["fingerprint"],
        )
        assert result["resolved"] and store.get("original").status == ResultStatus.SUCCESS
        store.close()
    finally:
        if owner.poll() is None:
            owner.kill()
        owner.wait(timeout=5)
        for stream in (owner.stdin, owner.stdout, owner.stderr):
            stream.close()


def test_actual_crash_during_backup_cleanup_leaves_healthy_data_and_explicit_cleanup(tmp_path):
    program = """
import os,sys,shutil
from pathlib import Path
from datetime import datetime
from truthound.stores.backends.concurrent_filesystem import ConcurrentFileSystemStore
from truthound.stores.results import ValidationResult,ResultStatus
store=ConcurrentFileSystemStore(base_path=sys.argv[1]);batch=store.batch()
batch.save(ValidationResult('committed',datetime(2026,1,1),'local',ResultStatus.FAILURE))
native=shutil.rmtree
def interrupted(path,*args,**kwargs):
 path=Path(path)
 if path.name.startswith('.batch-'):
  for item in path.iterdir():
   if item.is_file():item.unlink();break
  os._exit(75)
 return native(path,*args,**kwargs)
shutil.rmtree=interrupted;batch.commit()
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", program, str(tmp_path)], capture_output=True, timeout=15
    )
    assert completed.returncode == 75, completed.stderr
    store = ConcurrentFileSystemStore(base_path=tmp_path)
    store.initialize()
    assert store.get("committed").status == ResultStatus.FAILURE
    descriptor = store.list_batch_recovery()[0]
    assert descriptor["state"] == "cleanup_pending"
    assert store.cleanup_batch_recovery(
        descriptor["recovery_id"], expected_fingerprint=descriptor["fingerprint"]
    )["cleaned"]
    assert store.list_batch_recovery() == []
    assert store.get("committed").status == ResultStatus.FAILURE
    store.close()
