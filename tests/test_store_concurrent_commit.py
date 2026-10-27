"""Native concurrent indexes must retain other owners and acknowledge disk failures."""

from types import SimpleNamespace

import pytest

from truthound.stores.concurrency.index import ConcurrentIndex

pytestmark = pytest.mark.fault


def test_two_initialized_native_index_owners_preserve_disjoint_commits(tmp_path):
    first, second = ConcurrentIndex(tmp_path), ConcurrentIndex(tmp_path)
    first.initialize()
    second.initialize()
    a, b = first.begin_transaction(), second.begin_transaction()
    a.add("first", {"value": 1})
    b.add("second", {"value": 2})
    a.commit()
    b.commit()
    reopened = ConcurrentIndex(tmp_path)
    assert set(reopened.snapshot().list_ids()) == {"first", "second"}


def test_native_index_concurrent_add_of_same_identity_conflicts(tmp_path):
    first, second = ConcurrentIndex(tmp_path), ConcurrentIndex(tmp_path)
    first.initialize()
    second.initialize()
    a, b = first.begin_transaction(), second.begin_transaction()
    a.add("same", {"value": 1})
    b.add("same", {"value": 2})
    a.commit()
    with pytest.raises(RuntimeError, match="Conflict"):
        b.commit()
    assert ConcurrentIndex(tmp_path).snapshot().get("same").metadata == {"value": 1}


def test_native_index_disk_failure_is_not_committed_or_reused(tmp_path, monkeypatch):
    import truthound.stores.concurrency.index as module

    index = ConcurrentIndex(tmp_path)
    index.initialize()
    txn = index.begin_transaction()
    txn.add("pending", {})
    monkeypatch.setattr(
        module,
        "atomic_write",
        lambda *args, **kwargs: SimpleNamespace(success=False, error="synthetic-private-write"),
    )
    with pytest.raises(OSError):
        txn.commit()
    assert txn.is_active
    with pytest.raises(RuntimeError, match="uncertain"):
        index.begin_transaction()


def test_native_wal_recovery_preserves_preexisting_commits(tmp_path):
    import json

    from truthound.stores.concurrency.index import IndexEntry

    index = ConcurrentIndex(tmp_path)
    index.add("existing", {"value": 1})
    pending = {
        "type": "add",
        "item_id": "pending",
        "entry": IndexEntry("pending", {"value": 2}).to_dict(),
    }
    (tmp_path / "_index.json.wal").write_text(json.dumps(pending) + "\n")
    reopened = ConcurrentIndex(tmp_path)
    assert set(reopened.snapshot().list_ids()) == {"existing", "pending"}


def test_native_corrupt_wal_is_preserved_and_does_not_publish_partial_recovery(tmp_path):
    import json

    from truthound.stores.concurrency.index import IndexEntry

    index = ConcurrentIndex(tmp_path)
    index.add("existing", {})
    pending = {"type": "add", "item_id": "pending", "entry": IndexEntry("pending", {}).to_dict()}
    wal = tmp_path / "_index.json.wal"
    wal.write_text(json.dumps(pending) + "\ninvalid-last-record\n")
    before = (tmp_path / "_index.json").read_bytes()
    with pytest.raises(ValueError):
        ConcurrentIndex(tmp_path).initialize()
    assert wal.exists() and (tmp_path / "_index.json").read_bytes() == before


def test_actual_process_owners_do_not_lose_disjoint_commits(tmp_path):
    import subprocess
    import sys
    from concurrent.futures import ThreadPoolExecutor

    program = """
import sys
from truthound.stores.concurrency.index import ConcurrentIndex
index = ConcurrentIndex(sys.argv[1]); index.initialize()
txn = index.begin_transaction(); txn.add(sys.argv[2], {})
print('ready', flush=True); sys.stdin.readline(); txn.commit()
"""
    owners = [
        subprocess.Popen(
            [sys.executable, "-I", "-c", program, str(tmp_path), identity],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for identity in ("first", "second")
    ]
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            reads = [pool.submit(owner.stdout.readline) for owner in owners]
            assert all(read.result(timeout=10).strip() == "ready" for read in reads)
        for owner in owners:
            owner.stdin.write("commit\n")
            owner.stdin.flush()
        for owner in owners:
            assert owner.wait(timeout=15) == 0, owner.stderr.read()
        assert set(ConcurrentIndex(tmp_path).snapshot().list_ids()) == {"first", "second"}
    finally:
        for owner in owners:
            if owner.poll() is None:
                owner.kill()
            owner.wait(timeout=5)
            for stream in (owner.stdin, owner.stdout, owner.stderr):
                stream.close()
