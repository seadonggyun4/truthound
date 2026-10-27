"""Native key persistence must not reinterpret corruption as absent key material."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from truthound.stores.encryption.base import EncryptionError
from truthound.stores.encryption.keys import FileKeyStore, InMemoryKeyStore, KeyManager

pytestmark = pytest.mark.fault


def test_corrupt_key_cannot_be_silently_replaced_by_get_or_create(tmp_path):
    store = FileKeyStore(tmp_path / "keys", master_key=bytes(range(32)))
    manager = KeyManager(store)
    manager.create_key("synthetic-fixed")
    path = store._get_key_path("synthetic-fixed")
    path.write_bytes(b"synthetic-corrupt-encrypted-key")
    with pytest.raises(EncryptionError):
        manager.get_or_create_key("synthetic-fixed")
    assert path.read_bytes() == b"synthetic-corrupt-encrypted-key"


def test_failed_key_publish_keeps_old_record_intact(tmp_path, monkeypatch):
    from truthound.stores.encryption import keys

    store = FileKeyStore(tmp_path / "keys", master_key=bytes(range(32)))
    manager = KeyManager(store)
    value = manager.create_key("synthetic-fixed")
    path = store._get_key_path("synthetic-fixed")
    original = path.read_bytes()
    replace = keys.os.replace

    def failed(source, target):
        if Path(target) == path:
            raise OSError("synthetic-private-key-publish")
        return replace(source, target)

    monkeypatch.setattr(keys.os, "replace", failed)
    with pytest.raises((EncryptionError, OSError)):
        store.put(value)
    assert path.read_bytes() == original
    assert not list(path.parent.glob(".truthound-key-*"))


def test_duplicate_key_creation_requires_rotation_instead_of_overwrite():
    manager = KeyManager(InMemoryKeyStore())
    manager.create_key("synthetic-fixed")
    with pytest.raises(EncryptionError):
        manager.create_key("synthetic-fixed")


def test_cleanup_expired_does_not_count_false_delete_receipt():
    class Store(InMemoryKeyStore):
        def delete(self, identity):
            return False

    store = Store()
    manager = KeyManager(store)
    value = manager.create_key("synthetic-expired")
    value.expires_at = datetime.now(UTC) - timedelta(seconds=1)
    with pytest.raises(EncryptionError):
        manager.cleanup_expired()
    assert store.exists("synthetic-expired")


def test_failed_key_unlink_does_not_destroy_previous_record(tmp_path, monkeypatch):
    store = FileKeyStore(tmp_path / "keys", master_key=bytes(range(32)))
    KeyManager(store).create_key("synthetic-fixed")
    path = store._get_key_path("synthetic-fixed")
    original = path.read_bytes()
    unlink = Path.unlink

    def failed(target, *args, **kwargs):
        if target == path:
            raise OSError("synthetic-private-key-unlink")
        return unlink(target, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", failed)
    with pytest.raises(OSError):
        store.delete("synthetic-fixed")
    assert path.read_bytes() == original


def test_key_repr_does_not_expose_key_material():
    from truthound.stores.encryption.base import EncryptionAlgorithm, EncryptionKey

    value = EncryptionKey(
        "synthetic", b"synthetic-sensitive-key-material", EncryptionAlgorithm.AES_256_GCM
    )
    assert "synthetic-sensitive-key-material" not in repr(value)


def test_rotation_audit_revocation_blocks_following_store_write():
    revoked = False

    def audit(event, details):
        nonlocal revoked
        if event == "key_accessed":
            revoked = True

    def guard():
        if revoked:
            raise PermissionError("synthetic-private-key-revoked")

    store = InMemoryKeyStore()
    manager = KeyManager(store, audit_hook=audit)
    manager.create_key("synthetic-fixed")
    manager.set_execution_guard(guard)
    with pytest.raises(PermissionError):
        manager.rotate_key("synthetic-fixed", archive_old=False)
    assert store.get("synthetic-fixed").version == 1


def test_independent_processes_cannot_create_the_same_native_file_key(tmp_path):
    import os
    import subprocess
    import sys
    from concurrent.futures import ThreadPoolExecutor

    code = """
import sys
from truthound.stores.encryption.base import EncryptionError
from truthound.stores.encryption.keys import FileKeyStore, KeyManager
manager = KeyManager(FileKeyStore(sys.argv[1], master_key=bytes(range(32))))
print('ready', flush=True)
sys.stdin.readline()
try:
    manager.create_key('synthetic-shared')
except EncryptionError:
    print('exists', flush=True)
else:
    print('created', flush=True)
"""
    root = tmp_path / "shared-keys"
    processes = []
    readers = ThreadPoolExecutor(max_workers=2)
    try:
        for _ in range(2):
            processes.append(subprocess.Popen(
                [sys.executable, "-c", code, str(root)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True,
                env={**os.environ, "POLARS_MAX_THREADS": "1", "OMP_NUM_THREADS": "1"},
            ))
        ready = [readers.submit(process.stdout.readline) for process in processes]
        assert [result.result(timeout=15).strip() for result in ready] == ["ready", "ready"]
        for process in processes:
            process.stdin.write("go\n")
            process.stdin.flush()
        results = [process.communicate(timeout=15)[0].strip() for process in processes]
        assert all(process.returncode == 0 for process in processes)
        assert sorted(results) == ["created", "exists"]
        assert FileKeyStore(root, master_key=bytes(range(32))).get("synthetic-shared").version == 1
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        readers.shutdown(wait=True, cancel_futures=True)
