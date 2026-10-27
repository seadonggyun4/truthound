"""Resource fault contracts execute the native fcntl strategy without a platform dependency."""

import threading
from types import SimpleNamespace

import pytest

import truthound.stores.concurrency.locks as locks
from truthound.stores.concurrency.manager import LockTimeout

pytestmark = pytest.mark.fault


def strategy(flock):
    native = locks.FcntlLockStrategy.__new__(locks.FcntlLockStrategy)
    native._fcntl = SimpleNamespace(LOCK_SH=1, LOCK_EX=2, LOCK_NB=4, LOCK_UN=8, flock=flock)
    native._lock = threading.RLock()
    native._locks = {}
    return native


def test_reused_descriptor_rejects_old_handle_without_closing_current_owner(tmp_path, monkeypatch):
    native = strategy(lambda *args: None)
    path = tmp_path / "reused-descriptor"
    target = native._get_lock_path(path)
    sentinel, closed = -12345, []
    real_open, real_close = locks.os.open, locks.os.close

    def open_owned(value, *args, **kwargs):
        return sentinel if str(value) == str(target) else real_open(value, *args, **kwargs)

    def close_owned(value):
        if value == sentinel:
            closed.append(value)
        else:
            real_close(value)

    with monkeypatch.context() as patch:
        patch.setattr(locks.os, "open", open_owned)
        patch.setattr(locks.os, "close", close_owned)
        previous = native.acquire(path, locks.LockMode.EXCLUSIVE, timeout=0)
        native.release(previous)
        current = native.acquire(path, locks.LockMode.EXCLUSIVE, timeout=0)
        try:
            with pytest.raises(ValueError):
                native.release(previous)
            assert closed == [sentinel] and native.is_locked(path)
        finally:
            native.release(current)
    assert closed == [sentinel, sentinel]


def test_native_timeout_preserves_timeout_and_closes_descriptor_once(tmp_path):
    def blocked(*args):
        raise BlockingIOError()

    native = strategy(blocked)
    path = tmp_path / "timed-out"
    with pytest.raises(LockTimeout):
        native.acquire(path, locks.LockMode.EXCLUSIVE, timeout=0)
    assert not locks._get_process_local_state(path).writer


def test_native_open_failure_releases_process_local_ownership(tmp_path, monkeypatch):
    native = strategy(lambda *args: None)
    path = tmp_path / "denied-open"

    def denied(*args, **kwargs):
        raise PermissionError("local-open-denied")

    with monkeypatch.context() as context:
        context.setattr(locks.os, "open", denied)
        with pytest.raises(PermissionError):
            native.acquire(path, locks.LockMode.EXCLUSIVE, timeout=0)
    state = locks._get_process_local_state(path)
    try:
        assert not state.writer
    finally:
        if state.writer:
            locks._release_process_local_lock(path, locks.LockMode.EXCLUSIVE)
    handle = native.acquire(path, locks.LockMode.EXCLUSIVE, timeout=0)
    native.release(handle)
