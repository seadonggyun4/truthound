"""Actual cross-platform providers must bound contention without fake runtime passes."""

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress

import pytest

from truthound.stores.concurrency import locks

pytestmark = pytest.mark.fault


def test_actual_filelock_acquired_on_owner_thread_is_released_by_async_executor(tmp_path):
    native = locks.FileLockStrategy()
    path = tmp_path / "async-owner"
    handle = native.acquire(path, locks.LockMode.EXCLUSIVE, timeout=0)
    entry = native._locks[str(path)]
    raw = entry[1] if isinstance(entry, tuple) else entry
    probe, next_handle = locks.FileLockStrategy(), None
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(native.release, handle).result(timeout=2)
        next_handle = probe.acquire(path, locks.LockMode.EXCLUSIVE, timeout=0, blocking=False)
    finally:
        if next_handle is not None:
            probe.release(next_handle)
        # Keep the exact SDK object to release a broken implementation's
        # thread-local owner context without touching any unrelated descriptor.
        raw.release(force=True)


@pytest.mark.parametrize("provider", ["filelock", "portalocker"])
def test_actual_provider_stale_handle_cannot_unlock_a_new_owner(tmp_path, provider):
    native = {"filelock": locks.FileLockStrategy, "portalocker": locks.PortalockerStrategy}[provider]()
    path = tmp_path / "reused-handle"
    previous = native.acquire(path, locks.LockMode.EXCLUSIVE, timeout=0)
    native.release(previous)
    current = native.acquire(path, locks.LockMode.EXCLUSIVE, timeout=0)
    try:
        with pytest.raises(ValueError):
            native.release(previous)
        assert native.is_locked(path)
    finally:
        with suppress(ValueError):
            native.release(current)


def test_actual_filelock_unlock_failure_retains_owned_resource_for_cleanup_retry(tmp_path, monkeypatch):
    native = locks.FileLockStrategy()
    path = tmp_path / "unlock-failure"
    handle = native.acquire(path, locks.LockMode.EXCLUSIVE, timeout=0)
    raw = native._locks[str(path)][1]

    def failed_unlock():
        raise OSError("synthetic-private-unlock")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(raw, "_release", failed_unlock)
            with pytest.raises(OSError):
                native.release(handle)
        assert native.is_locked(path)
        native.release(handle)
        assert not native.is_locked(path)
    finally:
        if str(path) in native._locks:
            native.release(handle)


@pytest.mark.parametrize("provider", ["filelock", "portalocker"])
def test_actual_provider_nonblocking_contention_is_bounded(tmp_path, provider):
    program = """
import sys
from pathlib import Path
from truthound.stores.concurrency.locks import FileLockStrategy,PortalockerStrategy,LockMode
from truthound.stores.concurrency.manager import LockTimeout
native={'filelock':FileLockStrategy,'portalocker':PortalockerStrategy}[sys.argv[1]]()
path=Path(sys.argv[2])
if sys.argv[3]=='owner':
 handle=native.acquire(path,LockMode.EXCLUSIVE)
 print('ready',flush=True);sys.stdin.readline();native.release(handle)
else:
 try:native.acquire(path,LockMode.EXCLUSIVE,blocking=False)
 except LockTimeout:print('contended',flush=True)
 else:raise AssertionError('Native provider admitted a second owner')
"""
    args = [sys.executable, "-I", "-c", program, provider, str(tmp_path / "owned")]
    owner = subprocess.Popen(
        [*args, "owner"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(owner.stdout.readline).result(timeout=10).strip() == "ready"
        try:
            contender = subprocess.run(
                [*args, "contender"], capture_output=True, text=True, timeout=2
            )
        except subprocess.TimeoutExpired:
            pytest.fail("Native nonblocking acquisition blocked")
        assert contender.returncode == 0 and contender.stdout.strip() == "contended", (
            contender.stderr
        )
    finally:
        owner.stdin.close()
        try:
            owner.wait(timeout=5)
        finally:
            if owner.poll() is None:
                owner.kill()
            owner.wait(timeout=5)
            owner.stdout.close()
            owner.stderr.close()


def test_actual_portalocker_open_failure_does_not_retain_local_owner(tmp_path, monkeypatch):
    native = locks.PortalockerStrategy()
    path = tmp_path / "denied-open"

    def denied(*args, **kwargs):
        raise PermissionError("local-open-denied")

    with monkeypatch.context() as context:
        context.setattr(locks, "open", denied, raising=False)
        with pytest.raises(PermissionError):
            native.acquire(path, locks.LockMode.EXCLUSIVE)
    state = locks._get_process_local_state(path)
    try:
        assert not state.writer
    finally:
        if state.writer:
            locks._release_process_local_lock(path, locks.LockMode.EXCLUSIVE)


def test_actual_portalocker_shared_readers_keep_distinct_owned_descriptors(tmp_path):
    import os

    native = locks.PortalockerStrategy()
    path = tmp_path / "shared-readers"
    first = native.acquire(path, locks.LockMode.SHARED)
    second = native.acquire(path, locks.LockMode.SHARED)
    try:
        os.fstat(first.fd)
        os.fstat(second.fd)
        native.release(first)
        os.fstat(second.fd)
        native.release(second)
        assert locks._get_process_local_state(path).readers == 0
    finally:
        for stream in native._locks.values():
            if not stream.closed:
                stream.close()
        native._locks.clear()
        state = locks._get_process_local_state(path)
        while state.readers:
            locks._release_process_local_lock(path, locks.LockMode.SHARED)


def test_actual_default_provider_keeps_lock_identity_across_waiting_and_new_owners(tmp_path):
    program = """
import sys
from pathlib import Path
from truthound.stores.concurrency.locks import get_default_lock_strategy,LockMode
from truthound.stores.concurrency.manager import LockTimeout
native=get_default_lock_strategy();path=Path(sys.argv[1]);mode=sys.argv[2]
if mode=='waiter':
 if hasattr(native,'_fcntl'):
  original=native._fcntl.flock;signalled=False
  def tracked(fd,flags):
   global signalled
   try:return original(fd,flags)
   except BlockingIOError:
    if not signalled:print('waiting',flush=True);signalled=True
    raise
  native._fcntl.flock=tracked
 else:print('waiting',flush=True)
if mode=='try':
 try:handle=native.acquire(path,LockMode.EXCLUSIVE,blocking=False)
 except LockTimeout:print('blocked',flush=True)
 else:
  native.release(handle);raise AssertionError('Concurrent owners admitted after handoff')
else:
 handle=native.acquire(path,LockMode.EXCLUSIVE,timeout=5)
 print('owned',flush=True);sys.stdin.readline();native.release(handle)
"""
    args = [sys.executable, "-I", "-c", program, str(tmp_path / "handoff")]
    first = subprocess.Popen(
        [*args, "owner"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    waiter = None

    def line(owner):
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(owner.stdout.readline).result(timeout=10).strip()

    try:
        assert line(first) == "owned"
        waiter = subprocess.Popen(
            [*args, "waiter"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert line(waiter) == "waiting"
        first.stdin.write("release\n")
        first.stdin.flush()
        assert first.wait(timeout=5) == 0
        assert line(waiter) == "owned"
        contender = subprocess.run([*args, "try"], capture_output=True, text=True, timeout=2)
        assert contender.returncode == 0 and contender.stdout.strip() == "blocked", contender.stderr
    finally:
        for owner in (first, waiter):
            if owner is None:
                continue
            owner.stdin.close()
            try:
                owner.wait(timeout=5)
            finally:
                if owner.poll() is None:
                    owner.kill()
                owner.wait(timeout=5)
                owner.stdout.close()
                owner.stderr.close()
