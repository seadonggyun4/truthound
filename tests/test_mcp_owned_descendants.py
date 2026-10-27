"""A dead driver must not erase ownership of its still-running descendants."""

import asyncio
import os
import subprocess
import sys
import threading
import time
from contextlib import suppress

import psutil
import pytest

from truthound.mcp.model_process import OwnedModelProcess

pytestmark = pytest.mark.fault


def test_pipe_reader_preserves_frames_and_enforces_partial_frame_limit():
    from truthound.mcp.native_stdio import CancellablePipeReader

    read_fd, write_fd = os.pipe()
    with os.fdopen(read_fd, "rb") as stream:
        reader = CancellablePipeReader(stream)
        done = threading.Event()
        replies, errors = [], []

        def receive_frames():
            try:
                for limit in (10, 10, 16, 16, 16):
                    replies.append(reader.readline(limit))
            except BaseException as error:
                errors.append(error)
            finally:
                done.set()

        os.write(write_fd, b'first\nsecond\n' + b'x' * 17)
        os.close(write_fd)
        thread = threading.Thread(target=receive_frames, daemon=True)
        thread.start()
        try:
            assert done.wait(2), "Available bytes must be read even on a byte-mode Windows pipe"
            assert not errors
            assert replies == [b'first\n', b'second\n', b'x' * 16, b'x', b'']
        finally:
            reader.stop()
            thread.join(timeout=2)


def test_worker_timeout_finishes_while_another_writer_keeps_pipe_open(monkeypatch):
    original = subprocess.Popen
    read_fd, write_fd = os.pipe()

    def start_owned(command, **kwargs):
        kwargs["stdout"] = write_fd
        process = original([command[0], "-I", "-c", "import time; time.sleep(600)"], **kwargs)
        process.stdout = os.fdopen(read_fd, "rb")
        return process

    monkeypatch.setattr("truthound.mcp.model_process.subprocess.Popen", start_owned)
    worker = OwnedModelProcess(30)
    done = threading.Event()
    errors = []

    def call():
        try:
            asyncio.run(worker.call({}, timeout=0.1))
        except TimeoutError:
            errors.append("timeout")
        finally:
            done.set()

    thread = threading.Thread(target=call, daemon=True)
    thread.start()
    try:
        assert done.wait(2), "A retained writer must not make timeout cleanup unbounded"
        assert errors == ["timeout"] and worker.process.poll() is not None
    finally:
        os.close(write_fd)
        thread.join(timeout=5)
        worker.close()


def test_native_worker_wire_excludes_native_child_and_atexit_output():
    program = '''
import atexit, os, subprocess, sys
from truthound.mcp.native_stdio import native_worker_wire
sys.stdout.reconfigure(encoding="cp1252")
sys.stderr.reconfigure(encoding="cp1252")
# Windows standard-stream duplication returns an inheritable descriptor.
original_dup = os.dup
def standard_stream_dup(fd):
    descriptor = original_dup(fd)
    if fd in (0, 1, 2):
        os.set_inheritable(descriptor, True)
    return descriptor
os.dup = standard_stream_dup
wire = native_worker_wire()
assert not os.get_inheritable(wire.fileno())
print("synthetic-private-unicode-\u2713")
print("synthetic-private-unicode-\u03bb", file=sys.stderr)
os.write(1, b"synthetic-private-native-stdout\\n")
os.write(2, b"synthetic-private-native-stderr\\n")
subprocess.run([sys.executable, "-I", "-c",
                "import os; os.write(1,b'synthetic-private-child\\\\n')"], check=True)
atexit.register(os.write, 1, b"synthetic-private-native-exit\\n")
if os.name == "nt":
    import _winapi, msvcrt
    assert _winapi.GetStdHandle(-11) == msvcrt.get_osfhandle(1)
    assert _winapi.GetStdHandle(-12) == msvcrt.get_osfhandle(2)
wire.write('{"ok":true}\\n')
wire.close()
'''
    result = subprocess.run([sys.executable, "-I", "-c", program],
                            capture_output=True, check=True, timeout=15)
    assert result.stdout == b'{"ok":true}\n' and result.stderr == b""


def test_dead_worker_reaps_recorded_descendant_and_leaves_sibling_alive(monkeypatch):
    original = subprocess.Popen
    sibling = original([sys.executable, "-I", "-c", "import time;time.sleep(600)"])
    program = (
        "import json,subprocess,sys,time;sys.stdin.readline();"
        "child=subprocess.Popen([sys.executable,'-I','-c','import time;time.sleep(600)']);"
        "print(json.dumps(dict(ok=True,child=child.pid)),flush=True);sys.stdin.read()"
    )

    def start_owned(command, **kwargs):
        return original([command[0], "-I", "-c", program], **kwargs)

    monkeypatch.setattr("truthound.mcp.model_process.subprocess.Popen", start_owned)
    monkeypatch.setattr(OwnedModelProcess, "validate_reply", lambda self, reply: reply)
    worker = OwnedModelProcess(30)
    child, created = None, None
    try:
        result = asyncio.run(worker.call({}, timeout=10))
        child = psutil.Process(result["child"])
        created = child.create_time()
        deadline = time.monotonic() + 2
        while child.pid not in worker._descendants and time.monotonic() < deadline:
            time.sleep(0.01)
        assert child.pid in worker._descendants
        worker.process.kill()
        worker.process.wait(timeout=5)
        assert child.is_running()
        worker.close()
        with suppress(psutil.NoSuchProcess):
            assert child.create_time() != created or child.status() == psutil.STATUS_ZOMBIE
        assert sibling.poll() is None
    finally:
        worker.close()
        if child is not None:
            try:
                if child.create_time() == created and child.status() != psutil.STATUS_ZOMBIE:
                    child.kill()
                    child.wait(timeout=5)
            except psutil.NoSuchProcess:
                pass
        sibling.kill()
        sibling.wait(timeout=5)
