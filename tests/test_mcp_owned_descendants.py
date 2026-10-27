"""A dead driver must not erase ownership of its still-running descendants."""

import asyncio
import subprocess
import sys
import time
from contextlib import suppress

import psutil
import pytest

from truthound.mcp.model_process import OwnedModelProcess

pytestmark = pytest.mark.fault


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
