"""Native owner EOF terminates its actual owned process group and descendants."""

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path

import psutil
import pytest

pytestmark = pytest.mark.fault


def test_native_process_group_is_reaped_when_owning_pipe_disconnects():
    program = """
import json, subprocess, sys, time
from truthound._process_supervision import watch_resident_parent
watch_resident_parent(sys.stdin.fileno(), own_process_group=True)
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(json.dumps({'pid':child.pid}), flush=True)
time.sleep(60)
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    options = (
        {"start_new_session": True}
        if os.name != "nt"
        else {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    )
    process = subprocess.Popen(
        [sys.executable, "-c", program],
        env=environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        **options,
    )
    child = None
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            line = pool.submit(process.stdout.readline).result(timeout=5)
        assert line, process.stderr.read()
        child = psutil.Process(json.loads(line)["pid"])
        process.stdin.close()
        process.wait(timeout=5)
        _, alive = psutil.wait_procs([child], timeout=5)
        assert not alive or all(p.status() == psutil.STATUS_ZOMBIE for p in alive)
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        if child is not None:
            with suppress(psutil.NoSuchProcess):
                child.kill()
        for stream in (process.stdin, process.stdout, process.stderr):
            stream.close()


def test_owned_worker_is_reaped_when_os_denies_descendant_enumeration(monkeypatch):
    from truthound.mcp.model_process import OwnedModelProcess

    owner = OwnedModelProcess(60)
    process = owner.process

    def denied(*args, **kwargs):
        raise PermissionError("local process enumeration denied")

    monkeypatch.setattr(psutil.Process, "children", denied)
    try:
        owner.close()
        assert owner.closed and process.poll() is not None
        assert process.stdin.closed and process.stdout.closed
        assert not Path(owner.directory.name).exists()
        owner.close()
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
