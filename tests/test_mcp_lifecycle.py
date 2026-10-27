"""Real stdio cancellation/EOF/timeout with actual child PIDs and private copies."""

import json
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import psutil
import pytest

pytest.importorskip("mcp")

pytestmark = pytest.mark.fault
GRACE_SECONDS = 5


@pytest.mark.parametrize("failed_artifact", [None, "wheel.whl", "source.tar.gz"])
def test_artifact_smoke_parallel_isolation_and_failure_barrier(tmp_path, monkeypatch, failed_artifact):
    """All eight stages survive concurrency; either branch failure blocks both launchers."""
    import importlib.util
    from types import SimpleNamespace

    path = Path(__file__).resolve().parents[1] / "verification/mcp_release.py"
    spec = importlib.util.spec_from_file_location("parallel_install_contract", path)
    release = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(release)
    wheel, sdist = tmp_path / "wheel.whl", tmp_path / "source.tar.gz"
    barrier = threading.Barrier(2, timeout=GRACE_SECONDS)
    stages, directories, installs = {}, {}, []
    lock = threading.Lock()

    def install(command, **kwargs):
        with lock:
            installs.append(list(map(str, command)))

    def smoke(command, code, artifact, stage, cwd):
        if stage in {"uv-launcher", "core-uv-launcher"}:
            assert failed_artifact is None
            assert all(not directory.exists() for directory in directories.values())
            assert all(value == ["base-install", "mcp-install", "registry-cli-install"]
                       for key, value in stages.items() if key != "launcher")
            assert list(stages.get("launcher", [])) == (
                [] if stage == "uv-launcher" else ["uv-launcher"]
            )
            stages.setdefault("launcher", []).append(stage)
            return
        with lock:
            stages.setdefault(artifact.name, []).append(stage)
            directories[artifact.name] = Path(cwd)
        assert Path(cwd).is_dir()
        assert Path(command[0]).is_relative_to(Path(cwd) / "venv")
        if stage == "base-install":
            assert code == release.BASE_SMOKE
            barrier.wait()  # Serial execution must fail rather than silently regress.
        elif stage == "mcp-install":
            assert code == release.SMOKE
            if artifact.name == failed_artifact:
                raise RuntimeError("synthetic installed artifact failure")
        else:
            assert '"truthound.cli"' in code and code != release.SMOKE

    monkeypatch.setattr(release, "run", install)
    args = SimpleNamespace(expected_python="3.11")
    if failed_artifact:
        with pytest.raises(RuntimeError, match="synthetic installed artifact failure"):
            release.smoke_artifacts(args, tmp_path, wheel, sdist, "3.2.0", "uv", smoke)
        assert "launcher" not in stages
    else:
        release.smoke_artifacts(args, tmp_path, wheel, sdist, "3.2.0", "uv", smoke)
        assert stages.pop("launcher") == ["uv-launcher", "core-uv-launcher"]
    assert set(stages) == {wheel.name, sdist.name}
    assert len(set(directories.values())) == 2
    assert all(not directory.exists() for directory in directories.values())
    assert len(installs) == 6  # Both independent venv/base/extra installs complete.
    for artifact in (wheel, sdist):
        expected = ["base-install", "mcp-install"]
        if artifact.name != failed_artifact:
            expected.append("registry-cli-install")
        assert stages[artifact.name] == expected


def test_installed_smoke_owns_actual_interpreter_and_preserves_venv(tmp_path):
    """A Windows redirector PID is not the interpreter whose locks we await."""
    import ast
    import importlib.util
    import os

    from mcp import StdioServerParameters

    path = Path(__file__).resolve().parents[1] / "verification/mcp_release.py"
    spec = importlib.util.spec_from_file_location("installed_ownership_contract", path)
    release = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(release)
    function = next(node for node in ast.parse(release.SMOKE).body
                    if isinstance(node, ast.FunctionDef) and node.name == "installed_server_parameters")
    namespace = {"sys": sys, "StdioServerParameters": StdioServerParameters}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<installed-owner>", "exec"), namespace)
    program = ("import os,sys,json; held=open('owned-file','w'); "
               "print(json.dumps([os.getpid(),sys.prefix]),flush=True); sys.stdin.read()")
    params = namespace["installed_server_parameters"](["-I", "-c", program])
    child = subprocess.Popen([params.command, *params.args], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, env={**os.environ, **params.env}, cwd=tmp_path)
    try:
        pid, prefix = json.loads(child.stdout.readline())
        assert pid == child.pid  # Fails for the Windows venv redirector.
        assert Path(prefix).resolve() == Path(sys.prefix).resolve()
    finally:
        child.kill()
        child.wait(timeout=GRACE_SECONDS)
        for stream in (child.stdin, child.stdout, child.stderr):
            stream.close()
    # On Windows an open file denies deletion. Awaiting the real interpreter
    # must release it immediately, without retry, sleep or ignored cleanup.
    (tmp_path / "owned-file").unlink()


def wait_until(predicate, seconds=GRACE_SECONDS):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(0.02)
    assert predicate(), "Lifecycle condition did not complete within the declared grace"


class WireServer:
    def __init__(self, root, timeout=30):
        self.data = root / "data"
        self.state = root / "state"
        self.data.mkdir()
        self.state.mkdir()
        (self.data / "hold.csv").write_text("hold\n1\n", encoding="utf-8")
        (self.data / "normal.csv").write_text("x\n1\n2\n", encoding="utf-8")
        self.ready = None
        self.messages = queue.Queue()
        fixture = Path(__file__).with_name("fixtures") / "mcp_lifecycle_server.py"
        self.process = subprocess.Popen(
            [sys.executable, str(fixture), str(self.data), str(self.state), str(timeout)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )

        def read():
            for line in self.process.stdout:
                self.messages.put(json.loads(line))

        self.reader = threading.Thread(target=read, daemon=True)
        self.reader.start()

    def send(self, message):
        self.process.stdin.write(json.dumps({"jsonrpc": "2.0", **message}) + "\n")
        self.process.stdin.flush()

    def result(self, id, seconds=10):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            message = self.messages.get(timeout=max(0.01, end - time.monotonic()))
            if message.get("id") == id:
                return message
        raise AssertionError("No matching MCP response")

    def initialize(self):
        self.send(
            {
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {},
                    "clientInfo": {"name": "lifecycle-contract", "version": "1"},
                },
            }
        )
        result = self.result(1)["result"]
        assert result["protocolVersion"] == "2025-11-25"
        self.send({"method": "notifications/initialized"})
        return result

    def start_long(self):
        self.send(
            {
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "check_file",
                    "arguments": {"path": str(self.data / "hold.csv")},
                },
            }
        )
        ready = self.state / "ready.json"
        wait_until(ready.exists, seconds=10)
        self.ready = json.loads(ready.read_text(encoding="utf-8"))
        self.ready["created"] = psutil.Process(self.ready["pid"]).create_time()
        assert Path(self.ready["directory"]).is_dir()

    def worker_exists(self):
        try:
            return psutil.Process(self.ready["pid"]).create_time() == self.ready["created"]
        except psutil.NoSuchProcess:
            return False

    def assert_reclaimed(self):
        wait_until(lambda: not self.worker_exists())  # Includes zombie/reap detection.
        wait_until(lambda: not Path(self.ready["directory"]).exists())

    def close(self):
        # Only this fixture's owned process tree is eligible for cleanup.
        if self.ready and self.worker_exists():
            psutil.Process(self.ready["pid"]).kill()
        if self.process.poll() is None:
            try:
                self.process.stdin.close()
                self.process.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                self.process.kill()
                self.process.wait(timeout=5)
        if self.ready:
            directory = Path(self.ready["directory"])
            if directory.exists():
                assert directory.parent.resolve() == Path(tempfile.gettempdir()).resolve()
                assert directory.name.startswith("truthound-mcp-") and not directory.is_symlink()
                shutil.rmtree(directory)
        self.reader.join(timeout=1)
        self.process.stdout.close()


def test_stdio_cancel_reaps_worker_cleans_copy_and_allows_followup(tmp_path):
    server = WireServer(tmp_path)
    try:
        server.initialize()
        server.start_long()
        server.send(
            {"method": "notifications/cancelled", "params": {"requestId": 2, "reason": "contract"}}
        )
        server.assert_reclaimed()
        assert server.process.poll() is None
        server.send(
            {
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "profile_file",
                    "arguments": {"path": str(server.data / "normal.csv")},
                },
            }
        )
        result = server.result(3)["result"]
        assert not result["isError"]
        assert result["structuredContent"]["data"]["row_count"] == 2
    finally:
        server.close()


def test_stdio_eof_reaps_worker_cleans_copy_and_exits(tmp_path):
    server = WireServer(tmp_path)
    try:
        server.initialize()
        server.start_long()
        server.process.stdin.close()  # Actual transport EOF, not task cancellation.
        server.assert_reclaimed()
        assert server.process.wait(timeout=GRACE_SECONDS) == 0
    finally:
        server.close()


def test_stdio_timeout_reaps_worker_and_cleans_copy(tmp_path):
    server = WireServer(tmp_path, timeout=1)
    try:
        server.initialize()
        server.start_long()
        result = server.result(2)["result"]
        assert result["isError"]
        assert result["structuredContent"]["error"]["code"] == "timeout"
        server.assert_reclaimed()
    finally:
        server.close()
