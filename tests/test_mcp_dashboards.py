"""Real loopback native dashboard lifecycle, process ownership and title safety."""

import http.client
import json
import socket
import sys
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from tests import test_mcp_plugin_bindings as plugins
from tests.test_mcp_brokers import approved

runtime = plugins.runtime
invoke = plugins.invoke
pytestmark = pytest.mark.contract


def free_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def test_dashboard_real_start_status_stop_and_release(runtime, monkeypatch):
    from truthound.mcp import model_process

    adapter, control, _, _, _ = runtime
    port = free_port()
    (control.parent / "dashboard-settings.json").write_text(
        json.dumps({"port": port}), encoding="utf-8"
    )
    monkeypatch.setattr(
        model_process,
        "sys",
        SimpleNamespace(
            executable=adapter._worker_argv()[0],
            platform=sys.platform,
            _base_executable=sys._base_executable,
        ),
    )
    path = control / "bindings.json"
    settings = json.loads(path.read_text())
    settings["plugins"]["fixture"]["capabilities"]["dashboard-factory"] = "dashboard-factory"
    settings["dashboards"] = {"local": "dashboard-factory"}
    path.write_text(json.dumps(settings), encoding="utf-8")
    identity = approved(
        runtime, "dashboards_open", binding_id="local", idempotency_key="dashboard-open"
    )["data"]["resource_id"]

    def command(action, key):
        response = approved(
            runtime,
            "dashboards_execute",
            binding_id="local",
            service_id=identity,
            command=action,
            idempotency_key=key,
        )
        return plugins.source_artifact(adapter, response["data"]["resource_id"])

    assert command("status", "dashboard-before-start")["running"] is False
    result = command("start", "dashboard-start")
    assert result["running"] is True and result["ready"] is True
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
    try:
        connection.request("GET", "/_stcore/health")
        assert connection.getresponse().status == 200
    finally:
        connection.close()
    assert command("stop", "dashboard-stop")["running"] is False
    assert command("start", "dashboard-restart")["ready"] is True
    worker = adapter._session._administration["dashboards"].workers[identity]
    assert (
        invoke(adapter, "resources_release", resource_id=identity)["execution_status"]
        == "succeeded"
    )
    assert worker.closed and worker.process.poll() is not None
    with socket.socket() as probe:
        probe.settimeout(2)
        assert probe.connect_ex(("127.0.0.1", port)) != 0


def test_dashboard_does_not_claim_readiness_of_an_unrelated_port():
    from truthound.profiler.dashboard.config import DashboardConfig
    from truthound.profiler.dashboard.server import DashboardServer

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        server = DashboardServer(DashboardConfig(host="127.0.0.1", port=listener.getsockname()[1]))
        with pytest.raises(OSError):
            server.start()
        assert server._process is None
        assert listener.fileno() >= 0
        server.close()


def test_dashboard_title_is_rendered_as_data_and_cannot_execute_python(tmp_path):
    from streamlit.testing.v1 import AppTest

    from truthound.profiler.dashboard.config import DashboardConfig

    marker = tmp_path / "injected-code"
    title = f'"; __import__("pathlib").Path({str(marker)!r}).touch(); #'
    config = tmp_path / "configuration.json"
    config.write_text(json.dumps(asdict(DashboardConfig(title=title))), encoding="utf-8")
    script = tmp_path / "dashboard.py"
    script.write_text(
        "from truthound.profiler.dashboard.server import render\n" + f"render({str(config)!r})\n",
        encoding="utf-8",
    )
    app = AppTest.from_file(str(script)).run()
    assert not app.exception
    assert not marker.exists()


def test_native_dashboard_child_exits_after_owning_process_is_hard_killed(tmp_path):
    import os
    import subprocess
    import time
    from pathlib import Path

    import psutil

    port = free_port()
    environment = dict(os.environ)
    environment['PYTHONPATH'] = str(Path(__file__).resolve().parents[1] / 'src')
    program = """
import json, sys
from truthound.profiler.dashboard.server import DashboardServer
from truthound.profiler.dashboard.config import DashboardConfig
server = DashboardServer(DashboardConfig(host='127.0.0.1', port=int(sys.argv[1])))
try:
    server.start()
    print(json.dumps({'pid': server._process.pid}), flush=True)
    server.wait()
finally:
    server.close()
"""
    owner = subprocess.Popen([sys.executable, '-c', program, str(port)], env=environment,
                             stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    child = None
    try:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=1) as pool:
            line = pool.submit(owner.stdout.readline).result(timeout=15)
        assert line, owner.stderr.read()
        child = psutil.Process(json.loads(line)['pid'])
        owner.kill()
        owner.wait(timeout=5)
        deadline = time.monotonic() + 5
        while child.is_running() and child.status() != psutil.STATUS_ZOMBIE and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not child.is_running() or child.status() == psutil.STATUS_ZOMBIE
        with socket.socket() as probe:
            assert probe.connect_ex(('127.0.0.1', port)) != 0
    finally:
        if owner.poll() is None:
            owner.kill()
        owner.wait(timeout=5)
        if child is not None and child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
            child.kill()
        owner.stdout.close()
        owner.stderr.close()
