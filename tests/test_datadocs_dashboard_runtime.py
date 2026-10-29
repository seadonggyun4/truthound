"""Actual Reflex page compilation and prepared loopback server ownership."""

import http.client
import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pytest

from tests import test_mcp_plugin_bindings as plugins
from tests.test_mcp_brokers import approved
from tests.test_mcp_dashboards import free_port

runtime = plugins.runtime
pytestmark = pytest.mark.contract


@pytest.mark.fault
def test_changed_prepared_configuration_is_rejected_before_occupied_port(tmp_path):
    import hashlib
    from truthound.datadocs.dashboard import DashboardConfig, ReflexDashboardServer

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        config = DashboardConfig(host="127.0.0.1", port=listener.getsockname()[1])
        original = json.dumps(asdict(config))
        (tmp_path / "configuration.json").write_text(original)
        (tmp_path / "prepared.json").write_text(json.dumps({
            "config_sha256": hashlib.sha256(original.encode()).hexdigest(),
            "profile_sha256": None,
        }))
        (tmp_path / "configuration.json").write_text(json.dumps({**asdict(config), "title": "Changed"}))
        server = ReflexDashboardServer(tmp_path, config)
        with pytest.raises(ValueError, match="configuration has changed"):
            server.start()
        assert server._process is None
        assert listener.getsockname()[1] == config.port


def test_reflex_native_page_compiles_with_real_profile_and_reactive_state():
    from truthound.datadocs.dashboard import DashboardApp, DashboardConfig, ProfileState

    profile = {
        "row_count": 3,
        "column_count": 1,
        "estimated_memory_bytes": 2048,
        "columns": [
            {
                "name": "x",
                "null_ratio": 0.6,
                "unique_ratio": 0.4,
                "distinct_count": 2,
                "patterns": [{"name": "numeric"}],
            }
        ],
        "correlations": [{"column1": "x", "column2": "y", "correlation": 0.5}],
    }
    owner = DashboardApp(DashboardConfig(profile_data=profile, theme="dark"))
    app = owner._create_app()
    app._compile_page("index")
    state = owner._state_class(_reflex_internal_init=True)
    expected = ProfileState.from_profile(profile)
    assert state.row_count == expected.row_count and not state.is_loading
    assert state.quality_score == expected.quality_score and state.alerts == expected.alerts
    assert state.correlations == expected.correlations and state.format_memory == "2.0 KB"
    state.search_query = "missing"
    assert state.filtered_columns == []
    state.search_query = "X"
    assert state.filtered_columns[0]["name"] == "x"


@pytest.fixture(scope="module")
def prepared_reflex(tmp_path_factory):
    from truthound.datadocs.dashboard import DashboardConfig, ReflexDashboardServer

    root = tmp_path_factory.mktemp("native-reflex")
    config = DashboardConfig(
        host="127.0.0.1",
        port=free_port(),
        title="Verified native profile",
        profile_data={
            "row_count": 3,
            "column_count": 1,
            "columns": [{"name": "x", "null_ratio": 0.5}],
        },
    )
    server = ReflexDashboardServer(root / "project", config)
    try:
        server.prepare()
        yield server
    finally:
        server.close()


def test_prepared_reflex_native_and_approved_mcp_lifecycle(prepared_reflex, runtime, monkeypatch):
    from types import SimpleNamespace

    from truthound.mcp import model_process

    server = prepared_reflex
    try:
        assert server.start()["ready"]
        connection = http.client.HTTPConnection(server.config.host, server.config.port, timeout=2)
        try:
            connection.request("GET", "/")
            response = connection.getresponse()
            assert response.status == 200
            assert b"Verified native profile" in response.read()
        finally:
            connection.close()
        assert not server.stop()["running"]
        assert server.start()["ready"]
    finally:
        server.close()
    adapter, control, marker, _, _ = runtime
    monkeypatch.setattr(
        model_process,
        "sys",
        SimpleNamespace(
            executable=adapter._worker_argv()[0],
            platform=sys.platform,
            _base_executable=sys._base_executable,
        ),
    )
    (control.parent / "dashboard-settings.json").write_text(
        json.dumps({"project": str(server.project_directory)})
    )
    settings = json.loads((control / "bindings.json").read_text())
    settings["plugins"]["fixture"]["capabilities"]["reflex-dashboard"] = "reflex-dashboard-factory"
    settings["dashboards"] = {"local": "reflex-dashboard"}
    (control / "bindings.json").write_text(json.dumps(settings))
    identity = approved(
        runtime, "dashboards_open", binding_id="local", idempotency_key="reflex-dashboard-open"
    )["data"]["resource_id"]
    result = approved(
        runtime,
        "dashboards_execute",
        binding_id="local",
        service_id=identity,
        command="start",
        idempotency_key="reflex-dashboard-start",
    )
    assert plugins.source_artifact(adapter, result["data"]["resource_id"])["ready"]
    assert (
        plugins.invoke(adapter, "resources_release", resource_id=identity)["execution_status"]
        == "succeeded"
    )
    assert "reflex-dashboard-close" in marker.read_text().splitlines()
    with socket.socket() as probe:
        assert probe.connect_ex((server.config.host, server.config.port)) != 0


@pytest.mark.fault
@pytest.mark.parametrize("previous_state", ["fresh", "stopped"])
def test_prepared_reflex_child_exits_on_owner_death_and_changed_config_is_refused(
    prepared_reflex, previous_state
):
    import psutil

    server = prepared_reflex
    program = """
import json, sys
from pathlib import Path
from truthound.datadocs.dashboard import DashboardConfig, ReflexDashboardServer
root = Path(sys.argv[1])
server = ReflexDashboardServer(root, DashboardConfig(**json.loads((root/'configuration.json').read_text())))
try:
    server.start()
    print(json.dumps({'pid': server._process.pid}), flush=True)
    server.wait()
finally: server.close()
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    owner = subprocess.Popen(
        [sys.executable, "-c", program, str(server.project_directory)],
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    child = None
    try:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=1) as pool:
            line = pool.submit(owner.stdout.readline).result(timeout=15)
        assert line, owner.stderr.read()
        child = psutil.Process(json.loads(line)["pid"])
        owner.kill()
        owner.wait(timeout=5)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            exited = not child.is_running() or child.status() == psutil.STATUS_ZOMBIE
            with socket.socket() as probe:
                port_closed = probe.connect_ex((server.config.host, server.config.port)) != 0
            if exited and port_closed:
                break
            time.sleep(0.01)
        assert not child.is_running() or child.status() == psutil.STATUS_ZOMBIE
        with socket.socket() as probe:
            assert probe.connect_ex((server.config.host, server.config.port)) != 0
    finally:
        if owner.poll() is None:
            owner.kill()
        owner.wait(timeout=5)
        if child is not None and child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
            child.kill()
        owner.stdout.close()
        owner.stderr.close()
    # Each shard must establish its own process state, independently of whether
    # the lifecycle test ran earlier against the module-scoped prepared project.
    server = type(server)(server.project_directory, server.config)
    if previous_state == "stopped":
        assert server.start()["ready"]
        assert not server.stop()["running"]
        assert server._process.poll() is not None
    else:
        assert server._process is None
    config = server.project_directory / "configuration.json"
    original = config.read_bytes()
    previous_process = server._process
    try:
        config.write_text(json.dumps({**asdict(server.config), "title": "Changed"}))
        with pytest.raises(ValueError, match="configuration has changed"):
            server.start()
        assert server._process is previous_process
        if previous_state == "stopped":
            assert server._process.poll() is not None
        else:
            assert server._process is None
    finally:
        config.write_bytes(original)
