"""Actual native polling, installed plugin approval and loopback HTTP delivery."""

import asyncio
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

pytest.importorskip("mcp")

from tests import test_mcp_plugin_bindings as plugin_tests
from tests.test_mcp_plugin_bindings import invoke, source_artifact
from truthound.mcp.approval import ApprovalStore

pytestmark = pytest.mark.contract
runtime = plugin_tests.runtime


@pytest.fixture
def receiver(tmp_path):
    messages = []
    response_code = [200]

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            messages.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(response_code[0])
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    (tmp_path / "endpoint.txt").write_text(
        f"http://127.0.0.1:{server.server_port}/", encoding="utf-8"
    )
    try:
        yield messages, response_code
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def prepare(runtime, monkeypatch, tmp_path, mode, max_events=10, false_notifier=False):
    from truthound.mcp import model_process

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
    path = control / "bindings.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["plugins"]["fixture"]["capabilities"].update(
        {
            "schema-factory": "schema-factory",
            "alert-factory": "alert-factory",
        }
    )
    config["schema_monitors"] = {
        "approved-watch": {"source": "schema-factory", "notifier": "alert-factory"}
    }
    if false_notifier:
        config["plugins"]["fixture"]["capabilities"]["alert-factory"] = "false-notifier"
    path.write_text(json.dumps(config), encoding="utf-8")
    change(tmp_path, {"x": "Int64", "y": "String"})
    request = dict(
        binding_id="approved-watch",
        mode=mode,
        poll_interval=0.1,
        max_events=max_events,
        idempotency_key="monitor-test-open",
    )
    pending = invoke(adapter, "schema_monitor_open", **request)
    assert pending["issues"] == [{"code": "approval_required"}], pending
    assert set(pending["data"]["effects"]) == {
        "installed_extension_execute",
        "continuous_schema_read",
        "external_schema_alert_send",
    }
    assert not marker.exists()
    store = ApprovalStore(control)
    try:
        store.approve(pending["data"]["plan_id"])
    finally:
        store.close()
    return request


def change(tmp_path, schema):
    temporary = tmp_path / "schema.next"
    temporary.write_text(json.dumps(schema), encoding="utf-8")
    temporary.replace(tmp_path / "schema.json")


def status(adapter, identity, operation="schema_monitor_status"):
    result = invoke(adapter, operation, monitor_id=identity)
    assert result["execution_status"] == "succeeded", result
    return source_artifact(adapter, result["data"]["resource_id"])


def until(adapter, identity, predicate):
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        value = status(adapter, identity)
        if predicate(value):
            return value
        time.sleep(0.05)
    pytest.fail(f"native poller did not reach expected state: {value}")


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_native_polling_history_and_http_delivery_match_core(
    runtime, monkeypatch, tmp_path, receiver, mode
):
    from truthound.profiler.evolution.breaking_alerts import BreakingChangeAlertManager
    from truthound.profiler.evolution.history import InMemorySchemaHistory, SchemaHistory
    from truthound.profiler.evolution.watcher import (
        AlertingEventHandler,
        AsyncSchemaWatcher,
        FileSchemaSource,
        HistoryEventHandler,
        SchemaWatcher,
    )

    adapter, _, marker, _, _ = runtime
    request = prepare(runtime, monkeypatch, tmp_path, mode)
    direct = SchemaWatcher() if mode == "sync" else AsyncSchemaWatcher()
    history, alerts = InMemorySchemaHistory(), BreakingChangeAlertManager()
    direct.add_source(FileSchemaSource(tmp_path / "schema.json"))
    for handler in (HistoryEventHandler(SchemaHistory(history)), AlertingEventHandler(alerts)):
        direct.add_handler(handler if mode == "sync" else handler.handle)
    opened = invoke(adapter, "schema_monitor_open", **request)
    assert opened["execution_status"] == "succeeded", opened
    identity = opened["data"]["resource_id"]
    assert invoke(adapter, "schema_monitor_open", **request) == opened
    until(adapter, identity, lambda state: state["polls"] >= 1)
    change(tmp_path, {"x": "Int64"})
    events = direct.check_now() if mode == "sync" else asyncio.run(direct.check_now())
    assert len(events) == 1
    actual = until(adapter, identity, lambda state: state["notifications_sent"] == 1)
    assert actual["state"] == "running" and actual["error"] is None
    assert actual["events"] == 1 and len(actual["versions"]) == 1
    expected = alerts.get_alert_history()[0].to_dict()
    assert set(actual["alerts"][0]) == set(expected)
    for key in set(expected) - {"detected_at"}:
        assert actual["alerts"][0][key] == expected[key], key
    assert receiver[0] == actual["alerts"]
    expected_version = history.list_versions()[0].to_dict()
    assert set(actual["versions"][0]) == set(expected_version)
    for key in set(expected_version) - {"version_id", "created_at"}:
        assert actual["versions"][0][key] == expected_version[key], key
    from datetime import datetime
    from uuid import UUID

    UUID(actual["versions"][0]["version_id"])
    datetime.fromisoformat(actual["versions"][0]["created_at"])
    datetime.fromisoformat(actual["alerts"][0]["detected_at"])
    stopped = status(adapter, identity, "schema_monitor_stop")
    assert stopped["state"] == "stopped"
    assert marker.read_text(encoding="utf-8").splitlines().count("teardown") == 1
    change(tmp_path, {})
    time.sleep(0.25)
    assert status(adapter, identity) == stopped and len(receiver[0]) == 1
    assert status(adapter, identity, "schema_monitor_stop") == stopped


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize(
    "failure",
    ["http", "false", "corrupt", "binding", "budget", "release", "close", "expiry", "policy"],
)
def test_failure_or_revocation_stops_effects_without_retry(
    runtime, monkeypatch, tmp_path, receiver, mode, failure
):
    adapter, control, marker, _, _ = runtime
    request = prepare(
        runtime,
        monkeypatch,
        tmp_path,
        mode,
        max_events=1 if failure == "budget" else 10,
        false_notifier=failure == "false",
    )
    opened = invoke(adapter, "schema_monitor_open", **request)
    assert opened["execution_status"] == "succeeded", opened
    identity = opened["data"]["resource_id"]
    worker = adapter._session._monitors.workers[identity]
    until(adapter, identity, lambda state: state["polls"] >= 1)
    if failure == "http":
        receiver[1][0] = 503
    elif failure == "corrupt":
        (tmp_path / "schema.json").write_text("invalid json", encoding="utf-8")
    elif failure == "binding":
        path = control / "bindings.json"
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    elif failure == "release":
        result = invoke(adapter, "resources_release", resource_id=identity)
        assert result["execution_status"] == "succeeded", result
    elif failure == "close":
        adapter.close()
    elif failure in {"expiry", "policy"}:
        if failure == "expiry":
            worker.expire_after(0)
        else:
            from dataclasses import replace

            adapter.policy = replace(adapter.policy, allow_sensitive_results=False)
        deadline = time.monotonic() + 5
        while worker.process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
    if failure != "corrupt":
        change(tmp_path, {"x": "Int64"})
    if failure in {"http", "false", "corrupt", "budget"}:
        state = until(adapter, identity, lambda state: state["state"] == "failed")
        assert state["error"] and "synthetic" not in json.dumps(state)
        assert state["notifications_sent"] == (1 if failure == "budget" else 0)
    elif failure == "binding":
        failed = invoke(adapter, "schema_monitor_status", monitor_id=identity)
        assert failed["execution_status"] == "failed", failed
    if failure in {"binding", "release", "close", "expiry", "policy"}:
        assert worker.closed and worker.process.poll() is not None
    sent = len(receiver[0])
    change(tmp_path, {})
    time.sleep(0.25)
    assert len(receiver[0]) == sent
    assert sent == (1 if failure in {"http", "budget"} else 0)


@pytest.mark.parametrize("state", ["starting", "error"])
def test_native_stop_joins_from_intermediate_or_failed_state(state):
    from truthound.profiler.evolution.watcher import SchemaWatcher, WatcherState

    watcher = SchemaWatcher()
    watcher._state = WatcherState(state)
    watcher._watch_thread = threading.Thread(target=lambda: watcher._stop_event.wait(2))
    watcher._watch_thread.start()
    watcher.stop()
    assert watcher._stop_event.is_set() and not watcher._watch_thread.is_alive()


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_unexpected_native_poller_exit_cannot_report_healthy(mode):
    from truthound.mcp.schema_monitor_runtime import MonitorRuntime
    from truthound.profiler.evolution.watcher import DictSchemaSource

    monitor = MonitorRuntime(
        DictSchemaSource({"x": "Int64"}),
        None,
        mode=mode,
        interval=0.1,
        max_events=10,
        budget=65536,
        guard=lambda: None,
    )
    try:
        monitor.start()
        if mode == "sync":
            monitor.watcher.stop()
        else:
            asyncio.run_coroutine_threadsafe(monitor.watcher.stop(), monitor.loop).result()
        state = monitor.snapshot()
        assert state["state"] == "failed"
        assert state["error"] == "monitor_poller_terminated"
    finally:
        monitor.stop()
