"""Real worker restart, signed checkpoints and loopback outbox delivery contracts."""

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("mcp")

from tests import test_mcp_schema_monitor as monitor_tests
from tests.test_mcp_plugin_bindings import invoke, source_artifact
from tests.test_mcp_schema_monitor import change
from truthound.mcp.approval import ApprovalStore

pytestmark = pytest.mark.contract
runtime = monitor_tests.runtime
receiver = monitor_tests.receiver


def test_disconnected_parent_reaps_blocked_one_shot_worker(tmp_path):
    import time

    marker = tmp_path / "entered"
    program = (
        "from pathlib import Path; import time; "
        "from truthound.mcp import durable_schema_worker as worker\n"
        "def blocked(request):\n"
        f"    Path({str(marker)!r}).write_text('entered', encoding='utf-8')\n"
        "    while True: time.sleep(0.01)\n"
        "worker.execute = blocked\nworker.main()\n"
    )
    process = subprocess.Popen(
        [sys.executable, "-I", "-c", program],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        process.stdin.write(b"{}\n")
        process.stdin.flush()
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert marker.exists() and process.poll() is None
        process.stdin.close()  # Actual parent pipe loss during a blocked operation.
        assert process.wait(timeout=5) == 75
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        for stream in (process.stdin, process.stdout, process.stderr):
            stream.close()


def register(runtime, monkeypatch, tmp_path, mode="sync", **options):
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
    document = json.loads(path.read_text(encoding="utf-8"))
    document["plugins"]["fixture"]["capabilities"].update(
        {"schema-factory": "schema-factory", "alert-factory": "alert-factory"}
    )
    document["schema_monitors"] = {
        "durable": {"source": "schema-factory", "notifier": "alert-factory"}
    }
    path.write_text(json.dumps(document), encoding="utf-8")
    change(tmp_path, {"x": "Int64", "y": "String", "z": "String"})
    request = dict(
        binding_id="durable",
        mode=mode,
        poll_interval=0.1,
        idempotency_key="durable-watch-intent",
        **options,
    )
    plan = invoke(adapter, "schema_watch_register", **request)
    assert plan["issues"] == [{"code": "approval_required"}], plan
    assert "automatic_restart_reads" in plan["data"]["effects"]
    assert "persistent_external_notification_queue" in plan["data"]["effects"]
    assert not marker.exists()
    store = ApprovalStore(control)
    try:
        store.approve(plan["data"]["plan_id"])
    finally:
        store.close()
    result = invoke(adapter, "schema_watch_register", **request)
    assert result["execution_status"] == "succeeded", result
    assert invoke(adapter, "schema_watch_register", **request) == result
    return result["data"]["watch_id"]


def worker(runtime, *, crash_after_send=False, crash_before_checkpoint=False):
    adapter, control, *_ = runtime
    args = [
        "worker",
        "--once",
        "--profile",
        "core",
        "--allow-sensitive-results",
        "--allow-dir",
        str(adapter.policy.allowed_directories[0]),
        "--control-dir",
        str(control),
    ]
    command = [adapter._worker_argv()[0], "-I", "-m", "truthound.mcp", *args]
    if crash_after_send or crash_before_checkpoint:
        code = """
import os, sys, runpy
from truthound.mcp import durable_schema
original = durable_schema.write
def crash(store, identity, record):
    if (os.environ.get('TRUTHOUND_TEST_CRASH_POLL') == '1' and
        (record['checkpoint'] or {}).get('polls') == 2 and not record['busy']):
        os._exit(74)
    if (os.environ.get('TRUTHOUND_TEST_CRASH_POLL') != '1' and
        any(item['state'] == 'sent' for item in record['outbox'])):
        os._exit(73)
    return original(store, identity, record)
durable_schema.write = crash
sys.argv = ['truthound.mcp'] + sys.argv[1:]
runpy.run_module('truthound.mcp', run_name='__main__')
"""
        command = [adapter._worker_argv()[0], "-I", "-c", code, *args]
    import os

    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=45,
        env={**os.environ, "TRUTHOUND_TEST_CRASH_POLL": "1" if crash_before_checkpoint else "0"},
    )


def state(adapter, identity):
    result = invoke(adapter, "schema_watch_status", watch_id=identity, include_snapshot=True)
    assert result["execution_status"] == "succeeded", result
    return result["data"], source_artifact(adapter, result["data"]["artifact"]["resource_id"])


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_restart_checkpoint_outbox_matches_native_core(
    runtime, monkeypatch, tmp_path, receiver, mode
):
    import asyncio

    from truthound.profiler.evolution.breaking_alerts import BreakingChangeAlertManager
    from truthound.profiler.evolution.history import InMemorySchemaHistory, SchemaHistory
    from truthound.profiler.evolution.watcher import (
        AlertingEventHandler,
        AsyncSchemaWatcher,
        FileSchemaSource,
        HistoryEventHandler,
        SchemaWatcher,
    )

    adapter = runtime[0]
    identity = register(runtime, monkeypatch, tmp_path, mode)
    direct = SchemaWatcher() if mode == "sync" else AsyncSchemaWatcher()
    history, alerts = InMemorySchemaHistory(), BreakingChangeAlertManager()
    direct.add_source(FileSchemaSource(tmp_path / "schema.json"))
    for handler in (HistoryEventHandler(SchemaHistory(history)), AlertingEventHandler(alerts)):
        direct.add_handler(handler if mode == "sync" else handler.handle)
    assert worker(runtime).returncode == 0
    for schema in ({"x": "Int64", "y": "String"}, {"x": "Int64"}):
        change(tmp_path, schema)
        direct.check_now() if mode == "sync" else asyncio.run(direct.check_now())
        assert worker(runtime).returncode == 0  # Separate process recovers old baseline.
        info, snapshot = state(adapter, identity)
        assert info["notifications"].get("pending") == 1
        assert len(receiver[0]) == len(snapshot["outbox"]) - 1  # No send before durable commit.
        assert worker(runtime).returncode == 0  # New process consumes queued alert.
        info, snapshot = state(adapter, identity)
        actual = snapshot["checkpoint"]
        import sqlite3
        from contextlib import closing

        with closing(sqlite3.connect(runtime[1] / "control.sqlite3")) as fresh:
            fresh_states = [
                row[0]
                for row in fresh.execute(
                    "SELECT json_extract(record,'$.outbox[0].state') FROM mcp_schema_watches"
                )
            ]
        assert fresh_states == ["sent"]
        assert not adapter._session.approvals().connection.in_transaction
        assert info["notifications"] == {"sent": len(alerts.get_alert_history())}
        assert sorted(receiver[0], key=lambda item: item["alert_id"]) == sorted(
            actual["alerts"], key=lambda item: item["alert_id"]
        )
        for left, right in zip(
            actual["alerts"], [a.to_dict() for a in alerts.get_alert_history()], strict=True
        ):
            assert {k: v for k, v in left.items() if k != "detected_at"} == {
                k: v for k, v in right.items() if k != "detected_at"
            }
        for left, right in zip(
            actual["versions"], [v.to_dict() for v in history.list_versions()], strict=True
        ):
            for key in ("schema", "version", "metadata", "changes_from_parent"):
                assert left[key] == right[key]
        assert actual["events"] == len(actual["versions"]) == len(alerts.get_alert_history())
        assert actual["versions"][-1]["parent_id"] is None
        if len(actual["versions"]) == 2:
            assert actual["versions"][0]["parent_id"] == actual["versions"][1]["version_id"]
    assert invoke(adapter, "schema_watch_disable", watch_id=identity)["data"]["state"] == "disabled"
    before = state(adapter, identity)
    change(tmp_path, {})
    assert worker(runtime).returncode == 0
    assert state(adapter, identity)[1] == before[1]
    assert len(receiver[0]) == 2


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("failure", ["http", "crash", "binding", "tamper", "quota"])
def test_durable_failure_never_replays_unknown_send(
    runtime, monkeypatch, tmp_path, receiver, mode, failure
):
    adapter, control, *_ = runtime
    identity = register(
        runtime, monkeypatch, tmp_path, mode, max_events=1 if failure == "quota" else 10
    )
    assert worker(runtime).returncode == 0
    change(tmp_path, {"x": "Int64"})
    assert worker(runtime).returncode == 0
    assert state(adapter, identity)[0]["notifications"] == {"pending": 1}
    if failure == "binding":
        path = control / "bindings.json"
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        assert worker(runtime).returncode != 0
        assert not receiver[0]
        return
    if failure == "tamper":
        store = ApprovalStore(control)
        try:
            store.connection.execute(
                "UPDATE mcp_schema_watches SET record='{}' WHERE identity=?", (identity,)
            )
        finally:
            store.close()
        assert worker(runtime).returncode != 0
        assert not receiver[0]
        return
    if failure == "http":
        receiver[1][0] = 503
    result = worker(runtime, crash_after_send=failure == "crash")
    assert result.returncode == (73 if failure == "crash" else 0 if failure == "quota" else 2)
    assert len(receiver[0]) == 1
    if failure == "quota":
        assert worker(runtime).returncode != 0
        assert state(adapter, identity)[0]["state"] == "failed"
    else:
        assert worker(runtime).returncode == 0
        assert state(adapter, identity)[0]["notifications"] == {"unknown": 1}
    assert worker(runtime).returncode == 0
    assert len(receiver[0]) == 1


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_dead_poll_recovery_commits_one_checkpoint_and_one_notification(
    runtime, monkeypatch, tmp_path, receiver, mode
):
    identity = register(runtime, monkeypatch, tmp_path, mode)
    assert worker(runtime).returncode == 0
    change(tmp_path, {"x": "Int64"})
    assert worker(runtime, crash_before_checkpoint=True).returncode == 74
    info, snapshot = state(runtime[0], identity)
    assert info["polls"] == 1 and snapshot["outbox"] == [] and not receiver[0]
    assert worker(runtime).returncode == 0
    info, snapshot = state(runtime[0], identity)
    assert info["polls"] == 2 and len(snapshot["checkpoint"]["versions"]) == 1
    assert info["notifications"] == {"pending": 1}
    assert worker(runtime).returncode == 0
    assert len(receiver[0]) == 1
    assert state(runtime[0], identity)[0]["notifications"] == {"sent": 1}


@pytest.mark.parametrize("boundary", ["live-owner", "expired", "disabled"])
def test_queue_authority_and_live_lease_block_new_effects(
    runtime, monkeypatch, tmp_path, receiver, boundary
):
    import os

    import psutil

    from truthound.mcp.durable_schema import read, write

    adapter, control, *_ = runtime
    identity = register(runtime, monkeypatch, tmp_path)
    assert worker(runtime).returncode == 0
    change(tmp_path, {"x": "Int64"})
    assert worker(runtime).returncode == 0
    store = ApprovalStore(control)
    try:
        with store.transaction():
            record = read(store, identity)
            if boundary == "live-owner":
                record["busy"] = dict(
                    pid=os.getpid(), created=psutil.Process().create_time(), task="poll", index=None
                )
            elif boundary == "expired":
                record["expires_at"] = store.clock() - 1
            else:
                record["state"] = "disabled"
            write(store, identity, record)
    finally:
        store.close()
    assert worker(runtime).returncode == 0
    info, snapshot = state(adapter, identity)
    assert not receiver[0] and info["notifications"] == {"pending": 1}
    assert info["polls"] == 2
    assert info["state"] == ("active" if boundary == "live-owner" else boundary)
