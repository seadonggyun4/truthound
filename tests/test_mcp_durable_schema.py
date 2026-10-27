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


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("kind", ["polars", "json"])
def test_restored_native_types_preserve_core_detection_and_version_history(mode, kind):
    import asyncio

    import polars as pl

    from truthound.mcp.durable_schema_worker import checkpoint, restore
    from truthound.mcp.schema_monitor_runtime import MonitorRuntime
    from truthound.profiler.evolution.breaking_alerts import BreakingChangeAlertManager
    from truthound.profiler.evolution.history import InMemorySchemaHistory, SchemaHistory
    from truthound.profiler.evolution.watcher import (
        AlertingEventHandler,
        AsyncSchemaWatcher,
        HistoryEventHandler,
        SchemaWatcher,
    )

    class Source:
        name = "typed-source"
        revision = 0

        def get_schema(self):
            return self.schema.copy()

        def get_checksum(self):
            return str(self.revision)

    source = Source()
    first, second, nested = (
        (pl.Int32, pl.Int64, pl.List(pl.Struct({"at": pl.Datetime("ns", "UTC")})))
        if kind == "polars"
        else (1, 2, ["a", {"z": 2}])
    )
    source.schema = {"x": first, "nested": nested, "gone": "String"}
    direct = SchemaWatcher() if mode == "sync" else AsyncSchemaWatcher()
    history, alerts = InMemorySchemaHistory(), BreakingChangeAlertManager()
    direct.add_source(source)
    for handler in (HistoryEventHandler(SchemaHistory(history)), AlertingEventHandler(alerts)):
        direct.add_handler(handler if mode == "sync" else handler.handle)

    def new_runtime():
        return MonitorRuntime(
            source, None, mode=mode, interval=1, max_events=10, budget=1024**2, guard=lambda: None
        )

    saved = json.loads(json.dumps(checkpoint(new_runtime())))
    for index, schema in enumerate(
        ({"x": second, "nested": nested, "gone": "String"}, {"x": second, "nested": nested}), 1
    ):
        source.schema, source.revision = schema, index
        resumed = new_runtime()
        restore(resumed, saved)
        expected = direct.check_now() if mode == "sync" else asyncio.run(direct.check_now())
        actual = (
            resumed.watcher.check_now()
            if mode == "sync"
            else asyncio.run(resumed.watcher.check_now())
        )
        assert resumed.error is None and len(expected) == len(actual) == 1
        assert [change.to_dict() for change in actual[0].changes] == [
            change.to_dict() for change in expected[0].changes
        ]
        assert [change.column for change in actual[0].changes] == ["x" if index == 1 else "gone"]
        for left, right in zip(
            resumed.history.list_versions(), history.list_versions(), strict=True
        ):
            assert left.schema == right.schema
            assert left.version == right.version
            assert [change.to_dict() for change in left.changes_from_parent] == [
                change.to_dict() for change in right.changes_from_parent
            ]
        assert len(resumed.alerts.get_alert_history()) == len(alerts.get_alert_history())
        saved = json.loads(json.dumps(checkpoint(resumed)))


@pytest.mark.parametrize("failure", ["unsupported-object", "corrupt-ipc"])
def test_checkpoint_type_codec_rejects_unsafe_or_corrupt_state(failure):
    import base64

    from truthound.mcp.durable_schema_worker import pack_schema, unpack_schema
    from truthound.mcp.policy import MCPError

    if failure == "unsupported-object":
        from enum import Enum

        class CustomType(Enum):
            VALUE = "not-a-string-type"

        for value in (object(), CustomType.VALUE):
            with pytest.raises(MCPError):
                pack_schema({"x": value})
    else:
        import polars as pl

        with pytest.raises((OSError, pl.exceptions.PolarsError)):
            unpack_schema({"x": {"kind": "polars", "ipc": base64.b64encode(b"not-arrow").decode()}})


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


def control_request(runtime, identity, action, **options):
    import uuid

    current = invoke(runtime[0], "schema_watch_status", watch_id=identity)
    return dict(
        watch_id=identity,
        expected_revision=current["data"]["revision"],
        action=action,
        idempotency_key=str(uuid.uuid4()),
        **options,
    )


def approve_update(runtime, request):
    plan = invoke(runtime[0], "schema_watch_update", **request)
    assert plan["issues"] == [{"code": "approval_required"}], plan
    store = ApprovalStore(runtime[1])
    try:
        store.approve(plan["data"]["plan_id"])
    finally:
        store.close()
    return invoke(runtime[0], "schema_watch_update", **request)


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_pause_resume_preserves_native_baseline_across_processes(
    runtime, monkeypatch, tmp_path, receiver, mode
):
    from truthound.profiler.evolution.detector import SchemaEvolutionDetector

    identity = register(runtime, monkeypatch, tmp_path, mode)
    assert worker(runtime).returncode == 0
    before, saved = state(runtime[0], identity)
    request = control_request(runtime, identity, "pause")
    result = approve_update(runtime, request)
    assert result["data"]["state"] == "paused"
    assert invoke(runtime[0], "schema_watch_update", **request) == result
    change(tmp_path, {"x": "Int64"})
    assert worker(runtime).returncode == 0
    paused, snapshot = state(runtime[0], identity)
    assert snapshot == saved and paused["polls"] == before["polls"] and not receiver[0]
    assert (
        approve_update(runtime, control_request(runtime, identity, "resume"))["data"]["expires_at"]
        == before["expires_at"]
    )
    assert worker(runtime).returncode == 0
    _, snapshot = state(runtime[0], identity)
    expected = SchemaEvolutionDetector().detect_changes(
        {"x": "Int64"}, {"x": "Int64", "y": "String", "z": "String"}
    )
    # Core history starts at the first changed schema, so its first parent diff
    # is empty. The native alert carries the initial-baseline change evidence.
    actual = snapshot["checkpoint"]["alerts"][0]["changes"]
    assert snapshot["checkpoint"]["versions"][0]["schema"] == {"x": "Int64"}
    assert sorted(actual, key=lambda item: item["column"]) == sorted(
        [item.to_dict() for item in expected], key=lambda item: item["column"]
    )
    assert worker(runtime).returncode == 0
    assert receiver[0] == [snapshot["outbox"][0]["alert"]]


@pytest.mark.parametrize("action", ["retry", "confirm_delivery", "discard"])
def test_unknown_operator_decision_survives_restart_without_hidden_resend(
    runtime, monkeypatch, tmp_path, receiver, action
):
    identity = register(runtime, monkeypatch, tmp_path)
    assert worker(runtime).returncode == 0
    change(tmp_path, {"x": "Int64"})
    assert worker(runtime).returncode == 0
    assert worker(runtime, crash_after_send=True).returncode == 73
    assert worker(runtime).returncode == 0
    _, snapshot = state(runtime[0], identity)
    alert = snapshot["outbox"][0]["alert"]
    assert snapshot["outbox"][0]["state"] == "unknown" and receiver[0] == [alert]
    request = control_request(
        runtime,
        identity,
        action,
        alert_id=alert["alert_id"],
        evidence_reference="operator-case-001",
        accept_duplicate_risk=action == "retry",
    )
    result = approve_update(runtime, request)
    assert result["execution_status"] == "succeeded", result
    assert invoke(runtime[0], "schema_watch_update", **request) == result
    assert worker(runtime).returncode == 0
    _, snapshot = state(runtime[0], identity)
    item = snapshot["outbox"][0]
    assert (
        item["state"]
        == {"retry": "sent", "confirm_delivery": "confirmed", "discard": "discarded"}[action]
    )
    assert item["decisions"][0]["evidence_reference"] == "operator-case-001"
    assert receiver[0] == [alert] * (2 if action == "retry" else 1)
    assert worker(runtime).returncode == 0
    assert len(receiver[0]) == (2 if action == "retry" else 1)


@pytest.mark.parametrize("failure", ["stale", "busy", "expired", "revoked", "quota"])
def test_approved_watch_updates_fail_closed_and_transactionally(
    runtime, monkeypatch, tmp_path, failure
):
    from truthound.mcp.durable_schema import read, storage, write

    identity = register(runtime, monkeypatch, tmp_path)
    request = control_request(runtime, identity, "pause")
    if failure != "stale":
        store = storage(runtime[0]._session)
        with store.transaction():
            record = read(store, identity)
            if failure == "busy":
                record["state"] = "paused"
                record["busy"] = {"pid": 1, "created": 0, "task": "poll", "index": None}
            elif failure == "expired":
                record["expires_at"] = 0
            elif failure == "quota":
                record["budget"] = len(json.dumps(record)) + 100
            write(store, identity, record)
        request = control_request(
            runtime,
            identity,
            "resume" if failure == "busy" else "retention" if failure == "quota" else "pause",
            **({"retention": {"max_count": 1}} if failure == "quota" else {}),
        )
    plan = invoke(runtime[0], "schema_watch_update", **request)
    assert plan["issues"] == [{"code": "approval_required"}]
    store = ApprovalStore(runtime[1])
    try:
        store.approve(plan["data"]["plan_id"])
    finally:
        store.close()
    if failure == "stale":
        invoke(runtime[0], "schema_watch_disable", watch_id=identity)
    elif failure == "revoked":
        path = runtime[1] / "bindings.json"
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    elif failure == "quota":
        # Inject a storage failure after approval; neither intent nor state may commit.
        monkeypatch.setattr(
            "truthound.mcp.schema_watch_control.prune",
            lambda record: record.update(over_budget="x" * 10000),
        )
    result = invoke(runtime[0], "schema_watch_update", **request)
    assert result["execution_status"] != "succeeded", result
    record = read(storage(runtime[0]._session), identity)
    assert "over_budget" not in record and record.get("retention") is None


@pytest.mark.parametrize("mode", ["all", "any"])
def test_queue_retention_matches_native_and_preserves_unresolved(mode):
    import copy
    from datetime import datetime, timedelta

    from truthound.mcp.schema_watch_control import prune
    from truthound.stores.retention.base import ItemMetadata, PolicyMode
    from truthound.stores.retention.policies import (
        CompositePolicy,
        CountBasedPolicy,
        TimeBasedPolicy,
    )

    now = datetime.now()
    items = [
        ItemMetadata(str(i), "schema-outbox", now - timedelta(minutes=20 - i * 5)) for i in range(4)
    ]
    native = CompositePolicy(
        [TimeBasedPolicy(max_age_minutes=12), CountBasedPolicy(2)], mode=PolicyMode(mode)
    )
    native.prepare_batch(items)
    expected = {item.item_id for item in items if native.should_retain(item)}
    protected = [
        {"state": state, "alert": {"alert_id": state}, "resolved_at": 0}
        for state in ("pending", "sending", "unknown")
    ]
    protected.append({"state": "sent", "alert": {"alert_id": "legacy"}})
    record = {
        "busy": None,
        "retention": {"max_age_minutes": 12, "max_count": 2, "mode": mode},
        "outbox": [
            {
                "state": "sent",
                "alert": {"alert_id": item.item_id},
                "resolved_at": item.created_at.timestamp(),
            }
            for item in items
        ]
        + copy.deepcopy(protected),
    }
    prune(record)
    assert {item["alert"]["alert_id"] for item in record["outbox"]} == expected | {
        item["alert"]["alert_id"] for item in protected
    }
    assert record["pruned_notifications"] == 4 - len(expected)
    saved = copy.deepcopy(record)
    prune(record)
    assert record == saved


def test_approved_retention_runs_after_restart_and_can_be_disabled(runtime, monkeypatch, tmp_path):
    from truthound.mcp.durable_schema import read, storage, write

    identity = register(runtime, monkeypatch, tmp_path)
    assert (
        approve_update(runtime, control_request(runtime, identity, "pause"))["execution_status"]
        == "succeeded"
    )
    request = control_request(runtime, identity, "retention", retention={"max_age_minutes": 1})
    assert approve_update(runtime, request)["execution_status"] == "succeeded"
    store = storage(runtime[0]._session)

    def seed():
        with store.transaction():
            record = read(store, identity)
            record["outbox"] = [
                {"state": status, "alert": {"alert_id": status}, "resolved_at": 0}
                for status in ("sent", "confirmed", "discarded", "unknown", "pending")
            ]
            write(store, identity, record)

    seed()
    assert worker(runtime).returncode == 0
    info, snapshot = state(runtime[0], identity)
    assert info["state"] == "paused" and info["pruned_notifications"] == 3
    assert {item["state"] for item in snapshot["outbox"]} == {"unknown", "pending"}
    assert (
        approve_update(runtime, control_request(runtime, identity, "retention", retention=None))[
            "execution_status"
        ]
        == "succeeded"
    )
    seed()
    assert worker(runtime).returncode == 0
    assert len(state(runtime[0], identity)[1]["outbox"]) == 5


@pytest.mark.parametrize(
    "fields",
    [
        {"action": "retry", "alert_id": "a", "evidence_reference": "case"},
        {"action": "retry", "alert_id": "a", "accept_duplicate_risk": True},
        {
            "action": "confirm_delivery",
            "alert_id": "a",
            "evidence_reference": "case",
            "accept_duplicate_risk": True,
        },
        {"action": "pause", "alert_id": "a"},
        {"action": "retention", "retention": {}},
    ],
)
def test_watch_update_rejects_missing_or_misplaced_consent(fields):
    from pydantic import ValidationError

    from truthound.mcp.schema_watch_control import WatchUpdate

    with pytest.raises(ValidationError):
        WatchUpdate.model_validate(
            {
                "watch_id": "watch_" + "a" * 64,
                "expected_revision": "b" * 64,
                "idempotency_key": "operator-request",
                **fields,
            }
        )


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


@pytest.mark.parametrize("initial", ["expired", "failed"])
def test_expiry_renewal_and_failed_restart_preserve_checkpoint(
    runtime, monkeypatch, tmp_path, initial
):
    from truthound.mcp.durable_schema import read, storage, write

    identity = register(runtime, monkeypatch, tmp_path)
    assert worker(runtime).returncode == 0
    store = storage(runtime[0]._session)
    with store.transaction():
        record = read(store, identity)
        record["state"] = initial
        if initial == "expired":
            record["expires_at"] = 0
        saved = record["checkpoint"]
        write(store, identity, record)
    action = "renew" if initial == "expired" else "restart"
    options = {"lifetime_seconds": 3600} if action == "renew" else {}
    request = control_request(runtime, identity, action, **options)
    result = approve_update(runtime, request)
    assert result["execution_status"] == "succeeded", result
    assert result["data"]["state"] == "paused"
    assert invoke(runtime[0], "schema_watch_update", **request) == result
    assert read(store, identity)["checkpoint"] == saved
    assert worker(runtime).returncode == 0
    assert read(store, identity)["checkpoint"] == saved
    assert (
        approve_update(runtime, control_request(runtime, identity, "resume"))["execution_status"]
        == "succeeded"
    )
    assert worker(runtime).returncode == 0
    assert state(runtime[0], identity)[0]["polls"] == saved["polls"] + 1


def test_definition_deletion_protects_unresolved_outbox_and_replays_receipt(
    runtime, monkeypatch, tmp_path
):
    from truthound.mcp.durable_schema import read, storage, write

    identity = register(runtime, monkeypatch, tmp_path)
    invoke(runtime[0], "schema_watch_disable", watch_id=identity)
    store = storage(runtime[0]._session)
    with store.transaction():
        record = read(store, identity)
        record["outbox"] = [{"state": "unknown", "alert": {"alert_id": "unresolved"}}]
        write(store, identity, record)
    protected = approve_update(
        runtime, control_request(runtime, identity, "delete", accept_history_loss=True)
    )
    assert protected["issues"] == [{"code": "invalid_control_state"}]
    assert read(store, identity)["outbox"][0]["state"] == "unknown"
    request = control_request(
        runtime, identity, "delete", accept_history_loss=True, discard_unresolved=True
    )
    result = approve_update(runtime, request)
    assert result["execution_status"] == "succeeded", result
    assert result["data"]["state"] == "deleted"
    assert invoke(runtime[0], "schema_watch_status", watch_id=identity)["issues"] == [
        {"code": "invalid_handle"}
    ]
    assert store.connection.execute("SELECT COUNT(*) FROM mcp_schema_watches").fetchone()[0] == 0
    assert invoke(runtime[0], "schema_watch_update", **request) == result
    changed = dict(request, discard_unresolved=False)
    assert invoke(runtime[0], "schema_watch_update", **changed)["issues"] == [
        {"code": "idempotency_conflict"}
    ]


def test_history_retention_and_reset_keep_alert_ids_monotonic(
    runtime, monkeypatch, tmp_path, receiver
):
    from truthound.mcp.durable_schema import read, storage, write

    identity = register(runtime, monkeypatch, tmp_path)
    assert worker(runtime).returncode == 0
    for schema in ({"x": "Int64", "y": "String"}, {"x": "Int64"}):
        change(tmp_path, schema)
        assert worker(runtime).returncode == 0
        assert worker(runtime).returncode == 0
    store = storage(runtime[0]._session)
    before = read(store, identity)
    ids = {item["alert"]["alert_id"] for item in before["outbox"]}
    assert len(ids) == 2
    # Native queue retention permits history removal only after its outbox
    # reference disappears; unresolved messages remain protected separately.
    with store.transaction():
        record = read(store, identity)
        record["outbox"] = []
        write(store, identity, record)
    result = approve_update(
        runtime, control_request(runtime, identity, "history_retention", retention={"max_count": 1})
    )
    assert result["execution_status"] == "succeeded", result
    record = read(store, identity)
    assert len(record["checkpoint"]["alerts"]) == len(record["checkpoint"]["versions"]) == 1
    assert record["checkpoint"]["schemas"] == before["checkpoint"]["schemas"]
    assert record["checkpoint"]["events"] == before["checkpoint"]["events"]
    with store.transaction():
        record["state"] = "failed"
        record["checkpoint"]["schema_format"] = 0
        write(store, identity, record)
    denied = approve_update(runtime, control_request(runtime, identity, "restart"))
    assert denied["issues"] == [{"code": "invalid_control_state"}]
    reset = approve_update(
        runtime,
        control_request(
            runtime, identity, "restart", reset_checkpoint=True, accept_history_loss=True
        ),
    )
    assert reset["execution_status"] == "succeeded", reset
    assert (
        approve_update(runtime, control_request(runtime, identity, "resume"))["execution_status"]
        == "succeeded"
    )
    assert worker(runtime).returncode == 0
    change(tmp_path, {"x": "String"})
    assert worker(runtime).returncode == 0
    record = read(store, identity)
    new_ids = {item["alert"]["alert_id"] for item in record["outbox"]}
    assert new_ids and ids.isdisjoint(new_ids)
    assert record["alert_counter"] > before["checkpoint"]["alert_counter"]


def test_delete_receipt_commit_failure_restores_definition(runtime, monkeypatch, tmp_path):
    from truthound.mcp.durable_schema import read, storage

    identity = register(runtime, monkeypatch, tmp_path)
    invoke(runtime[0], "schema_watch_disable", watch_id=identity)
    request = control_request(runtime, identity, "delete", accept_history_loss=True)
    store = storage(runtime[0]._session)
    previous = read(store, identity)
    original = store.finish_in_transaction

    def fail(*args, **kwargs):
        raise OSError("injected receipt publication failure")

    monkeypatch.setattr(store, "finish_in_transaction", fail)
    result = approve_update(runtime, request)
    assert result["execution_status"] != "succeeded"
    assert read(store, identity) == previous
    monkeypatch.setattr(store, "finish_in_transaction", original)
    result = invoke(runtime[0], "schema_watch_update", **request)
    assert result["execution_status"] == "succeeded", result
    path = runtime[1] / "bindings.json"
    path.write_text(path.read_text() + "\n")
    assert invoke(runtime[0], "schema_watch_update", **request)["execution_status"] != "succeeded"


@pytest.mark.parametrize(
    "options",
    [
        {"action": "delete"},
        {"action": "restart", "reset_checkpoint": True},
        {"action": "renew"},
        {"action": "renew", "lifetime_seconds": 2592001},
        {"action": "resume", "lifetime_seconds": 1},
        {"action": "resume", "discard_unresolved": True},
        {"action": "retention", "reset_checkpoint": True, "accept_history_loss": True},
    ],
)
def test_lifecycle_fields_require_explicit_scoped_loss_and_duration(options):
    from pydantic import ValidationError

    from truthound.mcp.schema_watch_control import WatchUpdate

    with pytest.raises(ValidationError):
        WatchUpdate(
            watch_id="watch_" + "a" * 64,
            expected_revision="b" * 64,
            idempotency_key="closed-lifecycle",
            **options,
        )
