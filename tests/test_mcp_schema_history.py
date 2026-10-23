"""Real resident history must preserve native version and lifecycle semantics."""

import asyncio
import base64
import json
from datetime import datetime

import polars as pl
import pytest

pytest.importorskip("mcp")

from truthound.mcp.policy import MCPPolicy
from truthound.mcp.service import MCPService
from truthound.mcp.tables import decode
from truthound.profiler.evolution.history import (
    IncrementalVersionStrategy,
    InMemorySchemaHistory,
    SemanticVersionStrategy,
)

pytestmark = pytest.mark.contract


def semantics(value):
    """Only native clocks/UUIDs differ; retain all computed content and relationships."""
    if isinstance(value, list):
        return [semantics(item) for item in value]
    if isinstance(value, dict):
        for key in ("timestamp", "created_at", "detected_at", "acknowledged_at", "resolved_at"):
            if value.get(key):
                datetime.fromisoformat(value[key])
        return {
            key: semantics(item)
            for key, item in value.items()
            if key
            not in {"timestamp", "created_at", "detected_at", "acknowledged_at", "resolved_at"}
        }
    return value


class HistoryClient:
    def __init__(self, service, state=None):
        self.service = service
        self.state = state or call(
            service, "schema_history_create", backend="file", max_versions=10
        )

    def execute(self, action, **options):
        self.state = call(
            self.service,
            "schema_history_execute",
            history_id=self.state["history"]["resource_id"],
            expected_revision=self.state["revision"],
            command={"action": action, **options},
        )
        return artifact(self.service, self.state["artifact"]["resource_id"])


def approve(service, operation, request):
    from truthound.mcp.approval import ApprovalStore

    pending = invoke(service, operation, **request)
    assert pending["issues"] == [{"code": "approval_required"}], pending
    store = ApprovalStore(service.policy.control_directory)
    try:
        store.approve(pending["data"]["plan_id"])
    finally:
        store.close()
    return call(service, operation, **request)


@pytest.mark.parametrize("backend", ["memory", "file"])
def test_watcher_and_local_alert_lifecycle_native_equivalence(runtime, backend):
    from truthound.profiler.evolution.breaking_alerts import BreakingChangeAlertManager
    from truthound.profiler.evolution.history import SchemaHistory
    from truthound.profiler.evolution.watcher import (
        AlertingEventHandler,
        DictSchemaSource,
        HistoryEventHandler,
        SchemaWatcher,
    )

    service, frames, assets = runtime
    client = HistoryClient(service, call(service, "schema_history_create", backend=backend))
    native_history = SchemaHistory(InMemorySchemaHistory())
    native_alerts = BreakingChangeAlertManager()
    watcher = SchemaWatcher(history=native_history)
    watcher.add_handler(HistoryEventHandler(native_history))
    watcher.add_handler(AlertingEventHandler(native_alerts))
    source = DictSchemaSource({k: str(v) for k, v in frames[0].schema.items()}, name="orders")
    watcher.add_source(source)
    assert (
        client.execute("watch_add", name="orders", asset_id=assets[0])
        == watcher.get_source_status()
    )
    for i in (0, 1, 2, 2):
        source.update({k: str(v) for k, v in frames[i].schema.items()})
        expected = watcher.check_now()
        actual = client.execute("watch_check", name="orders", asset_id=assets[i])
        assert semantics(actual) == semantics([event.to_dict() for event in expected])
        assert client.execute("count") == native_history.version_count
        assert client.execute("watch_status") == watcher.get_source_status()
        assert client.execute("alert_stats") == native_alerts.get_stats()
        assert semantics(client.execute("alerts")) == semantics(
            [v.to_dict() for v in native_alerts.get_alert_history()]
        )
    alert = native_alerts.get_alert_history()[0]
    assert client.execute(
        "alert_acknowledge", alert_id=alert.alert_id
    ) == native_alerts.acknowledge_alert(alert.alert_id)
    assert semantics(client.execute("alert_get", alert_id=alert.alert_id)) == semantics(
        alert.to_dict()
    )
    assert client.execute("alert_resolve", alert_id=alert.alert_id) == native_alerts.resolve_alert(
        alert.alert_id
    )
    assert semantics(client.execute("alert_get", alert_id=alert.alert_id)) == semantics(
        alert.to_dict()
    )
    assert client.execute("watch_remove", name="orders") == watcher.remove_source("orders")
    assert client.execute("watch_status") == {}


@pytest.fixture
def durable_history(tmp_path):
    data, control = tmp_path / "data", tmp_path / "control"
    data.mkdir()
    control.mkdir(mode=0o700)
    policy = MCPPolicy(
        (data,), profile="core", control_directory=control, allow_sensitive_results=True
    )
    service = MCPService(policy)
    try:
        assets = []
        for i, frame in enumerate((pl.DataFrame({"x": [1], "y": ["a"]}), pl.DataFrame({"x": [2]}))):
            path = data / f"{i}.parquet"
            frame.write_parquet(path)
            assets.append(call(service, "assets_capture", path=str(path))["resource_id"])
        client = HistoryClient(service)
        first = client.execute("save", asset_id=assets[0])
        client.execute("watch_add", name="orders", asset_id=assets[0])
        client.execute("watch_check", name="orders", asset_id=assets[1])
        client.execute("alert_acknowledge", alert_id="ALERT-000001")
        request = dict(
            history_id=client.state["history"]["resource_id"],
            expected_revision=client.state["revision"],
            idempotency_key="persist-file-history",
        )
        checkpoint = approve(service, "schema_history_persist", request)
        assert call(service, "schema_history_persist", **request) == checkpoint
        yield service, client, checkpoint, first, policy
    finally:
        service.close()


def test_file_history_exact_restart_recovery_independent_writers_and_deletion(durable_history):
    service, client, checkpoint, first, policy = durable_history
    expected_versions, expected_alerts = client.execute("list"), client.execute("alerts")
    old = client.state["history"]["resource_id"]
    service.close()
    resumed = MCPService(policy)
    try:
        restored = HistoryClient(
            resumed, call(resumed, "schema_history_resume", snapshot_id=checkpoint["snapshot_id"])
        )
        assert restored.state["revision"] == 0 and restored.state["history"]["resource_id"] != old
        assert restored.execute("list") == expected_versions
        assert restored.execute("alerts") == expected_alerts
        assert restored.execute("watch_status") == {}  # Never restart subscriptions silently.
        assert restored.execute("delete", version_id=first["version_id"])
        second = HistoryClient(
            resumed, call(resumed, "schema_history_resume", idempotency_key="persist-file-history")
        )
        assert second.execute("list") == expected_versions  # No shared mutable writer.
        preview = call(resumed, "storage_retention_preview", kind="histories", max_count=1)
        assert preview["preserved"] == [checkpoint["snapshot_id"]]
        approve(
            resumed,
            "storage_delete",
            dict(identities=[checkpoint["snapshot_id"]], idempotency_key="delete-file-history"),
        )
        denied = invoke(resumed, "schema_history_resume", snapshot_id=checkpoint["snapshot_id"])
        assert denied["issues"] == [{"code": "invalid_handle"}]
        denied = invoke(
            resumed,
            "schema_history_execute",
            history_id=second.state["history"]["resource_id"],
            expected_revision=second.state["revision"],
            command={"action": "latest"},
        )
        assert denied["execution_status"] == "failed"
    finally:
        resumed.close()


@pytest.mark.parametrize("failure", ["policy", "payload", "signature"])
def test_durable_history_rejects_revocation_and_tampering(durable_history, failure):
    from dataclasses import replace

    from truthound.mcp.durable_history import storage

    service, client, checkpoint, first, policy = durable_history
    if failure == "policy":
        service.policy = replace(policy, max_items=policy.max_items + 1)
    else:
        store = storage(service._session)
        field, value = ("payload", b"{}") if failure == "payload" else ("signature", "corrupt")
        store.connection.execute(
            f"UPDATE history_snapshots SET {field}=? WHERE identity=?",
            (value, checkpoint["snapshot_id"]),
        )
    response = invoke(service, "schema_history_resume", snapshot_id=checkpoint["snapshot_id"])
    assert response["execution_status"] == "failed"
    assert response["issues"] == [
        {"code": "access_denied" if failure == "policy" else "invalid_control_state"}
    ]


def test_history_automatic_retention_reuses_native_selection_after_restart(durable_history):
    from truthound.mcp import automation

    service, client, checkpoint, _, policy = durable_history
    second = approve(
        service,
        "schema_history_persist",
        {
            "history_id": client.state["history"]["resource_id"],
            "expected_revision": client.state["revision"],
            "idempotency_key": "history-second-snapshot",
        },
    )
    preview = call(service, "storage_retention_preview", kind="histories", max_count=1)
    assert preview["candidates"] == [checkpoint["snapshot_id"]]
    saved = approve(
        service,
        "storage_policy_register",
        {
            "kind": "histories",
            "max_count": 1,
            "idempotency_key": "history-retention-policy",
        },
    )
    service.close()
    resumed = MCPService(policy)
    try:
        call(resumed, "storage_policy_status", control_id=saved["control_id"])
        asyncio.run(automation.tick(resumed._session))
        assert (
            call(resumed, "storage_policy_status", control_id=saved["control_id"])["deleted"] == 1
        )
        assert (
            invoke(resumed, "schema_history_resume", snapshot_id=checkpoint["snapshot_id"])[
                "execution_status"
            ]
            == "failed"
        )
        assert (
            call(resumed, "schema_history_resume", snapshot_id=second["snapshot_id"])["revision"]
            == 0
        )
    finally:
        resumed.close()


def test_history_failed_durable_commit_is_unknown_not_retryable_success(
    durable_history, monkeypatch
):
    from truthound.mcp.approval import ApprovalStore
    from truthound.mcp.durable_history import storage

    service, client, checkpoint, _, _ = durable_history
    request = {
        "history_id": client.state["history"]["resource_id"],
        "expected_revision": client.state["revision"],
        "idempotency_key": "history-failed-commit",
    }
    pending = invoke(service, "schema_history_persist", **request)
    plan_id = pending["data"]["plan_id"]
    store = storage(service._session)
    store.approve(plan_id)
    original = ApprovalStore.finish_in_transaction

    def interrupted(self, *args, **kwargs):
        original(self, *args, **kwargs)
        raise OSError("synthetic interrupted commit")

    monkeypatch.setattr(ApprovalStore, "finish_in_transaction", interrupted)
    failed = invoke(service, "schema_history_persist", **request)
    assert failed["issues"] == [{"code": "outcome_unknown"}]
    assert store.connection.execute("SELECT COUNT(*) FROM history_snapshots").fetchone()[0] == 1
    replay = invoke(service, "schema_history_persist", **request)
    assert replay["execution_status"] == "outcome_unknown"
    assert replay["issues"] == [{"code": "outcome_unknown"}]


@pytest.mark.parametrize("action", ["get", "delete", "diff", "rollback"])
def test_file_history_rejects_path_escape_before_native_access(runtime, action):
    service, _, _ = runtime
    client = HistoryClient(service)
    key = "source_id" if action == "diff" else "version_id"
    result = invoke(
        service,
        "schema_history_execute",
        history_id=client.state["history"]["resource_id"],
        expected_revision=0,
        command={"action": action, key: "../../outside"},
    )
    assert result["execution_status"] == "failed"
    assert service._session._histories.workers == {}


def test_history_snapshot_closed_files_and_native_io_failure(tmp_path):
    import logging

    from truthound.mcp.encoding import encode
    from truthound.mcp.history_files import restore
    from truthound.mcp.policy import MCPError
    from truthound.mcp.schema_history import HistoryCreate
    from truthound.mcp.schema_history_worker import FailClosed
    from truthound.profiler.evolution.history import FileSchemaHistory

    config = HistoryCreate(backend="file")
    path = tmp_path / "snapshot.json"
    path.write_text(
        json.dumps(
            encode(
                {
                    "format": 1,
                    "configuration": config.model_dump(),
                    "files": {"index.json": "[]", "alerts.json": "[]", "../escape": "secret"},
                }
            )
        )
    )
    with pytest.raises(MCPError):
        restore(path, tmp_path / "restore", config)
    assert not (tmp_path / "escape").exists()
    native = FileSchemaHistory(tmp_path / "history")
    native._index_file.mkdir()  # Native _save_index normally logs and silently succeeds.
    logger = logging.getLogger("truthound.profiler.evolution.history")
    handler = FailClosed()
    handler.setLevel(logging.WARNING)
    logger.addHandler(handler)
    try:
        with pytest.raises(MCPError):
            native.save_version({"id": "Int64"})
    finally:
        logger.removeHandler(handler)


def invoke(service, operation, **arguments):
    return asyncio.run(service.call_operation_async(operation, arguments))


def call(service, operation, **arguments):
    result = invoke(service, operation, **arguments)
    assert result["execution_status"] == "succeeded", result
    return result["data"]


def artifact(service, identity):
    chunks, offset = [], 0
    while offset is not None:
        data = call(service, "artifacts_read", resource_id=identity, offset=offset)
        chunks.append(base64.b64decode(data["base64"]))
        offset = data["next_offset"]
    return decode(json.loads(b"".join(chunks)))


@pytest.fixture
def runtime(tmp_path):
    service = MCPService(MCPPolicy((tmp_path,), profile="core", allow_sensitive_results=True))
    frames = [
        pl.DataFrame({"x": [1]}),
        pl.DataFrame({"x": [1], "y": ["a"]}),
        pl.DataFrame({"y": ["b"]}),
    ]
    assets = []
    for i, frame in enumerate(frames):
        path = tmp_path / f"history-{i}.parquet"
        frame.write_parquet(path)
        assets.append(call(service, "assets_capture", path=str(path))["resource_id"])
    try:
        yield service, frames, assets
    finally:
        service.close()


@pytest.mark.parametrize(
    "strategy,cls",
    [("semantic", SemanticVersionStrategy), ("incremental", IncrementalVersionStrategy)],
)
@pytest.mark.parametrize("retention", [2, 10])
def test_native_history_versions_queries_diff_rollback_and_retention(
    runtime, strategy, cls, retention, tmp_path
):
    assert_native_history(runtime, strategy, cls, retention, "memory", tmp_path)


@pytest.mark.parametrize(
    "strategy,cls",
    [("semantic", SemanticVersionStrategy), ("incremental", IncrementalVersionStrategy)],
)
@pytest.mark.parametrize("retention", [2, 10])
def test_native_file_history_versions_queries_diff_rollback_and_retention(
    runtime, strategy, cls, retention, tmp_path
):
    assert_native_history(runtime, strategy, cls, retention, "file", tmp_path)


def assert_native_history(runtime, strategy, cls, retention, backend, tmp_path):
    service, frames, assets = runtime
    state = call(
        service, "schema_history_create", max_versions=retention, strategy=strategy, backend=backend
    )
    from truthound.profiler.evolution.history import FileSchemaHistory

    native = (
        FileSchemaHistory(
            tmp_path / "native-history", max_versions=retention, version_strategy=cls()
        )
        if backend == "file"
        else InMemorySchemaHistory(max_versions=retention, version_strategy=cls())
    )
    aliases = {}

    def comparable(value):
        if isinstance(value, list):
            return [comparable(v) for v in value]
        if isinstance(value, dict):
            for key in ("created_at", "computed_at", "detected_at"):
                if key in value:
                    assert datetime.fromisoformat(value[key]).tzinfo is None
            return {
                k: comparable(v)
                for k, v in value.items()
                if k not in {"created_at", "computed_at", "detected_at"}
            }
        return aliases.get(value, value) if isinstance(value, str) else value

    def execute(command, expected, creates=False):
        nonlocal state
        before = state["revision"]
        state = call(
            service,
            "schema_history_execute",
            history_id=state["history"]["resource_id"],
            expected_revision=before,
            command=command,
        )
        actual = artifact(service, state["artifact"]["resource_id"])
        wanted = (
            [v.to_dict() for v in expected]
            if isinstance(expected, list)
            else expected.to_dict()
            if hasattr(expected, "to_dict")
            else expected
        )
        if creates:
            aliases[actual["version_id"]] = wanted["version_id"]
        assert comparable(actual) == comparable(wanted)
        assert state["revision"] == before + int(
            command["action"] in {"save", "delete", "rollback"}
        )
        return actual

    execute({"action": "latest"}, None)
    versions = []
    for index, (frame, asset) in enumerate(zip(frames, assets, strict=True)):
        kwargs = {"version": "explicit"} if index == 0 else {}
        metadata = {"author": "local-test", "tags": ["schema", index]}
        expected = native.save_version(frame.schema, metadata=metadata, **kwargs)
        actual = execute(
            {"action": "save", "asset_id": asset, "metadata": metadata, **kwargs},
            expected,
            creates=True,
        )
        versions.append((actual["version_id"], expected.version_id))
    execute({"action": "get", "version_id": versions[0][0]}, native.get_version(versions[0][1]))
    execute(
        {"action": "by_version", "version": "explicit"}, native.get_version_by_string("explicit")
    )
    execute({"action": "list", "limit": 0}, native.list_versions(limit=0))
    execute(
        {
            "action": "list",
            "limit": 1,
            "since": "2000-01-01T00:00:00",
            "until": "2100-01-01T00:00:00",
        },
        native.list_versions(limit=1, since=datetime(2000, 1, 1), until=datetime(2100, 1, 1)),
    )
    execute({"action": "baseline"}, native.get_baseline())
    execute(
        {"action": "diff", "source_id": versions[1][0], "target_id": versions[2][0]},
        native.diff(versions[1][1], versions[2][1]),
    )
    execute(
        {"action": "rollback", "version_id": versions[1][0]},
        native.rollback_to(versions[1][1]),
        creates=True,
    )
    execute({"action": "latest"}, native.get_latest())
    execute(
        {"action": "delete", "version_id": versions[2][0]}, native.delete_version(versions[2][1])
    )
    for action in ("get", "delete", "rollback"):
        execute(
            {"action": action, "version_id": "missing"},
            {
                "get": native.get_version,
                "delete": native.delete_version,
                "rollback": native.rollback_to,
            }[action]("missing"),
        )
    execute({"action": "diff", "source_id": "missing", "target_id": versions[2][0]}, None)


@pytest.mark.parametrize(
    "reason", ["release", "input_release", "expiry", "policy", "close", "worker_exit"]
)
def test_history_ownership_revocation_reaps_worker_and_artifacts(runtime, reason):
    assert_history_ownership(runtime, reason, "memory")


@pytest.mark.parametrize(
    "reason", ["release", "input_release", "expiry", "policy", "close", "worker_exit"]
)
def test_file_history_ownership_revocation_reaps_worker_and_artifacts(runtime, reason):
    assert_history_ownership(runtime, reason, "file")


def assert_history_ownership(runtime, reason, backend):
    from dataclasses import replace

    service, _, assets = runtime
    state = call(service, "schema_history_create", backend=backend)
    identity = state["history"]["resource_id"]
    state = call(
        service,
        "schema_history_execute",
        history_id=identity,
        expected_revision=0,
        command={"action": "save", "asset_id": assets[0]},
    )
    session = service._session
    worker = session._histories.workers[identity]
    if reason == "release":
        call(service, "resources_release", resource_id=identity)
    elif reason == "input_release":
        call(service, "resources_release", resource_id=assets[0])
    elif reason == "expiry":
        expires = session.resources.get(identity).expires
        session.resources.clock = lambda: expires + 1
    elif reason == "policy":
        other = service.policy.allowed_directories[0] / "other"
        other.mkdir()
        service.policy = replace(service.policy, allowed_directories=(other,))
    elif reason == "close":
        service.close()
    else:
        worker.close()
    result = invoke(
        service,
        "schema_history_execute",
        history_id=identity,
        expected_revision=1,
        command={"action": "latest"},
    )
    assert result["execution_status"] == "failed"
    assert (
        invoke(service, "artifacts_read", resource_id=state["artifact"]["resource_id"])[
            "execution_status"
        ]
        == "failed"
    )
    assert worker.closed and worker.process.poll() is not None
    from pathlib import Path

    assert not Path(worker.directory.name).exists()


def test_native_corrupt_file_is_failure_not_missing_version(runtime):
    from pathlib import Path

    service, _, assets = runtime
    client = HistoryClient(service)
    version = client.execute("save", asset_id=assets[0])
    identity = client.state["history"]["resource_id"]
    worker = service._session._histories.workers[identity]
    path = Path(worker.directory.name) / "history" / "versions" / f"{version['version_id']}.json"
    path.write_text("{corrupt", encoding="utf-8")
    failed = invoke(
        service,
        "schema_history_execute",
        history_id=identity,
        expected_revision=1,
        command={"action": "get", "version_id": version["version_id"]},
    )
    assert failed["issues"] == [{"code": "history_execution_failed"}]
    assert worker.closed and not Path(worker.directory.name).exists()


def test_watcher_limits_and_unregistered_source_fail_closed(runtime):
    service, _, assets = runtime
    client = HistoryClient(
        service, call(service, "schema_history_create", backend="file", max_versions=1)
    )
    client.execute("watch_add", name="orders", asset_id=assets[0])
    client.execute("watch_check", name="orders", asset_id=assets[2])
    assert client.execute("alert_stats")["total"] == 1
    failed = invoke(
        service,
        "schema_history_execute",
        history_id=client.state["history"]["resource_id"],
        expected_revision=client.state["revision"],
        command={"action": "watch_check", "name": "orders", "asset_id": assets[0]},
    )
    assert failed["execution_status"] == "failed"
    assert service._session._histories.workers == {}


def test_history_revision_conflict_and_closed_wire_do_not_mutate(runtime):
    service, _, assets = runtime
    state = call(service, "schema_history_create")
    identity = state["history"]["resource_id"]
    args = {
        "history_id": identity,
        "expected_revision": 0,
        "command": {"action": "save", "asset_id": assets[0]},
    }
    state = call(service, "schema_history_execute", **args)
    assert invoke(service, "schema_history_execute", **args)["issues"] == [
        {"code": "revision_conflict"}
    ]
    for command in (
        {"action": "save", "asset_id": assets[0], "base_path": "/private"},
        {"action": "import", "module": "os"},
    ):
        assert (
            invoke(
                service,
                "schema_history_execute",
                history_id=identity,
                expected_revision=1,
                command=command,
            )["execution_status"]
            == "failed"
        )
    state = call(
        service,
        "schema_history_execute",
        history_id=identity,
        expected_revision=1,
        command={"action": "list"},
    )
    assert len(artifact(service, state["artifact"]["resource_id"])) == 1


def test_history_core_datetime_error_invalidates_only_owned_state(runtime):
    service, _, assets = runtime
    state = call(service, "schema_history_create")
    identity = state["history"]["resource_id"]
    call(
        service,
        "schema_history_execute",
        history_id=identity,
        expected_revision=0,
        command={"action": "save", "asset_id": assets[0]},
    )
    worker = service._session._histories.workers[identity]
    result = invoke(
        service,
        "schema_history_execute",
        history_id=identity,
        expected_revision=1,
        command={"action": "list", "since": "2000-01-01T00:00:00Z"},
    )
    assert result["execution_status"] == "failed"
    assert worker.closed
    assert call(service, "resources_describe", resource_id=assets[0])["resource_id"] == assets[0]
    native = InMemorySchemaHistory()
    native.save_version({"x": pl.Int64})
    with pytest.raises(TypeError):
        native.list_versions(since=datetime.fromisoformat("2000-01-01T00:00:00+00:00"))


@pytest.mark.parametrize("strategy", ["timestamp", "git_like"])
def test_history_nondeterministic_native_versions_are_preserved_on_query(runtime, strategy):
    import re

    from truthound.profiler.evolution.history import (
        GitLikeVersionStrategy,
        TimestampVersionStrategy,
    )

    service, frames, assets = runtime
    cls = TimestampVersionStrategy if strategy == "timestamp" else GitLikeVersionStrategy
    native = InMemorySchemaHistory(version_strategy=cls())
    state = call(service, "schema_history_create", strategy=strategy)
    identity = state["history"]["resource_id"]
    state = call(
        service,
        "schema_history_execute",
        history_id=identity,
        expected_revision=0,
        command={"action": "save", "asset_id": assets[0]},
    )
    actual = artifact(service, state["artifact"]["resource_id"])
    expected = native.save_version(frames[0].schema).to_dict()
    pattern = r"\d{8}\.\d{6}" if strategy == "timestamp" else r"[0-9a-f]{8}"
    assert re.fullmatch(pattern, actual["version"]) and re.fullmatch(pattern, expected["version"])
    assert actual["schema"] == expected["schema"]
    state = call(
        service,
        "schema_history_execute",
        history_id=identity,
        expected_revision=1,
        command={"action": "by_version", "version": actual["version"]},
    )
    assert artifact(service, state["artifact"]["resource_id"]) == actual


def test_history_output_budget_failure_reaps_ambiguous_mutation(runtime):
    from dataclasses import replace

    service, _, assets = runtime
    state = call(service, "schema_history_create")
    identity = state["history"]["resource_id"]
    worker = service._session._histories.workers[identity]
    used = sum(item.size for item in service._session.resources._entries.values())
    service.policy = replace(service.policy, max_session_bytes=used + 1)
    result = invoke(
        service,
        "schema_history_execute",
        history_id=identity,
        expected_revision=0,
        command={"action": "save", "asset_id": assets[0]},
    )
    assert result["execution_status"] == "failed"
    assert worker.closed and worker.process.poll() is not None
    assert identity not in service._session._histories.workers


def test_history_foreign_handle_and_invalid_config_are_rejected(runtime):
    service, _, assets = runtime
    for options in (
        {"strategy": "os.system"},
        {"max_versions": 0},
        {"max_versions": 1001},
        {"max_versions": 101},
        {"base_path": "/tmp/unapproved"},
    ):
        assert invoke(service, "schema_history_create", **options)["execution_status"] == "failed"
    assert (
        invoke(
            service,
            "schema_history_execute",
            history_id=assets[0],
            expected_revision=0,
            command={"action": "latest"},
        )["execution_status"]
        == "failed"
    )


def test_history_manager_count_breaking_and_version_string_rollback(runtime):
    from truthound.profiler.evolution.history import SchemaHistory

    service, frames, assets = runtime
    native = SchemaHistory.create()
    state = call(service, "schema_history_create")
    identity = state["history"]["resource_id"]
    for i in (0, 2):
        revision = state["revision"]
        state = call(
            service,
            "schema_history_execute",
            history_id=identity,
            expected_revision=revision,
            command={"action": "save", "asset_id": assets[i], "version": f"v{i}"},
        )
        native.save(frames[i].schema, version=f"v{i}")
    for command, expected in (
        ({"action": "count"}, native.version_count),
        ({"action": "breaking_since", "version_id": "v0"}, native.has_breaking_changes_since("v0")),
        (
            {"action": "breaking_since", "version_id": "missing"},
            native.has_breaking_changes_since("missing"),
        ),
    ):
        result = call(
            service,
            "schema_history_execute",
            history_id=identity,
            expected_revision=2,
            command=command,
        )
        assert artifact(service, result["artifact"]["resource_id"]) == expected
    diff = call(
        service,
        "schema_history_execute",
        history_id=identity,
        expected_revision=2,
        command={"action": "diff", "source_id": "v0"},
    )
    actual = artifact(service, diff["artifact"]["resource_id"])
    assert actual["changes"] == native.diff("v0").to_dict()["changes"]
    rolled = call(
        service,
        "schema_history_execute",
        history_id=identity,
        expected_revision=2,
        command={"action": "rollback", "version_id": "v0", "reason": "reviewed rollback"},
    )
    actual = artifact(service, rolled["artifact"]["resource_id"])
    expected = native.rollback("v0", reason="reviewed rollback").to_dict()
    assert actual["schema"] == expected["schema"]
    assert actual["changes_from_parent"] == expected["changes_from_parent"]
    assert actual["metadata"]["rollback_reason"] == expected["metadata"]["rollback_reason"]
    assert rolled["revision"] == 3
