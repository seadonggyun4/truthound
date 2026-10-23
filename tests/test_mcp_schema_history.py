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
    runtime, strategy, cls, retention
):
    service, frames, assets = runtime
    state = call(service, "schema_history_create", max_versions=retention, strategy=strategy)
    native = InMemorySchemaHistory(max_versions=retention, version_strategy=cls())
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
    from dataclasses import replace

    service, _, assets = runtime
    state = call(service, "schema_history_create")
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
