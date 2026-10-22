"""Actual context state transitions, intent replay and workspace confinement."""

import asyncio
import base64
import json

import polars as pl
import pytest

pytest.importorskip("mcp")

from truthound.context import TruthoundContext
from truthound.mcp.approval import ApprovalStore
from truthound.mcp.policy import MCPPolicy
from truthound.mcp.service import MCPService
from truthound.mcp.tables import decode

pytestmark = pytest.mark.contract


def invoke(service, name, **arguments):
    return asyncio.run(service.call_operation_async(name, arguments))


def call(service, name, **arguments):
    value = invoke(service, name, **arguments)
    assert value["execution_status"] == "succeeded", value
    return value["data"]


def read(service, identity):
    chunks, offset = [], 0
    while offset is not None:
        chunk = call(service, "artifacts_read", resource_id=identity, offset=offset)
        chunks.append(base64.b64decode(chunk["base64"]))
        offset = chunk["next_offset"]
    return decode(json.loads(b"".join(chunks)))


def perform(service, control, context, operation, key):
    request = {"context_id": context, "operation": operation, "idempotency_key": key}
    pending = invoke(service, "contexts_execute", **request)
    assert pending["issues"] == [{"code": "approval_required"}], pending
    store = ApprovalStore(control)
    try:
        assert (
            operation.get("value", "not-a-value")
            not in store.review(pending["data"]["plan_id"])["request"].values()
        )
        store.approve(pending["data"]["plan_id"])
    finally:
        store.close()
    done = call(service, "contexts_execute", **request)
    assert not done["replayed"]
    replay = call(service, "contexts_execute", **request)
    assert (
        replay["replayed"] and replay["artifact"]["resource_id"] == done["artifact"]["resource_id"]
    )
    return read(service, done["artifact"]["resource_id"])["result"]


def test_context_native_baseline_applied_suite_metric_history_and_artifact_lifecycle(tmp_path):
    from truthound._applied_suite import AppliedSuiteRecord
    from truthound.core.results import ValidationRunResult

    data, workspace, control = [tmp_path / name for name in ("data", "workspace", "control")]
    for path in (data, workspace, control):
        path.mkdir(mode=0o700)
    service = MCPService(
        MCPPolicy(
            (data,),
            profile="core",
            workspace_directories=(workspace,),
            control_directory=control,
            allow_sensitive_results=True,
        )
    )
    try:
        frame = pl.DataFrame({"x": [1, 2, 2]})
        asset = call(
            service,
            "assets_inline",
            columns=[{"name": "x", "dtype": {"kind": "Int64"}, "values": [1, 2, 2]}],
        )["resource_id"]
        context = call(
            service, "contexts_open", path=str(workspace), config={"max_metric_history_entries": 2}
        )["context_id"]
        assert not (workspace / ".truthound").exists()
        direct = TruthoundContext(tmp_path / "direct")

        def operation(action, intent_key, **fields):
            return perform(service, control, context, {"action": action, **fields}, intent_key)

        key = operation("resolve_source_key", "source-key-01", asset_id=asset)
        assert key == direct.resolve_source_key(data=frame)
        assert operation(
            "resolve_fingerprint", "fingerprint-01", asset_id=asset
        ) == direct.resolve_fingerprint(data=frame)
        tracked = operation("track_asset", "track-01", asset_id=asset)
        assert tracked["fingerprint"] == direct.track_asset(data=frame)["fingerprint"]
        first = operation(
            "get_or_create_schema_baseline", "baseline-01", asset_id=asset, infer_constraints=True
        )
        expected, created = direct.get_or_create_schema_baseline(data=frame)
        assert first == {"schema": expected.to_dict(), "created": created}
        assert (
            operation("get_or_create_schema_baseline", "baseline-02", asset_id=asset)["created"]
            is False
        )
        for i in range(3):
            operation(
                "metric_append_history", f"metric-history-{i}", key="measurements", value={"n": i}
            )
        assert operation("metric_get", "metric-read-01", key="measurements") == [{"n": 1}, {"n": 2}]
        operation(
            "metric_put",
            "metric-put-01",
            key="settings",
            value={"$type": "literal-data", "private_fixture": "synthetic"},
        )
        assert operation("metric_get", "metric-read-02", key="settings") == {
            "$type": "literal-data",
            "private_fixture": "synthetic",
        }
        record = {
            "source_key": key,
            "proposal_id": "reviewed-local-proposal",
            "diff_hash": "a" * 64,
            "applied_by": {"actor_id": "metadata-not-consent", "actor_name": "Fixture"},
            "applied_at": "2026-10-21T00:00:00",
            "checks": [
                {
                    "check_key": "required-x",
                    "validator_name": "null",
                    "category": "completeness",
                    "columns": ["x"],
                    "params": {},
                    "rationale": "Required data",
                }
            ],
            "effective_suite_snapshot": {"name": "approved-suite", "checks": []},
        }
        operation("write_applied_suite", "applied-write-01", record=record)
        direct.write_applied_suite(AppliedSuiteRecord.from_dict(record))
        assert (
            operation("read_applied_suite", "applied-read-01", source_key=key)
            == direct.read_applied_suite(source_key=key).to_dict()
        )
        run = call(service, "assets_validate", asset_id=asset, checks=[{"validator": "null"}])[
            "artifact"
        ]["resource_id"]
        result = operation("persist_run", "persist-run-01", resource_id=run)
        assert result["relative_path"].startswith("runs/")
        assert (
            json.loads(
                (workspace / ".truthound" / result["relative_path"]).read_text(encoding="utf-8")
            )["checks"]
            == ValidationRunResult.from_dict(read(service, run)).to_dict()["checks"]
        )
        docs = operation("persist_docs", "persist-docs-01", resource_id=run)
        assert (
            "<html"
            in (workspace / ".truthound" / docs["relative_path"])
            .read_text(encoding="utf-8")
            .lower()
        )
        assert operation("list_runs", "list-runs-01")["total"] == 1
        assert operation("list_docs", "list-docs-01")["total"] == 1
        call(service, "resources_release", resource_id=context)
        assert (workspace / ".truthound" / docs["relative_path"]).exists()
    finally:
        service.close()


@pytest.mark.fault
def test_context_index_cannot_grant_paths_outside_approved_workspace(tmp_path):
    data, workspace, control = [tmp_path / name for name in ("data", "workspace", "control")]
    for path in (data, workspace, control):
        path.mkdir(mode=0o700)
    source = workspace / ".truthound/suites"
    source.mkdir(parents=True)
    (source / "index.json").write_text(
        json.dumps({"dataframe": {"suite_file": "../../../../ungranted.json"}})
    )
    service = MCPService(
        MCPPolicy(
            (data,), profile="core", workspace_directories=(workspace,), control_directory=control
        )
    )
    try:
        response = invoke(service, "contexts_open", path=str(workspace))
        assert response["issues"] == [{"code": "invalid_workspace_reference"}]
        assert not (workspace / ".truthound/config.yaml").exists()
    finally:
        service.close()
