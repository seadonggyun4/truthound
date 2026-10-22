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


def test_durable_job_native_validation_restart_result_and_cancel(tmp_path):
    import truthound as th
    from truthound.mcp.operations import _context
    from truthound.validators import get_validator

    data, control = tmp_path / "data", tmp_path / "control"
    data.mkdir()
    control.mkdir(mode=0o700)
    policy = MCPPolicy(
        (data,), profile="core", allow_sensitive_results=True, control_directory=control
    )
    frame = pl.DataFrame({"x": [1, None, 3]})
    frame.write_parquet(data / "input.parquet")
    direct = th.check(
        frame,
        validators=[get_validator("not_null")()],
        result_format="complete",
        context=_context(str(tmp_path / "direct/input.ipc")),
    ).to_dict()
    service = MCPService(policy)
    try:
        asset = call(service, "assets_capture", path=str(data / "input.parquet"))["resource_id"]
        request = {
            "validation": {
                "asset_id": asset,
                "checks": [{"validator": "not_null"}],
                "result_format": "complete",
            },
            "idempotency_key": "durable-job-01",
        }
        pending = invoke(service, "jobs_submit", **request)
        assert pending["issues"] == [{"code": "approval_required"}]
        store = ApprovalStore(control)
        try:
            store.approve(pending["data"]["plan_id"])
        finally:
            store.close()
        queued = call(service, "jobs_submit", **request)
        assert queued["state"] == "queued"
        assert call(service, "jobs_submit", **request) == queued
    finally:
        service.close()
    service = MCPService(policy)
    try:
        job = {"job_id": queued["job_id"]}
        assert call(service, "jobs_status", **job)["state"] == "queued"
        assert (
            call(service, "jobs_status", idempotency_key="durable-job-01")["job_id"]
            == job["job_id"]
        )
        completed = call(service, "jobs_execute", **job)
        assert invoke(service, "models_resume", idempotency_key="durable-job-01")["issues"] == [
            {"code": "invalid_handle"}
        ]
        assert completed["state"] == "succeeded"  # execution, NOT quality verdict
        assert call(service, "jobs_execute", **job) == completed
        saved = call(service, "jobs_result", **job)
        actual = read(service, saved["artifact"]["resource_id"])
        for field in ("checks", "issues", "execution_issues", "row_count", "column_count"):
            assert actual[field] == direct[field], field
    finally:
        service.close()
    service = MCPService(policy)
    try:
        again = call(service, "jobs_result", **job)
        assert again["result_sha256"] == completed["result_sha256"]
        recovered = call(service, "jobs_result", idempotency_key="durable-job-01")
        assert recovered["result_sha256"] == completed["result_sha256"]
        assert read(service, again["artifact"]["resource_id"])["issues"] == direct["issues"]
        assert call(service, "jobs_recover", **job)["state"] == "succeeded"
    finally:
        service.close()


def test_durable_job_crash_lease_is_unknown_not_retryable_success(tmp_path):
    from truthound.mcp.durable_jobs import storage, write
    from truthound.mcp.session import MCPSession

    data, control = tmp_path / "data", tmp_path / "control"
    data.mkdir()
    control.mkdir(mode=0o700)
    policy = MCPPolicy(
        (data,), profile="core", allow_sensitive_results=True, control_directory=control
    )
    service = MCPService(policy)
    try:
        # Fault injection at the persisted lease boundary, before any output
        # commit. A different session must not infer success or repeat execution.
        session = MCPSession(service)
        service._session = session
        store = storage(session)
        import hashlib

        identity = "job_" + "a" * 64
        record = {
            "state": "running",
            "due_at": 0.0,
            "lease_until": 0.0,
            "result_sha256": None,
            "input_sha256": hashlib.sha256(b"input").hexdigest(),
            "policy": session._policy_stamp(),
            "origins": [],
        }
        with store.transaction():
            write(store, identity, record, b"input", None)
    finally:
        service.close()
    service = MCPService(policy)
    try:
        state = call(service, "jobs_recover", job_id=identity)
        assert state["state"] == "outcome_unknown" and state["artifact"] is None
        assert call(service, "jobs_execute", job_id=identity) == state
        assert call(service, "jobs_result", job_id=identity) == state
    finally:
        service.close()


@pytest.mark.parametrize("fault", ["not_due", "cancel", "input_tamper", "runtime_error"])
def test_durable_job_never_promotes_unexecuted_or_failed_work(tmp_path, fault):
    data, control = tmp_path / "data", tmp_path / "control"
    data.mkdir()
    control.mkdir(mode=0o700)
    service = MCPService(
        MCPPolicy((data,), profile="core", allow_sensitive_results=True, control_directory=control)
    )
    try:
        asset = call(
            service,
            "assets_inline",
            columns=[{"name": "x", "dtype": {"kind": "Int64"}, "values": [1, 2, 3]}],
        )["resource_id"]
        check = (
            {"validator": "between", "settings": {"min_value": 2**256}}
            if fault == "runtime_error"
            else {"validator": "not_null"}
        )
        request = {
            "validation": {"asset_id": asset, "checks": [check]},
            "idempotency_key": "job-fault-test-01",
        }
        if fault == "not_due":
            request["schedule"] = {
                "expression": "@daily",
                "after": "2099-01-01T00:00:00+00:00",
                "count": 1,
            }
        pending = invoke(service, "jobs_submit", **request)
        assert pending["issues"] == [{"code": "approval_required"}]
        store = ApprovalStore(control)
        try:
            store.approve(pending["data"]["plan_id"])
        finally:
            store.close()
        queued = call(service, "jobs_submit", **request)
        job = {"job_id": queued["job_id"]}
        if fault == "cancel":
            assert call(service, "jobs_cancel", **job)["state"] == "cancelled"
            assert call(service, "jobs_execute", **job)["state"] == "cancelled"
        elif fault == "not_due":
            assert invoke(service, "jobs_execute", **job)["issues"] == [{"code": "job_not_due"}]
            assert call(service, "jobs_status", **job)["state"] == "queued"
        elif fault == "input_tamper":
            store = ApprovalStore(control)
            try:
                store.connection.execute(
                    "UPDATE validation_jobs SET input=? WHERE identity=?",
                    (b"corrupt", job["job_id"]),
                )
            finally:
                store.close()
            assert invoke(service, "jobs_execute", **job)["issues"] == [
                {"code": "invalid_control_state"}
            ]
        else:
            assert call(service, "jobs_execute", **job)["state"] == "failed"
            assert call(service, "jobs_recover", **job)["state"] == "failed"
            result = call(service, "jobs_result", **job)
            assert read(service, result["artifact"]["resource_id"])["execution_issues"]
    finally:
        service.close()


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
