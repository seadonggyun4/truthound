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


def test_repeated_occurrences_restart_independent_state_and_native_retention(tmp_path):
    from datetime import datetime

    from truthound.scheduling import CronExpression
    from truthound.stores.retention.base import ItemMetadata
    from truthound.stores.retention.policies import CountBasedPolicy

    data, control = tmp_path / "data", tmp_path / "control"
    data.mkdir()
    control.mkdir(mode=0o700)
    pl.DataFrame({"x": [1, None]}).write_parquet(data / "input.parquet")
    policy = MCPPolicy(
        (data,), profile="core", allow_sensitive_results=True, control_directory=control
    )
    service = MCPService(policy)

    def approved(name, request):
        pending = invoke(service, name, **request)
        assert pending["issues"] == [{"code": "approval_required"}]
        store = ApprovalStore(control)
        try:
            store.approve(pending["data"]["plan_id"])
        finally:
            store.close()
        return call(service, name, **request)

    try:
        asset = call(service, "assets_capture", path=str(data / "input.parquet"))["resource_id"]
        request = dict(
            validation={"asset_id": asset, "checks": [{"validator": "not_null"}]},
            schedule={"expression": "0 * * * *", "after": "2020-01-01T00:00:00+00:00", "count": 3},
            idempotency_key="repeated-test-01",
        )
        queued = approved("jobs_submit", request)
        assert call(service, "jobs_submit", **request) == queued
        identities = queued["occurrence_ids"]
        assert len(set(identities)) == 3
        service.close()
        service = MCPService(policy)
        expected = CronExpression.parse("0 * * * *").next_n(
            3, datetime.fromisoformat("2020-01-01T00:00:00+00:00")
        )
        for identity, due in zip(identities, expected, strict=True):
            assert call(service, "jobs_status", job_id=identity)["due_at"] == due.timestamp()
        assert set(
            call(service, "storage_retention_preview", kind="jobs", max_count=1)["excluded"]
        ) == set(identities)
        assert invoke(
            service,
            "storage_delete",
            identities=[identities[0]],
            idempotency_key="active-delete-test",
        )["issues"] == [{"code": "busy"}]
        completed = call(service, "jobs_execute", job_id=identities[0])
        assert completed["state"] == "succeeded"
        assert call(service, "jobs_status", job_id=identities[1])["state"] == "queued"
        for identity in identities[1:]:
            assert call(service, "jobs_cancel", job_id=identity)["state"] == "cancelled"
        preview = call(service, "storage_retention_preview", kind="jobs", max_count=1)
        native = CountBasedPolicy(1)
        # Equal creation timestamps have the same deterministic identity ordering.
        items = [
            ItemMetadata(identity, "jobs", datetime(2020, 1, 1)) for identity in sorted(identities)
        ]
        native.prepare_batch(items)
        assert preview["candidates"] == [
            item.item_id for item in items if not native.should_retain(item)
        ]
        retained = call(service, "jobs_result", job_id=identities[0])["artifact"]["resource_id"]
        deletion = {"identities": identities, "idempotency_key": "delete-repeated-test"}
        assert approved("storage_delete", deletion)["deleted"] == sorted(identities)
        assert call(service, "storage_delete", **deletion)["deleted"] == sorted(identities)
        assert invoke(service, "jobs_status", job_id=identities[0])["issues"] == [
            {"code": "invalid_handle"}
        ]
        assert (
            invoke(service, "artifacts_read", resource_id=retained)["execution_status"] == "failed"
        )
    finally:
        service.close()


def test_repeated_submission_rolls_back_every_occurrence_on_partial_write(tmp_path, monkeypatch):
    from truthound.mcp import durable_jobs
    from truthound.mcp.policy import MCPError

    data, control = tmp_path / "data", tmp_path / "control"
    data.mkdir()
    control.mkdir(mode=0o700)
    pl.DataFrame({"x": [1, 2]}).write_parquet(data / "input.parquet")
    service = MCPService(
        MCPPolicy((data,), profile="core", allow_sensitive_results=True, control_directory=control)
    )
    try:
        asset = call(service, "assets_capture", path=str(data / "input.parquet"))["resource_id"]
        args = {
            "validation": {"asset_id": asset},
            "idempotency_key": "atomic-series-test",
            "schedule": {"expression": "@daily", "after": "2020-01-01T00:00:00+00:00", "count": 3},
        }
        pending = invoke(service, "jobs_submit", **args)
        assert pending["issues"] == [{"code": "approval_required"}]
        store = ApprovalStore(control)
        try:
            store.approve(pending["data"]["plan_id"])
        finally:
            store.close()
        original = durable_jobs.write
        writes = []

        def interrupted(*arguments):
            original(*arguments)
            writes.append(arguments[1])
            if len(writes) == 2:
                raise MCPError("injected_partial_write")

        monkeypatch.setattr(durable_jobs, "write", interrupted)
        assert invoke(service, "jobs_submit", **args)["issues"] == [{"code": "outcome_unknown"}]
        assert len(writes) == 2
        assert (
            service._session.approvals()
            .connection.execute("SELECT COUNT(*) FROM validation_jobs")
            .fetchone()[0]
            == 0
        )
        assert invoke(service, "jobs_submit", **args)["issues"] == [{"code": "outcome_unknown"}]
        assert len(writes) == 2
    finally:
        service.close()


def test_persistent_automation_restart_native_cron_and_approved_retention(tmp_path):
    from datetime import datetime

    from truthound.mcp.automation import tick
    from truthound.scheduling import CronExpression

    data, control = tmp_path / "data", tmp_path / "control"
    data.mkdir()
    control.mkdir(mode=0o700)
    policy = MCPPolicy(
        (data,), profile="core", allow_sensitive_results=True, control_directory=control
    )
    service = MCPService(policy)

    def approved(name, **request):
        pending = invoke(service, name, **request)
        assert pending["issues"] == [{"code": "approval_required"}], pending
        store = ApprovalStore(control)
        try:
            store.approve(pending["data"]["plan_id"])
        finally:
            store.close()
        return call(service, name, **request)

    try:
        asset = call(
            service,
            "assets_inline",
            columns=[{"name": "x", "dtype": {"kind": "Int64"}, "values": [1, None]}],
        )["resource_id"]
        request = dict(
            validation={"asset_id": asset, "checks": [{"validator": "not_null"}]},
            schedule={"expression": "@hourly", "after": "2020-01-01T00:00:00+00:00", "count": 1},
            idempotency_key="persistent-recurring-01",
        )
        schedule = approved("schedules_register", **request)
        assert call(service, "schedules_register", **request) == schedule
        retention = approved(
            "storage_policy_register",
            kind="jobs",
            max_count=1,
            idempotency_key="automatic-retention-01",
        )
        # An already-signed pre-composition policy must retain its original ALL
        # semantics after an upgrade and restart; do not require re-registration.
        from truthound.mcp import automation

        legacy = automation.read(service._session, retention["control_id"])
        legacy["retention"].pop("mode")
        legacy["retention"].pop("max_size_bytes")
        automation.write(service._session.approvals(), retention["control_id"], legacy)
        service.close()
        service = MCPService(policy)
        control_arg = {"control_id": schedule["control_id"]}
        call(service, "schedules_status", **control_arg)
        expected = CronExpression.parse("@hourly").next_n(
            4, datetime.fromisoformat(request["schedule"]["after"])
        )
        for generation, due in enumerate(expected):
            asyncio.run(tick(service._session))
            state = call(service, "schedules_status", **control_arg)
            assert state["generation"] == generation
            job = call(service, "jobs_status", job_id=state["job_id"])
            assert job["state"] == "succeeded" and job["due_at"] == due.timestamp()
            assert invoke(
                service,
                "storage_delete",
                identities=[job["job_id"]],
                idempotency_key=f"protected-head-{generation}",
            )["issues"] == [{"code": "busy"}]
        other = MCPService(policy)
        try:
            call(other, "schedules_status", **control_arg)

            async def competing_workers():
                return await asyncio.gather(tick(service._session), tick(other._session))

            asyncio.run(competing_workers())
            concurrent = call(service, "schedules_status", **control_arg)
            assert concurrent["generation"] == len(expected)
            assert call(service, "jobs_status", job_id=concurrent["job_id"])["state"] == "succeeded"
            state = concurrent
        finally:
            other.close()
        assert (
            call(service, "storage_policy_status", control_id=retention["control_id"])["deleted"]
            >= 1
        )
        assert invoke(service, "storage_policy_disable", **control_arg)["issues"] == [
            {"code": "invalid_handle"}
        ]
        assert call(service, "schedules_disable", **control_arg)["state"] == "disabled"
        call(service, "storage_policy_disable", control_id=retention["control_id"])
        assert asyncio.run(tick(service._session)) == 0
        approved(
            "storage_delete", identities=[state["job_id"]], idempotency_key="disabled-head-delete"
        )
        assert call(service, "schedules_disable", **control_arg)["state"] == "disabled"
    finally:
        service.close()


@pytest.mark.parametrize("fault", ["partial_write", "expired_lease", "tamper", "policy_change"])
def test_recurring_automation_faults_never_retry_or_advance(tmp_path, monkeypatch, fault):
    from truthound.mcp import automation, durable_jobs
    from truthound.mcp.policy import MCPError

    data, control = tmp_path / "data", tmp_path / "control"
    data.mkdir()
    control.mkdir(mode=0o700)
    policy = MCPPolicy(
        (data,), profile="core", allow_sensitive_results=True, control_directory=control
    )
    service = MCPService(policy)
    try:
        asset = call(
            service,
            "assets_inline",
            columns=[{"name": "x", "dtype": {"kind": "Int64"}, "values": [1]}],
        )["resource_id"]
        request = dict(
            validation={"asset_id": asset},
            schedule={"expression": "@hourly", "after": "2020-01-01T00:00:00+00:00", "count": 1},
            idempotency_key="fault-recurring-01",
        )
        pending = invoke(service, "schedules_register", **request)
        assert pending["issues"] == [{"code": "approval_required"}]
        store = service._session.approvals()
        store.approve(pending["data"]["plan_id"])
        if fault == "partial_write":
            original = durable_jobs.write

            def interrupted(*args):
                original(*args)
                raise MCPError("injected_partial_write")

            monkeypatch.setattr(durable_jobs, "write", interrupted)
            assert invoke(service, "schedules_register", **request)["issues"] == [
                {"code": "outcome_unknown"}
            ]
            for table in ("validation_jobs", "mcp_automations"):
                assert store.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
            return
        state = call(service, "schedules_register", **request)
        if fault == "expired_lease":
            record, payload, output = durable_jobs.read(store, state["job_id"])
            record.update(state="running", lease_until=0)
            durable_jobs.write(store, state["job_id"], record, payload, output)
            asyncio.run(automation.tick(service._session))
            paused = call(service, "schedules_status", control_id=state["control_id"])
            assert paused["state"] == "paused" and paused["generation"] == 0
            assert (
                call(service, "jobs_status", job_id=state["job_id"])["state"] == "outcome_unknown"
            )
            assert asyncio.run(automation.tick(service._session)) == 0
        else:
            if fault == "tamper":
                store.connection.execute("UPDATE mcp_automations SET signature='corrupt'")
            else:
                service.close()
                service = MCPService(
                    MCPPolicy(
                        (data,),
                        profile="core",
                        allow_sensitive_results=True,
                        control_directory=control,
                        max_items=99,
                    )
                )
                invoke(service, "schedules_status", control_id=state["control_id"])
            with pytest.raises(MCPError) as failure:
                asyncio.run(automation.tick(service._session))
            assert failure.value.code == (
                "invalid_control_state" if fault == "tamper" else "access_denied"
            )
    finally:
        service.close()


@pytest.mark.fault
def test_foreground_worker_rejects_invalid_policy_and_stops_owned_process(tmp_path):
    import os
    import subprocess
    import sys
    import time

    data, control = tmp_path / "data", tmp_path / "control"
    data.mkdir()
    control.mkdir(mode=0o700)
    args = [
        sys.executable,
        "-m",
        "truthound.mcp",
        "worker",
        "--allow-dir",
        str(data),
        "--control-dir",
        str(control),
    ]
    denied = subprocess.run([*args, "--once"], capture_output=True, timeout=30)
    assert denied.returncode != 0
    assert b"Traceback" not in denied.stderr
    valid = [*args, "--profile", "core", "--allow-sensitive-results"]
    assert subprocess.run([*valid, "--once"], capture_output=True, timeout=30).returncode == 0
    if os.name == "nt":
        # Windows terminate is not POSIX SIGTERM; use once to verify clean lifecycle there.
        return
    process = subprocess.Popen(valid, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        time.sleep(1)
        assert process.poll() is None
        process.terminate()  # Only this test-owned foreground process.
        _, stderr = process.communicate(timeout=30)
        assert process.returncode == 0, stderr
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)


@pytest.mark.parametrize("mode", ["all", "any"])
@pytest.mark.parametrize("commit_fault", [False, True])
def test_native_size_composite_retention_restart_and_atomic_cleanup(
    tmp_path, monkeypatch, mode, commit_fault
):
    from datetime import datetime

    from truthound.mcp import automation
    from truthound.mcp.durable_jobs import read as job_read
    from truthound.stores.retention.base import ItemMetadata, PolicyMode
    from truthound.stores.retention.policies import (
        CompositePolicy,
        CountBasedPolicy,
        SizeBasedPolicy,
    )

    data, control = tmp_path / "data", tmp_path / "control"
    data.mkdir()
    control.mkdir(mode=0o700)
    policy = MCPPolicy(
        (data,), profile="core", allow_sensitive_results=True, control_directory=control
    )
    service = MCPService(policy)

    def approve(name, request):
        pending = invoke(service, name, **request)
        assert pending["issues"] == [{"code": "approval_required"}], pending
        service._session.approvals().approve(pending["data"]["plan_id"])
        return call(service, name, **request)

    try:
        asset = call(
            service,
            "assets_inline",
            columns=[{"name": "x", "dtype": {"kind": "Int64"}, "values": [1, 2]}],
        )["resource_id"]
        series = approve(
            "jobs_submit",
            dict(
                validation={"asset_id": asset},
                schedule={
                    "expression": "@hourly",
                    "after": "2020-01-01T00:00:00+00:00",
                    "count": 3,
                },
                idempotency_key="size-retention-series",
            ),
        )
        items = []
        for identity in sorted(series["occurrence_ids"]):
            call(service, "jobs_cancel", job_id=identity)
            record, payload, output = job_read(service._session.approvals(), identity)
            items.append(
                ItemMetadata(
                    identity,
                    "jobs",
                    datetime.fromtimestamp(record["created_at"]),
                    len(payload) + len(output or b""),
                )
            )
        size = items[0].size_bytes * 2
        native = CompositePolicy(
            [CountBasedPolicy(1), SizeBasedPolicy(max_size_bytes=size)], mode=PolicyMode(mode)
        )
        native.prepare_batch(items)
        candidates = [item.item_id for item in items if not native.should_retain(item)]
        assert len(candidates) == (2 if mode == "all" else 1)
        request = dict(kind="jobs", max_count=1, max_size_bytes=size, mode=mode)
        assert call(service, "storage_retention_preview", **request)["candidates"] == candidates
        saved = approve(
            "storage_policy_register", {**request, "idempotency_key": "size-policy-register"}
        )
        service.close()
        service = MCPService(policy)
        call(service, "storage_policy_status", control_id=saved["control_id"])
        if commit_fault:
            original = automation.write

            def interrupted(*args):
                original(*args)
                raise OSError("synthetic control write failure")

            monkeypatch.setattr(automation, "write", interrupted)
            with pytest.raises(OSError):
                asyncio.run(automation.tick(service._session))
            assert (
                call(service, "storage_policy_status", control_id=saved["control_id"])["deleted"]
                == 0
            )
            for identity in series["occurrence_ids"]:
                assert call(service, "jobs_status", job_id=identity)["state"] == "cancelled"
        else:
            asyncio.run(automation.tick(service._session))
            assert call(service, "storage_policy_status", control_id=saved["control_id"])[
                "deleted"
            ] == len(candidates)
            for identity in series["occurrence_ids"]:
                response = invoke(service, "jobs_status", job_id=identity)
                assert response["execution_status"] == (
                    "failed" if identity in candidates else "succeeded"
                )
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
