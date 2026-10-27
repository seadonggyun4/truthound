"""Actual native owner crashes leave private, explicitly reconciled batch state."""

import pytest

from tests.test_mcp_brokers import approved
from tests.test_mcp_execution_stores import command, plugins, prepare
from truthound.mcp.approval import ApprovalStore

runtime = plugins.runtime
pytestmark = pytest.mark.fault


@pytest.mark.parametrize("resolution", ["rollback", "keep_current"])
def test_actual_mcp_worker_crash_reopen_and_explicit_native_reconciliation(
    runtime, monkeypatch, resolution
):
    identity = prepare(runtime, monkeypatch, "store_services", mode="concurrent-crash")
    request = {
        "binding_id": "local",
        "service_id": identity,
        "idempotency_key": "crash-native-batch",
        "command": "concurrent_apply",
        "dry_run": False,
        "actions": [{"kind": "save", "record_id": "seed"}, {"kind": "save", "record_id": "fail"}],
    }
    pending = plugins.invoke(runtime[0], "store_services_execute", **request)
    assert pending["issues"] == [{"code": "approval_required"}]
    store = ApprovalStore(runtime[1])
    try:
        store.approve(pending["data"]["plan_id"])
    finally:
        store.close()
    owner = runtime[0]._session._administration["store_services"].workers[identity]
    response = plugins.invoke(runtime[0], "store_services_execute", **request)
    assert response["execution_status"] == "outcome_unknown" and owner.closed
    identity = approved(
        runtime, "store_services_open", binding_id="local", idempotency_key="reopen-crashed-store"
    )["data"]["resource_id"]
    rows = command(runtime, "store_services", identity, "concurrent_recovery_list")["recoveries"]
    assert len(rows) == 1 and rows[0]["state"] == "prepared" and rows[0]["file_count"] == 2
    assert (
        "private-native-run" not in str(rows)
        and ".batch-recovery-" not in str(rows)
        and "fingerprint" not in str(rows)
    )
    result = command(
        runtime,
        "store_services",
        identity,
        "concurrent_reconcile",
        recovery_id=rows[0]["recovery_id"],
        resolution=resolution,
    )
    assert result["resolved"] and not result["quarantined"]
    assert command(runtime, "store_services", identity, "count")["count"] == (
        0 if resolution == "rollback" else 1
    )
    assert (
        command(
            runtime,
            "store_services",
            identity,
            "concurrent_recovery_list",
            key="after-native-reconciliation",
        )["recoveries"]
        == []
    )
    assert "private-native-run" not in str(result)


def test_actual_cleanup_crash_preserves_committed_data_and_approved_cleanup(runtime, monkeypatch):
    identity = prepare(runtime, monkeypatch, "store_services", mode="concurrent-cleanup-crash")
    request = {
        "binding_id": "local",
        "service_id": identity,
        "idempotency_key": "crash-cleanup-batch",
        "command": "concurrent_apply",
        "dry_run": False,
        "actions": [{"kind": "save", "record_id": "seed"}],
    }
    pending = plugins.invoke(runtime[0], "store_services_execute", **request)
    assert pending["issues"] == [{"code": "approval_required"}]
    store = ApprovalStore(runtime[1])
    try:
        store.approve(pending["data"]["plan_id"])
    finally:
        store.close()
    response = plugins.invoke(runtime[0], "store_services_execute", **request)
    assert response["execution_status"] == "outcome_unknown"
    # The next owner uses the same native persisted backend, without the injected crash.
    runtime[2].with_name("store-mode").write_text("concurrent")
    identity = approved(
        runtime, "store_services_open", binding_id="local", idempotency_key="cleanup-crash-reopen"
    )["data"]["resource_id"]
    assert command(runtime, "store_services", identity, "count")["count"] == 1
    rows = command(runtime, "store_services", identity, "concurrent_recovery_list")["recoveries"]
    assert len(rows) == 1 and rows[0]["state"] == "cleanup_pending"
    result = command(
        runtime,
        "store_services",
        identity,
        "concurrent_recovery_cleanup",
        recovery_id=rows[0]["recovery_id"],
    )
    assert result["cleaned"] and not result["quarantined"]
    assert (
        command(
            runtime,
            "store_services",
            identity,
            "concurrent_recovery_list",
            key="after-approved-cleanup",
        )["recoveries"]
        == []
    )
    assert (
        command(runtime, "store_services", identity, "count", key="after-cleanup-count")["count"]
        == 1
    )
    assert "private-native-run" not in str(rows) and ".batch-resolved-" not in str(rows)
