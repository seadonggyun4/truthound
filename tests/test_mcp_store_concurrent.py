"""Installed native concurrent batches preserve approvals, rollback and opaque IDs."""

import pytest

from tests.test_mcp_execution_stores import command, plugins, prepare
from truthound.mcp.approval import ApprovalStore

runtime = plugins.runtime
pytestmark = pytest.mark.contract


@pytest.mark.parametrize("mode", ["concurrent", "concurrent-filelock", "concurrent-yaml"])
def test_native_concurrent_batch_preview_apply_delete_index_stats_and_reopen(
    runtime, monkeypatch, mode
):
    identity = prepare(runtime, monkeypatch, "store_services", mode=mode)

    def call(name, **args):
        return command(runtime, "store_services", identity, name, **args)

    actions = [{"kind": "save", "record_id": "seed"}, {"kind": "save", "record_id": "fail"}]
    preview = call("concurrent_apply", actions=actions)
    assert preview["dry_run"] and preview["buffered"] == 2
    assert call("count")["count"] == 0
    applied = call("concurrent_apply", actions=actions, dry_run=False, key="concurrent-save")
    assert applied["applied"] == 2 and not applied["dry_run"]
    assert call("get", record_id="seed")["record"]["status"] == "failure"
    assert call("concurrent_stats")["statistics"]["total_acquisitions"] > 0
    assert call("rebuild_index")["indexed"] == 2
    call(
        "concurrent_apply",
        actions=[{"kind": "delete", "record_id": "seed"}],
        dry_run=False,
        key="concurrent-delete",
    )
    assert not call("exists", record_id="seed")["exists"]
    assert "private-native-run" not in str(applied)
    plugins.invoke(runtime[0], "resources_release", resource_id=identity)
    from truthound.stores.backends.concurrent_filesystem import ConcurrentFileSystemStore

    reopened = ConcurrentFileSystemStore(
        base_path=runtime[2].with_name("native-store"),
        **(
            dict(serialization_format="yaml", file_extension=".yaml", compression=True)
            if mode == "concurrent-yaml"
            else {}
        ),
    )
    try:
        assert reopened.list_ids() == ["private-fail"]
    finally:
        reopened.close()


@pytest.mark.fault
def test_native_failed_batch_is_not_replayed_and_owner_is_quarantined(runtime, monkeypatch):
    identity = prepare(runtime, monkeypatch, "store_services", mode="concurrent-failure")
    request = {
        "binding_id": "local",
        "service_id": identity,
        "idempotency_key": "concurrent-failure",
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
    worker = runtime[0]._session._administration["store_services"].workers[identity]
    response = plugins.invoke(runtime[0], "store_services_execute", **request)
    assert response["execution_status"] == "outcome_unknown" and worker.closed
    assert "synthetic-private-batch-failure" not in str(response)
    from truthound.stores.backends.concurrent_filesystem import ConcurrentFileSystemStore

    reopened = ConcurrentFileSystemStore(base_path=runtime[2].with_name("native-store"))
    try:
        assert reopened.list_ids() == []
    finally:
        reopened.close()
    assert (
        plugins.invoke(runtime[0], "store_services_execute", **request)["execution_status"]
        != "succeeded"
    )


def test_native_concurrent_expectation_codec_ordering_and_tags(runtime, monkeypatch):
    identity = prepare(runtime, monkeypatch, "store_services", mode="concurrent-expectation")

    def call(name, **args):
        return command(runtime, "store_services", identity, name, **args)

    saved = {}
    for name in ("zeta", "alpha", "middle"):
        saved[name] = call(
            "save_suite",
            key="save-suite-" + name,
            suite={
                "name": name,
                "data_asset_id": "local",
                "tags": {"owner": "synthetic"},
                "expectations": [],
            },
        )["record_id"]
    result = call(
        "query",
        query={"order_by": "name", "ascending": True, "tags": {"owner": "synthetic"}},
        offset=1,
        limit=1,
    )
    assert [item["record_id"] for item in result["records"]] == [saved["middle"]]
    call("shutdown")
    from truthound.stores.backends.concurrent_filesystem import ConcurrentFileSystemExpectationStore

    reopened = ConcurrentFileSystemExpectationStore(
        base_path=runtime[2].with_name("native-store"),
        serialization_format="yaml",
        file_extension=".yaml",
        use_compression=True,
    )
    try:
        assert reopened.get("middle").tags == {"owner": "synthetic"}
    finally:
        reopened.close()
