"""Installed registered ports execute together and export canonical native runs."""

import json

import pytest

from tests import test_mcp_plugin_bindings as plugins
from tests.test_mcp_brokers import approved
from tests.test_mcp_platform_services import command, prepare

runtime = plugins.runtime
pytestmark = pytest.mark.contract


def register(runtime, identity):
    group = "plugin_services"
    for name, alias in [
        ("check_factory_register", "native-checks"),
        ("provider_register", "native-data"),
        ("reporter_register", "native-json"),
    ]:
        assert command(runtime, group, identity, name, alias=alias)["registered"]
    for alias, event in [
        ("before", "before_validation"),
        ("after", "after_validation"),
        ("issue", "on_issue_found"),
        ("report", "on_report_generate"),
    ]:
        assert command(
            runtime,
            group,
            identity,
            "hook_register",
            alias=alias,
            hook_type=event,
            key="register-" + alias,
        )["registered"]


def test_same_owner_provider_checks_hooks_reporter_export_and_external_native_store(
    runtime, monkeypatch
):
    import polars as pl

    from truthound.core import (
        CheckSpec,
        ScanPlanner,
        ValidationRuntime,
        ValidationSuite,
        build_validation_asset,
    )
    from truthound.core.suite import EvidencePolicy
    from truthound.validators import get_validator

    document = json.loads((runtime[1] / "bindings.json").read_text())
    document["plugins"]["fixture"]["capabilities"]["store-factory"] = "store-factory"
    document["store_services"] = {"local": "store-factory"}
    (runtime[1] / "bindings.json").write_text(json.dumps(document))
    identity = prepare(runtime, monkeypatch, "plugin_services")
    register(runtime, identity)
    actual = command(
        runtime,
        "plugin_services",
        identity,
        "validate",
        provider_id="native-data",
        check_aliases=["native-checks"],
        result_format="complete",
    )
    assert not actual["hooks_failed"]
    assert all(row["success"] for rows in actual["hooks"].values() for row in rows)
    spec = CheckSpec(
        id="native-positive",
        name="selected-positive",
        category="range",
        factory=lambda: get_validator("between")(columns=["x"], min_value=0),
    )
    asset = build_validation_asset(pl.DataFrame({"x": [-1, 1]}))
    suite = ValidationSuite(
        name="plugin-suite", checks=(spec,), evidence_policy=EvidencePolicy.from_any("complete")
    )
    direct = (
        ValidationRuntime()
        .execute(asset=asset, plan=ScanPlanner().plan(suite=suite, asset=asset))
        .to_dict()
    )
    for key in (
        "checks",
        "issues",
        "execution_issues",
        "source",
        "row_count",
        "column_count",
        "result_format",
        "execution_mode",
        "planned_execution_mode",
    ):
        assert actual["run"][key] == direct[key], key
    rendered = command(
        runtime, "plugin_services", identity, "render", alias="native-json", run_id=actual["run_id"]
    )
    assert (
        not rendered["hooks_failed"]
        and json.loads(rendered["content"])["result"]["run_id"] == actual["run"]["run_id"]
    )
    exported = approved(
        runtime,
        "plugin_services_execute",
        binding_id="local",
        service_id=identity,
        command="run_export",
        run_id=actual["run_id"],
        idempotency_key="native-run-export",
    )["data"]
    assert exported["media_type"] == "application/vnd.truthound.validation+json"
    assert plugins.source_artifact(runtime[0], exported["resource_id"]) == actual["run"]
    # Exported native results are ordinary canonical inputs for existing consumers.
    report = plugins.invoke(
        runtime[0], "artifacts_render", resource_id=exported["resource_id"], format="json"
    )
    assert report["execution_status"] == "succeeded"
    from tests.test_mcp_execution_stores import command as store_command

    store_identity = approved(
        runtime, "store_services_open", binding_id="local", idempotency_key="plugin-result-store"
    )["data"]["resource_id"]
    saved = store_command(
        runtime, "store_services", store_identity, "save", result_id=exported["resource_id"]
    )
    assert (
        store_command(
            runtime, "store_services", store_identity, "get", record_id=saved["record_id"]
        )["record"]["status"]
        == "failure"
    )
    command(runtime, "plugin_services", identity, "run_release", run_id=actual["run_id"])
    events = runtime[2].read_text().splitlines()
    assert events.count("native-provider-read") == events.count("native-check-build") == 1
    assert (
        events.count("native-before")
        == events.count("native-after")
        == events.count("native-issue")
        == events.count("native-report")
        == 1
    )
    plugins.invoke(runtime[0], "resources_release", resource_id=identity)
    assert "platform-plugin-close" in runtime[2].read_text()


def test_registered_checks_consume_retained_asset_and_revoked_input_closes_owner(
    runtime, monkeypatch
):
    identity = prepare(runtime, monkeypatch, "plugin_services")
    register(runtime, identity)
    result = command(
        runtime,
        "plugin_services",
        identity,
        "validate",
        asset_id=runtime[4],
        check_aliases=["native-checks"],
    )
    assert result["run"]["row_count"] == 2 and not result["hooks_failed"]
    assert "native-provider-read" not in runtime[2].read_text()
    plugins.invoke(runtime[0], "resources_release", resource_id=runtime[4])
    assert "platform-plugin-close" in runtime[2].read_text()
    failed = plugins.invoke(
        runtime[0],
        "plugin_services_execute",
        binding_id="local",
        service_id=identity,
        command="run_get",
        run_id=result["run_id"],
        idempotency_key="after-asset-release",
    )
    assert failed["issues"] == [{"code": "invalid_handle"}]


@pytest.mark.fault
def test_detailed_hook_error_is_a_native_failed_outcome_without_exception_body(
    runtime, monkeypatch
):
    identity = prepare(runtime, monkeypatch, "plugin_services")
    command(runtime, "plugin_services", identity, "hook_register", alias="failure")
    result = command(runtime, "plugin_services", identity, "hook_fire_detailed")
    assert result["native_failed"] and not result["hooks"][0]["success"]
    assert result["hooks"][0]["error_type"] == "RuntimeError"
    assert "synthetic-private-hook-detail" not in str(result)
    assert runtime[2].read_text().splitlines().count("native-hook-failed") == 1


def test_native_registered_port_removal_is_manager_local_and_exact(runtime, monkeypatch):
    identity = prepare(runtime, monkeypatch, "plugin_services")
    register(runtime, identity)
    command(
        runtime, "plugin_services", identity, "capability_register", alias="approved-capability"
    )
    for name, alias in [
        ("check_factory_unregister", "native-checks"),
        ("provider_unregister", "native-data"),
        ("reporter_unregister", "native-json"),
        ("capability_unregister", "approved-capability"),
    ]:
        assert command(runtime, "plugin_services", identity, name, alias=alias)["removed"]
    ports = command(runtime, "plugin_services", identity, "ports")
    assert ports == {"capabilities": [], "providers": [], "reporters": {}, "check_factories": []}


@pytest.mark.fault
def test_native_artifact_reads_reauthorize_policy_without_waiting_for_watchdog(
    runtime, monkeypatch
):
    from dataclasses import replace

    from truthound.mcp.schema_monitor import MonitorProcess

    # Simulate an unavailable watchdog; actual worker and synchronous grants remain real.
    monkeypatch.setattr(MonitorProcess, "watch_policy", lambda self, authorized: None)
    identity = prepare(runtime, monkeypatch, "plugin_services")
    artifact = approved(
        runtime,
        "plugin_services_execute",
        binding_id="local",
        service_id=identity,
        command="ports",
        idempotency_key="policy-bound-artifact",
    )["data"]["resource_id"]
    worker = runtime[0]._session._administration["plugin_services"].workers[identity]
    runtime[0].policy = replace(runtime[0].policy, max_items=runtime[0].policy.max_items - 1)
    response = plugins.invoke(runtime[0], "artifacts_read", resource_id=artifact)
    assert response["execution_status"] != "succeeded"
    assert worker.closed


@pytest.mark.fault
def test_native_failed_cleanup_receipt_preserves_partial_completion_without_error_body(
    runtime, monkeypatch
):
    identity = prepare(runtime, monkeypatch, "plugin_services")
    runtime[2].with_name("failed-target-cleanup").write_text("local failure")
    command(runtime, "plugin_services", identity, "load", alias="target")
    command(runtime, "plugin_services", identity, "activate", alias="target")
    result = command(runtime, "plugin_services", identity, "unload", alias="target")
    assert not result["loaded"] and result["native_failed"]
    assert result["cleanup"]["removed"] and not result["cleanup"]["success"]
    assert result["cleanup"]["error_types"] == ["ValueError", "RuntimeError"]
    assert "synthetic-private-target" not in str(result)
    events = runtime[2].read_text().splitlines()
    assert events.count("target-unregister") == events.count("target-teardown") == 1


@pytest.mark.fault
def test_revoked_bound_plugin_still_runs_owned_teardown_before_factory_exit(runtime, monkeypatch):
    identity = prepare(runtime, monkeypatch, "plugin_services")
    command(runtime, "plugin_services", identity, "load", alias="target")
    command(runtime, "plugin_services", identity, "activate", alias="target")
    command(
        runtime,
        "plugin_services",
        identity,
        "hook_register",
        alias="local-hook",
        hook_type="on_plugin_unload",
    )
    document = json.loads((runtime[1] / "bindings.json").read_text())
    document["plugins"]["fixture"]["version"] = "9.9.9"
    (runtime[1] / "bindings.json").write_text(json.dumps(document))
    result = plugins.invoke(runtime[0], "resources_describe", resource_id=identity)
    assert result["execution_status"] != "succeeded"
    events = runtime[2].read_text().splitlines()
    assert events.count("target-unregister") == events.count("target-teardown") == 1
    assert "custom-hook" not in events and "platform-plugin-close" in events


@pytest.mark.parametrize("fail_setup", [False, True])
def test_approved_reload_runs_native_replacement_or_observed_rollback(
    runtime, monkeypatch, fail_setup
):
    identity = prepare(runtime, monkeypatch, "plugin_services")
    command(runtime, "plugin_services", identity, "load", alias="target")
    command(runtime, "plugin_services", identity, "activate", alias="target")
    if fail_setup:
        runtime[2].with_name("failed-replacement-setup").write_text("local fault")
    result = command(
        runtime, "plugin_services", identity, "reload", alias="target", reload_strategy="graceful"
    )
    assert result["loaded"]
    assert result["reloaded"] is (not fail_setup)
    assert result["rolled_back"] is fail_setup
    assert result["native_failed"] is fail_setup
    assert "synthetic-private-reload" not in str(result)
    events = runtime[2].read_text().splitlines()
    assert events.count("target-unregister") == 1
    assert events.count("target-teardown") == 1
    assert events.count("replacement-setup") == 1
    assert events.count("target-setup") == (2 if fail_setup else 1)
    if fail_setup:
        assert events.count("replacement-teardown") == 1
    command(runtime, "plugin_services", identity, "unload", alias="target")


@pytest.mark.parametrize("fail_cleanup", [False, True])
def test_acknowledged_plugin_shutdown_waits_for_teardown_and_factory(
    runtime, monkeypatch, fail_cleanup
):
    from truthound.mcp.approval import ApprovalStore

    identity = prepare(runtime, monkeypatch, "plugin_services")
    command(runtime, "plugin_services", identity, "load", alias="target")
    command(runtime, "plugin_services", identity, "activate", alias="target")
    if fail_cleanup:
        runtime[2].with_name("failed-target-cleanup").write_text("local failure")
    request = dict(
        binding_id="local",
        service_id=identity,
        command="shutdown",
        idempotency_key="plugin-shutdown",
    )
    pending = plugins.invoke(runtime[0], "plugin_services_execute", **request)
    assert pending["issues"] == [{"code": "approval_required"}]
    approvals = ApprovalStore(runtime[1])
    try:
        approvals.approve(pending["data"]["plan_id"])
    finally:
        approvals.close()
    result = plugins.invoke(runtime[0], "plugin_services_execute", **request)
    assert (result["execution_status"] == "succeeded") is (not fail_cleanup)
    assert "synthetic-private-target" not in str(result)
    events = runtime[2].read_text().splitlines()
    assert events.count("target-unregister") == events.count("target-teardown") == 1
    assert events.count("platform-plugin-close") == 1
    if not fail_cleanup:
        assert plugins.source_artifact(runtime[0], result["data"]["resource_id"]) == {
            "shutdown": True
        }
    repeated = plugins.invoke(runtime[0], "plugin_services_execute", **request)
    assert runtime[2].read_text().splitlines().count("target-teardown") == 1
    if fail_cleanup:
        assert repeated["execution_status"] != "succeeded"
