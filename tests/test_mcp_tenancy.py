"""Actual native tenant store, scoped context and quota calls via pinned aliases."""

import json
import sys
from types import SimpleNamespace

import pytest

from tests import test_mcp_plugin_bindings as plugins
from tests.test_mcp_brokers import approved

runtime = plugins.runtime
pytestmark = pytest.mark.contract


def prepare(runtime, monkeypatch):
    import time
    from truthound.mcp import model_process
    from truthound.mcp.tenancy import TenantServices

    adapter, control, _, _, _ = runtime
    original = getattr(TenantServices.process_type, "_test_original", TenantServices.process_type)

    def tracked(ttl):
        worker = original(ttl)
        adapter._test_native_worker = worker
        worker._test_exchanges = []
        native_call = worker.call

        async def measured(payload, timeout):
            started = time.perf_counter()
            before = getattr(worker, "exchange_request_index", 0)
            try:
                return await native_call(payload, timeout)
            finally:
                worker._test_exchanges.append({
                    "kind": "prepare" if payload == {"prepare": True} else "native",
                    "budget": timeout, "seconds": time.perf_counter() - started,
                    "phase": getattr(worker, "exchange_phase", None),
                    "failure": getattr(worker, "exchange_failure", None),
                    "category": getattr(worker, "exchange_worker_category", None),
                    "request_index": getattr(worker, "exchange_request_index", None),
                    "dispatch_started": getattr(worker, "exchange_request_index", 0) > before,
                })

        worker.call = measured
        return worker

    tracked._test_original = original
    monkeypatch.setattr(TenantServices, "process_type", staticmethod(tracked))
    monkeypatch.setattr(
        model_process,
        "sys",
        SimpleNamespace(
            executable=adapter._worker_argv()[0],
            platform=sys.platform,
            _base_executable=sys._base_executable,
        ),
    )
    document = json.loads((control / "bindings.json").read_text())
    document["plugins"]["fixture"]["capabilities"]["tenant-factory"] = "tenant-factory"
    document["tenant_services"] = {"local": "tenant-factory"}
    (control / "bindings.json").write_text(json.dumps(document))
    return approved(
        runtime, "tenant_services_open", binding_id="local", idempotency_key="tenant-open"
    )["data"]["resource_id"]


def command(runtime, identity, command_name, *, key=None, **arguments):
    result = approved(
        runtime,
        "tenant_services_execute",
        binding_id="local",
        service_id=identity,
        command=command_name,
        idempotency_key=key or "tenant-" + command_name,
        **arguments,
    )
    return plugins.source_artifact(runtime[0], result["data"]["resource_id"])


def test_native_row_policy_lifecycle_applies_to_pinned_input_and_revokes_with_it(
    runtime, monkeypatch
):
    identity = prepare(runtime, monkeypatch)
    command(runtime, identity, "create", tenant_id="local", patch={"name": "Local"})
    asset = plugins.invoke(
        runtime[0],
        "assets_inline",
        columns=[
            {
                "name": "tenant_id",
                "dtype": {"kind": "String"},
                "values": ["outside", "internal-private-tenant", "internal-private-tenant"],
            },
            {"name": "x", "dtype": {"kind": "Int64"}, "values": [99, -1, 1]},
        ],
    )["data"]["resource_id"]

    def filtered(key, **arguments):
        return command(
            runtime,
            identity,
            "isolation_apply",
            key=key,
            tenant_id="local",
            asset_id=asset,
            columns=["x"],
            **arguments,
        )

    assert filtered("unrestricted-local")["rows"] == [{"x": -1}, {"x": 1}]
    command(runtime, identity, "policy_add", tenant_id="local", policy_id="positive")
    assert filtered("positive-local", limit=1)["rows"] == [{"x": 1}]
    catalog = command(runtime, identity, "policy_catalog", tenant_id="local")
    assert catalog["policies"][0] == {
        "policy_id": "positive",
        "active": True,
        "enabled": True,
        "priority": 0,
    }
    assert "private-positive-policy" not in str(catalog)
    command(
        runtime, identity, "policy_add", key="add-untyped", tenant_id="local", policy_id="untyped"
    )
    assert filtered("untyped-denied")["rows"] == []
    command(runtime, identity, "policy_remove", tenant_id="local", policy_id="untyped")
    command(
        runtime,
        identity,
        "policy_update",
        tenant_id="local",
        policy_id="positive",
        policy_patch={"enabled": False, "priority": 2},
    )
    assert filtered("disabled-local")["rows"] == [{"x": -1}, {"x": 1}]
    worker = runtime[0]._session._administration["tenant_services"].workers[identity]
    plugins.invoke(runtime[0], "resources_release", resource_id=asset)
    assert worker.closed and worker.process.poll() is not None
    denied = plugins.invoke(
        runtime[0],
        "tenant_services_execute",
        binding_id="local",
        service_id=identity,
        command="get",
        tenant_id="local",
        idempotency_key="after-input-revoke",
    )
    assert denied["issues"] == [{"code": "invalid_handle"}], denied


def test_native_tenant_lifecycle_quota_and_sqlite_worker_reopen(runtime, monkeypatch):
    identity = prepare(runtime, monkeypatch)
    created = command(
        runtime,
        identity,
        "create",
        tenant_id="local",
        patch={"name": "Local", "settings_id": "local", "quota": {"api_calls_per_day": 2}},
    )["tenant"]
    assert created["tenant_id"] == "local" and created["status"] == "active"
    assert "internal-private-tenant" not in str(created)
    assert command(runtime, identity, "context", tenant_id="local")["current"]
    assert command(runtime, identity, "quota_check", tenant_id="local", resource="api_calls")[
        "allowed"
    ]
    assert (
        command(
            runtime, identity, "usage_track", tenant_id="local", resource="api_calls", amount=2
        )["usage"]
        == 2
    )
    assert not command(
        runtime, identity, "quota_check", key="quota-after", tenant_id="local", resource="api_calls"
    )["allowed"]
    usage = command(runtime, identity, "usage_summary", tenant_id="local", resource="api_calls")[
        "usage"
    ]
    assert usage["total_usage"] == 2 and usage["tenant_id"] == "local"
    assert (
        command(runtime, identity, "suspend", tenant_id="local", reason="local review")["tenant"][
            "status"
        ]
        == "suspended"
    )
    assert (
        plugins.invoke(runtime[0], "resources_release", resource_id=identity)["execution_status"]
        == "succeeded"
    )
    assert "tenant-close" in runtime[2].read_text()
    identity = approved(
        runtime, "tenant_services_open", binding_id="local", idempotency_key="tenant-reopen"
    )["data"]["resource_id"]
    assert (
        command(runtime, identity, "get", key="tenant-restored", tenant_id="local")["tenant"][
            "status"
        ]
        == "suspended"
    )
    assert command(runtime, identity, "activate", tenant_id="local")["tenant"]["status"] == "active"
    assert (
        command(
            runtime,
            identity,
            "set_tier",
            tenant_id="local",
            tier="professional",
            update_quota=False,
        )["tenant"]["quota"]["api_calls_per_day"]
        == 2
    )
    assert (
        command(
            runtime,
            identity,
            "update",
            tenant_id="local",
            patch={"name": "Changed", "quota": {"api_calls_per_day": 3}},
        )["tenant"]["name"]
        == "Changed"
    )
    assert command(runtime, identity, "delete", tenant_id="local")["deleted"]


@pytest.mark.fault
def test_tenant_physical_paths_and_provider_configuration_are_not_wire_fields(runtime, monkeypatch):
    identity = prepare(runtime, monkeypatch)
    for extra in (
        {"database_name": "private-db"},
        {"metadata": {"credential": "not-real"}},
        {"resolver": "arbitrary.module"},
        {"store_path": "/private/file"},
    ):
        response = plugins.invoke(
            runtime[0],
            "tenant_services_execute",
            binding_id="local",
            service_id=identity,
            command="catalog",
            idempotency_key="tenant-invalid-fields",
            **extra,
        )
        assert response["issues"] == [{"code": "invalid_arguments"}]


@pytest.mark.fault
def test_tenant_alias_does_not_grant_access_to_another_native_tenant(runtime, monkeypatch):
    from truthound.mcp.approval import ApprovalStore

    identity = prepare(runtime, monkeypatch)
    request = dict(
        binding_id="local",
        service_id=identity,
        command="get",
        tenant_id="outside",
        idempotency_key="tenant-reject-native-id",
    )
    pending = plugins.invoke(runtime[0], "tenant_services_execute", **request)
    assert pending["issues"] == [{"code": "approval_required"}]
    store = ApprovalStore(runtime[1])
    try:
        store.approve(pending["data"]["plan_id"])
    finally:
        store.close()
    failed = plugins.invoke(runtime[0], "tenant_services_execute", **request)
    assert failed["execution_status"] == "outcome_unknown"
    assert "Outside" not in str(failed)


def test_native_tenant_resolver_contexts_are_private_and_selected_creation_survives_reopen(
    runtime, monkeypatch
):
    identity = prepare(runtime, monkeypatch)
    catalog = command(runtime, identity, "resolver_catalog")
    assert catalog == {"contexts": ["missing", "outside", "selected"], "total": 3}
    assert "synthetic-private-context-token" not in str(catalog)
    assert command(runtime, identity, "resolve", context_id="missing") == {
        "resolved": False,
        "tenant_id": None,
    }
    assert (
        command(runtime, identity, "resolve", key="selected-before-create", context_id="selected")[
            "resolved"
        ]
        is False
    )
    created = command(
        runtime,
        identity,
        "resolve_or_create",
        context_id="selected",
        tenant_id="local",
        default_name="Resolved",
    )
    assert created["resolved"] is True and created["tenant_id"] == "local"
    assert created["tenant"]["name"] == "Resolved"
    assert "internal-private-tenant" not in str(created)
    assert "synthetic-private-context-token" not in str(created)
    plugins.invoke(runtime[0], "resources_release", resource_id=identity)
    reopened = approved(
        runtime,
        "tenant_services_open",
        binding_id="local",
        idempotency_key="resolved-tenant-reopened",
    )["data"]["resource_id"]
    resolved = command(
        runtime, reopened, "resolve", key="resolved-after-reopen", context_id="selected"
    )
    assert resolved["tenant_id"] == "local" and resolved["tenant"]["name"] == "Resolved"
    plugins.invoke(runtime[0], "resources_release", resource_id=reopened)


def test_native_tenant_resolver_cannot_create_unselected_identity(runtime, monkeypatch):
    from truthound.mcp.approval import ApprovalStore

    identity = prepare(runtime, monkeypatch)
    request = dict(
        binding_id="local",
        service_id=identity,
        command="resolve_or_create",
        tenant_id="local",
        context_id="outside",
        default_name="Denied",
        idempotency_key="resolver-outside-create",
    )
    pending = plugins.invoke(runtime[0], "tenant_services_execute", **request)
    assert pending["issues"] == [{"code": "approval_required"}]
    approvals = ApprovalStore(runtime[1])
    try:
        approvals.approve(pending["data"]["plan_id"])
    finally:
        approvals.close()
    failed = plugins.invoke(runtime[0], "tenant_services_execute", **request)
    assert failed["execution_status"] == "outcome_unknown"
    assert "synthetic-private-context-token" not in str(failed)
    # The failed native operation can invalidate the worker; inspect persisted
    # state through a new selected owner rather than replaying the creation.
    plugins.invoke(runtime[0], "resources_release", resource_id=identity)
    reopened = approved(
        runtime,
        "tenant_services_open",
        binding_id="local",
        idempotency_key="resolver-denied-reopened",
    )["data"]["resource_id"]
    assert (
        command(runtime, reopened, "get", key="denied-not-created", tenant_id="local")["tenant"]
        is None
    )
    plugins.invoke(runtime[0], "resources_release", resource_id=reopened)


@pytest.mark.parametrize("provider", ["sqlite", "duckdb"])
def test_installed_structured_source_reads_same_database_after_owner_restart(runtime, monkeypatch, provider):
    runtime[2].with_name('tenant-source-mode').write_text(provider)
    identity = prepare(runtime, monkeypatch)
    assert command(runtime, identity, 'source_catalog') == {'sources': ['records'], 'total': 1}
    first = command(runtime, identity, 'source_read', tenant_id='structured', source_id='records', limit=1)
    assert first['rows'] == [{'x': 1}] and first['truncated'] is True
    assert 'private-tenant-data' not in str(first) and 'private-structured' not in str(first)
    plugins.invoke(runtime[0], 'resources_release', resource_id=identity)
    reopened = approved(runtime, 'tenant_services_open', binding_id='local', idempotency_key='source-reopened')['data']['resource_id']
    second = command(runtime, reopened, 'source_read', key='source-second', tenant_id='structured', source_id='records', offset=1, columns=['x'])
    assert second['rows'] == [{'x': 2}] and second['truncated'] is False
    plugins.invoke(runtime[0], 'resources_release', resource_id=reopened)
    assert runtime[2].read_text().count('tenant-close') == 2
