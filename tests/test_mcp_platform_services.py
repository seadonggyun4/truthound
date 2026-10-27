"""Installed native security/plugin owners enforce exact typed approvals and scope."""

import json
import sys
from types import SimpleNamespace

import pytest

from tests import test_mcp_plugin_bindings as plugins
from tests.test_mcp_brokers import approved

runtime = plugins.runtime
invoke = plugins.invoke
pytestmark = pytest.mark.contract


def prepare(runtime, monkeypatch, group):
    from truthound.mcp import model_process

    adapter, control, _, _, _ = runtime
    monkeypatch.setattr(
        model_process,
        "sys",
        SimpleNamespace(
            executable=adapter._worker_argv()[0],
            platform=sys.platform,
            _base_executable=sys._base_executable,
        ),
    )
    capability = "security-factory" if group == "security_services" else "plugin-factory"
    path = control / "bindings.json"
    settings = json.loads(path.read_text())
    settings["plugins"]["fixture"]["capabilities"][capability] = capability
    settings[group] = {"local": capability}
    path.write_text(json.dumps(settings), encoding="utf-8")
    return approved(runtime, group + "_open", binding_id="local", idempotency_key=group + "-open")[
        "data"
    ]["resource_id"]


def command(runtime, group, identity, command_name, *, key=None, **args):
    response = approved(
        runtime,
        group + "_execute",
        binding_id="local",
        service_id=identity,
        command=command_name,
        idempotency_key=key or group + "-" + command_name,
        **args,
    )
    return plugins.source_artifact(runtime[0], response["data"]["resource_id"])


def test_native_security_role_assignment_revocation_and_plain_resource_boundary(
    runtime, monkeypatch
):
    from truthound.rbac import RBACManager

    group = "security_services"
    identity = prepare(runtime, monkeypatch, group)
    command(
        runtime,
        group,
        identity,
        "role_create",
        role_id="reader",
        role_patch={"name": "Reader", "permissions": ["dataset:read"]},
    )
    command(
        runtime,
        group,
        identity,
        "principal_create",
        principal_id="reviewer",
        principal_patch={"name": "Reviewer", "attributes": {"department": "quality"}},
    )
    command(runtime, group, identity, "assign_role", principal_id="reviewer", role_id="reader")
    allowed = command(
        runtime,
        group,
        identity,
        "check",
        principal_id="reviewer",
        resource="dataset",
        action="read",
    )
    native = RBACManager()
    role = native.create_role(
        "Reader", role_id="reader", tenant_id="local", permissions={"dataset:read"}
    )
    principal = native.create_principal("Reviewer", principal_id="reviewer", tenant_id="local")
    native.assign_role(principal.id, role.id)
    assert allowed["allowed"] == native.check(principal.id, "dataset", "read").allowed is True
    assert (
        command(
            runtime,
            group,
            identity,
            "check",
            key="check-secret",
            principal_id="reviewer",
            resource="secret",
            action="read",
        )["allowed"]
        is False
    )
    command(runtime, group, identity, "revoke_role", principal_id="reviewer", role_id="reader")
    assert (
        command(
            runtime,
            group,
            identity,
            "check",
            key="check-after-revoke",
            principal_id="reviewer",
            resource="dataset",
            action="read",
        )["allowed"]
        is False
    )
    command(
        runtime,
        group,
        identity,
        "principal_update",
        principal_id="reviewer",
        principal_patch={"enabled": False},
    )
    assert (
        command(runtime, group, identity, "principal", principal_id="reviewer")["principal"][
            "enabled"
        ]
        is False
    )
    assert command(runtime, group, identity, "role_delete", role_id="reader") == {"deleted": True}
    assert command(runtime, group, identity, "principal_delete", principal_id="reviewer") == {
        "deleted": True
    }


def test_security_fixed_tenant_rejects_access_to_another_tenant(runtime, monkeypatch):
    from truthound.mcp.approval import ApprovalStore

    group = "security_services"
    identity = prepare(runtime, monkeypatch, group)
    request = dict(
        binding_id="local",
        service_id=identity,
        command="role_delete",
        role_id="other-role",
        idempotency_key="reject-cross-tenant",
    )
    pending = invoke(runtime[0], group + "_execute", **request)
    store = ApprovalStore(runtime[1])
    store.approve(pending["data"]["plan_id"])
    store.close()
    result = invoke(runtime[0], group + "_execute", **request)
    assert result["issues"] == [{"code": "outcome_unknown"}]


def test_security_structured_deny_and_conditional_role_survive_worker_reopen(runtime, monkeypatch):
    adapter, control, _, _, _ = runtime
    (control.parent / "security-settings.json").write_text("{}")
    group = "security_services"
    identity = prepare(runtime, monkeypatch, group)
    deny = {"resource": "dataset", "action": "read", "effect": "deny"}
    command(
        runtime,
        group,
        identity,
        "role_create",
        role_id="reader",
        role_patch={
            "name": "Reader",
            "permissions": ["dataset:read"],
            "conditions": [
                {"field": "resource.owner_id", "operator": "eq", "value": "${principal.id}"}
            ],
        },
    )
    command(
        runtime,
        group,
        identity,
        "principal_create",
        principal_id="reviewer",
        principal_patch={"name": "Reviewer"},
    )
    command(runtime, group, identity, "assign_role", principal_id="reviewer", role_id="reader")
    assert command(
        runtime,
        group,
        identity,
        "check",
        key="check-owned-before",
        principal_id="reviewer",
        resource="dataset:orders",
        action="read",
        resource_attributes={"owner_id": "reviewer"},
    )["allowed"]
    assert not command(
        runtime,
        group,
        identity,
        "check",
        key="check-other-before",
        principal_id="reviewer",
        resource="dataset:orders",
        action="read",
        resource_attributes={"owner_id": "other"},
    )["allowed"]
    command(runtime, group, identity, "permission_add", role_id="reader", permission=deny)
    assert (
        invoke(adapter, "resources_release", resource_id=identity)["execution_status"]
        == "succeeded"
    )
    identity = approved(
        runtime, group + "_open", binding_id="local", idempotency_key="security-reopened"
    )["data"]["resource_id"]
    role = command(runtime, group, identity, "role", key="reopened-role", role_id="reader")["role"]
    assert any(p["effect"] == "deny" for p in role["permissions"])
    assert role["conditions"][0]["field"] == "resource.owner_id"
    assert not command(
        runtime,
        group,
        identity,
        "check",
        key="check-owned-after",
        principal_id="reviewer",
        resource="dataset:orders",
        action="read",
        resource_attributes={"owner_id": "reviewer"},
    )["allowed"]
    command(runtime, group, identity, "permission_remove", role_id="reader", permission=deny)
    assert command(
        runtime,
        group,
        identity,
        "check",
        key="check-owned-restored",
        principal_id="reviewer",
        resource="dataset:orders",
        action="read",
        resource_attributes={"owner_id": "reviewer"},
    )["allowed"]


@pytest.mark.fault
def test_security_condition_cannot_read_unapproved_principal_attributes(runtime, monkeypatch):
    from truthound.mcp.approval import ApprovalStore

    group = "security_services"
    identity = prepare(runtime, monkeypatch, group)
    request = dict(
        binding_id="local",
        service_id=identity,
        command="role_create",
        role_id="probe",
        role_patch={
            "name": "probe",
            "permissions": ["dataset:read"],
            "conditions": [
                {
                    "field": "resource.value",
                    "operator": "eq",
                    "value": "${principal.attributes.secret}",
                }
            ],
        },
        idempotency_key="reject-condition-secret",
    )
    pending = invoke(runtime[0], group + "_execute", **request)
    store = ApprovalStore(runtime[1])
    store.approve(pending["data"]["plan_id"])
    store.close()
    result = invoke(runtime[0], group + "_execute", **request)
    assert result["issues"] == [{"code": "outcome_unknown"}]


def test_plugin_native_lifecycle_custom_hook_and_alias_only_registration(runtime, monkeypatch):
    group = "plugin_services"
    identity = prepare(runtime, monkeypatch, group)
    marker = runtime[2]
    assert command(runtime, group, identity, "catalog") == {"aliases": ["target"]}
    assert "target-setup" not in marker.read_text()
    loaded = command(runtime, group, identity, "load", alias="target")
    assert loaded["state"] == "loaded"
    assert command(runtime, group, identity, "activate", alias="target")["state"] == "active"
    assert "bound-target-capability" in command(runtime, group, identity, "ports")["capabilities"]
    command(runtime, group, identity, "hook_register", alias="local-hook")
    assert command(runtime, group, identity, "hook_fire", payload={"scope": "local"}) == {
        "results": ["local"]
    }
    assert command(runtime, group, identity, "hook_unregister", alias="local-hook") == {
        "removed": 1
    }
    assert command(runtime, group, identity, "deactivate", alias="target")["state"] == "inactive"
    assert command(runtime, group, identity, "unload", alias="target") == {"loaded": False}
    assert marker.read_text().count("target-setup") == 1
    assert marker.read_text().count("target-teardown") == 1
    assert marker.read_text().count("custom-hook") == 1


@pytest.mark.parametrize("group", ["security_services", "plugin_services"])
def test_platform_wire_cannot_choose_python_credentials_or_store_paths(runtime, monkeypatch, group):
    identity = prepare(runtime, monkeypatch, group)
    name = "roles" if group == "security_services" else "catalog"
    for extra in ({"module": "os.system"}, {"password": "private"}, {"database": "/private/state"}):
        result = invoke(
            runtime[0],
            group + "_execute",
            binding_id="local",
            service_id=identity,
            command=name,
            idempotency_key="invalid-platform-fields",
            **extra,
        )
        assert result["issues"] == [{"code": "invalid_arguments"}]
