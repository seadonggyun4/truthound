"""Native rate budget and secret-cache lifecycle through actual approved owners."""

import json
import sys
from types import SimpleNamespace

import pytest

from tests import test_mcp_plugin_bindings as plugins
from tests.test_mcp_brokers import approved

runtime = plugins.runtime
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
    capability = "rate-limit-factory" if group == "rate_limit_services" else "secret-factory"
    document = json.loads((control / "bindings.json").read_text())
    document["plugins"]["fixture"]["capabilities"][capability] = capability
    document[group] = {"local": capability}
    (control / "bindings.json").write_text(json.dumps(document))
    return approved(runtime, group + "_open", binding_id="local", idempotency_key=group + "-open")[
        "data"
    ]["resource_id"]


def command(runtime, group, identity, command_name, *, key=None, replay=False, **args):
    dispatch = (
        (lambda runtime, operation, **request: plugins.invoke(runtime[0], operation, **request))
        if replay
        else approved
    )
    result = dispatch(
        runtime,
        group + "_execute",
        binding_id="local",
        service_id=identity,
        command=command_name,
        idempotency_key=key or group + "-" + command_name,
        **args,
    )
    assert result["execution_status"] == "succeeded", result
    return plugins.source_artifact(runtime[0], result["data"]["resource_id"])


def test_native_rate_tokens_quota_policy_and_exact_replay(runtime, monkeypatch):
    from truthound.ratelimit import RateLimiter
    from truthound.ratelimit.core import RateLimitConfig

    group = "rate_limit_services"
    identity = prepare(runtime, monkeypatch, group)
    first = command(runtime, group, identity, "acquire", subject_id="reviewer", key="rate-first")
    assert first["allowed"] and first["remaining"] == 1
    assert (
        command(
            runtime,
            group,
            identity,
            "acquire",
            subject_id="reviewer",
            key="rate-first",
            replay=True,
        )
        == first
    )
    second = command(runtime, group, identity, "acquire", subject_id="reviewer", key="rate-second")
    assert second["allowed"] and second["remaining"] == 0
    assert not command(
        runtime, group, identity, "acquire", subject_id="reviewer", key="rate-denied"
    )["allowed"]
    native = RateLimiter(RateLimitConfig(requests_per_second=0.0001, burst_size=2))
    assert native.acquire("private").remaining == first["remaining"]
    assert native.acquire("private").remaining == second["remaining"]
    assert not native.acquire("private").allowed
    assert "internal-private-subject" not in json.dumps(first)
    command(runtime, group, identity, "reset", subject_id="reviewer")
    assert command(runtime, group, identity, "peek", subject_id="reviewer")["remaining"] == 2
    command(runtime, group, identity, "quota_register", quota_id="daily")
    usage = command(
        runtime, group, identity, "quota_consume", subject_id="reviewer", quota_id="daily", tokens=2
    )
    assert usage["usage"]["used"] == 2
    assert not command(
        runtime, group, identity, "quota_check", subject_id="reviewer", quota_id="daily"
    )["allowed"]
    command(runtime, group, identity, "quota_reset", subject_id="reviewer", quota_id="daily")
    assert (
        command(runtime, group, identity, "quota_usage", subject_id="reviewer", quota_id="daily")[
            "usage"
        ]
        is None
    )
    command(runtime, group, identity, "policy_register", policy_id="default")
    assert command(runtime, group, identity, "policy_resolve", context_id="local")[
        "policy_ids"
    ] == ["default"]
    assert command(runtime, group, identity, "policy_unregister", policy_id="default")["removed"]


def test_secret_resolution_never_returns_material_and_cache_provider_cleanup_is_native(
    runtime, monkeypatch
):
    group = "secret_services"
    identity = prepare(runtime, monkeypatch, group)
    marker = runtime[2]
    first = command(
        runtime, group, identity, "resolve", reference_id="primary", key="resolve-first"
    )
    assert first["resolved"] and not first["expired"]
    assert set(first) == {"resolved", "expired", "created_at", "expires_at"}
    assert (
        command(
            runtime,
            group,
            identity,
            "resolve",
            reference_id="primary",
            key="resolve-first",
            replay=True,
        )
        == first
    )
    command(runtime, group, identity, "resolve", reference_id="primary", key="resolve-cached")
    events = marker.read_text().splitlines()
    assert events.count("secret-fetch") == 1 and events.count("secret-read") == 2
    command(runtime, group, identity, "cache_clear", provider_id="local")
    refreshed = command(
        runtime, group, identity, "resolve", reference_id="primary", key="resolve-refresh"
    )
    assert refreshed["resolved"] and marker.read_text().splitlines().count("secret-fetch") == 2
    assert command(
        runtime, group, identity, "resolve", reference_id="missing", key="resolve-missing"
    ) == {"resolved": False}
    command(runtime, group, identity, "provider_remove", provider_id="local")
    assert not command(runtime, group, identity, "catalog")["providers"]["local"]["registered"]
    command(runtime, group, identity, "provider_add", provider_id="local")
    catalog = command(runtime, group, identity, "catalog", key="secret-catalog-after")
    assert catalog["providers"]["local"]["registered"]
    for private in (
        "synthetic-not-a-real-credential",
        "internal-private-secret-key",
        "internal-provider-alias",
    ):
        assert private not in json.dumps(
            {"first": first, "refresh": refreshed, "catalog": catalog}, default=str
        )
        assert private not in marker.read_text()
    assert (
        plugins.invoke(runtime[0], "resources_release", resource_id=identity)["execution_status"]
        == "succeeded"
    )
    assert "secret-close" in marker.read_text()


@pytest.mark.parametrize("group", ["rate_limit_services", "secret_services"])
def test_governance_wire_cannot_set_native_keys_credentials_or_provider_paths(
    runtime, monkeypatch, group
):
    identity = prepare(runtime, monkeypatch, group)
    for extra in (
        {"key": "other-user"},
        {"value": "not-a-real-secret"},
        {"path": "/private/data"},
        {"provider_url": "https://example.invalid"},
    ):
        response = plugins.invoke(
            runtime[0],
            group + "_execute",
            binding_id="local",
            service_id=identity,
            command="catalog",
            idempotency_key="reject-governance-fields",
            **extra,
        )
        assert response["issues"] == [{"code": "invalid_arguments"}]


@pytest.mark.fault
def test_secret_provider_error_is_redacted_and_never_reexecuted_by_replay(runtime, monkeypatch):
    from truthound.mcp.approval import ApprovalStore

    group = "secret_services"
    identity = prepare(runtime, monkeypatch, group)
    request = dict(
        binding_id="local",
        service_id=identity,
        command="resolve",
        reference_id="failure",
        idempotency_key="secret-provider-failure",
    )
    pending = plugins.invoke(runtime[0], group + "_execute", **request)
    assert pending["issues"] == [{"code": "approval_required"}]
    store = ApprovalStore(runtime[1])
    try:
        store.approve(pending["data"]["plan_id"])
    finally:
        store.close()
    failed = plugins.invoke(runtime[0], group + "_execute", **request)
    assert failed["execution_status"] == "outcome_unknown"
    assert "synthetic-sensitive-provider-message" not in str(failed)
    events = runtime[2].read_text().splitlines()
    assert events.count("secret-fetch") == 1
    replay = plugins.invoke(runtime[0], group + "_execute", **request)
    assert replay["execution_status"] != "succeeded"
    assert runtime[2].read_text().splitlines().count("secret-fetch") == 1
