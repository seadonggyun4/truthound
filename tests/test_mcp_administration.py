"""Local native audit storage and telemetry exporters preserve observable effects."""

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
    capability = "audit-factory" if group == "audit_services" else "telemetry-factory"
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


def test_audit_native_sqlite_record_query_and_explicit_retention(runtime, monkeypatch):
    from truthound.audit.storage import SQLiteAuditStorage

    group = "audit_services"
    identity = prepare(runtime, monkeypatch, group)
    event = command(
        runtime,
        group,
        identity,
        "record",
        action="review",
        outcome="success",
        data={"scope": "local"},
    )["event"]
    assert event["actor"]["id"] == "mcp-approved-operator"
    assert command(runtime, group, identity, "read", event_id=event["id"])["event"] == event
    queried = command(
        runtime, group, identity, "query", event_types=["custom"], actor_id="mcp-approved-operator"
    )
    assert queried["events"] == [event]
    native = SQLiteAuditStorage(str(runtime[1].parent / "audit.sqlite"))
    try:
        assert native.read(event["id"]).to_dict() == event
        assert command(runtime, group, identity, "count")["count"] == native.count() == 1
        command(runtime, group, identity, "flush")
        assert (
            command(runtime, group, identity, "delete_before", before="2100-01-01T00:00:00+00:00")[
                "removed"
            ]
            == 1
        )
        assert native.count() == 0
    finally:
        native.close()


def test_telemetry_native_metric_export_span_and_failed_flush_are_preserved(runtime, monkeypatch):
    from truthound.observability.metrics import Counter, MetricsRegistry

    group = "telemetry_services"
    identity = prepare(runtime, monkeypatch, group)
    actual = command(
        runtime,
        group,
        identity,
        "metric",
        metric_id="requests",
        update="inc",
        value=2.0,
        labels={"kind": "local"},
    )
    counter = Counter("requests", labels=["kind"])
    registry = MetricsRegistry()
    registry.register(counter)
    counter.inc(2, kind="local")
    assert actual["metrics"] == registry.collect_all()
    command(runtime, group, identity, "export", exporter_id="local")
    assert runtime[2].read_text().count("metric-export") == 1
    span = command(
        runtime, group, identity, "span", span_name="approved-review", attributes={"scope": "local"}
    )
    assert span["trace_id"] and span["span_id"]
    assert runtime[2].read_text().count("span-export") == 1
    assert command(runtime, group, identity, "flush") == {"flushed": False}
    assert command(runtime, group, identity, "shutdown") == {"shutdown": True}


@pytest.mark.parametrize("group", ["audit_services", "telemetry_services"])
def test_administration_cannot_override_operator_transport_or_actor(runtime, monkeypatch, group):
    identity = prepare(runtime, monkeypatch, group)
    adapter = runtime[0]
    operation = group + "_execute"
    command_name = "query" if group == "audit_services" else "collect"
    for extra in (
        {"endpoint": "private://destination"},
        {"actor": "admin"},
        {"sql": "SELECT secret"},
    ):
        result = invoke(
            adapter,
            operation,
            binding_id="local",
            service_id=identity,
            command=command_name,
            idempotency_key="invalid-administration",
            **extra,
        )
        assert result["issues"] == [{"code": "invalid_arguments"}]
