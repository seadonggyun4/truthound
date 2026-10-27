"""Installed pinned emitter delivers only approved requests to an owned receiver."""

import json
import sys
from types import SimpleNamespace

import pytest

from tests import test_mcp_plugin_bindings as plugins
from tests.test_mcp_brokers import approved
from tests.verification_support import receiver
from truthound.mcp.approval import ApprovalStore

runtime = plugins.runtime
pytestmark = pytest.mark.contract


def prepare(runtime, monkeypatch, endpoint):
    from truthound.mcp import model_process

    adapter, control, marker, _, _ = runtime
    monkeypatch.setattr(
        model_process,
        "sys",
        SimpleNamespace(
            executable=adapter._worker_argv()[0],
            platform=sys.platform,
            _base_executable=sys._base_executable,
        ),
    )
    marker.with_name("lineage-endpoint").write_text(endpoint)
    path = control / "bindings.json"
    settings = json.loads(path.read_text())
    settings["plugins"]["fixture"]["capabilities"]["openlineage-factory"] = "openlineage-factory"
    settings["openlineage_services"] = {"local": "openlineage-factory"}
    path.write_text(json.dumps(settings))
    return approved(
        runtime, "openlineage_services_open", binding_id="local", idempotency_key="openlineage-open"
    )["data"]["resource_id"]


def call(runtime, service, command, key, **fields):
    request = dict(
        binding_id="local", service_id=service, command=command, idempotency_key=key, **fields
    )
    pending = plugins.invoke(runtime[0], "openlineage_services_execute", **request)
    assert pending["issues"] == [{"code": "approval_required"}], pending
    store = ApprovalStore(runtime[1])
    try:
        store.approve(pending["data"]["plan_id"])
    finally:
        store.close()
    result = plugins.invoke(runtime[0], "openlineage_services_execute", **request)
    assert result["issues"] in ([], [{"code": "outcome_unknown"}]), result
    return result, plugins.source_artifact(runtime[0], result["data"]["resource_id"]), request


def test_installed_lineage_unknown_resolution_graph_and_acknowledged_shutdown(runtime, monkeypatch):
    with receiver() as remote:
        service = prepare(runtime, monkeypatch, remote.endpoint)
        assert not remote.events
        result, value, request = call(
            runtime, service, "start", "lineage-start", job_name="synthetic", datasets=["input"]
        )
        assert result["execution_status"] == "succeeded"
        run = value["run"]["run_id"]
        assert remote.events[0]["inputs"][0]["name"] == "synthetic-input"
        remote.status = 503
        result, failure, request = call(
            runtime, service, "complete", "lineage-failed-terminal", run_id=run, datasets=["output"]
        )
        assert result["execution_status"] == "outcome_unknown"
        assert failure == {"run_id": run, "event_type": "COMPLETE", "outcome_unknown": True}
        assert "synthetic-lineage-secret" not in json.dumps(result)
        assert len(remote.events) == 2
        assert plugins.invoke(runtime[0], "openlineage_services_execute", **request) == result
        _, repeated, _ = call(runtime, service, "complete", "lineage-no-retry", run_id=run)
        assert repeated == failure
        assert len(remote.events) == 2
        _, state, _ = call(runtime, service, "status", "lineage-unknown-status")
        assert state["active"] == [{"run_id": run, "event_type": "START"}]
        assert state["unknown"] == [{"run_id": run, "event_type": "COMPLETE"}]
        call(runtime, service, "resolve", "lineage-resolve", run_id=run, delivered=True)
        assert len(remote.events) == 2
        remote.status = 200
        graph = plugins.invoke(
            runtime[0],
            "lineage_create",
            nodes=[
                {"id": "a", "name": "synthetic-a", "node_type": "source"},
                {"id": "b", "name": "synthetic-b", "node_type": "transformation"},
            ],
            edges=[{"source": "a", "target": "b", "edge_type": "derived_from"}],
        )["data"]["resource_id"]
        _, emitted, _ = call(
            runtime, service, "graph", "lineage-graph", graph_id=graph, job_name="graph"
        )
        assert len(emitted["runs"]) == 1 and emitted["runs"][0]["event_type"] == "COMPLETE"
        assert len(remote.events) == 4
        call(runtime, service, "shutdown", "lineage-shutdown")
        assert "lineage-close" in runtime[2].read_text()
        rejected = plugins.invoke(
            runtime[0],
            "openlineage_services_execute",
            binding_id="local",
            service_id=service,
            command="status",
            idempotency_key="lineage-after-close",
        )
        assert rejected["issues"] == [{"code": "invalid_handle"}]


def test_installed_lineage_parent_fail_abort_and_graph_grant_revocation(runtime, monkeypatch):
    with receiver() as remote:
        service = prepare(runtime, monkeypatch, remote.endpoint)
        pending = plugins.invoke(
            runtime[0],
            "openlineage_services_execute",
            binding_id="local",
            service_id=service,
            command="start",
            job_name="pending",
            idempotency_key="lineage-unapproved",
        )
        assert pending["issues"] == [{"code": "approval_required"}]
        assert not remote.events
        _, parent, _ = call(runtime, service, "start", "lineage-parent", job_name="parent")
        parent_id = parent["run"]["run_id"]
        _, child, _ = call(
            runtime, service, "start", "lineage-child", job_name="child", parent_run_id=parent_id
        )
        child_id = child["run"]["run_id"]
        assert remote.events[-1]["run"]["facets"]["parent"]["run"]["runId"] == parent_id
        call(runtime, service, "running", "lineage-running", run_id=child_id)
        call(runtime, service, "fail", "lineage-fail", run_id=child_id)
        call(runtime, service, "abort", "lineage-abort", run_id=parent_id)
        assert [event["eventType"] for event in remote.events] == [
            "START",
            "START",
            "RUNNING",
            "FAIL",
            "ABORT",
        ]
        graph = plugins.invoke(runtime[0], "lineage_create")["data"]["resource_id"]
        call(runtime, service, "graph", "lineage-empty-graph", graph_id=graph, job_name="empty")
        plugins.invoke(runtime[0], "resources_release", resource_id=graph)
        rejected = plugins.invoke(
            runtime[0],
            "openlineage_services_execute",
            binding_id="local",
            service_id=service,
            command="status",
            idempotency_key="lineage-revoked",
        )
        assert rejected["issues"] == [{"code": "invalid_handle"}]
        assert "lineage-close" in runtime[2].read_text()
        assert len(remote.events) == 5
