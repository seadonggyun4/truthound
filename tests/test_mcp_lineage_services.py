"""Installed approved native tracker factory: real contexts, history and reopen."""

import json
import sys
from types import SimpleNamespace

import pytest

from tests import test_mcp_plugin_bindings as plugins
from tests.test_mcp_brokers import approved
from tests.test_mcp_lineage import raw_artifact
from truthound.mcp.approval import ApprovalStore

runtime = plugins.runtime
pytestmark = pytest.mark.contract


def test_installed_tracker_context_failure_history_files_and_shutdown(runtime, monkeypatch):
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
    path = control / "bindings.json"
    settings = json.loads(path.read_text())
    settings["plugins"]["fixture"]["capabilities"]["lineage-factory"] = "lineage-factory"
    settings["lineage_services"] = {"local": "lineage-factory"}
    path.write_text(json.dumps(settings))
    service = approved(
        runtime, "lineage_services_open", binding_id="local", idempotency_key="lineage-open"
    )["data"]["resource_id"]

    def execute(command, key, **fields):
        arguments = dict(
            binding_id="local", service_id=service, idempotency_key=key, command=command, **fields
        )
        pending = plugins.invoke(adapter, "lineage_services_execute", **arguments)
        assert pending["issues"] == [{"code": "approval_required"}], pending
        store = ApprovalStore(control)
        try:
            store.approve(pending["data"]["plan_id"])
        finally:
            store.close()
        reply = plugins.invoke(adapter, "lineage_services_execute", **arguments)
        assert reply["issues"] in ([], [{"code": "native_lineage_incomplete"}]), reply
        assert plugins.invoke(adapter, "lineage_services_execute", **arguments) == reply
        return reply

    def value(reply):
        return plugins.source_artifact(adapter, reply["data"]["resource_id"])

    result = value(execute("run", "lineage-run-success", job_id="success"))
    assert result["native_completed"] and result["state"]["history_count"] == 1
    history = value(execute("history", "lineage-history-success", limit=1))
    assert history["history"][0]["completed"]
    assert history["history"][0]["operation_id"] == result["operation_id"]
    assert value(execute("save", "lineage-backup", file_id="backup"))["saved"]
    failed = execute("run", "lineage-run-failure", job_id="failure")
    assert failed["execution_status"] == "failed"
    history = value(execute("history", "lineage-history-failure"))
    assert not history["history"][-1]["completed"]
    assert history["history"][-1]["error"] == "native_tracking_failed"
    assert "synthetic-private-lineage-secret" not in json.dumps(history, default=str)
    assert str(control) not in json.dumps(history, default=str)
    snapshot = value(execute("snapshot", "lineage-snapshot-failure"))
    assert any(node["id"] == "partial" for node in snapshot["graph"]["nodes"])
    rendered = execute(
        "render", "lineage-html", rendering={"renderer": "d3", "output_format": "html"}
    )
    assert rendered["data"]["media_type"] == "text/html"
    assert b"<!DOCTYPE html>" in raw_artifact(adapter, rendered["data"]["resource_id"])
    failed_image = execute(
        "render",
        "lineage-image-failure",
        rendering={
            "renderer": "graphviz",
            "output_format": "png",
            "settings": {"node_colors": {"file": 'red", image="synthetic.png'}},
        },
    )
    assert failed_image["execution_status"] == "failed"
    assert failed_image["data"]["media_type"] == "application/vnd.truthound.lineage_services+json"
    assert value(failed_image)["native_completed"] is False
    assert value(execute("load", "lineage-restore-backup", file_id="backup"))["loaded"]
    assert value(execute("save", "lineage-commit-restored", file_id="primary"))["saved"]
    execute("shutdown", "lineage-close-first")
    assert "lineage-close" in marker.read_text()
    service = approved(
        runtime, "lineage_services_open", binding_id="local", idempotency_key="lineage-reopen"
    )["data"]["resource_id"]
    state = value(execute("inspect", "lineage-inspect-reopened"))
    assert state["history_count"] == 0  # Native graph files do not serialize context history.
    snapshot = value(execute("snapshot", "lineage-restored-graph"))
    assert any(node["id"] == "output" for node in snapshot["graph"]["nodes"])
    assert not any(node["id"] == "partial" for node in snapshot["graph"]["nodes"])
    execute("shutdown", "lineage-close-last")
