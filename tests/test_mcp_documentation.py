"""Actual installed documentation owner: approvals, file receipts and cleanup."""

import base64
import hashlib
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tests import test_mcp_plugin_bindings as plugins
from tests.test_mcp_brokers import approved
from truthound.mcp.approval import ApprovalStore

runtime = plugins.runtime
pytestmark = pytest.mark.contract


def test_installed_documentation_files_failure_receipt_and_shutdown(runtime, monkeypatch, tmp_path):
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
    settings["plugins"]["fixture"]["capabilities"]["documentation-factory"] = (
        "documentation-factory"
    )
    settings["documentation_services"] = {"local": "documentation-factory"}
    path.write_text(json.dumps(settings))
    service = approved(
        runtime, "documentation_services_open", binding_id="local", idempotency_key="docs-open"
    )["data"]["resource_id"]

    def execute(command, key, **fields):
        args = dict(
            binding_id="local", service_id=service, command=command, idempotency_key=key, **fields
        )
        pending = plugins.invoke(adapter, "documentation_services_execute", **args)
        assert pending["issues"] == [{"code": "approval_required"}], pending
        store = ApprovalStore(control)
        try:
            store.approve(pending["data"]["plan_id"])
        finally:
            store.close()
        reply = plugins.invoke(adapter, "documentation_services_execute", **args)
        assert reply["issues"] in ([], [{"code": "native_documentation_incomplete"}]), reply
        assert plugins.invoke(adapter, "documentation_services_execute", **args) == reply
        return reply

    reply = execute("generate", "docs-generate", job_id="local")
    generated = plugins.source_artifact(adapter, reply["data"]["resource_id"])
    assert generated["result"]["success"]
    assert generated["result"]["functions_found"] == 1
    for file in generated["result"]["files"]:
        reply = execute(
            "read",
            "docs-read-" + file["file_id"],
            result_id=generated["result_id"],
            file_id=file["file_id"],
        )
        chunks, offset = [], 0
        while offset is not None:
            page = plugins.invoke(
                adapter, "artifacts_read", resource_id=reply["data"]["resource_id"], offset=offset
            )
            assert not page["issues"], page
            chunks.append(base64.b64decode(page["data"]["base64"]))
            offset = page["data"]["next_offset"]
        assert hashlib.sha256(b"".join(chunks)).hexdigest() == file["sha256"]
    failed = execute("generate", "docs-failure", job_id="failure")
    assert failed["execution_status"] == "failed"
    value = plugins.source_artifact(adapter, failed["data"]["resource_id"])
    assert not value["result"]["success"]
    assert value["result"]["errors"] == ["native_documentation_failed"]
    scaffold = execute("generate", "docs-scaffold", job_id="scaffold")
    files = plugins.source_artifact(adapter, scaffold["data"]["resource_id"])["result"]["files"]
    assert any(entry["name"].endswith("reporter.py") for entry in files)
    assert len({entry["name"] for entry in files}) == len(files)
    assert all(type(entry["executable"]) is bool for entry in files)
    scaffold_value = plugins.source_artifact(adapter, scaffold["data"]["resource_id"])
    for entry in files:
        if not entry["name"].endswith(".py"):
            continue
        reply = execute(
            "read",
            "scaffold-read-" + entry["file_id"],
            result_id=scaffold_value["result_id"],
            file_id=entry["file_id"],
        )
        chunks, offset = [], 0
        while offset is not None:
            page = plugins.invoke(
                adapter, "artifacts_read", resource_id=reply["data"]["resource_id"], offset=offset
            )
            assert not page["issues"], page
            chunks.append(base64.b64decode(page["data"]["base64"]))
            offset = page["data"]["next_offset"]
        content = b"".join(chunks)
        assert hashlib.sha256(content).hexdigest() == entry["sha256"]
        compile(content, entry["name"], "exec")
        if entry["name"].endswith("/reporter.py"):
            script_path = tmp_path / "generated_reporter.py"
            script_path.write_bytes(content)
            script = f"""
import importlib.util, sys, json
import polars as pl
import truthound as th
spec = importlib.util.spec_from_file_location("generated_reporter", {str(script_path)!r})
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
Reporter, Config = module.SyntheticComponentReporter, module.SyntheticComponentReporterConfig
passed = th.check(pl.DataFrame({{"x": [1, 2]}}), validators=["null"])
shown = json.loads(Reporter(Config(include_passed=True)).render(passed))
assert any(row["success"] for row in shown["issues"])
failed = th.check(pl.DataFrame({{"x": [None, 2]}}), validators=["null"])
assert json.loads(Reporter(Config(max_issues=0)).render(failed))["issues"] == []
"""
            executed = subprocess.run(
                [adapter._worker_argv()[0], "-I", "-c", script],
                cwd=tmp_path,
                capture_output=True,
                text=True,
                timeout=30,
            )
            assert executed.returncode == 0, executed.stdout + executed.stderr

    invalid = execute("generate", "docs-invalid-scaffold", job_id="invalid_scaffold")
    assert invalid["execution_status"] == "failed"
    assert not plugins.source_artifact(adapter, invalid["data"]["resource_id"])["result"]["files"]
    validation = execute("generate", "docs-validate", job_id="validate")
    assert validation["execution_status"] == "failed"
    assert plugins.source_artifact(adapter, validation["data"]["resource_id"])["result"]["files"]
    execute("shutdown", "docs-close")
    assert "documentation-close" in marker.read_text()
