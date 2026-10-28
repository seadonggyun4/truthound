"""Shared native-service startup must not consume approval before factory entry."""

import json
from types import SimpleNamespace

import pytest

from tests import test_mcp_plugin_bindings as plugins
from truthound.mcp.approval import ApprovalStore

runtime = plugins.runtime
pytestmark = pytest.mark.fault


@pytest.mark.parametrize("asynchronous", [False, True])
def test_checkpoint_native_import_failure_precedes_approval_consumption(runtime, monkeypatch, asynchronous):
    from tests.test_mcp_checkpoints import prepare

    adapter, control, marker, source, _ = runtime
    operation = "checkpoints_run_async" if asynchronous else "checkpoints_run"
    request = prepare(runtime, monkeypatch, approve=False, asynchronous=asynchronous)
    pending = plugins.invoke(adapter, operation, **request)
    plan = pending["data"]["plan_id"]
    store = ApprovalStore(control)
    try:
        store.approve(plan)
    finally:
        store.close()
    customizer = source.parent.parent / "sitecustomize.py"
    assert not customizer.exists()
    customizer.write_text(
        "import builtins\n"
        "original_import = builtins.__import__\n"
        "def import_without_checkpoint(name, *args, **kwargs):\n"
        "    if name == 'truthound.checkpoint':\n"
        "        raise ImportError('synthetic missing native dependency')\n"
        "    return original_import(name, *args, **kwargs)\n"
        "builtins.__import__ = import_without_checkpoint\n", encoding="utf-8")
    try:
        failed = plugins.invoke(adapter, operation, **request)
        assert failed["issues"] == [{"code": "checkpoint_execution_failed"}], failed
        assert not marker.exists()
        assert adapter._session.approvals().status(plan, session=adapter._session.identity)["state"] == "approved"
        assert adapter._test_checkpoint_worker.process.poll() is not None
    finally:
        customizer.unlink()
    # A dependency repaired before factory entry must not require another grant.
    recovered = plugins.invoke(adapter, operation, **request)
    assert recovered["execution_status"] == "succeeded", recovered
    assert marker.read_text().splitlines().count("checkpoint-enter") == 1


def approved_open(runtime):
    adapter, control, _, _, _ = runtime
    settings = json.loads((control / "bindings.json").read_text())
    settings["plugins"]["fixture"]["capabilities"]["broker-factory"] = "broker-factory"
    settings["brokers"] = {"local-broker": "broker-factory"}
    (control / "bindings.json").write_text(json.dumps(settings))
    request = dict(binding_id="local-broker", idempotency_key="shared-bootstrap")
    pending = plugins.invoke(adapter, "brokers_open", **request)
    assert pending["issues"] == [{"code": "approval_required"}]
    store = ApprovalStore(control)
    try:
        store.approve(pending["data"]["plan_id"])
    finally:
        store.close()
    return request, pending["data"]["plan_id"]


def test_shared_bootstrap_failure_preserves_approval_and_has_no_factory_effect(runtime, monkeypatch):
    from truthound.mcp.brokers import BrokerServices

    request, plan = approved_open(runtime)
    created = []

    class Unready:
        def __init__(self, ttl):
            self.closed = False
            created.append(self)
        def watch_policy(self, callback):
            assert callback()
        async def call(self, payload, timeout):
            assert payload == {"prepare": True}
            raise TimeoutError
        def close(self):
            self.closed = True

    monkeypatch.setattr(BrokerServices, "process_type", Unready)
    for _ in range(2):
        result = plugins.invoke(runtime[0], "brokers_open", **request)
        assert result["issues"] == [{"code": "timeout"}]
        assert created[-1].closed
        assert not runtime[2].exists()
        assert runtime[0]._session.approvals().status(plan, session=runtime[0]._session.identity)["state"] == "approved"
    assert len(created) == 2


def test_shared_readiness_and_factory_share_deadline_and_no_uncertain_replay(runtime, monkeypatch):
    from truthound.mcp import bound_services
    from truthound.mcp.brokers import BrokerServices

    request, _ = approved_open(runtime)
    calls = []
    ticks = iter([100., 101., 102., 107.])
    monkeypatch.setattr(bound_services, "time", SimpleNamespace(monotonic=lambda: next(ticks)))

    class Ready:
        def __init__(self, ttl):
            self.closed = False
        def watch_policy(self, callback):
            assert callback()
        async def call(self, payload, timeout):
            calls.append((payload, timeout))
            if payload == {"prepare": True}:
                return {"ok": True}
            raise TimeoutError
        def close(self):
            self.closed = True

    monkeypatch.setattr(BrokerServices, "process_type", Ready)
    failed = plugins.invoke(runtime[0], "brokers_open", **request)
    assert failed["issues"] == [{"code": "outcome_unknown"}]
    assert len(calls) == 2
    assert calls[0][1] == runtime[0].policy.timeout_seconds - 1
    assert calls[1][1] == runtime[0].policy.timeout_seconds - 7
    assert plugins.invoke(runtime[0], "brokers_open", **request)["issues"] == failed["issues"]
    assert len(calls) == 2


def test_actual_native_workers_prepare_without_loading_operator_factory(runtime, monkeypatch):
    import asyncio
    import importlib
    import sys
    from truthound.mcp import model_process

    adapter = runtime[0]
    monkeypatch.setattr(model_process, "sys", SimpleNamespace(executable=adapter._worker_argv()[0],
        platform=sys.platform, _base_executable=sys._base_executable))
    processes = [
        ("brokers", "BrokerProcess"), ("profiler_services", "ProfilerProcess"),
        ("execution_services", "ExecutionProcess"), ("store_services", "StoreProcess"),
        ("streaming_store_services", "StreamingStoreProcess"), ("store_utilities", "StoreUtilityProcess"),
        ("platform_services", "PlatformProcess"), ("governance", "GovernanceProcess"),
        ("tenancy", "TenantProcess"), ("administration", "AdministrationProcess"),
        ("ai_services", "AIProcess"), ("ml_services", "MLProcess"),
        ("lineage_services", "LineageProcess"), ("openlineage_services", "OpenLineageProcess"),
        ("benchmark_services", "BenchmarkProcess"), ("documentation_services", "DocumentationProcess"),
        ("dashboards", "DashboardProcess"),
        ("sources", "SourceProcess"), ("checkpoints", "CheckpointProcess"),
        ("schema_monitor", "MonitorProcess"),
    ]
    for module, name in processes:
        process_type = getattr(importlib.import_module("truthound.mcp." + module), name)
        worker = process_type(adapter.policy.handle_ttl_seconds)
        try:
            reply = asyncio.run(worker.call({"prepare": True}, adapter.policy.timeout_seconds))
            assert reply == ({"ok": True, "completed": True} if module == "sources" else {"ok": True}), module
            assert not runtime[2].exists(), module
        finally:
            worker.close()
        assert worker.process.poll() is not None, module


def test_parent_artifact_failure_before_factory_request_preserves_approval(runtime, monkeypatch):
    from pathlib import Path
    from truthound.mcp.brokers import BrokerServices

    request, plan = approved_open(runtime)
    calls, workers = [], []

    class Ready:
        def __init__(self, ttl):
            self.closed = False
            workers.append(self)
        def watch_policy(self, callback):
            assert callback()
        async def call(self, payload, timeout):
            calls.append(payload)
            assert payload == {"prepare": True}
            return {"ok": True}
        def close(self):
            self.closed = True

    original = Path.write_text
    root = runtime[0]._session.resources.root

    def failed_write(path, *args, **kwargs):
        if path.parent == root:
            raise OSError(28, "synthetic disk full")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(BrokerServices, "process_type", Ready)
    monkeypatch.setattr(Path, "write_text", failed_write)
    result = plugins.invoke(runtime[0], "brokers_open", **request)
    assert result["execution_status"] not in {"succeeded", "outcome_unknown"}
    assert runtime[0]._session.approvals().status(plan, session=runtime[0]._session.identity)["state"] == "approved"
    assert calls == [{"prepare": True}] and workers[0].closed
    assert not runtime[2].exists()


@pytest.mark.parametrize("group", ["source", "checkpoint", "monitor"])
def test_other_native_factory_bootstrap_failure_preserves_approval(runtime, monkeypatch, group):
    import importlib

    adapter, control, marker, _, _ = runtime
    document = json.loads((control / "bindings.json").read_text())
    capability = {"source": "source-factory", "checkpoint": "checkpoint-factory", "monitor": "schema-factory"}[group]
    document["plugins"]["fixture"]["capabilities"][capability] = capability
    if group == "monitor":
        document["schema_monitors"] = {"local": {"source": capability}}
        operation, module, name = "schema_monitor_open", "schema_monitor", "MonitorProcess"
    elif group == "source":
        document["sources"] = {"local": capability}
        operation, module, name = "sources_open_binding", "sources", "SourceProcess"
    else:
        document["checkpoints"] = {"local": capability}
        operation, module, name = "checkpoints_run", "checkpoints", "CheckpointProcess"
    (control / "bindings.json").write_text(json.dumps(document))
    request = dict(binding_id="local", idempotency_key="bootstrap-" + group)
    pending = plugins.invoke(adapter, operation, **request)
    assert pending["issues"] == [{"code": "approval_required"}], pending
    plan = pending["data"]["plan_id"]
    store = ApprovalStore(control)
    try:
        store.approve(plan)
    finally:
        store.close()
    created = []

    class Unready:
        def __init__(self, ttl):
            self.closed = False
            created.append(self)
        def watch_policy(self, callback):
            assert callback()
        async def call(self, payload, timeout):
            assert payload == {"prepare": True}
            raise TimeoutError
        def close(self):
            self.closed = True

    monkeypatch.setattr(importlib.import_module("truthound.mcp." + module), name, Unready)
    result = plugins.invoke(adapter, operation, **request)
    assert result["issues"] == [{"code": "timeout"}]
    assert adapter._session.approvals().status(plan, session=adapter._session.identity)["state"] == "approved"
    assert created[0].closed and not marker.exists()


def test_actual_missing_interpreter_preserves_approval_and_can_retry_without_reapproval(runtime, monkeypatch, tmp_path):
    import sys
    from truthound.mcp import model_process

    request, plan = approved_open(runtime)
    invalid = str(tmp_path / "missing-python")
    monkeypatch.setattr(model_process, "sys", SimpleNamespace(executable=invalid,
        platform=sys.platform, _base_executable=invalid))
    failed = plugins.invoke(runtime[0], "brokers_open", **request)
    assert failed["execution_status"] not in {"succeeded", "outcome_unknown"}
    assert runtime[0]._session.approvals().status(plan, session=runtime[0]._session.identity)["state"] == "approved"
    assert not runtime[2].exists()
    monkeypatch.setattr(model_process, "sys", SimpleNamespace(executable=runtime[0]._worker_argv()[0],
        platform=sys.platform, _base_executable=sys._base_executable))
    opened = plugins.invoke(runtime[0], "brokers_open", **request)
    assert opened["execution_status"] == "succeeded", opened
    assert plugins.invoke(runtime[0], "resources_release", resource_id=opened["data"]["resource_id"])["execution_status"] == "succeeded"
