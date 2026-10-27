"""Installed native benchmark workloads preserve measurements, failures and cleanup."""

import json
import sys
from types import SimpleNamespace

import pytest

from tests import test_mcp_plugin_bindings as plugins
from tests.test_mcp_brokers import approved
from truthound.mcp.approval import ApprovalStore

runtime = plugins.runtime
pytestmark = pytest.mark.contract


def prepare(runtime, monkeypatch):
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
    path = control / "bindings.json"
    settings = json.loads(path.read_text())
    settings["plugins"]["fixture"]["capabilities"]["benchmark-factory"] = "benchmark-factory"
    settings["benchmark_services"] = {"local": "benchmark-factory"}
    path.write_text(json.dumps(settings))
    return approved(
        runtime, "benchmark_services_open", binding_id="local", idempotency_key="benchmark-open"
    )["data"]["resource_id"]


def call(runtime, service, command, key, **fields):
    request = dict(
        binding_id="local", service_id=service, command=command, idempotency_key=key, **fields
    )
    pending = plugins.invoke(runtime[0], "benchmark_services_execute", **request)
    assert pending["issues"] == [{"code": "approval_required"}], pending
    store = ApprovalStore(runtime[1])
    try:
        store.approve(pending["data"]["plan_id"])
    finally:
        store.close()
    result = plugins.invoke(runtime[0], "benchmark_services_execute", **request)
    assert result["issues"] in ([], [{"code": "native_benchmark_incomplete"}]), result
    if command == "report" and fields.get("format", "json") != "json":
        import base64

        chunks, offset = [], 0
        while offset is not None:
            page = plugins.invoke(
                runtime[0],
                "artifacts_read",
                resource_id=result["data"]["resource_id"],
                offset=offset,
            )
            assert not page["issues"], page
            chunks.append(base64.b64decode(page["data"]["base64"]))
            offset = page["data"]["next_offset"]
        return result, b"".join(chunks), request
    return result, plugins.source_artifact(runtime[0], result["data"]["resource_id"]), request


def test_native_benchmark_profile_failed_result_comparison_and_shutdown(runtime, monkeypatch):
    service = prepare(runtime, monkeypatch)
    import io

    import polars as pl

    from truthound.benchmark.generators import GeneratorConfig, TabularDataGenerator

    _, generated, _ = call(runtime, service, "run", "benchmark-generated", workload_id="generated")
    assert generated["result"]["rows"] == 8
    _, page, _ = call(
        runtime,
        service,
        "data_page",
        "benchmark-generated-page",
        result_id=generated["result_id"],
        limit=2,
    )
    assert len(page["rows"]) == 2 and page["next_offset"] == 2
    _, ipc, _ = call(
        runtime,
        service,
        "report",
        "benchmark-generated-ipc",
        result_id=generated["result_id"],
        format="ipc",
    )
    expected = TabularDataGenerator().generate(
        GeneratorConfig(row_count=8, seed=0, null_ratio=0.25, duplicate_ratio=0.25)
    )
    assert pl.read_ipc(io.BytesIO(ipc)).equals(expected)
    _, first, _ = call(runtime, service, "run", "benchmark-profile", workload_id="profile")
    assert first["result"]["success"]
    assert first["result"]["metrics"]["timing"]["mean_seconds"] > 0
    assert "parameters" not in first["result"]
    _, comparison, _ = call(
        runtime,
        service,
        "compare",
        "benchmark-compare",
        result_id=first["result_id"],
        baseline_id=first["result_id"],
        comparator_id="native",
    )
    assert not comparison["comparison"]["has_regression"]
    _, threshold, _ = call(
        runtime,
        service,
        "threshold",
        "benchmark-threshold",
        result_id=first["result_id"],
        threshold_id="strict",
    )
    assert threshold["passed"] is False and threshold["violations"]
    failed, value, request = call(
        runtime, service, "run", "benchmark-failed", workload_id="failure"
    )
    assert failed["execution_status"] == "failed"
    assert value["result"]["success"] is False
    assert value["result"]["error"] == "native_benchmark_failed"
    assert "synthetic-private-benchmark-secret" not in json.dumps(value)
    assert plugins.invoke(runtime[0], "benchmark_services_execute", **request) == failed
    _, saved, _ = call(
        runtime, service, "result", "benchmark-inspect-failed", result_id=value["result_id"]
    )
    assert saved == value
    failed_comparison, compared, _ = call(
        runtime,
        service,
        "compare",
        "benchmark-failed-comparison",
        result_id=value["result_id"],
        baseline_id=value["result_id"],
        comparator_id="native",
    )
    assert failed_comparison["execution_status"] == "failed"
    assert compared == {"comparable": False}
    _, report, _ = call(
        runtime, service, "report", "benchmark-report", result_id=first["result_id"], format="json"
    )
    assert report == first["result"]
    _, suite, _ = call(runtime, service, "run", "benchmark-suite", workload_id="suite")
    assert suite["result"]["summary"]["successful"] == 1
    _, duplicate, _ = call(runtime, service, "run", "benchmark-duplicate", workload_id="duplicate")
    incomplete, payload, _ = call(
        runtime,
        service,
        "compare",
        "benchmark-duplicate-compare",
        result_id=duplicate["result_id"],
        baseline_id=suite["result_id"],
        comparator_id="native",
    )
    assert incomplete["execution_status"] == "failed"
    assert payload == {"comparable": False}

    call(
        runtime,
        service,
        "history_save",
        "benchmark-save-baseline",
        history_id="local",
        result_id=suite["result_id"],
    )
    call(runtime, service, "shutdown", "benchmark-restart-close")
    service = approved(
        runtime, "benchmark_services_open", binding_id="local", idempotency_key="benchmark-reopen"
    )["data"]["resource_id"]
    _, restored, _ = call(
        runtime, service, "history_baseline", "benchmark-restored", history_id="local"
    )
    assert restored["result"] == suite["result"]
    _, checked, _ = call(
        runtime,
        service,
        "history_check",
        "benchmark-history-check",
        history_id="local",
        result_id=restored["result_id"],
    )
    assert checked == {"regressions": []}
    _, loaded, _ = call(
        runtime, service, "history_load", "benchmark-history-load", history_id="local"
    )
    assert loaded["loaded"] == 1
    call(
        runtime,
        service,
        "history_threshold",
        "benchmark-history-threshold",
        history_id="local",
        threshold_id="strict",
    )
    _, thresholds, _ = call(
        runtime,
        service,
        "history_thresholds",
        "benchmark-history-thresholds",
        history_id="local",
        result_id=restored["result_id"],
    )
    assert thresholds["checks"]["profile"][0] is False
    call(runtime, service, "release", "benchmark-release", result_id=restored["result_id"])
    call(runtime, service, "shutdown", "benchmark-shutdown")
    assert "benchmark-close" in runtime[2].read_text()
