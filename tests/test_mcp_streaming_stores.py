"""Native stream persistence and sync/async ownership via an installed factory."""

import json
import sys
from types import SimpleNamespace

import pytest

from tests import test_mcp_plugin_bindings as plugins
from tests.test_mcp_brokers import approved

runtime = plugins.runtime
pytestmark = pytest.mark.contract


def prepare(runtime, monkeypatch, mode):
    from truthound.mcp import model_process

    adapter, control, marker, _, _ = runtime
    monkeypatch.setattr(model_process, "sys", SimpleNamespace(
        executable=adapter._worker_argv()[0], platform=sys.platform,
        _base_executable=sys._base_executable,
    ))
    marker.with_name("streaming-store-mode").write_text(mode)
    document = json.loads((control / "bindings.json").read_text())
    document["plugins"]["fixture"]["capabilities"]["streaming-store-factory"] = "streaming-store-factory"
    document["streaming_store_services"] = {"local": "streaming-store-factory"}
    (control / "bindings.json").write_text(json.dumps(document))
    return reopen(runtime, "initial-open")


def reopen(runtime, key):
    return approved(runtime, "streaming_store_services_open", binding_id="local", idempotency_key=key)["data"]["resource_id"]


def call(runtime, service, command, key=None, **arguments):
    response = approved(runtime, "streaming_store_services_execute", binding_id="local", service_id=service,
                        command=command, idempotency_key=key or "stream-" + command, **arguments)
    value = plugins.source_artifact(runtime[0], response["data"]["resource_id"])
    assert "private-native-run" not in json.dumps(value)
    assert "private-native-asset" not in json.dumps(value)
    return value


@pytest.mark.parametrize("mode,asynchronous", [("filesystem", False), ("sqlite", True)])
def test_installed_native_stream_durable_result_and_fresh_owner_read(runtime, monkeypatch, mode, asynchronous):
    service = prepare(runtime, monkeypatch, mode)
    current = call(runtime, service, "session_create", run_id="local")
    writer = call(runtime, service, "writer_open", session_id=current["session_id"], asynchronous=asynchronous)["writer_id"]
    records = [{"validator_name": "native", "success": False, "count": 3, "severity": "high", "details": {"flag": False}},
               {"validator_name": "native", "success": True, "count": 0}]
    written = call(runtime, service, "writer_write", writer_id=writer, records=records)
    assert written["metrics"]["records_written"] == 2
    call(runtime, service, "writer_close", writer_id=writer)
    final = call(runtime, service, "finalize", session_id=current["session_id"])
    assert final["statistics"]["total_validators"] == 2
    assert final["statistics"]["total_issues"] == 3
    assert final["results"][0]["success"] is False
    assert final["results"][0]["details"] == {"flag": False}
    plugins.invoke(runtime[0], "resources_release", resource_id=service)
    assert "streaming-store-close" in runtime[2].read_text()
    fresh = reopen(runtime, "fresh-open")
    restored = call(runtime, fresh, "session_open", key="fresh-session", run_id="local")
    assert restored["status"] == "completed"
    assert restored["config"]["compression"] == "gzip"
    reader = call(runtime, fresh, "reader_open", run_id="local", asynchronous=asynchronous)["reader_id"]
    assert call(runtime, fresh, "reader_read", reader_id=reader, limit=1)["records"][0]["success"] is False
    status = call(runtime, fresh, "reader_status", reader_id=reader)
    assert status["metrics"]["records_read"] == 2 and not status["closed"]
    call(runtime, fresh, "reader_reset", reader_id=reader)
    assert call(runtime, fresh, "reader_read", key="stream-reread", reader_id=reader, limit=2)["count"] == 2
    call(runtime, fresh, "reader_close", reader_id=reader)
    chunks = call(runtime, fresh, "chunks_list", run_id="local")["chunks"]
    assert call(runtime, fresh, "chunk_get", chunk_id=chunks[0]["chunk_id"])["records"][0]["count"] == 3
    assert call(runtime, fresh, "chunks_delete", run_id="local")["deleted"] == 1
    assert not call(runtime, fresh, "stats", run_id="local")["exists"]


def test_schema_refuses_paths_and_irrelevant_fields_before_approval(runtime, monkeypatch):
    service = prepare(runtime, monkeypatch, "filesystem")
    for arguments in ({"command": "session_create", "run_id": "local", "path": "/not-selected"},
                      {"command": "session_status", "session_id": "unknown", "records": []}):
        result = plugins.invoke(runtime[0], "streaming_store_services_execute", binding_id="local",
                                service_id=service, idempotency_key="bad-shape", **arguments)
        assert result["issues"] == [{"code": "invalid_arguments"}]


@pytest.mark.parametrize("mode", ["filesystem", "sqlite"])
def test_actual_installed_owner_crash_resumes_checkpoint_without_duplicate_write(runtime, monkeypatch, mode):
    from truthound.mcp.service import MCPService

    service = prepare(runtime, monkeypatch, mode)
    current = call(runtime, service, "session_create", run_id="local")
    writer = call(runtime, service, "writer_open", session_id=current["session_id"])["writer_id"]
    records = [{"validator_name": "before", "success": False, "count": 2},
               {"validator_name": "before", "success": True}]
    first = call(runtime, service, "writer_write", writer_id=writer, records=records)
    # An identical committed approval must return its receipt without appending.
    replay = plugins.invoke(runtime[0], "streaming_store_services_execute", binding_id="local", service_id=service,
                            command="writer_write", idempotency_key="stream-writer_write", writer_id=writer, records=records)
    assert plugins.source_artifact(runtime[0], replay["data"]["resource_id"]) == first
    worker = runtime[0]._session._administration["streaming_store_services"].workers[service]
    worker.process.kill()  # This test's exact recorded native owner only.
    worker.process.wait(timeout=5)
    runtime[0].close()
    recovered = MCPService(runtime[0].policy)
    recovered._worker_argv = runtime[0]._worker_argv
    state = (recovered, *runtime[1:])
    try:
        fresh = reopen(state, "crashed-new-owner")
        opened = call(state, fresh, "session_open", key="crashed-session-open", run_id="local")
        assert opened["metrics"]["records_written"] == 2
        resumed = call(state, fresh, "session_resume", session_id=opened["session_id"])
        assert resumed["status"] == "active"
        next_writer = call(state, fresh, "writer_open", key="crashed-writer-open", session_id=opened["session_id"])["writer_id"]
        call(state, fresh, "writer_write", key="crashed-write", writer_id=next_writer,
             records=[{"validator_name": "after", "success": True}])
        final = call(state, fresh, "finalize", session_id=opened["session_id"])
        assert final["statistics"]["total_validators"] == 3
        assert [value["validator_name"] for value in final["results"]] == ["before", "before", "after"]
    finally:
        recovered.close()


def test_canonical_run_input_and_source_release_revoke_native_stream_owner(runtime, monkeypatch):
    service = prepare(runtime, monkeypatch, "filesystem")
    current = call(runtime, service, "session_create", run_id="local")
    writer = call(runtime, service, "writer_open", session_id=current["session_id"])["writer_id"]
    result = plugins.invoke(runtime[0], "assets_validate", asset_id=runtime[4],
                            checks=[{"validator": "not_null", "settings": {"columns": ["x"]}}])
    assert result["execution_status"] == "succeeded", result
    result_id = result["data"]["artifact"]["resource_id"]
    call(runtime, service, "writer_write_run", writer_id=writer, result_id=result_id)
    status = call(runtime, service, "writer_status", writer_id=writer)
    assert status["buffered"] == 1 and status["metrics"]["records_written"] == 0
    plugins.invoke(runtime[0], "resources_release", resource_id=result_id)
    denied = plugins.invoke(runtime[0], "streaming_store_services_execute", binding_id="local", service_id=service,
                            command="writer_flush", writer_id=writer, idempotency_key="revoked-flush")
    assert denied["execution_status"] == "failed"
    assert denied["issues"][0]["code"] in {"invalid_handle", "access_denied", "resource_expired"}
    run_path = runtime[2].with_name("native-stream") / "private-native-run"
    assert not list(run_path.glob("*.csv.gz"))
    assert json.loads((run_path / "_manifest.json").read_text())["status"] == "aborted"


@pytest.mark.parametrize("resolution", ["adopt", "discard"])
@pytest.mark.parametrize("backend", ["filesystem", "sqlite", "s3"])
def test_installed_crash_recovery_uses_opaque_evidence_and_explicit_resolution(runtime, monkeypatch, resolution, backend):
    service = prepare(runtime, monkeypatch, backend + "-uncheckpointed")
    current = call(runtime, service, "session_create", run_id="local")
    writer = call(runtime, service, "writer_open", session_id=current["session_id"])["writer_id"]
    call(runtime, service, "writer_write", writer_id=writer, records=[{"validator_name": "uncertain", "success": False, "count": 4}])
    worker = runtime[0]._session._administration["streaming_store_services"].workers[service]
    worker.process.kill()
    worker.process.wait(timeout=5)
    fresh = reopen(runtime, "unconfirmed-fresh-owner")
    state = call(runtime, fresh, "recovery_status", run_id="local")
    assert state["unconfirmed_chunks"] == 1 and state["confirmed_chunks"] == 0
    assert "fingerprint" not in state and "path" not in state
    result = call(runtime, fresh, "recovery_reconcile", recovery_id=state["recovery_id"], resolution=resolution)
    assert result["status"] == "paused"
    current = call(runtime, fresh, "session_open", key="unconfirmed-open-session", run_id="local")
    call(runtime, fresh, "session_resume", session_id=current["session_id"])
    writer = call(runtime, fresh, "writer_open", key="unconfirmed-new-writer", session_id=current["session_id"])["writer_id"]
    call(runtime, fresh, "writer_write", key="unconfirmed-after-write", writer_id=writer, records=[{"validator_name": "after", "success": True}])
    final = call(runtime, fresh, "finalize", session_id=current["session_id"])
    assert final["statistics"]["total_validators"] == (2 if resolution == "adopt" else 1)
    assert final["statistics"]["total_issues"] == (4 if resolution == "adopt" else 0)


def test_restored_native_metric_error_bodies_stay_private(runtime, monkeypatch):
    service = prepare(runtime, monkeypatch, "filesystem")
    call(runtime, service, "session_create", run_id="local")
    plugins.invoke(runtime[0], "resources_release", resource_id=service)
    path = runtime[2].with_name("native-stream") / "private-native-run" / "_manifest.json"
    data = json.loads(path.read_text())
    secret = "synthetic-private-provider-credential-error"
    data["metrics"]["errors"] = [secret]
    path.write_text(json.dumps(data))
    fresh = reopen(runtime, "legacy-errors-fresh-owner")
    restored = call(runtime, fresh, "session_open", key="legacy-error-session", run_id="local")
    assert secret not in json.dumps(restored)
    assert restored["metrics"]["error_count"] == 1
    assert json.loads(path.read_text())["metrics"]["errors"] == [secret]


@pytest.mark.parametrize("phase", ["create", "part"])
def test_installed_multipart_process_loss_requires_owned_or_selected_receipt(runtime, monkeypatch, phase):
    from truthound.mcp.approval import ApprovalStore
    from truthound.mcp.service import MCPService

    service = prepare(runtime, monkeypatch, "s3-multipart-" + phase)
    current = call(runtime, service, "session_create", run_id="local")
    writer = call(runtime, service, "writer_open", session_id=current["session_id"])["writer_id"]
    request = dict(binding_id="local",service_id=service,command="writer_write",writer_id=writer,
                   records=[{"validator_name":"synthetic","success":False}],idempotency_key="multipart-crash-write")
    pending = plugins.invoke(runtime[0], "streaming_store_services_execute", **request)
    approval = ApprovalStore(runtime[1])
    try:
        approval.approve(pending["data"]["plan_id"])
    finally:
        approval.close()
    failed = plugins.invoke(runtime[0], "streaming_store_services_execute", **request)
    assert failed["execution_status"] == "outcome_unknown", failed
    runtime[0].close()
    recovered = MCPService(runtime[0].policy)
    recovered._worker_argv = runtime[0]._worker_argv
    state = (recovered, *runtime[1:])
    try:
        fresh = reopen(state, "multipart-fresh-owner")
        inspected = call(state, fresh, "recovery_status", run_id="local")
        assert inspected["multipart_uploads"] == 1
        assert inspected["unresolved_upload_intents"] == (1 if phase == "create" else 0)
        assert "synthetic-upload-" not in str(inspected) and "private-native-run" not in str(inspected)
        parameters = {"receipt_id":"recovered"} if phase == "create" else {}
        result = call(state, fresh, "multipart_reconcile", recovery_id=inspected["recovery_id"], **parameters)
        assert result["multipart_uploads"] == 0 and result["aborted_uploads"] == 1
        assert "fingerprint" not in result and "synthetic-upload-" not in str(result)
        opened = call(state, fresh, "session_open", run_id="local")
        call(state, fresh, "session_resume", session_id=opened["session_id"])
        writer = call(state, fresh, "writer_open", key="recovered-writer", session_id=opened["session_id"])["writer_id"]
        call(state, fresh, "writer_write", key="after-recovery", writer_id=writer, records=[{"validator_name":"after","success":True}])
        final = call(state, fresh, "finalize", session_id=opened["session_id"])
        assert [row["validator_name"] for row in final["results"]] == ["after"]
    finally:
        recovered.close()
