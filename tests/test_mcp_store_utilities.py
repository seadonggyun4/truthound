"""Installed native codec roundtrip, persisted ciphertext and input revocation."""

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
    document = json.loads((control / "bindings.json").read_text())
    document["plugins"]["fixture"]["capabilities"]["store-utility-factory"] = (
        "store-utility-factory"
    )
    document["store_utilities"] = {"local": "store-utility-factory"}
    (control / "bindings.json").write_text(json.dumps(document))
    return reopen(runtime, "utility-open-local")


def reopen(runtime, key):
    return approved(runtime, "store_utilities_open", binding_id="local", idempotency_key=key)[
        "data"
    ]["resource_id"]


def call(runtime, service, command, key=None, failed=False, **arguments):
    request = dict(
        binding_id="local",
        service_id=service,
        command=command,
        idempotency_key=key or "utility-" + command,
        **arguments,
    )
    pending = plugins.invoke(runtime[0], "store_utilities_execute", **request)
    assert pending["issues"] == [{"code": "approval_required"}], pending
    store = ApprovalStore(runtime[1])
    try:
        store.approve(pending["data"]["plan_id"])
    finally:
        store.close()
    response = plugins.invoke(runtime[0], "store_utilities_execute", **request)
    assert response["execution_status"] == ("failed" if failed else "succeeded"), response
    payload = plugins.source_artifact(runtime[0], response["data"]["resource_id"])
    assert "synthetic-private-" not in str(payload)
    return response, payload


@pytest.mark.parametrize("kind", ["compression", "encryption", "adaptive", "chunked_compression", "chunked_encryption"])
def test_native_codec_slot_roundtrip_survives_acknowledged_shutdown_and_new_owner(
    runtime, monkeypatch, kind
):
    service = prepare(runtime, monkeypatch)
    _, seed = call(runtime, service, "payload_status", payload_id="seed")
    _, encoded = call(
        runtime, service, "codec_process", payload_id="seed", codec_kind=kind, codec_id="local"
    )
    call(runtime, service, "slot_save", payload_id=encoded["payload_id"], slot_id="local")
    if kind in {"encryption", "chunked_encryption"}:
        assert (
            b"synthetic-storage-payload"
            not in runtime[2].with_name("operator-payload.bin").read_bytes()
        )
    call(runtime, service, "shutdown")
    assert "utility-close" in runtime[2].read_text()
    plugins.invoke(runtime[0], "resources_release", resource_id=service)
    fresh = reopen(runtime, "utility-fresh-owner")
    _, restored = call(runtime, fresh, "slot_load", slot_id="local")
    assert restored["sha256"] == encoded["sha256"]
    _, decoded = call(
        runtime,
        fresh,
        "codec_reverse",
        payload_id=restored["payload_id"],
        codec_kind=kind,
        codec_id="local",
    )
    assert decoded["sha256"] == seed["sha256"] and decoded["size_bytes"] == seed["size_bytes"]
    exported, data = call(runtime, fresh, "payload_export", payload_id=decoded["payload_id"])
    assert data["data"] == b"synthetic-storage-payload" * 100
    importer = reopen(runtime, "utility-import-owner")
    _, imported = call(
        runtime, importer, "payload_import", artifact_id=exported["data"]["resource_id"]
    )
    assert imported["sha256"] == seed["sha256"]
    call(runtime, fresh, "slot_delete", slot_id="local")
    assert not runtime[2].with_name("operator-payload.bin").exists()


def test_uncertain_pressure_cannot_be_reacquired_until_explicit_native_reset(runtime, monkeypatch):
    runtime[2].with_name("utility-pressure-failed").touch()
    service = prepare(runtime, monkeypatch)
    response, state = call(runtime, service, "pressure_acquire", pressure_id="local", failed=True)
    assert state["uncertain"] and response["issues"] == [{"code": "native_pressure_incomplete"}]
    call(
        runtime,
        service,
        "pressure_acquire",
        key="pressure-repeat-unknown",
        pressure_id="local",
        failed=True,
    )
    assert runtime[2].read_text().count("utility-acquire") == 1
    runtime[2].with_name("utility-pressure-failed").unlink()
    _, reset = call(runtime, service, "pressure_reset", pressure_id="local")
    assert not reset["uncertain"]
    _, acquired = call(
        runtime, service, "pressure_acquire", key="pressure-after-reset", pressure_id="local"
    )
    assert acquired["acquired"] is True and runtime[2].read_text().count("utility-acquire") == 2


def test_captured_asset_revocation_reaps_owner_and_invalidates_export(runtime, monkeypatch):
    service = prepare(runtime, monkeypatch)
    _, captured = call(runtime, service, "payload_capture", asset_id=runtime[4])
    exported, _ = call(runtime, service, "payload_export", payload_id=captured["payload_id"])
    _, active = call(runtime, service, "stream_open", codec_kind="encryption", codec_id="local", direction="encode")
    call(runtime, service, "stream_write", stream_id=active["stream_id"], payload_id=captured["payload_id"])
    worker = runtime[0]._session._administration["store_utilities"].workers[service]
    plugins.invoke(runtime[0], "resources_release", resource_id=runtime[4])
    assert worker.closed and worker.process.poll() is not None
    rejected = plugins.invoke(
        runtime[0], "artifacts_read", resource_id=exported["data"]["resource_id"]
    )
    assert rejected["issues"] == [{"code": "invalid_handle"}]


def test_native_key_rotation_archive_and_cipher_restore_without_key_disclosure(
    runtime, monkeypatch
):
    service = prepare(runtime, monkeypatch)
    _, created = call(runtime, service, "key_create", key_id="local")
    assert created["version"] == 1 and "key_material" not in created
    call(runtime, service, "key_apply", key_id="local", codec_id="local")
    _, encrypted = call(
        runtime,
        service,
        "codec_process",
        payload_id="seed",
        codec_kind="encryption",
        codec_id="local",
    )
    call(runtime, service, "slot_save", slot_id="local", payload_id=encrypted["payload_id"])
    _, rotated = call(runtime, service, "key_rotate", key_id="local")
    assert rotated["version"] == 2
    _, history = call(runtime, service, "key_history", key_id="local")
    assert [key["version"] for key in history["keys"]] == [2, 1]
    call(runtime, service, "shutdown")
    plugins.invoke(runtime[0], "resources_release", resource_id=service)
    fresh = reopen(runtime, "key-fresh-owner")
    _, status = call(runtime, fresh, "key_status", key_id="local")
    assert status["version"] == 2
    call(
        runtime,
        fresh,
        "key_apply",
        key="fresh-archived-key-apply",
        key_id="local",
        key_version=1,
        codec_id="local",
    )
    _, loaded = call(runtime, fresh, "slot_load", slot_id="local")
    _, restored = call(
        runtime,
        fresh,
        "codec_reverse",
        payload_id=loaded["payload_id"],
        codec_kind="encryption",
        codec_id="local",
    )
    _, plaintext = call(runtime, fresh, "payload_export", payload_id=restored["payload_id"])
    assert plaintext["data"] == b"synthetic-storage-payload" * 100
    call(runtime, fresh, "key_delete", key_id="local", key_version=1)
    call(runtime, fresh, "key_delete", key="delete-current-key", key_id="local")
    _, missing = call(runtime, fresh, "key_status", key="deleted-key-status", key_id="local")
    assert missing["found"] is False


def test_unconfirmed_key_creation_is_not_repeated_and_status_retains_uncertainty(
    runtime, monkeypatch
):
    runtime[2].with_name("utility-key-receipt-failed").touch()
    service = prepare(runtime, monkeypatch)
    response, state = call(runtime, service, "key_create", key_id="local", failed=True)
    assert state["uncertain"] and response["issues"] == [{"code": "native_key_incomplete"}]
    call(
        runtime,
        service,
        "key_create",
        key="unknown-key-second-attempt",
        key_id="local",
        failed=True,
    )
    assert runtime[2].read_text().count("utility-key-key_created") == 1
    _, status = call(runtime, service, "key_status", key_id="local")
    assert status["uncertain"] and status["version"] == 1


def test_native_zero_ttl_key_is_expired_and_cleanup_counts_actual_deletion(runtime, monkeypatch):
    runtime[2].with_name("utility-key-expired").touch()
    service = prepare(runtime, monkeypatch)
    _, created = call(runtime, service, "key_create", key_id="local")
    assert created["expired"] is True
    _, deleted = call(runtime, service, "key_cleanup_expired", key_id="local")
    assert deleted["deleted"] == 1
    _, missing = call(runtime, service, "key_status", key_id="local")
    assert missing["found"] is False


@pytest.mark.parametrize("kind", ["chunked_compression", "chunked_encryption"])
def test_selected_native_chunk_range_and_adaptive_analysis_expose_only_public_fields(runtime, monkeypatch, kind):
    import hashlib

    service = prepare(runtime, monkeypatch)
    _, encoded = call(runtime, service, "codec_process", payload_id="seed", codec_kind=kind, codec_id="local")
    _, part = call(runtime, service, "codec_range", payload_id=encoded["payload_id"], codec_kind=kind, codec_id="local", start=7, end=80)
    assert part["size_bytes"] == 73
    assert part["sha256"] == hashlib.sha256((b"synthetic-storage-payload" * 100)[7:80]).hexdigest()
    _, analysis = call(runtime, service, "codec_analyze", payload_id="seed", codec_kind="adaptive", codec_id="local")
    assert analysis["analysis"]["sample_ratios"]["gzip"] > 1
    assert set(analysis["analysis"]) == {"size", "entropy", "repetition_score", "pattern_score", "compressibility_estimate", "detected_type", "sample_ratios", "sample_errors", "analysis_budget_exhausted"}
    _, recommended = call(runtime, service, "codec_recommend", payload_id="seed", codec_kind="adaptive", codec_id="local")
    assert recommended["profiles"][0]["algorithm"] == "gzip"
    assert "reason" not in recommended["profiles"][0]
    call(runtime, service, "shutdown")


@pytest.mark.parametrize("kind", ["compression", "encryption"])
def test_incremental_codec_session_observes_finalization_and_decoder_lifetime(runtime, monkeypatch, kind):
    import hashlib

    service = prepare(runtime, monkeypatch)
    _, encoder = call(runtime, service, "stream_open", codec_kind=kind, codec_id="local", direction="encode")
    encoded_id = encoder["stream_id"]
    _, pending = call(runtime, service, "stream_write", stream_id=encoded_id, payload_id="seed")
    assert pending["state"] == "open" and pending["output"] is None
    _, final = call(runtime, service, "stream_finalize", stream_id=encoded_id)
    assert final["state"] == "finalized"
    encoded = final["output"]["payload_id"]
    _, decoder = call(runtime, service, "stream_open", key="decoder-open", codec_kind=kind, codec_id="local", direction="decode", **({"payload_id": encoded} if kind == "encryption" else {}))
    decoded_id = decoder["stream_id"]
    if kind == "compression":
        _, output = call(runtime, service, "stream_write", key="decoder-write", stream_id=decoded_id, payload_id=encoded)
        _, closed = call(runtime, service, "stream_finalize", key="decoder-finalize", stream_id=decoded_id)
    else:
        _, output = call(runtime, service, "stream_finalize", key="decoder-finalize", stream_id=decoded_id)
        closed = output
        assert closed["authenticated_complete"] is True
    assert closed["state"] == "finalized"
    assert output["output"]["sha256"] == hashlib.sha256(b"synthetic-storage-payload" * 100).hexdigest()
    _, released = call(runtime, service, "stream_release", stream_id=encoded_id)
    assert released["released"] is True
    call(runtime, service, "stream_release", key="decoder-release", stream_id=decoded_id)
    call(runtime, service, "shutdown")


def test_incremental_decoder_failure_is_visible_and_abort_never_finalizes_pending_writer(runtime, monkeypatch):
    service = prepare(runtime, monkeypatch)
    _, decoder = call(runtime, service, "stream_open", codec_kind="compression", codec_id="local", direction="decode")
    response, failure = call(runtime, service, "stream_write", stream_id=decoder["stream_id"], payload_id="seed", failed=True)
    assert failure["state"] == "failed"
    assert response["issues"] == [{"code": "native_codec_incomplete"}]
    _, status = call(runtime, service, "stream_status", stream_id=decoder["stream_id"])
    assert status["state"] == "failed"
    _, aborted = call(runtime, service, "stream_abort", stream_id=decoder["stream_id"])
    assert aborted["state"] == "aborted"
    _, writer = call(runtime, service, "stream_open", key="pending-writer", codec_kind="encryption", codec_id="local", direction="encode")
    call(runtime, service, "stream_write", key="pending-write", stream_id=writer["stream_id"], payload_id="seed")
    _, aborted_writer = call(runtime, service, "stream_abort", key="pending-abort", stream_id=writer["stream_id"])
    assert aborted_writer["state"] == "aborted" and aborted_writer["output"] is None
    call(runtime, service, "shutdown")


def test_selected_native_context_exit_failure_cannot_be_reported_as_success(runtime, monkeypatch):
    service = prepare(runtime, monkeypatch)
    runtime[2].with_name("utility-stream-exit-failed").touch()
    _, writer = call(runtime, service, "stream_open", codec_kind="compression", codec_id="local", direction="encode")
    call(runtime, service, "stream_write", stream_id=writer["stream_id"], payload_id="seed")
    response, failed = call(runtime, service, "stream_finalize", stream_id=writer["stream_id"], failed=True)
    assert response["issues"] == [{"code": "native_codec_incomplete"}]
    assert failed["state"] == "failed" and failed["cleanup_uncertain"] is True
    assert runtime[2].read_text().splitlines().count("utility-stream-close") == 1
    _, status = call(runtime, service, "stream_status", stream_id=writer["stream_id"])
    assert status["cleanup_uncertain"] is True
    plugins.invoke(runtime[0], "resources_release", resource_id=service)
    assert runtime[2].read_text().splitlines().count("utility-stream-close") == 1


def test_native_encrypted_stream_read_chunk_does_not_claim_total_completion(runtime, monkeypatch):
    service = prepare(runtime, monkeypatch)
    _, writer = call(runtime, service, "stream_open", codec_kind="encryption", codec_id="local", direction="encode")
    call(runtime, service, "stream_write", stream_id=writer["stream_id"], payload_id="seed")
    _, encoded = call(runtime, service, "stream_finalize", stream_id=writer["stream_id"])
    _, reader = call(runtime, service, "stream_open", key="read-open", codec_kind="encryption", codec_id="local", direction="decode", payload_id=encoded["output"]["payload_id"])
    _, chunk = call(runtime, service, "stream_read", stream_id=reader["stream_id"])
    assert chunk["output"]["size_bytes"] == 64
    assert chunk["authenticated_complete"] is False
    _, final = call(runtime, service, "stream_finalize", key="read-finalize", stream_id=reader["stream_id"])
    assert final["authenticated_complete"] is True and final["output"]["size_bytes"] == len(b"synthetic-storage-payload" * 100) - 64
    call(runtime, service, "shutdown")
