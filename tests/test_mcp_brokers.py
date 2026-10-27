"""Installed local adapter exercises native broker effects and closed approvals."""

import asyncio
import json
import sys
from types import SimpleNamespace

import pytest

from tests import test_mcp_plugin_bindings as plugins
from truthound.mcp.approval import ApprovalStore

runtime = plugins.runtime
invoke = plugins.invoke
pytestmark = pytest.mark.contract


def approved(runtime, operation, **request):
    adapter, control, _, _, _ = runtime
    pending = invoke(adapter, operation, **request)
    assert pending["issues"] == [{"code": "approval_required"}], pending
    store = ApprovalStore(control)
    store.approve(pending["data"]["plan_id"])
    store.close()
    result = invoke(adapter, operation, **request)
    assert result["execution_status"] == "succeeded", result
    return result


def prepare(runtime, monkeypatch):
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
    settings["plugins"]["fixture"]["capabilities"]["broker-factory"] = "broker-factory"
    settings["brokers"] = {"local-broker": "broker-factory"}
    path.write_text(json.dumps(settings), encoding="utf-8")
    before = marker.read_bytes() if marker.exists() else b""
    request = dict(binding_id="local-broker", idempotency_key="broker-open-01")
    pending = invoke(adapter, "brokers_open", **request)
    assert pending["issues"] == [{"code": "approval_required"}]
    assert (marker.read_bytes() if marker.exists() else b"") == before
    result = approved(runtime, "brokers_open", **request)
    assert "broker-connect" not in marker.read_text()
    return result["data"]["resource_id"]


def command(runtime, identity, action, *, key=None, **arguments):
    result = approved(
        runtime,
        "brokers_execute",
        binding_id="local-broker",
        service_id=identity,
        command=action,
        idempotency_key=key or "broker-command-" + action,
        **arguments,
    )
    return plugins.source_artifact(runtime[0], result["data"]["resource_id"])


def test_native_broker_consume_window_ack_and_produce_with_replay(runtime, monkeypatch):
    from dataclasses import asdict

    from truthound.realtime.adapters.mock import MockAdapter, MockAdapterConfig
    from truthound.realtime.processing.windows import (
        CountAggregator,
        TumblingWindowProcessor,
        WindowConfig,
    )

    identity = prepare(runtime, monkeypatch)
    command(runtime, identity, "connect")
    batch = command(runtime, identity, "consume", limit=3)

    async def direct():
        adapter = MockAdapter(
            MockAdapterConfig(num_messages=3, seed=42, error_rate=0, enable_health_check=False)
        )
        await adapter.connect()
        native = await adapter.consume_batch(3)
        window = TumblingWindowProcessor(WindowConfig(size_ms=60000), CountAggregator)
        for message in native.messages:
            await window.add(message)
        output = await window.force_emit()
        await adapter.commit_batch(native.messages)
        assert adapter.get_committed_offsets() == {0: native.messages[-1].offset}
        await adapter.disconnect()
        return [asdict(message)["value"]["id"] for message in native.messages], output

    values, windows = asyncio.run(direct())
    assert [message["value"]["id"] for message in batch["messages"]] == values
    command(runtime, identity, "window_add", batch_id=batch["batch_id"])
    result = command(runtime, identity, "window_flush")
    assert (
        sum(row["message_count"] for row in result["windows"])
        == sum(row.message_count for row in windows)
        == 3
    )
    marker = runtime[2]
    request = dict(
        binding_id="local-broker",
        service_id=identity,
        command="ack",
        batch_id=batch["batch_id"],
        idempotency_key="broker-ack-exact",
    )
    receipt = approved(runtime, "brokers_execute", **request)
    effects = marker.read_bytes()
    assert invoke(runtime[0], "brokers_execute", **request) == receipt
    assert marker.read_bytes() == effects
    assert marker.read_text().count("broker-ack") == 1
    command(
        runtime,
        identity,
        "produce",
        message={"value": {"x": 1}, "timestamp": "2026-10-24T00:00:00+00:00"},
    )
    assert marker.read_text().count("broker-produce") == 1
    command(runtime, identity, "disconnect")


def test_native_broker_processing_and_recovery_are_not_delivery_claims(runtime, monkeypatch):
    identity = prepare(runtime, monkeypatch)
    command(runtime, identity, "connect")
    batch = command(runtime, identity, "consume", limit=3)
    results = command(runtime, identity, "process", batch_id=batch["batch_id"])
    assert all(row["success"] for row in results["results"])
    assert runtime[2].read_text().count("broker-ack") == 3
    assert command(runtime, identity, "recover") == {"rolled_back": 0}
    command(runtime, identity, "ack", batch_id=batch["batch_id"])
    command(runtime, identity, "seek", partition=0, offset=0)
    command(runtime, identity, "disconnect")


def test_broker_wire_cannot_choose_transport_or_import_and_release_reaps_owner(
    runtime, monkeypatch
):
    identity = prepare(runtime, monkeypatch)
    adapter = runtime[0]
    before = runtime[2].read_bytes()
    for arguments in (
        {"topic": "arbitrary"},
        {"factory": "os.system"},
        {"endpoint": "private://broker"},
    ):
        result = invoke(
            adapter,
            "brokers_execute",
            binding_id="local-broker",
            service_id=identity,
            command="connect",
            idempotency_key="broker-invalid-command",
            **arguments,
        )
        assert result["issues"] == [{"code": "invalid_arguments"}]
    assert runtime[2].read_bytes() == before
    worker = adapter._session._brokers.workers[identity]
    assert (
        invoke(adapter, "resources_release", resource_id=identity)["execution_status"]
        == "succeeded"
    )
    assert worker.closed and worker.process.poll() is not None


def test_lost_ack_receipt_quarantines_command_instead_of_committing_again(runtime, monkeypatch):
    adapter = runtime[0]
    identity = prepare(runtime, monkeypatch)
    command(runtime, identity, "connect")
    batch = command(runtime, identity, "consume", limit=3)
    worker = adapter._session._brokers.workers[identity]
    original = worker.call

    async def lose_receipt(request, timeout):
        await original(request, timeout)
        raise TimeoutError("synthetic lost receipt")

    monkeypatch.setattr(worker, "call", lose_receipt)
    request = dict(
        binding_id="local-broker",
        service_id=identity,
        command="ack",
        batch_id=batch["batch_id"],
        idempotency_key="broker-lost-ack",
    )
    pending = invoke(adapter, "brokers_execute", **request)
    store = ApprovalStore(runtime[1])
    store.approve(pending["data"]["plan_id"])
    store.close()
    result = invoke(adapter, "brokers_execute", **request)
    assert result["issues"] == [{"code": "outcome_unknown"}]
    assert runtime[2].read_text().count("broker-ack") == 1
    assert worker.closed and worker.process.poll() is not None
    assert invoke(adapter, "brokers_execute", **request)["execution_status"] != "succeeded"
    assert runtime[2].read_text().count("broker-ack") == 1


@pytest.mark.asyncio
async def test_broker_recovery_calls_native_rollback_for_real_pending_transaction():
    from truthound.mcp.broker_worker import BrokerOwner
    from truthound.mcp.brokers import BrokerRuntime
    from truthound.realtime.adapters.mock import MockAdapter
    from truthound.realtime.processing.exactly_once import ExactlyOnceProcessor

    adapter = MockAdapter()
    processor = ExactlyOnceProcessor(adapter)
    transaction = await processor._txn_manager.begin()
    owner = BrokerOwner(BrokerRuntime(adapter, processor=processor))
    assert (await owner.execute({"command": "recover"}, 65536)) == {"rolled_back": 1}
    assert (await owner.execute({"command": "recover"}, 65536)) == {"rolled_back": 0}
    stored = await processor._state.get("transactions", transaction.id)
    assert stored["state"] == "rolled_back"
