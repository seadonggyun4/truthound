"""Actual database stores and redacted pool lifecycle via installed factories."""

import pytest

from tests import test_mcp_plugin_bindings as plugins
from tests.test_mcp_brokers import approved
from tests.test_mcp_execution_stores import command, prepare
from truthound.mcp.approval import ApprovalStore

runtime = plugins.runtime
pytestmark = pytest.mark.contract


@pytest.mark.parametrize("backend", ["database", "database-legacy", "database-prefixed"])
def test_database_persists_after_pool_recycle_shutdown_and_factory_reopen(
    runtime, monkeypatch, backend
):
    identity = prepare(runtime, monkeypatch, "store_services", backend)

    def call(name, **arguments):
        return command(runtime, "store_services", identity, name, **arguments)

    call("initialize")
    assert call("save", record_id="seed")["record_id"] == "seed"
    status = call("pool_status")
    assert status["initialized"]
    assert "config" not in status and "connection_url" not in str(status)
    assert "native-store.db" not in str(status)
    recycled = call("pool_recycle")["recycled"]
    if backend == "database":
        assert recycled >= 1
    assert call("get", record_id="seed")["record"]["status"] == "failure"
    assert call("count")["count"] == 1
    if backend == "database-prefixed":
        import sqlite3

        with sqlite3.connect(runtime[2].with_name("native-store.db")) as connection:
            assert connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall() == [("isolated_validation_results",)]
    assert call("shutdown")["shutdown"]
    assert "store-close" in runtime[2].read_text()
    plugins.invoke(runtime[0], "resources_release", resource_id=identity)
    identity = approved(
        runtime, "store_services_open", binding_id="local", idempotency_key="database-fresh-owner"
    )["data"]["resource_id"]
    assert call("count", key="reopened-database-count")["count"] == 1
    assert call("delete", record_id="seed")["deleted"]
    assert call("count", key="after-database-delete")["count"] == 0
    call("shutdown", key="reopened-database-shutdown")


@pytest.mark.parametrize("operation", ["query", "count", "exists"])
def test_sql_failure_cannot_become_a_successful_empty_mcp_result(runtime, monkeypatch, operation):
    identity = prepare(runtime, monkeypatch, "store_services", "database-failure")
    request = dict(
        binding_id="local",
        service_id=identity,
        command=operation,
        idempotency_key="database-fault-" + operation,
    )
    if operation == "exists":
        request["record_id"] = "seed"
    pending = plugins.invoke(runtime[0], "store_services_execute", **request)
    store = ApprovalStore(runtime[1])
    try:
        store.approve(pending["data"]["plan_id"])
    finally:
        store.close()
    failed = plugins.invoke(runtime[0], "store_services_execute", **request)
    assert failed["execution_status"] == "outcome_unknown", failed
    assert failed["data"] is None
    assert "synthetic-private-db-connection" not in str(failed)
    assert "native-store.db" not in str(failed)
