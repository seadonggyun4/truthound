"""Bound observable stores expose native audit verdicts and selected scalar metrics."""

import pytest

from tests.test_mcp_execution_stores import command, plugins, prepare

runtime = plugins.runtime
pytestmark = pytest.mark.contract


def test_observable_store_audit_metric_configuration_and_owned_teardown(runtime, monkeypatch):
    group = "store_services"
    identity = prepare(runtime, monkeypatch, group, mode="observable")

    def call(name, **args):
        return command(runtime, group, identity, name, **args)

    call("initialize")
    call("save", record_id="seed")
    assert call("metric_read", metric_id="saves")["value"] == 1
    events = call("audit_query", record_id="seed", audit={"event_type": "create"})["events"]
    assert (
        len(events) == 1 and events[0]["status"] == "success" and events[0]["record_id"] == "seed"
    )
    assert events[0]["event_type"] == "create"
    for private in (
        "private-native-run",
        "private-user",
        "private-tenant",
        "private-correlation",
        "synthetic-private-observability",
    ):
        assert private not in str(events)
    assert (
        call("audit_query", audit={"event_type": "delete"}, key="no-delete-events")["events"] == []
    )
    assert call("observability_flush")["flushed"]
    worker = runtime[0]._session._administration["store_services"].workers[identity]
    plugins.invoke(runtime[0], "resources_release", resource_id=identity)
    assert worker.closed and "store-close" in runtime[2].read_text()
