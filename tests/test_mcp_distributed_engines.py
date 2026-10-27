"""Approved installed factories own actual local Ray and Spark lifetimes."""

import pytest

from tests import test_mcp_plugin_bindings as plugins
from tests.test_mcp_brokers import approved
from tests.test_mcp_execution_stores import command, prepare
from tests.verification_support import environment

runtime = plugins.runtime
pytestmark = pytest.mark.contract


@pytest.mark.parametrize("backend", ["ray", "spark"])
def test_approved_native_distributed_engines_children_and_reopen(runtime, monkeypatch, backend):
    adapter, _, marker, _, _ = runtime
    if backend == "spark":
        java = environment().get("JAVA_HOME")
        if java:
            marker.with_name("execution-java-home").write_text(java)
    identity = prepare(runtime, monkeypatch, "execution_services", backend)
    process = adapter._session._administration["execution_services"].workers[identity]

    def call(name, **arguments):
        return command(runtime, "execution_services", identity, name, **arguments)

    assert call("inspect")["engine_type"] == backend
    assert call("count_nulls_all")["value"] == {"x": 1, "g": 0}
    assert call("get_stats", column="x")["value"]["count"] == 2
    assert call("aggregate", aggregations={"x": "median"})["value"] == {"x_median": 2.0}
    child = call("sample", limit=2, seed=42)["engine_id"]
    assert call("count_rows", engine_id=child, key="sample-actual-count")["value"] == 2
    assert call("release", engine_id=child)["released"]
    repartitioned = call("repartition", num_partitions=2)["engine_id"]
    assert (
        call("partitions", engine_id=repartitioned, key="repartitioned-actual-partitions")[
            "num_partitions"
        ]
        == 2
    )
    assert call("count_rows", engine_id=repartitioned, key="repartitioned-row-count")["value"] == 3
    coalesced = call("coalesce", engine_id=repartitioned, num_partitions=1)["engine_id"]
    assert (
        call("partitions", engine_id=coalesced, key="coalesced-actual-partitions")["num_partitions"]
        == 1
    )
    call("release", engine_id=coalesced, key="coalesced-release")
    call("release", engine_id=repartitioned, key="repartitioned-release")
    assert len(call("to_arrow", limit=3)["rows"]) == 3
    if backend == "ray":
        child = call("filter_rows", row_filter_id="positive")["engine_id"]
        assert call("count_rows", engine_id=child, key="filtered-actual-count")["value"] == 2
        assert call("materialize")["engine_id"] == "root"
    else:
        child = call("execute_query", query_id="count")["engine_id"]
        assert call("get_column_values", engine_id=child, column="total")["value"] == [3]
        assert call("persist")["engine_id"] == "root"
        assert call("unpersist")["engine_id"] == "root"
    assert call("release", engine_id=child, key="final-child-release")["released"]
    assert call("shutdown")["shutdown"]
    replayed = plugins.invoke(
        adapter,
        "execution_services_execute",
        binding_id="local",
        service_id=identity,
        command="shutdown",
        idempotency_key="execution_services-shutdown",
    )
    assert replayed["execution_status"] == "succeeded", replayed
    assert plugins.source_artifact(adapter, replayed["data"]["resource_id"])["shutdown"]
    rejected = plugins.invoke(
        adapter,
        "execution_services_execute",
        binding_id="local",
        service_id=identity,
        command="count_rows",
        idempotency_key="post-shutdown-count",
    )
    assert rejected["issues"] == [{"code": "invalid_handle"}], rejected
    assert (
        plugins.invoke(adapter, "resources_release", resource_id=identity)["execution_status"]
        == "succeeded"
    )
    assert process.closed and process.process.poll() is not None
    assert "execution-close" in marker.read_text()
    # Reopen the same installed factory after its own SDK has been stopped.
    identity = approved(
        runtime, "execution_services_open", binding_id="local", idempotency_key="fresh-native-open"
    )["data"]["resource_id"]
    assert call("count_rows", key="fresh-native-count")["value"] == 3
    assert call("shutdown", key="fresh-native-shutdown")["shutdown"]
    plugins.invoke(adapter, "resources_release", resource_id=identity)
