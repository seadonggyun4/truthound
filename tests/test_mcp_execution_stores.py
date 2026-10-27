"""Native engines/stores via installed factories, persistence and input ownership."""

import json
import sys
from types import SimpleNamespace

import polars as pl
import pytest

from tests import test_mcp_plugin_bindings as plugins
from tests.test_mcp_brokers import approved

runtime = plugins.runtime
pytestmark = pytest.mark.contract


def prepare(runtime, monkeypatch, group, mode=None):
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
    if mode:
        marker.with_name(
            "execution-mode" if group == "execution_services" else "store-mode"
        ).write_text(mode)
    capability = "execution-factory" if group == "execution_services" else "store-factory"
    document = json.loads((control / "bindings.json").read_text())
    document["plugins"]["fixture"]["capabilities"][capability] = capability
    document[group] = {"local": capability}
    (control / "bindings.json").write_text(json.dumps(document))
    return approved(runtime, group + "_open", binding_id="local", idempotency_key=group + "-open")[
        "data"
    ]["resource_id"]


def command(runtime, group, identity, command_name, *, key=None, **arguments):
    response = approved(
        runtime,
        group + "_execute",
        binding_id="local",
        service_id=identity,
        command=command_name,
        idempotency_key=key or group + "-" + command_name,
        **arguments,
    )
    return plugins.source_artifact(runtime[0], response["data"]["resource_id"])



def test_native_descriptor_output_cannot_corrupt_resident_protocol(runtime, monkeypatch):
    runtime[2].with_name("native-descriptor-output").touch()
    identity = prepare(runtime, monkeypatch, "execution_services")
    result = command(runtime, "execution_services", identity, "count_rows")
    assert result["value"] == 3
    assert "synthetic-private-native" not in json.dumps(result)
    assert command(runtime, "execution_services", identity, "shutdown")["shutdown"] is True


def test_engine_cleanup_attempts_every_child_and_root_after_failure():
    import asyncio

    from truthound.execution.polars_engine import PolarsExecutionEngine
    from truthound.mcp.execution_services import ExecutionRuntime
    from truthound.mcp.execution_worker import ExecutionOwner

    closed = []

    class FailingExit(PolarsExecutionEngine):
        def __init__(self, label):
            super().__init__(pl.DataFrame({"x": [1]}).lazy())
            self.label = label

        def __exit__(self, *args):
            super().__exit__(*args)
            closed.append(self.label)
            raise RuntimeError("synthetic private cleanup " + self.label)

    owner = ExecutionOwner(
        ExecutionRuntime(FailingExit("root")), {"native_group": "execution_services"}
    )
    owner.retain(FailingExit("first"))
    owner.retain(FailingExit("second"))
    with pytest.raises(ExceptionGroup) as failed:
        asyncio.run(owner.close())
    assert closed == ["second", "first", "root"]
    assert len(failed.value.exceptions) == 3
    assert not owner.children and not owner.engines


def test_native_shutdown_failure_is_unknown_and_never_replayed(runtime, monkeypatch):
    from truthound.mcp.approval import ApprovalStore

    identity = prepare(runtime, monkeypatch, "execution_services", "polars-close-failure")
    request = dict(
        binding_id="local",
        service_id=identity,
        command="shutdown",
        idempotency_key="failed-native-shutdown",
    )
    pending = plugins.invoke(runtime[0], "execution_services_execute", **request)
    store = ApprovalStore(runtime[1])
    try:
        store.approve(pending["data"]["plan_id"])
    finally:
        store.close()
    failed = plugins.invoke(runtime[0], "execution_services_execute", **request)
    assert failed["execution_status"] == "outcome_unknown", failed
    assert "synthetic-private-native-close" not in str(failed)
    effects = runtime[2].read_text()
    assert effects.count("execution-close") == 1
    repeated = plugins.invoke(runtime[0], "execution_services_execute", **request)
    assert repeated["execution_status"] != "succeeded"
    assert runtime[2].read_text() == effects


@pytest.mark.parametrize("backend", ["polars", "sql"])
def test_native_engine_queries_children_and_cleanup_match_direct_calls(
    runtime, monkeypatch, backend
):
    from truthound.datasources.sql.sqlite import SQLiteDataSource
    from truthound.execution._protocols import AggregationType
    from truthound.execution.polars_engine import PolarsExecutionEngine
    from truthound.execution.sql_engine import SQLExecutionEngine

    identity = prepare(
        runtime, monkeypatch, "execution_services", "sql" if backend == "sql" else None
    )
    source = (
        SQLiteDataSource(database=str(runtime[2].with_name("operator.db")), table="items")
        if backend == "sql"
        else None
    )
    if source is not None:
        source.__enter__()
    native = (
        SQLExecutionEngine(source)
        if source is not None
        else PolarsExecutionEngine(pl.DataFrame({"x": [-1, 1, None]}).lazy())
    )

    def call(name, **args):
        return command(runtime, "execution_services", identity, name, **args)

    try:
        assert call("count_rows")["value"] == native.count_rows()
        assert call("get_stats", column="x")["value"] == native.get_stats("x")
        assert call("count_nulls", column="x")["value"] == native.count_nulls("x")
        assert call("get_completeness_ratio", column="x")["value"] == native.get_completeness_ratio(
            "x"
        )
        assert call("get_column_values", column="x", limit=2, offset=1)[
            "value"
        ] == native.get_column_values("x", 2, 1)
        assert call("aggregate", aggregations={"x": "sum"})["value"] == native.aggregate(
            {"x": AggregationType.SUM}
        )
        assert call("aggregate", key="aggregate-median", aggregations={"x": "median"})["value"] == {
            "x_median": 0.0
        }
        assert call("count_duplicates", columns=["x"])["value"] == 0
        assert call("get_duplicate_values", columns=["x"])["value"] == []
        assert call("count_in_set", column="x", values=[1])["value"] == native.count_in_set(
            "x", {1}
        )
        assert call("count_in_range", column="x", min_value=-1, max_value=0)[
            "value"
        ] == native.count_in_range("x", -1, 0)
        assert call("count_matching", predicate_id="positive")["value"] == native.count_matching(
            "x > 0"
        )
        child = call("filter_by_condition", predicate_id="positive")["engine_id"]
        assert call("count_rows", engine_id=child, key="child-count")["value"] == 1
        assert call("release", engine_id=child)["released"]
        sampled = call("sample", limit=1, seed=42)["engine_id"]
        assert call("count_rows", engine_id=sampled, key="sample-count")["value"] == 1
        call("clear_cache")
        if backend == "sql":
            assert call("execute_query", query_id="count")["value"] == {
                "rows": [{"total": 3}],
                "truncated": False,
            }
        else:
            child = call("with_columns", expression_ids=["double"])["engine_id"]
            assert call(
                "get_column_values", engine_id=child, column="double", key="computed-values"
            )["value"] == [-2, 2, None]
            from truthound.mcp.encoding import encode

            assert encode(
                call("to_numpy_safe", columns=["x"], limit=1, seed=42)["value"]
            ) == encode(native.to_numpy_safe(["x"], 1, 42).tolist())
        plugins.invoke(runtime[0], "resources_release", resource_id=identity)
        assert "execution-close" in runtime[2].read_text()
        if backend == "sql":
            assert "source-close" in runtime[2].read_text()
    finally:
        native.__exit__(None, None, None)
        if source is not None:
            source.__exit__(None, None, None)


def test_filesystem_store_native_result_save_worker_reopen_and_bounded_delete(runtime, monkeypatch):
    group = "store_services"
    identity = prepare(runtime, monkeypatch, group)

    def call(name, **args):
        return command(runtime, group, identity, name, **args)

    call("initialize")
    assert call("save", record_id="seed")["record_id"] == "seed"
    assert call("get", record_id="seed")["record"]["status"] == "failure"
    saved = (
        runtime[2].with_name("native-store") / "default" / "validations" / "private-native-run.json"
    )
    assert json.loads(saved.read_text())["metadata"] == {
        "private-record": "synthetic-private-record"
    }
    assert "synthetic-private-record" not in str(call("query"))
    validation = plugins.invoke(
        runtime[0],
        "assets_validate",
        asset_id=runtime[4],
        checks=[{"validator": "not_null", "settings": {"columns": ["x"]}}],
    )
    assert validation["execution_status"] == "succeeded", validation
    run_id = validation["data"]["artifact"]["resource_id"]
    saved_id = call("save", result_id=run_id, key="save-actual-run")["record_id"]
    assert call("exists", record_id=saved_id)["exists"]
    assert call("count")["count"] == 2
    assert len(call("iter_query", limit=1, batch_size=1)["records"]) == 1
    plugins.invoke(runtime[0], "resources_release", resource_id=identity)
    assert "store-close" in runtime[2].read_text()
    identity = approved(
        runtime, group + "_open", binding_id="local", idempotency_key="store-reopen"
    )["data"]["resource_id"]
    assert call("count", key="restored-count")["count"] == 2
    assert call("clear", limit=1)["deleted"] == 1
    assert call("count", key="remaining-count")["count"] == 1


@pytest.mark.parametrize(
    "mode", ["cached", "versioned", "batched", "expectation", "memory-expectation"]
)
def test_native_store_wrapper_specific_lifecycle(runtime, monkeypatch, mode):
    group = "store_services"
    identity = prepare(runtime, monkeypatch, group, mode)

    def call(name, **args):
        return command(runtime, group, identity, name, **args)

    if mode in {"expectation", "memory-expectation"}:
        record = call(
            "save_suite",
            suite={
                "name": "local_suite",
                "data_asset_id": "local",
                "expectations": [{"expectation_type": "not_null", "column": "x"}],
            },
        )["record_id"]
        assert call("get_suite", record_id=record)["record"]["expectations"][0]["column"] == "x"
        assert call("list_suites")["record_ids"] == [record]
    elif mode == "batched":
        assert call("batch_add", record_id="seed")["added"]
        assert not call("exists", record_id="seed")["exists"]
        assert call("batch_flush")["batches"] == 1
        assert call("exists", record_id="seed", key="exists-after-flush")["exists"]
        assert not call("batch_stop")["stats"]["state"]["is_active"]
    else:
        call("save", record_id="seed")
        if mode == "cached":
            assert call("cache_get_many", record_ids=["seed"])["records"][0]["record_id"] == "seed"
            assert call("cache_invalidate", record_id="seed")["invalidated"]
            assert call("cache_warm", record_ids=["seed"])["warmed"] == 1
            assert call("cache_refresh", record_id="seed")["record"]["status"] == "failure"
            assert call("cache_invalidate_all")["invalidated"] == 1
        else:
            call(
                "save", record_id="seed", expected_version=1, message="second", key="second-version"
            )
            assert call("count_versions", record_id="seed")["count"] == 2
            assert call("version_info", record_id="seed", version=1)["version"]["version"] == 1
            assert call("diff", record_id="seed", version=1, version_b=2)["changes"] == []
            call("rollback", record_id="seed", version=1)
            assert call("latest_version", record_id="seed")["version"]["version"] == 3
    plugins.invoke(runtime[0], "resources_release", resource_id=identity)
    assert "store-close" in runtime[2].read_text()


def test_store_run_input_is_reauthorized_and_released_with_owner(runtime, monkeypatch):
    identity = prepare(runtime, monkeypatch, "store_services")
    validation = plugins.invoke(
        runtime[0],
        "assets_validate",
        asset_id=runtime[4],
        checks=[{"validator": "not_null", "settings": {"columns": ["x"]}}],
    )
    run_id = validation["data"]["artifact"]["resource_id"]
    command(runtime, "store_services", identity, "save", result_id=run_id)
    plugins.invoke(runtime[0], "resources_release", resource_id=run_id)
    assert "store-close" in runtime[2].read_text()
    response = plugins.invoke(
        runtime[0],
        "store_services_execute",
        binding_id="local",
        service_id=identity,
        command="inspect",
        idempotency_key="after-input-release",
    )
    assert response["issues"] == [{"code": "invalid_handle"}]


@pytest.mark.parametrize("group", ["execution_services", "store_services"])
def test_native_engine_store_wire_rejects_backend_and_code_selection(runtime, monkeypatch, group):
    identity = prepare(runtime, monkeypatch, group)
    for fields in (
        {"sql": "DROP TABLE items"},
        {"path": "/private/source"},
        {"factory": "arbitrary.module"},
        {"config": {"password": "synthetic"}},
    ):
        response = plugins.invoke(
            runtime[0],
            group + "_execute",
            binding_id="local",
            service_id=identity,
            command="inspect",
            idempotency_key="reject-native-config",
            **fields,
        )
        assert response["issues"] == [{"code": "invalid_arguments"}]


@pytest.mark.parametrize("mode", ["retention", "tiered", "replicated"])
def test_native_store_policies_tiers_replica_effects_and_close(runtime, monkeypatch, mode):
    identity = prepare(runtime, monkeypatch, "store_services", mode)

    def call(name, **args):
        return command(runtime, "store_services", identity, name, **args)

    assert call("inspect")["store_type"]
    if mode == "retention":
        call("save", record_id="seed")
        call("save", record_id="second", key="save-second")
        preview = call("retention_cleanup")["cleanup"]
        assert preview["dry_run"] and preview["items_deleted"] == 1 and not preview["native_failed"]
        assert not call("retention_info", record_id="seed")["retention"]["will_be_retained"]
        actual = call("retention_cleanup", dry_run=False, key="retention-real-delete")["cleanup"]
        assert actual["items_deleted"] == 1 and not actual["native_failed"]
        assert not call("exists", record_id="seed")["exists"]
        assert (
            call("retention_stats")["stats"]["total_deleted"] == 2
        )  # Native history includes previews.
    elif mode == "tiered":
        assert call("tier_info", record_id="seed")["tier"]["tier_id"] == "hot"
        preview = call("tier_run")["tiering"]
        assert preview["items_migrated"] == 1 and preview["dry_run"]
        assert preview["migrations"][0]["to_tier_id"] == "cold"
        moved = call("tier_run", dry_run=False, key="tier-actual")["tiering"]
        assert moved["items_migrated"] == 1 and not moved["native_failed"]
        assert call("tier_info", record_id="seed", key="cold-info")["tier"]["tier_id"] == "cold"
        assert call("tier_move", record_id="seed", tier_id="hot")["moved"]
        assert call("tier_stats")["tiers"]["hot"]["item_count"] == 1
        assert "private-" not in str(call("tier_list"))
    else:
        call("replication_start")
        call("save", record_id="seed")
        result = call("replication_sync", replica_id="local")["replicas"]["local"]
        assert result["success"] and result["items_synced"] == 1
        stats = call("replication_flush")["stats"]
        assert stats["metrics"]["writes_replicated"] == 2 and stats["pending_tasks"] == 0
        assert "private-" not in str(stats)
        assert stats["monitor"]["failures"] == 0
        assert stats["monitor"]["handler_failures"] == 0
        assert call("delete", record_id="seed")["deleted"]
        assert not list(runtime[2].with_name("native-secondary").rglob("private-native-run.json"))
        call("replication_stop")
    plugins.invoke(runtime[0], "resources_release", resource_id=identity)
    assert "store-close" in runtime[2].read_text()


@pytest.mark.fault
def test_replica_partial_failure_is_unknown_and_exact_request_is_never_replayed(
    runtime, monkeypatch
):
    from truthound.mcp.approval import ApprovalStore

    identity = prepare(runtime, monkeypatch, "store_services", "replicated-failure")
    request = dict(
        binding_id="local",
        service_id=identity,
        command="save",
        record_id="seed",
        idempotency_key="partial-replica-write",
    )
    pending = plugins.invoke(runtime[0], "store_services_execute", **request)
    assert pending["issues"] == [{"code": "approval_required"}]
    store = ApprovalStore(runtime[1])
    try:
        store.approve(pending["data"]["plan_id"])
    finally:
        store.close()
    first = plugins.invoke(runtime[0], "store_services_execute", **request)
    assert first["execution_status"] == "outcome_unknown"
    replay = plugins.invoke(runtime[0], "store_services_execute", **request)
    assert replay["execution_status"] != "succeeded"
    assert runtime[2].read_text().splitlines().count("replica-save") == 1
    assert list(runtime[2].with_name("native-store").rglob("private-native-run.json"))
    assert "synthetic-private-replica-failure" not in str(first)


def test_actual_local_dask_partitions_native_results_persistence_and_child_lifecycle(
    runtime, monkeypatch
):
    import dask.dataframe as dd
    import pandas as pd

    from truthound.execution._protocols import AggregationType
    from truthound.execution.distributed.dask_engine import DaskExecutionEngine

    identity = prepare(runtime, monkeypatch, "execution_services", "dask")
    native = DaskExecutionEngine(
        dd.from_pandas(pd.DataFrame({"x": [-1.0, 1.0, None]}), npartitions=2)
    )

    def call(name, **args):
        return command(runtime, "execution_services", identity, name, **args)

    try:
        assert call("partitions") == {"backend": "dask", "num_partitions": 2}
        assert call("count_rows")["value"] == native.count_rows()
        assert call("aggregate", aggregations={"x": "sum"})["value"] == native.aggregate(
            {"x": AggregationType.SUM}
        )
        assert call("count_nulls", column="x")["value"] == native.count_nulls("x")
        assert call("persist")["engine_id"] == "root"
        repartitioned = call("repartition", num_partitions=3)["engine_id"]
        assert (
            call("partitions", engine_id=repartitioned, key="repartitioned-info")["num_partitions"]
            == 3
        )
        filtered = call("native_filter", predicate_id="positive")["engine_id"]
        assert call("compute", engine_id=filtered)["rows"] == [{"x": 1.0}]
        selected = call("select", columns=["x"])["engine_id"]
        assert call("head", engine_id=selected, limit=1)["rows"] == native.head(1).to_dict(
            orient="records"
        )
        assert (
            call("to_arrow", engine_id=filtered)["rows"]
            == native.filter("x > 0").to_arrow_table().to_pylist()
        )
        metrics = call("metrics")["metrics"]
        assert any(row["operation"] == "aggregate" and row["success"] for row in metrics)
        call("clear_metrics")
        assert not call("metrics", key="cleared-metrics")["metrics"]
        call("release", engine_id=repartitioned)
        plugins.invoke(runtime[0], "resources_release", resource_id=identity)
        assert "execution-close" in runtime[2].read_text()
    finally:
        native.__exit__(None, None, None)


@pytest.mark.parametrize("mode", ["dask", "polars", "dask-synchronous", "dask-eager", "dask-hash"])
def test_factory_selected_aggregation_plans_stream_and_native_validators(
    runtime, monkeypatch, mode
):
    identity = prepare(runtime, monkeypatch, "execution_services", mode)
    inspect = command(runtime, "execution_services", identity, "inspect")
    assert inspect["aggregation_plans"] == ["partitions", "stats"]
    assert inspect["validators"] == ["nulls"]
    if mode == "dask-eager":
        assert inspect["config"]["execution_mode"] == "eager"
        assert runtime[2].read_text().count("eager-native-partition") == 2
    result = command(
        runtime, "execution_services", identity, "aggregate_plan", aggregation_plan_id="stats"
    )
    assert result["value"]["x_count"] == 2
    assert result["value"]["x_nulls"] == {"null_count": 1, "total_count": 3}
    stream = command(
        runtime,
        "execution_services",
        identity,
        "aggregate_stream",
        aggregation_plan_id="stats",
        batch_size=1,
    )
    assert stream["complete"] is True and len(stream["snapshots"]) == 3
    assert stream["snapshots"][-1] == result["value"]
    partial = command(
        runtime,
        "execution_services",
        identity,
        "aggregate_stream",
        key="truncated-aggregation",
        aggregation_plan_id="stats",
        batch_size=1,
        limit=1,
    )
    assert partial["complete"] is False and len(partial["snapshots"]) == 1
    if mode in {"dask", "dask-synchronous", "dask-eager", "dask-hash"}:
        validated = command(
            runtime, "execution_services", identity, "validate", validator_ids=["nulls"]
        )
        assert validated["issues"][0]["issue_type"] == "null"
        assert validated["issues"][0]["count"] == 1
        assert validated["issues"][0]["result"]["element_count"] == 3
    separate = command(
        runtime,
        "execution_services",
        identity,
        "aggregate_plan",
        key="partition-plan",
        aggregation_plan_id="partitions",
    )
    partitions = separate["value"]["partitions"]
    assert len(partitions) == (
        3 if mode == "dask-hash" else 2 if mode in {"dask", "dask-synchronous", "dask-eager"} else 1
    )
    assert sum(row["result"]["*_count"] for row in partitions) == 3
    assert sum(row["result"]["x_sum"] for row in partitions) == 0
    partition_stream = command(
        runtime,
        "execution_services",
        identity,
        "aggregate_stream",
        key="partition-stream",
        aggregation_plan_id="partitions",
        batch_size=1,
    )
    assert partition_stream["complete"] is True
    assert partition_stream["snapshots"][-1] == separate["value"]
    closed = command(runtime, "execution_services", identity, "shutdown")
    assert closed["shutdown"] is True
    if mode == "dask-eager":
        assert runtime[2].read_text().count("eager-native-partition") == 2


@pytest.mark.parametrize("mode", ["filesystem-yaml", "filesystem-zstd"])
def test_selected_filesystem_format_survives_native_worker_shutdown_and_reopen(
    runtime, monkeypatch, mode
):
    import gzip

    import yaml

    group = "store_services"
    identity = prepare(runtime, monkeypatch, group, mode)
    command(runtime, group, identity, "save", record_id="seed")
    before = command(runtime, group, identity, "get", key="format-before", record_id="seed")
    suffix = ".yaml.gz" if mode == "filesystem-yaml" else ".json.zst"
    path = (
        runtime[2].with_name("native-store")
        / "default"
        / "validations"
        / ("private-native-run" + suffix)
    )
    if mode == "filesystem-yaml":
        document = yaml.safe_load(gzip.decompress(path.read_bytes()))
    else:
        from truthound.stores.compression.providers import get_compressor

        document = json.loads(get_compressor("zstd").decompress(path.read_bytes()))
    assert document["run_id"] == "private-native-run"
    command(runtime, group, identity, "shutdown")
    plugins.invoke(runtime[0], "resources_release", resource_id=identity)
    fresh = approved(
        runtime, group + "_open", binding_id="local", idempotency_key="filesystem-format-reopened"
    )["data"]["resource_id"]
    after = command(runtime, group, fresh, "get", key="format-after", record_id="seed")
    assert after["record"] == before["record"]
    command(runtime, group, fresh, "shutdown", key="filesystem-format-final-close")


def test_native_memory_store_is_owner_local_with_selected_configuration(runtime, monkeypatch):
    group = "store_services"
    identity = prepare(runtime, monkeypatch, group, "memory")
    command(runtime, group, identity, "save", record_id="seed")
    assert (
        command(runtime, group, identity, "get", record_id="seed")["record"]["status"] == "failure"
    )
    command(runtime, group, identity, "shutdown")
    plugins.invoke(runtime[0], "resources_release", resource_id=identity)
    fresh = approved(
        runtime, group + "_open", binding_id="local", idempotency_key="memory-reopened"
    )["data"]["resource_id"]
    assert (
        command(runtime, group, fresh, "exists", key="memory-fresh-exists", record_id="seed")[
            "exists"
        ]
        is False
    )
    command(runtime, group, fresh, "shutdown", key="memory-final-close")


def test_corrupt_filesystem_index_is_visible_and_new_owner_can_explicitly_rebuild(
    runtime, monkeypatch
):
    from truthound.mcp.approval import ApprovalStore

    group = "store_services"
    service = prepare(runtime, monkeypatch, group)
    command(runtime, group, service, "save", record_id="seed")
    command(runtime, group, service, "shutdown")
    plugins.invoke(runtime[0], "resources_release", resource_id=service)
    path = runtime[2].with_name("native-store") / "default" / "validations" / "_index.json"
    path.write_bytes(b"corrupt synthetic index")
    broken = approved(
        runtime, group + "_open", binding_id="local", idempotency_key="corrupt-index-open"
    )["data"]["resource_id"]
    request = dict(
        binding_id="local",
        service_id=broken,
        command="initialize",
        idempotency_key="corrupt-index-initialize",
    )
    pending = plugins.invoke(runtime[0], group + "_execute", **request)
    approval = ApprovalStore(runtime[1])
    try:
        approval.approve(pending["data"]["plan_id"])
    finally:
        approval.close()
    failed = plugins.invoke(runtime[0], group + "_execute", **request)
    assert failed["execution_status"] == "outcome_unknown" and failed["data"] is None
    assert path.read_bytes() == b"corrupt synthetic index"
    rejected = plugins.invoke(runtime[0], group + "_execute", **request)
    assert rejected["execution_status"] == "failed"
    assert rejected["issues"] == [{"code": "invalid_handle"}]
    plugins.invoke(runtime[0], "resources_release", resource_id=broken)
    restored = approved(
        runtime, group + "_open", binding_id="local", idempotency_key="explicit-rebuild-open"
    )["data"]["resource_id"]
    assert command(runtime, group, restored, "rebuild_index")["indexed"] == 1
    assert (
        command(runtime, group, restored, "get", key="get-after-rebuild", record_id="seed")[
            "record"
        ]["status"]
        == "failure"
    )
    command(runtime, group, restored, "shutdown", key="rebuild-final-close")


@pytest.mark.parametrize("mode", ["ordinary-s3", "ordinary-s3-yaml", "ordinary-s3-zstd"])
def test_selected_ordinary_s3_native_format_and_metadata_survive_worker_reopen(
    runtime, monkeypatch, mode
):
    import base64

    import yaml

    from truthound.stores.compression import get_compressor

    group = "store_services"
    identity = prepare(runtime, monkeypatch, group, mode)
    command(runtime, group, identity, "save", record_id="seed")
    before = command(runtime, group, identity, "get", key="s3-before", record_id="seed")
    command(runtime, group, identity, "shutdown")
    objects = json.loads(runtime[2].with_name("ordinary-s3-objects.json").read_text())
    key = next(key for key in objects if "/results/" in key)
    content = base64.b64decode(objects[key])
    if mode == "ordinary-s3-zstd":
        content = get_compressor("zstd").decompress(content)
        assert key.endswith(".json.zst")
    document = yaml.safe_load(content) if mode == "ordinary-s3-yaml" else json.loads(content)
    assert document["run_id"] == "private-native-run"
    assert document["metadata"]["configured"] == "synthetic-private-object-default"
    assert "synthetic-private-object-default" not in str(before)
    plugins.invoke(runtime[0], "resources_release", resource_id=identity)
    fresh = approved(runtime, group + "_open", binding_id="local", idempotency_key="s3-reopened")[
        "data"
    ]["resource_id"]
    after = command(runtime, group, fresh, "get", key="s3-after", record_id="seed")
    assert after["record"] == before["record"]
    assert command(runtime, group, fresh, "rebuild_index", key="s3-rebuild")["indexed"] == 1
    assert command(runtime, group, fresh, "delete", record_id="seed")["deleted"] is True
    assert command(runtime, group, fresh, "exists", record_id="seed")["exists"] is False
    command(runtime, group, fresh, "shutdown", key="s3-final-close")


@pytest.mark.parametrize("mode", ["ordinary-gcs", "ordinary-gcs-yaml", "ordinary-gcs-zstd"])
def test_native_google_sdk_http_persistence_survives_installed_owner_restart(
    runtime, monkeypatch, mode
):
    import base64
    import gzip

    import yaml

    from truthound.stores.compression import get_compressor

    group = "store_services"
    identity = prepare(runtime, monkeypatch, group, mode)
    command(runtime, group, identity, "save", record_id="seed")
    before = command(runtime, group, identity, "get", key="gcs-before", record_id="seed")
    command(runtime, group, identity, "shutdown")
    state = json.loads(runtime[2].with_name("native-google-objects.json").read_text())
    key = next(key for key in state["objects"] if "/results/" in key)
    content = base64.b64decode(state["objects"][key]["data"])
    content = (
        get_compressor("zstd").decompress(content)
        if mode == "ordinary-gcs-zstd"
        else gzip.decompress(content)
    )
    document = yaml.safe_load(content) if mode == "ordinary-gcs-yaml" else json.loads(content)
    assert document["metadata"]["configured"] == "synthetic-private-gcs-default"
    assert "synthetic-private-gcs-default" not in str(before)
    plugins.invoke(runtime[0], "resources_release", resource_id=identity)
    fresh = approved(runtime, group + "_open", binding_id="local", idempotency_key="gcs-reopened")[
        "data"
    ]["resource_id"]
    assert (
        command(runtime, group, fresh, "get", key="gcs-after", record_id="seed")["record"]
        == before["record"]
    )
    assert command(runtime, group, fresh, "rebuild_index", key="gcs-rebuild")["indexed"] == 1
    assert command(runtime, group, fresh, "delete", record_id="seed")["deleted"] is True
    assert command(runtime, group, fresh, "exists", record_id="seed")["exists"] is False
    command(runtime, group, fresh, "shutdown", key="gcs-final-close")


@pytest.mark.parametrize("mode", ["ordinary-azure", "ordinary-azure-yaml", "ordinary-azure-zstd"])
def test_native_azure_sdk_tier_and_persistence_survive_installed_owner_restart(
    runtime, monkeypatch, mode
):
    import base64
    import gzip

    import yaml

    from truthound.stores.compression import get_compressor

    group = "store_services"
    identity = prepare(runtime, monkeypatch, group, mode)
    command(runtime, group, identity, "save", record_id="seed")
    before = command(runtime, group, identity, "get", key="azure-before", record_id="seed")
    assert command(runtime, group, identity, "get_access_tier", record_id="seed") == {
        "record_id": "seed",
        "tier": "Cool",
        "archive_status": None,
    }
    changed = command(
        runtime, group, identity, "set_access_tier", record_id="seed", access_tier="Hot"
    )
    assert changed == {
        "record_id": "seed",
        "tier": "Hot",
        "archive_status": None,
        "acknowledged": True,
    }
    command(runtime, group, identity, "shutdown")
    state = json.loads(runtime[2].with_name("native-azure-objects.json").read_text())
    key = next(key for key in state["objects"] if "/results/" in key)
    row = state["objects"][key]
    content = base64.b64decode(row["data"])
    content = (
        get_compressor("zstd").decompress(content)
        if mode == "ordinary-azure-zstd"
        else gzip.decompress(content)
    )
    document = yaml.safe_load(content) if mode == "ordinary-azure-yaml" else json.loads(content)
    assert document["metadata"]["configured"] == "synthetic-private-azure-default"
    assert row["metadata"]["owner"] == "synthetic-private-azure-header"
    assert "synthetic-private-azure" not in str(before)
    assert "c3ludGhldGljLWF6dXJlLWtleS1vbmx5" not in str(before)
    plugins.invoke(runtime[0], "resources_release", resource_id=identity)
    fresh = approved(
        runtime, group + "_open", binding_id="local", idempotency_key="azure-reopened"
    )["data"]["resource_id"]
    assert (
        command(runtime, group, fresh, "get", key="azure-after", record_id="seed")["record"]
        == before["record"]
    )
    assert (
        command(
            runtime,
            group,
            fresh,
            "get_access_tier",
            key="azure-tier-after-reopen",
            record_id="seed",
        )["tier"]
        == "Hot"
    )
    assert command(runtime, group, fresh, "rebuild_index", key="azure-rebuild")["indexed"] == 1
    assert command(runtime, group, fresh, "delete", record_id="seed")["deleted"] is True
    assert command(runtime, group, fresh, "exists", record_id="seed")["exists"] is False
    command(runtime, group, fresh, "shutdown", key="azure-final-close")


def test_installed_dask_checkpoint_owner_restart_uses_saved_native_data(runtime, monkeypatch):
    group = "execution_services"
    identity = prepare(runtime, monkeypatch, group, "dask-checkpoint")
    assert command(runtime, group, identity, "count_rows")["value"] == 3
    child = command(runtime, group, identity, "checkpoint")
    assert "native-engine-checkpoints" not in str(child)
    assert (
        command(
            runtime,
            group,
            identity,
            "count_rows",
            key="checkpoint-child-count",
            engine_id=child["engine_id"],
        )["value"]
        == 3
    )
    command(runtime, group, identity, "shutdown")
    assert runtime[2].read_text().count("checkpoint-source-partition") == 2
    plugins.invoke(runtime[0], "resources_release", resource_id=identity)
    runtime[2].with_name("execution-mode").write_text("dask-checkpoint-reopen")
    fresh = approved(
        runtime, group + "_open", binding_id="local", idempotency_key="checkpoint-reopened"
    )["data"]["resource_id"]
    assert (
        command(runtime, group, fresh, "count_rows", key="checkpoint-restored-count")["value"] == 3
    )
    assert runtime[2].read_text().count("checkpoint-source-partition") == 2
    command(runtime, group, fresh, "shutdown", key="checkpoint-restored-close")


def test_installed_native_arrow_stream_is_lazy_paged_and_explicitly_closed(runtime, monkeypatch):
    group = "execution_services"
    identity = prepare(runtime, monkeypatch, group, "dask-streaming")
    opened = command(runtime, group, identity, "arrow_stream_open", batch_size=1)
    stream = opened["stream_id"]
    assert "streaming-native-partition" not in runtime[2].read_text()
    first = command(
        runtime,
        group,
        identity,
        "arrow_stream_read",
        key="arrow-read-first",
        stream_id=stream,
        limit=1,
    )
    assert first["rows"] == [{"x": -1.0}]
    assert first["complete"] is False
    assert runtime[2].read_text().count("streaming-native-partition") == 1
    second = command(
        runtime,
        group,
        identity,
        "arrow_stream_read",
        key="arrow-read-second",
        stream_id=stream,
        limit=1,
    )
    assert second["rows"] == [{"x": 1.0}]
    assert runtime[2].read_text().count("streaming-native-partition") == 1
    final = command(
        runtime,
        group,
        identity,
        "arrow_stream_read",
        key="arrow-read-final",
        stream_id=stream,
        limit=2,
    )
    assert final["rows"] == [{"x": None}]
    assert final["complete"] is True
    assert runtime[2].read_text().count("streaming-native-partition") == 2
    assert (
        command(runtime, group, identity, "arrow_stream_close", stream_id=stream)["closed"] is True
    )
    assert command(runtime, group, identity, "inspect")["arrow_streams"] == []
    command(runtime, group, identity, "shutdown")


def test_installed_write_behind_flush_and_reopen(runtime, monkeypatch):
    identity = prepare(runtime, monkeypatch, "store_services", "cached-behind")

    def call(name, **args):
        return command(runtime, "store_services", identity, name, **args)

    call("save", record_id="seed")
    stats = call("cache_flush")["stats"]
    assert stats["write_behind"]["pending"] == 0
    assert stats["write_behind"]["completed"] == 1
    assert not stats["write_behind"]["outcome_unknown"]
    assert "private-" not in str(stats)
    call("cache_invalidate_all")
    assert call("get", record_id="seed")["record"]
    worker = runtime[0]._session._administration["store_services"].workers[identity]
    plugins.invoke(runtime[0], "resources_release", resource_id=identity)
    assert worker.closed and worker.process.poll() is not None
    restored = approved(
        runtime, "store_services_open", binding_id="local", idempotency_key="write-behind-reopen"
    )["data"]["resource_id"]
    row = command(
        runtime,
        "store_services",
        restored,
        "get",
        key="write-behind-restored-get",
        record_id="seed",
    )
    assert row["record"]
    plugins.invoke(runtime[0], "resources_release", resource_id=restored)
