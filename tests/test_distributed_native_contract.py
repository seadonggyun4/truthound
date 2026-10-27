"""Actual SDK regressions, isolated from shared cluster and JVM lifetimes."""

import psutil
import pytest

from tests.verification_support import OwnedProbe
from truthound.mcp.tables import decode

pytestmark = pytest.mark.contract


def test_windows_native_scratch_is_removed_only_after_its_owner_closes(tmp_path, monkeypatch):
    import tempfile
    from pathlib import Path
    from types import SimpleNamespace

    from tests.verification_support import _load

    probe = _load("native_distributed_probe")
    with tempfile.TemporaryDirectory(dir=tmp_path) as owner:
        with monkeypatch.context() as patch:
            patch.chdir(owner)
            patch.setattr(probe, "os", SimpleNamespace(name="nt"))
            with probe.runtime_directory("native-sdk-") as directory:
                scratch = Path(directory)
                assert scratch.is_relative_to(Path(owner))
                (scratch / "native.log").write_text("owned synthetic log")
            assert scratch.is_dir()
        assert scratch.is_dir()
    assert not Path(owner).exists() and not scratch.exists()


@pytest.mark.parametrize("backend", ["ray", "spark"])
def test_null_statistics_and_sampling_use_observed_rows(tmp_path, backend):
    runtime = OwnedProbe(backend, tmp_path / (backend + ".log"))
    try:
        assert runtime.receive()["ready"] is True
        basic = decode(runtime.call("basics"))
        sample = decode(runtime.call("sample"))
        assert (basic["rows"], basic["nulls"], basic["sum"]) == (3, 1, 4)
        assert (basic["stats"]["count"], sample["rows"]) == (2, 2)
        assert basic["stats"]["std"] == pytest.approx(2**0.5)
        assert decode(runtime.call("polars")) == {"rows": 3, "columns": ["x", "g"]}
        aggregates = decode(runtime.call("aggregations"))
        assert aggregates["median"] == 2
        assert aggregates["min"] == 1 and aggregates["max"] == 3
        assert aggregates["count"] == 2 and aggregates["null_count"] == 1
        assert aggregates["count_distinct"] == basic["distinct"]
        assert aggregates["first"] == 1 and aggregates["last"] == 3
        values = decode(runtime.call("values"))
        assert values["counts"] == {"a": 2, "b": 1}
        assert {row["value"]: row["count"] for row in values["nullable_counts"]} == {
            1: 1,
            None: 1,
            3: 1,
        }
        assert values["group_context_unchanged"] is True
        assert values["duplicates"] == 1 and values["pair_duplicates"] == 0
        # Spark's documented approximate rank quantile differs from Core's
        # linear Polars interpolation; preserve that native contract.
        assert values["quantiles"] == ([1, 1, 3] if backend == "spark" else [1, 2, 3])
        assert decode(runtime.call("partition")) == {"value": 3, "rows": 3, "errors": 0}
        plan_result = decode(runtime.call("aggregation_plan"))
        assert len(plan_result["snapshots"]) == 3
        assert plan_result["snapshots"][-1] == plan_result["final"]
        groups = {row["keys"]["g"]: row["aggregations"] for row in plan_result["final"]["groups"]}
        assert groups == {
            "a": {"*_count": 2, "x_count": 2, "x_mean": 2, "x_std": 1},
            "b": {"*_count": 1, "x_count": 0, "x_mean": 0, "x_std": 0},
        }
        native_spec = decode(runtime.call("aggregation_spec"))
        assert {row["keys"]["g"]: row["aggregations"] for row in native_spec["groups"]} == groups

    finally:
        runtime.close()
    for pid, created in runtime.owned.items():
        try:
            child = psutil.Process(pid)
            assert child.create_time() != created or child.status() == psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            pass


def test_ray_native_task_limit_actor_execution_and_block_target_are_applied(tmp_path):
    runtime = OwnedProbe("ray", tmp_path / "ray-configuration.log")
    try:
        assert runtime.receive()["ready"] is True
        result = decode(runtime.call("ray_configuration"))
        assert result["task_rows"] == result["actor_rows"] == result["calls"] == 4
        assert result["peak"] == 1 and result["all_actor"] is True
        assert result["target"] == 1024 * 1024 and result["global_unchanged"] is True
    finally:
        runtime.close()


@pytest.mark.parametrize("backend", ["ray", "spark"])
def test_native_row_cache_honors_disabled_cache_and_explicit_invalidation(tmp_path, backend):
    runtime = OwnedProbe(backend, tmp_path / (backend + "-cache.log"))
    try:
        assert runtime.receive()["ready"] is True
        assert decode(runtime.call("cache_configuration")) == {
            "uncached_calls": 2,
            "cached_calls": 1,
            "cleared_calls": 2,
            "rows": [3] * 5,
        }
    finally:
        runtime.close()


@pytest.mark.parametrize("backend", ["ray", "spark"])
def test_native_initial_partition_defaults_and_partition_scope(tmp_path, backend):
    runtime = OwnedProbe(backend, tmp_path / (backend + "-partitions.log"))
    try:
        assert runtime.receive()["ready"] is True
        result = decode(runtime.call("partition_configuration"))
        assert {
            key: result[key] for key in ("initial", "explicit", "rows", "captured", "root")
        } == {
            "initial": 2,
            "explicit": 1,
            "rows": 3,
            "captured": 2,
            "root": 2,
        }
        partitions = result["separate"]["partitions"]
        assert len(partitions) == 2
        assert sum(row["result"]["*_count"] for row in partitions) == 3
        assert sum(row["result"]["x_sum"] for row in partitions) == 4
        assert [row["result"] for row in result["empty"]["partitions"]] == [
            {"*_count": 0, "x_sum": 0}
        ] * 2
    finally:
        runtime.close()


def test_native_spark_false_flags_override_previous_true_and_capture_configuration(tmp_path):
    runtime = OwnedProbe("spark", tmp_path / "spark-flags.log")
    try:
        assert runtime.receive()["ready"] is True
        assert decode(runtime.call("spark_false_flags")) == {
            "arrow": "false",
            "adaptive": "false",
            "captured": {},
        }
    finally:
        runtime.close()


@pytest.mark.parametrize("backend", ["ray", "spark"])
def test_native_eager_mode_materializes_with_owned_spark_cache_cleanup(tmp_path, backend):
    runtime = OwnedProbe(backend, tmp_path / (backend + "-eager.log"))
    try:
        assert runtime.receive()["ready"] is True
        result = decode(runtime.call("eager_configuration"))
        assert result["materialized"] is True
        assert result["rows"] == (3 if backend == "ray" else 2)
        if backend == "spark":
            assert result["owned_cleaned"] is result["borrowed_retained"] is True
            assert result["borrowed_rows"] == 2
    finally:
        runtime.close()


@pytest.mark.parametrize("backend", ["ray", "spark"])
def test_native_partition_strategy_keeps_key_groups_ranges_and_round_robin_rows(tmp_path, backend):
    runtime = OwnedProbe(backend, tmp_path / (backend + "-strategies.log"))
    try:
        assert runtime.receive()["ready"] is True
        for strategy in ("column", "row_range", "round_robin", "custom"):
            result = decode(runtime.call("partition_strategy_" + strategy))
            assert result["count"] == len(result["partitions"]) == 3
            assert result["context_unchanged"] is True
            groups = [row["result"]["groups"] for row in result["partitions"]]
            assert sum(group["aggregations"]["*_count"] for part in groups for group in part) == 12
            if strategy == "column":
                assert all(
                    sum(any(group["keys"]["g"] == key for group in part) for part in groups) == 1
                    for key in ("a", "b")
                )
            elif strategy == "row_range":
                bounds = sorted(
                    (
                        min(group["aggregations"]["x_minmax"]["min"] for group in part),
                        max(group["aggregations"]["x_minmax"]["max"] for group in part),
                    )
                    for part in groups
                    if part
                )
                assert all(
                    left[1] <= right[0] for left, right in zip(bounds, bounds[1:], strict=False)
                )
            elif strategy == "round_robin":
                assert result["single_rows"] == 12
                assert [
                    sum(group["aggregations"]["*_count"] for group in part) for part in groups
                ] == [4, 4, 4]
    finally:
        runtime.close()


@pytest.mark.parametrize("backend", ["ray", "spark"])
def test_native_checkpoint_enabled_creates_real_snapshot(tmp_path, backend):
    runtime = OwnedProbe(backend, tmp_path / (backend + "-checkpoint.log"))
    try:
        assert runtime.receive()["ready"] is True
        result = decode(runtime.call("checkpoint_configuration"))
        assert result["rows"] == 3
        if backend == "ray":
            assert result == {"rows": 3, "restored_rows": 3, "sum": 4, "committed": True}
        else:
            assert result["checkpointed"] is result["files"] is True
    finally:
        runtime.close()


@pytest.mark.parametrize("backend", ["ray", "spark"])
def test_native_streaming_arrow_batches_and_early_close_preserve_runtime(tmp_path, backend):
    runtime = OwnedProbe(backend, tmp_path / (backend + "-streaming.log"))
    try:
        assert runtime.receive()["ready"] is True
        assert decode(runtime.call("streaming_configuration")) == {
            "first": 1,
            "sizes": [1, 1, 1],
            "rows": 3,
            "columns": ["x", "g"],
            "nulls": 1,
            "after_close_rows": 3,
            "memory_blocked": True,
        }
    finally:
        runtime.close()


def test_native_ray_empty_checkpoint_preserves_schema(tmp_path):
    runtime = OwnedProbe("ray", tmp_path / "ray-empty-checkpoint.log")
    try:
        assert runtime.receive()["ready"] is True
        assert decode(runtime.call("empty_checkpoint")) == {
            "rows": 0,
            "columns": ["x", "g"],
            "committed": True,
        }
    finally:
        runtime.close()


def test_native_spark_manual_arrow_fallback_enforces_transfer_policy(tmp_path):
    runtime = OwnedProbe("spark", tmp_path / "spark-manual-arrow.log")
    try:
        assert runtime.receive()["ready"] is True
        assert decode(runtime.call("manual_arrow_policy")) == {
            "sizes": [1, 1, 1], "invalid": [True, True, True], "blocked": True,
        }
    finally:
        runtime.close()


@pytest.mark.parametrize("backend", ["ray", "spark"])
def test_native_arrow_bridge_applies_selected_policy_without_session_mutation(tmp_path, backend):
    runtime = OwnedProbe(backend, tmp_path / (backend + "-arrow-bridge.log"))
    try:
        assert runtime.receive()["ready"] is True
        assert decode(runtime.call("arrow_bridge_policy")) == {
            "rows": 3, "sizes": [1, 1, 1], "null_rejected": True, "session_unchanged": True,
        }
    finally:
        runtime.close()
