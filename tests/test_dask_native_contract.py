"""Actual partitioned Dask computations; no mocked scheduler/runtime evidence."""

import dask.dataframe as dd
import pandas as pd
import pytest

from truthound.execution._protocols import AggregationType
from truthound.execution.distributed.dask_engine import DaskExecutionEngine

pytestmark = pytest.mark.contract


def test_native_first_last_aggregations_preserve_order_and_nulls_across_partitions():
    engine = DaskExecutionEngine(
        dd.from_pandas(pd.DataFrame({"x": [4.0, 9.0, 12.0]}), npartitions=2)
    )
    assert engine.aggregate_column("x", AggregationType.FIRST) == 4.0
    assert engine.aggregate_column("x", AggregationType.LAST) == 12.0
    import math

    nulls = DaskExecutionEngine(
        dd.from_pandas(pd.DataFrame({"x": [None, 9.0, None]}), npartitions=2)
    )
    assert math.isnan(nulls.aggregate_column("x", AggregationType.FIRST))
    assert math.isnan(nulls.aggregate_column("x", AggregationType.LAST))
    empty = DaskExecutionEngine(
        dd.from_pandas(pd.DataFrame({"x": pd.Series(dtype=float)}), npartitions=1)
    )
    assert empty.aggregate_column("x", AggregationType.FIRST) is None
    assert empty.aggregate_column("x", AggregationType.LAST) is None


def test_native_sample_fills_requested_size_across_tiny_partitions():
    engine = DaskExecutionEngine(dd.from_pandas(pd.DataFrame({"x": [1, 2, 3]}), npartitions=3))
    with engine.sample(n=1, seed=42) as sampled:
        assert sampled.count_rows() == 1
        assert sampled.to_polars_lazyframe().collect().height == 1


def test_configured_synchronous_scheduler_and_arrow_conversion_execute_each_partition_once():
    import threading

    from truthound.execution.distributed.dask_engine import DaskEngineConfig

    calls = []
    caller = threading.get_ident()
    frame = dd.from_pandas(pd.DataFrame({"x": [1, 2, 3]}), npartitions=3)

    def observe(partition):
        calls.append(threading.get_ident())
        return partition

    native = frame.map_partitions(observe, meta=frame._meta)
    engine = DaskExecutionEngine(native, DaskEngineConfig(scheduler="synchronous"))
    assert engine.to_arrow_table().num_rows == 3
    assert calls == [caller] * 3


def test_disabled_row_cache_observes_changed_native_graph():
    from truthound.execution.distributed.dask_engine import DaskEngineConfig

    source = dd.from_pandas(pd.DataFrame({"x": [1, 2, 3]}), npartitions=1)
    rows = [1]

    def changing(partition):
        return partition.head(rows[0])

    native = source.map_partitions(changing, meta=source._meta)
    engine = DaskExecutionEngine(
        native, DaskEngineConfig(cache_results=False, scheduler="synchronous")
    )
    assert engine.count_rows() == 1
    rows[0] = 3
    assert engine.count_rows() == 3


def test_owned_distributed_client_closes_but_child_and_borrowed_client_survive():
    from distributed import Client

    from truthound.execution.distributed.dask_engine import DaskEngineConfig

    data = dd.from_pandas(pd.DataFrame({"x": [1, 2, 3]}), npartitions=3)
    config = DaskEngineConfig(
        scheduler="distributed",
        n_workers=1,
        threads_per_worker=1,
        memory_per_worker="0",
        dashboard_address=None,
    )
    with DaskExecutionEngine(data, config) as root:
        client = root.client
        assert root.count_rows() == 3
        with root.sample(n=1, seed=42) as child:
            assert child.count_rows() == 1
        assert client.status == "running"
    assert client.status == "closed"
    with Client(
        n_workers=1,
        threads_per_worker=1,
        processes=False,
        memory_limit=0,
        dashboard_address=None,
        set_as_default=False,
    ) as borrowed:
        with DaskExecutionEngine(data, config, client=borrowed) as root:
            assert root.count_nulls("x") == 0
        assert borrowed.status == "running"


def test_native_grouped_spec_preserves_all_keys_filter_and_scope():
    from truthound.execution.distributed.aggregations import AggregationPlan
    from truthound.execution.distributed.protocols import AggregationScope

    engine = DaskExecutionEngine(
        dd.from_pandas(pd.DataFrame({"g": ["a", "a", "b"], "x": [1, 3, 9]}), npartitions=2)
    )
    spec = AggregationPlan().add_mean("x").add_group_by("g").add_filter("x > 1").to_spec()
    result = engine._aggregate_distributed(spec)
    assert {row["keys"]["g"]: row["aggregations"]["x_mean"] for row in result["groups"]} == {
        "a": 3,
        "b": 9,
    }
    assert spec.scope == AggregationScope.GROUPED


def test_grouped_scope_without_keys_is_not_silently_global():
    from truthound.execution.distributed.protocols import AggregationScope, AggregationSpec

    engine = DaskExecutionEngine(dd.from_pandas(pd.DataFrame({"x": [1, 3]}), npartitions=2))
    spec = AggregationSpec(scope=AggregationScope.GROUPED).add("x", "mean")
    with pytest.raises(ValueError, match="group columns"):
        engine._aggregate_distributed(spec)


def test_initial_partition_setting_does_not_override_explicit_children():
    from truthound.execution.distributed.dask_engine import DaskEngineConfig

    source = dd.from_pandas(pd.DataFrame({"x": range(12)}), npartitions=4)
    config = DaskEngineConfig(num_partitions=3, scheduler="synchronous")
    with DaskExecutionEngine(source, config) as root:
        assert root.num_partitions == 3
        config.num_partitions = 4
        with root.repartition(2) as child:
            assert child.num_partitions == 2
            with child.select(["x"]) as selected:
                assert selected.num_partitions == 2
                assert selected.count_rows() == 12
        assert root.num_partitions == 3


def test_partition_scope_retains_separate_native_partitions_and_empty_results():
    from truthound.execution.distributed.protocols import AggregationScope, AggregationSpec

    source = dd.from_pandas(pd.DataFrame({"x": range(6)}), npartitions=3)
    engine = DaskExecutionEngine(source)
    spec = AggregationSpec(scope=AggregationScope.PARTITION).add("*", "count").add("x", "sum")
    partitions = engine._aggregate_distributed(spec)["partitions"]
    assert [row["partition_id"] for row in partitions] == [0, 1, 2]
    assert [row["result"] for row in partitions] == [
        {"*_count": 2, "x_sum": 1},
        {"*_count": 2, "x_sum": 5},
        {"*_count": 2, "x_sum": 9},
    ]
    spec.filter_condition = "x < 0"
    assert [row["result"] for row in engine._aggregate_distributed(spec)["partitions"]] == [
        {"*_count": 0, "x_sum": 0},
    ] * 3


def test_eager_mode_materializes_once_and_lazy_mode_defers_native_partitions():
    from truthound.execution.distributed.dask_engine import DaskEngineConfig
    from truthound.execution.distributed.protocols import ExecutionMode

    calls = []
    source = dd.from_pandas(pd.DataFrame({"x": range(6)}), npartitions=3)

    def observe(frame):
        calls.append(len(frame))
        return frame

    data = source.map_partitions(observe, meta=source._meta)
    with DaskExecutionEngine(
        data, DaskEngineConfig(execution_mode=ExecutionMode.LAZY, scheduler="synchronous")
    ):
        assert calls == []
    with DaskExecutionEngine(
        data,
        DaskEngineConfig(
            execution_mode=ExecutionMode.EAGER, scheduler="synchronous", cache_results=False
        ),
    ) as eager:
        assert len(calls) == 3
        assert eager.count_rows() == eager.count_rows() == 6
        assert len(calls) == 3


def test_distributed_eager_constructor_waits_for_selected_native_partitions(tmp_path):
    from truthound.execution.distributed.dask_engine import DaskEngineConfig
    from truthound.execution.distributed.protocols import ExecutionMode

    source = dd.from_pandas(pd.DataFrame({"x": range(3)}), npartitions=3)

    def observe(frame, partition_info=None):
        import time

        time.sleep(0.1)
        (tmp_path / str(partition_info["number"])).write_text("native partition complete")
        return frame

    data = source.map_partitions(observe, meta=source._meta)
    config = DaskEngineConfig(
        execution_mode=ExecutionMode.EAGER,
        scheduler="distributed",
        n_workers=1,
        threads_per_worker=1,
        memory_per_worker="0",
        dashboard_address=None,
    )
    with DaskExecutionEngine(data, config):
        assert sorted(path.name for path in tmp_path.iterdir()) == ["0", "1", "2"]


def test_partition_strategies_group_keys_sort_ranges_and_round_robin_rows():
    from truthound.execution.distributed.dask_engine import DaskEngineConfig
    from truthound.execution.distributed.protocols import PartitionStrategy

    source = dd.from_pandas(pd.DataFrame({"g": ["a", "b"] * 6, "x": range(12)}), npartitions=2)
    for strategy in (PartitionStrategy.ROW_HASH, PartitionStrategy.COLUMN):
        with DaskExecutionEngine(
            source,
            DaskEngineConfig(
                num_partitions=3,
                partition_strategy=strategy,
                partition_columns=("g",),
                scheduler="synchronous",
            ),
        ) as engine:
            partitions = [
                part.compute(scheduler="synchronous") for part in engine.dask_dataframe.to_delayed()
            ]
            assert len(partitions) == 3
            assert sorted(value for part in partitions for value in part.x) == list(range(12))
            assert all(
                sum(group in set(part.g) for part in partitions) == 1 for group in ("a", "b")
            )
    with DaskExecutionEngine(
        source,
        DaskEngineConfig(
            num_partitions=3,
            partition_strategy=PartitionStrategy.ROW_RANGE,
            partition_columns=("x",),
            scheduler="synchronous",
        ),
    ) as ranged:
        partitions = [
            part.compute(scheduler="synchronous") for part in ranged.dask_dataframe.to_delayed()
        ]
        assert [value for part in partitions for value in part.x] == list(range(12))
        assert len(partitions) == 3
    with DaskExecutionEngine(
        source,
        DaskEngineConfig(
            num_partitions=3,
            partition_strategy=PartitionStrategy.ROUND_ROBIN,
            scheduler="synchronous",
        ),
    ) as round_robin:
        partitions = [
            part.compute(scheduler="synchronous")
            for part in round_robin.dask_dataframe.to_delayed()
        ]
        assert [set(part.x) for part in partitions] == [{0, 3, 6, 9}, {1, 4, 7, 10}, {2, 5, 8, 11}]


def test_custom_partitioner_is_selected_once_and_invalid_output_is_rejected():
    from truthound.execution.distributed.dask_engine import DaskEngineConfig
    from truthound.execution.distributed.protocols import PartitionStrategy

    source = dd.from_pandas(pd.DataFrame({"x": range(6)}), npartitions=2)
    calls = []

    def partition(data, count):
        calls.append(count)
        return data.repartition(npartitions=count)

    config = DaskEngineConfig(
        num_partitions=3, partition_strategy=PartitionStrategy.CUSTOM, partitioner=partition
    )
    with DaskExecutionEngine(source, config) as engine:
        assert engine.num_partitions == 3 and engine.count_rows() == 6
        with engine.select(["x"]) as child:
            assert child.count_rows() == 6
    assert calls == [3]
    config.partitioner = lambda data, count: data.drop(columns=["x"])
    with pytest.raises(ValueError, match="schema"):
        DaskExecutionEngine(source, config)


def test_range_planning_uses_selected_scheduler_and_preserves_null_rows_or_fails_explicitly():
    import threading

    from truthound.execution.distributed.dask_engine import DaskEngineConfig
    from truthound.execution.distributed.protocols import PartitionStrategy

    source = dd.from_pandas(pd.DataFrame({"x": [5.0, 1.0, 9.0, 2.0, None, 7.0]}), npartitions=2)
    caller = threading.get_ident()

    def selected_thread(frame):
        assert threading.get_ident() == caller, "range planning ignored selected scheduler"
        return frame

    native = source.map_partitions(selected_thread, meta=source._meta)
    config = DaskEngineConfig(
        num_partitions=3,
        scheduler="synchronous",
        partition_strategy=PartitionStrategy.ROW_RANGE,
        partition_columns=("x",),
    )
    with DaskExecutionEngine(native, config) as engine:
        assert engine.count_rows() == 6
        assert engine.count_nulls("x") == 1


def test_native_dask_configured_checkpoint_and_reopen_do_not_reexecute_source(tmp_path):
    from truthound.execution.distributed.dask_engine import DaskEngineConfig

    calls = []
    source = dd.from_pandas(pd.DataFrame({"x": [1, 2, 3], "g": ["a", None, "b"]}), npartitions=2)

    def observed(partition):
        calls.append(len(partition))
        return partition

    selected = DaskEngineConfig(
        scheduler="synchronous",
        checkpoint_enabled=True,
        checkpoint_dir=str(tmp_path),
        cache_results=False,
    )
    with DaskExecutionEngine(
        source.map_partitions(observed, meta=source._meta), selected
    ) as engine:
        assert sorted(calls) == [1, 2]
        location = engine.checkpoint_path
        assert engine.count_rows() == engine.count_rows() == 3
        assert sorted(calls) == [1, 2]
        assert engine.checkpoint().count_rows() == 3
    with DaskExecutionEngine.from_checkpoint(
        location, DaskEngineConfig(scheduler="synchronous")
    ) as restored:
        assert restored.count_rows() == 3
        assert restored.count_nulls("g") == 1
        assert restored.aggregate_column("x", AggregationType.SUM) == 6
    assert sorted(calls) == [1, 2]


def test_native_dask_checkpoint_rejects_partial_and_corrupt_snapshot(tmp_path, monkeypatch):
    import os
    import stat
    from pathlib import Path

    from truthound.execution.distributed.dask_engine import DaskEngineConfig

    native_sync = os.fsync

    def writable_file_sync(descriptor):
        if stat.S_ISREG(os.fstat(descriptor).st_mode):
            # Enforce Windows CRT's writable-fd requirement on every platform.
            os.write(descriptor, b"")
        native_sync(descriptor)

    monkeypatch.setattr(os, "fsync", writable_file_sync)

    source = dd.from_pandas(pd.DataFrame({"x": [1, 2, 3]}), npartitions=2)
    engine = DaskExecutionEngine(
        source, DaskEngineConfig(scheduler="synchronous", checkpoint_dir=str(tmp_path))
    )
    with pytest.raises(ValueError):
        DaskExecutionEngine.from_checkpoint(tmp_path)
    checked = engine.checkpoint()
    artifact = next(Path(checked.checkpoint_path).rglob("*.parquet"))
    artifact.write_bytes(b"synthetic corrupted parquet")
    with pytest.raises(ValueError, match="checkpoint"):
        DaskExecutionEngine.from_checkpoint(checked.checkpoint_path)


def test_native_dask_streaming_conversion_stops_before_next_partition(tmp_path):
    from truthound.execution.distributed.dask_engine import DaskEngineConfig
    from truthound.execution.distributed.protocols import ExecutionMode

    calls = []
    source = dd.from_pandas(pd.DataFrame({"x": list(range(6))}), npartitions=3)

    def observed(partition):
        calls.append(partition["x"].tolist())
        return partition

    engine = DaskExecutionEngine(
        source.map_partitions(observed, meta=source._meta),
        DaskEngineConfig(
            scheduler="synchronous", execution_mode=ExecutionMode.STREAMING, arrow_batch_size=1
        ),
    )
    stream = engine.iter_arrow_batches()
    assert calls == []
    assert next(stream).column(0).to_pylist() == [0]
    stream.close()
    assert calls == [[0, 1]]
    assert engine.to_arrow_table().column("x").to_pylist() == list(range(6))


def test_native_dask_timeout_rejects_late_success_and_does_not_cache_it():
    import time

    from truthound.execution.distributed.dask_engine import DaskEngineConfig

    calls = []
    source = dd.from_pandas(pd.DataFrame({"x": [1]}), npartitions=1)

    def slow(partition):
        calls.append(1)
        time.sleep(0.03)
        return partition

    engine = DaskExecutionEngine(
        source.map_partitions(slow, meta=source._meta),
        DaskEngineConfig(scheduler="synchronous", timeout_seconds=0.01),
    )
    with pytest.raises(TimeoutError):
        engine.count_rows()
    assert calls == [1]
    assert not engine._cache


def test_native_dask_retry_requires_explicit_repeatable_error_and_obeys_count():
    from truthound.execution.distributed.dask_engine import DaskEngineConfig
    from truthound.execution.distributed.policies import RetryableExecutionError

    calls = []
    source = dd.from_pandas(pd.DataFrame({"x": [1]}), npartitions=1)

    def retryable(partition):
        calls.append(1)
        if len(calls) < 3:
            raise RetryableExecutionError("synthetic repeatable read failure")
        return partition

    engine = DaskExecutionEngine(
        source.map_partitions(retryable, meta=source._meta),
        DaskEngineConfig(scheduler="synchronous", retry_count=2, cache_results=False),
    )
    assert engine.count_rows() == 1
    assert len(calls) == 3
    calls.clear()
    limited = DaskExecutionEngine(
        source.map_partitions(retryable, meta=source._meta),
        DaskEngineConfig(scheduler="synchronous", retry_count=1),
    )
    with pytest.raises(RetryableExecutionError):
        limited.count_rows()
    assert len(calls) == 2
    calls.clear()

    def uncertain(partition):
        calls.append(1)
        raise OSError("synthetic uncertain read callback")

    uncertain_engine = DaskExecutionEngine(
        source.map_partitions(uncertain, meta=source._meta),
        DaskEngineConfig(scheduler="synchronous", retry_count=3),
    )
    with pytest.raises(OSError):
        uncertain_engine.count_rows()
    assert calls == [1]


def test_streaming_empty_arrow_keeps_native_schema_and_invalid_batch_size_fails():
    from truthound.execution.distributed.dask_engine import DaskEngineConfig
    from truthound.execution.distributed.protocols import ExecutionMode

    frame = dd.from_pandas(
        pd.DataFrame({"x": pd.Series(dtype="int64"), "s": pd.Series(dtype="string")}), npartitions=1
    )
    engine = DaskExecutionEngine(
        frame, DaskEngineConfig(scheduler="synchronous", execution_mode=ExecutionMode.STREAMING)
    )
    assert engine.to_arrow_table().column_names == ["x", "s"]
    assert str(engine.to_arrow_table().schema.field("x").type) == "int64"
    for value in [0, -1, True, 1.5]:
        with pytest.raises(ValueError):
            list(engine.iter_arrow_batches(value))


def test_arrow_iterator_failure_preserves_original_and_cleanup_errors():
    from truthound.execution.distributed.policies import checked_batches

    engine = DaskExecutionEngine(dd.from_pandas(pd.DataFrame({"x": [1]}), npartitions=1))

    class FailedIterator:
        def __iter__(self):
            return self

        def __next__(self):
            raise ValueError("synthetic native iterator failure")

        def close(self):
            raise OSError("synthetic native iterator cleanup failure")

    with pytest.raises(ExceptionGroup) as raised:
        next(checked_batches(engine, FailedIterator()))
    assert [type(error) for error in raised.value.exceptions] == [ValueError, OSError]


def test_native_arrow_transfer_rejects_batch_above_selected_memory_fraction():
    from truthound.execution.distributed.dask_engine import DaskEngineConfig
    from truthound.execution.distributed.protocols import ExecutionMode

    source = dd.from_pandas(pd.DataFrame({"x": ["synthetic-value" * 100]}), npartitions=1)
    engine = DaskExecutionEngine(
        source,
        DaskEngineConfig(
            scheduler="synchronous", execution_mode=ExecutionMode.STREAMING, memory_fraction=1e-15
        ),
    )
    with pytest.raises(MemoryError):
        next(engine.iter_arrow_batches())
