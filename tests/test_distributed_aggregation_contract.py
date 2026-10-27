"""Plans keep all groups and streamed aggregates retain actual merge state."""

from copy import deepcopy

import dask.dataframe as dd
import polars as pl
import pytest

from truthound.execution.distributed.aggregations import AggregationPlan, DistributedAggregator
from truthound.execution.distributed.protocols import DistinctCountAggregator, MinMaxAggregator

pytestmark = pytest.mark.contract


def test_spec_conversion_never_consumes_plan_aliases():
    plan = AggregationPlan().add_mean("x", alias="observed")
    before = deepcopy(plan)
    assert plan.to_spec().aggregations[0].alias == "observed"
    assert plan == before
    assert plan.to_spec().aggregations[0].alias == "observed"


@pytest.mark.parametrize("backend", ["local", "dask"])
def test_grouped_filtered_plan_returns_every_typed_key_with_correct_ddof(backend):
    source = pl.DataFrame({"g": ["a", "a", None, None], "x": [1, 3, 5, 9]})
    data = dd.from_pandas(source.to_pandas(), npartitions=2) if backend == "dask" else source
    plan = (
        AggregationPlan()
        .add_count()
        .add_mean("x")
        .add_std("x", ddof=0)
        .add_group_by("g")
        .add_filter("x > 1")
    )
    result = DistributedAggregator(data).execute(plan)
    grouped = {row["keys"]["g"]: row["aggregations"] for row in result["groups"]}
    assert grouped == {
        "a": {"*_count": 1, "x_mean": 3.0, "x_std": 0.0},
        None: {"*_count": 2, "x_mean": 7.0, "x_std": 2.0},
    }


@pytest.mark.parametrize("backend", ["local", "dask"])
def test_streaming_emits_cumulative_states_without_averaging_partition_means(backend):
    source = pl.DataFrame({"x": [1, 2, 100, None]})
    data = dd.from_pandas(source.to_pandas(), npartitions=2) if backend == "dask" else source
    plan = AggregationPlan().add_count().add_count("x").add_mean("x").add_std("x", ddof=0)
    snapshots = list(DistributedAggregator(data).execute_streaming(plan, batch_size=1))
    assert len(snapshots) == 4
    assert [row["*_count"] for row in snapshots] == [1, 2, 3, 4]
    final = snapshots[-1]
    assert final["x_count"] == 3
    assert final["x_mean"] == pytest.approx(103 / 3)
    assert final["x_std"] == pytest.approx(pl.Series([1, 2, 100]).std(ddof=0))
    assert DistributedAggregator(data).execute(plan) == final


def test_distinct_is_exact_across_hash_collisions_and_exhaustion_is_visible():
    assert hash(-1) == hash(-2)
    agg = DistinctCountAggregator(max_sample=2)
    state = agg.initialize()
    for value in [-1, -2]:
        state = agg.accumulate(state, value)
    assert agg.finalize(state) == 2
    with pytest.raises(MemoryError):
        agg.accumulate(state, 3)
    another = agg.accumulate(agg.initialize(), 3)
    with pytest.raises(MemoryError):
        agg.merge(state, another)


def test_minmax_merge_preserves_zero_and_exact_large_integer():
    agg = MinMaxAggregator()
    zero = agg.accumulate(agg.initialize(), 0)
    merged = agg.merge(zero, agg.initialize())
    assert agg.finalize(merged) == {"min": 0, "max": 0}
    large = 2**63 - 1
    state = agg.accumulate(agg.initialize(), large)
    assert agg.finalize(state) == {"min": large, "max": large}


def test_revoked_stream_does_not_start_the_next_native_batch(monkeypatch):
    from truthound.execution.distributed.protocols import AGGREGATOR_REGISTRY, SumAggregator

    calls = []
    allowed = [True]

    class Observed(SumAggregator):
        def accumulate(self, state, value):
            calls.append(value)
            return super().accumulate(state, value)

    def guard():
        if not allowed[0]:
            raise PermissionError("revoked")

    monkeypatch.setitem(AGGREGATOR_REGISTRY, "observed", Observed)
    native = DistributedAggregator(pl.DataFrame({"x": [1, 2]}))
    native.set_execution_guard(guard)
    stream = native.execute_streaming(AggregationPlan().add("x", "observed"), batch_size=1)
    assert next(stream) == {"x_observed": 1}
    allowed[0] = False
    with pytest.raises(PermissionError, match="revoked"):
        next(stream)
    assert calls == [1]


def test_spec_edit_cannot_change_the_original_executable_plan():
    plan = AggregationPlan().add_mean("x")
    spec = plan.to_spec()
    spec.group_by.append("g")
    data = pl.DataFrame({"g": ["a", "b"], "x": [1, 3]})
    assert DistributedAggregator(data).execute(plan) == {"x_mean": 2}


def test_local_polars_plan_does_not_require_pandas(tmp_path):
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent("""
        import importlib.abc, sys
        class NoPandas(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "pandas" or fullname.startswith("pandas."):
                    raise ModuleNotFoundError("pandas intentionally absent", name="pandas")
        sys.meta_path.insert(0, NoPandas())
        import polars as pl
        from truthound.execution.distributed.aggregations import AggregationPlan, DistributedAggregator
        assert DistributedAggregator(pl.DataFrame({"x":[1,3]})).execute(AggregationPlan().add_mean("x")) == {"x_mean":2}
    """)
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
