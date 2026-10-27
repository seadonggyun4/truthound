"""General-engine paging, configuration ownership and native failure contracts."""

import sqlite3
from contextlib import ExitStack

import polars as pl
import pytest

from truthound.datasources.sql.sqlite import SQLiteDataSource
from truthound.execution.base import ExecutionConfig
from truthound.execution.pandas_engine import PandasExecutionEngine
from truthound.execution.polars_engine import PolarsExecutionEngine
from truthound.execution.sql_engine import SQLExecutionEngine

pytestmark = pytest.mark.contract


@pytest.fixture(params=["polars", "pandas", "sqlite"])
def engine(request, tmp_path):
    frame = pl.DataFrame({"x": [1, 2, 3, 4]})
    if request.param == "polars":
        yield PolarsExecutionEngine(frame)
    elif request.param == "pandas":
        import pandas as pd

        yield PandasExecutionEngine(pd.DataFrame({"x": [1, 2, 3, 4]}))
    else:
        path = tmp_path / "engine.db"
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE records (x INTEGER)")
            connection.executemany("INSERT INTO records VALUES (?)", [(1,), (2,), (3,), (4,)])
        source = SQLiteDataSource(table="records", database=str(path))
        with source:
            yield SQLExecutionEngine(source)


def test_zero_limit_and_offset_without_implicit_cap(engine):
    assert engine.get_column_values("x", limit=0, offset=1) == []
    assert engine.get_distinct_values("x", limit=0) == []
    assert engine.get_value_counts("x", limit=0) == {}
    assert engine.get_column_values("x", offset=2) == [3, 4]
    for options in ({"limit": -1}, {"offset": -1}, {"limit": True}):
        with pytest.raises(ValueError):
            engine.get_column_values("x", **options)


def test_config_and_cached_results_do_not_share_mutable_state():
    config = ExecutionConfig(max_rows_for_numpy=2)
    engine = PolarsExecutionEngine(pl.DataFrame({"x": [1, None]}), config)
    config.max_rows_for_numpy = 100
    assert engine.config.max_rows_for_numpy == 2
    result = engine.count_nulls_all()
    result["x"] = 99
    assert engine.count_nulls_all()["x"] == 1


def test_native_filter_source_failure_is_not_reported_as_unsupported(monkeypatch):
    engine = PolarsExecutionEngine(pl.DataFrame({"x": [1]}))

    def fail(*args, **kwargs):
        raise OSError("source read failed")

    monkeypatch.setattr(pl.LazyFrame, "collect", fail)
    with pytest.raises(OSError, match="source read failed"):
        engine.count_matching("x > 0")


def test_polars_offset_does_not_silently_cap_at_one_million():
    frame = pl.DataFrame({"x": pl.int_range(0, 1_000_003, eager=True)})
    values = PolarsExecutionEngine(frame).get_column_values("x", offset=1)
    assert len(values) == 1_000_002 and values[-1] == 1_000_002


@pytest.mark.parametrize(
    "settings",
    [
        {"max_rows_for_numpy": 0},
        {"max_rows_for_polars": -1},
        {"sample_size_for_stats": True},
        {"cache_results": 1},
    ],
)
def test_invalid_engine_configuration_rejected(settings):
    with pytest.raises(ValueError):
        ExecutionConfig(**settings)


@pytest.mark.parametrize("column", ["count", "__truthound_count"])
def test_frequency_aggregation_preserves_reserved_column_names(column):
    engine = PolarsExecutionEngine(pl.DataFrame({column: [1, 1, 2]}))
    assert engine.get_value_counts(column) == {1: 2, 2: 1}


def test_pandas_cache_clear_refreshes_materialized_conversion():
    import pandas as pd

    engine = PandasExecutionEngine(pd.DataFrame({"x": ["before"]}))
    assert engine.to_polars_lazyframe().collect()["x"].to_list() == ["before"]
    engine.dataframe.loc[0, "x"] = "after"
    engine.clear_cache()
    assert engine.to_polars_lazyframe().collect()["x"].to_list() == ["after"]


def test_pandas_disabled_cache_observes_changed_input():
    import pandas as pd

    engine = PandasExecutionEngine(
        pd.DataFrame({"x": ["before"]}), ExecutionConfig(cache_results=False)
    )
    assert engine.to_polars_lazyframe().collect()["x"].to_list() == ["before"]
    engine.dataframe.loc[0, "x"] = "after"
    assert engine.to_polars_lazyframe().collect()["x"].to_list() == ["after"]


@pytest.mark.parametrize("backend", ["polars", "pandas", "sqlite"])
def test_duplicate_groups_preserve_nulls_and_reserved_column_names(backend, tmp_path):
    rows = [(1, "a"), (1, "a"), (None, "b"), (None, "b"), (2, "c")]
    stack = ExitStack()
    if backend == "sqlite":
        path = tmp_path / "duplicates.db"
        with sqlite3.connect(path) as connection:
            connection.execute('CREATE TABLE records ("_count" INTEGER, cnt TEXT)')
            connection.executemany("INSERT INTO records VALUES (?, ?)", rows)
        source = stack.enter_context(SQLiteDataSource(table="records", database=str(path)))
        selected = SQLExecutionEngine(source)
    elif backend == "pandas":
        import pandas as pd

        selected = PandasExecutionEngine(pd.DataFrame(rows, columns=["_count", "cnt"]))
    else:
        selected = PolarsExecutionEngine(pl.DataFrame(rows, schema=["_count", "cnt"], orient="row"))
    try:
        assert selected.count_duplicates(["_count", "cnt"]) == 2
        actual = selected.get_duplicate_values(["_count", "cnt"])
        assert {tuple(row[name] for name in ("_count", "cnt")) for row in actual} == {
            (1, "a"),
            (None, "b"),
        }
        assert selected.get_duplicate_values(["_count", "cnt"], limit=0) == []
        with pytest.raises(ValueError):
            selected.get_duplicate_values(["_count", "cnt"], limit=-1)
    finally:
        stack.close()


def test_sql_unsupported_pushdown_aggregation_uses_bounded_native_fallback(tmp_path):
    from truthound.execution._protocols import AggregationType
    from truthound.execution.base import ExecutionSizeError

    path = tmp_path / "aggregates.db"
    with sqlite3.connect(path) as connection:
        connection.execute('CREATE TABLE records ("value space" INTEGER)')
        connection.executemany("INSERT INTO records VALUES (?)", [(1,), (2,), (3,), (4,)])
    with SQLiteDataSource(table="records", database=str(path)) as source:
        selected = SQLExecutionEngine(source)
        assert selected.aggregate({"value space": AggregationType.MEDIAN}) == {
            "value space_median": 2.5
        }
        assert selected.aggregate({"value space": AggregationType.SUM}) == {"value space_sum": 10}
        assert selected.aggregate({}) == {}
        limited = SQLExecutionEngine(source, ExecutionConfig(max_rows_for_polars=2))
        with pytest.raises(ExecutionSizeError):
            limited.aggregate({"value space": AggregationType.MEDIAN})


def test_sql_statistics_return_native_dispersion_and_respect_materialization_budget(tmp_path):
    from truthound.execution.base import ExecutionSizeError

    path = tmp_path / "stats.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE records (x REAL)")
        connection.executemany("INSERT INTO records VALUES (?)", [(1,), (2,), (4,), (None,)])
    with SQLiteDataSource(table="records", database=str(path)) as source:
        actual = SQLExecutionEngine(source).get_stats("x")
        expected = PolarsExecutionEngine(pl.DataFrame({"x": [1.0, 2.0, 4.0, None]})).get_stats("x")
        assert actual == expected
        with pytest.raises(ExecutionSizeError):
            SQLExecutionEngine(source, ExecutionConfig(max_rows_for_polars=2)).get_stats("x")


def test_sql_range_values_are_bound_data_not_sql_syntax(tmp_path):
    path = tmp_path / "range.db"
    values = ["a", "b", "x' OR 1=1 --", None]
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE records (x TEXT)")
        connection.executemany("INSERT INTO records VALUES (?)", [(value,) for value in values])
    with SQLiteDataSource(table="records", database=str(path)) as source:
        selected = SQLExecutionEngine(source)
        assert selected.count_in_range("x", "a", "b") == 2
        assert selected.count_in_range("x", "x' OR 1=1 --", "x' OR 1=1 --") == 1
        assert selected.count_in_set("x", {"x' OR 1=1 --"}) == 1
        assert selected.count_rows() == 4


def test_duckdb_bound_conditions_and_complete_statistics(tmp_path):
    import duckdb

    from truthound.datasources.sql.duckdb import DuckDBDataSource

    path = tmp_path / "native.duckdb"
    with duckdb.connect(str(path)) as connection:
        connection.execute("CREATE TABLE records (x DOUBLE, label VARCHAR)")
        connection.executemany(
            "INSERT INTO records VALUES (?, ?)",
            [(1, "a"), (2, "x' OR 1=1 --"), (4, "c"), (None, None)],
        )
    with DuckDBDataSource(table="records", database=str(path)) as source:
        selected = SQLExecutionEngine(source)
        assert selected.get_stats("x")["median"] == 2
        assert selected.count_in_range("label", "x' OR 1=1 --", "x' OR 1=1 --") == 1
        assert selected.count_in_set("label", {"x' OR 1=1 --"}) == 1
        with selected.filter_by_condition("x >= 2") as child:
            assert child.count_in_range("x", 2, 4) == 2
            assert child.count_in_set("x", {4}) == 1


def test_shared_range_fallback_counts_rows_when_both_bounds_are_absent():
    from truthound.execution.base import BaseExecutionEngine

    engine = PolarsExecutionEngine(pl.DataFrame({"x": [5, None, 10]}))
    assert BaseExecutionEngine.count_in_range(engine, "x") == 3
    assert BaseExecutionEngine.count_outside_range(engine, "x") == 0
