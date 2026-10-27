"""Incremental reuse and chunked failure must not produce stale or complete results."""

import logging

import polars as pl
import pytest

from truthound.profiler import ColumnProfiler, DataProfiler
from truthound.profiler.column_profiler import BasicStatsAnalyzer, ColumnAnalyzer
from truthound.profiler.incremental import IncrementalConfig, IncrementalProfiler
from truthound.profiler.streaming import IncrementalStats, StreamingProfiler

pytestmark = pytest.mark.fault


def test_native_incremental_detects_value_changes_with_equal_counts_after_restore(tmp_path):
    from truthound.profiler.table_profiler import load_profile, save_profile

    native = IncrementalProfiler(config=IncrementalConfig(max_age=None))
    first = native.profile(pl.DataFrame({"x": [1, 2, 3]}))
    path = tmp_path / "baseline.json"
    save_profile(first, path)
    second = native.profile(pl.DataFrame({"x": [100, 200, 300]}), previous=load_profile(path))
    assert second["x"].distribution.mean == 200
    assert native.last_profiled_columns == {"x"} and not native.last_skipped_columns
    unchanged = native.profile(pl.DataFrame({"x": [100, 200, 300]}), previous=second)
    assert unchanged["x"].distribution.mean == 200
    assert native.last_skipped_columns == {"x"} and not native.last_profiled_columns


def test_native_incremental_does_not_reuse_incomplete_column():
    class Failed(ColumnAnalyzer):
        name = "failed"

        def analyze(self, *args):
            raise RuntimeError("synthetic-private-incremental")

    previous = DataProfiler(column_profiler=ColumnProfiler(analyzers=[BasicStatsAnalyzer(), Failed()])).profile(pl.DataFrame({"x": [1, 2]}).lazy())
    assert not previous.complete
    native = IncrementalProfiler(config=IncrementalConfig(max_age=None))
    result = native.profile(pl.DataFrame({"x": [1, 2]}), previous=previous)
    assert result.complete and native.last_profiled_columns == {"x"}


def test_native_streaming_chunk_failure_retains_safe_current_run_diagnostics(monkeypatch, caplog):
    original = IncrementalStats.update_from_chunk
    failed = False

    def update(self, frame, column, *args, **kwargs):
        nonlocal failed
        if column == "x" and not failed:
            failed = True
            raise RuntimeError("synthetic-private-stream-query")
        return original(self, frame, column, *args, **kwargs)

    monkeypatch.setattr(IncrementalStats, "update_from_chunk", update)
    native = StreamingProfiler(chunk_size=1)
    with caplog.at_level(logging.DEBUG):
        partial = native.profile_dataframe(pl.DataFrame({"x": [1, 2]}))
        fresh = native.profile_dataframe(pl.DataFrame({"x": [1, 2]}))
    assert not partial.complete and partial["x"].execution_issues
    assert partial["x"].execution_issues[0].error_type == "RuntimeError"
    assert fresh.complete and fresh["x"].row_count == 2
    assert "synthetic-private-stream-query" not in caplog.text + str(partial.to_dict())


def test_native_chunk_schema_failure_closes_generator_before_returning():
    from truthound.profiler.errors import AnalysisError

    events = []

    class Chunks:
        total_chunks = 2

        def __iter__(self):
            try:
                yield pl.DataFrame({"x": [1]})
                yield pl.DataFrame({"private-different-column": [2]})
            finally:
                events.append("closed")

    with pytest.raises(AnalysisError, match="schema changed"):
        StreamingProfiler(chunk_size=1).profile_chunks(Chunks())
    assert events == ["closed"]
