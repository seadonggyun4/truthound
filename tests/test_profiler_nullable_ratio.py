import polars as pl
import pytest

from truthound.profiler.base import ProfilerConfig
from truthound.profiler.column_profiler import BasicStatsAnalyzer


@pytest.mark.parametrize(
    "values,expected",
    [
        ([1, 2, None, 4], 1.0),
        ([1, 1, None], 0.5),
        ([None, None], 0.0),
        ([], 0.0),
        ([1, 2, 3], 1.0),
    ],
)
def test_non_null_unique_ratio_excludes_null_bucket(values, expected):
    frame = pl.DataFrame({"value": pl.Series(values, dtype=pl.Int64)}).lazy()
    result = BasicStatsAnalyzer().analyze("value", frame, ProfilerConfig())
    assert result["unique_ratio"] == expected
    assert 0 <= result["unique_ratio"] <= 1
    # Preserve the existing distinct-count contract, including its null bucket.
    assert result["distinct_count"] == len(set(values))
