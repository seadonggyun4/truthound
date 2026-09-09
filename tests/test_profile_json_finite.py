"""Strict JSON for undefined profile moments, without masking source values."""

from __future__ import annotations

import json
import math

import polars as pl
import pytest

from truthound.profiler import DataProfiler
from truthound.profiler.base import DistributionStats, ValueFrequency


@pytest.mark.parametrize(
    "values",
    [
        pytest.param([1], id="singleton"),
        pytest.param([1, 1], id="constant"),
        pytest.param([1, None, 1], id="nullable-constant"),
        pytest.param([1, None, 3, 5], id="nullable-varying"),
        pytest.param([None, None], id="all-null"),
        pytest.param([], id="empty"),
    ],
)
def test_measured_profile_round_trips_through_strict_json(values):
    data = pl.DataFrame({"measure": pl.Series(values, dtype=pl.Int64)})
    profile = DataProfiler(sample_size=None).profile(data.lazy())
    payload = profile.to_dict()
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload
    assert payload["row_count"] == len(values)
    assert payload["columns"][0]["null_count"] == values.count(None)
    finite = [value for value in values if value is not None]
    distribution = payload["columns"][0].get("distribution", {})
    if finite:
        assert distribution["min"] == min(finite)
        assert distribution["max"] == max(finite)
        assert distribution["mean"] == sum(finite) / len(finite)
        if len(set(finite)) == 1:
            assert distribution["skewness"] is None
            assert distribution["kurtosis"] is None
            # Serialization does not change the immutable measured object.
            assert math.isnan(profile.columns[0].distribution.skewness)


@pytest.mark.parametrize("field", ["skewness", "kurtosis"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_undefined_standardized_moments_are_null_not_zero(field, value):
    statistics = DistributionStats(mean=0.0, std=0.0, **{field: value})
    result = statistics.to_dict()
    assert result[field] is None
    assert result["mean"] == 0.0 and result["std"] == 0.0
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert not math.isfinite(getattr(statistics, field))


@pytest.mark.parametrize("field", ["skewness", "kurtosis"])
@pytest.mark.parametrize("value", [0.0, -0.0, 1.25, -2.75, 1e20])
def test_finite_standardized_moments_are_preserved_exactly(field, value):
    result = DistributionStats(**{field: value}).to_dict()
    assert result[field] == value
    assert math.copysign(1.0, result[field]) == math.copysign(1.0, value)


@pytest.mark.parametrize("field", ["min", "max", "mean", "std", "median", "q1", "q3"])
def test_nonmoment_measurements_are_not_silently_sanitized(field):
    result = DistributionStats(**{field: float("nan")}).to_dict()
    assert math.isnan(result[field])
    with pytest.raises(ValueError):
        json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_source_values_remain_strict_json_failures(value):
    result = ValueFrequency(value=value, count=1, ratio=1.0).to_dict()
    assert not math.isfinite(result["value"])
    with pytest.raises(ValueError):
        json.dumps(result, allow_nan=False)


def test_unmeasured_statistics_keep_existing_omission_contract():
    assert DistributionStats().to_dict() == {}
