"""Drift input selection and settings cannot silently change requested analysis."""

import math

import polars as pl
import pytest

from truthound.drift import compare
from truthound.drift.detectors import KSTestDetector, MMDDetector, PSIDetector


def test_zero_threshold_is_preserved_and_unknown_selected_column_rejected():
    frame = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
    result = compare(frame, frame, method="psi", threshold=0.0)
    assert result.columns[0].result.threshold == 0.0
    with pytest.raises(ValueError):
        compare(frame, frame, columns=["missing"])


@pytest.mark.parametrize(
    "constructor,settings",
    [
        (KSTestDetector, {"threshold": math.nan}),
        (KSTestDetector, {"threshold": 2}),
        (PSIDetector, {"n_bins": 0}),
        (MMDDetector, {"kernel": "unknown"}),
        (MMDDetector, {"max_samples": 1}),
        (MMDDetector, {"bandwidth": -1}),
    ],
)
def test_invalid_native_detector_settings_fail_at_construction(constructor, settings):
    with pytest.raises(ValueError):
        constructor(**settings)


def test_nonfinite_numeric_data_cannot_report_no_drift():
    with pytest.raises(ValueError):
        KSTestDetector().detect(pl.Series([math.nan, 1.0]), pl.Series([1.0, 2.0]))


def test_compare_rejects_invalid_sample_before_reading_sources():
    with pytest.raises(ValueError):
        compare("absent-baseline.csv", "absent-current.csv", sample_size=0)
