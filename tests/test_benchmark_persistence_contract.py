"""Persisted benchmark evidence must retain actual measurements across restart."""

from datetime import datetime, timedelta

import pytest

from truthound.benchmark import (
    BenchmarkCategory,
    BenchmarkMetrics,
    BenchmarkResult,
    EnvironmentInfo,
)
from truthound.benchmark.comparison import RegressionDetector
from truthound.benchmark.runner import SuiteResult


def suite(durations):
    environment = EnvironmentInfo.capture()
    started = datetime(2026, 1, 2, 3, 4, 5)
    result = BenchmarkResult(
        benchmark_name="synthetic",
        category=BenchmarkCategory.PROFILING,
        success=True,
        metrics=BenchmarkMetrics(durations=list(durations), rows_processed=10),
        environment=environment,
        started_at=started,
        completed_at=started + timedelta(seconds=5),
        parameters={"rows": 10},
        tags={"suite": "synthetic"},
    )
    return SuiteResult("synthetic", [result], environment, started, started + timedelta(seconds=5))


def test_saved_suite_preserves_raw_measurements_and_detects_restart_regression(tmp_path):
    baseline = suite([1.0, 1.1, 1.2])
    detector = RegressionDetector(tmp_path)
    detector.save_baseline(baseline)
    restarted = RegressionDetector(tmp_path)
    restored = restarted.get_baseline()
    assert restored.to_dict() == baseline.to_dict()
    assert restored.results[0].metrics.durations == [1.0, 1.1, 1.2]
    assert restarted.check(suite([3.0, 3.1, 3.2]))[0].has_regression
    restarted.load_history()
    assert restarted._history and all(isinstance(item, SuiteResult) for item in restarted._history)
    count = len(restarted._history)
    restarted.load_history()
    assert len(restarted._history) == count


def test_corrupt_existing_baseline_is_not_reported_as_missing(tmp_path):
    (tmp_path / "baseline.json").write_text('{"broken":')
    with pytest.raises(ValueError):
        RegressionDetector(tmp_path).get_baseline()


@pytest.mark.parametrize(
    "corruption", ["missing_samples", "duplicate_key", "nonfinite", "iteration_count"]
)
def test_invalid_measurement_evidence_cannot_be_used_as_a_baseline(tmp_path, corruption):
    import json

    path = tmp_path / "baseline.json"
    value = suite([1, 2, 3]).to_dict()
    if corruption == "missing_samples":
        del value["results"][0]["metrics"]["timing"]["durations_seconds"]
    elif corruption == "iteration_count":
        value["results"][0]["metrics"]["timing"]["iterations"] = 50
    payload = json.dumps(value)
    if corruption == "duplicate_key":
        payload = payload.replace('"success": true', '"success": true, "success": false')
    elif corruption == "nonfinite":
        payload = payload.replace('"rows": 10', '"rows": 1e999')
    path.write_text(payload)
    with pytest.raises(ValueError):
        RegressionDetector(tmp_path).get_baseline()


def test_failed_atomic_replacement_keeps_previous_measurements(tmp_path, monkeypatch):
    from pathlib import Path

    path = tmp_path / "baseline.json"
    original = suite([1, 2, 3])
    original.save(path)
    saved = path.read_bytes()
    replace = Path.replace

    def reject(selected, target):
        if Path(target) == path:
            raise OSError("synthetic replacement failure")
        return replace(selected, target)

    monkeypatch.setattr(Path, "replace", reject)
    with pytest.raises(OSError):
        suite([3, 4, 5]).save(path)
    assert path.read_bytes() == saved
    assert SuiteResult.load(path).to_dict() == original.to_dict()
