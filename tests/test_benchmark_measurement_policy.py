"""Resource-bounded measurement must retain samples and genuine failures."""

import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from truthound.benchmark.parity import (
    NATIVE_THREAD_ENVIRON,
    BenchmarkMethodology,
    FrameworkObservation,
    ParityResult,
    ParityRunner,
    evaluate_parity_assertions,
)
from truthound.benchmark.workloads import load_suite_workloads

pytestmark = pytest.mark.contract


def test_empty_workload_catalog_cannot_pass():
    assertions = evaluate_parity_assertions(
        [], [], BenchmarkMethodology(), requested_frameworks=("truthound", "gx")
    )
    assert any(not item.passed and item.severity == "error" for item in assertions)


def workload():
    return next(w for w in load_suite_workloads("release-ga") if w.id == "local-mixed-core-suite")


def observation(framework="truthound"):
    w = workload()
    return FrameworkObservation(
        framework=framework,
        framework_version="fixture",
        workload_id=w.id,
        dataset_fingerprint=w.dataset_fingerprint,
        backend=w.backend.value,
        exactness=w.exactness,
        cold_start_seconds=0.01,
        warm_median_seconds=0.002,
        peak_rss_bytes=1024,
        correctness_passed=True,
        expected_issue_count=3,
        observed_issue_count=3,
        metadata={
            "row_count": 4,
            "warm_iterations": 7,
            "warm_durations_seconds": [0.002] * 7,
            "warm_process_cpu_seconds": [0.001] * 7,
            "iteration_correctness": [True] * 8,
            "worker_threads": 1,
            "polars_thread_pool_size": 1,
            "native_thread_limits": {name: "1" for name in NATIVE_THREAD_ENVIRON},
            "workload_contract_sha256": w.contract_sha256,
        },
    )


def test_new_methodology_is_bounded_but_legacy_artifacts_keep_original_meaning():
    current = BenchmarkMethodology()
    assert current.name == "truthound-parity-thread-budget-v2"
    assert current.warm_iterations == 7
    assert current.worker_threads == 1
    assert current.local_speedup_threshold == 1.5
    assert current.local_memory_ratio_threshold == 0.60
    assert BenchmarkMethodology.from_dict(current.to_dict()) == current
    old = BenchmarkMethodology.from_dict(
        {"name": "truthound-3.0-parity-gate", "warm_iterations": 2}
    )
    assert old.warm_iterations == 2 and old.worker_threads is None


@pytest.mark.parametrize("framework", ["truthound", "gx"])
def test_worker_sets_equal_thread_limits_before_child_imports(monkeypatch, tmp_path, framework):
    monkeypatch.setenv("POLARS_MAX_THREADS", "8")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "32")
    captured = []

    def run(command, **kwargs):
        captured.append((command, kwargs))
        Path(command[command.index("--output") + 1]).write_text(
            json.dumps(observation(framework).to_dict())
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("truthound.benchmark.parity.subprocess.run", run)
    result = ParityRunner()._run_worker(
        workload=workload(), framework=framework, artifact_root=tmp_path
    )
    command, kwargs = captured[0]
    assert command[command.index("--warm-iterations") + 1] == "7"
    for name in (
        "POLARS_MAX_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "TRUTHOUND_BENCHMARK_WORKER_THREADS",
    ):
        assert kwargs["env"][name] == "1"
    assert result.correctness_passed


@pytest.mark.parametrize(
    "mutation",
    ["missing", "short", "negative", "nonfinite", "wrong_median", "pool", "prior_failure"],
)
def test_measurement_receipt_rejects_missing_or_inconsistent_samples(mutation):
    item = observation()
    data = dict(item.metadata)
    if mutation == "missing":
        data.pop("warm_durations_seconds")
    elif mutation == "short":
        data["warm_durations_seconds"] = [0.002] * 2
    elif mutation == "negative":
        data["warm_durations_seconds"] = [-0.002] * 7
    elif mutation == "nonfinite":
        data["warm_durations_seconds"] = [float("nan")] * 7
    elif mutation == "wrong_median":
        data["warm_durations_seconds"] = [0.02] * 7
    elif mutation == "pool":
        data["polars_thread_pool_size"] = 8
    else:
        data["iteration_correctness"] = [False] + [True] * 7
    assertions = evaluate_parity_assertions(
        [workload()],
        [replace(item, metadata=data)],
        BenchmarkMethodology(),
        requested_frameworks=("truthound",),
    )
    check = next(a for a in assertions if a.name.endswith(":measurement-integrity"))
    assert not check.passed and check.severity == "error"


def test_valid_sample_receipt_preserves_actual_quality_failure_as_expected():
    item = observation()
    checks = evaluate_parity_assertions(
        [workload()], [item], BenchmarkMethodology(), requested_frameworks=("truthound",)
    )
    assert all(a.passed for a in checks)
    assert item.observed_issue_count == 3  # genuine failures are not changed into passes


@pytest.mark.parametrize("mismatch", ["methodology", "dataset", "backend", "exactness"])
def test_baseline_mismatch_is_not_reported_as_a_comparable_speedup(mismatch):
    current = observation()
    old = current
    methodology = BenchmarkMethodology()
    if mismatch == "methodology":
        methodology = replace(methodology, worker_threads=8)
    else:
        key = {"dataset": "dataset_fingerprint", "backend": "backend", "exactness": "exactness"}[
            mismatch
        ]
        old = replace(old, **{key: "different"})
    baseline = ParityResult(
        "release-ga", (old,), (), methodology, {}, "synthetic", datetime.now(), datetime.now()
    )
    checks = evaluate_parity_assertions(
        [workload()],
        [current],
        BenchmarkMethodology(),
        requested_frameworks=("truthound",),
        baseline_result=baseline,
    )
    assert any(a.name.endswith(":baseline-compatible") and not a.passed for a in checks)
    assert not any(a.name.endswith(":baseline-regression") for a in checks)


def test_worker_retains_earlier_incorrect_result_when_final_iteration_matches(
    monkeypatch, tmp_path
):
    import truthound
    from truthound.benchmark import _parity_worker as worker

    results = iter(
        [
            SimpleNamespace(success=False, issues=[SimpleNamespace(count=1)], metadata={}),
            SimpleNamespace(success=False, issues=[SimpleNamespace(count=3)], metadata={}),
        ]
    )
    monkeypatch.setattr(truthound, "check", lambda *args, **kwargs: next(results))
    monkeypatch.setattr(worker._PeakRSSMonitor, "start", lambda self: None)
    monkeypatch.setattr(worker._PeakRSSMonitor, "stop", lambda self: None)
    result = worker.execute_framework_observation(
        workload(), framework="truthound", artifact_dir=tmp_path, warm_iterations=1
    )
    assert result.correctness_passed is False
    assert result.observed_issue_count == 3
    assert result.metadata["iteration_correctness"] == [False, True]
    assert len(result.metadata["warm_durations_seconds"]) == 1
    assert len(result.metadata["warm_process_cpu_seconds"]) == 1


def test_gx_worker_retains_earlier_incorrect_result(monkeypatch, tmp_path):
    from truthound.benchmark import _parity_worker as worker

    w = next(w for w in load_suite_workloads("release-ga") if w.id == "local-null")
    validations = iter([{"success": True}, {"success": False, "result": {"unexpected_count": 1}}])
    monkeypatch.setitem(sys.modules, "great_expectations", ModuleType("great_expectations"))
    monkeypatch.setitem(
        sys.modules,
        "great_expectations.expectations",
        ModuleType("great_expectations.expectations"),
    )
    monkeypatch.setattr(worker.GreatExpectationsAdapter, "is_available", lambda self: (True, None))
    monkeypatch.setattr(worker, "_gx_get_context", lambda gx: object())
    monkeypatch.setattr(worker, "_gx_add_pandas_datasource", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        worker,
        "_gx_local_batch",
        lambda *args, **kwargs: SimpleNamespace(validate=lambda x: next(validations)),
    )
    monkeypatch.setattr(worker, "_gx_expectation_object", lambda *args: object())
    monkeypatch.setattr(type(w), "load_pandas", lambda self: object())
    monkeypatch.setattr(worker._PeakRSSMonitor, "start", lambda self: None)
    monkeypatch.setattr(worker._PeakRSSMonitor, "stop", lambda self: None)
    result = worker.execute_framework_observation(
        w, framework="gx", artifact_dir=tmp_path, warm_iterations=1
    )
    assert result.correctness_passed is False
    assert result.metadata["iteration_correctness"] == [False, True]
    assert result.observed_issue_count == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("worker_threads", None),
        ("worker_threads", 8),
        ("warm_iterations", 2),
        ("local_speedup_threshold", 1.0),
    ],
)
def test_release_claim_rejects_noncanonical_measurement_policy(monkeypatch, tmp_path, field, value):
    runner = ParityRunner(replace(BenchmarkMethodology(), **{field: value}))
    monkeypatch.setattr(runner, "_run_worker", lambda **kwargs: observation(kwargs["framework"]))
    result = runner.run_suite("release-ga", artifact_root=tmp_path)
    check = next(a for a in result.assertions if a.name == "release-ga:measurement-policy")
    assert not check.passed
    assert not result.metadata["release_claim_ready"]
