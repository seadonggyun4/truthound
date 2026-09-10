"""Malformed observations must never manufacture a release performance pass."""

from dataclasses import replace
from datetime import datetime

import pytest

from tests.test_benchmark_measurement_policy import observation as _observation
from tests.test_benchmark_measurement_policy import workload
from truthound.benchmark.parity import (
    NATIVE_THREAD_ENVIRON,
    BenchmarkMethodology,
    FrameworkObservation,
    ParityAssertion,
    ParityResult,
    evaluate_parity_assertions,
)

pytestmark = pytest.mark.contract


def _blocking(observations, *, baseline=None, methodology=None, frameworks=("truthound",)):
    checks = evaluate_parity_assertions(
        [workload()],
        observations,
        methodology or BenchmarkMethodology(),
        requested_frameworks=frameworks,
        baseline_result=baseline,
    )
    return [check.name for check in checks if not check.passed and check.severity == "error"]


def observation(framework="truthound"):
    candidate = _observation(framework)
    # A malformed shared fixture must not make every negative case pass.
    assert not _blocking([candidate], frameworks=(framework,))
    return candidate


def _baseline(old):
    return ParityResult(
        "release-ga",
        (old,),
        (),
        BenchmarkMethodology(),
        {},
        "synthetic",
        datetime.now(),
        datetime.now(),
    )


def test_duplicate_observation_cannot_overwrite_an_earlier_failure():
    good = observation()
    assert _blocking([replace(good, correctness_passed=False), good])


def test_nan_baseline_median_cannot_silently_skip_regression_gate():
    good = observation()
    assert _blocking(
        [good],
        baseline=_baseline(replace(good, warm_median_seconds=float("nan"))),
    )


def test_missing_baseline_samples_cannot_support_a_regression_claim():
    good = observation()
    metadata = dict(good.metadata)
    metadata.pop("warm_durations_seconds")
    assert _blocking([good], baseline=_baseline(replace(good, metadata=metadata)))


def test_v2_missing_thread_budget_cannot_downgrade_to_legacy_checks():
    try:
        methodology = BenchmarkMethodology.from_dict(
            {"name": "truthound-parity-thread-budget-v2", "warm_iterations": 7},
        )
    except ValueError:
        return
    malformed = replace(observation(), warm_median_seconds=float("nan"))
    assert _blocking([malformed], methodology=methodology)


def test_conflicting_native_limits_cannot_receive_resource_integrity_pass():
    good = observation()
    metadata = dict(good.metadata)
    metadata["native_thread_limits"] = {name: "999" for name in NATIVE_THREAD_ENVIRON}
    assert _blocking([replace(good, metadata=metadata)])


def test_infinite_comparator_rss_cannot_grant_zero_memory_ratio():
    good = observation()
    comparator = observation("gx")
    metadata = dict(comparator.metadata)
    metadata["warm_durations_seconds"] = [0.004] * 7
    comparator = replace(
        comparator,
        metadata=metadata,
        warm_median_seconds=0.004,
        peak_rss_bytes=float("inf"),
    )
    assert _blocking([good, comparator], frameworks=("truthound", "gx"))


def test_correctness_flag_cannot_override_mismatched_expected_issue_counts():
    invalid = replace(observation(), expected_issue_count=99, observed_issue_count=99)
    assert _blocking([invalid])


def test_duplicate_baseline_cannot_overwrite_an_earlier_failure():
    good = observation()
    baseline = replace(
        _baseline(good),
        observations=(replace(good, correctness_passed=False), good),
    )
    assert _blocking([good], baseline=baseline)


@pytest.mark.parametrize("worker_threads", [True, 1.9])
def test_v2_thread_budget_cannot_coerce_non_integer_values(worker_threads):
    try:
        methodology = BenchmarkMethodology.from_dict(
            {
                "name": "truthound-parity-thread-budget-v2",
                "warm_iterations": 7,
                "worker_threads": worker_threads,
            },
        )
    except (TypeError, ValueError):
        return
    assert _blocking([observation()], methodology=methodology)


@pytest.mark.parametrize(
    "field,value",
    [
        ("correctness_passed", "false"),
        ("available", "false"),
        ("expected_issue_count", 3.9),
        ("observed_issue_count", 3.9),
    ],
)
def test_observation_deserialization_cannot_coerce_failure_or_counts(field, value):
    payload = observation().to_dict()
    payload[field] = value
    try:
        malformed = FrameworkObservation.from_dict(payload)
    except (TypeError, ValueError):
        return
    assert _blocking([malformed])


def test_assertion_deserialization_cannot_coerce_false_string_to_pass():
    try:
        assertion = ParityAssertion.from_dict(
            {"name": "synthetic:correctness", "passed": "false"},
        )
    except (TypeError, ValueError):
        return
    assert assertion.passed is False
