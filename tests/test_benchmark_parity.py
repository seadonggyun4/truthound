from __future__ import annotations

import builtins
import json
from datetime import datetime
from pathlib import Path

import pytest

from truthound.benchmark import (
    BenchmarkMethodology,
    FrameworkObservation,
    ParityAssertion,
    ParityResult,
    ParityRunner,
    benchmark_artifact_root,
    capture_parity_environment,
    classify_release_blockers,
    default_env_manifest_path,
    default_release_summary_path,
    evaluate_parity_assertions,
    evaluate_release_environment_assertions,
    load_suite_workloads,
    load_workload,
    write_environment_manifest,
    write_release_summary,
)
from truthound.benchmark._parity_worker import _PeakRSSMonitor
from truthound.benchmark.base import EnvironmentInfo
from truthound.benchmark.workloads import workload_root


def _observation(
    *,
    framework: str,
    workload_id: str,
    backend: str,
    warm_median_seconds: float,
    peak_rss_bytes: int,
    observed_issue_count: int,
    correctness_passed: bool = True,
) -> FrameworkObservation:
    return FrameworkObservation(
        framework=framework,
        framework_version="test",
        workload_id=workload_id,
        dataset_fingerprint="abc123",
        backend=backend,
        exactness="exact",
        cold_start_seconds=warm_median_seconds * 1.25,
        warm_median_seconds=warm_median_seconds,
        peak_rss_bytes=peak_rss_bytes,
        correctness_passed=correctness_passed,
        expected_issue_count=1,
        observed_issue_count=observed_issue_count,
        artifact_paths={},
        metadata={"row_count": 4},
    )


@pytest.mark.contract
def test_release_ga_suite_contains_local_and_sql_exact_workloads():
    workloads = load_suite_workloads("release-ga")
    assert {workload.id for workload in workloads} == {
        "local-null",
        "local-unique",
        "local-range",
        "local-schema",
        "local-mixed-core-suite",
        "sqlite-null",
        "sqlite-unique",
        "sqlite-range",
    }


@pytest.mark.contract
def test_workload_dataset_fingerprint_is_stable():
    workload = load_workload(
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "workloads"
        / "local-null.json"
    )
    assert workload.dataset_fingerprint == workload.dataset_fingerprint
    assert len(workload.dataset_fingerprint) == 16


@pytest.mark.contract
def test_workload_root_prefers_repo_benchmarks_from_cwd(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("TRUTHOUND_BENCHMARK_WORKLOAD_ROOT", raising=False)
    monkeypatch.chdir(Path(__file__).resolve().parents[1])

    root = workload_root()

    assert root.name == "workloads"
    assert (root / "local-null.json").exists()


@pytest.mark.contract
@pytest.mark.parametrize(
    ("platform_name", "ru_maxrss", "expected_bytes"),
    (
        ("darwin", 4096, 4096),
        ("linux", 4096, 4096 * 1024),
    ),
)
def test_peak_rss_monitor_resource_fallback_respects_platform_units(
    monkeypatch: pytest.MonkeyPatch,
    platform_name: str,
    ru_maxrss: int,
    expected_bytes: int,
):
    import resource

    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "psutil":
            raise ImportError("psutil intentionally unavailable")
        return original_import(name, globals, locals, fromlist, level)

    class FakeUsage:
        def __init__(self, value: int) -> None:
            self.ru_maxrss = value

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(resource, "getrusage", lambda _: FakeUsage(ru_maxrss))
    monkeypatch.setattr("truthound.benchmark._parity_worker.sys.platform", platform_name)

    monitor = _PeakRSSMonitor()

    assert monitor._rss_bytes() == expected_bytes


@pytest.mark.contract
def test_evaluate_parity_assertions_detects_local_speed_and_memory_thresholds():
    methodology = BenchmarkMethodology()
    workload = next(
        item for item in load_suite_workloads("nightly-core") if item.id == "local-null"
    )
    observations = [
        _observation(
            framework="truthound",
            workload_id="local-null",
            backend="local",
            warm_median_seconds=1.0,
            peak_rss_bytes=80,
            observed_issue_count=1,
        ),
        _observation(
            framework="gx",
            workload_id="local-null",
            backend="local",
            warm_median_seconds=1.2,
            peak_rss_bytes=100,
            observed_issue_count=1,
        ),
    ]

    assertions = evaluate_parity_assertions(
        [workload],
        observations,
        methodology,
        requested_frameworks=("truthound", "gx"),
    )

    by_name = {assertion.name: assertion for assertion in assertions}
    assert by_name["local-null:truthound:correctness"].passed is True
    assert by_name["local-null:gx:correctness"].passed is True
    assert by_name["local-null:issue-parity"].passed is True
    assert by_name["local-null:local-speedup"].passed is False
    assert by_name["local-null:local-memory-ratio"].passed is False


@pytest.mark.contract
def test_evaluate_parity_assertions_uses_sql_thresholds():
    methodology = BenchmarkMethodology()
    workload = next(
        item for item in load_suite_workloads("release-ga") if item.id == "sqlite-null"
    )
    observations = [
        _observation(
            framework="truthound",
            workload_id="sqlite-null",
            backend="sqlite",
            warm_median_seconds=0.9,
            peak_rss_bytes=40,
            observed_issue_count=1,
        ),
        _observation(
            framework="gx",
            workload_id="sqlite-null",
            backend="sqlite",
            warm_median_seconds=1.0,
            peak_rss_bytes=50,
            observed_issue_count=1,
        ),
    ]

    assertions = evaluate_parity_assertions(
        [workload],
        observations,
        methodology,
        requested_frameworks=("truthound", "gx"),
    )
    by_name = {assertion.name: assertion for assertion in assertions}
    assert by_name["sqlite-null:sql-speedup"].passed is True


@pytest.mark.contract
def test_parity_result_round_trip_serialization():
    result = ParityResult(
        suite_name="pr-fast",
        observations=(
            _observation(
                framework="truthound",
                workload_id="local-null",
                backend="local",
                warm_median_seconds=0.1,
                peak_rss_bytes=1024,
                observed_issue_count=1,
            ),
        ),
        assertions=(
            ParityAssertion(
                name="local-null:truthound:correctness",
                passed=True,
                message="ok",
            ),
        ),
        methodology=BenchmarkMethodology(),
        environment=EnvironmentInfo.capture().to_dict(),
        artifact_root="/tmp/truthound-benchmarks",
        started_at=datetime.now(),
        completed_at=datetime.now(),
        metadata={"requested_frameworks": ("truthound",)},
    )

    restored = ParityResult.from_dict(json.loads(result.to_json()))
    assert restored.suite_name == result.suite_name
    assert restored.observations[0].framework == "truthound"
    assert restored.assertions[0].passed is True


@pytest.mark.contract
def test_capture_parity_environment_records_runner_metadata(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TRUTHOUND_BENCHMARK_RUNNER_CLASS", "self-hosted-fixed")
    monkeypatch.setenv("TRUTHOUND_BENCHMARK_RUNNER_LABELS", "self-hosted,benchmark-fixed")
    monkeypatch.setenv("TRUTHOUND_BENCHMARK_STORAGE_CLASS", "local-nvme")
    monkeypatch.setenv("TRUTHOUND_BENCHMARK_CPU_MODEL", "Test CPU")
    monkeypatch.setenv("TRUTHOUND_BENCHMARK_RAM_BYTES", str(32 * 1024**3))
    monkeypatch.setenv("TRUTHOUND_BENCHMARK_RELEASE_VERDICT", "true")

    environment = capture_parity_environment()

    assert environment["runner"]["class"] == "self-hosted-fixed"
    assert environment["runner"]["labels"] == ("self-hosted", "benchmark-fixed")
    assert environment["runner"]["storage_class"] == "local-nvme"
    assert environment["runner"]["release_verdict"] is True
    assert environment["machine"]["cpu_model"] == "Test CPU"
    assert environment["machine"]["ram_bytes"] == 32 * 1024**3
    assert environment["platform"]["python_minor"]


@pytest.mark.contract
def test_release_environment_assertions_require_fixed_runner():
    assertions = evaluate_release_environment_assertions(
        suite_name="release-ga",
        requested_frameworks=("truthound", "gx"),
        backend_filter=None,
        workload_count=8,
        environment={
            "platform": {"python_minor": "3.11"},
            "machine": {
                "cpu_model": "unknown",
                "cpu_logical_cores": 8,
                "ram_bytes": 16 * 1024**3,
            },
            "runner": {
                "class": "github-hosted-nightly",
                "labels": ("ubuntu-latest",),
                "storage_class": "network-ssd",
                "release_verdict": False,
            },
        },
    )

    by_name = {assertion.name: assertion for assertion in assertions}
    assert by_name["release-ga:runner-policy"].passed is False
    assert by_name["release-ga:runner-metadata"].passed is True
    assert by_name["release-ga:framework-selector"].passed is True
    assert by_name["release-ga:backend-filter"].passed is True
    assert by_name["release-ga:catalog-size"].passed is True


@pytest.mark.contract
def test_classify_release_blockers_separates_environment_and_performance():
    result = ParityResult(
        suite_name="release-ga",
        observations=(
            _observation(
                framework="truthound",
                workload_id="local-null",
                backend="local",
                warm_median_seconds=0.1,
                peak_rss_bytes=100,
                observed_issue_count=1,
            ),
            _observation(
                framework="gx",
                workload_id="local-null",
                backend="local",
                warm_median_seconds=0.3,
                peak_rss_bytes=200,
                observed_issue_count=1,
            ),
        ),
        assertions=(
            ParityAssertion(
                name="release-ga:runner-policy",
                passed=False,
                message="runner policy mismatch",
            ),
            ParityAssertion(
                name="local-null:local-memory-ratio",
                passed=False,
                message="memory ratio too high",
            ),
        ),
        methodology=BenchmarkMethodology(),
        environment={},
        artifact_root="/tmp/truthound-benchmarks",
        started_at=datetime.now(),
        completed_at=datetime.now(),
    )

    blockers = classify_release_blockers(result)

    assert blockers["primary"] == "environment"
    assert blockers["categories"] == ("environment", "performance")


@pytest.mark.contract
def test_write_environment_manifest_and_release_summary(tmp_path: Path):
    result = ParityResult(
        suite_name="release-ga",
        observations=(
            _observation(
                framework="truthound",
                workload_id="local-null",
                backend="local",
                warm_median_seconds=0.10,
                peak_rss_bytes=60,
                observed_issue_count=1,
            ),
            _observation(
                framework="gx",
                workload_id="local-null",
                backend="local",
                warm_median_seconds=0.20,
                peak_rss_bytes=120,
                observed_issue_count=1,
            ),
        ),
        assertions=(
            ParityAssertion(name="local-null:issue-parity", passed=True, message="ok"),
        ),
        methodology=BenchmarkMethodology(),
        environment={
            "runner": {
                "class": "self-hosted-fixed",
                "storage_class": "local-nvme",
            },
            "machine": {
                "cpu_model": "Benchmark CPU",
                "cpu_logical_cores": 8,
                "ram_bytes": 16 * 1024**3,
            },
        },
        artifact_root="/tmp/truthound-benchmarks",
        started_at=datetime.now(),
        completed_at=datetime.now(),
        metadata={"release_claim_ready": True},
    )

    env_manifest_path = write_environment_manifest(result, default_env_manifest_path(tmp_path / "release-ga.json"))
    summary_path = write_release_summary(
        result,
        default_release_summary_path(tmp_path / "release-ga.json"),
        env_manifest_path=env_manifest_path,
    )

    manifest_payload = json.loads(env_manifest_path.read_text())
    summary_text = summary_path.read_text()
    assert manifest_payload["suite_name"] == "release-ga"
    assert manifest_payload["environment"]["runner"]["class"] == "self-hosted-fixed"
    assert "Official benchmark claims" not in summary_text
    assert "eligible to back official benchmark claims" in summary_text
    assert "local-null" in summary_text
    assert "2.00x" in summary_text


@pytest.mark.contract
def test_benchmark_artifact_root_respects_env_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    override_root = tmp_path / "benchmark-root"
    monkeypatch.setenv("TRUTHOUND_BENCHMARK_ROOT", str(override_root))

    root = benchmark_artifact_root()

    assert root == override_root
    assert (root / "results").is_dir()
    assert (root / "baselines").is_dir()
    assert (root / "artifacts").is_dir()
    assert (root / "release").is_dir()


@pytest.mark.contract
def test_parity_runner_executes_truthound_pr_fast_suite(tmp_path):
    runner = ParityRunner(BenchmarkMethodology(warm_iterations=1))
    result = runner.run_suite(
        "pr-fast",
        frameworks="truthound",
        backend="local",
        artifact_root=tmp_path / ".truthound" / "benchmarks",
    )

    assert result.has_blocking_failures is False
    assert len(result.observations) == 3
    assert all(observation.framework == "truthound" for observation in result.observations)
    assert all(observation.correctness_passed for observation in result.observations)
    assert all(observation.artifact_paths["workspace_root"] for observation in result.observations)


@pytest.mark.contract
def test_parity_runner_executes_truthound_sqlite_suite(tmp_path: Path):
    runner = ParityRunner(BenchmarkMethodology(warm_iterations=1))
    result = runner.run_suite(
        "nightly-sql",
        frameworks="truthound",
        backend="sqlite",
        artifact_root=tmp_path / ".truthound" / "benchmarks",
    )

    assert result.has_blocking_failures is False
    assert len(result.observations) == 3
    assert all(observation.framework == "truthound" for observation in result.observations)
    assert all(observation.backend == "sqlite" for observation in result.observations)
    assert all(observation.correctness_passed for observation in result.observations)
    assert all(observation.artifact_paths["database_path"].endswith(".sqlite") for observation in result.observations)
