from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from truthound.benchmark import (
    BenchmarkMethodology,
    FrameworkObservation,
    ParityAssertion,
    ParityResult,
)
from truthound.cli import app


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.mark.contract
def test_benchmark_help_lists_parity(runner: CliRunner) -> None:
    result = runner.invoke(app, ["benchmark", "--help"])

    assert result.exit_code == 0
    assert "parity" in result.output


@pytest.mark.contract
def test_benchmark_parity_truthound_only_writes_json_artifact(runner: CliRunner) -> None:
    with runner.isolated_filesystem():
        output = Path("parity.json")
        result = runner.invoke(
            app,
            [
                "benchmark",
                "parity",
                "--suite",
                "pr-fast",
                "--frameworks",
                "truthound",
                "--backend",
                "local",
                "--output",
                str(output),
                "--strict",
            ],
        )

        assert result.exit_code == 0, result.output
        payload = json.loads(output.read_text())
        assert payload["suite_name"] == "pr-fast"
        assert payload["summary"]["blocking_failures"] == 0
        assert len(payload["observations"]) == 3
        assert output.with_suffix(".md").exists()
        assert output.with_suffix(".html").exists()
        assert Path("env-manifest.json").exists()


@pytest.mark.contract
def test_benchmark_parity_release_ga_requires_both_frameworks(runner: CliRunner) -> None:
    result = runner.invoke(
        app,
        [
            "benchmark",
            "parity",
            "--suite",
            "release-ga",
            "--frameworks",
            "truthound",
        ],
    )

    assert result.exit_code == 1
    assert "must run both frameworks" in result.output


@pytest.mark.contract
def test_benchmark_parity_release_ga_rejects_backend_filter(runner: CliRunner) -> None:
    result = runner.invoke(
        app,
        [
            "benchmark",
            "parity",
            "--suite",
            "release-ga",
            "--frameworks",
            "both",
            "--backend",
            "local",
        ],
    )

    assert result.exit_code == 1
    assert "without --backend" in result.output


@pytest.mark.contract
def test_benchmark_parity_release_ga_writes_release_summary(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeRunner:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def run_suite(self, *args, **kwargs) -> ParityResult:
            return ParityResult(
                suite_name="release-ga",
                observations=(
                    FrameworkObservation(
                        framework="truthound",
                        framework_version="test",
                        workload_id="local-null",
                        dataset_fingerprint="abc123",
                        backend="local",
                        exactness="exact",
                        cold_start_seconds=0.2,
                        warm_median_seconds=0.1,
                        peak_rss_bytes=100,
                        correctness_passed=True,
                        expected_issue_count=1,
                        observed_issue_count=1,
                        artifact_paths={},
                    ),
                    FrameworkObservation(
                        framework="gx",
                        framework_version="test",
                        workload_id="local-null",
                        dataset_fingerprint="abc123",
                        backend="local",
                        exactness="exact",
                        cold_start_seconds=0.4,
                        warm_median_seconds=0.2,
                        peak_rss_bytes=200,
                        correctness_passed=True,
                        expected_issue_count=1,
                        observed_issue_count=1,
                        artifact_paths={},
                    ),
                ),
                assertions=(
                    ParityAssertion(
                        name="release-ga:runner-policy",
                        passed=True,
                        message="ok",
                    ),
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
                artifact_root="/tmp/benchmarks",
                started_at=datetime.now(),
                completed_at=datetime.now(),
                metadata={"release_claim_ready": True},
            )

    monkeypatch.setattr("truthound.benchmark.ParityRunner", FakeRunner)

    with runner.isolated_filesystem():
        output = Path("release-ga.json")
        result = runner.invoke(
            app,
            [
                "benchmark",
                "parity",
                "--suite",
                "release-ga",
                "--frameworks",
                "both",
                "--output",
                str(output),
            ],
        )

        assert result.exit_code == 0, result.output
        assert Path("env-manifest.json").exists()
        assert Path("latest-benchmark-summary.md").exists()
        assert "Release summary" in result.output


@pytest.mark.contract
def test_benchmark_parity_strict_exits_on_blocking_failures(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeRunner:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def run_suite(self, *args, **kwargs) -> ParityResult:
            return ParityResult(
                suite_name="pr-fast",
                observations=(
                    FrameworkObservation(
                        framework="truthound",
                        framework_version="test",
                        workload_id="local-null",
                        dataset_fingerprint="abc123",
                        backend="local",
                        exactness="exact",
                        cold_start_seconds=0.1,
                        warm_median_seconds=0.1,
                        peak_rss_bytes=1024,
                        correctness_passed=False,
                        expected_issue_count=1,
                        observed_issue_count=0,
                        artifact_paths={},
                    ),
                ),
                assertions=(
                    ParityAssertion(
                        name="local-null:truthound:correctness",
                        passed=False,
                        message="broken",
                    ),
                ),
                methodology=BenchmarkMethodology(),
                environment={},
                artifact_root="/tmp/benchmarks",
                started_at=datetime.now(),
                completed_at=datetime.now(),
            )

    monkeypatch.setattr("truthound.benchmark.ParityRunner", FakeRunner)

    with runner.isolated_filesystem():
        result = runner.invoke(
            app,
            [
                "benchmark",
                "parity",
                "--suite",
                "pr-fast",
                "--frameworks",
                "truthound",
                "--strict",
            ],
        )

        assert result.exit_code == 1
