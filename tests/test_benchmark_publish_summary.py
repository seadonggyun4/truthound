from __future__ import annotations

import importlib.util
from datetime import datetime
from pathlib import Path

import pytest

from truthound.benchmark import (
    BenchmarkMethodology,
    FrameworkObservation,
    ParityAssertion,
    ParityResult,
)


def _load_publish_module():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "scripts"
        / "publish_benchmark_summary.py"
    )
    spec = importlib.util.spec_from_file_location("publish_benchmark_summary", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.contract
def test_render_docs_summary_includes_artifact_links_and_claim_status():
    module = _load_publish_module()
    result = ParityResult(
        suite_name="release-ga",
        observations=(
            FrameworkObservation(
                framework="truthound",
                framework_version="3.0.0rc1",
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
                framework_version="1.15.0",
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
            ParityAssertion(name="release-ga:runner-policy", passed=True, message="ok"),
        ),
        methodology=BenchmarkMethodology(),
        environment={},
        artifact_root="/tmp/truthound-benchmarks",
        started_at=datetime.now(),
        completed_at=datetime.now(),
        metadata={"release_claim_ready": True},
    )

    rendered = module.render_docs_summary(
        result,
        artifact_base_url="https://example.com/releases/v3.0.0",
    )

    assert "Official claim eligible: `yes`" in rendered
    assert "[release-ga.json](https://example.com/releases/v3.0.0/release-ga.json)" in rendered
    assert "| local-null | 0.100000 | 0.200000 | 2.00x | 50.00% | pass |" in rendered
