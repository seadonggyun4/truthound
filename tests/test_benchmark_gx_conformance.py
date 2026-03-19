from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from truthound.benchmark import load_workload, workload_root
from truthound.benchmark._parity_worker import execute_framework_observation

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", autouse=True)
def _require_gx_benchmark_dependencies() -> None:
    pytest.importorskip("great_expectations")
    pytest.importorskip("pandas")


@pytest.mark.integration
@pytest.mark.parametrize(
    "manifest_name",
    [
        "local-null.json",
        "local-unique.json",
        "local-range.json",
        "local-schema.json",
        "sqlite-null.json",
        "sqlite-unique.json",
        "sqlite-range.json",
    ],
)
def test_gx_conformance_workloads_match_manifest(tmp_path: Path, manifest_name: str) -> None:
    workload = load_workload(workload_root() / manifest_name)

    observation = execute_framework_observation(
        workload,
        framework="gx",
        artifact_dir=tmp_path / workload.id,
        warm_iterations=1,
    )

    assert observation.available is True
    assert observation.status == "ok"
    assert observation.correctness_passed is True
    assert observation.expected_issue_count == workload.expected.issue_count
    assert observation.observed_issue_count == workload.expected.issue_count
