from __future__ import annotations

from datetime import datetime

import pytest

import truthound
from truthound.checkpoint.checkpoint import CheckpointResult, CheckpointStatus
from truthound.core.results import CheckResult, ValidationRunResult
from truthound.types import ResultFormat


def _sample_validation_run() -> ValidationRunResult:
    return ValidationRunResult(
        run_id="run_20260319_130000_checkpoint",
        run_time=datetime(2026, 3, 19, 13, 0, 0),
        suite_name="checkpoint_suite",
        source="customers.csv",
        row_count=10,
        column_count=3,
        result_format=ResultFormat.SUMMARY,
        execution_mode="sequential",
        checks=(
            CheckResult(
                name="unique_check",
                category="uniqueness",
                success=True,
            ),
        ),
    )


def test_truthound_dir_exposes_core_surface_only():
    exported = dir(truthound)

    assert "check" in exported
    assert "ValidationRunResult" in exported
    assert "checkpoint" not in exported
    assert "profiler" not in exported
    assert "ml" not in exported


def test_truthound_top_level_advanced_access_warns():
    with pytest.raises(AttributeError, match="has no attribute 'compare'"):
        _ = truthound.compare

    from truthound.drift import compare
    assert callable(compare)


def test_checkpoint_result_exposes_validation_run_and_view_only():
    run_result = _sample_validation_run()
    checkpoint_result = CheckpointResult(
        run_id="chk_001",
        checkpoint_name="daily_users",
        run_time=datetime(2026, 3, 19, 13, 5, 0),
        status=CheckpointStatus.SUCCESS,
        validation_run=run_result,
    )

    assert checkpoint_result.validation_run is run_result
    legacy_view = checkpoint_result.validation_view

    assert legacy_view.run_id == run_result.run_id
    assert legacy_view.statistics.total_validators == 1
    with pytest.raises(AttributeError, match="validation_result"):
        _ = checkpoint_result.validation_result

    restored = CheckpointResult.from_dict(checkpoint_result.to_dict())
    assert restored.validation_run is not None
    assert restored.validation_run.to_dict() == run_result.to_dict()
