from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from truthound._datasets import (
    QualityGateContext,
    QualityGateDisposition,
    QualityGatePolicy,
    QualityGateStatus,
    QualityGateType,
    evaluate_quality_gate,
)
from truthound._redaction import RedactionViolationError
from truthound.core.results import CheckResult, ExecutionIssue, ValidationRunResult
from truthound.types import Severity
from truthound.validators.base import ValidationIssue

RUN_TIME = datetime(2026, 5, 19, 12, 0, tzinfo=UTC)


def _context(
    *,
    gate_type: QualityGateType | str = QualityGateType.MERGE,
    **overrides: Any,
) -> QualityGateContext:
    values = {
        "quality_gate_id": "gate_001",
        "gate_type": gate_type,
        "asset_id": "asset_customers",
        "snapshot_id": "snapshot_001",
        "suite_ref": "suite://merge",
    }
    values.update(overrides)
    return QualityGateContext(**values)


def _issue(
    *,
    severity: Severity = Severity.LOW,
    validator_name: str = "null_check",
    issue_type: str = "null",
) -> ValidationIssue:
    return ValidationIssue(
        column="name",
        issue_type=issue_type,
        count=2,
        severity=severity,
        details="raw debug rows are deliberately not copied",
        expected="non-null",
        actual="null",
        sample_values=["person@example.com"],
        validator_name=validator_name,
    )


def _run(
    *,
    checks: tuple[CheckResult, ...],
    issues: tuple[ValidationIssue, ...] = (),
    execution_issues: tuple[ExecutionIssue, ...] = (),
) -> ValidationRunResult:
    return ValidationRunResult(
        run_id="run_001",
        run_time=RUN_TIME,
        suite_name="suite",
        source="dict",
        row_count=3,
        column_count=2,
        checks=checks,
        issues=issues,
        execution_issues=execution_issues,
    )


def test_clean_validation_run_projects_to_passed_gate() -> None:
    run = _run(checks=(CheckResult(name="schema", category="schema", success=True),))

    result = evaluate_quality_gate(run, _context())

    assert result.status == QualityGateStatus.PASSED
    assert result.blocking_failures == ()
    assert result.warnings == ()
    assert result.summary["check_count"] == 1
    assert result.run_ref == "run_001"


def test_default_policy_blocks_any_validation_issue() -> None:
    issue = _issue(severity=Severity.LOW)
    run = _run(
        checks=(
            CheckResult(
                name="not_null",
                category="completeness",
                success=False,
                issue_count=1,
                issues=(issue,),
            ),
        ),
        issues=(issue,),
    )

    result = evaluate_quality_gate(run, _context())

    assert result.status == QualityGateStatus.FAILED
    assert result.blocking_failures == (
        {
            "source": "validation_issue",
            "check": "not_null",
            "validator": "null_check",
            "issue_type": "null",
            "column": "name",
            "severity": "low",
            "count": 2,
            "disposition": "blocking",
        },
    )
    serialized = str(result.to_dict())
    assert "sample_values" not in serialized
    assert "person@example.com" not in serialized
    assert "raw debug rows" not in serialized


def test_policy_can_downgrade_low_severity_to_warning() -> None:
    issue = _issue(severity=Severity.LOW)
    run = _run(
        checks=(CheckResult(name="not_null", success=False, issue_count=1, issues=(issue,)),),
        issues=(issue,),
    )
    policy = QualityGatePolicy(severity_dispositions={Severity.LOW: "warning"})

    result = evaluate_quality_gate(run, _context(), policy)

    assert result.status == QualityGateStatus.WARNING
    assert result.blocking_failures == ()
    assert result.warnings[0]["disposition"] == QualityGateDisposition.WARNING.value


def test_issue_type_informational_override_keeps_gate_passed() -> None:
    issue = _issue(severity=Severity.HIGH, issue_type="profile_drift")
    run = _run(
        checks=(CheckResult(name="profile", success=False, issue_count=1, issues=(issue,)),),
        issues=(issue,),
    )
    policy = QualityGatePolicy(issue_type_dispositions={"profile_drift": "informational"})

    result = evaluate_quality_gate(run, _context(), policy)

    assert result.status == QualityGateStatus.PASSED
    assert result.blocking_failures == ()
    assert result.warnings == ()
    assert result.summary["informational_count"] == 1


def test_execution_issue_defaults_to_error_without_copying_message() -> None:
    run = _run(
        checks=(CheckResult(name="schema", success=True),),
        execution_issues=(
            ExecutionIssue(
                check_name="schema",
                message="database failure for person@example.com",
                exception_type="RuntimeError",
                failure_category="runtime",
                retry_count=1,
            ),
        ),
    )

    result = evaluate_quality_gate(run, _context())

    assert result.status == QualityGateStatus.ERROR
    assert result.blocking_failures[0]["source"] == "execution_issue"
    assert "database failure" not in str(result.to_dict())
    assert "person@example.com" not in str(result.to_dict())


def test_empty_checks_are_error_unless_policy_allows_them() -> None:
    run = _run(checks=())

    blocked = evaluate_quality_gate(run, _context())
    allowed = evaluate_quality_gate(
        run,
        _context(quality_gate_id="gate_002"),
        QualityGatePolicy(allow_empty_checks=True),
    )

    assert blocked.status == QualityGateStatus.ERROR
    assert blocked.blocking_failures[0]["reason"] == "empty_check_set"
    assert allowed.status == QualityGateStatus.PASSED


def test_skip_reason_projects_to_skipped_status() -> None:
    run = _run(checks=())

    result = evaluate_quality_gate(
        run,
        _context(skip_reason="manual maintenance window"),
    )

    assert result.status == QualityGateStatus.SKIPPED
    assert result.summary["skipped"] is True
    assert result.summary["skip_reason"] == "manual maintenance window"


def test_rollback_missing_evidence_is_error() -> None:
    run = _run(checks=(CheckResult(name="schema", success=True),))

    result = evaluate_quality_gate(run, _context(gate_type=QualityGateType.ROLLBACK))

    assert result.status == QualityGateStatus.ERROR
    assert result.blocking_failures[0]["reason"] == "missing_rollback_evidence"


def test_rollback_missing_target_is_failed() -> None:
    run = _run(checks=(CheckResult(name="schema", success=True),))

    result = evaluate_quality_gate(
        run,
        _context(
            gate_type=QualityGateType.ROLLBACK,
            target_snapshot_exists=False,
            previous_successful_gate_refs=("gate_passed",),
        ),
    )

    assert result.status == QualityGateStatus.FAILED
    assert result.blocking_failures[0]["reason"] == "rollback_target_missing"


def test_rollback_requires_prior_successful_gate_ref() -> None:
    run = _run(checks=(CheckResult(name="schema", success=True),))

    result = evaluate_quality_gate(
        run,
        _context(
            gate_type=QualityGateType.ROLLBACK,
            target_snapshot_exists=True,
            previous_successful_gate_refs=(),
        ),
    )

    assert result.status == QualityGateStatus.FAILED
    assert result.blocking_failures[0]["reason"] == "previous_successful_gate_missing"


def test_rollback_with_evidence_and_clean_run_passes() -> None:
    run = _run(checks=(CheckResult(name="schema", success=True),))

    result = evaluate_quality_gate(
        run,
        _context(
            gate_type=QualityGateType.ROLLBACK,
            target_snapshot_exists=True,
            previous_successful_gate_refs=("gate_passed",),
        ),
    )

    assert result.status == QualityGateStatus.PASSED
    assert result.summary["previous_successful_gate_count"] == 1


def test_quality_gate_context_rejects_unsafe_metadata() -> None:
    with pytest.raises(RedactionViolationError, match="raw_rows"):
        _context(metadata={"raw_rows": [{"email": "person@example.com"}]})


def test_quality_gate_runtime_remains_private_to_root_package() -> None:
    import truthound

    exported = dir(truthound)

    assert "QualityGatePolicy" not in exported
    assert "QualityGateContext" not in exported
    assert "evaluate_quality_gate" not in exported
