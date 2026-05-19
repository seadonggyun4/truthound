"""Private quality gate projection runtime for dataset repository contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from truthound._datasets._serialization import (
    DatasetArtifactContractError,
    normalize_str_tuple,
    require_mapping,
    require_non_empty_str,
)
from truthound._datasets.primitives import (
    QualityGateResult,
    QualityGateStatus,
    QualityGateType,
)
from truthound._datasets.redaction import assert_dataset_artifact_safe
from truthound.types import Severity

if TYPE_CHECKING:
    from collections.abc import Mapping

    from truthound.core.results import CheckResult, ExecutionIssue, ValidationRunResult
    from truthound.validators.base import ValidationIssue


class QualityGateDisposition(StrEnum):
    BLOCKING = "blocking"
    WARNING = "warning"
    INFORMATIONAL = "informational"


@dataclass(frozen=True)
class QualityGatePolicy:
    """Controls how validation issues are classified for a dataset gate."""

    default_issue_disposition: QualityGateDisposition | str = QualityGateDisposition.BLOCKING
    severity_dispositions: Mapping[str | Severity, QualityGateDisposition | str] = field(
        default_factory=dict
    )
    check_dispositions: Mapping[str, QualityGateDisposition | str] = field(default_factory=dict)
    validator_dispositions: Mapping[str, QualityGateDisposition | str] = field(default_factory=dict)
    issue_type_dispositions: Mapping[str, QualityGateDisposition | str] = field(
        default_factory=dict
    )
    execution_issue_disposition: QualityGateDisposition | str = QualityGateDisposition.BLOCKING
    allow_empty_checks: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "default_issue_disposition",
            _coerce_disposition(
                self.default_issue_disposition, field_name="default_issue_disposition"
            ),
        )
        object.__setattr__(
            self,
            "severity_dispositions",
            _normalize_disposition_map(
                self.severity_dispositions,
                field_name="severity_dispositions",
                normalize_key=_normalize_severity_key,
            ),
        )
        object.__setattr__(
            self,
            "check_dispositions",
            _normalize_disposition_map(self.check_dispositions, field_name="check_dispositions"),
        )
        object.__setattr__(
            self,
            "validator_dispositions",
            _normalize_disposition_map(
                self.validator_dispositions,
                field_name="validator_dispositions",
            ),
        )
        object.__setattr__(
            self,
            "issue_type_dispositions",
            _normalize_disposition_map(
                self.issue_type_dispositions,
                field_name="issue_type_dispositions",
            ),
        )
        object.__setattr__(
            self,
            "execution_issue_disposition",
            _coerce_disposition(
                self.execution_issue_disposition,
                field_name="execution_issue_disposition",
            ),
        )
        object.__setattr__(self, "allow_empty_checks", bool(self.allow_empty_checks))

    def disposition_for_issue(
        self,
        issue: ValidationIssue,
        *,
        check_name: str | None,
    ) -> QualityGateDisposition:
        """Resolve issue disposition with check > validator > issue_type > severity > default."""

        if check_name:
            disposition = self.check_dispositions.get(_normalize_key(check_name))
            if disposition is not None:
                return disposition
        if issue.validator_name:
            disposition = self.validator_dispositions.get(_normalize_key(issue.validator_name))
            if disposition is not None:
                return disposition
        disposition = self.issue_type_dispositions.get(_normalize_key(issue.issue_type))
        if disposition is not None:
            return disposition
        disposition = self.severity_dispositions.get(_normalize_severity_key(issue.severity))
        if disposition is not None:
            return disposition
        return self.default_issue_disposition


@dataclass(frozen=True)
class QualityGateContext:
    """Context supplied by Depot or Orchestration for a gate projection."""

    quality_gate_id: str
    gate_type: QualityGateType | str
    asset_id: str
    snapshot_id: str
    suite_ref: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    skip_reason: str | None = None
    target_snapshot_exists: bool | None = None
    previous_successful_gate_refs: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "quality_gate_id",
            require_non_empty_str(self.quality_gate_id, field_name="quality_gate_id"),
        )
        object.__setattr__(self, "gate_type", _coerce_gate_type(self.gate_type))
        object.__setattr__(
            self, "asset_id", require_non_empty_str(self.asset_id, field_name="asset_id")
        )
        object.__setattr__(
            self,
            "snapshot_id",
            require_non_empty_str(self.snapshot_id, field_name="snapshot_id"),
        )
        object.__setattr__(
            self,
            "suite_ref",
            require_non_empty_str(self.suite_ref, field_name="suite_ref"),
        )
        object.__setattr__(self, "metadata", require_mapping(self.metadata, field_name="metadata"))
        if self.skip_reason is not None:
            object.__setattr__(
                self,
                "skip_reason",
                require_non_empty_str(self.skip_reason, field_name="skip_reason"),
            )
        if self.previous_successful_gate_refs is not None:
            object.__setattr__(
                self,
                "previous_successful_gate_refs",
                normalize_str_tuple(
                    self.previous_successful_gate_refs,
                    field_name="previous_successful_gate_refs",
                ),
            )
        assert_dataset_artifact_safe(self.to_metadata(), label="quality gate context")

    def to_metadata(self) -> dict[str, Any]:
        return {
            "quality_gate_id": self.quality_gate_id,
            "gate_type": self.gate_type.value,
            "asset_id": self.asset_id,
            "snapshot_id": self.snapshot_id,
            "suite_ref": self.suite_ref,
            "metadata": dict(self.metadata),
            "skip_reason": self.skip_reason,
            "target_snapshot_exists": self.target_snapshot_exists,
            "previous_successful_gate_refs": list(self.previous_successful_gate_refs)
            if self.previous_successful_gate_refs is not None
            else None,
        }


def evaluate_quality_gate(
    run_result: ValidationRunResult,
    context: QualityGateContext,
    policy: QualityGatePolicy | None = None,
) -> QualityGateResult:
    """Project a ValidationRunResult into a deterministic QualityGateResult."""

    resolved_policy = policy or QualityGatePolicy()
    if context.skip_reason:
        return _build_result(
            run_result=run_result,
            context=context,
            status=QualityGateStatus.SKIPPED,
            blocking_failures=(),
            warnings=(),
            summary=_summary(
                run_result,
                context=context,
                blocking_count=0,
                warning_count=0,
                informational_count=0,
                skipped=True,
                error_count=0,
            ),
            metadata=_metadata(context, skipped=True),
        )

    blocking_failures: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    informational_count = 0

    issue_check_names = _issue_check_names(run_result.checks)
    for issue in run_result.issues:
        check_name = (
            issue_check_names.get(_issue_signature(issue))
            or issue.validator_name
            or issue.issue_type
        )
        disposition = resolved_policy.disposition_for_issue(issue, check_name=check_name)
        item = _validation_issue_item(issue, check_name=check_name, disposition=disposition)
        if disposition == QualityGateDisposition.BLOCKING:
            blocking_failures.append(item)
        elif disposition == QualityGateDisposition.WARNING:
            warnings.append(item)
        else:
            informational_count += 1

    blocking_execution_count = 0
    for execution_issue in run_result.execution_issues:
        disposition = resolved_policy.execution_issue_disposition
        item = _execution_issue_item(execution_issue, disposition=disposition)
        if disposition == QualityGateDisposition.BLOCKING:
            blocking_execution_count += 1
            blocking_failures.append(item)
        elif disposition == QualityGateDisposition.WARNING:
            warnings.append(item)
        else:
            informational_count += 1

    rollback_error_count, rollback_failure_count = _append_rollback_failures(
        context,
        blocking_failures=blocking_failures,
    )
    empty_check_error = 0
    if not run_result.checks and not resolved_policy.allow_empty_checks:
        empty_check_error = 1
        blocking_failures.append(
            {
                "source": "gate_policy",
                "reason": "empty_check_set",
                "disposition": QualityGateDisposition.BLOCKING.value,
            }
        )

    status = _status(
        blocking_failures=blocking_failures,
        warnings=warnings,
        blocking_execution_count=blocking_execution_count,
        rollback_error_count=rollback_error_count,
        rollback_failure_count=rollback_failure_count,
        empty_check_error=empty_check_error,
    )
    summary = _summary(
        run_result,
        context=context,
        blocking_count=len(blocking_failures),
        warning_count=len(warnings),
        informational_count=informational_count,
        skipped=False,
        error_count=blocking_execution_count + rollback_error_count + empty_check_error,
    )
    return _build_result(
        run_result=run_result,
        context=context,
        status=status,
        blocking_failures=tuple(blocking_failures),
        warnings=tuple(warnings),
        summary=summary,
        metadata=_metadata(context, policy=resolved_policy),
    )


def _build_result(
    *,
    run_result: ValidationRunResult,
    context: QualityGateContext,
    status: QualityGateStatus,
    blocking_failures: tuple[Mapping[str, Any], ...],
    warnings: tuple[Mapping[str, Any], ...],
    summary: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> QualityGateResult:
    return QualityGateResult(
        quality_gate_id=context.quality_gate_id,
        gate_type=context.gate_type,
        status=status,
        asset_id=context.asset_id,
        snapshot_id=context.snapshot_id,
        suite_ref=context.suite_ref,
        run_ref=run_result.run_id,
        summary=summary,
        blocking_failures=blocking_failures,
        warnings=warnings,
        created_at=run_result.run_time,
        metadata=metadata,
    )


def _append_rollback_failures(
    context: QualityGateContext,
    *,
    blocking_failures: list[dict[str, Any]],
) -> tuple[int, int]:
    if context.gate_type != QualityGateType.ROLLBACK:
        return 0, 0
    if context.target_snapshot_exists is None or context.previous_successful_gate_refs is None:
        blocking_failures.append(
            {
                "source": "rollback_evidence",
                "reason": "missing_rollback_evidence",
                "disposition": QualityGateDisposition.BLOCKING.value,
            }
        )
        return 1, 0
    if context.target_snapshot_exists is False:
        blocking_failures.append(
            {
                "source": "rollback_evidence",
                "reason": "rollback_target_missing",
                "disposition": QualityGateDisposition.BLOCKING.value,
            }
        )
        return 0, 1
    if not context.previous_successful_gate_refs:
        blocking_failures.append(
            {
                "source": "rollback_evidence",
                "reason": "previous_successful_gate_missing",
                "disposition": QualityGateDisposition.BLOCKING.value,
            }
        )
        return 0, 1
    return 0, 0


def _status(
    *,
    blocking_failures: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
    blocking_execution_count: int,
    rollback_error_count: int,
    rollback_failure_count: int,
    empty_check_error: int,
) -> QualityGateStatus:
    if blocking_execution_count or rollback_error_count or empty_check_error:
        return QualityGateStatus.ERROR
    if blocking_failures or rollback_failure_count:
        return QualityGateStatus.FAILED
    if warnings:
        return QualityGateStatus.WARNING
    return QualityGateStatus.PASSED


def _summary(
    run_result: ValidationRunResult,
    *,
    context: QualityGateContext,
    blocking_count: int,
    warning_count: int,
    informational_count: int,
    skipped: bool,
    error_count: int,
) -> dict[str, Any]:
    return {
        "gate_type": context.gate_type.value,
        "skipped": skipped,
        "skip_reason": context.skip_reason if skipped else None,
        "check_count": len(run_result.checks),
        "passed_check_count": sum(1 for check in run_result.checks if check.success),
        "failed_check_count": sum(1 for check in run_result.checks if not check.success),
        "validation_issue_count": len(run_result.issues),
        "execution_issue_count": len(run_result.execution_issues),
        "blocking_count": blocking_count,
        "warning_count": warning_count,
        "informational_count": informational_count,
        "error_count": error_count,
        "run_success": run_result.success,
        "row_count": run_result.row_count,
        "column_count": run_result.column_count,
        "rollback_target_exists": context.target_snapshot_exists,
        "previous_successful_gate_count": len(context.previous_successful_gate_refs or ()),
    }


def _metadata(
    context: QualityGateContext,
    *,
    policy: QualityGatePolicy | None = None,
    skipped: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "engine": "truthound._datasets.gates",
        "projection": "ValidationRunResult->QualityGateResult",
        "context": context.to_metadata(),
        "skipped": skipped,
    }
    if policy is not None:
        payload["policy"] = {
            "default_issue_disposition": policy.default_issue_disposition.value,
            "severity_dispositions": _disposition_map_to_dict(policy.severity_dispositions),
            "check_dispositions": _disposition_map_to_dict(policy.check_dispositions),
            "validator_dispositions": _disposition_map_to_dict(policy.validator_dispositions),
            "issue_type_dispositions": _disposition_map_to_dict(policy.issue_type_dispositions),
            "execution_issue_disposition": policy.execution_issue_disposition.value,
            "allow_empty_checks": policy.allow_empty_checks,
        }
    return payload


def _validation_issue_item(
    issue: ValidationIssue,
    *,
    check_name: str,
    disposition: QualityGateDisposition,
) -> dict[str, Any]:
    return {
        "source": "validation_issue",
        "check": check_name,
        "validator": issue.validator_name,
        "issue_type": issue.issue_type,
        "column": issue.column,
        "severity": issue.severity.value,
        "count": int(issue.count),
        "disposition": disposition.value,
    }


def _execution_issue_item(
    execution_issue: ExecutionIssue,
    *,
    disposition: QualityGateDisposition,
) -> dict[str, Any]:
    return {
        "source": "execution_issue",
        "check": execution_issue.check_name,
        "exception_type": execution_issue.exception_type,
        "failure_category": execution_issue.failure_category,
        "retry_count": execution_issue.retry_count,
        "disposition": disposition.value,
    }


def _issue_check_names(checks: tuple[CheckResult, ...]) -> dict[tuple[Any, ...], str]:
    names: dict[tuple[Any, ...], str] = {}
    for check in checks:
        for issue in check.issues:
            names.setdefault(_issue_signature(issue), check.name)
    return names


def _issue_signature(issue: ValidationIssue) -> tuple[Any, ...]:
    return (
        issue.validator_name,
        issue.issue_type,
        issue.column,
        int(issue.count),
        issue.severity.value,
    )


def _coerce_gate_type(value: QualityGateType | str) -> QualityGateType:
    if isinstance(value, QualityGateType):
        return value
    try:
        return QualityGateType(str(value))
    except ValueError as exc:
        raise DatasetArtifactContractError(
            f"Dataset quality gate type has unsupported value: {value!r}"
        ) from exc


def _coerce_disposition(
    value: QualityGateDisposition | str,
    *,
    field_name: str,
) -> QualityGateDisposition:
    if isinstance(value, QualityGateDisposition):
        return value
    try:
        return QualityGateDisposition(str(value).lower())
    except ValueError as exc:
        raise DatasetArtifactContractError(
            f"Dataset quality gate {field_name} has unsupported value: {value!r}"
        ) from exc


def _normalize_disposition_map(
    value: Mapping[Any, QualityGateDisposition | str],
    *,
    field_name: str,
    normalize_key: Any = None,
) -> dict[str, QualityGateDisposition]:
    mapping = require_mapping(value, field_name=field_name)
    key_normalizer = normalize_key or _normalize_key
    return {
        key_normalizer(key): _coerce_disposition(item, field_name=f"{field_name}.{key}")
        for key, item in mapping.items()
    }


def _normalize_key(value: Any) -> str:
    return str(value).strip().lower()


def _normalize_severity_key(value: Any) -> str:
    if isinstance(value, Severity):
        return value.value
    return str(value).strip().lower()


def _disposition_map_to_dict(
    value: Mapping[str, QualityGateDisposition],
) -> dict[str, str]:
    return {str(key): item.value for key, item in value.items()}


__all__ = [
    "QualityGateContext",
    "QualityGateDisposition",
    "QualityGatePolicy",
    "evaluate_quality_gate",
]
