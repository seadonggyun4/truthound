from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Any
from uuid import uuid4

from truthound.core.execution_modes import (
    PlannedExecutionMode,
    RuntimeExecutionMode,
    coarse_planned_execution_mode,
    normalize_planned_execution_mode,
    normalize_runtime_execution_mode,
)
from truthound.types import ResultFormat, Severity
from truthound.validators.base import ValidationIssue


def _get_reporter(name: str, **kwargs: Any) -> Any:
    """Load the reporter registry lazily from the outer adapter layer."""

    return import_module('truthound.reporters').get_reporter(name, **kwargs)


def _generate_validation_report(run_result: ValidationRunResult, **kwargs: Any) -> str:
    """Load validation docs generation lazily from the outer adapter layer."""

    return import_module('truthound.datadocs').generate_validation_report(run_result, **kwargs)


@dataclass(frozen=True)
class ExecutionIssue:
    check_name: str
    message: str
    exception_type: str | None = None
    failure_category: str | None = None
    retry_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            'check_name': self.check_name,
            'message': self.message,
            'exception_type': self.exception_type,
            'failure_category': self.failure_category,
            'retry_count': self.retry_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExecutionIssue:
        return cls(
            check_name=str(data.get('check_name', '')),
            message=str(data.get('message', '')),
            exception_type=data.get('exception_type'),
            failure_category=data.get('failure_category'),
            retry_count=int(data.get('retry_count', 0)),
        )


@dataclass(frozen=True)
class CheckResult:
    name: str
    category: str = 'general'
    success: bool = True
    issue_count: int = 0
    issues: tuple[ValidationIssue, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            'name': self.name,
            'category': self.category,
            'success': self.success,
            'issue_count': self.issue_count,
            'issues': [issue.to_dict() for issue in self.issues],
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckResult:
        issues = tuple(_validation_issue_from_dict(issue) for issue in data.get('issues', []))
        return cls(
            name=str(data.get('name', '')),
            category=str(data.get('category', 'general')),
            success=bool(data.get('success', False)),
            issue_count=int(data.get('issue_count', len(issues))),
            issues=issues,
            metadata=dict(data.get('metadata', {})),
        )


@dataclass(frozen=True)
class ValidationRunResult:
    suite_name: str
    source: str
    row_count: int
    column_count: int
    run_id: str = field(default_factory=lambda: f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}")
    run_time: datetime = field(default_factory=datetime.now)
    result_format: ResultFormat = ResultFormat.SUMMARY
    execution_mode: str = RuntimeExecutionMode.SEQUENTIAL.value
    planned_execution_mode: str | None = PlannedExecutionMode.SEQUENTIAL.value
    checks: tuple[CheckResult, ...] = field(default_factory=tuple)
    issues: tuple[ValidationIssue, ...] = field(default_factory=tuple)
    execution_issues: tuple[ExecutionIssue, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        runtime_mode = normalize_runtime_execution_mode(self.execution_mode)
        planned_mode = normalize_planned_execution_mode(
            self.planned_execution_mode or coarse_planned_execution_mode(runtime_mode)
        )
        object.__setattr__(self, "execution_mode", runtime_mode)
        object.__setattr__(self, "planned_execution_mode", planned_mode)

    @classmethod
    def from_suite(
        cls,
        *,
        suite: Any,
        issues: list[ValidationIssue],
        source: str,
        row_count: int,
        column_count: int,
        execution_mode: str,
        planned_execution_mode: str | None = None,
        execution_issues: list[ExecutionIssue] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ValidationRunResult:
        by_check: dict[str, list[ValidationIssue]] = {}
        for issue in issues:
            check_name = issue.validator_name or issue.issue_type
            by_check.setdefault(check_name, []).append(issue)

        checks: list[CheckResult] = []
        known_checks: set[str] = set()
        for spec in suite.checks:
            spec_issues = tuple(by_check.get(spec.name, []))
            checks.append(
                CheckResult(
                    name=spec.name,
                    category=spec.category,
                    success=len(spec_issues) == 0,
                    issue_count=len(spec_issues),
                    issues=spec_issues,
                    metadata=dict(spec.metadata),
                )
            )
            known_checks.add(spec.name)

        for name, extra_issues in by_check.items():
            if name in known_checks:
                continue
            checks.append(
                CheckResult(
                    name=name,
                    success=len(extra_issues) == 0,
                    issue_count=len(extra_issues),
                    issues=tuple(extra_issues),
                )
            )

        return cls(
            run_id=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}",
            run_time=datetime.now(),
            suite_name=suite.name,
            source=source,
            row_count=row_count,
            column_count=column_count,
            result_format=suite.evidence_policy.result_format.format,
            execution_mode=execution_mode,
            planned_execution_mode=planned_execution_mode,
            checks=tuple(checks),
            issues=tuple(issues),
            execution_issues=tuple(execution_issues or ()),
            metadata=dict(metadata or {}),
        )

    @property
    def has_issues(self) -> bool:
        return bool(self.issues)

    @property
    def has_failures(self) -> bool:
        return self.has_issues or bool(self.execution_issues)

    @property
    def success(self) -> bool:
        return not self.has_failures

    def filter_by_severity(self, min_severity: Severity) -> ValidationRunResult:
        filtered_issues = tuple(
            issue for issue in self.issues if issue.severity >= min_severity
        )
        filtered_checks = tuple(
            CheckResult(
                name=check.name,
                category=check.category,
                success=not any(issue.severity >= min_severity for issue in check.issues),
                issue_count=sum(1 for issue in check.issues if issue.severity >= min_severity),
                issues=tuple(
                    issue for issue in check.issues if issue.severity >= min_severity
                ),
                metadata=dict(check.metadata),
            )
            for check in self.checks
        )
        return ValidationRunResult(
            run_id=self.run_id,
            run_time=self.run_time,
            suite_name=self.suite_name,
            source=self.source,
            row_count=self.row_count,
            column_count=self.column_count,
            result_format=self.result_format,
            execution_mode=self.execution_mode,
            planned_execution_mode=self.planned_execution_mode,
            checks=filtered_checks,
            issues=filtered_issues,
            execution_issues=self.execution_issues,
            metadata=dict(self.metadata),
        )

    def with_metadata(self, **metadata: Any) -> ValidationRunResult:
        return ValidationRunResult(
            run_id=self.run_id,
            run_time=self.run_time,
            suite_name=self.suite_name,
            source=self.source,
            row_count=self.row_count,
            column_count=self.column_count,
            result_format=self.result_format,
            execution_mode=self.execution_mode,
            planned_execution_mode=self.planned_execution_mode,
            checks=self.checks,
            issues=self.issues,
            execution_issues=self.execution_issues,
            metadata={
                **self.metadata,
                **metadata,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            'run_id': self.run_id,
            'run_time': self.run_time.isoformat(),
            'suite_name': self.suite_name,
            'source': self.source,
            'row_count': self.row_count,
            'column_count': self.column_count,
            'result_format': self.result_format.value,
            'execution_mode': self.execution_mode,
            'planned_execution_mode': self.planned_execution_mode,
            'checks': [check.to_dict() for check in self.checks],
            'issues': [issue.to_dict() for issue in self.issues],
            'execution_issues': [issue.to_dict() for issue in self.execution_issues],
            'metadata': self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def render(self, format: str = 'console', **kwargs: Any) -> str:
        reporter = _get_reporter(format, **kwargs)
        return reporter.render(self)

    def write(
        self,
        path: str | None = None,
        *,
        format: str | None = None,
        **kwargs: Any,
    ) -> Any:
        resolved_format = format
        if resolved_format is None and path:
            suffix = Path(path).suffix.lower()
            resolved_format = {
                '.json': 'json',
                '.html': 'html',
                '.htm': 'html',
                '.md': 'markdown',
                '.markdown': 'markdown',
                '.txt': 'console',
            }.get(suffix, 'json')
        reporter = _get_reporter(resolved_format or 'json', **kwargs)
        return reporter.write(self, path)

    def build_docs(self, **kwargs: Any) -> str:
        return _generate_validation_report(self, **kwargs)

    def print(self, format: str = 'console', **kwargs: Any) -> None:
        output = self.render(format=format, **kwargs)
        print(output)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ValidationRunResult:
        return cls(
            run_id=str(data.get('run_id', '')),
            run_time=datetime.fromisoformat(data['run_time']) if data.get('run_time') else datetime.now(),
            suite_name=str(data.get('suite_name', '')),
            source=str(data.get('source', '')),
            row_count=int(data.get('row_count', 0)),
            column_count=int(data.get('column_count', 0)),
            result_format=ResultFormat(str(data.get('result_format', ResultFormat.SUMMARY.value))),
            execution_mode=str(data.get('execution_mode', 'sequential')),
            planned_execution_mode=data.get('planned_execution_mode'),
            checks=tuple(CheckResult.from_dict(check) for check in data.get('checks', [])),
            issues=tuple(_validation_issue_from_dict(issue) for issue in data.get('issues', [])),
            execution_issues=tuple(
                ExecutionIssue.from_dict(issue) for issue in data.get('execution_issues', [])
            ),
            metadata=dict(data.get('metadata', {})),
        )


def _validation_issue_from_dict(data: dict[str, Any]) -> ValidationIssue:
    severity = data.get('severity', Severity.LOW.value)
    return ValidationIssue(
        column=str(data.get('column', '_table_')),
        issue_type=str(data.get('issue_type', 'unknown_issue')),
        count=int(data.get('count', 0)),
        severity=Severity(str(severity).lower()),
        details=data.get('details'),
        expected=data.get('expected'),
        actual=data.get('actual'),
        sample_values=list(data.get('sample_values', [])) or None,
        validator_name=data.get('validator_name'),
        success=bool(data.get('success', False)),
    )
