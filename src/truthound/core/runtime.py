from __future__ import annotations

import logging
from copy import copy
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from truthound.core.execution_modes import (
    PlannedExecutionMode,
    RuntimeExecutionMode,
)
from truthound.core.results import CheckResult, ExecutionIssue, ValidationRunResult
from truthound.types import Severity
from truthound.validators.base import ValidationIssue, ValidationResult, _validate_safe

if TYPE_CHECKING:
    from truthound.core.contracts import DataAsset
    from truthound.core.planning import ScanPlan

logger = logging.getLogger(__name__)


@dataclass
class _ExecutionBatch:
    # Input-position identity survives aliases, duplicate names and DAG order.
    checks: list[tuple[list[ValidationIssue], list[ExecutionIssue]]]

    @property
    def issues(self):
        return [issue for issues, _ in self.checks for issue in issues]

    @property
    def execution_issues(self):
        return [issue for _, errors in self.checks for issue in errors]

    def __iter__(self):
        # Preserve the two-value private helper unpacking convention.
        yield self.issues
        yield self.execution_issues


def _outcome(validator, result):
    issues = list(result.issues)
    errors = []
    if result.has_exception and result.exception_info is not None:
        info = result.exception_info
        errors.append(
            ExecutionIssue(
                check_name=validator.name,
                message=result.error_message or info.exception_message or "Validator failed.",
                exception_type=info.exception_type,
                failure_category=info.failure_category,
                retry_count=result.retry_count,
            )
        )
        if not issues:
            issues.append(
                ValidationIssue(
                    column="*",
                    issue_type="validator_error",
                    count=0,
                    severity=Severity.LOW,
                    details=result.error_message or info.exception_message,
                    validator_name=validator.name,
                    exception_info=info,
                )
            )
    elif result.status != ValidationResult.SUCCESS:
        errors.append(
            ExecutionIssue(
                check_name=validator.name,
                message=result.error_message or f"Validator execution {result.status.value}.",
                retry_count=result.retry_count,
            )
        )
    return issues, errors


class ValidationRuntime:
    def execute(self, *, asset: DataAsset, plan: ScanPlan) -> ValidationRunResult:
        validator_instances = []
        for step in plan.steps:
            validator = step.check.build_validator()
            if hasattr(getattr(validator, "config", None), "replace"):
                # Intent owns evidence policy without mutating a factory's
                # retained object or overriding exception/retry settings.
                validator = copy(validator)
                validator.config = validator.config.replace(
                    result_format=plan.suite.evidence_policy.result_format
                )
            validator_instances.append(validator)
        actual_execution_mode = RuntimeExecutionMode.SEQUENTIAL.value

        if (
            plan.planned_execution_mode == PlannedExecutionMode.PUSHDOWN.value
            and getattr(asset, "sql_source", None) is not None
        ):
            from truthound.validators.pushdown_support import PushdownValidationEngine

            outcomes = PushdownValidationEngine(asset.sql_source).validate_outcomes(
                validator_instances
            )
            batch = _ExecutionBatch(
                [
                    _outcome(validator, result)
                    for validator, result in zip(validator_instances, outcomes, strict=True)
                ]
            )
            actual_execution_mode = RuntimeExecutionMode.PUSHDOWN.value
        elif (
            plan.planned_execution_mode == PlannedExecutionMode.PARALLEL.value
            and len(validator_instances) > 1
        ):
            batch = self._execute_parallel(
                asset=asset,
                validator_instances=validator_instances,
                max_workers=plan.max_workers,
            )
            actual_execution_mode = RuntimeExecutionMode.PARALLEL.value
        else:
            batch = self._execute_sequential(
                asset=asset,
                validator_instances=validator_instances,
            )

        if len(batch.checks) != len(plan.steps):
            raise RuntimeError("Missing per-check execution outcome")
        checks, execution_issues = [], []
        for step, (issues, errors) in zip(plan.steps, batch.checks, strict=True):
            spec = step.check
            execution_issues.extend(replace(error, check_name=spec.name) for error in errors)
            checks.append(
                CheckResult(
                    name=spec.name,
                    category=spec.category,
                    success=not issues and not errors,
                    issue_count=len(issues),
                    issues=tuple(issues),
                    metadata={
                        **spec.metadata,
                        "_truthound_check_id": spec.id,
                        "_truthound_execution_incomplete": bool(errors),
                    },
                )
            )

        return ValidationRunResult.from_suite(
            suite=plan.suite,
            issues=batch.issues,
            source=asset.name,
            row_count=asset.row_count,
            column_count=asset.column_count,
            execution_mode=actual_execution_mode,
            planned_execution_mode=plan.planned_execution_mode,
            execution_issues=execution_issues,
            check_results=tuple(checks),
            metadata={
                **plan.metadata,
                "duplicate_check_count": plan.duplicate_check_count,
            },
        )

    def _execute_parallel(
        self,
        *,
        asset: DataAsset,
        validator_instances: list[Any],
        max_workers: int | None = None,
    ) -> _ExecutionBatch:
        lf = asset.to_lazyframe()
        from truthound.validators.optimization.orchestrator import (
            AdaptiveExecutionStrategy,
            ParallelExecutionStrategy,
            ValidatorDAG,
        )

        dag = ValidatorDAG(preserve_instances=True)
        dag.add_validators(validator_instances)
        plan = dag.build_execution_plan()
        if max_workers is not None:
            strategy = ParallelExecutionStrategy(max_workers=max_workers)
        elif len(validator_instances) > 4:
            strategy = ParallelExecutionStrategy(max_workers=4)
        else:
            strategy = AdaptiveExecutionStrategy()
        result = plan.execute(lf, strategy, skip_on_error=True)
        positions = {name: index for index, name in enumerate(dag.nodes)}
        outcomes = [None] * len(validator_instances)
        observed = set()
        for node in result.node_results:
            if node.node_id in observed:
                raise RuntimeError(f"Duplicate execution result: {node.node_id}")
            if node.node_id not in positions:
                raise RuntimeError("Unknown execution result")
            observed.add(node.node_id)
            index = positions[node.node_id]
            outcomes[index] = _outcome(validator_instances[index], node.result)
        for missing in dag.nodes.keys() - observed:
            outcomes[positions[missing]] = (
                [],
                [
                    ExecutionIssue(
                        check_name=dag.nodes[missing].validator.name,
                        message=f"Missing execution result for {missing}.",
                    )
                ],
            )
        if any(outcome is None for outcome in outcomes):
            raise RuntimeError("Missing execution outcome")
        return _ExecutionBatch(outcomes)

    def _execute_sequential(
        self,
        *,
        asset: DataAsset,
        validator_instances: list[Any],
    ) -> _ExecutionBatch:
        lf = asset.to_lazyframe()
        outcomes = []

        def run_single(validator: Any) -> tuple[list[ValidationIssue], list[ExecutionIssue]]:
            retries = getattr(getattr(validator, "config", None), "max_retries", 0)
            catch = getattr(getattr(validator, "config", None), "catch_exceptions", True)
            result = _validate_safe(
                validator,
                lf,
                skip_on_error=catch,
                log_errors=getattr(getattr(validator, "config", None), "log_errors", True),
                max_retries=retries,
            )
            return _outcome(validator, result)

        # Keep the sequential runtime truly sequential. Running multiple
        # validators concurrently against the same Polars lazy frame has caused
        # process aborts in CI under heavier validator mixes.
        for validator in validator_instances:
            outcomes.append(run_single(validator))

        return _ExecutionBatch(outcomes)
