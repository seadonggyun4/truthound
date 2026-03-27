from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from truthound.core.execution_modes import (
    PlannedExecutionMode,
    RuntimeExecutionMode,
)
from truthound.core.results import ExecutionIssue, ValidationRunResult
from truthound.types import Severity
from truthound.validators.base import ValidationIssue, _validate_safe

if TYPE_CHECKING:
    from truthound.core.contracts import DataAsset
    from truthound.core.planning import ScanPlan

logger = logging.getLogger(__name__)


class ValidationRuntime:
    def execute(self, *, asset: DataAsset, plan: ScanPlan) -> ValidationRunResult:
        validator_instances = [step.check.build_validator() for step in plan.steps]
        execution_issues: list[ExecutionIssue] = []
        actual_execution_mode = RuntimeExecutionMode.SEQUENTIAL.value

        if (
            plan.planned_execution_mode == PlannedExecutionMode.PUSHDOWN.value
            and getattr(asset, 'sql_source', None) is not None
        ):
            from truthound.validators.pushdown_support import PushdownValidationEngine

            issues = PushdownValidationEngine(asset.sql_source).validate(validator_instances)
            actual_execution_mode = RuntimeExecutionMode.PUSHDOWN.value
        elif (
            plan.planned_execution_mode == PlannedExecutionMode.PARALLEL.value
            and len(validator_instances) > 1
        ):
            issues = self._execute_parallel(
                asset=asset,
                validator_instances=validator_instances,
                max_workers=plan.max_workers,
            )
            actual_execution_mode = RuntimeExecutionMode.PARALLEL.value
        else:
            issues, execution_issues, used_threadpool = self._execute_sequential(
                asset=asset,
                validator_instances=validator_instances,
                max_workers=plan.max_workers,
            )
            if used_threadpool:
                actual_execution_mode = RuntimeExecutionMode.THREADPOOL.value

        return ValidationRunResult.from_suite(
            suite=plan.suite,
            issues=issues,
            source=asset.name,
            row_count=asset.row_count,
            column_count=asset.column_count,
            execution_mode=actual_execution_mode,
            planned_execution_mode=plan.planned_execution_mode,
            execution_issues=execution_issues,
            metadata={
                **plan.metadata,
                'duplicate_check_count': plan.duplicate_check_count,
            },
        )

    def _execute_parallel(
        self,
        *,
        asset: DataAsset,
        validator_instances: list[Any],
        max_workers: int | None = None,
    ) -> list[ValidationIssue]:
        lf = asset.to_lazyframe()
        from truthound.validators.optimization.orchestrator import (
            AdaptiveExecutionStrategy,
            ParallelExecutionStrategy,
            ValidatorDAG,
        )

        dag = ValidatorDAG()
        dag.add_validators(validator_instances)
        plan = dag.build_execution_plan()
        if max_workers is not None:
            strategy = ParallelExecutionStrategy(max_workers=max_workers)
        elif len(validator_instances) > 4:
            strategy = ParallelExecutionStrategy(max_workers=4)
        else:
            strategy = AdaptiveExecutionStrategy()
        result = plan.execute(lf, strategy, skip_on_error=True)
        return result.all_issues

    def _execute_sequential(
        self,
        *,
        asset: DataAsset,
        validator_instances: list[Any],
        max_workers: int | None = None,
    ) -> tuple[list[ValidationIssue], list[ExecutionIssue], bool]:
        lf = asset.to_lazyframe()
        all_issues: list[ValidationIssue] = []
        execution_issues: list[ExecutionIssue] = []
        used_threadpool = False

        def run_single(validator: Any) -> tuple[list[ValidationIssue], list[ExecutionIssue]]:
            retries = getattr(getattr(validator, 'config', None), 'max_retries', 0)
            catch = getattr(getattr(validator, 'config', None), 'catch_exceptions', True)
            result = _validate_safe(
                validator,
                lf,
                skip_on_error=catch,
                log_errors=True,
                max_retries=retries,
            )
            issues = list(result.issues)
            exec_issues: list[ExecutionIssue] = []
            if result.has_exception and result.exception_info is not None:
                exec_issues.append(
                    ExecutionIssue(
                        check_name=validator.name,
                        message=result.error_message or result.exception_info.exception_message or 'Validator failed.',
                        exception_type=result.exception_info.exception_type,
                        failure_category=result.exception_info.failure_category,
                        retry_count=result.retry_count,
                    )
                )
                if not issues:
                    issues.append(
                        ValidationIssue(
                            column='*',
                            issue_type='validator_error',
                            count=0,
                            severity=Severity.LOW,
                            details=result.error_message or result.exception_info.exception_message,
                            validator_name=validator.name,
                            exception_info=result.exception_info,
                        )
                    )
            return issues, exec_issues

        if len(validator_instances) < 5:
            for validator in validator_instances:
                issues, exec_issues = run_single(validator)
                all_issues.extend(issues)
                execution_issues.extend(exec_issues)
        else:
            used_threadpool = True
            with ThreadPoolExecutor(max_workers=max_workers or 4) as executor:
                for issues, exec_issues in executor.map(run_single, validator_instances):
                    all_issues.extend(issues)
                    execution_issues.extend(exec_issues)

        return all_issues, execution_issues, used_threadpool
