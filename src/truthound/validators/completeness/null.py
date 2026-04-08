"""Null value validators."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl

from truthound.validators.base import (
    ExpressionValidatorMixin,
    ValidationExpressionSpec,
    ValidationIssue,
    Validator,
)
from truthound.validators.pushdown_support import NullCheckPushdownMixin
from truthound.validators.registry import register_validator

if TYPE_CHECKING:
    from truthound.validators.metrics import MetricKey
    from truthound.validators.pushdown_support import PushdownResult


@register_validator
class NullValidator(Validator, ExpressionValidatorMixin, NullCheckPushdownMixin):
    """Detects null/missing values in columns.

    This validator uses the expression-based architecture for optimal
    performance when combined with other validators via ExpressionBatchExecutor.

    Example:
        # Standalone usage (backward compatible)
        validator = NullValidator()
        issues = validator.validate(lf)

        # Batched usage with other validators (single collect)
        from truthound.validators.base import ExpressionBatchExecutor
        executor = ExpressionBatchExecutor()
        executor.add_validator(NullValidator())
        executor.add_validator(RangeValidator(min_value=0))
        all_issues = executor.execute(lf)  # Single collect()!
    """

    name = "null"
    category = "completeness"
    dependencies = {"column_exists"}
    provides = {"null_checked", "null"}
    priority = 50

    def get_required_metrics(self, columns: list[str]) -> list[MetricKey]:
        from truthound.validators.metrics import CommonMetrics
        metrics = [CommonMetrics.row_count()[0]]
        for col in columns:
            metrics.append(CommonMetrics.null_count(col)[0])
        return metrics

    def get_validation_exprs(
        self,
        lf: pl.LazyFrame,
        columns: list[str],
    ) -> list[ValidationExpressionSpec]:
        """Get validation expressions for null checking.

        Args:
            lf: LazyFrame to validate
            columns: Columns to check for nulls

        Returns:
            List of ValidationExpressionSpec for batched execution
        """
        return [self._build_null_spec(col) for col in columns]

    def _build_null_spec(self, column: str) -> ValidationExpressionSpec:
        return ValidationExpressionSpec(
            column=column,
            validator_name=self.name,
            issue_type="null",
            count_expr=pl.col(column).null_count(),
            non_null_expr=pl.len(),
            details_template="{ratio:.1%} of values are null",
            filter_expr=pl.col(column).is_null(),
        )

    def process_pushdown_results(
        self,
        results: list[PushdownResult],
    ) -> list[ValidationIssue]:
        prefix_map = {
            f"_v{index}": self._build_null_spec(result.column)
            for index, result in enumerate(results)
        }
        result_map = {
            prefix: {
                "count": int(result.value or 0),
                "non_null": int(result.total_rows),
            }
            for prefix, result in zip(prefix_map, results, strict=False)
        }
        total_rows = max((int(result.total_rows) for result in results), default=0)
        return self.build_issues_from_results(
            specs=list(prefix_map.values()),
            results=result_map,
            total_rows=total_rows,
            prefix_map=prefix_map,
        )

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate using expression-based approach.

        This method uses the expression mixin for single-collect execution.
        """
        return self._validate_with_expressions(lf)


@register_validator
class NotNullValidator(Validator, ExpressionValidatorMixin, NullCheckPushdownMixin):
    """Validates that columns have no null values.

    Uses expression-based architecture for batched execution.
    """

    name = "not_null"
    category = "completeness"
    dependencies = {"column_exists"}
    provides = {"not_null_checked", "not_null"}
    priority = 50

    def get_required_metrics(self, columns: list[str]) -> list[MetricKey]:
        from truthound.validators.metrics import CommonMetrics
        metrics = [CommonMetrics.row_count()[0]]
        for col in columns:
            metrics.append(CommonMetrics.null_count(col)[0])
        return metrics

    def get_validation_exprs(
        self,
        lf: pl.LazyFrame,
        columns: list[str],
    ) -> list[ValidationExpressionSpec]:
        """Get validation expressions for not-null checking."""
        return [self._build_not_null_spec(col) for col in columns]

    def _build_not_null_spec(self, column: str) -> ValidationExpressionSpec:
        return ValidationExpressionSpec(
            column=column,
            validator_name=self.name,
            issue_type="not_null_violation",
            count_expr=pl.col(column).null_count(),
            non_null_expr=pl.len(),
            severity_ratio_thresholds=(0.01, 0.001, 0.0001),
            details_template="Expected no nulls, found {count} ({ratio:.1%})",
            expected=0,
            filter_expr=pl.col(column).is_null(),
        )

    def process_pushdown_results(
        self,
        results: list[PushdownResult],
    ) -> list[ValidationIssue]:
        prefix_map = {
            f"_v{index}": self._build_not_null_spec(result.column)
            for index, result in enumerate(results)
        }
        result_map = {
            prefix: {
                "count": int(result.value or 0),
                "non_null": int(result.total_rows),
            }
            for prefix, result in zip(prefix_map, results, strict=False)
        }
        total_rows = max((int(result.total_rows) for result in results), default=0)
        return self.build_issues_from_results(
            specs=list(prefix_map.values()),
            results=result_map,
            total_rows=total_rows,
            prefix_map=prefix_map,
        )

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate using expression-based approach."""
        return self._validate_with_expressions(lf)


@register_validator
class CompletenessRatioValidator(Validator, ExpressionValidatorMixin):
    """Validates that columns meet a minimum completeness ratio.

    Uses expression-based architecture for batched execution.
    """

    name = "completeness_ratio"
    category = "completeness"
    dependencies = {"column_exists"}
    provides = {"completeness_ratio"}
    priority = 50

    def get_required_metrics(self, columns: list[str]) -> list[MetricKey]:
        from truthound.validators.metrics import CommonMetrics
        metrics = [CommonMetrics.row_count()[0]]
        for col in columns:
            metrics.append(CommonMetrics.null_count(col)[0])
            metrics.append(CommonMetrics.non_null_count(col)[0])
        return metrics

    def __init__(
        self,
        min_ratio: float = 0.95,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.min_ratio = min_ratio

    def get_validation_exprs(
        self,
        lf: pl.LazyFrame,
        columns: list[str],
    ) -> list[ValidationExpressionSpec]:
        """Get validation expressions for completeness ratio checking.

        Note: This validator has custom logic that requires the min_ratio threshold,
        so we use a custom build_issues_from_results implementation.
        """
        specs = []
        for col in columns:
            specs.append(
                ValidationExpressionSpec(
                    column=col,
                    validator_name=self.name,
                    issue_type="completeness_ratio",
                    count_expr=pl.col(col).null_count(),  # We'll use this for null count
                    non_null_expr=pl.len(),  # Total rows
                    severity_ratio_thresholds=(0.5, 0.2, 0.05),
                    details_template=f"Completeness {{ratio:.1%}} < {self.min_ratio:.1%}",
                    expected=self.min_ratio,
                    filter_expr=pl.col(col).is_null(),
                )
            )
        return specs

    def build_issues_from_results(
        self,
        specs: list[ValidationExpressionSpec],
        results: dict[str, dict[str, Any]],
        total_rows: int,
        prefix_map: dict[str, ValidationExpressionSpec],
    ) -> list[ValidationIssue]:
        """Custom implementation for completeness ratio validation."""
        issues: list[ValidationIssue] = []

        for prefix, spec in prefix_map.items():
            spec_results = results[prefix]
            null_count = spec_results.get("count", 0)
            row_count = spec_results.get("non_null", total_rows)

            if row_count == 0:
                continue

            non_null_count = row_count - null_count
            ratio = non_null_count / row_count

            # Only report if below threshold
            if ratio >= self.min_ratio:
                continue

            # Calculate severity based on how far below threshold
            fail_ratio = 1 - ratio
            severity = self._calculate_severity_from_ratio(
                fail_ratio, spec.severity_ratio_thresholds
            )

            issues.append(
                ValidationIssue(
                    column=spec.column,
                    issue_type=spec.issue_type,
                    count=null_count,
                    severity=severity,
                    details=f"Completeness {ratio:.1%} < {self.min_ratio:.1%}",
                    expected=self.min_ratio,
                    actual=ratio,
                )
            )

        return issues

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Validate using expression-based approach."""
        return self._validate_with_expressions(lf)
