"""Unique value validators."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import polars as pl

from truthound.types import Severity, ValidationDetail
from truthound.validators.base import ValidationIssue, Validator
from truthound.validators.registry import register_validator

if TYPE_CHECKING:
    from truthound.validators.metrics import MetricKey, SharedMetricStore


@register_validator
class UniqueValidator(Validator):
    """Validates that column values are unique (no duplicates)."""

    name = "unique"
    category = "uniqueness"
    dependencies = {"column_exists"}
    provides = {"uniqueness_checked", "unique"}
    priority = 60

    def get_required_metrics(self, columns: list[str]) -> list[MetricKey]:
        from truthound.validators.metrics import CommonMetrics
        metrics = [CommonMetrics.row_count()[0]]
        for col in columns:
            metrics.append(CommonMetrics.n_unique(col)[0])
            metrics.append(CommonMetrics.non_null_count(col)[0])
        return metrics

    def validate_with_metrics(
        self, lf: pl.LazyFrame, metric_store: SharedMetricStore,
    ) -> list[ValidationIssue]:
        """Validate using pre-computed metrics from the store when available."""
        from truthound.validators.metrics import CommonMetrics

        issues: list[ValidationIssue] = []
        columns = self._get_target_columns(lf)
        if not columns:
            return issues

        # Try to read all needed metrics from the store
        row_count_key = CommonMetrics.row_count()[0]
        total_rows = metric_store.get(row_count_key)

        if total_rows is None:
            # Fallback to standard path
            return self.validate(lf)

        # Check if all column metrics are available
        col_metrics: dict[str, tuple[int, int]] = {}
        for col in columns:
            n_unique_key = CommonMetrics.n_unique(col)[0]
            non_null_key = CommonMetrics.non_null_count(col)[0]
            n_unique = metric_store.get(n_unique_key)
            non_null = metric_store.get(non_null_key)
            if n_unique is None or non_null is None:
                return self.validate(lf)
            col_metrics[col] = (n_unique, non_null)

        if total_rows == 0:
            return issues

        for col in columns:
            unique_count, non_null_count = col_metrics[col]
            duplicate_count = non_null_count - unique_count

            if duplicate_count > 0:
                dup_pct = duplicate_count / non_null_count if non_null_count > 0 else 0
                details = (
                    f"{duplicate_count} duplicate values ({dup_pct:.1%})"
                    if self._should_build_details() else None
                )
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="unique_violation",
                        count=duplicate_count,
                        severity=self._calculate_severity(
                            dup_pct, thresholds=(0.3, 0.1, 0.01)
                        ),
                        details=details,
                        validator_name=self.name,
                        success=False,
                        result=ValidationDetail.from_aggregates(
                            element_count=total_rows,
                            missing_count=total_rows - non_null_count,
                            unexpected_count=duplicate_count,
                            observed_value=f"{duplicate_count} duplicates ({dup_pct:.1%})",
                        ),
                    )
                )

        # Collect sample duplicate values if requested (BASIC+)
        if issues and self._should_collect_samples():
            sample_count = self._get_partial_count()
            for issue in issues:
                try:
                    dup_df = (
                        lf.group_by(issue.column)
                        .agg(pl.len().alias("_count"))
                        .filter(pl.col("_count") > 1)
                        .sort("_count", descending=True)
                        .head(sample_count)
                        .collect()
                    )
                    if len(dup_df) > 0:
                        issue.sample_values = dup_df[issue.column].to_list()
                except Exception:
                    pass

        return issues

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        columns = self._get_target_columns(lf)

        if not columns:
            return issues

        exprs = [
            pl.len().alias("_total"),
            *[pl.col(c).n_unique().alias(f"_unique_{c}") for c in columns],
            *[pl.col(c).count().alias(f"_count_{c}") for c in columns],
        ]
        result = lf.select(exprs).collect()

        total_rows = result["_total"][0]
        if total_rows == 0:
            return issues

        for col in columns:
            unique_count = result[f"_unique_{col}"][0]
            non_null_count = result[f"_count_{col}"][0]
            duplicate_count = non_null_count - unique_count

            if duplicate_count > 0:
                dup_pct = duplicate_count / non_null_count if non_null_count > 0 else 0
                details = (
                    f"{duplicate_count} duplicate values ({dup_pct:.1%})"
                    if self._should_build_details() else None
                )
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="unique_violation",
                        count=duplicate_count,
                        severity=self._calculate_severity(
                            dup_pct, thresholds=(0.3, 0.1, 0.01)
                        ),
                        details=details,
                        validator_name=self.name,
                        success=False,
                        result=ValidationDetail.from_aggregates(
                            element_count=total_rows,
                            missing_count=total_rows - non_null_count,
                            unexpected_count=duplicate_count,
                            observed_value=f"{duplicate_count} duplicates ({dup_pct:.1%})",
                        ),
                    )
                )

        # Collect sample duplicate values if requested (BASIC+)
        if issues and self._should_collect_samples():
            sample_count = self._get_partial_count()
            for issue in issues:
                try:
                    dup_df = (
                        lf.group_by(issue.column)
                        .agg(pl.len().alias("_count"))
                        .filter(pl.col("_count") > 1)
                        .sort("_count", descending=True)
                        .head(sample_count)
                        .collect()
                    )
                    if len(dup_df) > 0:
                        issue.sample_values = dup_df[issue.column].to_list()
                except Exception:
                    pass

        return issues


@register_validator
class UniqueRatioValidator(Validator):
    """Validates that uniqueness ratio is within expected range."""

    name = "unique_ratio"
    category = "uniqueness"
    dependencies = {"column_exists"}
    provides = {"unique_ratio"}
    priority = 60

    def get_required_metrics(self, columns: list[str]) -> list[MetricKey]:
        from truthound.validators.metrics import CommonMetrics
        metrics = [CommonMetrics.row_count()[0]]
        for col in columns:
            metrics.append(CommonMetrics.n_unique(col)[0])
            metrics.append(CommonMetrics.non_null_count(col)[0])
        return metrics

    def __init__(
        self,
        min_ratio: float | None = None,
        max_ratio: float | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        columns = self._get_target_columns(lf)

        if not columns:
            return issues

        exprs = [
            pl.len().alias("_total"),
            *[pl.col(c).n_unique().alias(f"_unique_{c}") for c in columns],
            *[pl.col(c).count().alias(f"_count_{c}") for c in columns],
        ]
        result = lf.select(exprs).collect()

        total_rows = result["_total"][0]
        if total_rows == 0:
            return issues

        build_details = self._should_build_details()

        for col in columns:
            unique_count = result[f"_unique_{col}"][0]
            non_null_count = result[f"_count_{col}"][0]

            if non_null_count == 0:
                continue

            ratio = unique_count / non_null_count

            if self.min_ratio is not None and ratio < self.min_ratio:
                deficit = int((self.min_ratio - ratio) * non_null_count)
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="unique_ratio_low",
                        count=deficit,
                        severity=Severity.MEDIUM,
                        details=(
                            f"Unique ratio {ratio:.1%} < min {self.min_ratio:.1%}"
                            if build_details else None
                        ),
                        expected=f">= {self.min_ratio:.1%}" if build_details else None,
                        actual=f"{ratio:.1%}" if build_details else None,
                        validator_name=self.name,
                        success=False,
                        result=ValidationDetail.from_aggregates(
                            element_count=total_rows,
                            missing_count=total_rows - non_null_count,
                            unexpected_count=deficit,
                            observed_value=f"{ratio:.1%}",
                        ),
                    )
                )

            if self.max_ratio is not None and ratio > self.max_ratio:
                excess = int((ratio - self.max_ratio) * non_null_count)
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="unique_ratio_high",
                        count=excess,
                        severity=Severity.MEDIUM,
                        details=(
                            f"Unique ratio {ratio:.1%} > max {self.max_ratio:.1%}"
                            if build_details else None
                        ),
                        expected=f"<= {self.max_ratio:.1%}" if build_details else None,
                        actual=f"{ratio:.1%}" if build_details else None,
                        validator_name=self.name,
                        success=False,
                        result=ValidationDetail.from_aggregates(
                            element_count=total_rows,
                            missing_count=total_rows - non_null_count,
                            unexpected_count=excess,
                            observed_value=f"{ratio:.1%}",
                        ),
                    )
                )

        return issues


@register_validator
class DistinctCountValidator(Validator):
    """Validates that distinct count is within expected range."""

    name = "distinct_count"
    category = "uniqueness"
    dependencies = {"column_exists"}
    provides = {"distinct_count"}
    priority = 60

    def get_required_metrics(self, columns: list[str]) -> list[MetricKey]:
        from truthound.validators.metrics import CommonMetrics
        metrics = [CommonMetrics.row_count()[0]]
        for col in columns:
            metrics.append(CommonMetrics.n_unique(col)[0])
        return metrics

    def __init__(
        self,
        min_count: int | None = None,
        max_count: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.min_count = min_count
        self.max_count = max_count

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        columns = self._get_target_columns(lf)

        if not columns:
            return issues

        exprs = [
            pl.len().alias("_total"),
            *[pl.col(c).n_unique().alias(f"_unique_{c}") for c in columns],
        ]
        result = lf.select(exprs).collect()
        total_rows = result["_total"][0]

        build_details = self._should_build_details()

        for col in columns:
            unique_count = result[f"_unique_{col}"][0]

            if self.min_count is not None and unique_count < self.min_count:
                deficit = self.min_count - unique_count
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="distinct_count_low",
                        count=deficit,
                        severity=Severity.MEDIUM,
                        details=(
                            f"Distinct count {unique_count} < min {self.min_count}"
                            if build_details else None
                        ),
                        expected=f">= {self.min_count}" if build_details else None,
                        actual=unique_count if build_details else None,
                        validator_name=self.name,
                        success=False,
                        result=ValidationDetail(
                            element_count=total_rows,
                            observed_value=unique_count,
                            unexpected_count=deficit,
                        ),
                    )
                )

            if self.max_count is not None and unique_count > self.max_count:
                excess = unique_count - self.max_count
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="distinct_count_high",
                        count=excess,
                        severity=Severity.MEDIUM,
                        details=(
                            f"Distinct count {unique_count} > max {self.max_count}"
                            if build_details else None
                        ),
                        expected=f"<= {self.max_count}" if build_details else None,
                        actual=unique_count if build_details else None,
                        validator_name=self.name,
                        success=False,
                        result=ValidationDetail(
                            element_count=total_rows,
                            observed_value=unique_count,
                            unexpected_count=excess,
                        ),
                    )
                )

        return issues
