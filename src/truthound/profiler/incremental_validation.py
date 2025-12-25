"""Incremental profiling validation framework.

This module provides comprehensive validation for incremental profiling:
- Change detection accuracy validation
- Fingerprint consistency validation
- Profile merge correctness validation
- Performance regression validation
- Data integrity validation

The framework is designed for high extensibility and maintainability:
- Protocol-based validators for easy extension
- Registry pattern for validator discovery
- Configurable validation strategies
- Detailed validation results with recommendations

Example:
    from truthound.profiler.incremental_validation import (
        IncrementalValidator,
        ValidationRunner,
        ValidationConfig,
    )

    # Create validator
    validator = IncrementalValidator()

    # Run validation
    result = validator.validate(
        original_profile=profile1,
        incremental_profile=profile2,
        data=df,
    )

    # Check results
    if result.passed:
        print("Validation passed!")
    else:
        for issue in result.issues:
            print(f"{issue.severity}: {issue.message}")
"""

from __future__ import annotations

import hashlib
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Protocol, Sequence, TypeVar

import polars as pl

from truthound.profiler.base import (
    ColumnProfile,
    TableProfile,
    ProfilerConfig,
)
from truthound.profiler.incremental import (
    ChangeReason,
    ColumnFingerprint,
    ChangeDetectionResult,
    FingerprintCalculator,
    IncrementalConfig,
    IncrementalProfiler,
    ProfileMerger,
)


# Set up logging
logger = logging.getLogger("truthound.profiler.incremental_validation")


# =============================================================================
# Validation Types
# =============================================================================


class ValidationSeverity(str, Enum):
    """Severity levels for validation issues."""

    INFO = "info"          # Informational message
    WARNING = "warning"    # Potential issue
    ERROR = "error"        # Definite problem
    CRITICAL = "critical"  # Validation cannot proceed


class ValidationCategory(str, Enum):
    """Categories of validation checks."""

    CHANGE_DETECTION = "change_detection"
    FINGERPRINT = "fingerprint"
    PROFILE_MERGE = "profile_merge"
    DATA_INTEGRITY = "data_integrity"
    PERFORMANCE = "performance"
    CONSISTENCY = "consistency"
    SCHEMA = "schema"


class ValidationType(str, Enum):
    """Types of validation operations."""

    FULL = "full"              # Complete validation
    QUICK = "quick"            # Fast essential checks
    CHANGE_ONLY = "change_only"  # Only change detection
    MERGE_ONLY = "merge_only"    # Only merge validation


# =============================================================================
# Validation Results
# =============================================================================


@dataclass
class ValidationIssue:
    """A single validation issue found during checks.

    Attributes:
        category: Category of the issue
        severity: How severe the issue is
        message: Human-readable description
        column_name: Affected column (if applicable)
        expected: Expected value
        actual: Actual value found
        recommendation: Suggested fix
        metadata: Additional context
    """

    category: ValidationCategory
    severity: ValidationSeverity
    message: str
    column_name: str | None = None
    expected: Any = None
    actual: Any = None
    recommendation: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "column_name": self.column_name,
            "expected": str(self.expected) if self.expected is not None else None,
            "actual": str(self.actual) if self.actual is not None else None,
            "recommendation": self.recommendation,
            "metadata": self.metadata,
        }


@dataclass
class ValidationMetrics:
    """Metrics from validation run.

    Attributes:
        total_checks: Total number of checks performed
        passed_checks: Number of checks that passed
        failed_checks: Number of checks that failed
        skipped_checks: Number of checks skipped
        duration_ms: Total validation time
        columns_validated: Number of columns validated
        changes_detected: Number of changes detected
        false_positives: Estimated false positive count
        false_negatives: Estimated false negative count
    """

    total_checks: int = 0
    passed_checks: int = 0
    failed_checks: int = 0
    skipped_checks: int = 0
    duration_ms: float = 0.0
    columns_validated: int = 0
    changes_detected: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def pass_rate(self) -> float:
        """Calculate pass rate."""
        if self.total_checks == 0:
            return 1.0
        return self.passed_checks / self.total_checks

    @property
    def accuracy(self) -> float:
        """Calculate change detection accuracy."""
        total = self.columns_validated
        if total == 0:
            return 1.0
        errors = self.false_positives + self.false_negatives
        return (total - errors) / total

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_checks": self.total_checks,
            "passed_checks": self.passed_checks,
            "failed_checks": self.failed_checks,
            "skipped_checks": self.skipped_checks,
            "pass_rate": self.pass_rate,
            "duration_ms": self.duration_ms,
            "columns_validated": self.columns_validated,
            "changes_detected": self.changes_detected,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "accuracy": self.accuracy,
        }


@dataclass
class ValidationResult:
    """Complete result of a validation run.

    Attributes:
        passed: Whether validation passed overall
        validation_type: Type of validation performed
        issues: List of issues found
        metrics: Validation metrics
        validated_at: When validation was performed
        config: Configuration used
        details: Additional details per category
    """

    passed: bool
    validation_type: ValidationType
    issues: list[ValidationIssue] = field(default_factory=list)
    metrics: ValidationMetrics = field(default_factory=ValidationMetrics)
    validated_at: datetime = field(default_factory=datetime.now)
    config: dict[str, Any] = field(default_factory=dict)
    details: dict[ValidationCategory, dict[str, Any]] = field(default_factory=dict)

    @property
    def error_count(self) -> int:
        """Count of error-level issues."""
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.ERROR)

    @property
    def warning_count(self) -> int:
        """Count of warning-level issues."""
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.WARNING)

    @property
    def critical_count(self) -> int:
        """Count of critical issues."""
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.CRITICAL)

    def get_issues_by_category(
        self,
        category: ValidationCategory,
    ) -> list[ValidationIssue]:
        """Get issues for a specific category."""
        return [i for i in self.issues if i.category == category]

    def get_issues_by_severity(
        self,
        severity: ValidationSeverity,
    ) -> list[ValidationIssue]:
        """Get issues for a specific severity."""
        return [i for i in self.issues if i.severity == severity]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "passed": self.passed,
            "validation_type": self.validation_type.value,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "critical_count": self.critical_count,
            "issues": [i.to_dict() for i in self.issues],
            "metrics": self.metrics.to_dict(),
            "validated_at": self.validated_at.isoformat(),
            "config": self.config,
            "details": {k.value: v for k, v in self.details.items()},
        }

    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            "# Incremental Profiling Validation Report",
            "",
            f"**Status**: {'✅ PASSED' if self.passed else '❌ FAILED'}",
            f"**Validation Type**: {self.validation_type.value}",
            f"**Validated At**: {self.validated_at.isoformat()}",
            "",
            "## Summary",
            "",
            f"- Total Checks: {self.metrics.total_checks}",
            f"- Passed: {self.metrics.passed_checks}",
            f"- Failed: {self.metrics.failed_checks}",
            f"- Pass Rate: {self.metrics.pass_rate:.1%}",
            f"- Duration: {self.metrics.duration_ms:.1f}ms",
            "",
        ]

        if self.issues:
            lines.extend([
                "## Issues Found",
                "",
            ])

            for severity in [
                ValidationSeverity.CRITICAL,
                ValidationSeverity.ERROR,
                ValidationSeverity.WARNING,
                ValidationSeverity.INFO,
            ]:
                issues = self.get_issues_by_severity(severity)
                if issues:
                    lines.append(f"### {severity.value.title()} ({len(issues)})")
                    lines.append("")
                    for issue in issues:
                        col_info = f" [{issue.column_name}]" if issue.column_name else ""
                        lines.append(f"- **{issue.category.value}**{col_info}: {issue.message}")
                        if issue.recommendation:
                            lines.append(f"  - *Recommendation*: {issue.recommendation}")
                    lines.append("")

        return "\n".join(lines)


# =============================================================================
# Validation Protocol and Base Classes
# =============================================================================


class ValidatorProtocol(Protocol):
    """Protocol for validators."""

    @property
    def name(self) -> str:
        """Validator name."""
        ...

    @property
    def category(self) -> ValidationCategory:
        """Validation category."""
        ...

    def validate(
        self,
        context: "ValidationContext",
    ) -> list[ValidationIssue]:
        """Perform validation.

        Args:
            context: Validation context with data and profiles

        Returns:
            List of issues found (empty if validation passes)
        """
        ...


@dataclass
class ValidationContext:
    """Context for validation operations.

    Contains all data needed for validation checks.
    """

    # Data
    data: pl.LazyFrame | pl.DataFrame

    # Profiles
    original_profile: TableProfile | None = None
    incremental_profile: TableProfile | None = None
    full_profile: TableProfile | None = None  # For comparison

    # Fingerprints
    original_fingerprints: dict[str, ColumnFingerprint] = field(default_factory=dict)
    current_fingerprints: dict[str, ColumnFingerprint] = field(default_factory=dict)

    # Change detection results
    change_results: dict[str, ChangeDetectionResult] = field(default_factory=dict)

    # Profiling metadata
    profiled_columns: set[str] = field(default_factory=set)
    skipped_columns: set[str] = field(default_factory=set)
    change_reasons: dict[str, ChangeReason] = field(default_factory=dict)

    # Configuration
    config: "ValidationConfig | None" = None


class BaseValidator(ABC):
    """Base class for validators.

    Provides common functionality for all validators.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._issues: list[ValidationIssue] = []

    @property
    @abstractmethod
    def name(self) -> str:
        """Validator name."""
        pass

    @property
    @abstractmethod
    def category(self) -> ValidationCategory:
        """Validation category."""
        pass

    @abstractmethod
    def validate(self, context: ValidationContext) -> list[ValidationIssue]:
        """Perform validation."""
        pass

    def add_issue(
        self,
        severity: ValidationSeverity,
        message: str,
        column_name: str | None = None,
        expected: Any = None,
        actual: Any = None,
        recommendation: str | None = None,
        **metadata: Any,
    ) -> None:
        """Add a validation issue."""
        self._issues.append(ValidationIssue(
            category=self.category,
            severity=severity,
            message=message,
            column_name=column_name,
            expected=expected,
            actual=actual,
            recommendation=recommendation,
            metadata=metadata,
        ))

    def reset(self) -> None:
        """Reset validator state."""
        self._issues = []


# =============================================================================
# Change Detection Validators
# =============================================================================


class ChangeDetectionAccuracyValidator(BaseValidator):
    """Validates accuracy of change detection.

    Compares incremental change detection against full re-profiling
    to identify false positives and false negatives.
    """

    @property
    def name(self) -> str:
        return "change_detection_accuracy"

    @property
    def category(self) -> ValidationCategory:
        return ValidationCategory.CHANGE_DETECTION

    def __init__(
        self,
        tolerance: float = 0.01,
        enabled: bool = True,
    ):
        """Initialize validator.

        Args:
            tolerance: Tolerance for numerical comparisons
            enabled: Whether validator is enabled
        """
        super().__init__(enabled)
        self.tolerance = tolerance

    def validate(self, context: ValidationContext) -> list[ValidationIssue]:
        """Validate change detection accuracy."""
        self.reset()

        if context.incremental_profile is None:
            self.add_issue(
                ValidationSeverity.ERROR,
                "No incremental profile provided for validation",
                recommendation="Provide an incremental profile to validate",
            )
            return self._issues

        if context.full_profile is None:
            self.add_issue(
                ValidationSeverity.WARNING,
                "No full profile for comparison; accuracy cannot be fully validated",
                recommendation="Provide a full profile for accurate comparison",
            )
            return self._issues

        inc_columns = {col.name: col for col in context.incremental_profile.columns}
        full_columns = {col.name: col for col in context.full_profile.columns}

        # Check for missing columns
        for col_name in full_columns:
            if col_name not in inc_columns:
                self.add_issue(
                    ValidationSeverity.ERROR,
                    f"Column missing from incremental profile",
                    column_name=col_name,
                    recommendation="Check if column was incorrectly skipped",
                )

        # Check for extra columns
        for col_name in inc_columns:
            if col_name not in full_columns:
                self.add_issue(
                    ValidationSeverity.WARNING,
                    f"Column in incremental but not in full profile",
                    column_name=col_name,
                )

        # Compare column profiles
        for col_name, inc_col in inc_columns.items():
            if col_name not in full_columns:
                continue

            full_col = full_columns[col_name]
            self._compare_columns(col_name, inc_col, full_col, context)

        return self._issues

    def _compare_columns(
        self,
        col_name: str,
        inc_col: ColumnProfile,
        full_col: ColumnProfile,
        context: ValidationContext,
    ) -> None:
        """Compare incremental and full profile for a column."""
        was_skipped = col_name in context.skipped_columns
        was_profiled = col_name in context.profiled_columns

        # If column was skipped, it should match the original
        if was_skipped and context.original_profile:
            orig_columns = {c.name: c for c in context.original_profile.columns}
            if col_name in orig_columns:
                # Compare with original - should be identical
                orig_col = orig_columns[col_name]
                if inc_col.row_count != orig_col.row_count:
                    self.add_issue(
                        ValidationSeverity.ERROR,
                        "Skipped column has different row count from original",
                        column_name=col_name,
                        expected=orig_col.row_count,
                        actual=inc_col.row_count,
                        recommendation="Column may have changed but was not detected",
                    )

        # Compare with full profile for accuracy
        issues_found = []

        # Row count
        if inc_col.row_count != full_col.row_count:
            issues_found.append(("row_count", inc_col.row_count, full_col.row_count))

        # Null count
        if inc_col.null_count != full_col.null_count:
            issues_found.append(("null_count", inc_col.null_count, full_col.null_count))

        # Null ratio (with tolerance)
        if abs(inc_col.null_ratio - full_col.null_ratio) > self.tolerance:
            issues_found.append(("null_ratio", inc_col.null_ratio, full_col.null_ratio))

        # Unique count
        if inc_col.distinct_count != full_col.distinct_count:
            issues_found.append(("distinct_count", inc_col.distinct_count, full_col.distinct_count))

        if issues_found and was_skipped:
            # False negative - change was not detected
            for field_name, inc_val, full_val in issues_found:
                self.add_issue(
                    ValidationSeverity.ERROR,
                    f"Change in {field_name} not detected (false negative)",
                    column_name=col_name,
                    expected=full_val,
                    actual=inc_val,
                    recommendation="Increase change detection sensitivity",
                    false_negative=True,
                )
        elif not issues_found and was_profiled:
            # Check if it was a false positive
            if col_name in context.change_reasons:
                reason = context.change_reasons[col_name]
                if reason not in [ChangeReason.NEW_COLUMN, ChangeReason.FORCED]:
                    # Might be a false positive if data hasn't changed
                    # This is informational since re-profiling is safe
                    self.add_issue(
                        ValidationSeverity.INFO,
                        f"Column was re-profiled but no differences found (potential false positive)",
                        column_name=col_name,
                        metadata={"reason": reason.value},
                        false_positive=True,
                    )


class SchemaChangeValidator(BaseValidator):
    """Validates schema change detection."""

    @property
    def name(self) -> str:
        return "schema_change"

    @property
    def category(self) -> ValidationCategory:
        return ValidationCategory.SCHEMA

    def validate(self, context: ValidationContext) -> list[ValidationIssue]:
        """Validate schema change detection."""
        self.reset()

        if not isinstance(context.data, (pl.LazyFrame, pl.DataFrame)):
            self.add_issue(
                ValidationSeverity.ERROR,
                "Invalid data type for validation",
            )
            return self._issues

        lf = context.data.lazy() if isinstance(context.data, pl.DataFrame) else context.data
        schema = lf.collect_schema()

        if context.original_profile:
            orig_columns = {c.name: c for c in context.original_profile.columns}

            for col_name, dtype in schema.items():
                if col_name in orig_columns:
                    orig_type = orig_columns[col_name].physical_type
                    current_type = str(dtype)

                    if orig_type != current_type:
                        # Schema changed
                        was_detected = (
                            col_name in context.change_reasons and
                            context.change_reasons[col_name] == ChangeReason.SCHEMA_CHANGED
                        )

                        if not was_detected:
                            self.add_issue(
                                ValidationSeverity.ERROR,
                                "Schema change not detected",
                                column_name=col_name,
                                expected=orig_type,
                                actual=current_type,
                                recommendation="Enable schema change detection",
                            )
                        else:
                            self.add_issue(
                                ValidationSeverity.INFO,
                                "Schema change correctly detected",
                                column_name=col_name,
                                expected=orig_type,
                                actual=current_type,
                            )

        return self._issues


class StalenessValidator(BaseValidator):
    """Validates staleness detection."""

    @property
    def name(self) -> str:
        return "staleness"

    @property
    def category(self) -> ValidationCategory:
        return ValidationCategory.CHANGE_DETECTION

    def __init__(
        self,
        max_age: timedelta | None = None,
        enabled: bool = True,
    ):
        super().__init__(enabled)
        self.max_age = max_age

    def validate(self, context: ValidationContext) -> list[ValidationIssue]:
        """Validate staleness detection."""
        self.reset()

        if context.original_profile is None:
            return self._issues

        config_max_age = (
            context.config.max_profile_age
            if context.config else self.max_age
        )

        if config_max_age is None:
            return self._issues

        for col in context.original_profile.columns:
            age = datetime.now() - col.profiled_at
            is_stale = age > config_max_age

            was_detected = (
                col.name in context.change_reasons and
                context.change_reasons[col.name] == ChangeReason.STALE
            )

            if is_stale and not was_detected:
                self.add_issue(
                    ValidationSeverity.WARNING,
                    "Stale column not re-profiled",
                    column_name=col.name,
                    expected=f"Re-profile after {config_max_age}",
                    actual=f"Age: {age}",
                    recommendation="Check staleness configuration",
                )

        return self._issues


# =============================================================================
# Fingerprint Validators
# =============================================================================


class FingerprintConsistencyValidator(BaseValidator):
    """Validates fingerprint consistency and correctness."""

    @property
    def name(self) -> str:
        return "fingerprint_consistency"

    @property
    def category(self) -> ValidationCategory:
        return ValidationCategory.FINGERPRINT

    def validate(self, context: ValidationContext) -> list[ValidationIssue]:
        """Validate fingerprint consistency."""
        self.reset()

        if not context.current_fingerprints:
            self.add_issue(
                ValidationSeverity.INFO,
                "No fingerprints to validate",
            )
            return self._issues

        # Check fingerprint stability (same data should produce same fingerprint)
        calculator = FingerprintCalculator()

        lf = (
            context.data.lazy()
            if isinstance(context.data, pl.DataFrame)
            else context.data
        )

        for col_name, fp in context.current_fingerprints.items():
            # Recalculate fingerprint
            try:
                recalc_fp = calculator.calculate(lf, col_name)

                if recalc_fp.sample_hash != fp.sample_hash:
                    self.add_issue(
                        ValidationSeverity.ERROR,
                        "Fingerprint not stable (different hashes for same data)",
                        column_name=col_name,
                        expected=fp.sample_hash,
                        actual=recalc_fp.sample_hash,
                        recommendation="Check fingerprint calculation determinism",
                    )

                if recalc_fp.row_count != fp.row_count:
                    self.add_issue(
                        ValidationSeverity.ERROR,
                        "Fingerprint row count mismatch",
                        column_name=col_name,
                        expected=fp.row_count,
                        actual=recalc_fp.row_count,
                    )

            except Exception as e:
                self.add_issue(
                    ValidationSeverity.ERROR,
                    f"Failed to calculate fingerprint: {e}",
                    column_name=col_name,
                )

        return self._issues


class FingerprintSensitivityValidator(BaseValidator):
    """Validates fingerprint sensitivity to changes."""

    @property
    def name(self) -> str:
        return "fingerprint_sensitivity"

    @property
    def category(self) -> ValidationCategory:
        return ValidationCategory.FINGERPRINT

    def __init__(
        self,
        min_change_detection_rate: float = 0.95,
        enabled: bool = True,
    ):
        """Initialize validator.

        Args:
            min_change_detection_rate: Minimum rate for detecting actual changes
            enabled: Whether validator is enabled
        """
        super().__init__(enabled)
        self.min_change_detection_rate = min_change_detection_rate

    def validate(self, context: ValidationContext) -> list[ValidationIssue]:
        """Validate fingerprint sensitivity."""
        self.reset()

        if not context.original_fingerprints or not context.current_fingerprints:
            self.add_issue(
                ValidationSeverity.INFO,
                "Need both original and current fingerprints for sensitivity validation",
            )
            return self._issues

        changes_detected = 0
        actual_changes = 0

        for col_name in context.current_fingerprints:
            if col_name not in context.original_fingerprints:
                continue

            orig_fp = context.original_fingerprints[col_name]
            curr_fp = context.current_fingerprints[col_name]

            # Check if data actually changed
            data_changed = (
                orig_fp.row_count != curr_fp.row_count or
                orig_fp.null_count != curr_fp.null_count or
                orig_fp.sample_hash != curr_fp.sample_hash
            )

            if data_changed:
                actual_changes += 1

                # Check if change was detected
                was_detected = col_name in context.profiled_columns

                if was_detected:
                    changes_detected += 1
                else:
                    self.add_issue(
                        ValidationSeverity.WARNING,
                        "Data change not detected by fingerprint",
                        column_name=col_name,
                        metadata={
                            "orig_hash": orig_fp.sample_hash,
                            "curr_hash": curr_fp.sample_hash,
                        },
                    )

        if actual_changes > 0:
            detection_rate = changes_detected / actual_changes
            if detection_rate < self.min_change_detection_rate:
                self.add_issue(
                    ValidationSeverity.ERROR,
                    f"Change detection rate below threshold",
                    expected=f">= {self.min_change_detection_rate:.0%}",
                    actual=f"{detection_rate:.0%}",
                    recommendation="Adjust fingerprint sensitivity settings",
                )

        return self._issues


# =============================================================================
# Profile Merge Validators
# =============================================================================


class ProfileMergeValidator(BaseValidator):
    """Validates profile merge correctness."""

    @property
    def name(self) -> str:
        return "profile_merge"

    @property
    def category(self) -> ValidationCategory:
        return ValidationCategory.PROFILE_MERGE

    def validate(self, context: ValidationContext) -> list[ValidationIssue]:
        """Validate profile merge operations."""
        self.reset()

        # This validator needs merge test data
        if not hasattr(context, 'merge_inputs') or not hasattr(context, 'merge_output'):
            return self._issues

        merge_inputs = getattr(context, 'merge_inputs', [])
        merge_output = getattr(context, 'merge_output', None)

        if not merge_inputs or merge_output is None:
            return self._issues

        self._validate_column_preservation(merge_inputs, merge_output)
        self._validate_row_count(merge_inputs, merge_output)
        self._validate_latest_wins(merge_inputs, merge_output)

        return self._issues

    def _validate_column_preservation(
        self,
        inputs: list[TableProfile],
        output: TableProfile,
    ) -> None:
        """Check all input columns appear in output."""
        all_input_columns = set()
        for profile in inputs:
            all_input_columns.update(c.name for c in profile.columns)

        output_columns = {c.name for c in output.columns}

        missing = all_input_columns - output_columns
        if missing:
            self.add_issue(
                ValidationSeverity.ERROR,
                f"Columns lost during merge: {missing}",
                recommendation="Check merge logic for column preservation",
            )

    def _validate_row_count(
        self,
        inputs: list[TableProfile],
        output: TableProfile,
    ) -> None:
        """Validate row count after merge."""
        expected_rows = sum(p.row_count for p in inputs)

        if output.row_count != expected_rows:
            self.add_issue(
                ValidationSeverity.WARNING,
                "Merged row count doesn't match sum of inputs",
                expected=expected_rows,
                actual=output.row_count,
                recommendation="This may be expected if merging overlapping data",
            )

    def _validate_latest_wins(
        self,
        inputs: list[TableProfile],
        output: TableProfile,
    ) -> None:
        """Validate that latest profile wins for duplicates."""
        # Sort inputs by profiled_at
        sorted_inputs = sorted(inputs, key=lambda p: p.profiled_at)

        # Build expected output
        expected_columns = {}
        for profile in sorted_inputs:
            for col in profile.columns:
                expected_columns[col.name] = col

        # Compare with actual output
        output_columns = {c.name: c for c in output.columns}

        for col_name, expected_col in expected_columns.items():
            if col_name not in output_columns:
                continue

            actual_col = output_columns[col_name]

            # The profile should match the latest input
            if actual_col.profiled_at != expected_col.profiled_at:
                self.add_issue(
                    ValidationSeverity.WARNING,
                    "Merged column doesn't use latest profile",
                    column_name=col_name,
                    expected=expected_col.profiled_at.isoformat(),
                    actual=actual_col.profiled_at.isoformat(),
                )


# =============================================================================
# Data Integrity Validators
# =============================================================================


class DataIntegrityValidator(BaseValidator):
    """Validates data integrity in profiles."""

    @property
    def name(self) -> str:
        return "data_integrity"

    @property
    def category(self) -> ValidationCategory:
        return ValidationCategory.DATA_INTEGRITY

    def validate(self, context: ValidationContext) -> list[ValidationIssue]:
        """Validate data integrity."""
        self.reset()

        profile = context.incremental_profile or context.full_profile
        if profile is None:
            self.add_issue(
                ValidationSeverity.ERROR,
                "No profile to validate",
            )
            return self._issues

        lf = (
            context.data.lazy()
            if isinstance(context.data, pl.DataFrame)
            else context.data
        )

        actual_row_count = lf.select(pl.len()).collect().item()

        # Row count check
        if profile.row_count != actual_row_count:
            self.add_issue(
                ValidationSeverity.ERROR,
                "Profile row count doesn't match data",
                expected=actual_row_count,
                actual=profile.row_count,
                recommendation="Profile may be stale",
            )

        # Column count check
        schema = lf.collect_schema()
        if profile.column_count != len(schema):
            self.add_issue(
                ValidationSeverity.ERROR,
                "Profile column count doesn't match data",
                expected=len(schema),
                actual=profile.column_count,
            )

        # Check each column
        for col in profile.columns:
            if col.name not in schema:
                self.add_issue(
                    ValidationSeverity.ERROR,
                    "Profile column not found in data",
                    column_name=col.name,
                    recommendation="Schema may have changed",
                )
                continue

            # Verify null count
            actual_nulls = lf.select(pl.col(col.name).null_count()).collect().item()
            if col.null_count != actual_nulls:
                self.add_issue(
                    ValidationSeverity.ERROR,
                    "Null count mismatch",
                    column_name=col.name,
                    expected=actual_nulls,
                    actual=col.null_count,
                )

        return self._issues


# =============================================================================
# Performance Validators
# =============================================================================


class PerformanceValidator(BaseValidator):
    """Validates performance improvements from incremental profiling."""

    @property
    def name(self) -> str:
        return "performance"

    @property
    def category(self) -> ValidationCategory:
        return ValidationCategory.PERFORMANCE

    def __init__(
        self,
        min_speedup: float = 1.0,
        max_overhead: float = 0.2,
        enabled: bool = True,
    ):
        """Initialize validator.

        Args:
            min_speedup: Minimum expected speedup ratio
            max_overhead: Maximum acceptable overhead ratio
            enabled: Whether validator is enabled
        """
        super().__init__(enabled)
        self.min_speedup = min_speedup
        self.max_overhead = max_overhead

    def validate(self, context: ValidationContext) -> list[ValidationIssue]:
        """Validate performance."""
        self.reset()

        inc_profile = context.incremental_profile
        full_profile = context.full_profile

        if inc_profile is None or full_profile is None:
            return self._issues

        inc_duration = inc_profile.profile_duration_ms
        full_duration = full_profile.profile_duration_ms

        if full_duration == 0:
            return self._issues

        speedup = full_duration / inc_duration if inc_duration > 0 else float('inf')
        columns_skipped = len(context.skipped_columns)
        columns_total = inc_profile.column_count

        if columns_skipped > 0:
            # Should see improvement if columns were skipped
            if speedup < self.min_speedup:
                self.add_issue(
                    ValidationSeverity.WARNING,
                    f"Incremental profiling slower than expected",
                    expected=f">= {self.min_speedup:.1f}x speedup",
                    actual=f"{speedup:.2f}x",
                    metadata={
                        "columns_skipped": columns_skipped,
                        "incremental_ms": inc_duration,
                        "full_ms": full_duration,
                    },
                )
        else:
            # If nothing skipped, check overhead isn't too high
            overhead = (inc_duration - full_duration) / full_duration if full_duration > 0 else 0
            if overhead > self.max_overhead:
                self.add_issue(
                    ValidationSeverity.WARNING,
                    "Incremental overhead too high when nothing skipped",
                    expected=f"<= {self.max_overhead:.0%} overhead",
                    actual=f"{overhead:.0%}",
                    recommendation="Check fingerprint calculation efficiency",
                )

        return self._issues


# =============================================================================
# Validator Registry
# =============================================================================


class ValidatorRegistry:
    """Registry for validators.

    Allows dynamic registration and discovery of validators.
    """

    def __init__(self):
        self._validators: dict[str, type[BaseValidator]] = {}
        self._instances: dict[str, BaseValidator] = {}

    def register(
        self,
        name: str,
        validator_class: type[BaseValidator],
    ) -> None:
        """Register a validator class."""
        self._validators[name] = validator_class

    def get(self, name: str, **kwargs: Any) -> BaseValidator:
        """Get or create a validator instance."""
        if name not in self._instances:
            if name not in self._validators:
                raise KeyError(f"Unknown validator: {name}")
            self._instances[name] = self._validators[name](**kwargs)
        return self._instances[name]

    def get_all(self, **kwargs: Any) -> list[BaseValidator]:
        """Get all registered validators."""
        return [
            self.get(name, **kwargs)
            for name in self._validators
        ]

    def get_by_category(
        self,
        category: ValidationCategory,
        **kwargs: Any,
    ) -> list[BaseValidator]:
        """Get validators for a category."""
        return [
            v for v in self.get_all(**kwargs)
            if v.category == category
        ]

    @property
    def registered_names(self) -> list[str]:
        """Get names of registered validators."""
        return list(self._validators.keys())


# Global registry
validator_registry = ValidatorRegistry()

# Register built-in validators
validator_registry.register("change_detection_accuracy", ChangeDetectionAccuracyValidator)
validator_registry.register("schema_change", SchemaChangeValidator)
validator_registry.register("staleness", StalenessValidator)
validator_registry.register("fingerprint_consistency", FingerprintConsistencyValidator)
validator_registry.register("fingerprint_sensitivity", FingerprintSensitivityValidator)
validator_registry.register("profile_merge", ProfileMergeValidator)
validator_registry.register("data_integrity", DataIntegrityValidator)
validator_registry.register("performance", PerformanceValidator)


def register_validator(name: str) -> Callable[[type[BaseValidator]], type[BaseValidator]]:
    """Decorator to register a validator."""
    def decorator(cls: type[BaseValidator]) -> type[BaseValidator]:
        validator_registry.register(name, cls)
        return cls
    return decorator


# =============================================================================
# Validation Configuration
# =============================================================================


@dataclass
class ValidationConfig:
    """Configuration for validation.

    Attributes:
        validation_type: Type of validation to perform
        enabled_validators: Set of validator names to run
        disabled_validators: Set of validator names to skip
        max_profile_age: Maximum profile age for staleness checks
        tolerance: Tolerance for numerical comparisons
        fail_on_warning: Whether warnings should fail validation
        fail_on_error: Whether errors should fail validation
        collect_all_issues: Collect all issues or stop on first failure
    """

    validation_type: ValidationType = ValidationType.FULL
    enabled_validators: set[str] | None = None
    disabled_validators: set[str] = field(default_factory=set)
    max_profile_age: timedelta | None = None
    tolerance: float = 0.01
    fail_on_warning: bool = False
    fail_on_error: bool = True
    collect_all_issues: bool = True

    @classmethod
    def quick(cls) -> "ValidationConfig":
        """Quick validation configuration."""
        return cls(
            validation_type=ValidationType.QUICK,
            enabled_validators={
                "change_detection_accuracy",
                "data_integrity",
            },
        )

    @classmethod
    def strict(cls) -> "ValidationConfig":
        """Strict validation configuration."""
        return cls(
            validation_type=ValidationType.FULL,
            fail_on_warning=True,
            tolerance=0.001,
        )

    @classmethod
    def change_detection_only(cls) -> "ValidationConfig":
        """Only validate change detection."""
        return cls(
            validation_type=ValidationType.CHANGE_ONLY,
            enabled_validators={
                "change_detection_accuracy",
                "schema_change",
                "staleness",
                "fingerprint_consistency",
                "fingerprint_sensitivity",
            },
        )


# =============================================================================
# Validation Runner
# =============================================================================


class ValidationRunner:
    """Runs validation checks.

    Orchestrates validators and collects results.

    Example:
        runner = ValidationRunner(ValidationConfig.strict())
        result = runner.run(context)
        if not result.passed:
            print(result.to_markdown())
    """

    def __init__(
        self,
        config: ValidationConfig | None = None,
        registry: ValidatorRegistry | None = None,
    ):
        """Initialize runner.

        Args:
            config: Validation configuration
            registry: Validator registry
        """
        self.config = config or ValidationConfig()
        self.registry = registry or validator_registry

    def run(self, context: ValidationContext) -> ValidationResult:
        """Run validation.

        Args:
            context: Validation context

        Returns:
            Validation result
        """
        start_time = time.perf_counter()
        context.config = self.config

        all_issues: list[ValidationIssue] = []
        metrics = ValidationMetrics()
        details: dict[ValidationCategory, dict[str, Any]] = {}

        # Get validators to run
        validators = self._get_validators()

        for validator in validators:
            if not validator.enabled:
                metrics.skipped_checks += 1
                continue

            try:
                issues = validator.validate(context)
                all_issues.extend(issues)

                metrics.total_checks += 1
                if not issues or all(i.severity == ValidationSeverity.INFO for i in issues):
                    metrics.passed_checks += 1
                else:
                    has_error = any(
                        i.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]
                        for i in issues
                    )
                    if has_error:
                        metrics.failed_checks += 1
                    else:
                        metrics.passed_checks += 1

                # Track details per category
                if validator.category not in details:
                    details[validator.category] = {"validators_run": []}
                details[validator.category]["validators_run"].append(validator.name)

                if not self.config.collect_all_issues:
                    # Check if we should stop
                    has_critical = any(
                        i.severity == ValidationSeverity.CRITICAL for i in issues
                    )
                    if has_critical:
                        break

            except Exception as e:
                logger.exception(f"Validator {validator.name} failed: {e}")
                all_issues.append(ValidationIssue(
                    category=validator.category,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Validator {validator.name} raised exception: {e}",
                ))
                metrics.failed_checks += 1

        # Calculate additional metrics
        metrics.duration_ms = (time.perf_counter() - start_time) * 1000
        metrics.columns_validated = len(context.profiled_columns | context.skipped_columns)
        metrics.changes_detected = len(context.profiled_columns)

        # Count false positives/negatives
        for issue in all_issues:
            if issue.metadata.get("false_positive"):
                metrics.false_positives += 1
            if issue.metadata.get("false_negative"):
                metrics.false_negatives += 1

        # Determine pass/fail
        passed = self._determine_passed(all_issues)

        return ValidationResult(
            passed=passed,
            validation_type=self.config.validation_type,
            issues=all_issues,
            metrics=metrics,
            config={"tolerance": self.config.tolerance},
            details=details,
        )

    def _get_validators(self) -> list[BaseValidator]:
        """Get validators to run based on config."""
        all_validators = self.registry.get_all()

        if self.config.enabled_validators:
            validators = [
                v for v in all_validators
                if v.name in self.config.enabled_validators
            ]
        else:
            validators = all_validators

        # Remove disabled
        validators = [
            v for v in validators
            if v.name not in self.config.disabled_validators
        ]

        return validators

    def _determine_passed(self, issues: list[ValidationIssue]) -> bool:
        """Determine if validation passed."""
        for issue in issues:
            if issue.severity == ValidationSeverity.CRITICAL:
                return False
            if issue.severity == ValidationSeverity.ERROR and self.config.fail_on_error:
                return False
            if issue.severity == ValidationSeverity.WARNING and self.config.fail_on_warning:
                return False
        return True


# =============================================================================
# Main Validator Class
# =============================================================================


class IncrementalValidator:
    """Main validator for incremental profiling.

    Provides a high-level interface for validation.

    Example:
        validator = IncrementalValidator()

        # Validate change detection
        result = validator.validate_change_detection(
            data=df,
            original_profile=profile1,
            incremental_profile=profile2,
        )

        # Validate profile merge
        result = validator.validate_merge(
            profiles=[profile1, profile2],
            merged_profile=merged,
        )

        # Full validation with profiling
        result = validator.validate_full(
            data=df,
            original_profile=profile1,
        )
    """

    def __init__(
        self,
        config: ValidationConfig | None = None,
    ):
        """Initialize validator.

        Args:
            config: Validation configuration
        """
        self.config = config or ValidationConfig()
        self.runner = ValidationRunner(self.config)

    def validate(
        self,
        data: pl.LazyFrame | pl.DataFrame,
        *,
        original_profile: TableProfile | None = None,
        incremental_profile: TableProfile | None = None,
        full_profile: TableProfile | None = None,
        profiled_columns: set[str] | None = None,
        skipped_columns: set[str] | None = None,
        change_reasons: dict[str, ChangeReason] | None = None,
    ) -> ValidationResult:
        """Validate incremental profiling results.

        Args:
            data: Data that was profiled
            original_profile: Previous profile
            incremental_profile: Incremental profile to validate
            full_profile: Full profile for comparison
            profiled_columns: Columns that were re-profiled
            skipped_columns: Columns that were skipped
            change_reasons: Reasons for re-profiling

        Returns:
            Validation result
        """
        context = ValidationContext(
            data=data,
            original_profile=original_profile,
            incremental_profile=incremental_profile,
            full_profile=full_profile,
            profiled_columns=profiled_columns or set(),
            skipped_columns=skipped_columns or set(),
            change_reasons=change_reasons or {},
        )

        return self.runner.run(context)

    def validate_change_detection(
        self,
        data: pl.LazyFrame | pl.DataFrame,
        original_profile: TableProfile,
        incremental_profile: TableProfile,
        *,
        full_profile: TableProfile | None = None,
    ) -> ValidationResult:
        """Validate change detection specifically.

        Focuses on accuracy of change detection.

        Args:
            data: Current data
            original_profile: Previous profile
            incremental_profile: New incremental profile
            full_profile: Optional full profile for comparison

        Returns:
            Validation result
        """
        config = ValidationConfig.change_detection_only()
        runner = ValidationRunner(config)

        context = ValidationContext(
            data=data,
            original_profile=original_profile,
            incremental_profile=incremental_profile,
            full_profile=full_profile,
        )

        return runner.run(context)

    def validate_merge(
        self,
        profiles: list[TableProfile],
        merged_profile: TableProfile,
    ) -> ValidationResult:
        """Validate profile merge.

        Args:
            profiles: Input profiles
            merged_profile: Merged output profile

        Returns:
            Validation result
        """
        config = ValidationConfig(
            validation_type=ValidationType.MERGE_ONLY,
            enabled_validators={"profile_merge"},
        )
        runner = ValidationRunner(config)

        # Create context with merge data
        context = ValidationContext(
            data=pl.DataFrame(),  # Empty, not needed for merge validation
        )
        setattr(context, 'merge_inputs', profiles)
        setattr(context, 'merge_output', merged_profile)

        return runner.run(context)

    def validate_full(
        self,
        data: pl.LazyFrame | pl.DataFrame,
        original_profile: TableProfile,
        *,
        incremental_config: IncrementalConfig | None = None,
    ) -> ValidationResult:
        """Full validation with actual profiling.

        Performs incremental profiling, full profiling, and compares results.

        Args:
            data: Data to profile
            original_profile: Previous profile
            incremental_config: Incremental profiling configuration

        Returns:
            Validation result
        """
        # Perform incremental profiling
        inc_profiler = IncrementalProfiler(config=incremental_config)
        inc_profile = inc_profiler.profile(data, previous=original_profile)

        # Perform full profiling for comparison
        from truthound.profiler.table_profiler import DataProfiler
        full_profiler = DataProfiler()

        lf = data.lazy() if isinstance(data, pl.DataFrame) else data
        full_profile = full_profiler.profile(lf)

        # Calculate fingerprints
        fp_calculator = FingerprintCalculator()
        current_fps = {}

        schema = lf.collect_schema()
        for col_name in schema.names():
            try:
                current_fps[col_name] = fp_calculator.calculate(lf, col_name)
            except Exception:
                pass

        context = ValidationContext(
            data=data,
            original_profile=original_profile,
            incremental_profile=inc_profile,
            full_profile=full_profile,
            current_fingerprints=current_fps,
            profiled_columns=inc_profiler.last_profiled_columns,
            skipped_columns=inc_profiler.last_skipped_columns,
            change_reasons=inc_profiler.last_change_reasons,
        )

        return self.runner.run(context)


# =============================================================================
# Convenience Functions
# =============================================================================


def validate_incremental(
    data: pl.LazyFrame | pl.DataFrame,
    original_profile: TableProfile,
    incremental_profile: TableProfile,
    *,
    full_profile: TableProfile | None = None,
    strict: bool = False,
) -> ValidationResult:
    """Convenience function to validate incremental profiling.

    Args:
        data: Data that was profiled
        original_profile: Previous profile
        incremental_profile: New incremental profile
        full_profile: Optional full profile for comparison
        strict: Use strict validation

    Returns:
        Validation result
    """
    config = ValidationConfig.strict() if strict else ValidationConfig()
    validator = IncrementalValidator(config)

    return validator.validate(
        data=data,
        original_profile=original_profile,
        incremental_profile=incremental_profile,
        full_profile=full_profile,
    )


def validate_merge(
    profiles: list[TableProfile],
    merged_profile: TableProfile,
) -> ValidationResult:
    """Convenience function to validate profile merge.

    Args:
        profiles: Input profiles
        merged_profile: Merged output

    Returns:
        Validation result
    """
    validator = IncrementalValidator()
    return validator.validate_merge(profiles, merged_profile)


def validate_fingerprints(
    data: pl.LazyFrame | pl.DataFrame,
    fingerprints: dict[str, ColumnFingerprint],
) -> ValidationResult:
    """Validate fingerprint consistency.

    Args:
        data: Data to check fingerprints against
        fingerprints: Fingerprints to validate

    Returns:
        Validation result
    """
    config = ValidationConfig(
        enabled_validators={"fingerprint_consistency"},
    )
    runner = ValidationRunner(config)

    context = ValidationContext(
        data=data,
        current_fingerprints=fingerprints,
    )

    return runner.run(context)
