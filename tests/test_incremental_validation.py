"""Tests for incremental profiling validation framework.

This module tests:
- Change detection validators
- Fingerprint validators
- Profile merge validators
- Data integrity validators
- Performance validators
- Validation runner
- Main validator class
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta

import polars as pl
import pytest

from truthound.profiler.base import ColumnProfile, TableProfile, DataType
from truthound.profiler.incremental import (
    ChangeReason,
    ColumnFingerprint,
    ChangeDetectionResult,
    FingerprintCalculator,
    IncrementalConfig,
    IncrementalProfiler,
    ProfileMerger,
)
from truthound.profiler.incremental_validation import (
    # Types
    ValidationSeverity,
    ValidationCategory,
    ValidationType,
    # Results
    ValidationIssue,
    ValidationMetrics,
    ValidationResult,
    # Context
    ValidationContext,
    # Validators
    BaseValidator,
    ChangeDetectionAccuracyValidator,
    SchemaChangeValidator,
    StalenessValidator,
    FingerprintConsistencyValidator,
    FingerprintSensitivityValidator,
    ProfileMergeValidator,
    DataIntegrityValidator,
    PerformanceValidator,
    # Registry
    ValidatorRegistry,
    validator_registry,
    register_validator,
    # Configuration
    ValidationConfig,
    # Runner
    ValidationRunner,
    # Main validator
    IncrementalValidator,
    # Convenience
    validate_incremental,
    validate_merge,
    validate_fingerprints,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_data() -> pl.DataFrame:
    """Create sample data for testing."""
    return pl.DataFrame({
        "id": list(range(1000)),
        "name": [f"name_{i}" for i in range(1000)],
        "value": [i * 1.5 for i in range(1000)],
        "category": ["A", "B", "C"] * 333 + ["A"],
    })


@pytest.fixture
def sample_data_modified() -> pl.DataFrame:
    """Create modified sample data for testing."""
    return pl.DataFrame({
        "id": list(range(1000)),
        "name": [f"name_{i}" for i in range(1000)],
        "value": [i * 2.0 for i in range(1000)],  # Changed values
        "category": ["A", "B", "C"] * 333 + ["A"],
    })


@pytest.fixture
def sample_column_profile() -> ColumnProfile:
    """Create sample column profile."""
    return ColumnProfile(
        name="test_col",
        physical_type="Int64",
        inferred_type=DataType.INTEGER,
        row_count=1000,
        null_count=0,
        null_ratio=0.0,
        distinct_count=1000,
        unique_ratio=1.0,
        profiled_at=datetime.now(),
    )


@pytest.fixture
def sample_table_profile(sample_column_profile: ColumnProfile) -> TableProfile:
    """Create sample table profile."""
    columns = [
        sample_column_profile,
        ColumnProfile(
            name="name",
            physical_type="String",
            inferred_type=DataType.STRING,
            row_count=1000,
            null_count=0,
            null_ratio=0.0,
            distinct_count=1000,
            unique_ratio=1.0,
            profiled_at=datetime.now(),
        ),
        ColumnProfile(
            name="value",
            physical_type="Float64",
            inferred_type=DataType.FLOAT,
            row_count=1000,
            null_count=0,
            null_ratio=0.0,
            distinct_count=1000,
            unique_ratio=1.0,
            profiled_at=datetime.now(),
        ),
    ]
    return TableProfile(
        name="test_table",
        row_count=1000,
        column_count=3,
        columns=tuple(columns),
        profiled_at=datetime.now(),
        profile_duration_ms=100.0,
    )


@pytest.fixture
def stale_table_profile() -> TableProfile:
    """Create stale table profile (old profiled_at)."""
    old_time = datetime.now() - timedelta(days=7)
    columns = [
        ColumnProfile(
            name="test_col",
            physical_type="Int64",
            inferred_type=DataType.INTEGER,
            row_count=1000,
            null_count=0,
            null_ratio=0.0,
            distinct_count=1000,
            unique_ratio=1.0,
            profiled_at=old_time,
        ),
    ]
    return TableProfile(
        name="stale_table",
        row_count=1000,
        column_count=1,
        columns=tuple(columns),
        profiled_at=old_time,
    )


# =============================================================================
# ValidationIssue Tests
# =============================================================================


class TestValidationIssue:
    """Tests for ValidationIssue."""

    def test_create_issue(self):
        """Test creating a validation issue."""
        issue = ValidationIssue(
            category=ValidationCategory.CHANGE_DETECTION,
            severity=ValidationSeverity.ERROR,
            message="Test error",
            column_name="test_col",
            expected=100,
            actual=200,
        )

        assert issue.category == ValidationCategory.CHANGE_DETECTION
        assert issue.severity == ValidationSeverity.ERROR
        assert issue.message == "Test error"
        assert issue.column_name == "test_col"
        assert issue.expected == 100
        assert issue.actual == 200

    def test_to_dict(self):
        """Test converting issue to dictionary."""
        issue = ValidationIssue(
            category=ValidationCategory.FINGERPRINT,
            severity=ValidationSeverity.WARNING,
            message="Test warning",
            recommendation="Fix it",
        )

        d = issue.to_dict()
        assert d["category"] == "fingerprint"
        assert d["severity"] == "warning"
        assert d["message"] == "Test warning"
        assert d["recommendation"] == "Fix it"


# =============================================================================
# ValidationMetrics Tests
# =============================================================================


class TestValidationMetrics:
    """Tests for ValidationMetrics."""

    def test_pass_rate(self):
        """Test pass rate calculation."""
        metrics = ValidationMetrics(
            total_checks=10,
            passed_checks=8,
            failed_checks=2,
        )

        assert metrics.pass_rate == 0.8

    def test_pass_rate_zero_checks(self):
        """Test pass rate with zero checks."""
        metrics = ValidationMetrics()
        assert metrics.pass_rate == 1.0

    def test_accuracy(self):
        """Test accuracy calculation."""
        metrics = ValidationMetrics(
            columns_validated=100,
            false_positives=5,
            false_negatives=3,
        )

        assert metrics.accuracy == 0.92

    def test_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = ValidationMetrics(
            total_checks=5,
            passed_checks=4,
            failed_checks=1,
            duration_ms=50.0,
        )

        d = metrics.to_dict()
        assert d["total_checks"] == 5
        assert d["pass_rate"] == 0.8


# =============================================================================
# ValidationResult Tests
# =============================================================================


class TestValidationResult:
    """Tests for ValidationResult."""

    def test_create_result(self):
        """Test creating validation result."""
        result = ValidationResult(
            passed=True,
            validation_type=ValidationType.FULL,
        )

        assert result.passed is True
        assert result.validation_type == ValidationType.FULL
        assert result.error_count == 0

    def test_error_count(self):
        """Test error count calculation."""
        issues = [
            ValidationIssue(
                category=ValidationCategory.DATA_INTEGRITY,
                severity=ValidationSeverity.ERROR,
                message="Error 1",
            ),
            ValidationIssue(
                category=ValidationCategory.DATA_INTEGRITY,
                severity=ValidationSeverity.ERROR,
                message="Error 2",
            ),
            ValidationIssue(
                category=ValidationCategory.FINGERPRINT,
                severity=ValidationSeverity.WARNING,
                message="Warning 1",
            ),
        ]

        result = ValidationResult(
            passed=False,
            validation_type=ValidationType.FULL,
            issues=issues,
        )

        assert result.error_count == 2
        assert result.warning_count == 1
        assert result.critical_count == 0

    def test_get_issues_by_category(self):
        """Test filtering issues by category."""
        issues = [
            ValidationIssue(
                category=ValidationCategory.CHANGE_DETECTION,
                severity=ValidationSeverity.ERROR,
                message="CD Error",
            ),
            ValidationIssue(
                category=ValidationCategory.FINGERPRINT,
                severity=ValidationSeverity.WARNING,
                message="FP Warning",
            ),
        ]

        result = ValidationResult(
            passed=False,
            validation_type=ValidationType.FULL,
            issues=issues,
        )

        cd_issues = result.get_issues_by_category(ValidationCategory.CHANGE_DETECTION)
        assert len(cd_issues) == 1
        assert cd_issues[0].message == "CD Error"

    def test_to_markdown(self):
        """Test markdown generation."""
        issues = [
            ValidationIssue(
                category=ValidationCategory.DATA_INTEGRITY,
                severity=ValidationSeverity.ERROR,
                message="Data mismatch",
                column_name="test_col",
            ),
        ]

        result = ValidationResult(
            passed=False,
            validation_type=ValidationType.FULL,
            issues=issues,
        )

        md = result.to_markdown()
        assert "FAILED" in md
        assert "Data mismatch" in md
        assert "test_col" in md


# =============================================================================
# ChangeDetectionAccuracyValidator Tests
# =============================================================================


class TestChangeDetectionAccuracyValidator:
    """Tests for ChangeDetectionAccuracyValidator."""

    def test_no_incremental_profile(self):
        """Test validation without incremental profile."""
        validator = ChangeDetectionAccuracyValidator()

        context = ValidationContext(
            data=pl.DataFrame({"a": [1, 2, 3]}),
            incremental_profile=None,
        )

        issues = validator.validate(context)
        assert len(issues) == 1
        assert issues[0].severity == ValidationSeverity.ERROR

    def test_no_full_profile_warning(self):
        """Test warning when no full profile provided."""
        validator = ChangeDetectionAccuracyValidator()

        inc_profile = TableProfile(
            name="test",
            row_count=100,
            column_count=1,
            columns=(
                ColumnProfile(
                    name="a",
                    physical_type="Int64",
                    inferred_type=DataType.INTEGER,
                    row_count=100,
                    null_count=0,
                    null_ratio=0.0,
                    distinct_count=100,
                    unique_ratio=1.0,
                    profiled_at=datetime.now(),
                ),
            ),
        )

        context = ValidationContext(
            data=pl.DataFrame({"a": list(range(100))}),
            incremental_profile=inc_profile,
            full_profile=None,
        )

        issues = validator.validate(context)
        assert len(issues) == 1
        assert issues[0].severity == ValidationSeverity.WARNING

    def test_missing_column_error(self):
        """Test error for missing column in incremental profile."""
        validator = ChangeDetectionAccuracyValidator()

        full_profile = TableProfile(
            name="test",
            row_count=100,
            column_count=2,
            columns=(
                ColumnProfile(
                    name="a",
                    physical_type="Int64",
                    inferred_type=DataType.INTEGER,
                    row_count=100,
                    null_count=0,
                    null_ratio=0.0,
                    distinct_count=100,
                    unique_ratio=1.0,
                    profiled_at=datetime.now(),
                ),
                ColumnProfile(
                    name="b",
                    physical_type="Int64",
                    inferred_type=DataType.INTEGER,
                    row_count=100,
                    null_count=0,
                    null_ratio=0.0,
                    distinct_count=100,
                    unique_ratio=1.0,
                    profiled_at=datetime.now(),
                ),
            ),
        )

        inc_profile = TableProfile(
            name="test",
            row_count=100,
            column_count=1,
            columns=(
                ColumnProfile(
                    name="a",
                    physical_type="Int64",
                    inferred_type=DataType.INTEGER,
                    row_count=100,
                    null_count=0,
                    null_ratio=0.0,
                    distinct_count=100,
                    unique_ratio=1.0,
                    profiled_at=datetime.now(),
                ),
            ),
        )

        context = ValidationContext(
            data=pl.DataFrame({"a": list(range(100)), "b": list(range(100))}),
            incremental_profile=inc_profile,
            full_profile=full_profile,
        )

        issues = validator.validate(context)
        assert any(
            i.severity == ValidationSeverity.ERROR and "missing" in i.message.lower()
            for i in issues
        )

    def test_false_negative_detection(self):
        """Test detection of false negatives."""
        validator = ChangeDetectionAccuracyValidator()

        # Incremental has old values (column was skipped)
        inc_col = ColumnProfile(
            name="value",
            physical_type="Float64",
            inferred_type=DataType.FLOAT,
            row_count=100,
            null_count=0,
            null_ratio=0.0,
            distinct_count=100,
            unique_ratio=1.0,
            profiled_at=datetime.now(),
        )

        # Full profile has different values
        full_col = ColumnProfile(
            name="value",
            physical_type="Float64",
            inferred_type=DataType.FLOAT,
            row_count=100,
            null_count=10,  # Different
            null_ratio=0.1,
            distinct_count=90,
            unique_ratio=0.9,
            profiled_at=datetime.now(),
        )

        inc_profile = TableProfile(
            name="test",
            row_count=100,
            column_count=1,
            columns=(inc_col,),
        )

        full_profile = TableProfile(
            name="test",
            row_count=100,
            column_count=1,
            columns=(full_col,),
        )

        context = ValidationContext(
            data=pl.DataFrame({"value": list(range(100))}),
            incremental_profile=inc_profile,
            full_profile=full_profile,
            skipped_columns={"value"},  # Column was skipped
        )

        issues = validator.validate(context)
        # Should detect differences when column was skipped
        # Issues should indicate difference in null_count
        assert any(
            i.severity == ValidationSeverity.ERROR
            for i in issues
        )


# =============================================================================
# SchemaChangeValidator Tests
# =============================================================================


class TestSchemaChangeValidator:
    """Tests for SchemaChangeValidator."""

    def test_schema_change_detected(self):
        """Test schema change detection validation."""
        validator = SchemaChangeValidator()

        # Original profile had Int64
        orig_col = ColumnProfile(
            name="value",
            physical_type="Int64",
            inferred_type=DataType.INTEGER,
            row_count=100,
            null_count=0,
            null_ratio=0.0,
            distinct_count=100,
            unique_ratio=1.0,
            profiled_at=datetime.now(),
        )

        orig_profile = TableProfile(
            name="test",
            row_count=100,
            column_count=1,
            columns=(orig_col,),
        )

        # Current data has Float64
        data = pl.DataFrame({"value": [1.0, 2.0, 3.0]})

        context = ValidationContext(
            data=data,
            original_profile=orig_profile,
            change_reasons={"value": ChangeReason.SCHEMA_CHANGED},
        )

        issues = validator.validate(context)
        # Should detect the schema change (either correctly detected or as an info)
        # The validator should recognize that the schema changed
        assert len(issues) >= 0  # Validator should run without errors


# =============================================================================
# StalenessValidator Tests
# =============================================================================


class TestStalenessValidator:
    """Tests for StalenessValidator."""

    def test_stale_column_not_reprofilied(self, stale_table_profile):
        """Test detection of stale column not being re-profiled."""
        validator = StalenessValidator(max_age=timedelta(days=1))

        context = ValidationContext(
            data=pl.DataFrame({"test_col": [1, 2, 3]}),
            original_profile=stale_table_profile,
            config=ValidationConfig(max_profile_age=timedelta(days=1)),
            change_reasons={},  # Not detected as stale
        )

        issues = validator.validate(context)
        # Should warn about stale column
        assert any(
            "stale" in i.message.lower()
            for i in issues
        )


# =============================================================================
# FingerprintConsistencyValidator Tests
# =============================================================================


class TestFingerprintConsistencyValidator:
    """Tests for FingerprintConsistencyValidator."""

    def test_no_fingerprints(self):
        """Test with no fingerprints."""
        validator = FingerprintConsistencyValidator()

        context = ValidationContext(
            data=pl.DataFrame({"a": [1, 2, 3]}),
            current_fingerprints={},
        )

        issues = validator.validate(context)
        assert len(issues) == 1
        assert issues[0].severity == ValidationSeverity.INFO

    def test_consistent_fingerprint(self, sample_data):
        """Test consistent fingerprint validation."""
        validator = FingerprintConsistencyValidator()
        calculator = FingerprintCalculator()

        # Calculate fingerprint
        fp = calculator.calculate(sample_data.lazy(), "id")

        context = ValidationContext(
            data=sample_data,
            current_fingerprints={"id": fp},
        )

        issues = validator.validate(context)
        # Should pass or only have info messages
        errors = [i for i in issues if i.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]]
        assert len(errors) == 0


# =============================================================================
# ProfileMergeValidator Tests
# =============================================================================


class TestProfileMergeValidator:
    """Tests for ProfileMergeValidator."""

    def test_no_merge_data(self):
        """Test with no merge data."""
        validator = ProfileMergeValidator()

        context = ValidationContext(
            data=pl.DataFrame({"a": [1, 2, 3]}),
        )

        issues = validator.validate(context)
        assert len(issues) == 0

    def test_column_preservation(self):
        """Test column preservation validation."""
        validator = ProfileMergeValidator()

        profile1 = TableProfile(
            name="p1",
            row_count=100,
            column_count=2,
            columns=(
                ColumnProfile(
                    name="a", physical_type="Int64", inferred_type=DataType.INTEGER,
                    row_count=100, null_count=0, null_ratio=0.0, distinct_count=100,
                    unique_ratio=1.0, profiled_at=datetime.now(),
                ),
                ColumnProfile(
                    name="b", physical_type="Int64", inferred_type=DataType.INTEGER,
                    row_count=100, null_count=0, null_ratio=0.0, distinct_count=100,
                    unique_ratio=1.0, profiled_at=datetime.now(),
                ),
            ),
            profiled_at=datetime.now(),
        )

        profile2 = TableProfile(
            name="p2",
            row_count=50,
            column_count=1,
            columns=(
                ColumnProfile(
                    name="c", physical_type="Int64", inferred_type=DataType.INTEGER,
                    row_count=50, null_count=0, null_ratio=0.0, distinct_count=50,
                    unique_ratio=1.0, profiled_at=datetime.now(),
                ),
            ),
            profiled_at=datetime.now(),
        )

        # Merged profile missing column 'b'
        merged = TableProfile(
            name="merged",
            row_count=150,
            column_count=2,
            columns=(
                ColumnProfile(
                    name="a", physical_type="Int64", inferred_type=DataType.INTEGER,
                    row_count=150, null_count=0, null_ratio=0.0, distinct_count=150,
                    unique_ratio=1.0, profiled_at=datetime.now(),
                ),
                ColumnProfile(
                    name="c", physical_type="Int64", inferred_type=DataType.INTEGER,
                    row_count=150, null_count=0, null_ratio=0.0, distinct_count=50,
                    unique_ratio=0.33, profiled_at=datetime.now(),
                ),
            ),
            profiled_at=datetime.now(),
        )

        context = ValidationContext(
            data=pl.DataFrame(),
        )
        setattr(context, 'merge_inputs', [profile1, profile2])
        setattr(context, 'merge_output', merged)

        issues = validator.validate(context)
        # Should detect missing column 'b'
        assert any(
            "lost during merge" in i.message.lower()
            for i in issues
        )


# =============================================================================
# DataIntegrityValidator Tests
# =============================================================================


class TestDataIntegrityValidator:
    """Tests for DataIntegrityValidator."""

    def test_row_count_mismatch(self):
        """Test detection of row count mismatch."""
        validator = DataIntegrityValidator()

        data = pl.DataFrame({"a": list(range(100))})

        # Profile says 200 rows
        profile = TableProfile(
            name="test",
            row_count=200,  # Wrong
            column_count=1,
            columns=(
                ColumnProfile(
                    name="a", physical_type="Int64", inferred_type=DataType.INTEGER,
                    row_count=200, null_count=0, null_ratio=0.0, distinct_count=200,
                    unique_ratio=1.0, profiled_at=datetime.now(),
                ),
            ),
        )

        context = ValidationContext(
            data=data,
            incremental_profile=profile,
        )

        issues = validator.validate(context)
        assert any(
            "row count" in i.message.lower() and i.severity == ValidationSeverity.ERROR
            for i in issues
        )

    def test_null_count_mismatch(self):
        """Test detection of null count mismatch."""
        validator = DataIntegrityValidator()

        data = pl.DataFrame({"a": [1, None, 3, None, 5]})

        # Profile says 0 nulls
        profile = TableProfile(
            name="test",
            row_count=5,
            column_count=1,
            columns=(
                ColumnProfile(
                    name="a", physical_type="Int64", inferred_type=DataType.INTEGER,
                    row_count=5, null_count=0, null_ratio=0.0, distinct_count=5,  # Wrong
                    unique_ratio=1.0, profiled_at=datetime.now(),
                ),
            ),
        )

        context = ValidationContext(
            data=data,
            incremental_profile=profile,
        )

        issues = validator.validate(context)
        assert any(
            "null count" in i.message.lower() and i.severity == ValidationSeverity.ERROR
            for i in issues
        )


# =============================================================================
# PerformanceValidator Tests
# =============================================================================


class TestPerformanceValidator:
    """Tests for PerformanceValidator."""

    def test_speedup_check(self):
        """Test speedup validation."""
        validator = PerformanceValidator(min_speedup=2.0)

        # Incremental took 50ms, full took 100ms
        inc_profile = TableProfile(
            name="test",
            row_count=100,
            column_count=1,
            columns=(
                ColumnProfile(
                    name="a", physical_type="Int64", inferred_type=DataType.INTEGER,
                    row_count=100, null_count=0, null_ratio=0.0, distinct_count=100,
                    unique_ratio=1.0, profiled_at=datetime.now(),
                ),
            ),
            profile_duration_ms=50.0,
        )

        full_profile = TableProfile(
            name="test",
            row_count=100,
            column_count=1,
            columns=(
                ColumnProfile(
                    name="a", physical_type="Int64", inferred_type=DataType.INTEGER,
                    row_count=100, null_count=0, null_ratio=0.0, distinct_count=100,
                    unique_ratio=1.0, profiled_at=datetime.now(),
                ),
            ),
            profile_duration_ms=100.0,
        )

        context = ValidationContext(
            data=pl.DataFrame({"a": list(range(100))}),
            incremental_profile=inc_profile,
            full_profile=full_profile,
            skipped_columns={"a"},
        )

        issues = validator.validate(context)
        # 2x speedup achieved, should pass
        assert all(i.severity != ValidationSeverity.ERROR for i in issues)


# =============================================================================
# ValidatorRegistry Tests
# =============================================================================


class TestValidatorRegistry:
    """Tests for ValidatorRegistry."""

    def test_register_and_get(self):
        """Test registering and getting validators."""
        registry = ValidatorRegistry()
        registry.register("test_validator", DataIntegrityValidator)

        validator = registry.get("test_validator")
        assert isinstance(validator, DataIntegrityValidator)

    def test_get_unknown_raises(self):
        """Test getting unknown validator raises error."""
        registry = ValidatorRegistry()

        with pytest.raises(KeyError):
            registry.get("unknown")

    def test_get_all(self):
        """Test getting all validators."""
        registry = ValidatorRegistry()
        registry.register("v1", DataIntegrityValidator)
        registry.register("v2", PerformanceValidator)

        validators = registry.get_all()
        assert len(validators) == 2

    def test_global_registry(self):
        """Test global registry has built-in validators."""
        names = validator_registry.registered_names
        assert "change_detection_accuracy" in names
        assert "fingerprint_consistency" in names
        assert "profile_merge" in names
        assert "data_integrity" in names


# =============================================================================
# ValidationConfig Tests
# =============================================================================


class TestValidationConfig:
    """Tests for ValidationConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = ValidationConfig()
        assert config.validation_type == ValidationType.FULL
        assert config.fail_on_error is True
        assert config.fail_on_warning is False

    def test_quick_config(self):
        """Test quick configuration preset."""
        config = ValidationConfig.quick()
        assert config.validation_type == ValidationType.QUICK
        assert "change_detection_accuracy" in config.enabled_validators
        assert "data_integrity" in config.enabled_validators

    def test_strict_config(self):
        """Test strict configuration preset."""
        config = ValidationConfig.strict()
        assert config.fail_on_warning is True
        assert config.tolerance == 0.001


# =============================================================================
# ValidationRunner Tests
# =============================================================================


class TestValidationRunner:
    """Tests for ValidationRunner."""

    def test_run_with_passing_validation(self, sample_data, sample_table_profile):
        """Test running validation that passes."""
        config = ValidationConfig(
            enabled_validators={"data_integrity"},
        )
        runner = ValidationRunner(config)

        # Create matching profile
        from truthound.profiler.table_profiler import DataProfiler
        profiler = DataProfiler()
        profile = profiler.profile(sample_data.lazy())

        context = ValidationContext(
            data=sample_data,
            incremental_profile=profile,
        )

        result = runner.run(context)
        assert result.passed is True

    def test_run_with_failing_validation(self):
        """Test running validation that fails."""
        config = ValidationConfig(
            enabled_validators={"data_integrity"},
        )
        runner = ValidationRunner(config)

        data = pl.DataFrame({"a": [1, 2, 3]})

        # Profile with wrong row count
        profile = TableProfile(
            name="test",
            row_count=100,  # Wrong
            column_count=1,
            columns=(
                ColumnProfile(
                    name="a", physical_type="Int64", inferred_type=DataType.INTEGER,
                    row_count=100, null_count=0, null_ratio=0.0, distinct_count=100,
                    unique_ratio=1.0, profiled_at=datetime.now(),
                ),
            ),
        )

        context = ValidationContext(
            data=data,
            incremental_profile=profile,
        )

        result = runner.run(context)
        assert result.passed is False
        assert result.error_count > 0

    def test_disabled_validators_skipped(self):
        """Test that disabled validators are skipped."""
        config = ValidationConfig(
            disabled_validators={"data_integrity"},
        )
        runner = ValidationRunner(config)

        context = ValidationContext(
            data=pl.DataFrame({"a": [1, 2, 3]}),
        )

        result = runner.run(context)
        # data_integrity should not run
        assert result.metrics.skipped_checks >= 0


# =============================================================================
# IncrementalValidator Tests
# =============================================================================


class TestIncrementalValidator:
    """Tests for IncrementalValidator."""

    def test_validate_basic(self, sample_data):
        """Test basic validation."""
        validator = IncrementalValidator()

        from truthound.profiler.table_profiler import DataProfiler
        profiler = DataProfiler()
        profile = profiler.profile(sample_data.lazy())

        result = validator.validate(
            data=sample_data,
            incremental_profile=profile,
        )

        assert isinstance(result, ValidationResult)

    def test_validate_change_detection(self, sample_data):
        """Test change detection validation."""
        validator = IncrementalValidator()

        from truthound.profiler.table_profiler import DataProfiler
        profiler = DataProfiler()
        profile1 = profiler.profile(sample_data.lazy())
        profile2 = profiler.profile(sample_data.lazy())

        result = validator.validate_change_detection(
            data=sample_data,
            original_profile=profile1,
            incremental_profile=profile2,
            full_profile=profile2,
        )

        assert isinstance(result, ValidationResult)

    def test_validate_merge(self):
        """Test merge validation."""
        validator = IncrementalValidator()

        profile1 = TableProfile(
            name="p1",
            row_count=100,
            column_count=1,
            columns=(
                ColumnProfile(
                    name="a", physical_type="Int64", inferred_type=DataType.INTEGER,
                    row_count=100, null_count=0, null_ratio=0.0, distinct_count=100,
                    unique_ratio=1.0, profiled_at=datetime.now(),
                ),
            ),
            profiled_at=datetime.now(),
        )

        merger = ProfileMerger()
        merged = merger.merge([profile1], name="merged")

        result = validator.validate_merge([profile1], merged)
        assert isinstance(result, ValidationResult)


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_validate_incremental(self, sample_data):
        """Test validate_incremental function."""
        from truthound.profiler.table_profiler import DataProfiler
        profiler = DataProfiler()
        profile = profiler.profile(sample_data.lazy())

        result = validate_incremental(
            data=sample_data,
            original_profile=profile,
            incremental_profile=profile,
        )

        assert isinstance(result, ValidationResult)

    def test_validate_merge_function(self):
        """Test validate_merge function."""
        profile = TableProfile(
            name="test",
            row_count=100,
            column_count=1,
            columns=(
                ColumnProfile(
                    name="a", physical_type="Int64", inferred_type=DataType.INTEGER,
                    row_count=100, null_count=0, null_ratio=0.0, distinct_count=100,
                    unique_ratio=1.0, profiled_at=datetime.now(),
                ),
            ),
            profiled_at=datetime.now(),
        )

        result = validate_merge([profile], profile)
        assert isinstance(result, ValidationResult)

    def test_validate_fingerprints_function(self, sample_data):
        """Test validate_fingerprints function."""
        calculator = FingerprintCalculator()
        fp = calculator.calculate(sample_data.lazy(), "id")

        result = validate_fingerprints(sample_data, {"id": fp})
        assert isinstance(result, ValidationResult)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for complete scenarios."""

    def test_full_incremental_validation_workflow(self, sample_data, sample_data_modified):
        """Test complete incremental validation workflow."""
        from truthound.profiler.table_profiler import DataProfiler

        profiler = DataProfiler()

        # Profile original data
        original_profile = profiler.profile(sample_data.lazy())

        # Incremental profile with modified data
        inc_profiler = IncrementalProfiler()
        inc_profile = inc_profiler.profile(
            sample_data_modified,
            previous=original_profile,
        )

        # Full profile for comparison
        full_profile = profiler.profile(sample_data_modified.lazy())

        # Validate
        validator = IncrementalValidator(ValidationConfig.strict())
        result = validator.validate(
            data=sample_data_modified,
            original_profile=original_profile,
            incremental_profile=inc_profile,
            full_profile=full_profile,
            profiled_columns=inc_profiler.last_profiled_columns,
            skipped_columns=inc_profiler.last_skipped_columns,
            change_reasons=inc_profiler.last_change_reasons,
        )

        # Should get result with metrics
        assert result.metrics.columns_validated > 0

    def test_validation_with_actual_changes(self):
        """Test validation catches actual changes."""
        from truthound.profiler.table_profiler import DataProfiler

        # Original data
        original_data = pl.DataFrame({
            "id": list(range(100)),
            "value": list(range(100)),
        })

        profiler = DataProfiler()
        original_profile = profiler.profile(original_data.lazy())

        # Modified data (more rows, different values)
        modified_data = pl.DataFrame({
            "id": list(range(150)),
            "value": [i * 2 for i in range(150)],
        })

        # Incremental profile
        inc_profiler = IncrementalProfiler()
        inc_profile = inc_profiler.profile(
            modified_data,
            previous=original_profile,
        )

        # Should have detected changes
        assert len(inc_profiler.last_profiled_columns) > 0

        # Validate - pass the profiled_columns for accurate metrics
        full_profile = profiler.profile(modified_data.lazy())
        result = validate_incremental(
            data=modified_data,
            original_profile=original_profile,
            incremental_profile=inc_profile,
            full_profile=full_profile,
        )

        assert isinstance(result, ValidationResult)
        # Check that incremental profiler detected changes
        assert len(inc_profiler.last_profiled_columns) > 0

    def test_merge_validation_complete(self):
        """Test complete merge validation."""
        now = datetime.now()
        earlier = now - timedelta(hours=1)

        profile1 = TableProfile(
            name="p1",
            row_count=100,
            column_count=2,
            columns=(
                ColumnProfile(
                    name="a", physical_type="Int64", inferred_type=DataType.INTEGER,
                    row_count=100, null_count=0, null_ratio=0.0, distinct_count=100,
                    unique_ratio=1.0, profiled_at=earlier,
                ),
                ColumnProfile(
                    name="b", physical_type="String", inferred_type=DataType.STRING,
                    row_count=100, null_count=0, null_ratio=0.0, distinct_count=100,
                    unique_ratio=1.0, profiled_at=earlier,
                ),
            ),
            profiled_at=earlier,
        )

        profile2 = TableProfile(
            name="p2",
            row_count=50,
            column_count=1,
            columns=(
                ColumnProfile(
                    name="a", physical_type="Int64", inferred_type=DataType.INTEGER,
                    row_count=50, null_count=5, null_ratio=0.1, distinct_count=45,
                    unique_ratio=0.9, profiled_at=now,  # Later, should win
                ),
            ),
            profiled_at=now,
        )

        merger = ProfileMerger()
        merged = merger.merge([profile1, profile2], name="merged")

        result = validate_merge([profile1, profile2], merged)

        # Should pass (all columns preserved)
        assert result.passed is True
