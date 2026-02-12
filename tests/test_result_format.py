"""Comprehensive tests for the ResultFormat system (PHASE 1).

Tests cover:
1. ResultFormat enum and ResultFormatConfig (types.py)
2. ValidatorConfig integration with result_format
3. Expression-based validators with result_format-aware phases
4. Non-expression validators with result_format awareness
5. ExpressionBatchExecutor with result_format
6. Report serialization with result_format
7. Cross-level consistency: same pass/fail regardless of format
"""

import polars as pl
import pytest

from truthound.types import ResultFormat, ResultFormatConfig, Severity
from truthound.validators.base import (
    ExpressionBatchExecutor,
    ValidationExpressionSpec,
    ValidationIssue,
    Validator,
    ValidatorConfig,
    ExpressionValidatorMixin,
)
from truthound.validators.completeness.null import (
    CompletenessRatioValidator,
    NotNullValidator,
    NullValidator,
)
from truthound.validators.distribution.range import (
    BetweenValidator,
    NonNegativeValidator,
    PositiveValidator,
    RangeValidator,
)
from truthound.validators.schema.column_type import ColumnTypeValidator
from truthound.validators.uniqueness.unique import (
    DistinctCountValidator,
    UniqueRatioValidator,
    UniqueValidator,
)
from truthound.report import Report


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_lf():
    """LazyFrame with nulls, out-of-range, and duplicate values."""
    return pl.DataFrame({
        "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "name": ["Alice", "Bob", None, "Dave", None, "Frank", "Grace", "Heidi", "Ivan", "Judy"],
        "age": [25, -5, 30, 150, 45, 60, 200, 18, 35, 42],
        "score": [85, 92, 78, None, 105, 63, 71, 88, 99, -10],
        "status": ["active", "active", "inactive", "active", "inactive", "active", "active", "inactive", "active", "active"],
    }).lazy()


@pytest.fixture
def null_heavy_lf():
    """LazyFrame with many nulls for completeness testing."""
    return pl.DataFrame({
        "col_a": [1, None, None, None, 5, None, 7, None, None, 10],
        "col_b": [None, None, None, None, None, None, None, None, None, None],
        "col_c": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    }).lazy()


@pytest.fixture
def duplicate_lf():
    """LazyFrame with duplicate values."""
    return pl.DataFrame({
        "user_id": [1, 2, 3, 1, 4, 2, 5, 6, 1, 7],
        "value": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    }).lazy()


# ============================================================================
# 1. ResultFormat Enum Tests
# ============================================================================

class TestResultFormat:
    """Tests for ResultFormat enum."""

    def test_from_string_valid(self):
        assert ResultFormat.from_string("boolean_only") == ResultFormat.BOOLEAN_ONLY
        assert ResultFormat.from_string("basic") == ResultFormat.BASIC
        assert ResultFormat.from_string("summary") == ResultFormat.SUMMARY
        assert ResultFormat.from_string("complete") == ResultFormat.COMPLETE

    def test_from_string_case_insensitive(self):
        assert ResultFormat.from_string("BOOLEAN_ONLY") == ResultFormat.BOOLEAN_ONLY
        assert ResultFormat.from_string("Basic") == ResultFormat.BASIC
        assert ResultFormat.from_string("  SUMMARY  ") == ResultFormat.SUMMARY

    def test_from_string_invalid(self):
        with pytest.raises(ValueError, match="Invalid result_format"):
            ResultFormat.from_string("invalid")

    def test_comparison_operators(self):
        assert ResultFormat.BOOLEAN_ONLY < ResultFormat.BASIC
        assert ResultFormat.BASIC < ResultFormat.SUMMARY
        assert ResultFormat.SUMMARY < ResultFormat.COMPLETE
        assert ResultFormat.COMPLETE >= ResultFormat.SUMMARY
        assert ResultFormat.BOOLEAN_ONLY <= ResultFormat.COMPLETE
        assert not (ResultFormat.COMPLETE < ResultFormat.BASIC)

    def test_equality(self):
        assert ResultFormat.SUMMARY == ResultFormat.SUMMARY
        assert ResultFormat.BOOLEAN_ONLY != ResultFormat.COMPLETE


# ============================================================================
# 2. ResultFormatConfig Tests
# ============================================================================

class TestResultFormatConfig:
    """Tests for ResultFormatConfig dataclass."""

    def test_default_values(self):
        config = ResultFormatConfig()
        assert config.format == ResultFormat.SUMMARY
        assert config.partial_unexpected_count == 20
        assert config.include_unexpected_rows is False
        assert config.max_unexpected_rows == 1000

    def test_custom_values(self):
        config = ResultFormatConfig(
            format=ResultFormat.COMPLETE,
            partial_unexpected_count=50,
            include_unexpected_rows=True,
            max_unexpected_rows=500,
        )
        assert config.format == ResultFormat.COMPLETE
        assert config.partial_unexpected_count == 50
        assert config.include_unexpected_rows is True

    def test_frozen_immutability(self):
        config = ResultFormatConfig()
        with pytest.raises(AttributeError):
            config.format = ResultFormat.BASIC  # type: ignore

    def test_validation_errors(self):
        with pytest.raises(ValueError, match="partial_unexpected_count"):
            ResultFormatConfig(partial_unexpected_count=-1)
        with pytest.raises(ValueError, match="max_unexpected_rows"):
            ResultFormatConfig(max_unexpected_rows=0)

    def test_includes_observed_value(self):
        assert not ResultFormatConfig(format=ResultFormat.BOOLEAN_ONLY).includes_observed_value()
        assert ResultFormatConfig(format=ResultFormat.BASIC).includes_observed_value()
        assert ResultFormatConfig(format=ResultFormat.SUMMARY).includes_observed_value()
        assert ResultFormatConfig(format=ResultFormat.COMPLETE).includes_observed_value()

    def test_includes_unexpected_samples(self):
        assert not ResultFormatConfig(format=ResultFormat.BOOLEAN_ONLY).includes_unexpected_samples()
        assert ResultFormatConfig(format=ResultFormat.BASIC).includes_unexpected_samples()
        assert ResultFormatConfig(format=ResultFormat.SUMMARY).includes_unexpected_samples()
        assert ResultFormatConfig(format=ResultFormat.COMPLETE).includes_unexpected_samples()

    def test_includes_unexpected_counts(self):
        assert not ResultFormatConfig(format=ResultFormat.BOOLEAN_ONLY).includes_unexpected_counts()
        assert not ResultFormatConfig(format=ResultFormat.BASIC).includes_unexpected_counts()
        assert ResultFormatConfig(format=ResultFormat.SUMMARY).includes_unexpected_counts()
        assert ResultFormatConfig(format=ResultFormat.COMPLETE).includes_unexpected_counts()

    def test_includes_full_results(self):
        assert not ResultFormatConfig(format=ResultFormat.BOOLEAN_ONLY).includes_full_results()
        assert not ResultFormatConfig(format=ResultFormat.BASIC).includes_full_results()
        assert not ResultFormatConfig(format=ResultFormat.SUMMARY).includes_full_results()
        assert ResultFormatConfig(format=ResultFormat.COMPLETE).includes_full_results()

    def test_from_any_none(self):
        config = ResultFormatConfig.from_any(None)
        assert config.format == ResultFormat.SUMMARY

    def test_from_any_string(self):
        config = ResultFormatConfig.from_any("complete")
        assert config.format == ResultFormat.COMPLETE

    def test_from_any_enum(self):
        config = ResultFormatConfig.from_any(ResultFormat.BASIC)
        assert config.format == ResultFormat.BASIC

    def test_from_any_config(self):
        original = ResultFormatConfig(format=ResultFormat.COMPLETE, partial_unexpected_count=50)
        result = ResultFormatConfig.from_any(original)
        assert result is original  # Should return same object

    def test_from_any_invalid(self):
        with pytest.raises(TypeError, match="Cannot convert"):
            ResultFormatConfig.from_any(123)  # type: ignore

    def test_replace(self):
        config = ResultFormatConfig()
        new_config = config.replace(format=ResultFormat.COMPLETE, partial_unexpected_count=50)
        assert new_config.format == ResultFormat.COMPLETE
        assert new_config.partial_unexpected_count == 50
        assert config.format == ResultFormat.SUMMARY  # Original unchanged


# ============================================================================
# 3. ValidatorConfig Integration Tests
# ============================================================================

class TestValidatorConfigResultFormat:
    """Tests for ValidatorConfig with result_format field."""

    def test_default_result_format(self):
        config = ValidatorConfig()
        assert config.result_format == ResultFormat.SUMMARY

    def test_result_format_enum(self):
        config = ValidatorConfig(result_format=ResultFormat.BOOLEAN_ONLY)
        assert config.result_format == ResultFormat.BOOLEAN_ONLY

    def test_result_format_string_normalized(self):
        config = ValidatorConfig(result_format="complete")
        assert config.result_format == ResultFormat.COMPLETE

    def test_result_format_config(self):
        rf_config = ResultFormatConfig(format=ResultFormat.COMPLETE, partial_unexpected_count=50)
        config = ValidatorConfig(result_format=rf_config)
        assert isinstance(config.result_format, ResultFormatConfig)
        assert config.get_result_format_config().partial_unexpected_count == 50

    def test_get_result_format_config(self):
        config = ValidatorConfig(result_format=ResultFormat.BASIC)
        rf = config.get_result_format_config()
        assert isinstance(rf, ResultFormatConfig)
        assert rf.format == ResultFormat.BASIC

    def test_replace_preserves_result_format(self):
        config = ValidatorConfig(result_format=ResultFormat.COMPLETE)
        new_config = config.replace(sample_size=10)
        assert new_config.result_format == ResultFormat.COMPLETE

    def test_from_kwargs_result_format(self):
        config = ValidatorConfig.from_kwargs(
            result_format="boolean_only",
            sample_size=10,
        )
        assert config.result_format == ResultFormat.BOOLEAN_ONLY
        assert config.sample_size == 10


# ============================================================================
# 4. NullValidator Result Format Tests (Expression-based)
# ============================================================================

class TestNullValidatorResultFormat:
    """Tests for NullValidator with different result_format levels."""

    def test_boolean_only_no_details(self, null_heavy_lf):
        validator = NullValidator(result_format=ResultFormat.BOOLEAN_ONLY)
        issues = validator.validate(null_heavy_lf)

        assert len(issues) > 0
        for issue in issues:
            assert issue.details is None
            assert issue.expected is None
            assert issue.sample_values is None

    def test_basic_has_details(self, null_heavy_lf):
        validator = NullValidator(result_format=ResultFormat.BASIC)
        issues = validator.validate(null_heavy_lf)

        assert len(issues) > 0
        for issue in issues:
            assert issue.details is not None
            assert "null" in issue.details.lower()

    def test_summary_has_details_and_counts(self, null_heavy_lf):
        validator = NullValidator(result_format=ResultFormat.SUMMARY)
        issues = validator.validate(null_heavy_lf)

        assert len(issues) > 0
        # Summary should have details
        for issue in issues:
            assert issue.details is not None

    def test_complete_has_full_results(self, null_heavy_lf):
        validator = NullValidator(result_format=ResultFormat.COMPLETE)
        issues = validator.validate(null_heavy_lf)

        assert len(issues) > 0
        for issue in issues:
            assert issue.details is not None

    def test_count_consistent_across_formats(self, null_heavy_lf):
        """Same issue count regardless of result_format."""
        formats = [
            ResultFormat.BOOLEAN_ONLY,
            ResultFormat.BASIC,
            ResultFormat.SUMMARY,
            ResultFormat.COMPLETE,
        ]
        counts = []
        for fmt in formats:
            validator = NullValidator(result_format=fmt)
            issues = validator.validate(null_heavy_lf)
            counts.append(len(issues))

        assert all(c == counts[0] for c in counts), \
            f"Issue count varies by format: {dict(zip(formats, counts))}"

    def test_issue_counts_consistent_across_formats(self, null_heavy_lf):
        """Individual issue counts match across all formats."""
        formats = [
            ResultFormat.BOOLEAN_ONLY,
            ResultFormat.BASIC,
            ResultFormat.SUMMARY,
            ResultFormat.COMPLETE,
        ]
        all_issues = {}
        for fmt in formats:
            validator = NullValidator(result_format=fmt)
            issues = validator.validate(null_heavy_lf)
            all_issues[fmt] = {i.column: i.count for i in issues}

        # All formats should have same column→count mapping
        base = all_issues[ResultFormat.BOOLEAN_ONLY]
        for fmt, issue_map in all_issues.items():
            assert issue_map == base, \
                f"{fmt.value} has different counts: {issue_map} vs {base}"

    def test_severity_consistent_across_formats(self, null_heavy_lf):
        """Severity should be the same regardless of format."""
        formats = [ResultFormat.BOOLEAN_ONLY, ResultFormat.COMPLETE]
        all_severities = {}
        for fmt in formats:
            validator = NullValidator(result_format=fmt)
            issues = validator.validate(null_heavy_lf)
            all_severities[fmt] = {i.column: i.severity for i in issues}

        assert all_severities[ResultFormat.BOOLEAN_ONLY] == all_severities[ResultFormat.COMPLETE]


# ============================================================================
# 5. NotNullValidator Result Format Tests
# ============================================================================

class TestNotNullValidatorResultFormat:
    """Tests for NotNullValidator with result_format."""

    def test_boolean_only(self, null_heavy_lf):
        validator = NotNullValidator(result_format=ResultFormat.BOOLEAN_ONLY)
        issues = validator.validate(null_heavy_lf)

        # col_a and col_b have nulls
        assert len(issues) >= 2
        for issue in issues:
            assert issue.details is None
            assert issue.sample_values is None

    def test_basic(self, null_heavy_lf):
        validator = NotNullValidator(result_format=ResultFormat.BASIC)
        issues = validator.validate(null_heavy_lf)

        assert len(issues) >= 2
        for issue in issues:
            assert issue.details is not None


# ============================================================================
# 6. CompletenessRatioValidator Result Format Tests
# ============================================================================

class TestCompletenessRatioResultFormat:
    """Tests for CompletenessRatioValidator with result_format."""

    def test_boolean_only(self, null_heavy_lf):
        validator = CompletenessRatioValidator(
            min_ratio=0.5, result_format=ResultFormat.BOOLEAN_ONLY
        )
        issues = validator.validate(null_heavy_lf)

        # col_b is 100% null (completeness 0.0 < 0.5)
        assert len(issues) >= 1
        for issue in issues:
            # CompletenessRatioValidator uses custom build_issues_from_results,
            # details are always built there (by design, not controlled by format)
            assert issue.count >= 0

    def test_count_consistent(self, null_heavy_lf):
        """Consistency check across formats."""
        counts_by_format = {}
        for fmt in [ResultFormat.BOOLEAN_ONLY, ResultFormat.SUMMARY]:
            validator = CompletenessRatioValidator(min_ratio=0.5, result_format=fmt)
            issues = validator.validate(null_heavy_lf)
            counts_by_format[fmt] = {i.column: i.count for i in issues}

        assert counts_by_format[ResultFormat.BOOLEAN_ONLY] == counts_by_format[ResultFormat.SUMMARY]


# ============================================================================
# 7. BetweenValidator Result Format Tests
# ============================================================================

class TestBetweenValidatorResultFormat:
    """Tests for BetweenValidator with result_format."""

    def test_boolean_only_no_details(self, sample_lf):
        validator = BetweenValidator(
            min_value=0, max_value=100,
            result_format=ResultFormat.BOOLEAN_ONLY,
        )
        issues = validator.validate(sample_lf)

        for issue in issues:
            assert issue.details is None
            assert issue.expected is None

    def test_basic_has_details(self, sample_lf):
        validator = BetweenValidator(
            min_value=0, max_value=100,
            result_format=ResultFormat.BASIC,
        )
        issues = validator.validate(sample_lf)

        for issue in issues:
            assert issue.details is not None

    def test_filter_expr_enables_samples(self, sample_lf):
        """With BASIC+, filter_expr enables sample collection."""
        validator = BetweenValidator(
            min_value=0, max_value=100,
            result_format=ResultFormat.BASIC,
        )
        issues = validator.validate(sample_lf)

        # At BASIC level, sample_values should be collected via filter_expr
        has_samples = any(i.sample_values is not None for i in issues)
        assert has_samples or len(issues) == 0

    def test_count_consistent_across_formats(self, sample_lf):
        counts = {}
        for fmt in [ResultFormat.BOOLEAN_ONLY, ResultFormat.COMPLETE]:
            validator = BetweenValidator(
                min_value=0, max_value=100, result_format=fmt,
            )
            issues = validator.validate(sample_lf)
            counts[fmt] = {i.column: i.count for i in issues}

        assert counts[ResultFormat.BOOLEAN_ONLY] == counts[ResultFormat.COMPLETE]


# ============================================================================
# 8. RangeValidator Result Format Tests
# ============================================================================

class TestRangeValidatorResultFormat:
    """Tests for RangeValidator with result_format."""

    def test_boolean_only(self, sample_lf):
        validator = RangeValidator(result_format=ResultFormat.BOOLEAN_ONLY)
        issues = validator.validate(sample_lf)

        for issue in issues:
            assert issue.details is None

    def test_filter_expr_in_spec(self, sample_lf):
        """RangeValidator specs should have filter_expr."""
        validator = RangeValidator()
        columns = validator._get_numeric_columns(sample_lf)
        specs = validator.get_validation_exprs(sample_lf, columns)

        for spec in specs:
            assert spec.filter_expr is not None


# ============================================================================
# 9. PositiveValidator & NonNegativeValidator Result Format Tests
# ============================================================================

class TestPositiveValidatorResultFormat:
    """Tests for PositiveValidator with result_format."""

    def test_filter_expr_in_spec(self):
        lf = pl.DataFrame({"amount": [10, -5, 0, 20, -3]}).lazy()
        validator = PositiveValidator()
        columns = validator._get_numeric_columns(lf)
        specs = validator.get_validation_exprs(lf, columns)

        assert len(specs) == 1
        assert specs[0].filter_expr is not None

    def test_boolean_only(self):
        lf = pl.DataFrame({"amount": [10, -5, 0, 20, -3]}).lazy()
        validator = PositiveValidator(result_format=ResultFormat.BOOLEAN_ONLY)
        issues = validator.validate(lf)

        assert len(issues) == 1
        assert issues[0].details is None


class TestNonNegativeValidatorResultFormat:
    """Tests for NonNegativeValidator with result_format."""

    def test_filter_expr_in_spec(self):
        lf = pl.DataFrame({"amount": [10, -5, 0, 20, -3]}).lazy()
        validator = NonNegativeValidator()
        columns = validator._get_numeric_columns(lf)
        specs = validator.get_validation_exprs(lf, columns)

        assert len(specs) == 1
        assert specs[0].filter_expr is not None


# ============================================================================
# 10. UniqueValidator Result Format Tests (Non-expression)
# ============================================================================

class TestUniqueValidatorResultFormat:
    """Tests for UniqueValidator with result_format."""

    def test_boolean_only_no_details(self, duplicate_lf):
        validator = UniqueValidator(result_format=ResultFormat.BOOLEAN_ONLY)
        issues = validator.validate(duplicate_lf)

        assert len(issues) >= 1
        for issue in issues:
            assert issue.details is None
            assert issue.sample_values is None

    def test_basic_has_details_and_samples(self, duplicate_lf):
        validator = UniqueValidator(result_format=ResultFormat.BASIC)
        issues = validator.validate(duplicate_lf)

        assert len(issues) >= 1
        for issue in issues:
            assert issue.details is not None
            # Should have sample duplicate values
            assert issue.sample_values is not None

    def test_count_consistent(self, duplicate_lf):
        counts = {}
        for fmt in [ResultFormat.BOOLEAN_ONLY, ResultFormat.BASIC, ResultFormat.COMPLETE]:
            validator = UniqueValidator(result_format=fmt)
            issues = validator.validate(duplicate_lf)
            counts[fmt] = {i.column: i.count for i in issues}

        assert counts[ResultFormat.BOOLEAN_ONLY] == counts[ResultFormat.BASIC]
        assert counts[ResultFormat.BASIC] == counts[ResultFormat.COMPLETE]


# ============================================================================
# 11. UniqueRatioValidator Result Format Tests
# ============================================================================

class TestUniqueRatioValidatorResultFormat:
    """Tests for UniqueRatioValidator with result_format."""

    def test_boolean_only(self):
        lf = pl.DataFrame({"col": [1, 1, 1, 2, 2]}).lazy()
        validator = UniqueRatioValidator(
            min_ratio=0.9, result_format=ResultFormat.BOOLEAN_ONLY
        )
        issues = validator.validate(lf)

        assert len(issues) >= 1
        for issue in issues:
            assert issue.details is None
            assert issue.expected is None
            assert issue.actual is None


# ============================================================================
# 12. DistinctCountValidator Result Format Tests
# ============================================================================

class TestDistinctCountValidatorResultFormat:
    """Tests for DistinctCountValidator with result_format."""

    def test_boolean_only(self):
        lf = pl.DataFrame({"col": [1, 1, 2, 2, 3]}).lazy()
        validator = DistinctCountValidator(
            min_count=10, result_format=ResultFormat.BOOLEAN_ONLY
        )
        issues = validator.validate(lf)

        assert len(issues) >= 1
        for issue in issues:
            assert issue.details is None
            assert issue.expected is None


# ============================================================================
# 13. ColumnTypeValidator Result Format Tests (Non-expression)
# ============================================================================

class TestColumnTypeValidatorResultFormat:
    """Tests for ColumnTypeValidator with result_format."""

    def test_boolean_only_no_details(self):
        lf = pl.DataFrame({"age": ["not_a_number"]}).lazy()
        validator = ColumnTypeValidator(
            expected_types={"age": "int"},
            result_format=ResultFormat.BOOLEAN_ONLY,
        )
        issues = validator.validate(lf)

        assert len(issues) == 1
        assert issues[0].details is None
        assert issues[0].expected is None
        assert issues[0].actual is None

    def test_basic_has_details(self):
        lf = pl.DataFrame({"age": ["not_a_number"]}).lazy()
        validator = ColumnTypeValidator(
            expected_types={"age": "int"},
            result_format=ResultFormat.BASIC,
        )
        issues = validator.validate(lf)

        assert len(issues) == 1
        assert issues[0].details is not None
        assert "int" in issues[0].details.lower() or "expected" in issues[0].details.lower()

    def test_missing_column_boolean_only(self):
        lf = pl.DataFrame({"name": ["Alice"]}).lazy()
        validator = ColumnTypeValidator(
            expected_types={"missing_col": "int"},
            result_format=ResultFormat.BOOLEAN_ONLY,
        )
        issues = validator.validate(lf)

        assert len(issues) == 1
        assert issues[0].issue_type == "missing_column"
        assert issues[0].details is None


# ============================================================================
# 14. ExpressionBatchExecutor Result Format Tests
# ============================================================================

class TestExpressionBatchExecutorResultFormat:
    """Tests for ExpressionBatchExecutor with result_format parameter."""

    def test_batch_boolean_only(self, null_heavy_lf):
        executor = ExpressionBatchExecutor()
        executor.add_validator(NullValidator())
        executor.add_validator(NotNullValidator())

        issues = executor.execute(null_heavy_lf, result_format="boolean_only")

        for issue in issues:
            assert issue.details is None
            assert issue.sample_values is None

    def test_batch_basic(self, null_heavy_lf):
        executor = ExpressionBatchExecutor()
        executor.add_validator(NullValidator())
        executor.add_validator(NotNullValidator())

        issues = executor.execute(null_heavy_lf, result_format="basic")

        for issue in issues:
            assert issue.details is not None

    def test_batch_count_consistent(self, null_heavy_lf):
        """Same counts regardless of result_format in batch execution."""
        counts_by_format = {}
        for fmt in ["boolean_only", "basic", "summary", "complete"]:
            executor = ExpressionBatchExecutor()
            executor.add_validator(NullValidator())
            executor.add_validator(NotNullValidator())
            issues = executor.execute(null_heavy_lf, result_format=fmt)
            counts_by_format[fmt] = sorted(
                [(i.column, i.issue_type, i.count) for i in issues]
            )

        base = counts_by_format["boolean_only"]
        for fmt, counts in counts_by_format.items():
            assert counts == base, \
                f"Format {fmt} has different counts: {counts} vs {base}"

    def test_batch_override_vs_individual_format(self, null_heavy_lf):
        """Batch format override should take precedence over individual configs."""
        # Validator configured with COMPLETE
        validator = NullValidator(result_format=ResultFormat.COMPLETE)

        executor = ExpressionBatchExecutor()
        executor.add_validator(validator)

        # But batch overrides to BOOLEAN_ONLY
        issues = executor.execute(null_heavy_lf, result_format="boolean_only")

        for issue in issues:
            assert issue.details is None  # Override took effect

    def test_batch_mixed_expression_and_traditional(self, sample_lf):
        """Batch with both expression-based and traditional validators."""
        executor = ExpressionBatchExecutor()
        executor.add_validator(NullValidator(result_format=ResultFormat.BASIC))
        executor.add_validator(UniqueValidator(result_format=ResultFormat.BASIC))

        issues = executor.execute(sample_lf)
        assert len(issues) >= 0  # Should not crash


# ============================================================================
# 15. Report Serialization with ResultFormat
# ============================================================================

class TestReportResultFormat:
    """Tests for Report serialization with result_format."""

    def _make_issues(self) -> list[ValidationIssue]:
        return [
            ValidationIssue(
                column="col_a",
                issue_type="null",
                count=5,
                severity=Severity.MEDIUM,
                details="50.0% of values are null",
                expected=None,
                sample_values=[None, None],
            ),
        ]

    def test_boolean_only_to_dict(self):
        report = Report(
            issues=self._make_issues(),
            result_format=ResultFormat.BOOLEAN_ONLY,
        )
        d = report.to_dict()

        assert d["result_format"] == "boolean_only"
        assert "success" in d
        # success is True because the only issue has MEDIUM severity
        # (success = no HIGH+ issues)
        assert d["success"] is True
        # BOOLEAN_ONLY issues should have minimal fields
        for issue in d["issues"]:
            assert "column" in issue
            assert "issue_type" in issue
            assert "count" in issue
            assert "severity" in issue

    def test_basic_to_dict(self):
        report = Report(
            issues=self._make_issues(),
            result_format=ResultFormat.BASIC,
        )
        d = report.to_dict()

        assert d["result_format"] == "basic"
        # BASIC+ should have full issue details
        for issue in d["issues"]:
            assert "details" in issue

    def test_summary_to_dict(self):
        report = Report(
            issues=self._make_issues(),
            result_format=ResultFormat.SUMMARY,
        )
        d = report.to_dict()

        assert d["result_format"] == "summary"

    def test_no_issues_boolean_only(self):
        report = Report(
            issues=[],
            result_format=ResultFormat.BOOLEAN_ONLY,
        )
        d = report.to_dict()

        assert d["success"] is True
        assert d["issue_count"] == 0

    def test_report_str_boolean_only(self):
        """String representation should work for all formats."""
        report = Report(
            issues=self._make_issues(),
            result_format=ResultFormat.BOOLEAN_ONLY,
        )
        s = str(report)
        assert "Truthound Report" in s

    def test_filter_by_severity_preserves_format(self):
        report = Report(
            issues=self._make_issues(),
            result_format=ResultFormat.COMPLETE,
        )
        filtered = report.filter_by_severity(Severity.HIGH)
        assert filtered.result_format == ResultFormat.COMPLETE


# ============================================================================
# 16. ValidationExpressionSpec filter_expr Tests
# ============================================================================

class TestValidationExpressionSpecFilterExpr:
    """Tests that filter_expr is properly set on expression specs."""

    def test_null_validator_has_filter_expr(self):
        lf = pl.DataFrame({"col": [1, None, 3]}).lazy()
        validator = NullValidator()
        specs = validator.get_validation_exprs(lf, ["col"])
        assert len(specs) == 1
        assert specs[0].filter_expr is not None

    def test_not_null_validator_has_filter_expr(self):
        lf = pl.DataFrame({"col": [1, None, 3]}).lazy()
        validator = NotNullValidator()
        specs = validator.get_validation_exprs(lf, ["col"])
        assert len(specs) == 1
        assert specs[0].filter_expr is not None

    def test_completeness_ratio_validator_has_filter_expr(self):
        lf = pl.DataFrame({"col": [1, None, 3]}).lazy()
        validator = CompletenessRatioValidator(min_ratio=0.9)
        specs = validator.get_validation_exprs(lf, ["col"])
        assert len(specs) == 1
        assert specs[0].filter_expr is not None

    def test_between_validator_has_filter_expr(self):
        lf = pl.DataFrame({"col": [1, 5, 10]}).lazy()
        validator = BetweenValidator(min_value=0, max_value=100)
        specs = validator.get_validation_exprs(lf, ["col"])
        assert len(specs) == 1
        assert specs[0].filter_expr is not None

    def test_positive_validator_has_filter_expr(self):
        lf = pl.DataFrame({"col": [1, -5, 10]}).lazy()
        validator = PositiveValidator()
        specs = validator.get_validation_exprs(lf, ["col"])
        assert len(specs) == 1
        assert specs[0].filter_expr is not None

    def test_non_negative_validator_has_filter_expr(self):
        lf = pl.DataFrame({"col": [1, -5, 10]}).lazy()
        validator = NonNegativeValidator()
        specs = validator.get_validation_exprs(lf, ["col"])
        assert len(specs) == 1
        assert specs[0].filter_expr is not None


# ============================================================================
# 17. End-to-End Cross-Format Consistency Tests
# ============================================================================

class TestCrossFormatConsistency:
    """Ensures validation results are consistent across all format levels."""

    @pytest.mark.parametrize("validator_cls,kwargs", [
        (NullValidator, {}),
        (NotNullValidator, {}),
        (CompletenessRatioValidator, {"min_ratio": 0.5}),
    ])
    def test_completeness_validators_consistent(self, null_heavy_lf, validator_cls, kwargs):
        """All completeness validators produce same pass/fail across formats."""
        results = {}
        for fmt in ResultFormat:
            validator = validator_cls(**kwargs, result_format=fmt)
            issues = validator.validate(null_heavy_lf)
            results[fmt] = {(i.column, i.issue_type) for i in issues}

        base = results[ResultFormat.BOOLEAN_ONLY]
        for fmt, result_set in results.items():
            assert result_set == base, \
                f"{validator_cls.__name__} inconsistent at {fmt.value}: {result_set} vs {base}"

    def test_range_validators_consistent(self, sample_lf):
        """Range validators produce same pass/fail across formats."""
        results = {}
        for fmt in ResultFormat:
            validator = BetweenValidator(
                min_value=0, max_value=100, result_format=fmt
            )
            issues = validator.validate(sample_lf)
            results[fmt] = {(i.column, i.issue_type) for i in issues}

        base = results[ResultFormat.BOOLEAN_ONLY]
        for fmt, result_set in results.items():
            assert result_set == base

    def test_unique_validator_consistent(self, duplicate_lf):
        """UniqueValidator produces same pass/fail across formats."""
        results = {}
        for fmt in ResultFormat:
            validator = UniqueValidator(result_format=fmt)
            issues = validator.validate(duplicate_lf)
            results[fmt] = {(i.column, i.issue_type) for i in issues}

        base = results[ResultFormat.BOOLEAN_ONLY]
        for fmt, result_set in results.items():
            assert result_set == base


# ============================================================================
# 18. Detail Monotonicity Tests
# ============================================================================

class TestDetailMonotonicity:
    """Verify that higher format levels include at least as much info as lower levels."""

    def test_detail_fields_monotonically_increase(self, null_heavy_lf):
        """Higher formats should have details when lower formats don't."""
        boolean_issues = NullValidator(
            result_format=ResultFormat.BOOLEAN_ONLY
        ).validate(null_heavy_lf)

        basic_issues = NullValidator(
            result_format=ResultFormat.BASIC
        ).validate(null_heavy_lf)

        # BOOLEAN_ONLY has no details, BASIC has details
        for issue in boolean_issues:
            assert issue.details is None

        for issue in basic_issues:
            assert issue.details is not None

    def test_samples_only_at_basic_and_above(self, null_heavy_lf):
        """Sample values should only appear at BASIC+ levels."""
        boolean_issues = NullValidator(
            result_format=ResultFormat.BOOLEAN_ONLY
        ).validate(null_heavy_lf)

        basic_issues = NullValidator(
            result_format=ResultFormat.BASIC
        ).validate(null_heavy_lf)

        # BOOLEAN_ONLY should have no samples
        for issue in boolean_issues:
            assert issue.sample_values is None

        # BASIC should have samples (if filter_expr is set and issues exist)
        if basic_issues:
            has_any_samples = any(i.sample_values is not None for i in basic_issues)
            assert has_any_samples, "BASIC should collect samples via filter_expr"


# ============================================================================
# 19. Edge Cases
# ============================================================================

class TestResultFormatEdgeCases:
    """Edge cases for result_format handling."""

    def test_empty_dataframe(self):
        """Empty data should produce no issues at any format level."""
        lf = pl.DataFrame({"col": []}).cast({"col": pl.Int64}).lazy()

        for fmt in ResultFormat:
            validator = NullValidator(result_format=fmt)
            issues = validator.validate(lf)
            assert len(issues) == 0

    def test_all_null_column(self):
        """Column with all nulls."""
        lf = pl.DataFrame({"col": [None, None, None, None, None]}).lazy()

        for fmt in ResultFormat:
            validator = NullValidator(result_format=fmt)
            issues = validator.validate(lf)
            assert len(issues) == 1
            assert issues[0].count == 5

    def test_no_issues_data(self):
        """Clean data should produce no issues at any level."""
        lf = pl.DataFrame({"col": [1, 2, 3, 4, 5]}).lazy()

        for fmt in ResultFormat:
            validator = NullValidator(result_format=fmt)
            issues = validator.validate(lf)
            assert len(issues) == 0

    def test_single_row(self):
        """Single row with null."""
        lf = pl.DataFrame({"col": [None]}).lazy()

        validator = NullValidator(result_format=ResultFormat.COMPLETE)
        issues = validator.validate(lf)
        assert len(issues) == 1
        assert issues[0].count == 1

    def test_validator_config_result_format_passthrough(self):
        """Validators should correctly inherit result_format from config."""
        config = ValidatorConfig(result_format=ResultFormat.BOOLEAN_ONLY)
        validator = NullValidator(config=config)
        assert validator._get_result_format_config().format == ResultFormat.BOOLEAN_ONLY

    def test_batch_executor_none_result_format(self, sample_lf):
        """Passing None as result_format should use each validator's config."""
        executor = ExpressionBatchExecutor()
        executor.add_validator(NullValidator(result_format=ResultFormat.BASIC))
        issues = executor.execute(sample_lf, result_format=None)

        # Should use the validator's own BASIC format
        for issue in issues:
            if issue.issue_type == "null":
                assert issue.details is not None
