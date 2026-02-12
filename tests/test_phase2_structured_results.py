"""Comprehensive tests for PHASE 2: Structured Validation Results.

Tests cover:
1. ValidationDetail dataclass (types.py)
2. ValidationIssue PHASE 2 extensions (base.py)
3. ExpressionValidatorMixin structured result population
4. Enrichment phases (samples, value counts, full results)
5. Debug query generation
6. ReportStatistics (report.py)
7. Non-expression validator structured results
8. Cross-format consistency with structured results
9. Serialization (to_dict / to_json)
"""

import json

import polars as pl
import pytest

from truthound.types import (
    ResultFormat,
    ResultFormatConfig,
    Severity,
    ValidationDetail,
)
from truthound.validators.base import (
    ExpressionBatchExecutor,
    ExpressionValidatorMixin,
    ValidationExpressionSpec,
    ValidationIssue,
    Validator,
    ValidatorConfig,
)
from truthound.report import Report, ReportStatistics


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_lf():
    """LazyFrame with nulls, out-of-range, and duplicate values."""
    return pl.DataFrame({
        "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "name": ["Alice", "Bob", None, "Dave", None, "Frank", "Grace", "Heidi", "Ivan", "Judy"],
        "age": [25, -5, 30, 150, 45, 60, 200, 18, 35, 42],
        "score": [85, 92, 78, None, 105, 63, 71, 88, 99, -10],
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
# 1. ValidationDetail Dataclass Tests
# ============================================================================

class TestValidationDetail:
    """Tests for ValidationDetail dataclass."""

    def test_default_values(self):
        detail = ValidationDetail()
        assert detail.element_count == 0
        assert detail.missing_count == 0
        assert detail.observed_value is None
        assert detail.unexpected_count == 0
        assert detail.unexpected_percent == 0.0
        assert detail.unexpected_percent_nonmissing == 0.0
        assert detail.partial_unexpected_list is None
        assert detail.partial_unexpected_counts is None
        assert detail.partial_unexpected_index_list is None
        assert detail.unexpected_list is None
        assert detail.unexpected_index_list is None
        assert detail.unexpected_rows is None
        assert detail.debug_query is None

    def test_from_aggregates_basic(self):
        detail = ValidationDetail.from_aggregates(
            element_count=100,
            missing_count=10,
            unexpected_count=5,
            observed_value=5,
        )
        assert detail.element_count == 100
        assert detail.missing_count == 10
        assert detail.unexpected_count == 5
        assert detail.observed_value == 5
        assert detail.unexpected_percent == pytest.approx(5.0)  # 5/100*100
        assert detail.unexpected_percent_nonmissing == pytest.approx(5 / 90 * 100)

    def test_from_aggregates_zero_total(self):
        detail = ValidationDetail.from_aggregates(element_count=0)
        assert detail.unexpected_percent == 0.0
        assert detail.unexpected_percent_nonmissing == 0.0

    def test_from_aggregates_all_missing(self):
        detail = ValidationDetail.from_aggregates(
            element_count=100,
            missing_count=100,
            unexpected_count=0,
        )
        assert detail.unexpected_percent == 0.0
        assert detail.unexpected_percent_nonmissing == 0.0

    def test_to_dict_excludes_none_and_zero(self):
        detail = ValidationDetail()
        d = detail.to_dict()
        # Zero ints and None values should be excluded by default
        assert len(d) == 0

    def test_to_dict_includes_zeros_when_requested(self):
        detail = ValidationDetail()
        d = detail.to_dict(include_zeros=True)
        assert "element_count" in d
        assert "missing_count" in d
        assert "unexpected_count" in d

    def test_to_dict_with_data(self):
        detail = ValidationDetail.from_aggregates(
            element_count=100,
            missing_count=10,
            unexpected_count=5,
            observed_value=5,
        )
        detail.partial_unexpected_list = [1, 2, 3]
        d = detail.to_dict()

        assert d["element_count"] == 100
        assert d["missing_count"] == 10
        assert d["unexpected_count"] == 5
        assert d["observed_value"] == 5
        assert "unexpected_percent" in d
        assert d["partial_unexpected_list"] == [1, 2, 3]

    def test_to_dict_with_dataframe(self):
        detail = ValidationDetail()
        detail.unexpected_rows = pl.DataFrame({"col": [1, 2, 3]})
        d = detail.to_dict()
        assert d["unexpected_rows"] == [{"col": 1}, {"col": 2}, {"col": 3}]

    def test_to_dict_excludes_none_fields(self):
        detail = ValidationDetail.from_aggregates(
            element_count=100,
            unexpected_count=5,
        )
        d = detail.to_dict()
        assert "partial_unexpected_list" not in d
        assert "partial_unexpected_counts" not in d
        assert "unexpected_list" not in d
        assert "debug_query" not in d


# ============================================================================
# 2. ValidationIssue PHASE 2 Extensions
# ============================================================================

class TestValidationIssuePHASE2:
    """Tests for ValidationIssue PHASE 2 fields and accessors."""

    def test_new_fields_defaults(self):
        issue = ValidationIssue(
            column="col",
            issue_type="test",
            count=5,
            severity=Severity.MEDIUM,
        )
        assert issue.result is None
        assert issue.validator_name is None
        assert issue.success is False

    def test_new_fields_populated(self):
        detail = ValidationDetail.from_aggregates(element_count=100, unexpected_count=5)
        issue = ValidationIssue(
            column="col",
            issue_type="test",
            count=5,
            severity=Severity.MEDIUM,
            result=detail,
            validator_name="test_validator",
            success=False,
        )
        assert issue.result is detail
        assert issue.validator_name == "test_validator"
        assert issue.success is False

    def test_unexpected_percent_accessor(self):
        detail = ValidationDetail.from_aggregates(element_count=100, unexpected_count=5)
        issue = ValidationIssue(
            column="col", issue_type="test", count=5,
            severity=Severity.MEDIUM, result=detail,
        )
        assert issue.unexpected_percent == pytest.approx(5.0)

    def test_unexpected_percent_no_result(self):
        issue = ValidationIssue(
            column="col", issue_type="test", count=5,
            severity=Severity.MEDIUM,
        )
        assert issue.unexpected_percent is None

    def test_unexpected_rows_accessor(self):
        detail = ValidationDetail()
        detail.unexpected_rows = pl.DataFrame({"x": [1, 2]})
        issue = ValidationIssue(
            column="col", issue_type="test", count=2,
            severity=Severity.LOW, result=detail,
        )
        assert issue.unexpected_rows is not None
        assert len(issue.unexpected_rows) == 2

    def test_debug_query_accessor(self):
        detail = ValidationDetail()
        detail.debug_query = 'df.filter(pl.col("x").is_null()).select("x")'
        issue = ValidationIssue(
            column="col", issue_type="test", count=1,
            severity=Severity.LOW, result=detail,
        )
        assert 'filter' in issue.debug_query

    def test_to_dict_includes_phase2_fields(self):
        detail = ValidationDetail.from_aggregates(element_count=100, unexpected_count=5)
        issue = ValidationIssue(
            column="col", issue_type="null", count=5,
            severity=Severity.MEDIUM,
            result=detail,
            validator_name="null_validator",
            success=False,
        )
        d = issue.to_dict()

        assert d["success"] is False
        assert d["validator_name"] == "null_validator"
        assert "result" in d
        assert d["result"]["element_count"] == 100
        assert d["result"]["unexpected_count"] == 5

    def test_to_dict_backward_compatible(self):
        """Old-style issues without PHASE 2 fields still serialize correctly."""
        issue = ValidationIssue(
            column="col", issue_type="null", count=5,
            severity=Severity.MEDIUM,
            details="5 nulls",
            sample_values=[None, None],
        )
        d = issue.to_dict()

        assert "column" in d
        assert "issue_type" in d
        assert "count" in d
        assert "severity" in d
        assert d["details"] == "5 nulls"
        assert d["sample_values"] == [None, None]
        assert d["success"] is False  # default


# ============================================================================
# 3. Expression Validator Structured Result Population
# ============================================================================

class TestExpressionValidatorStructuredResults:
    """Tests that expression validators populate structured results."""

    def test_null_validator_populates_result(self, null_heavy_lf):
        from truthound.validators.completeness.null import NullValidator

        validator = NullValidator(result_format=ResultFormat.BASIC)
        issues = validator.validate(null_heavy_lf)

        assert len(issues) > 0
        for issue in issues:
            assert issue.result is not None
            assert issue.result.element_count == 10
            assert issue.result.unexpected_count > 0
            assert issue.result.unexpected_percent > 0
            assert issue.validator_name == "null"
            assert issue.success is False

    def test_boolean_only_has_minimal_result(self, null_heavy_lf):
        from truthound.validators.completeness.null import NullValidator

        validator = NullValidator(result_format=ResultFormat.BOOLEAN_ONLY)
        issues = validator.validate(null_heavy_lf)

        assert len(issues) > 0
        for issue in issues:
            assert issue.result is not None
            assert issue.result.element_count == 10
            # BOOLEAN_ONLY should NOT have observed_value
            assert issue.result.observed_value is None

    def test_basic_has_observed_value(self, null_heavy_lf):
        from truthound.validators.completeness.null import NullValidator

        validator = NullValidator(result_format=ResultFormat.BASIC)
        issues = validator.validate(null_heavy_lf)

        for issue in issues:
            assert issue.result is not None
            # BASIC+ should have observed_value
            assert issue.result.observed_value is not None

    def test_between_validator_populates_result(self, sample_lf):
        from truthound.validators.distribution.range import BetweenValidator

        validator = BetweenValidator(
            min_value=0, max_value=100,
            result_format=ResultFormat.BASIC,
        )
        issues = validator.validate(sample_lf)

        for issue in issues:
            assert issue.result is not None
            assert issue.result.element_count > 0
            assert issue.validator_name == "between"
            assert issue.success is False


# ============================================================================
# 4. Enrichment Phases
# ============================================================================

class TestEnrichmentPhases:
    """Tests for the enrichment phases of expression validators."""

    def test_basic_collects_samples(self, null_heavy_lf):
        from truthound.validators.completeness.null import NullValidator

        validator = NullValidator(result_format=ResultFormat.BASIC)
        issues = validator.validate(null_heavy_lf)

        for issue in issues:
            if issue.count > 0:
                # Legacy field
                assert issue.sample_values is not None
                # Structured field
                assert issue.result is not None
                assert issue.result.partial_unexpected_list is not None

    def test_summary_collects_value_counts(self, null_heavy_lf):
        from truthound.validators.completeness.null import NullValidator

        validator = NullValidator(result_format=ResultFormat.SUMMARY)
        issues = validator.validate(null_heavy_lf)

        for issue in issues:
            if issue.count > 0 and issue.result is not None:
                # SUMMARY level should have value counts
                assert issue.result.partial_unexpected_counts is not None
                # Each entry should be a dict with value and count
                for vc in issue.result.partial_unexpected_counts:
                    assert "value" in vc
                    assert "count" in vc

    def test_complete_collects_full_results(self):
        from truthound.validators.distribution.range import BetweenValidator

        lf = pl.DataFrame({
            "val": [1, 2, 150, 200, -5, 50, 300],
        }).lazy()

        validator = BetweenValidator(
            min_value=0, max_value=100,
            result_format=ResultFormatConfig(
                format=ResultFormat.COMPLETE,
                include_unexpected_rows=True,
                return_debug_query=True,
            ),
        )
        issues = validator.validate(lf)

        assert len(issues) == 1
        issue = issues[0]
        assert issue.result is not None

        # unexpected_list should contain the actual bad values
        assert issue.result.unexpected_list is not None
        assert len(issue.result.unexpected_list) > 0

        # unexpected_index_list should contain row indices
        assert issue.result.unexpected_index_list is not None
        assert len(issue.result.unexpected_index_list) > 0

        # unexpected_rows should be a DataFrame
        assert issue.result.unexpected_rows is not None
        assert isinstance(issue.result.unexpected_rows, pl.DataFrame)

        # debug_query should be a string
        assert issue.result.debug_query is not None
        assert "filter" in issue.result.debug_query

    def test_complete_without_include_unexpected_rows(self):
        from truthound.validators.distribution.range import BetweenValidator

        lf = pl.DataFrame({"val": [1, 200, -5]}).lazy()

        validator = BetweenValidator(
            min_value=0, max_value=100,
            result_format=ResultFormatConfig(
                format=ResultFormat.COMPLETE,
                include_unexpected_rows=False,
            ),
        )
        issues = validator.validate(lf)
        assert len(issues) == 1
        # unexpected_rows should NOT be set
        assert issues[0].result.unexpected_rows is None

    def test_boolean_only_no_enrichment(self, null_heavy_lf):
        from truthound.validators.completeness.null import NullValidator

        validator = NullValidator(result_format=ResultFormat.BOOLEAN_ONLY)
        issues = validator.validate(null_heavy_lf)

        for issue in issues:
            assert issue.sample_values is None
            assert issue.result is not None
            assert issue.result.partial_unexpected_list is None
            assert issue.result.partial_unexpected_counts is None
            assert issue.result.unexpected_list is None


# ============================================================================
# 5. Debug Query Generation
# ============================================================================

class TestDebugQueryGeneration:
    """Tests for the debug query generation feature."""

    def test_generate_debug_query_with_filter_expr(self):
        spec = ValidationExpressionSpec(
            column="age",
            validator_name="between",
            issue_type="out_of_range",
            count_expr=pl.lit(0),
            filter_expr=pl.col("age").is_between(0, 100).not_(),
        )
        query = ExpressionValidatorMixin._generate_debug_query(spec)
        assert "filter" in query
        assert '"age"' in query

    def test_generate_debug_query_without_filter_expr(self):
        spec = ValidationExpressionSpec(
            column="col",
            validator_name="test",
            issue_type="test_issue",
            count_expr=pl.lit(0),
        )
        query = ExpressionValidatorMixin._generate_debug_query(spec)
        assert "test" in query
        assert "col" in query

    def test_generate_debug_query_with_sample_columns(self):
        spec = ValidationExpressionSpec(
            column="age",
            validator_name="between",
            issue_type="out_of_range",
            count_expr=pl.lit(0),
            filter_expr=pl.col("age") > 100,
            sample_columns=["age", "name"],
        )
        query = ExpressionValidatorMixin._generate_debug_query(spec)
        assert '"age"' in query
        assert '"name"' in query


# ============================================================================
# 6. ReportStatistics Tests
# ============================================================================

class TestReportStatistics:
    """Tests for ReportStatistics dataclass and Report integration."""

    def test_empty_report_statistics(self):
        report = Report(issues=[])
        assert report.statistics is not None
        assert report.statistics.unsuccessful_validations == 0
        assert report.statistics.issues_by_severity == {}
        assert report.success is True

    def test_report_statistics_computed(self):
        issues = [
            ValidationIssue(
                column="col_a", issue_type="null", count=5,
                severity=Severity.HIGH, validator_name="null",
            ),
            ValidationIssue(
                column="col_a", issue_type="range", count=3,
                severity=Severity.MEDIUM, validator_name="range",
            ),
            ValidationIssue(
                column="col_b", issue_type="null", count=10,
                severity=Severity.CRITICAL, validator_name="null",
            ),
        ]
        report = Report(issues=issues)

        stats = report.statistics
        assert stats is not None
        assert stats.unsuccessful_validations == 3
        assert stats.issues_by_severity == {"high": 1, "medium": 1, "critical": 1}
        assert stats.issues_by_column == {"col_a": 2, "col_b": 1}
        assert stats.issues_by_validator == {"null": 2, "range": 1}
        assert stats.issues_by_type == {"null": 2, "range": 1}

    def test_report_statistics_most_problematic_columns(self):
        issues = [
            ValidationIssue(column="col_a", issue_type="null", count=5, severity=Severity.HIGH),
            ValidationIssue(column="col_a", issue_type="range", count=3, severity=Severity.MEDIUM),
            ValidationIssue(column="col_b", issue_type="null", count=10, severity=Severity.CRITICAL),
        ]
        report = Report(issues=issues)

        assert report.statistics.most_problematic_columns[0] == ("col_a", 2)
        assert report.statistics.most_problematic_columns[1] == ("col_b", 1)

    def test_report_success_with_high_issues(self):
        issues = [
            ValidationIssue(column="col", issue_type="null", count=5, severity=Severity.HIGH),
        ]
        report = Report(issues=issues)
        assert report.success is False

    def test_report_success_with_only_medium_issues(self):
        issues = [
            ValidationIssue(column="col", issue_type="null", count=5, severity=Severity.MEDIUM),
        ]
        report = Report(issues=issues)
        assert report.success is True

    def test_report_statistics_updated_on_add_issue(self):
        report = Report(issues=[])
        assert report.statistics.unsuccessful_validations == 0

        report.add_issue(ValidationIssue(
            column="col", issue_type="null", count=5,
            severity=Severity.HIGH, validator_name="null",
        ))
        assert report.statistics.unsuccessful_validations == 1
        assert report.statistics.issues_by_severity == {"high": 1}
        assert report.success is False

    def test_report_statistics_updated_on_add_issues(self):
        report = Report(issues=[])
        report.add_issues([
            ValidationIssue(column="a", issue_type="null", count=1, severity=Severity.LOW),
            ValidationIssue(column="b", issue_type="null", count=2, severity=Severity.MEDIUM),
        ])
        assert report.statistics.unsuccessful_validations == 2
        assert report.success is True  # no HIGH+ issues

    def test_statistics_to_dict(self):
        issues = [
            ValidationIssue(column="col", issue_type="null", count=5,
                            severity=Severity.HIGH, validator_name="null"),
        ]
        report = Report(issues=issues)
        d = report.statistics.to_dict()

        assert "unsuccessful_validations" in d
        assert "issues_by_severity" in d
        assert "issues_by_column" in d
        assert "issues_by_validator" in d
        assert "most_problematic_columns" in d
        assert d["most_problematic_columns"][0] == {"column": "col", "issue_count": 1}

    def test_report_to_dict_includes_statistics(self):
        issues = [
            ValidationIssue(column="col", issue_type="null", count=5,
                            severity=Severity.HIGH, validator_name="null"),
        ]
        report = Report(issues=issues)
        d = report.to_dict()

        assert "statistics" in d
        assert "success" in d
        assert d["success"] is False

    def test_report_to_json_includes_statistics(self):
        issues = [
            ValidationIssue(column="col", issue_type="null", count=5,
                            severity=Severity.HIGH, validator_name="null"),
        ]
        report = Report(issues=issues)
        j = report.to_json()
        parsed = json.loads(j)
        assert "statistics" in parsed
        assert "success" in parsed


# ============================================================================
# 7. Non-Expression Validator Structured Results
# ============================================================================

class TestNonExpressionValidatorStructuredResults:
    """Tests that non-expression validators populate structured results."""

    def test_unique_validator_has_structured_result(self, duplicate_lf):
        from truthound.validators.uniqueness.unique import UniqueValidator

        validator = UniqueValidator(result_format=ResultFormat.BASIC)
        issues = validator.validate(duplicate_lf)

        assert len(issues) >= 1
        for issue in issues:
            assert issue.validator_name == "unique"
            assert issue.success is False
            assert issue.result is not None
            assert issue.result.element_count > 0

    def test_column_type_validator_has_structured_result(self):
        from truthound.validators.schema.column_type import ColumnTypeValidator

        lf = pl.DataFrame({"age": ["not_a_number"]}).lazy()
        validator = ColumnTypeValidator(
            expected_types={"age": "int"},
            result_format=ResultFormat.BASIC,
        )
        issues = validator.validate(lf)

        assert len(issues) == 1
        assert issues[0].validator_name == "column_type"
        assert issues[0].success is False
        assert issues[0].result is not None

    def test_column_exists_validator_has_structured_result(self):
        from truthound.validators.schema.column_exists import ColumnExistsValidator

        lf = pl.DataFrame({"existing": [1, 2, 3]}).lazy()
        validator = ColumnExistsValidator(
            columns=["existing", "missing"],
        )
        issues = validator.validate(lf)

        assert len(issues) == 1
        assert issues[0].validator_name == "column_exists"
        assert issues[0].success is False
        assert issues[0].result is not None

    def test_duplicate_validator_has_structured_result(self, duplicate_lf):
        from truthound.validators.uniqueness.duplicate import DuplicateValidator

        validator = DuplicateValidator(result_format=ResultFormat.BASIC)
        issues = validator.validate(duplicate_lf)

        for issue in issues:
            assert issue.validator_name == "duplicate"
            assert issue.success is False
            assert issue.result is not None

    def test_primary_key_validator_has_structured_result(self):
        from truthound.validators.uniqueness.primary_key import PrimaryKeyValidator

        lf = pl.DataFrame({
            "id": [1, 2, 3, 1, 5],
        }).lazy()
        validator = PrimaryKeyValidator(column="id")
        issues = validator.validate(lf)

        assert len(issues) >= 1
        for issue in issues:
            assert issue.validator_name == "primary_key"
            assert issue.result is not None


# ============================================================================
# 8. Cross-Format Consistency with Structured Results
# ============================================================================

class TestCrossFormatStructuredConsistency:
    """Ensures structured results are consistent across formats."""

    def test_element_count_same_across_formats(self, null_heavy_lf):
        from truthound.validators.completeness.null import NullValidator

        element_counts = {}
        for fmt in ResultFormat:
            validator = NullValidator(result_format=fmt)
            issues = validator.validate(null_heavy_lf)
            for issue in issues:
                key = (issue.column, fmt)
                if issue.result:
                    element_counts[key] = issue.result.element_count

        # All formats should report same element_count
        columns = {k[0] for k in element_counts}
        for col in columns:
            counts = [element_counts.get((col, fmt), -1) for fmt in ResultFormat]
            valid_counts = [c for c in counts if c != -1]
            assert len(set(valid_counts)) == 1, \
                f"element_count varies for {col}: {valid_counts}"

    def test_unexpected_count_same_across_formats(self, null_heavy_lf):
        from truthound.validators.completeness.null import NullValidator

        for fmt in [ResultFormat.BASIC, ResultFormat.SUMMARY, ResultFormat.COMPLETE]:
            validator = NullValidator(result_format=fmt)
            issues = validator.validate(null_heavy_lf)
            for issue in issues:
                assert issue.result is not None
                assert issue.result.unexpected_count == issue.count


# ============================================================================
# 9. Batch Executor with Structured Results
# ============================================================================

class TestBatchExecutorStructuredResults:
    """Tests that batch executor properly populates structured results."""

    def test_batch_basic_has_structured_results(self, null_heavy_lf):
        from truthound.validators.completeness.null import NullValidator, NotNullValidator

        executor = ExpressionBatchExecutor()
        executor.add_validator(NullValidator())
        executor.add_validator(NotNullValidator())

        issues = executor.execute(null_heavy_lf, result_format="basic")

        for issue in issues:
            assert issue.result is not None
            assert issue.result.element_count > 0
            assert issue.validator_name is not None

    def test_batch_boolean_only_minimal_results(self, null_heavy_lf):
        from truthound.validators.completeness.null import NullValidator

        executor = ExpressionBatchExecutor()
        executor.add_validator(NullValidator())

        issues = executor.execute(null_heavy_lf, result_format="boolean_only")

        for issue in issues:
            assert issue.result is not None
            # BOOLEAN_ONLY: no observed_value
            assert issue.result.observed_value is None
            # No samples
            assert issue.sample_values is None
            assert issue.result.partial_unexpected_list is None

    def test_batch_complete_has_full_results(self):
        from truthound.validators.distribution.range import BetweenValidator

        lf = pl.DataFrame({"val": [1, 200, -5, 50, 300]}).lazy()

        executor = ExpressionBatchExecutor()
        executor.add_validator(BetweenValidator(min_value=0, max_value=100))

        issues = executor.execute(
            lf,
            result_format=ResultFormatConfig(
                format=ResultFormat.COMPLETE,
                include_unexpected_rows=True,
                return_debug_query=True,
            ),
        )

        assert len(issues) == 1
        issue = issues[0]
        assert issue.result is not None
        assert issue.result.unexpected_list is not None
        assert issue.result.unexpected_index_list is not None
        assert issue.result.unexpected_rows is not None
        assert issue.result.debug_query is not None


# ============================================================================
# 10. Serialization Tests
# ============================================================================

class TestSerialization:
    """Tests for complete serialization pipeline."""

    def test_issue_with_result_json_serializable(self):
        detail = ValidationDetail.from_aggregates(
            element_count=100, unexpected_count=5, observed_value=5,
        )
        detail.partial_unexpected_list = [1, 2, 3]
        issue = ValidationIssue(
            column="col", issue_type="test", count=5,
            severity=Severity.MEDIUM,
            result=detail, validator_name="test_v", success=False,
        )
        d = issue.to_dict()
        # Should be JSON serializable
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        assert parsed["result"]["element_count"] == 100

    def test_report_with_structured_issues_json_serializable(self):
        detail = ValidationDetail.from_aggregates(
            element_count=100, unexpected_count=5,
        )
        issues = [
            ValidationIssue(
                column="col", issue_type="null", count=5,
                severity=Severity.HIGH,
                result=detail, validator_name="null", success=False,
            ),
        ]
        report = Report(issues=issues)
        j = report.to_json()
        parsed = json.loads(j)

        assert "statistics" in parsed
        assert "success" in parsed
        assert parsed["issues"][0]["result"]["element_count"] == 100

    def test_report_boolean_only_serialization(self):
        issues = [
            ValidationIssue(
                column="col", issue_type="null", count=5,
                severity=Severity.HIGH,
                validator_name="null", success=False,
            ),
        ]
        report = Report(issues=issues, result_format=ResultFormat.BOOLEAN_ONLY)
        d = report.to_dict()

        # BOOLEAN_ONLY should still have success field
        assert "success" in d
        assert d["result_format"] == "boolean_only"
        # Issues should have minimal fields
        for issue in d["issues"]:
            assert "column" in issue
            assert "success" in issue


# ============================================================================
# 11. Edge Cases
# ============================================================================

class TestPhase2EdgeCases:
    """Edge cases for PHASE 2 structured results."""

    def test_empty_dataframe_structured_result(self):
        from truthound.validators.completeness.null import NullValidator

        lf = pl.DataFrame({"col": []}).cast({"col": pl.Int64}).lazy()
        validator = NullValidator(result_format=ResultFormat.COMPLETE)
        issues = validator.validate(lf)
        assert len(issues) == 0

    def test_single_row_structured_result(self):
        from truthound.validators.completeness.null import NullValidator

        lf = pl.DataFrame({"col": [None]}).lazy()
        validator = NullValidator(result_format=ResultFormat.BASIC)
        issues = validator.validate(lf)

        assert len(issues) == 1
        assert issues[0].result is not None
        assert issues[0].result.element_count == 1
        assert issues[0].result.unexpected_count == 1

    def test_all_valid_no_structured_issues(self):
        from truthound.validators.completeness.null import NullValidator

        lf = pl.DataFrame({"col": [1, 2, 3, 4, 5]}).lazy()
        validator = NullValidator(result_format=ResultFormat.COMPLETE)
        issues = validator.validate(lf)
        assert len(issues) == 0

    def test_validation_detail_immutability(self):
        """ValidationDetail should be a mutable dataclass (not frozen)."""
        detail = ValidationDetail()
        detail.element_count = 100  # Should work
        detail.partial_unexpected_list = [1, 2]  # Should work
        assert detail.element_count == 100

    def test_report_filter_preserves_statistics(self):
        issues = [
            ValidationIssue(column="a", issue_type="null", count=5, severity=Severity.HIGH),
            ValidationIssue(column="b", issue_type="null", count=3, severity=Severity.LOW),
        ]
        report = Report(issues=issues)
        filtered = report.filter_by_severity(Severity.HIGH)

        # Filtered report should have its own statistics
        assert filtered.statistics is not None
        assert filtered.statistics.unsuccessful_validations == 1
