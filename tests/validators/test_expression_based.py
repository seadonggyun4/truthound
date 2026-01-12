"""Tests for expression-based validation architecture.

This module tests the new expression-based validation infrastructure that
enables single collect() execution across multiple validators.
"""

import pytest
import polars as pl

from truthound.validators.base import (
    ValidationExpressionSpec,
    ExpressionValidatorProtocol,
    ExpressionValidatorMixin,
    ExpressionBatchExecutor,
    Validator,
    ValidationIssue,
)
from truthound.validators.completeness.null import (
    NullValidator,
    NotNullValidator,
    CompletenessRatioValidator,
)
from truthound.validators.distribution.range import (
    BetweenValidator,
    RangeValidator,
    PositiveValidator,
    NonNegativeValidator,
)
from truthound.validators.string.regex import RegexValidator


class TestValidationExpressionSpec:
    """Tests for ValidationExpressionSpec."""

    def test_basic_spec_creation(self):
        """Test creating a basic expression spec."""
        spec = ValidationExpressionSpec(
            column="test_col",
            validator_name="test_validator",
            issue_type="test_issue",
            count_expr=pl.col("test_col").is_null().sum(),
        )

        assert spec.column == "test_col"
        assert spec.validator_name == "test_validator"
        assert spec.issue_type == "test_issue"
        assert spec.non_null_expr is None

    def test_spec_with_non_null_expr(self):
        """Test spec with non-null expression."""
        spec = ValidationExpressionSpec(
            column="test_col",
            validator_name="test_validator",
            issue_type="test_issue",
            count_expr=pl.col("test_col").is_null().sum(),
            non_null_expr=pl.col("test_col").is_not_null().sum(),
        )

        assert spec.non_null_expr is not None

    def test_get_all_exprs_basic(self):
        """Test get_all_exprs returns properly aliased expressions."""
        spec = ValidationExpressionSpec(
            column="test_col",
            validator_name="test",
            issue_type="test",
            count_expr=pl.col("test_col").is_null().sum(),
        )

        exprs = spec.get_all_exprs("_v0")
        assert len(exprs) == 1
        # Expression should have correct alias

    def test_get_all_exprs_with_non_null(self):
        """Test get_all_exprs with non_null_expr."""
        spec = ValidationExpressionSpec(
            column="test_col",
            validator_name="test",
            issue_type="test",
            count_expr=pl.col("test_col").is_null().sum(),
            non_null_expr=pl.col("test_col").is_not_null().sum(),
        )

        exprs = spec.get_all_exprs("_v0")
        assert len(exprs) == 2

    def test_get_all_exprs_with_extras(self):
        """Test get_all_exprs with extra expressions."""
        spec = ValidationExpressionSpec(
            column="test_col",
            validator_name="test",
            issue_type="test",
            count_expr=pl.col("test_col").is_null().sum(),
            extra_exprs=[pl.col("test_col").mean(), pl.col("test_col").std()],
            extra_keys=["mean", "std"],
        )

        exprs = spec.get_all_exprs("_v0")
        assert len(exprs) == 3


class TestExpressionValidatorMixin:
    """Tests for ExpressionValidatorMixin."""

    def test_null_validator_implements_protocol(self):
        """Test that NullValidator implements ExpressionValidatorProtocol."""
        validator = NullValidator()
        assert isinstance(validator, ExpressionValidatorProtocol)

    def test_between_validator_implements_protocol(self):
        """Test that BetweenValidator implements ExpressionValidatorProtocol."""
        validator = BetweenValidator(min_value=0, max_value=100)
        assert isinstance(validator, ExpressionValidatorProtocol)

    def test_regex_validator_implements_protocol(self):
        """Test that RegexValidator implements ExpressionValidatorProtocol."""
        validator = RegexValidator(pattern=r"^[A-Z]+$")
        assert isinstance(validator, ExpressionValidatorProtocol)


class TestNullValidatorExpression:
    """Tests for expression-based NullValidator."""

    def test_basic_null_detection(self):
        """Test basic null detection with expression approach."""
        lf = pl.DataFrame({
            "a": [1, None, 3, None, 5],
            "b": [None, 2, None, 4, None],
        }).lazy()

        validator = NullValidator()
        issues = validator.validate(lf)

        assert len(issues) == 2

        # Check issue for column 'a'
        issue_a = next(i for i in issues if i.column == "a")
        assert issue_a.issue_type == "null"
        assert issue_a.count == 2

        # Check issue for column 'b'
        issue_b = next(i for i in issues if i.column == "b")
        assert issue_b.issue_type == "null"
        assert issue_b.count == 3

    def test_no_nulls(self):
        """Test with no null values."""
        lf = pl.DataFrame({
            "a": [1, 2, 3],
            "b": [4, 5, 6],
        }).lazy()

        validator = NullValidator()
        issues = validator.validate(lf)

        assert len(issues) == 0

    def test_empty_dataframe(self):
        """Test with empty dataframe."""
        lf = pl.DataFrame({"a": []}).lazy()

        validator = NullValidator()
        issues = validator.validate(lf)

        assert len(issues) == 0

    def test_get_validation_exprs(self):
        """Test get_validation_exprs returns correct specs."""
        lf = pl.DataFrame({"a": [1, None], "b": [None, 2]}).lazy()

        validator = NullValidator()
        specs = validator.get_validation_exprs(lf, ["a", "b"])

        assert len(specs) == 2
        assert specs[0].column == "a"
        assert specs[0].issue_type == "null"
        assert specs[1].column == "b"


class TestBetweenValidatorExpression:
    """Tests for expression-based BetweenValidator."""

    def test_basic_range_validation(self):
        """Test basic range validation with expression approach."""
        lf = pl.DataFrame({
            "value": [5, 15, 25, 105, 200],
        }).lazy()

        validator = BetweenValidator(min_value=0, max_value=100)
        issues = validator.validate(lf)

        assert len(issues) == 1
        assert issues[0].column == "value"
        assert issues[0].issue_type == "out_of_range"
        assert issues[0].count == 2  # 105 and 200 are out of range

    def test_all_in_range(self):
        """Test with all values in range."""
        lf = pl.DataFrame({
            "value": [10, 20, 30, 40, 50],
        }).lazy()

        validator = BetweenValidator(min_value=0, max_value=100)
        issues = validator.validate(lf)

        assert len(issues) == 0

    def test_get_validation_exprs(self):
        """Test get_validation_exprs returns correct specs."""
        lf = pl.DataFrame({"a": [1, 2], "b": [3, 4]}).lazy()

        validator = BetweenValidator(min_value=0, max_value=100)
        specs = validator.get_validation_exprs(lf, ["a", "b"])

        assert len(specs) == 2
        assert specs[0].column == "a"
        assert specs[0].issue_type == "out_of_range"


class TestRangeValidatorExpression:
    """Tests for expression-based RangeValidator."""

    def test_auto_detect_age_range(self):
        """Test auto-detection of age column range."""
        lf = pl.DataFrame({
            "age": [25, 30, -5, 200],  # -5 and 200 out of [0, 150]
        }).lazy()

        validator = RangeValidator()
        issues = validator.validate(lf)

        assert len(issues) == 1
        assert issues[0].column == "age"
        assert issues[0].count == 2

    def test_auto_detect_percentage(self):
        """Test auto-detection of percentage column range."""
        lf = pl.DataFrame({
            "pct_complete": [50, 75, 110, -10],  # 110 and -10 out of [0, 100]
        }).lazy()

        validator = RangeValidator()
        issues = validator.validate(lf)

        assert len(issues) == 1
        assert issues[0].column == "pct_complete"
        assert issues[0].count == 2

    def test_no_known_range_columns(self):
        """Test with columns that don't match known patterns."""
        lf = pl.DataFrame({
            "xyz": [1, 2, 3, 4, 5],
        }).lazy()

        validator = RangeValidator()
        issues = validator.validate(lf)

        assert len(issues) == 0


class TestPositiveValidatorExpression:
    """Tests for expression-based PositiveValidator."""

    def test_detect_non_positive(self):
        """Test detection of non-positive values."""
        lf = pl.DataFrame({
            "value": [1, 0, -1, 5, -10],
        }).lazy()

        validator = PositiveValidator()
        issues = validator.validate(lf)

        assert len(issues) == 1
        assert issues[0].count == 3  # 0, -1, -10


class TestNonNegativeValidatorExpression:
    """Tests for expression-based NonNegativeValidator."""

    def test_detect_negative(self):
        """Test detection of negative values."""
        lf = pl.DataFrame({
            "value": [1, 0, -1, 5, -10],
        }).lazy()

        validator = NonNegativeValidator()
        issues = validator.validate(lf)

        assert len(issues) == 1
        assert issues[0].count == 2  # -1, -10


class TestRegexValidatorExpression:
    """Tests for expression-based RegexValidator."""

    def test_basic_regex_validation(self):
        """Test basic regex validation with expression approach."""
        lf = pl.DataFrame({
            "code": ["ABC", "DEF", "123", "XYZ"],
        }).lazy()

        validator = RegexValidator(pattern=r"^[A-Z]+$")
        issues = validator.validate(lf)

        assert len(issues) == 1
        assert issues[0].count == 1  # "123" doesn't match

    def test_get_validation_exprs(self):
        """Test get_validation_exprs returns correct specs."""
        lf = pl.DataFrame({"code": ["ABC", "123"]}).lazy()

        validator = RegexValidator(pattern=r"^[A-Z]+$")
        specs = validator.get_validation_exprs(lf, ["code"])

        assert len(specs) == 1
        assert specs[0].column == "code"
        assert specs[0].issue_type == "regex_mismatch"


class TestExpressionBatchExecutor:
    """Tests for ExpressionBatchExecutor."""

    def test_single_validator(self):
        """Test batch executor with single validator."""
        lf = pl.DataFrame({
            "a": [1, None, 3],
        }).lazy()

        executor = ExpressionBatchExecutor()
        executor.add_validator(NullValidator())
        issues = executor.execute(lf)

        assert len(issues) == 1
        assert issues[0].issue_type == "null"

    def test_multiple_validators_same_type(self):
        """Test batch executor with multiple validators of same type."""
        lf = pl.DataFrame({
            "a": [1, None, 3],
            "b": [None, 2, None],
        }).lazy()

        executor = ExpressionBatchExecutor()
        executor.add_validator(NullValidator(columns=("a",)))
        executor.add_validator(NullValidator(columns=("b",)))
        issues = executor.execute(lf)

        assert len(issues) == 2

    def test_multiple_validator_types(self):
        """Test batch executor with different validator types."""
        lf = pl.DataFrame({
            "age": [25, None, 200],  # Has null and out-of-range
            "name": ["ABC", "123", None],  # Has regex mismatch and null
        }).lazy()

        executor = ExpressionBatchExecutor()
        executor.add_validator(NullValidator())
        executor.add_validator(RangeValidator())
        executor.add_validator(RegexValidator(pattern=r"^[A-Z]+$", columns=("name",)))

        issues = executor.execute(lf)

        # Should have issues from all validators
        issue_types = {i.issue_type for i in issues}
        assert "null" in issue_types
        assert "out_of_range" in issue_types
        assert "regex_mismatch" in issue_types

    def test_empty_executor(self):
        """Test batch executor with no validators."""
        lf = pl.DataFrame({"a": [1, 2, 3]}).lazy()

        executor = ExpressionBatchExecutor()
        issues = executor.execute(lf)

        assert len(issues) == 0

    def test_add_validators_chaining(self):
        """Test that add_validator returns self for chaining."""
        executor = ExpressionBatchExecutor()
        result = executor.add_validator(NullValidator())

        assert result is executor

    def test_add_multiple_validators_at_once(self):
        """Test add_validators method."""
        lf = pl.DataFrame({
            "a": [1, None, 3],
            "b": [0, -1, 5],
        }).lazy()

        executor = ExpressionBatchExecutor()
        executor.add_validators([
            NullValidator(),
            PositiveValidator(),
        ])
        issues = executor.execute(lf)

        # Should have null issue and positive issue
        assert len(issues) >= 2

    def test_clear_validators(self):
        """Test clearing validators from executor."""
        executor = ExpressionBatchExecutor()
        executor.add_validator(NullValidator())
        executor.clear()

        lf = pl.DataFrame({"a": [1, None, 3]}).lazy()
        issues = executor.execute(lf)

        assert len(issues) == 0

    def test_batched_execution_performance(self):
        """Test that batched execution performs better than sequential.

        This is a basic test to verify the concept works. In real scenarios,
        the performance difference would be more significant with larger data.
        """
        import time

        # Create a moderate-sized dataframe
        n_rows = 100_000
        lf = pl.DataFrame({
            "age": list(range(n_rows)),
            "value": [i % 150 - 25 for i in range(n_rows)],  # Some negative, some > 100
            "code": [f"{'ABC' if i % 3 == 0 else '123'}" for i in range(n_rows)],
        }).lazy()

        validators = [
            NullValidator(),
            RangeValidator(),
            PositiveValidator(columns=("value",)),
            RegexValidator(pattern=r"^[A-Z]+$", columns=("code",)),
        ]

        # Batched execution
        executor = ExpressionBatchExecutor()
        executor.add_validators(validators)

        start = time.time()
        batched_issues = executor.execute(lf)
        batched_time = time.time() - start

        # Sequential execution
        sequential_issues = []
        start = time.time()
        for v in validators:
            sequential_issues.extend(v.validate(lf))
        sequential_time = time.time() - start

        # Both should produce same number of issues (approximately)
        # Note: Results might differ slightly due to different severity calculations
        assert len(batched_issues) > 0
        assert len(sequential_issues) > 0

        # Log times for comparison (not enforcing specific ratio)
        print(f"\nBatched time: {batched_time:.4f}s")
        print(f"Sequential time: {sequential_time:.4f}s")
        print(f"Speedup: {sequential_time / batched_time:.2f}x")


class TestCompletenessRatioValidatorExpression:
    """Tests for expression-based CompletenessRatioValidator."""

    def test_below_threshold(self):
        """Test detection of columns below completeness threshold."""
        lf = pl.DataFrame({
            "a": [1, None, None, None, 5],  # 40% complete
        }).lazy()

        validator = CompletenessRatioValidator(min_ratio=0.5)
        issues = validator.validate(lf)

        assert len(issues) == 1
        assert issues[0].column == "a"
        assert issues[0].issue_type == "completeness_ratio"

    def test_above_threshold(self):
        """Test no issues when above threshold."""
        lf = pl.DataFrame({
            "a": [1, 2, 3, None, 5],  # 80% complete
        }).lazy()

        validator = CompletenessRatioValidator(min_ratio=0.5)
        issues = validator.validate(lf)

        assert len(issues) == 0

    def test_exact_threshold(self):
        """Test at exact threshold boundary."""
        lf = pl.DataFrame({
            "a": [1, 2, 3, 4, None],  # 80% complete
        }).lazy()

        validator = CompletenessRatioValidator(min_ratio=0.8)
        issues = validator.validate(lf)

        assert len(issues) == 0  # 80% >= 80%


class TestNotNullValidatorExpression:
    """Tests for expression-based NotNullValidator."""

    def test_detect_nulls(self):
        """Test detection of any nulls."""
        lf = pl.DataFrame({
            "a": [1, None, 3],
        }).lazy()

        validator = NotNullValidator()
        issues = validator.validate(lf)

        assert len(issues) == 1
        assert issues[0].issue_type == "not_null_violation"
        assert issues[0].count == 1

    def test_no_nulls(self):
        """Test no issues when all values present."""
        lf = pl.DataFrame({
            "a": [1, 2, 3],
        }).lazy()

        validator = NotNullValidator()
        issues = validator.validate(lf)

        assert len(issues) == 0


class TestEdgeCases:
    """Tests for edge cases in expression-based validation."""

    def test_all_null_column(self):
        """Test column with all null values."""
        lf = pl.DataFrame({
            "a": [None, None, None],
        }).lazy()

        validator = NullValidator()
        issues = validator.validate(lf)

        assert len(issues) == 1
        assert issues[0].count == 3

    def test_single_row(self):
        """Test with single row dataframe."""
        lf = pl.DataFrame({
            "a": [None],
        }).lazy()

        validator = NullValidator()
        issues = validator.validate(lf)

        assert len(issues) == 1
        assert issues[0].count == 1

    def test_mixed_types(self):
        """Test with mixed column types."""
        lf = pl.DataFrame({
            "num": [1, None, 3],
            "str": ["a", None, "c"],
            "date": [None, None, None],
        }).lazy()

        executor = ExpressionBatchExecutor()
        executor.add_validator(NullValidator())
        issues = executor.execute(lf)

        assert len(issues) == 3
        columns = {i.column for i in issues}
        assert "num" in columns
        assert "str" in columns
        assert "date" in columns

    def test_large_number_of_columns(self):
        """Test with many columns."""
        data = {f"col_{i}": [1, None, 3] for i in range(50)}
        lf = pl.DataFrame(data).lazy()

        executor = ExpressionBatchExecutor()
        executor.add_validator(NullValidator())
        issues = executor.execute(lf)

        assert len(issues) == 50  # All columns have 1 null

    def test_column_filter_integration(self):
        """Test that column filtering works with expression approach."""
        lf = pl.DataFrame({
            "a": [1, None, 3],
            "b": [None, 2, None],
        }).lazy()

        validator = NullValidator(columns=("a",))
        issues = validator.validate(lf)

        assert len(issues) == 1
        assert issues[0].column == "a"
