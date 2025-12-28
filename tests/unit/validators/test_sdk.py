"""Tests for the Custom Validator SDK.

This test suite covers:
- Decorator registration and metadata
- Builder pattern for validators
- Testing utilities
- Template validators
"""

import pytest
import polars as pl

from truthound.types import Severity
from truthound.validators.sdk import (
    # Core
    Validator,
    ValidationIssue,
    ValidatorConfig,
    # Decorators
    custom_validator,
    register_validator,
    validator_metadata,
    deprecated_validator,
    # Builder
    ValidatorBuilder,
    # Testing
    ValidatorTestCase,
    create_test_dataframe,
    assert_no_issues,
    assert_has_issue,
    assert_issue_count,
    # Templates
    SimpleColumnValidator,
    SimplePatternValidator,
    SimpleRangeValidator,
    SimpleComparisonValidator,
    CompositeValidator,
)
from truthound.validators.sdk.decorators import (
    get_registered_validators,
    get_validator_by_name,
    get_validator_metadata,
    unregister_validator,
    clear_registry,
)


# ============================================================================
# Test Decorators
# ============================================================================


class TestCustomValidatorDecorator:
    """Test @custom_validator decorator."""

    def setup_method(self):
        """Clear registry before each test."""
        clear_registry()

    def test_registers_validator(self):
        """Test that decorator registers the validator."""

        @custom_validator(name="test_validator", category="test")
        class TestValidator(Validator):
            def validate(self, lf):
                return []

        assert "test_validator" in get_registered_validators()
        assert get_validator_by_name("test_validator") is TestValidator

    def test_sets_class_attributes(self):
        """Test that decorator sets name and category."""

        @custom_validator(name="attr_test", category="custom")
        class AttrValidator(Validator):
            def validate(self, lf):
                return []

        assert AttrValidator.name == "attr_test"
        assert AttrValidator.category == "custom"

    def test_stores_metadata(self):
        """Test that decorator stores metadata."""

        @custom_validator(
            name="meta_test",
            category="test",
            description="Test validator",
            version="1.0.0",
            author="Test Author",
            tags=["test", "example"],
        )
        class MetaValidator(Validator):
            def validate(self, lf):
                return []

        meta = get_validator_metadata("meta_test")
        assert meta is not None
        assert meta.name == "meta_test"
        assert meta.description == "Test validator"
        assert meta.version == "1.0.0"
        assert meta.author == "Test Author"
        assert "test" in meta.tags
        assert "example" in meta.tags

    def test_prevents_duplicate_registration(self):
        """Test that duplicate names raise ValueError."""

        @custom_validator(name="duplicate", category="test")
        class First(Validator):
            def validate(self, lf):
                return []

        with pytest.raises(ValueError, match="already registered"):

            @custom_validator(name="duplicate", category="test")
            class Second(Validator):
                def validate(self, lf):
                    return []

    def test_auto_register_false(self):
        """Test that auto_register=False skips registration."""

        @custom_validator(name="no_register", auto_register=False)
        class NoRegister(Validator):
            def validate(self, lf):
                return []

        assert "no_register" not in get_registered_validators()

    def test_requires_validate_method(self):
        """Test that classes without validate raise TypeError."""
        with pytest.raises(TypeError, match="must have a 'validate' method"):

            @custom_validator(name="no_validate")
            class NoValidate:
                pass


class TestDeprecatedValidatorDecorator:
    """Test @deprecated_validator decorator."""

    def setup_method(self):
        clear_registry()

    def test_issues_deprecation_warning(self):
        """Test that deprecated validators issue warnings."""

        @deprecated_validator(message="Use new_validator instead")
        @custom_validator(name="old_validator")
        class OldValidator(Validator):
            def validate(self, lf):
                return []

        with pytest.warns(DeprecationWarning, match="deprecated"):
            OldValidator()

    def test_includes_replacement_in_warning(self):
        """Test that replacement is included in warning."""

        @deprecated_validator(replacement="better_validator")
        @custom_validator(name="worse_validator")
        class WorseValidator(Validator):
            def validate(self, lf):
                return []

        with pytest.warns(DeprecationWarning, match="better_validator"):
            WorseValidator()


# ============================================================================
# Test Builder
# ============================================================================


class TestValidatorBuilder:
    """Test ValidatorBuilder fluent API."""

    def test_builds_simple_validator(self):
        """Test building a simple validator."""
        validator = (
            ValidatorBuilder("null_check")
            .category("completeness")
            .description("Checks for null values")
            .check_column(
                lambda col, lf: lf.filter(pl.col(col).is_null())
                .select(pl.len())
                .collect()
                .item()
            )
            .with_issue_type("null_value")
            .with_severity(Severity.HIGH)
            .build()
        )

        assert validator.name == "null_check"
        assert validator.category == "completeness"

    def test_validator_finds_issues(self):
        """Test that built validator finds issues."""
        validator = (
            ValidatorBuilder("negative_check")
            .for_numeric_columns()
            .check(
                lambda col, lf: lf.filter(pl.col(col) < 0)
                .select(pl.len())
                .collect()
                .item()
            )
            .with_issue_type("negative_value")
            .build()
        )

        lf = pl.LazyFrame({"value": [1, -2, 3, -4, 5]})
        issues = validator.validate(lf)

        assert len(issues) == 1
        assert issues[0].count == 2
        assert issues[0].issue_type == "negative_value"

    def test_no_issues_for_clean_data(self):
        """Test that no issues are found for valid data."""
        validator = (
            ValidatorBuilder("positive_only")
            .for_numeric_columns()
            .check(
                lambda col, lf: lf.filter(pl.col(col) < 0)
                .select(pl.len())
                .collect()
                .item()
            )
            .build()
        )

        lf = pl.LazyFrame({"value": [1, 2, 3, 4, 5]})
        issues = validator.validate(lf)

        assert len(issues) == 0

    def test_requires_at_least_one_check(self):
        """Test that builder requires at least one check."""
        with pytest.raises(ValueError, match="At least one check"):
            ValidatorBuilder("empty").build()

    def test_multiple_checks(self):
        """Test builder with multiple checks."""
        validator = (
            ValidatorBuilder("multi_check")
            .for_numeric_columns()
            .check(
                lambda col, lf: lf.filter(pl.col(col) < 0)
                .select(pl.len())
                .collect()
                .item()
            )
            .with_issue_type("negative")
            .check(
                lambda col, lf: lf.filter(pl.col(col) > 100)
                .select(pl.len())
                .collect()
                .item()
            )
            .with_issue_type("too_large")
            .build()
        )

        lf = pl.LazyFrame({"value": [-1, 50, 150]})
        issues = validator.validate(lf)

        issue_types = {i.issue_type for i in issues}
        assert "negative" in issue_types
        assert "too_large" in issue_types


# ============================================================================
# Test Templates
# ============================================================================


class TestSimplePatternValidator:
    """Test SimplePatternValidator template."""

    def test_finds_non_matching_values(self):
        """Test that non-matching values are flagged."""

        class EmailValidator(SimplePatternValidator):
            name = "email"
            pattern = r"^[\w.+-]+@[\w.-]+\.\w+$"
            issue_type = "invalid_email"

        validator = EmailValidator()
        lf = pl.LazyFrame(
            {"email": ["test@example.com", "invalid", "another@test.org", "bad"]}
        )
        issues = validator.validate(lf)

        assert len(issues) == 1
        assert issues[0].count == 2
        assert issues[0].issue_type == "invalid_email"

    def test_invert_match(self):
        """Test that invert_match flags matching values."""

        class NoSSNValidator(SimplePatternValidator):
            name = "no_ssn"
            pattern = r"\d{3}-\d{2}-\d{4}"
            invert_match = True
            issue_type = "contains_ssn"

        validator = NoSSNValidator()
        lf = pl.LazyFrame(
            {"data": ["normal text", "123-45-6789", "more text", "987-65-4321"]}
        )
        issues = validator.validate(lf)

        assert len(issues) == 1
        assert issues[0].count == 2
        assert issues[0].issue_type == "contains_ssn"


class TestSimpleRangeValidator:
    """Test SimpleRangeValidator template."""

    def test_finds_out_of_range_values(self):
        """Test that out-of-range values are flagged."""

        class PercentageValidator(SimpleRangeValidator):
            name = "percentage"
            min_value = 0
            max_value = 100
            issue_type = "invalid_percentage"

        validator = PercentageValidator()
        lf = pl.LazyFrame({"pct": [50, -10, 75, 150, 100]})
        issues = validator.validate(lf)

        assert len(issues) == 1
        assert issues[0].count == 2  # -10 and 150

    def test_exclusive_bounds(self):
        """Test exclusive bounds."""

        class PositiveValidator(SimpleRangeValidator):
            name = "strictly_positive"
            min_value = 0
            inclusive_min = False
            issue_type = "non_positive"

        validator = PositiveValidator()
        lf = pl.LazyFrame({"value": [0, 1, 2, -1]})
        issues = validator.validate(lf)

        assert len(issues) == 1
        assert issues[0].count == 2  # 0 and -1


class TestSimpleComparisonValidator:
    """Test SimpleComparisonValidator template."""

    def test_finds_comparison_failures(self):
        """Test that failed comparisons are flagged."""

        class StartBeforeEndValidator(SimpleComparisonValidator):
            name = "start_before_end"
            left_column = "start"
            right_column = "end"
            operator = "lt"
            issue_type = "invalid_range"

        validator = StartBeforeEndValidator()
        lf = pl.LazyFrame({"start": [1, 5, 3], "end": [2, 3, 4]})  # 5 > 3 is invalid
        issues = validator.validate(lf)

        assert len(issues) == 1
        assert issues[0].count == 1


class TestCompositeValidator:
    """Test CompositeValidator template."""

    def test_combines_multiple_validators(self):
        """Test that composite runs all validators."""

        class NullCheck(Validator):
            name = "null"
            category = "test"

            def validate(self, lf):
                count = (
                    lf.filter(pl.col("value").is_null())
                    .select(pl.len())
                    .collect()
                    .item()
                )
                if count > 0:
                    return [
                        ValidationIssue(
                            column="value",
                            issue_type="null",
                            count=count,
                            severity=Severity.HIGH,
                        )
                    ]
                return []

        class NegativeCheck(Validator):
            name = "negative"
            category = "test"

            def validate(self, lf):
                count = (
                    lf.filter(pl.col("value") < 0)
                    .select(pl.len())
                    .collect()
                    .item()
                )
                if count > 0:
                    return [
                        ValidationIssue(
                            column="value",
                            issue_type="negative",
                            count=count,
                            severity=Severity.MEDIUM,
                        )
                    ]
                return []

        composite = CompositeValidator(validators=[NullCheck(), NegativeCheck()])
        lf = pl.LazyFrame({"value": [1, None, -2, 3]})
        issues = composite.validate(lf)

        issue_types = {i.issue_type for i in issues}
        assert "null" in issue_types
        assert "negative" in issue_types


# ============================================================================
# Test Testing Utilities
# ============================================================================


class TestTestingUtilities:
    """Test testing utility functions."""

    def test_create_test_dataframe(self):
        """Test create_test_dataframe function."""
        lf = create_test_dataframe(rows=100)
        df = lf.collect()
        assert len(df) == 100

    def test_create_test_dataframe_with_nulls(self):
        """Test creating dataframe with nulls."""
        lf = create_test_dataframe(rows=1000, include_nulls=True, null_probability=0.5)
        df = lf.collect()
        # Should have some nulls
        null_count = df.null_count().sum_horizontal().item()
        assert null_count > 0

    def test_assert_no_issues(self):
        """Test assert_no_issues helper."""
        assert_no_issues([])

        with pytest.raises(AssertionError):
            assert_no_issues(
                [
                    ValidationIssue(
                        column="test", issue_type="error", count=1, severity=Severity.LOW
                    )
                ]
            )

    def test_assert_has_issue(self):
        """Test assert_has_issue helper."""
        issues = [
            ValidationIssue(
                column="col1", issue_type="type1", count=5, severity=Severity.HIGH
            ),
            ValidationIssue(
                column="col2", issue_type="type2", count=3, severity=Severity.LOW
            ),
        ]

        # Should find matching issue
        issue = assert_has_issue(issues, column="col1", issue_type="type1")
        assert issue.count == 5

        # Should raise when not found
        with pytest.raises(AssertionError):
            assert_has_issue(issues, column="col3")

    def test_assert_issue_count(self):
        """Test assert_issue_count helper."""
        issues = [
            ValidationIssue(
                column="c1", issue_type="t1", count=1, severity=Severity.LOW
            ),
            ValidationIssue(
                column="c2", issue_type="t2", count=1, severity=Severity.LOW
            ),
        ]

        assert_issue_count(issues, 2)

        with pytest.raises(AssertionError):
            assert_issue_count(issues, 3)


class TestValidatorTestCase:
    """Test ValidatorTestCase base class."""

    def test_create_df(self):
        """Test create_df method."""

        class MyTestCase(ValidatorTestCase):
            pass

        tc = MyTestCase()
        tc.setUp()

        lf = tc.create_df({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        df = lf.collect()

        assert len(df) == 3
        assert "a" in df.columns
        assert "b" in df.columns


# ============================================================================
# Integration Tests
# ============================================================================


class TestSDKIntegration:
    """Integration tests for the SDK."""

    def setup_method(self):
        clear_registry()

    def test_full_workflow(self):
        """Test complete validator development workflow."""
        # 1. Create validator with decorator
        @custom_validator(
            name="percentage",
            category="numeric",
            description="Validates percentage values are in [0, 100]",
            tags=["numeric", "range", "percentage"],
        )
        class PercentageValidator(Validator):
            def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
                issues = []
                schema = lf.collect_schema()

                for col in schema.names():
                    if schema[col] in (pl.Float64, pl.Float32, pl.Int64, pl.Int32):
                        count = (
                            lf.filter((pl.col(col) < 0) | (pl.col(col) > 100))
                            .select(pl.len())
                            .collect()
                            .item()
                        )
                        if count > 0:
                            issues.append(
                                ValidationIssue(
                                    column=col,
                                    issue_type="invalid_percentage",
                                    count=count,
                                    severity=Severity.HIGH,
                                )
                            )
                return issues

        # 2. Verify registration
        assert "percentage" in get_registered_validators()
        meta = get_validator_metadata("percentage")
        assert meta.description == "Validates percentage values are in [0, 100]"
        assert "numeric" in meta.tags

        # 3. Use validator
        validator = PercentageValidator()
        lf = pl.LazyFrame({"score": [50, -10, 100, 150]})
        issues = validator.validate(lf)

        assert len(issues) == 1
        assert issues[0].count == 2
        assert issues[0].issue_type == "invalid_percentage"

    def test_builder_and_testing_integration(self):
        """Test builder with testing utilities."""
        # Build validator
        validator = (
            ValidatorBuilder("not_empty")
            .category("completeness")
            .for_string_columns()
            .check(
                lambda col, lf: lf.filter(
                    (pl.col(col).is_null()) | (pl.col(col).str.len_chars() == 0)
                )
                .select(pl.len())
                .collect()
                .item()
            )
            .with_issue_type("empty_string")
            .with_severity(Severity.MEDIUM)
            .build()
        )

        # Test with clean data
        clean_lf = pl.LazyFrame({"name": ["Alice", "Bob", "Charlie"]})
        issues = validator.validate(clean_lf)
        assert_no_issues(issues)

        # Test with problematic data
        dirty_lf = pl.LazyFrame({"name": ["Alice", "", None, "Bob"]})
        issues = validator.validate(dirty_lf)
        assert_has_issue(issues, column="name", issue_type="empty_string", min_count=2)
