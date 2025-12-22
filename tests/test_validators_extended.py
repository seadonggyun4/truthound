"""Comprehensive tests for extended validators library.

Tests all 60 validators across 7 categories with real-world scenarios.
"""

import json
import random
from datetime import date, datetime, timedelta

import polars as pl
import pytest

from truthound.types import Severity

# =============================================================================
# Schema Validators Tests
# =============================================================================


class TestSchemaValidators:
    """Tests for schema category validators."""

    def test_column_exists_validator(self):
        """Test column existence validation."""
        from truthound.validators import ColumnExistsValidator

        df = pl.DataFrame({"id": [1, 2], "name": ["a", "b"]})

        # Should pass - columns exist
        validator = ColumnExistsValidator(columns=["id", "name"])
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

        # Should fail - missing column
        validator = ColumnExistsValidator(columns=["id", "email", "phone"])
        issues = validator.validate(df.lazy())
        assert len(issues) == 2
        assert all(i.issue_type == "missing_column" for i in issues)
        assert all(i.severity == Severity.CRITICAL for i in issues)

    def test_column_not_exists_validator(self):
        """Test deprecated column detection."""
        from truthound.validators import ColumnNotExistsValidator

        df = pl.DataFrame({"id": [1], "deprecated_field": ["x"], "name": ["a"]})

        validator = ColumnNotExistsValidator(columns=["deprecated_field", "old_id"])
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].column == "deprecated_field"

    def test_column_count_validator(self):
        """Test column count validation."""
        from truthound.validators import ColumnCountValidator

        df = pl.DataFrame({"a": [1], "b": [2], "c": [3]})

        # Exact count
        validator = ColumnCountValidator(exact_count=3)
        assert len(validator.validate(df.lazy())) == 0

        validator = ColumnCountValidator(exact_count=5)
        issues = validator.validate(df.lazy())
        assert len(issues) == 1
        assert issues[0].severity == Severity.CRITICAL

        # Range
        validator = ColumnCountValidator(min_count=2, max_count=5)
        assert len(validator.validate(df.lazy())) == 0

    def test_row_count_validator(self):
        """Test row count validation."""
        from truthound.validators import RowCountValidator

        df = pl.DataFrame({"a": list(range(100))})

        validator = RowCountValidator(min_count=50, max_count=200)
        assert len(validator.validate(df.lazy())) == 0

        validator = RowCountValidator(min_count=200)
        issues = validator.validate(df.lazy())
        assert len(issues) == 1
        assert issues[0].issue_type == "too_few_rows"

    def test_column_type_validator(self):
        """Test column type validation."""
        from truthound.validators import ColumnTypeValidator

        df = pl.DataFrame({
            "id": [1, 2, 3],
            "name": ["a", "b", "c"],
            "price": [1.5, 2.5, 3.5],
        })

        # Using type aliases
        validator = ColumnTypeValidator(expected_types={
            "id": "int",
            "name": "string",
            "price": "float",
        })
        assert len(validator.validate(df.lazy())) == 0

        # Type mismatch
        validator = ColumnTypeValidator(expected_types={
            "id": "string",
        })
        issues = validator.validate(df.lazy())
        assert len(issues) == 1
        assert issues[0].issue_type == "type_mismatch"

    def test_column_order_validator(self):
        """Test column order validation."""
        from truthound.validators import ColumnOrderValidator

        df = pl.DataFrame({"a": [1], "b": [2], "c": [3]})

        # Correct order
        validator = ColumnOrderValidator(expected_order=["a", "b", "c"], strict=True)
        assert len(validator.validate(df.lazy())) == 0

        # Wrong order
        validator = ColumnOrderValidator(expected_order=["c", "b", "a"], strict=True)
        issues = validator.validate(df.lazy())
        assert len(issues) == 1

    def test_table_schema_validator(self):
        """Test complete table schema validation."""
        from truthound.validators import TableSchemaValidator

        df = pl.DataFrame({
            "id": [1, 2],
            "name": ["a", "b"],
            "extra": [1.0, 2.0],
        })

        # Allow extra columns
        validator = TableSchemaValidator(
            expected_schema={"id": "int", "name": "string"},
            allow_extra_columns=True,
        )
        assert len(validator.validate(df.lazy())) == 0

        # Don't allow extra columns
        validator = TableSchemaValidator(
            expected_schema={"id": "int", "name": "string"},
            allow_extra_columns=False,
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 1
        assert issues[0].column == "extra"

    def test_column_pair_validator(self):
        """Test column relationship validation."""
        from truthound.validators import ColumnPairValidator

        df = pl.DataFrame({
            "start_value": [1, 5, 10],
            "end_value": [10, 8, 5],  # Last row violates: 10 > 5
        })

        validator = ColumnPairValidator(
            column_a="start_value",
            column_b="end_value",
            relationship="<",
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 1
        assert issues[0].count == 1

    def test_multi_column_unique_validator(self):
        """Test multi-column uniqueness validation."""
        from truthound.validators import MultiColumnUniqueValidator

        df = pl.DataFrame({
            "store_id": [1, 1, 2, 2],
            "product_id": [1, 2, 1, 1],  # (2, 1) appears twice
        })

        validator = MultiColumnUniqueValidator(columns=["store_id", "product_id"])
        issues = validator.validate(df.lazy())
        assert len(issues) == 1

    def test_referential_integrity_validator(self):
        """Test referential integrity validation."""
        from truthound.validators import ReferentialIntegrityValidator

        customers = pl.DataFrame({"id": [1, 2, 3]})
        orders = pl.DataFrame({
            "order_id": [1, 2, 3, 4],
            "customer_id": [1, 2, 99, 100],  # 99 and 100 don't exist
        })

        validator = ReferentialIntegrityValidator(
            column="customer_id",
            reference_data=customers,
            reference_column="id",
        )
        issues = validator.validate(orders.lazy())
        assert len(issues) == 1
        assert issues[0].count == 2


# =============================================================================
# Completeness Validators Tests
# =============================================================================


class TestCompletenessValidators:
    """Tests for completeness category validators."""

    def test_null_validator(self):
        """Test null value detection."""
        from truthound.validators import NullValidator

        df = pl.DataFrame({
            "a": [1, None, 3, None, 5],
            "b": [1, 2, 3, 4, 5],
        })

        validator = NullValidator()
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].column == "a"
        assert issues[0].count == 2

    def test_not_null_validator(self):
        """Test not null constraint validation."""
        from truthound.validators import NotNullValidator

        df = pl.DataFrame({
            "required_field": [1, None, 3],
        })

        validator = NotNullValidator(columns=["required_field"])
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].severity == Severity.HIGH

    def test_completeness_ratio_validator(self):
        """Test completeness ratio validation."""
        from truthound.validators import CompletenessRatioValidator

        df = pl.DataFrame({
            "a": [1, 2, 3, None, None],  # 60% complete
            "b": [1, 2, 3, 4, 5],  # 100% complete
        })

        validator = CompletenessRatioValidator(min_ratio=0.8)
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].column == "a"

    def test_empty_string_validator(self):
        """Test empty string detection."""
        from truthound.validators import EmptyStringValidator

        df = pl.DataFrame({
            "name": ["Alice", "", "Charlie", ""],
        })

        validator = EmptyStringValidator()
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].count == 2

    def test_whitespace_only_validator(self):
        """Test whitespace-only string detection."""
        from truthound.validators import WhitespaceOnlyValidator

        df = pl.DataFrame({
            "text": ["hello", "   ", "world", "\t\n"],
        })

        validator = WhitespaceOnlyValidator()
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].count == 2

    def test_conditional_null_validator(self):
        """Test conditional null validation."""
        from truthound.validators import ConditionalNullValidator

        df = pl.DataFrame({
            "subscription": ["free", "premium", "premium", "free"],
            "payment_method": [None, "card", None, None],  # premium needs payment
        })

        validator = ConditionalNullValidator(
            column="payment_method",
            condition_column="subscription",
            condition_values=["premium"],
        )
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].count == 1  # Only 1 premium with null payment

    def test_default_value_validator(self):
        """Test default/placeholder value detection."""
        from truthound.validators import DefaultValueValidator

        df = pl.DataFrame({
            "name": ["Alice", "N/A", "Bob", "unknown", "Charlie"],
        })

        validator = DefaultValueValidator(max_ratio=0.1)
        issues = validator.validate(df.lazy())

        assert len(issues) == 1


# =============================================================================
# Uniqueness Validators Tests
# =============================================================================


class TestUniquenessValidators:
    """Tests for uniqueness category validators."""

    def test_unique_validator(self):
        """Test uniqueness validation."""
        from truthound.validators import UniqueValidator

        df = pl.DataFrame({
            "id": [1, 2, 3, 1, 5],  # 1 is duplicated
        })

        validator = UniqueValidator(columns=["id"])
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].issue_type == "unique_violation"

    def test_unique_ratio_validator(self):
        """Test unique ratio validation."""
        from truthound.validators import UniqueRatioValidator

        df = pl.DataFrame({
            "category": ["A", "A", "A", "B", "B"],  # 40% unique
        })

        validator = UniqueRatioValidator(min_ratio=0.5)
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].issue_type == "unique_ratio_low"

    def test_distinct_count_validator(self):
        """Test distinct count validation."""
        from truthound.validators import DistinctCountValidator

        df = pl.DataFrame({
            "status": ["active", "inactive", "pending"],  # 3 distinct
        })

        validator = DistinctCountValidator(min_count=2, max_count=5)
        assert len(validator.validate(df.lazy())) == 0

        validator = DistinctCountValidator(min_count=5)
        issues = validator.validate(df.lazy())
        assert len(issues) == 1

    def test_duplicate_validator(self):
        """Test duplicate row detection."""
        from truthound.validators import DuplicateValidator

        df = pl.DataFrame({
            "a": [1, 1, 2, 3],
            "b": ["x", "x", "y", "z"],
        })

        validator = DuplicateValidator()
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].count == 2  # Both duplicate rows

    def test_duplicate_within_group_validator(self):
        """Test duplicate detection within groups."""
        from truthound.validators import DuplicateWithinGroupValidator

        df = pl.DataFrame({
            "customer_id": [1, 1, 2, 2],
            "order_id": [100, 100, 200, 201],  # Customer 1 has duplicate order
        })

        validator = DuplicateWithinGroupValidator(
            group_by=["customer_id"],
            check_column="order_id",
        )
        issues = validator.validate(df.lazy())

        assert len(issues) == 1

    def test_primary_key_validator(self):
        """Test primary key validation."""
        from truthound.validators import PrimaryKeyValidator

        df = pl.DataFrame({
            "id": [1, 2, None, 1],  # Has null and duplicate
        })

        validator = PrimaryKeyValidator(column="id")
        issues = validator.validate(df.lazy())

        assert len(issues) == 2  # Null and duplicate issues

    def test_compound_key_validator(self):
        """Test compound key validation."""
        from truthound.validators import CompoundKeyValidator

        df = pl.DataFrame({
            "store_id": [1, 1, 2],
            "date": ["2024-01-01", "2024-01-01", "2024-01-01"],  # (1, 2024-01-01) duplicated
        })

        validator = CompoundKeyValidator(columns=["store_id", "date"])
        issues = validator.validate(df.lazy())

        assert len(issues) == 1


# =============================================================================
# Distribution Validators Tests
# =============================================================================


class TestDistributionValidators:
    """Tests for distribution category validators."""

    def test_between_validator(self):
        """Test between range validation."""
        from truthound.validators import BetweenValidator

        df = pl.DataFrame({
            "age": [25, 30, 150, -5, 40],  # 150 and -5 out of range
        })

        validator = BetweenValidator(min_value=0, max_value=120, columns=["age"])
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].count == 2

    def test_range_validator_auto_detect(self):
        """Test auto range detection based on column name."""
        from truthound.validators import RangeValidator

        df = pl.DataFrame({
            "age": [25, -5, 200],  # -5 and 200 out of [0, 150]
            "percentage": [50, 110, -10],  # 110 and -10 out of [0, 100]
        })

        validator = RangeValidator()
        issues = validator.validate(df.lazy())

        assert len(issues) == 2

    def test_positive_validator(self):
        """Test positive value validation."""
        from truthound.validators import PositiveValidator

        df = pl.DataFrame({
            "amount": [100, 0, -50, 200],  # 0 and -50 are not positive
        })

        validator = PositiveValidator(columns=["amount"])
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].count == 2

    def test_non_negative_validator(self):
        """Test non-negative validation."""
        from truthound.validators import NonNegativeValidator

        df = pl.DataFrame({
            "count": [10, 0, -5, 20],  # Only -5 is negative
        })

        validator = NonNegativeValidator(columns=["count"])
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].count == 1

    def test_in_set_validator(self):
        """Test in-set validation."""
        from truthound.validators import InSetValidator

        df = pl.DataFrame({
            "status": ["active", "inactive", "unknown", "deleted"],
        })

        validator = InSetValidator(
            allowed_values=["active", "inactive", "pending"],
            columns=["status"],
        )
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].count == 2  # unknown and deleted

    def test_not_in_set_validator(self):
        """Test forbidden values validation."""
        from truthound.validators import NotInSetValidator

        df = pl.DataFrame({
            "data": ["normal", "test", "production", "debug"],
        })

        validator = NotInSetValidator(
            forbidden_values=["test", "debug"],
            columns=["data"],
        )
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].count == 2

    def test_increasing_validator(self):
        """Test monotonically increasing validation."""
        from truthound.validators import IncreasingValidator

        df = pl.DataFrame({
            "sequence": [1, 2, 3, 2, 5],  # 3 -> 2 violates
        })

        validator = IncreasingValidator(columns=["sequence"])
        issues = validator.validate(df.lazy())

        assert len(issues) == 1

    def test_decreasing_validator(self):
        """Test monotonically decreasing validation."""
        from truthound.validators import DecreasingValidator

        df = pl.DataFrame({
            "countdown": [5, 4, 3, 4, 1],  # 3 -> 4 violates
        })

        validator = DecreasingValidator(columns=["countdown"])
        issues = validator.validate(df.lazy())

        assert len(issues) == 1

    def test_outlier_validator(self):
        """Test IQR-based outlier detection."""
        from truthound.validators import OutlierValidator

        # Normal values with one extreme outlier
        values = [10, 11, 12, 13, 14, 15, 1000]
        df = pl.DataFrame({"value": values})

        validator = OutlierValidator(iqr_multiplier=1.5)
        issues = validator.validate(df.lazy())

        assert len(issues) == 1

    def test_zscore_outlier_validator(self):
        """Test Z-score based outlier detection."""
        from truthound.validators import ZScoreOutlierValidator

        values = [10, 11, 12, 13, 14, 15, 100]
        df = pl.DataFrame({"value": values})

        validator = ZScoreOutlierValidator(threshold=2.0)
        issues = validator.validate(df.lazy())

        assert len(issues) == 1

    def test_quantile_validator(self):
        """Test quantile range validation."""
        from truthound.validators import QuantileValidator

        df = pl.DataFrame({
            "score": list(range(1, 101)),  # 1 to 100
        })

        # Median should be around 50
        validator = QuantileValidator(quantile=0.5, min_value=40, max_value=60)
        assert len(validator.validate(df.lazy())) == 0

        validator = QuantileValidator(quantile=0.5, min_value=60)
        issues = validator.validate(df.lazy())
        assert len(issues) == 1

    def test_distribution_validator(self):
        """Test distribution comparison validation."""
        from truthound.validators import DistributionValidator

        # Reference distribution
        reference = pl.DataFrame({"value": list(range(100))})

        # Similar distribution - should pass
        similar = pl.DataFrame({"value": list(range(5, 105))})
        validator = DistributionValidator(reference_data=reference, threshold=0.1)
        issues = validator.validate(similar.lazy())
        assert len(issues) == 0

        # Very different distribution - should fail
        different = pl.DataFrame({"value": [1000] * 100})
        issues = validator.validate(different.lazy())
        assert len(issues) == 1


# =============================================================================
# String Validators Tests
# =============================================================================


class TestStringValidators:
    """Tests for string category validators."""

    def test_regex_validator(self):
        """Test regex pattern validation."""
        from truthound.validators import RegexValidator

        df = pl.DataFrame({
            "code": ["ABC-123", "XYZ-456", "invalid", "DEF-789"],
        })

        validator = RegexValidator(
            pattern=r"^[A-Z]{3}-\d{3}$",
            columns=["code"],
        )
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].count == 1

    def test_length_validator(self):
        """Test string length validation."""
        from truthound.validators import LengthValidator

        df = pl.DataFrame({
            "username": ["ab", "alice", "bob", "x"],  # ab, x too short
        })

        validator = LengthValidator(min_length=3, max_length=20, columns=["username"])
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].count == 2

    def test_email_validator(self):
        """Test email format validation."""
        from truthound.validators import EmailValidator

        df = pl.DataFrame({
            "contact": [
                "alice@example.com",
                "invalid-email",
                "bob@test.org",
                "not.an" "email",
            ],
        })

        validator = EmailValidator(columns=["contact"])
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].count == 2

    def test_url_validator(self):
        """Test URL format validation."""
        from truthound.validators import UrlValidator

        df = pl.DataFrame({
            "website": [
                "https://example.com",
                "http://test.org/path",
                "not-a-url",
                "ftp://wrong.protocol",
            ],
        })

        validator = UrlValidator(columns=["website"])
        issues = validator.validate(df.lazy())

        assert len(issues) == 1

    def test_phone_validator(self):
        """Test phone format validation."""
        from truthound.validators import PhoneValidator

        df = pl.DataFrame({
            "phone": [
                "+1-555-123-4567",
                "(555) 123-4567",
                "abc",
                "123",
            ],
        })

        validator = PhoneValidator(columns=["phone"])
        issues = validator.validate(df.lazy())

        assert len(issues) == 1

    def test_uuid_validator(self):
        """Test UUID format validation."""
        from truthound.validators import UuidValidator

        df = pl.DataFrame({
            "id": [
                "550e8400-e29b-41d4-a716-446655440000",
                "not-a-uuid",
                "123e4567-e89b-12d3-a456-426614174000",
            ],
        })

        validator = UuidValidator(columns=["id"])
        issues = validator.validate(df.lazy())

        assert len(issues) == 1

    def test_ip_address_validator(self):
        """Test IP address format validation."""
        from truthound.validators import IpAddressValidator

        df = pl.DataFrame({
            "ip": ["192.168.1.1", "10.0.0.1", "999.999.999.999", "not-ip"],
        })

        validator = IpAddressValidator(columns=["ip"])
        issues = validator.validate(df.lazy())

        assert len(issues) == 1

    def test_json_parseable_validator(self):
        """Test JSON parseability validation."""
        from truthound.validators import JsonParseableValidator

        df = pl.DataFrame({
            "data": [
                '{"key": "value"}',
                '[1, 2, 3]',
                "not json",
                '{"invalid": }',
            ],
        })

        validator = JsonParseableValidator(columns=["data"])
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].count == 2

    def test_alphanumeric_validator(self):
        """Test alphanumeric validation."""
        from truthound.validators import AlphanumericValidator

        df = pl.DataFrame({
            "code": ["ABC123", "test_code", "hello world", "valid42"],
        })

        validator = AlphanumericValidator(columns=["code"])
        issues = validator.validate(df.lazy())
        assert len(issues) == 1
        assert issues[0].count == 2  # underscore and space

        # Allow underscore
        validator = AlphanumericValidator(allow_underscore=True, columns=["code"])
        issues = validator.validate(df.lazy())
        assert len(issues) == 1
        assert issues[0].count == 1  # Only space

    def test_format_validator_auto_detect(self):
        """Test auto format detection based on column name."""
        from truthound.validators import FormatValidator

        df = pl.DataFrame({
            "email": ["valid@example.com", "invalid"],
            "phone": ["+1-555-1234", "not-phone"],
        })

        validator = FormatValidator()
        issues = validator.validate(df.lazy())

        assert len(issues) == 2  # Both columns have issues


# =============================================================================
# Datetime Validators Tests
# =============================================================================


class TestDatetimeValidators:
    """Tests for datetime category validators."""

    def test_date_format_validator(self):
        """Test date format validation."""
        from truthound.validators import DateFormatValidator

        df = pl.DataFrame({
            "date_str": ["2024-01-15", "2024-12-31", "01/15/2024", "invalid"],
        })

        validator = DateFormatValidator(format="%Y-%m-%d", columns=["date_str"])
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].count == 2

    def test_date_between_validator(self):
        """Test date range validation."""
        from truthound.validators import DateBetweenValidator

        df = pl.DataFrame({
            "event_date": [
                date(2024, 6, 15),
                date(2024, 1, 1),
                date(2023, 1, 1),  # Too old
                date(2025, 12, 31),  # Too new
            ],
        })

        validator = DateBetweenValidator(
            min_date="2024-01-01",
            max_date="2024-12-31",
            columns=["event_date"],
        )
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].count == 2

    def test_future_date_validator(self):
        """Test future date detection."""
        from truthound.validators import FutureDateValidator

        today = date.today()
        df = pl.DataFrame({
            "created_at": [
                today - timedelta(days=30),
                today - timedelta(days=1),
                today + timedelta(days=1),  # Future
                today + timedelta(days=365),  # Future
            ],
        })

        validator = FutureDateValidator(columns=["created_at"])
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].count == 2

    def test_past_date_validator(self):
        """Test old date detection."""
        from truthound.validators import PastDateValidator

        df = pl.DataFrame({
            "birth_date": [
                date(1990, 1, 1),
                date(2000, 1, 1),
                date(1800, 1, 1),  # Too old
            ],
        })

        validator = PastDateValidator(min_date="1900-01-01", columns=["birth_date"])
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].count == 1

    def test_date_order_validator(self):
        """Test date order validation."""
        from truthound.validators import DateOrderValidator

        df = pl.DataFrame({
            "start_date": [
                date(2024, 1, 1),
                date(2024, 6, 1),
                date(2024, 12, 1),  # Violates: start > end
            ],
            "end_date": [
                date(2024, 1, 31),
                date(2024, 6, 30),
                date(2024, 6, 1),
            ],
        })

        validator = DateOrderValidator(
            first_column="start_date",
            second_column="end_date",
        )
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].count == 1

    def test_timezone_validator(self):
        """Test timezone validation."""
        from truthound.validators import TimezoneValidator

        # Create timezone-aware and naive datetimes
        df = pl.DataFrame({
            "timestamp": [
                datetime(2024, 1, 1, 12, 0, 0),
                datetime(2024, 1, 2, 12, 0, 0),
            ],
        })

        validator = TimezoneValidator(require_timezone=True, columns=["timestamp"])
        issues = validator.validate(df.lazy())

        # Should warn about missing timezone
        assert len(issues) == 1


# =============================================================================
# Aggregate Validators Tests
# =============================================================================


class TestAggregateValidators:
    """Tests for aggregate category validators."""

    def test_mean_between_validator(self):
        """Test mean range validation."""
        from truthound.validators import MeanBetweenValidator

        df = pl.DataFrame({
            "score": [80, 85, 90, 95, 100],  # Mean = 90
        })

        validator = MeanBetweenValidator(min_value=85, max_value=95, columns=["score"])
        assert len(validator.validate(df.lazy())) == 0

        validator = MeanBetweenValidator(min_value=95, columns=["score"])
        issues = validator.validate(df.lazy())
        assert len(issues) == 1

    def test_median_between_validator(self):
        """Test median range validation."""
        from truthound.validators import MedianBetweenValidator

        df = pl.DataFrame({
            "value": [1, 2, 50, 98, 99],  # Median = 50
        })

        validator = MedianBetweenValidator(min_value=40, max_value=60, columns=["value"])
        assert len(validator.validate(df.lazy())) == 0

    def test_std_between_validator(self):
        """Test standard deviation range validation."""
        from truthound.validators import StdBetweenValidator

        # Low variance data
        df = pl.DataFrame({"value": [10, 10, 10, 11, 11]})

        validator = StdBetweenValidator(max_value=1, columns=["value"])
        assert len(validator.validate(df.lazy())) == 0

        # High variance data
        df = pl.DataFrame({"value": [1, 100, 1, 100, 1]})
        validator = StdBetweenValidator(max_value=10, columns=["value"])
        issues = validator.validate(df.lazy())
        assert len(issues) == 1

    def test_min_between_validator(self):
        """Test minimum value range validation."""
        from truthound.validators import MinBetweenValidator

        df = pl.DataFrame({"price": [10, 20, 30, 40, 50]})  # Min = 10

        validator = MinBetweenValidator(min_value=5, max_value=15, columns=["price"])
        assert len(validator.validate(df.lazy())) == 0

        validator = MinBetweenValidator(min_value=20, columns=["price"])
        issues = validator.validate(df.lazy())
        assert len(issues) == 1

    def test_max_between_validator(self):
        """Test maximum value range validation."""
        from truthound.validators import MaxBetweenValidator

        df = pl.DataFrame({"quantity": [1, 5, 10, 15, 100]})  # Max = 100

        validator = MaxBetweenValidator(max_value=150, columns=["quantity"])
        assert len(validator.validate(df.lazy())) == 0

        validator = MaxBetweenValidator(max_value=50, columns=["quantity"])
        issues = validator.validate(df.lazy())
        assert len(issues) == 1

    def test_sum_between_validator(self):
        """Test sum range validation."""
        from truthound.validators import SumBetweenValidator

        df = pl.DataFrame({"amount": [10, 20, 30, 40]})  # Sum = 100

        validator = SumBetweenValidator(min_value=90, max_value=110, columns=["amount"])
        assert len(validator.validate(df.lazy())) == 0

    def test_variance_between_validator(self):
        """Test variance range validation."""
        from truthound.validators import VarianceBetweenValidator

        df = pl.DataFrame({"value": [10, 10, 10, 10, 10]})  # Variance = 0

        validator = VarianceBetweenValidator(max_value=1, columns=["value"])
        assert len(validator.validate(df.lazy())) == 0

    def test_type_validator(self):
        """Test mixed type detection."""
        from truthound.validators import TypeValidator

        df = pl.DataFrame({
            "mixed": ["apple", "123", "banana", "456", "cherry"],
        })

        validator = TypeValidator(columns=["mixed"])
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].issue_type == "mixed_type"


# =============================================================================
# Large Scale Stress Tests
# =============================================================================


class TestLargeScaleValidation:
    """Stress tests with large datasets."""

    def test_million_row_null_validation(self):
        """Test null validation on 1M rows."""
        from truthound.validators import NullValidator

        # Create 1M rows with 5% nulls
        n = 1_000_000
        values = [i if random.random() > 0.05 else None for i in range(n)]
        df = pl.DataFrame({"value": values})

        validator = NullValidator()
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        # Approximately 5% should be null
        assert 40_000 < issues[0].count < 60_000

    def test_million_row_range_validation(self):
        """Test range validation on 1M rows."""
        from truthound.validators import BetweenValidator

        n = 1_000_000
        # 1% out of range
        values = [random.randint(0, 100) if random.random() > 0.01 else 999 for _ in range(n)]
        df = pl.DataFrame({"value": values})

        validator = BetweenValidator(min_value=0, max_value=100, columns=["value"])
        issues = validator.validate(df.lazy())

        assert len(issues) == 1

    def test_million_row_unique_validation(self):
        """Test uniqueness validation on 1M rows."""
        from truthound.validators import UniqueValidator

        n = 1_000_000
        # Create IDs with ~1% duplicates
        values = list(range(n))
        for i in range(n // 100):
            values[random.randint(0, n - 1)] = 0  # Create duplicates

        df = pl.DataFrame({"id": values})

        validator = UniqueValidator(columns=["id"])
        issues = validator.validate(df.lazy())

        assert len(issues) == 1


# =============================================================================
# Edge Cases and Boundary Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_dataframe(self):
        """Test validators handle empty DataFrames."""
        from truthound.validators import (
            NullValidator,
            UniqueValidator,
            BetweenValidator,
            EmailValidator,
        )

        df = pl.DataFrame({"col": []}).cast({"col": pl.Int64})

        for validator in [
            NullValidator(),
            UniqueValidator(),
            BetweenValidator(min_value=0, max_value=100),
        ]:
            issues = validator.validate(df.lazy())
            assert len(issues) == 0

    def test_single_row_dataframe(self):
        """Test validators handle single-row DataFrames."""
        from truthound.validators import (
            NullValidator,
            DuplicateValidator,
            OutlierValidator,
        )

        df = pl.DataFrame({"value": [42]})

        for validator in [NullValidator(), DuplicateValidator()]:
            issues = validator.validate(df.lazy())
            assert len(issues) == 0

    def test_all_null_column(self):
        """Test handling of all-null columns."""
        from truthound.validators import NullValidator, CompletenessRatioValidator

        df = pl.DataFrame({"null_col": [None, None, None]})

        validator = NullValidator()
        issues = validator.validate(df.lazy())
        assert len(issues) == 1
        assert issues[0].severity == Severity.CRITICAL

    def test_special_characters_in_strings(self):
        """Test string validators with special characters."""
        from truthound.validators import RegexValidator

        df = pl.DataFrame({
            "text": ["hello\nworld", "tab\there", "unicode: 日本語", "emoji: 😀"],
        })

        validator = RegexValidator(pattern=r".+", columns=["text"])
        issues = validator.validate(df.lazy())
        assert len(issues) == 0  # All should match .+

    def test_extreme_numeric_values(self):
        """Test validators with extreme numeric values."""
        from truthound.validators import BetweenValidator, OutlierValidator

        df = pl.DataFrame({
            "value": [1e-300, 1e300, float("inf"), float("-inf")],
        })

        validator = BetweenValidator(min_value=0, max_value=1e100, columns=["value"])
        issues = validator.validate(df.lazy())
        # Should detect inf values as out of range
        assert len(issues) >= 1

    def test_column_selection(self):
        """Test column selection functionality."""
        from truthound.validators import NullValidator

        df = pl.DataFrame({
            "a": [1, None, 3],
            "b": [None, None, None],
            "c": [1, 2, 3],
        })

        # Only check column 'a'
        validator = NullValidator(columns=["a"])
        issues = validator.validate(df.lazy())
        assert len(issues) == 1
        assert issues[0].column == "a"

        # Exclude column 'b'
        validator = NullValidator(exclude_columns=["b"])
        issues = validator.validate(df.lazy())
        assert len(issues) == 1
        assert issues[0].column == "a"


# =============================================================================
# Real-World Scenario Tests
# =============================================================================


class TestRealWorldScenarios:
    """Tests simulating real-world data quality scenarios."""

    def test_ecommerce_order_validation(self):
        """Test e-commerce order data validation."""
        from truthound.validators import (
            NotNullValidator,
            PositiveValidator,
            InSetValidator,
            DateOrderValidator,
            PrimaryKeyValidator,
        )

        orders = pl.DataFrame({
            "order_id": [1, 2, 3, 4, 5],
            "customer_id": [101, 102, None, 104, 105],  # Missing customer
            "amount": [99.99, 150.00, -10.00, 200.00, 0],  # Negative and zero
            "status": ["pending", "shipped", "invalid_status", "delivered", "pending"],
            "created_at": [
                date(2024, 1, 1),
                date(2024, 1, 2),
                date(2024, 1, 3),
                date(2024, 1, 4),
                date(2024, 1, 5),
            ],
            "shipped_at": [
                None,
                date(2024, 1, 3),
                None,
                date(2024, 1, 3),  # Shipped before created!
                None,
            ],
        })

        # Validate order_id is primary key
        pk_validator = PrimaryKeyValidator(column="order_id")
        assert len(pk_validator.validate(orders.lazy())) == 0

        # Validate customer_id is not null
        not_null_validator = NotNullValidator(columns=["customer_id"])
        issues = not_null_validator.validate(orders.lazy())
        assert len(issues) == 1

        # Validate amount is positive
        positive_validator = PositiveValidator(columns=["amount"])
        issues = positive_validator.validate(orders.lazy())
        assert len(issues) == 1
        assert issues[0].count == 2  # -10 and 0

        # Validate status is in allowed set
        status_validator = InSetValidator(
            allowed_values=["pending", "shipped", "delivered", "cancelled"],
            columns=["status"],
        )
        issues = status_validator.validate(orders.lazy())
        assert len(issues) == 1

    def test_user_registration_validation(self):
        """Test user registration data validation."""
        from truthound.validators import (
            EmailValidator,
            LengthValidator,
            UniqueValidator,
            RegexValidator,
        )

        users = pl.DataFrame({
            "email": [
                "alice@example.com",
                "bob@test",  # Invalid
                "charlie@demo.org",
                "alice@example.com",  # Duplicate
            ],
            "username": [
                "alice123",
                "bo",  # Too short
                "charlie_456",
                "david_user",
            ],
            "password_hash": [
                "a" * 64,
                "b" * 64,
                "short",  # Not 64 chars
                "d" * 64,
            ],
        })

        # Validate email format
        email_validator = EmailValidator(columns=["email"])
        issues = email_validator.validate(users.lazy())
        assert len(issues) == 1

        # Validate email uniqueness
        unique_validator = UniqueValidator(columns=["email"])
        issues = unique_validator.validate(users.lazy())
        assert len(issues) == 1

        # Validate username length
        length_validator = LengthValidator(min_length=3, max_length=20, columns=["username"])
        issues = length_validator.validate(users.lazy())
        assert len(issues) == 1

        # Validate password hash is 64 chars
        hash_validator = LengthValidator(exact_length=64, columns=["password_hash"])
        issues = hash_validator.validate(users.lazy())
        assert len(issues) == 1

    def test_financial_transaction_validation(self):
        """Test financial transaction data validation."""
        from truthound.validators import (
            CompoundKeyValidator,
            NonNegativeValidator,
            InSetValidator,
            SumBetweenValidator,
        )

        transactions = pl.DataFrame({
            "account_id": [1, 1, 2, 2, 3],
            "transaction_id": [100, 100, 200, 201, 300],  # (1, 100) duplicated
            "amount": [1000, -500, 2000, 150, -100],  # Negative amounts
            "type": ["credit", "debit", "credit", "transfer", "unknown"],  # unknown type
        })

        # Compound key should be unique
        key_validator = CompoundKeyValidator(columns=["account_id", "transaction_id"])
        issues = key_validator.validate(transactions.lazy())
        assert len(issues) == 1

        # Credit amounts should be non-negative
        credit_df = transactions.filter(pl.col("type") == "credit")
        amount_validator = NonNegativeValidator(columns=["amount"])
        issues = amount_validator.validate(credit_df.lazy())
        assert len(issues) == 0

        # Transaction type validation
        type_validator = InSetValidator(
            allowed_values=["credit", "debit", "transfer"],
            columns=["type"],
        )
        issues = type_validator.validate(transactions.lazy())
        assert len(issues) == 1

    def test_sensor_time_series_validation(self):
        """Test IoT sensor time series data validation."""
        from truthound.validators import (
            IncreasingValidator,
            BetweenValidator,
            NotNullValidator,
        )

        sensor_data = pl.DataFrame({
            "timestamp": [1, 2, 3, 2, 5],  # Out of order at position 3
            "temperature": [20.5, 21.0, 150.0, 22.0, 22.5],  # 150 is outlier
            "humidity": [45, 46, None, 48, 49],  # Missing value
        })

        # Timestamps should be increasing
        ts_validator = IncreasingValidator(columns=["timestamp"])
        issues = ts_validator.validate(sensor_data.lazy())
        assert len(issues) == 1

        # Temperature should be in reasonable range
        temp_validator = BetweenValidator(
            min_value=-50, max_value=60, columns=["temperature"]
        )
        issues = temp_validator.validate(sensor_data.lazy())
        assert len(issues) == 1

        # Humidity should not be null
        humidity_validator = NotNullValidator(columns=["humidity"])
        issues = humidity_validator.validate(sensor_data.lazy())
        assert len(issues) == 1
