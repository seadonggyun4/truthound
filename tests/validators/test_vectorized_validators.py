"""Tests for vectorized validators (within_record and format)."""

import polars as pl
import pytest

from truthound.validators.uniqueness.within_record import (
    UniqueWithinRecordValidator,
    AllColumnsUniqueWithinRecordValidator,
    ColumnPairUniqueValidator,
)
from truthound.validators.string.format import (
    EmailValidator,
    PhoneValidator,
    PhonePatterns,
    UuidValidator,
    IpAddressValidator,
    Ipv6AddressValidator,
    UrlValidator,
    FormatValidator,
)


class TestUniqueWithinRecordValidator:
    """Tests for vectorized UniqueWithinRecordValidator."""

    def test_detects_duplicates_within_row(self):
        """Should detect rows with duplicate values."""
        df = pl.DataFrame({
            "col_a": ["same", "diff1", "same"],
            "col_b": ["same", "diff2", "other"],
        })
        lf = df.lazy()

        validator = UniqueWithinRecordValidator(columns=["col_a", "col_b"])
        issues = validator.validate(lf)

        assert len(issues) == 1
        assert issues[0].count == 1  # Only first row has duplicates

    def test_no_duplicates_passes(self):
        """Should pass when no duplicates exist."""
        df = pl.DataFrame({
            "col_a": ["a", "b", "c"],
            "col_b": ["x", "y", "z"],
        })
        lf = df.lazy()

        validator = UniqueWithinRecordValidator(columns=["col_a", "col_b"])
        issues = validator.validate(lf)

        assert len(issues) == 0

    def test_ignore_nulls_default(self):
        """Should ignore null values by default."""
        df = pl.DataFrame({
            "col_a": [None, "a", None],
            "col_b": [None, "b", "x"],
        })
        lf = df.lazy()

        validator = UniqueWithinRecordValidator(columns=["col_a", "col_b"])
        issues = validator.validate(lf)

        assert len(issues) == 0  # Nulls don't count as duplicates

    def test_include_nulls_option(self):
        """Should count null == null when ignore_nulls=False."""
        df = pl.DataFrame({
            "col_a": [None, "a"],
            "col_b": [None, "b"],
        })
        lf = df.lazy()

        validator = UniqueWithinRecordValidator(
            columns=["col_a", "col_b"],
            ignore_nulls=False,
        )
        issues = validator.validate(lf)

        assert len(issues) == 1  # None == None counts as duplicate

    def test_three_columns(self):
        """Should work with more than 2 columns."""
        df = pl.DataFrame({
            "a": ["x", "x", "a"],
            "b": ["y", "x", "b"],  # Row 1 has x==x
            "c": ["z", "z", "c"],
        })
        lf = df.lazy()

        validator = UniqueWithinRecordValidator(columns=["a", "b", "c"])
        issues = validator.validate(lf)

        assert len(issues) == 1
        assert issues[0].count == 1

    def test_performance_large_dataset(self):
        """Should handle large datasets efficiently."""
        import time

        # 100K rows
        df = pl.DataFrame({
            "a": [f"a_{i}" for i in range(100_000)],
            "b": [f"b_{i}" for i in range(100_000)],
        })
        lf = df.lazy()

        validator = UniqueWithinRecordValidator(columns=["a", "b"])

        start = time.time()
        issues = validator.validate(lf)
        elapsed = time.time() - start

        # Should complete in under 1 second
        assert elapsed < 1.0
        assert len(issues) == 0

    def test_requires_at_least_two_columns(self):
        """Should raise error with less than 2 columns."""
        with pytest.raises(ValueError, match="At least 2 columns"):
            UniqueWithinRecordValidator(columns=["single"])


class TestAllColumnsUniqueWithinRecordValidator:
    """Tests for AllColumnsUniqueWithinRecordValidator."""

    def test_detects_any_duplicate_in_row(self):
        """Should detect duplicates across any columns."""
        df = pl.DataFrame({
            "a": ["x", "a"],
            "b": ["y", "b"],
            "c": ["x", "c"],  # Row 0: a==c (both "x")
        })
        lf = df.lazy()

        validator = AllColumnsUniqueWithinRecordValidator()
        issues = validator.validate(lf)

        assert len(issues) == 1
        assert issues[0].count == 1

    def test_empty_dataframe(self):
        """Should handle empty dataframe."""
        df = pl.DataFrame({"a": [], "b": []})
        lf = df.lazy()

        validator = AllColumnsUniqueWithinRecordValidator()
        issues = validator.validate(lf)

        assert len(issues) == 0


class TestColumnPairUniqueValidator:
    """Tests for ColumnPairUniqueValidator."""

    def test_detects_equal_values(self):
        """Should detect rows where column pair has equal values."""
        df = pl.DataFrame({
            "sender": ["A", "B", "C"],
            "receiver": ["X", "B", "Y"],  # Row 1: B == B
        })
        lf = df.lazy()

        validator = ColumnPairUniqueValidator(
            column_a="sender",
            column_b="receiver",
        )
        issues = validator.validate(lf)

        assert len(issues) == 1
        assert issues[0].count == 1

    def test_different_values_pass(self):
        """Should pass when all values are different."""
        df = pl.DataFrame({
            "a": ["1", "2", "3"],
            "b": ["x", "y", "z"],
        })
        lf = df.lazy()

        validator = ColumnPairUniqueValidator(column_a="a", column_b="b")
        issues = validator.validate(lf)

        assert len(issues) == 0


class TestEmailValidator:
    """Tests for vectorized EmailValidator."""

    def test_valid_emails_pass(self):
        """Should pass for valid email formats."""
        df = pl.DataFrame({
            "email": [
                "user@example.com",
                "user.name@sub.domain.org",
                "user+tag@example.co.uk",
            ]
        })
        lf = df.lazy()

        validator = EmailValidator(columns=["email"])
        issues = validator.validate(lf)

        assert len(issues) == 0

    def test_invalid_emails_detected(self):
        """Should detect invalid email formats."""
        df = pl.DataFrame({
            "email": [
                "not_an_email",
                "missing@tld",
                "@no-local.com",
                "spaces in@email.com",
            ]
        })
        lf = df.lazy()

        validator = EmailValidator(columns=["email"])
        issues = validator.validate(lf)

        assert len(issues) == 1
        assert issues[0].count == 4

    def test_performance_large_dataset(self):
        """Should be fast on large datasets."""
        import time

        df = pl.DataFrame({
            "email": [f"user{i}@example.com" for i in range(100_000)]
        })
        lf = df.lazy()

        validator = EmailValidator(columns=["email"])

        start = time.time()
        issues = validator.validate(lf)
        elapsed = time.time() - start

        # Should complete in under 0.5 seconds
        assert elapsed < 0.5
        assert len(issues) == 0


class TestPhoneValidator:
    """Tests for enhanced PhoneValidator."""

    def test_valid_phones_pass(self):
        """Should pass for valid phone formats."""
        df = pl.DataFrame({
            "phone": [
                "+1-234-567-8900",
                "(123) 456-7890",
                "123-456-7890",
                "+82 10 1234 5678",
                "02-123-4567",
            ]
        })
        lf = df.lazy()

        validator = PhoneValidator(columns=["phone"])
        issues = validator.validate(lf)

        assert len(issues) == 0

    def test_invalid_phones_detected(self):
        """Should reject invalid phone formats."""
        df = pl.DataFrame({
            "phone": [
                "(+1)",  # Incomplete
                "12345",  # Too short
                "not a phone",  # Text
            ]
        })
        lf = df.lazy()

        validator = PhoneValidator(columns=["phone"])
        issues = validator.validate(lf)

        assert len(issues) == 1
        assert issues[0].count == 3

    def test_north_america_pattern(self):
        """Should validate US/Canada phone format."""
        df = pl.DataFrame({
            "phone": [
                "(123) 456-7890",
                "123-456-7890",
                "+1 234 567 8900",
                "12345",  # Invalid
            ]
        })
        lf = df.lazy()

        validator = PhoneValidator(columns=["phone"], pattern="north_america")
        issues = validator.validate(lf)

        assert len(issues) == 1

    def test_korean_pattern(self):
        """Should validate Korean phone format."""
        df = pl.DataFrame({
            "phone": [
                "010-1234-5678",
                "02-123-4567",
                "031-123-4567",
                "123-456",  # Invalid
            ]
        })
        lf = df.lazy()

        validator = PhoneValidator(columns=["phone"], pattern="korean")
        issues = validator.validate(lf)

        assert len(issues) == 1

    def test_custom_pattern(self):
        """Should accept custom regex pattern."""
        df = pl.DataFrame({
            "phone": ["1234", "5678", "abc"]
        })
        lf = df.lazy()

        # Custom: exactly 4 digits
        validator = PhoneValidator(
            columns=["phone"],
            pattern=r"^\d{4}$",
        )
        issues = validator.validate(lf)

        assert len(issues) == 1
        assert issues[0].count == 1  # "abc" invalid


class TestUuidValidator:
    """Tests for UuidValidator."""

    def test_valid_uuids_pass(self):
        """Should pass for valid UUID format."""
        df = pl.DataFrame({
            "id": [
                "550e8400-e29b-41d4-a716-446655440000",
                "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
            ]
        })
        lf = df.lazy()

        validator = UuidValidator(columns=["id"])
        issues = validator.validate(lf)

        assert len(issues) == 0

    def test_invalid_uuids_detected(self):
        """Should detect invalid UUIDs."""
        df = pl.DataFrame({
            "id": [
                "not-a-uuid",
                "550e8400-e29b-41d4-a716",  # Too short
                "550e8400e29b41d4a716446655440000",  # No hyphens
            ]
        })
        lf = df.lazy()

        validator = UuidValidator(columns=["id"])
        issues = validator.validate(lf)

        assert len(issues) == 1
        assert issues[0].count == 3


class TestIpAddressValidator:
    """Tests for IpAddressValidator."""

    def test_valid_ips_pass(self):
        """Should pass for valid IPv4 addresses."""
        df = pl.DataFrame({
            "ip": [
                "192.168.1.1",
                "10.0.0.1",
                "255.255.255.255",
                "0.0.0.0",
            ]
        })
        lf = df.lazy()

        validator = IpAddressValidator(columns=["ip"])
        issues = validator.validate(lf)

        assert len(issues) == 0

    def test_invalid_ips_detected(self):
        """Should detect invalid IP addresses."""
        df = pl.DataFrame({
            "ip": [
                "256.1.1.1",  # Out of range
                "192.168.1",  # Incomplete
                "not.an.ip.address",
            ]
        })
        lf = df.lazy()

        validator = IpAddressValidator(columns=["ip"])
        issues = validator.validate(lf)

        assert len(issues) == 1


class TestUrlValidator:
    """Tests for UrlValidator."""

    def test_valid_urls_pass(self):
        """Should pass for valid URLs."""
        df = pl.DataFrame({
            "url": [
                "https://example.com",
                "http://localhost:8080/path",
                "https://sub.domain.org/page?query=1",
            ]
        })
        lf = df.lazy()

        validator = UrlValidator(columns=["url"])
        issues = validator.validate(lf)

        assert len(issues) == 0

    def test_invalid_urls_detected(self):
        """Should detect invalid URLs."""
        df = pl.DataFrame({
            "url": [
                "not a url",
                "ftp://example.com",  # Wrong protocol
                "example.com",  # No protocol
            ]
        })
        lf = df.lazy()

        validator = UrlValidator(columns=["url"])
        issues = validator.validate(lf)

        assert len(issues) == 1


class TestFormatValidator:
    """Tests for auto-detecting FormatValidator."""

    def test_auto_detects_email_column(self):
        """Should auto-detect email columns by name."""
        df = pl.DataFrame({
            "user_email": ["invalid", "user@example.com"],
            "other": ["text", "more text"],
        })
        lf = df.lazy()

        validator = FormatValidator()
        issues = validator.validate(lf)

        # Should only check user_email column
        assert len(issues) == 1
        assert issues[0].column == "user_email"
        assert issues[0].count == 1

    def test_auto_detects_phone_column(self):
        """Should auto-detect phone columns by name."""
        df = pl.DataFrame({
            "phone_number": ["12345", "123-456-7890"],
        })
        lf = df.lazy()

        validator = FormatValidator()
        issues = validator.validate(lf)

        assert len(issues) == 1
        assert "phone" in issues[0].column.lower()

    def test_ignores_unrecognized_columns(self):
        """Should not validate unrecognized column names."""
        df = pl.DataFrame({
            "random_data": ["anything", "goes here"],
        })
        lf = df.lazy()

        validator = FormatValidator()
        issues = validator.validate(lf)

        assert len(issues) == 0


class TestIpv6AddressValidator:
    """Tests for Ipv6AddressValidator."""

    def test_valid_ipv6_pass(self):
        """Should pass for valid IPv6 addresses."""
        df = pl.DataFrame({
            "ipv6": [
                "2001:0db8:85a3:0000:0000:8a2e:0370:7334",
                "::1",
                "::",
            ]
        })
        lf = df.lazy()

        validator = Ipv6AddressValidator(columns=["ipv6"])
        issues = validator.validate(lf)

        assert len(issues) == 0
