"""Tests for the main API functions."""

import polars as pl
import pytest

import truthound as th
from truthound.report import PIIReport, ProfileReport, Report


class TestCheck:
    """Tests for th.check()."""

    def test_check_dict_with_nulls(self):
        """Test check with dictionary input containing nulls."""
        data = {
            "name": ["Alice", "Bob", None, "David"],
            "age": [25, 30, 35, 40],
        }
        report = th.check(data)

        assert isinstance(report, Report)
        assert report.has_issues
        assert any(i.issue_type == "null" for i in report.issues)

    def test_check_polars_dataframe(self):
        """Test check with Polars DataFrame."""
        df = pl.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "value": [10, 20, 30, None],
            }
        )
        report = th.check(df)

        assert isinstance(report, Report)
        assert report.row_count == 4
        assert report.column_count == 2

    def test_check_with_specific_validators(self):
        """Test check with specific validators."""
        data = {"col": [1, 2, 3, None, 5]}
        report = th.check(data, validators=["null"])

        assert isinstance(report, Report)
        # Only null validator should run
        assert all(i.issue_type == "null" for i in report.issues)

    def test_check_min_severity_filter(self):
        """Test check with min_severity filter."""
        data = {
            "name": ["Alice", None, None, None, None],  # High null rate
            "age": [25, 30, 35, 40, None],  # Low null rate
        }
        report = th.check(data, min_severity="high")

        # Should only include high severity issues
        from truthound.types import Severity

        assert all(i.severity >= Severity.HIGH for i in report.issues)

    def test_check_no_issues(self):
        """Test check with clean data."""
        data = {
            "a": [1, 2, 3],
            "b": ["x", "y", "z"],
        }
        report = th.check(data, validators=["null"])

        assert not report.has_issues

    def test_check_duplicate_detection(self):
        """Test duplicate row detection."""
        data = {
            "a": [1, 1, 2, 2],
            "b": ["x", "x", "y", "y"],
        }
        report = th.check(data, validators=["duplicate"])

        assert report.has_issues
        assert any(i.issue_type == "duplicate_row" for i in report.issues)


class TestScan:
    """Tests for th.scan()."""

    def test_scan_with_email(self):
        """Test PII scan detecting email addresses."""
        data = {
            "email": ["alice@example.com", "bob@test.org", "charlie@demo.net"],
            "name": ["Alice", "Bob", "Charlie"],
        }
        report = th.scan(data)

        assert isinstance(report, PIIReport)
        assert report.has_pii
        assert any(f["pii_type"] == "Email Address" for f in report.findings)

    def test_scan_with_phone(self):
        """Test PII scan detecting phone numbers."""
        data = {
            "phone": ["555-123-4567", "555-987-6543", "555-456-7890"],
            "name": ["Alice", "Bob", "Charlie"],
        }
        report = th.scan(data)

        assert isinstance(report, PIIReport)
        assert report.has_pii
        assert any(f["pii_type"] == "Phone Number" for f in report.findings)

    def test_scan_no_pii(self):
        """Test PII scan with no PII data."""
        data = {
            "product": ["Apple", "Banana", "Orange"],
            "price": [1.50, 0.75, 2.00],
        }
        report = th.scan(data)

        assert isinstance(report, PIIReport)
        assert not report.has_pii


class TestMask:
    """Tests for th.mask()."""

    def test_mask_auto_detect(self):
        """Test mask with auto PII detection."""
        data = {
            "email": ["alice@example.com", "bob@test.org"],
            "name": ["Alice", "Bob"],
        }
        masked = th.mask(data)

        assert isinstance(masked, pl.DataFrame)
        # Email column should be masked
        emails = masked.get_column("email").to_list()
        assert "alice@example.com" not in emails

    def test_mask_specific_columns(self):
        """Test mask with specific columns."""
        data = {
            "secret": ["password123", "secret456"],
            "public": ["hello", "world"],
        }
        masked = th.mask(data, columns=["secret"])

        assert masked.get_column("secret").to_list() != ["password123", "secret456"]
        assert masked.get_column("public").to_list() == ["hello", "world"]

    def test_mask_hash_strategy(self):
        """Test mask with hash strategy."""
        data = {
            "email": ["alice@example.com", "bob@test.org"],
        }
        masked = th.mask(data, columns=["email"], strategy="hash")

        emails = masked.get_column("email").to_list()
        # Hash should produce hex strings
        assert all(len(e) == 16 for e in emails)


class TestProfile:
    """Tests for th.profile()."""

    def test_profile_basic(self):
        """Test basic profiling."""
        data = {
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "score": [85.5, 92.0, 78.5, 88.0, 95.5],
        }
        report = th.profile(data)

        assert isinstance(report, ProfileReport)
        assert report.row_count == 5
        assert report.column_count == 3
        assert len(report.columns) == 3

    def test_profile_with_nulls(self):
        """Test profiling with null values."""
        data = {
            "value": [1, None, 3, None, 5],
        }
        report = th.profile(data)

        assert report.row_count == 5
        col_info = report.columns[0]
        assert col_info["null_pct"] == "40.0%"


class TestCustomValidator:
    """Tests for custom validators."""

    def test_decorator_validator(self):
        """Test creating a validator with decorator."""

        @th.validator
        def check_positive(value: int) -> bool:
            return value > 0

        data = {"numbers": [1, -2, 3, -4, 5]}
        report = th.check(data, validators=[check_positive])

        assert report.has_issues
        assert any(i.issue_type == "check_positive" for i in report.issues)
