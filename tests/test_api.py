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


class TestRead:
    """Tests for th.read()."""

    def test_read_dict_data(self):
        """Test reading from dictionary data."""
        data = {"a": [1, 2, 3], "b": ["x", "y", "z"]}
        df = th.read(data)

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 3
        assert df.columns == ["a", "b"]

    def test_read_polars_dataframe(self):
        """Test reading from Polars DataFrame."""
        original = pl.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
        df = th.read(original)

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 3
        assert df.columns == original.columns
        assert df["id"].to_list() == original["id"].to_list()
        assert df["value"].to_list() == original["value"].to_list()

    def test_read_polars_lazyframe(self):
        """Test reading from Polars LazyFrame."""
        original = pl.LazyFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
        df = th.read(original)

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 3

    def test_read_with_sample_size(self):
        """Test reading with sample_size parameter."""
        data = {"id": list(range(1000))}
        df = th.read(data, sample_size=100)

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 100

    def test_read_sample_size_larger_than_data(self):
        """Test that sample_size larger than data returns all data."""
        data = {"id": [1, 2, 3, 4, 5]}
        df = th.read(data, sample_size=1000)

        assert len(df) == 5

    def test_read_dict_config_with_path_key(self):
        """Test reading with dict config containing 'path' key."""
        import tempfile
        import os

        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("a,b\n1,x\n2,y\n3,z\n")
            temp_path = f.name

        try:
            df = th.read({"path": temp_path})
            assert isinstance(df, pl.DataFrame)
            assert len(df) == 3
            assert "a" in df.columns
            assert "b" in df.columns
        finally:
            os.unlink(temp_path)

    def test_read_dict_config_missing_path_raises(self):
        """Test that dict config without 'path' key raises ValueError."""
        with pytest.raises(ValueError, match="must include 'path' key"):
            th.read({"delimiter": ","})

    def test_read_csv_file(self, tmp_path):
        """Test reading from CSV file."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("id,name\n1,Alice\n2,Bob\n3,Charlie\n")

        df = th.read(str(csv_file))

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 3
        assert df.columns == ["id", "name"]

    def test_read_parquet_file(self, tmp_path):
        """Test reading from Parquet file."""
        parquet_file = tmp_path / "test.parquet"
        pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}).write_parquet(parquet_file)

        df = th.read(str(parquet_file))

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 3
        assert set(df.columns) == {"x", "y"}

    def test_read_json_file(self, tmp_path):
        """Test reading from JSON file."""
        json_file = tmp_path / "test.json"
        json_file.write_text('[{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]')

        df = th.read(str(json_file))

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 2
