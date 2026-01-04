"""Tests for ProfileAdapter and generate_suite ProfileReport support."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pytest

from truthound.profiler.base import ColumnProfile, DataType, TableProfile
from truthound.profiler.generators.suite_generator import (
    ProfileAdapter,
    ProfileInput,
    generate_suite,
)


@dataclass
class MockProfileReport:
    """Mock ProfileReport for testing.

    Mimics the structure of truthound.report.ProfileReport.
    """

    source: str = "test_data.csv"
    row_count: int = 1000
    column_count: int = 5
    size_bytes: int = 50000
    columns: list[dict] = field(default_factory=list)


class TestProfileAdapterPercentageParsing:
    """Test percentage string parsing."""

    def test_parse_percentage_with_percent_sign(self):
        """Test parsing percentage strings with % sign."""
        assert ProfileAdapter._parse_percentage("10.5%") == 0.105
        assert ProfileAdapter._parse_percentage("0%") == 0.0
        assert ProfileAdapter._parse_percentage("100%") == 1.0

    def test_parse_percentage_without_percent_sign(self):
        """Test parsing percentage values without % sign."""
        # Values > 1 are treated as percentages
        assert ProfileAdapter._parse_percentage("50") == 0.5
        assert ProfileAdapter._parse_percentage("100") == 1.0

        # Values <= 1 are treated as ratios
        assert ProfileAdapter._parse_percentage("0.5") == 0.5
        assert ProfileAdapter._parse_percentage("0.1") == 0.1

    def test_parse_percentage_from_numbers(self):
        """Test parsing numeric values directly."""
        assert ProfileAdapter._parse_percentage(50) == 0.5
        assert ProfileAdapter._parse_percentage(0.5) == 0.5
        assert ProfileAdapter._parse_percentage(0) == 0.0

    def test_parse_percentage_invalid(self):
        """Test parsing invalid values."""
        assert ProfileAdapter._parse_percentage("invalid") == 0.0
        assert ProfileAdapter._parse_percentage("abc%") == 0.0


class TestProfileAdapterDataTypeInference:
    """Test data type inference from dtype strings."""

    def test_infer_integer_types(self):
        """Test inferring integer types."""
        assert ProfileAdapter._infer_data_type("Int64") == DataType.INTEGER
        assert ProfileAdapter._infer_data_type("Int32") == DataType.INTEGER
        assert ProfileAdapter._infer_data_type("i64") == DataType.INTEGER
        assert ProfileAdapter._infer_data_type("UInt8") == DataType.INTEGER

    def test_infer_float_types(self):
        """Test inferring float types."""
        assert ProfileAdapter._infer_data_type("Float64") == DataType.FLOAT
        assert ProfileAdapter._infer_data_type("Float32") == DataType.FLOAT
        assert ProfileAdapter._infer_data_type("f64") == DataType.FLOAT
        assert ProfileAdapter._infer_data_type("Decimal") == DataType.FLOAT

    def test_infer_boolean_type(self):
        """Test inferring boolean type."""
        assert ProfileAdapter._infer_data_type("Boolean") == DataType.BOOLEAN
        assert ProfileAdapter._infer_data_type("bool") == DataType.BOOLEAN

    def test_infer_string_types(self):
        """Test inferring string types."""
        assert ProfileAdapter._infer_data_type("String") == DataType.STRING
        assert ProfileAdapter._infer_data_type("Utf8") == DataType.STRING
        assert ProfileAdapter._infer_data_type("str") == DataType.STRING
        assert ProfileAdapter._infer_data_type("Categorical") == DataType.STRING

    def test_infer_datetime_types(self):
        """Test inferring datetime types."""
        assert ProfileAdapter._infer_data_type("Datetime") == DataType.DATETIME
        assert ProfileAdapter._infer_data_type("Date") == DataType.DATE
        assert ProfileAdapter._infer_data_type("Time") == DataType.TIME
        assert ProfileAdapter._infer_data_type("Duration") == DataType.DURATION

    def test_infer_unknown_type(self):
        """Test unknown types default to UNKNOWN."""
        assert ProfileAdapter._infer_data_type("Unknown") == DataType.UNKNOWN
        assert ProfileAdapter._infer_data_type("SomeRandomType") == DataType.UNKNOWN


class TestProfileAdapterFromProfileReport:
    """Test conversion from ProfileReport to TableProfile."""

    def test_basic_conversion(self):
        """Test basic ProfileReport to TableProfile conversion."""
        report = MockProfileReport(
            source="data.csv",
            row_count=1000,
            column_count=3,
            size_bytes=25000,
            columns=[
                {"name": "id", "dtype": "Int64", "null_pct": "0%", "unique_pct": "100%"},
                {"name": "name", "dtype": "String", "null_pct": "5%", "unique_pct": "95%"},
                {"name": "age", "dtype": "Int32", "null_pct": "10%", "unique_pct": "50%"},
            ],
        )

        table_profile = ProfileAdapter.to_table_profile(report)

        assert table_profile.name == "data.csv"
        assert table_profile.row_count == 1000
        assert table_profile.column_count == 3
        assert len(table_profile.columns) == 3

    def test_column_profiles_created(self):
        """Test column profiles are correctly created."""
        report = MockProfileReport(
            row_count=100,
            columns=[
                {"name": "id", "dtype": "Int64", "null_pct": "0%", "unique_pct": "100%"},
            ],
        )

        table_profile = ProfileAdapter.to_table_profile(report)

        assert len(table_profile.columns) == 1
        col = table_profile.columns[0]

        assert col.name == "id"
        assert col.physical_type == "Int64"
        assert col.inferred_type == DataType.INTEGER
        assert col.null_ratio == 0.0
        assert col.unique_ratio == 1.0
        assert col.is_unique is True

    def test_null_count_calculated(self):
        """Test null count is calculated from ratio."""
        report = MockProfileReport(
            row_count=1000,
            columns=[
                {"name": "col", "dtype": "String", "null_pct": "10%", "unique_pct": "50%"},
            ],
        )

        table_profile = ProfileAdapter.to_table_profile(report)
        col = table_profile.columns[0]

        assert col.null_count == 100  # 10% of 1000
        assert col.null_ratio == 0.1


class TestProfileAdapterFromDict:
    """Test conversion from dict to TableProfile."""

    def test_basic_dict_conversion(self):
        """Test basic dict to TableProfile conversion."""
        data = {
            "name": "test_table",
            "row_count": 500,
            "column_count": 2,
            "size_bytes": 10000,
            "columns": [
                {"name": "id", "dtype": "Int64", "null_ratio": 0.0, "unique_ratio": 1.0},
                {"name": "value", "dtype": "Float64", "null_ratio": 0.05, "unique_ratio": 0.8},
            ],
        }

        table_profile = ProfileAdapter.to_table_profile(data)

        assert table_profile.name == "test_table"
        assert table_profile.row_count == 500
        assert len(table_profile.columns) == 2

    def test_dict_with_source_field(self):
        """Test dict with 'source' instead of 'name'."""
        data = {
            "source": "data.parquet",
            "row_count": 100,
            "columns": [],
        }

        table_profile = ProfileAdapter.to_table_profile(data)
        assert table_profile.name == "data.parquet"

    def test_dict_with_table_profile_format(self):
        """Test dict in TableProfile format (with inferred_type)."""
        data = {
            "name": "test",
            "row_count": 100,
            "columns": [
                {
                    "name": "id",
                    "physical_type": "Int64",
                    "inferred_type": "integer",
                    "null_ratio": 0.0,
                    "unique_ratio": 1.0,
                    "row_count": 100,
                },
            ],
        }

        table_profile = ProfileAdapter.to_table_profile(data)
        col = table_profile.columns[0]

        assert col.inferred_type == DataType.INTEGER


class TestProfileAdapterTableProfilePassthrough:
    """Test TableProfile passthrough."""

    def test_table_profile_unchanged(self):
        """Test TableProfile is returned unchanged."""
        original = TableProfile(
            name="test",
            row_count=100,
            column_count=1,
            columns=(
                ColumnProfile(
                    name="id",
                    physical_type="Int64",
                    inferred_type=DataType.INTEGER,
                    row_count=100,
                ),
            ),
        )

        result = ProfileAdapter.to_table_profile(original)
        assert result is original


class TestProfileAdapterErrors:
    """Test error handling."""

    def test_unsupported_type_raises(self):
        """Test unsupported type raises TypeError."""
        with pytest.raises(TypeError, match="Unsupported profile type"):
            ProfileAdapter.to_table_profile([1, 2, 3])

    def test_string_not_supported(self):
        """Test string is not a valid profile type."""
        with pytest.raises(TypeError, match="Unsupported profile type"):
            ProfileAdapter.to_table_profile("not a profile")


class TestGenerateSuiteWithProfileReport:
    """Test generate_suite function with various profile types."""

    def test_with_profile_report(self):
        """Test generate_suite accepts ProfileReport."""
        report = MockProfileReport(
            source="test.csv",
            row_count=1000,
            column_count=2,
            columns=[
                {"name": "id", "dtype": "Int64", "null_pct": "0%", "unique_pct": "100%"},
                {"name": "name", "dtype": "String", "null_pct": "5%", "unique_pct": "90%"},
            ],
        )

        suite = generate_suite(report)

        assert suite is not None
        assert "test.csv" in suite.source_profile or suite.name == "test.csv"

    def test_with_dict(self):
        """Test generate_suite accepts dict."""
        data = {
            "name": "test_data",
            "row_count": 500,
            "column_count": 1,
            "columns": [
                {"name": "value", "dtype": "Float64", "null_pct": "10%", "unique_pct": "80%"},
            ],
        }

        suite = generate_suite(data)

        assert suite is not None

    def test_with_table_profile(self):
        """Test generate_suite accepts TableProfile."""
        profile = TableProfile(
            name="native_profile",
            row_count=100,
            column_count=1,
            columns=(
                ColumnProfile(
                    name="id",
                    physical_type="Int64",
                    inferred_type=DataType.INTEGER,
                    row_count=100,
                    null_count=0,
                    null_ratio=0.0,
                    distinct_count=100,
                    unique_ratio=1.0,
                    is_unique=True,
                ),
            ),
        )

        suite = generate_suite(profile)

        assert suite is not None
        assert suite.source_profile == "native_profile"

    def test_with_strictness_parameter(self):
        """Test strictness parameter works with all types."""
        report = MockProfileReport(
            source="test.csv",
            row_count=100,
            columns=[],
        )

        suite_loose = generate_suite(report, strictness="loose")
        suite_strict = generate_suite(report, strictness="strict")

        assert suite_loose.strictness.value == "loose"
        assert suite_strict.strictness.value == "strict"

    def test_with_custom_name(self):
        """Test custom name parameter."""
        report = MockProfileReport(
            source="data.csv",
            row_count=100,
            columns=[],
        )

        suite = generate_suite(report, name="my_custom_suite")

        assert suite.name == "my_custom_suite"


class TestProfileAdapterEdgeCases:
    """Test edge cases in profile adaptation."""

    def test_empty_columns(self):
        """Test handling empty columns list."""
        report = MockProfileReport(
            source="empty.csv",
            row_count=0,
            column_count=0,
            columns=[],
        )

        table_profile = ProfileAdapter.to_table_profile(report)

        assert table_profile.column_count == 0
        assert len(table_profile.columns) == 0

    def test_missing_column_fields(self):
        """Test handling columns with missing fields."""
        report = MockProfileReport(
            row_count=100,
            columns=[
                {"name": "col1"},  # Missing dtype, null_pct, unique_pct
            ],
        )

        table_profile = ProfileAdapter.to_table_profile(report)

        assert len(table_profile.columns) == 1
        col = table_profile.columns[0]
        assert col.name == "col1"
        assert col.inferred_type == DataType.UNKNOWN

    def test_zero_row_count(self):
        """Test handling zero row count."""
        report = MockProfileReport(
            row_count=0,
            columns=[
                {"name": "col", "dtype": "String", "null_pct": "0%", "unique_pct": "0%"},
            ],
        )

        table_profile = ProfileAdapter.to_table_profile(report)
        col = table_profile.columns[0]

        assert col.null_count == 0
        assert col.distinct_count == 0
