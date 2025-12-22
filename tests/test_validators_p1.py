"""Tests for P1 validators: Statistical, JSON Schema, Multi-column, Datetime freshness, Casing."""

from datetime import datetime, timedelta, date
import json

import polars as pl
import pytest

from truthound.types import Severity


# =============================================================================
# Statistical Validators Tests
# =============================================================================


class TestStatisticalValidators:
    """Tests for statistical distribution validators."""

    def test_kl_divergence_similar_distribution(self):
        """Test KL divergence with similar distributions."""
        from truthound.validators import KLDivergenceValidator

        # Reference distribution
        reference = pl.DataFrame({
            "category": ["A"] * 50 + ["B"] * 30 + ["C"] * 20,
        })

        # Similar current distribution
        current = pl.DataFrame({
            "category": ["A"] * 48 + ["B"] * 32 + ["C"] * 20,
        })

        validator = KLDivergenceValidator(
            column="category",
            reference_data=reference,
            max_divergence=0.1,
        )
        issues = validator.validate(current.lazy())
        assert len(issues) == 0

    def test_kl_divergence_different_distribution(self):
        """Test KL divergence with different distributions."""
        from truthound.validators import KLDivergenceValidator

        reference = pl.DataFrame({
            "category": ["A"] * 50 + ["B"] * 30 + ["C"] * 20,
        })

        # Very different distribution
        current = pl.DataFrame({
            "category": ["A"] * 10 + ["B"] * 10 + ["C"] * 80,
        })

        validator = KLDivergenceValidator(
            column="category",
            reference_data=reference,
            max_divergence=0.1,
        )
        issues = validator.validate(current.lazy())

        assert len(issues) == 1
        assert issues[0].issue_type == "kl_divergence_exceeded"

    def test_kl_divergence_with_dict_distribution(self):
        """Test KL divergence with predefined distribution."""
        from truthound.validators import KLDivergenceValidator

        current = pl.DataFrame({
            "status": ["active"] * 50 + ["inactive"] * 50,
        })

        validator = KLDivergenceValidator(
            column="status",
            reference_distribution={"active": 0.5, "inactive": 0.5},
            max_divergence=0.05,
        )
        issues = validator.validate(current.lazy())
        assert len(issues) == 0

    def test_chi_square_matching_distribution(self):
        """Test chi-square with expected distribution."""
        from truthound.validators import ChiSquareValidator

        # Data matching expected frequencies
        df = pl.DataFrame({
            "category": ["A"] * 50 + ["B"] * 30 + ["C"] * 20,
        })

        validator = ChiSquareValidator(
            column="category",
            expected_frequencies={"A": 0.5, "B": 0.3, "C": 0.2},
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_chi_square_mismatched_distribution(self):
        """Test chi-square with mismatched distribution."""
        from truthound.validators import ChiSquareValidator

        # Data very different from expected
        df = pl.DataFrame({
            "category": ["A"] * 10 + ["B"] * 10 + ["C"] * 80,
        })

        validator = ChiSquareValidator(
            column="category",
            expected_frequencies={"A": 0.5, "B": 0.3, "C": 0.2},
        )
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].issue_type == "chi_square_distribution_mismatch"

    def test_most_common_value_expected(self):
        """Test most common value is in expected set."""
        from truthound.validators import MostCommonValueValidator

        df = pl.DataFrame({
            "country": ["US"] * 50 + ["UK"] * 30 + ["CA"] * 20,
        })

        validator = MostCommonValueValidator(
            column="country",
            expected_values=["US", "UK", "CA"],
            top_n=3,
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_most_common_value_unexpected(self):
        """Test unexpected most common value."""
        from truthound.validators import MostCommonValueValidator

        df = pl.DataFrame({
            "country": ["XX"] * 50 + ["YY"] * 30 + ["US"] * 20,
        })

        validator = MostCommonValueValidator(
            column="country",
            expected_values=["US", "UK", "CA"],
            top_n=2,
        )
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].issue_type == "unexpected_most_common_value"


# =============================================================================
# JSON Schema Validator Tests
# =============================================================================


class TestJsonSchemaValidator:
    """Tests for JSON schema validation."""

    def test_json_schema_valid(self):
        """Test valid JSON against schema."""
        from truthound.validators import JsonSchemaValidator

        df = pl.DataFrame({
            "config": [
                '{"name": "test", "version": "1.0.0"}',
                '{"name": "app", "version": "2.0.0"}',
            ]
        })

        validator = JsonSchemaValidator(
            column="config",
            schema={
                "type": "object",
                "required": ["name", "version"],
                "properties": {
                    "name": {"type": "string"},
                    "version": {"type": "string"},
                }
            }
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_json_schema_missing_required(self):
        """Test JSON missing required field."""
        from truthound.validators import JsonSchemaValidator

        df = pl.DataFrame({
            "config": [
                '{"name": "test"}',  # Missing version
                '{"name": "app", "version": "2.0.0"}',
            ]
        })

        validator = JsonSchemaValidator(
            column="config",
            schema={
                "type": "object",
                "required": ["name", "version"],
            }
        )
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].issue_type == "json_schema_violation"

    def test_json_schema_type_mismatch(self):
        """Test JSON with wrong type."""
        from truthound.validators import JsonSchemaValidator

        df = pl.DataFrame({
            "data": [
                '{"count": 10}',
                '{"count": "ten"}',  # Should be integer
            ]
        })

        validator = JsonSchemaValidator(
            column="data",
            schema={
                "type": "object",
                "properties": {
                    "count": {"type": "integer"},
                }
            }
        )
        issues = validator.validate(df.lazy())

        assert len(issues) == 1

    def test_json_schema_invalid_json(self):
        """Test invalid JSON string."""
        from truthound.validators import JsonSchemaValidator

        df = pl.DataFrame({
            "data": [
                '{"valid": true}',
                'not valid json',
            ]
        })

        validator = JsonSchemaValidator(
            column="data",
            schema={"type": "object"}
        )
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert "Invalid JSON" in issues[0].details


# =============================================================================
# Multi-Column Validators Tests
# =============================================================================


class TestMultiColumnValidators:
    """Tests for multi-column aggregate validators."""

    def test_multi_column_sum_equals_column(self):
        """Test sum of columns equals another column."""
        from truthound.validators import MultiColumnSumValidator

        df = pl.DataFrame({
            "q1": [10, 20, 30],
            "q2": [20, 30, 40],
            "q3": [30, 40, 50],
            "total": [60, 90, 120],
        })

        validator = MultiColumnSumValidator(
            columns=["q1", "q2", "q3"],
            equals_column="total",
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_multi_column_sum_mismatch(self):
        """Test sum mismatch detection."""
        from truthound.validators import MultiColumnSumValidator

        df = pl.DataFrame({
            "q1": [10, 20, 30],
            "q2": [20, 30, 40],
            "q3": [30, 40, 50],
            "total": [60, 100, 120],  # Second row is wrong
        })

        validator = MultiColumnSumValidator(
            columns=["q1", "q2", "q3"],
            equals_column="total",
        )
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].issue_type == "multi_column_sum_mismatch"
        assert issues[0].count == 1

    def test_multi_column_sum_equals_value(self):
        """Test sum equals fixed value."""
        from truthound.validators import MultiColumnSumValidator

        df = pl.DataFrame({
            "a": [25, 30, 35],
            "b": [25, 30, 35],
            "c": [25, 20, 30],
            "d": [25, 20, 0],
        })

        validator = MultiColumnSumValidator(
            columns=["a", "b", "c", "d"],
            equals_value=100,
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_multi_column_calculation(self):
        """Test multi-column calculation validation."""
        from truthound.validators import MultiColumnCalculationValidator

        df = pl.DataFrame({
            "revenue": [100, 200, 300],
            "cost": [60, 120, 180],
            "profit": [40, 80, 120],
        })

        validator = MultiColumnCalculationValidator(
            left_column="revenue",
            operator="-",
            right_column="cost",
            equals_column="profit",
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 0


# =============================================================================
# Datetime Freshness Validators Tests
# =============================================================================


class TestDatetimeFreshnessValidators:
    """Tests for datetime freshness validators."""

    def test_recent_data_fresh(self):
        """Test data is recent."""
        from truthound.validators import RecentDataValidator

        now = datetime.now()
        df = pl.DataFrame({
            "created_at": [now - timedelta(hours=1), now - timedelta(hours=2)],
        })

        validator = RecentDataValidator(
            column="created_at",
            max_age_hours=24,
            reference_time=now,
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_recent_data_stale(self):
        """Test stale data detection."""
        from truthound.validators import RecentDataValidator

        now = datetime.now()
        df = pl.DataFrame({
            "created_at": [now - timedelta(hours=48), now - timedelta(hours=72)],
        })

        validator = RecentDataValidator(
            column="created_at",
            max_age_hours=24,
            reference_time=now,
        )
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].issue_type == "stale_data"

    def test_date_part_coverage_complete(self):
        """Test complete date coverage."""
        from truthound.validators import DatePartCoverageValidator

        # 7 consecutive days
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(7)]
        df = pl.DataFrame({"date": dates})

        validator = DatePartCoverageValidator(
            column="date",
            date_part="day",
            min_coverage=1.0,
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_date_part_coverage_with_gaps(self):
        """Test date coverage with gaps."""
        from truthound.validators import DatePartCoverageValidator

        # Days with gaps
        dates = [
            datetime(2024, 1, 1),
            datetime(2024, 1, 2),
            # Jan 3 missing
            datetime(2024, 1, 4),
            datetime(2024, 1, 5),
        ]
        df = pl.DataFrame({"date": dates})

        validator = DatePartCoverageValidator(
            column="date",
            date_part="day",
            min_coverage=1.0,
        )
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].issue_type == "date_part_coverage_gap"

    def test_grouped_recent_data(self):
        """Test grouped recent data validation."""
        from truthound.validators import GroupedRecentDataValidator

        now = datetime.now()
        df = pl.DataFrame({
            "store_id": ["A", "A", "B", "B"],
            "timestamp": [
                now - timedelta(hours=1),
                now - timedelta(hours=2),
                now - timedelta(hours=1),
                now - timedelta(hours=2),
            ],
        })

        validator = GroupedRecentDataValidator(
            datetime_column="timestamp",
            group_column="store_id",
            max_age_hours=24,
            reference_time=now,
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_grouped_recent_data_stale_group(self):
        """Test grouped data with stale group."""
        from truthound.validators import GroupedRecentDataValidator

        now = datetime.now()
        df = pl.DataFrame({
            "store_id": ["A", "A", "B", "B"],
            "timestamp": [
                now - timedelta(hours=1),  # Store A is fresh
                now - timedelta(hours=2),
                now - timedelta(hours=48),  # Store B is stale
                now - timedelta(hours=72),
            ],
        })

        validator = GroupedRecentDataValidator(
            datetime_column="timestamp",
            group_column="store_id",
            max_age_hours=24,
            reference_time=now,
        )
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].issue_type == "grouped_stale_data"
        assert issues[0].count == 1


# =============================================================================
# Casing Validators Tests
# =============================================================================


class TestCasingValidators:
    """Tests for string casing validators."""

    def test_consistent_casing_upper(self):
        """Test consistent uppercase."""
        from truthound.validators import ConsistentCasingValidator

        df = pl.DataFrame({
            "code": ["ABC", "DEF", "GHI"],
        })

        validator = ConsistentCasingValidator(
            column="code",
            expected_casing="upper",
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_consistent_casing_violation(self):
        """Test casing violation detection."""
        from truthound.validators import ConsistentCasingValidator

        df = pl.DataFrame({
            "code": ["ABC", "def", "GHI"],  # "def" is lowercase
        })

        validator = ConsistentCasingValidator(
            column="code",
            expected_casing="upper",
        )
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].issue_type == "casing_violation"

    def test_consistent_casing_title(self):
        """Test title case validation."""
        from truthound.validators import ConsistentCasingValidator

        df = pl.DataFrame({
            "name": ["John Doe", "Jane Smith", "Bob Wilson"],
        })

        validator = ConsistentCasingValidator(
            column="name",
            expected_casing="title",
        )
        issues = validator.validate(df.lazy())
        assert len(issues) == 0

    def test_inconsistent_casing_auto_detect(self):
        """Test auto-detection of inconsistent casing."""
        from truthound.validators import ConsistentCasingValidator

        df = pl.DataFrame({
            "status": ["ACTIVE", "INACTIVE", "pending"],  # Mixed upper and lower
        })

        validator = ConsistentCasingValidator(column="status")
        issues = validator.validate(df.lazy())

        assert len(issues) == 1
        assert issues[0].issue_type == "inconsistent_casing"


# =============================================================================
# Integration Tests
# =============================================================================


class TestP1Integration:
    """Integration tests for P1 validators."""

    def test_etl_data_quality_check(self):
        """Test comprehensive ETL data quality check."""
        from truthound.validators import (
            MultiColumnSumValidator,
            JsonSchemaValidator,
            ConsistentCasingValidator,
        )

        df = pl.DataFrame({
            "region": ["NORTH", "SOUTH", "EAST", "WEST"],
            "q1_sales": [100, 200, 150, 250],
            "q2_sales": [110, 210, 160, 260],
            "total_sales": [210, 410, 310, 510],
            "metadata": [
                '{"source": "crm", "verified": true}',
                '{"source": "erp", "verified": true}',
                '{"source": "crm", "verified": false}',
                '{"source": "erp", "verified": true}',
            ],
        })

        # Check sum
        sum_validator = MultiColumnSumValidator(
            columns=["q1_sales", "q2_sales"],
            equals_column="total_sales",
        )
        assert len(sum_validator.validate(df.lazy())) == 0

        # Check JSON
        json_validator = JsonSchemaValidator(
            column="metadata",
            schema={
                "type": "object",
                "required": ["source", "verified"],
            }
        )
        assert len(json_validator.validate(df.lazy())) == 0

        # Check casing
        casing_validator = ConsistentCasingValidator(
            column="region",
            expected_casing="upper",
        )
        assert len(casing_validator.validate(df.lazy())) == 0

    def test_time_series_monitoring(self):
        """Test time series data monitoring."""
        from truthound.validators import (
            RecentDataValidator,
            DatePartCoverageValidator,
            KLDivergenceValidator,
        )

        now = datetime.now()

        # Time series data
        df = pl.DataFrame({
            "timestamp": [now - timedelta(hours=i) for i in range(24)],
            "category": ["A"] * 12 + ["B"] * 12,
            "value": list(range(24)),
        })

        # Check freshness
        fresh_validator = RecentDataValidator(
            column="timestamp",
            max_age_hours=1,
            reference_time=now,
        )
        assert len(fresh_validator.validate(df.lazy())) == 0

        # Check distribution
        dist_validator = KLDivergenceValidator(
            column="category",
            reference_distribution={"A": 0.5, "B": 0.5},
            max_divergence=0.1,
        )
        assert len(dist_validator.validate(df.lazy())) == 0
