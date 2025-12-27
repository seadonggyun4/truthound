"""Tests for distributed execution mixins.

This module tests the mixin classes that provide additional functionality
to distributed execution engines.
"""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_engine():
    """Create a mock engine implementing DistributedEngineProtocol."""
    engine = MagicMock()

    # Basic operations
    engine.count_rows.return_value = 1000
    engine.get_columns.return_value = ["id", "name", "value", "category"]
    engine.count_nulls.return_value = 50
    engine.count_nulls_all.return_value = {
        "id": 0,
        "name": 50,
        "value": 100,
        "category": 0,
    }
    engine.count_distinct.return_value = 500
    engine.get_stats.return_value = {
        "count": 900,
        "null_count": 100,
        "mean": 50.5,
        "std": 10.2,
        "min": 0.0,
        "max": 100.0,
    }

    # Caching
    cache = {}

    def cache_key(*args):
        return str(args)

    def get_cached(key):
        return cache.get(key)

    def set_cached(key, value):
        cache[key] = value

    engine._cache_key = cache_key
    engine._get_cached = get_cached
    engine._set_cached = set_cached

    return engine


# =============================================================================
# StatisticalMixin Tests
# =============================================================================


class TestStatisticalMixin:
    """Tests for StatisticalMixin."""

    def test_get_percentiles_delegates_to_get_quantiles(self, mock_engine):
        """Test that get_percentiles uses get_quantiles if available."""
        from truthound.execution.distributed.mixins import StatisticalMixin

        # Add the mixin method to mock engine
        mock_engine.get_quantiles = MagicMock(return_value=[25.0, 50.0, 75.0])

        # Call mixin method directly
        result = StatisticalMixin.get_percentiles(mock_engine, "value")

        assert result == {"p25": 25.0, "p50": 50.0, "p75": 75.0}
        mock_engine.get_quantiles.assert_called_once_with("value", [0.25, 0.5, 0.75])

    def test_get_percentiles_custom_percentiles(self, mock_engine):
        """Test custom percentile values."""
        from truthound.execution.distributed.mixins import StatisticalMixin

        mock_engine.get_quantiles = MagicMock(return_value=[10.0, 90.0])

        result = StatisticalMixin.get_percentiles(
            mock_engine,
            "value",
            percentiles=[0.1, 0.9],
        )

        assert result == {"p10": 10.0, "p90": 90.0}


# =============================================================================
# DataQualityMixin Tests
# =============================================================================


class TestDataQualityMixin:
    """Tests for DataQualityMixin."""

    def test_get_completeness_single_column(self, mock_engine):
        """Test completeness for a single column."""
        from truthound.execution.distributed.mixins import DataQualityMixin

        # 50 nulls out of 1000 = 95% complete
        result = DataQualityMixin.get_completeness(mock_engine, "name")

        assert result == 0.95

    def test_get_completeness_all_columns(self, mock_engine):
        """Test completeness for all columns."""
        from truthound.execution.distributed.mixins import DataQualityMixin

        result = DataQualityMixin.get_completeness(mock_engine)

        assert isinstance(result, dict)
        assert result["id"] == 1.0  # 0 nulls
        assert result["name"] == 0.95  # 50 nulls
        assert result["value"] == 0.9  # 100 nulls
        assert result["category"] == 1.0  # 0 nulls

    def test_get_completeness_empty_data(self, mock_engine):
        """Test completeness with empty data."""
        from truthound.execution.distributed.mixins import DataQualityMixin

        mock_engine.count_rows.return_value = 0

        result = DataQualityMixin.get_completeness(mock_engine, "name")

        assert result == 1.0

    def test_get_uniqueness_single_column(self, mock_engine):
        """Test uniqueness for a single column."""
        from truthound.execution.distributed.mixins import DataQualityMixin

        # 500 distinct out of 1000 = 50% unique
        result = DataQualityMixin.get_uniqueness(mock_engine, "name")

        assert result == 0.5

    def test_get_uniqueness_all_columns(self, mock_engine):
        """Test uniqueness for all columns."""
        from truthound.execution.distributed.mixins import DataQualityMixin

        # Each column returns 500 distinct
        result = DataQualityMixin.get_uniqueness(mock_engine)

        assert isinstance(result, dict)
        for col in ["id", "name", "value", "category"]:
            assert result[col] == 0.5

    def test_get_data_quality_report(self, mock_engine):
        """Test data quality report generation."""
        from truthound.execution.distributed.mixins import DataQualityMixin

        report = DataQualityMixin.get_data_quality_report(mock_engine)

        assert report.total_rows == 1000
        assert report.total_columns == 4
        assert isinstance(report.completeness, dict)
        assert isinstance(report.uniqueness, dict)
        assert 0 <= report.overall_completeness <= 1
        assert 0 <= report.overall_score <= 1

    def test_check_null_threshold_pass(self, mock_engine):
        """Test null threshold check - passing."""
        from truthound.execution.distributed.mixins import DataQualityMixin

        # 5% null rate, threshold is 10%
        result = DataQualityMixin.check_null_threshold(
            mock_engine, "name", threshold=0.1
        )

        assert result is True

    def test_check_null_threshold_fail(self, mock_engine):
        """Test null threshold check - failing."""
        from truthound.execution.distributed.mixins import DataQualityMixin

        # 50 nulls out of 1000 = 5% null rate, threshold is 4%
        # Mock get_completeness to return 0.95 (95% complete = 5% null)
        # check_null_threshold calls self.get_completeness() which is on the mock
        mock_engine.get_completeness = MagicMock(return_value=0.95)

        result = DataQualityMixin.check_null_threshold(
            mock_engine, "name", threshold=0.04  # 4% threshold, 5% null = fail
        )

        assert result is False

    def test_check_unique_threshold_pass(self, mock_engine):
        """Test uniqueness threshold check - passing."""
        from truthound.execution.distributed.mixins import DataQualityMixin

        # 50% unique, threshold is 40%
        result = DataQualityMixin.check_unique_threshold(
            mock_engine, "name", threshold=0.4
        )

        assert result is True

    def test_check_unique_threshold_fail(self, mock_engine):
        """Test uniqueness threshold check - failing."""
        from truthound.execution.distributed.mixins import DataQualityMixin

        # count_distinct returns 500, count_rows returns 1000
        # So uniqueness = 500/1000 = 0.5 (50%)
        # Mock get_uniqueness to return 0.5 (50% unique)
        # check_unique_threshold calls self.get_uniqueness() which is on the mock
        mock_engine.get_uniqueness = MagicMock(return_value=0.5)

        # Threshold of 60% should fail
        result = DataQualityMixin.check_unique_threshold(
            mock_engine, "name", threshold=0.6  # 60% required, only 50% = fail
        )

        assert result is False


# =============================================================================
# PartitioningMixin Tests
# =============================================================================


class TestPartitioningMixin:
    """Tests for PartitioningMixin."""

    def test_get_partition_sizes_from_info(self, mock_engine):
        """Test getting partition sizes from partition info."""
        from truthound.execution.distributed.mixins import PartitioningMixin
        from truthound.execution.distributed.protocols import PartitionInfo

        mock_engine._get_partition_info = MagicMock(return_value=[
            PartitionInfo(0, 4, row_start=0, row_end=250),
            PartitionInfo(1, 4, row_start=250, row_end=500),
            PartitionInfo(2, 4, row_start=500, row_end=750),
            PartitionInfo(3, 4, row_start=750, row_end=1000),
        ])

        result = PartitioningMixin.get_partition_sizes(mock_engine)

        assert result == [250, 250, 250, 250]

    def test_check_partition_skew_not_skewed(self, mock_engine):
        """Test partition skew check - not skewed."""
        from truthound.execution.distributed.mixins import PartitioningMixin
        from truthound.execution.distributed.protocols import PartitionInfo

        mock_engine._get_partition_info = MagicMock(return_value=[
            PartitionInfo(0, 4, row_start=0, row_end=250),
            PartitionInfo(1, 4, row_start=250, row_end=500),
            PartitionInfo(2, 4, row_start=500, row_end=750),
            PartitionInfo(3, 4, row_start=750, row_end=1000),
        ])

        result = PartitioningMixin.check_partition_skew(mock_engine, threshold=3.0)

        assert result is False

    def test_check_partition_skew_is_skewed(self, mock_engine):
        """Test partition skew check - is skewed."""
        from truthound.execution.distributed.mixins import PartitioningMixin

        # Very uneven partitions: 100, 100, 100, 700
        # Mock get_partition_sizes to return the expected sizes
        # check_partition_skew calls self.get_partition_sizes() which is on the mock
        mock_engine.get_partition_sizes = MagicMock(return_value=[100, 100, 100, 700])

        # Max/min ratio is 700/100 = 7, which exceeds threshold of 3.0
        result = PartitioningMixin.check_partition_skew(mock_engine, threshold=3.0)

        assert result is True


# =============================================================================
# ValidationMixin Tests
# =============================================================================


class TestValidationMixin:
    """Tests for ValidationMixin."""

    def test_validate_not_null_pass(self, mock_engine):
        """Test not null validation - passing."""
        from truthound.execution.distributed.mixins import ValidationMixin

        # All columns have < 10% null rate
        result = ValidationMixin.validate_not_null(
            mock_engine,
            columns=["id", "category"],
            threshold=0.1,
        )

        assert result.passed is True
        assert "passed" in result.message.lower()

    def test_validate_not_null_fail(self, mock_engine):
        """Test not null validation - failing."""
        from truthound.execution.distributed.mixins import ValidationMixin

        # value has 10% null rate, threshold is 5%
        result = ValidationMixin.validate_not_null(
            mock_engine,
            columns=["value"],
            threshold=0.05,
        )

        assert result.passed is False
        assert "1 columns exceeded" in result.message

    def test_validate_not_null_empty_data(self, mock_engine):
        """Test not null validation with empty data."""
        from truthound.execution.distributed.mixins import ValidationMixin

        mock_engine.count_rows.return_value = 0

        result = ValidationMixin.validate_not_null(mock_engine)

        assert result.passed is True
        assert "no data" in result.message.lower()

    def test_validate_unique_pass(self, mock_engine):
        """Test unique validation - passing."""
        from truthound.execution.distributed.mixins import ValidationMixin

        mock_engine.count_duplicates = MagicMock(return_value=0)

        result = ValidationMixin.validate_unique(mock_engine, ["id"])

        assert result.passed is True
        assert "unique" in result.message.lower()

    def test_validate_unique_fail(self, mock_engine):
        """Test unique validation - failing."""
        from truthound.execution.distributed.mixins import ValidationMixin

        mock_engine.count_duplicates = MagicMock(return_value=100)

        result = ValidationMixin.validate_unique(mock_engine, ["name"])

        assert result.passed is False
        assert "100" in result.message

    def test_validate_range_pass(self, mock_engine):
        """Test range validation - passing."""
        from truthound.execution.distributed.mixins import ValidationMixin

        result = ValidationMixin.validate_range(
            mock_engine,
            "value",
            min_value=-10,
            max_value=200,
        )

        assert result.passed is True
        assert "in range" in result.message.lower()

    def test_validate_range_fail_below_min(self, mock_engine):
        """Test range validation - failing below min."""
        from truthound.execution.distributed.mixins import ValidationMixin

        result = ValidationMixin.validate_range(
            mock_engine,
            "value",
            min_value=10,  # Actual min is 0
        )

        assert result.passed is False
        assert "below" in result.message.lower()

    def test_validate_range_fail_above_max(self, mock_engine):
        """Test range validation - failing above max."""
        from truthound.execution.distributed.mixins import ValidationMixin

        result = ValidationMixin.validate_range(
            mock_engine,
            "value",
            max_value=50,  # Actual max is 100
        )

        assert result.passed is False
        assert "above" in result.message.lower()


# =============================================================================
# IOOperationsMixin Tests
# =============================================================================


class TestIOOperationsMixin:
    """Tests for IOOperationsMixin."""

    def test_write_parquet_delegates(self, mock_engine, tmp_path):
        """Test that write_parquet delegates to backend."""
        from truthound.execution.distributed.mixins import IOOperationsMixin

        mock_engine._write_parquet = MagicMock()

        path = str(tmp_path / "output.parquet")
        IOOperationsMixin.write_parquet(mock_engine, path)

        mock_engine._write_parquet.assert_called_once()

    def test_write_csv_delegates(self, mock_engine, tmp_path):
        """Test that write_csv delegates to backend."""
        from truthound.execution.distributed.mixins import IOOperationsMixin

        mock_engine._write_csv = MagicMock()

        path = str(tmp_path / "output.csv")
        IOOperationsMixin.write_csv(mock_engine, path)

        mock_engine._write_csv.assert_called_once()

    def test_write_parquet_fallback_to_polars(self, mock_engine, tmp_path):
        """Test fallback to Polars for parquet writing."""
        pytest.importorskip("polars")

        from truthound.execution.distributed.mixins import IOOperationsMixin
        import polars as pl

        # Remove backend-specific method
        if hasattr(mock_engine, "_write_parquet"):
            del mock_engine._write_parquet

        # Mock Polars conversion
        mock_lf = MagicMock(spec=pl.LazyFrame)
        mock_df = MagicMock(spec=pl.DataFrame)
        mock_lf.collect.return_value = mock_df
        mock_engine.to_polars_lazyframe = MagicMock(return_value=mock_lf)

        path = str(tmp_path / "output.parquet")
        IOOperationsMixin.write_parquet(mock_engine, path)

        mock_df.write_parquet.assert_called_once()


# =============================================================================
# FullFeaturedMixin Tests
# =============================================================================


class TestFullFeaturedMixin:
    """Tests for FullFeaturedMixin (combined mixin)."""

    def test_has_all_mixin_methods(self):
        """Test that FullFeaturedMixin includes all mixin methods."""
        from truthound.execution.distributed.mixins import FullFeaturedMixin

        # Statistical methods
        assert hasattr(FullFeaturedMixin, "get_percentiles")
        assert hasattr(FullFeaturedMixin, "get_skewness")
        assert hasattr(FullFeaturedMixin, "get_kurtosis")
        assert hasattr(FullFeaturedMixin, "get_correlation")

        # Data quality methods
        assert hasattr(FullFeaturedMixin, "get_completeness")
        assert hasattr(FullFeaturedMixin, "get_uniqueness")
        assert hasattr(FullFeaturedMixin, "get_data_quality_report")

        # Partitioning methods
        assert hasattr(FullFeaturedMixin, "get_partition_sizes")
        assert hasattr(FullFeaturedMixin, "check_partition_skew")

        # IO methods
        assert hasattr(FullFeaturedMixin, "write_parquet")
        assert hasattr(FullFeaturedMixin, "write_csv")
        assert hasattr(FullFeaturedMixin, "write_json")

        # Validation methods
        assert hasattr(FullFeaturedMixin, "validate_not_null")
        assert hasattr(FullFeaturedMixin, "validate_unique")
        assert hasattr(FullFeaturedMixin, "validate_range")


# =============================================================================
# DataQualityReport Tests
# =============================================================================


class TestDataQualityReport:
    """Tests for DataQualityReport dataclass."""

    def test_data_quality_report_creation(self):
        """Test creating a data quality report."""
        from truthound.execution.distributed.mixins import DataQualityReport

        report = DataQualityReport(
            total_rows=1000,
            total_columns=4,
            completeness={"a": 0.95, "b": 0.90},
            uniqueness={"a": 0.80, "b": 0.60},
            overall_completeness=0.925,
            overall_score=0.8125,
        )

        assert report.total_rows == 1000
        assert report.total_columns == 4
        assert report.completeness["a"] == 0.95
        assert report.uniqueness["b"] == 0.60


# =============================================================================
# ValidationResult Tests
# =============================================================================


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_validation_result_passed(self):
        """Test creating a passing validation result."""
        from truthound.execution.distributed.mixins import ValidationResult

        result = ValidationResult(
            passed=True,
            message="All checks passed",
            details={"checked_columns": ["a", "b"]},
        )

        assert result.passed is True
        assert "passed" in result.message.lower()
        assert result.details["checked_columns"] == ["a", "b"]

    def test_validation_result_failed(self):
        """Test creating a failing validation result."""
        from truthound.execution.distributed.mixins import ValidationResult

        result = ValidationResult(
            passed=False,
            message="Found 10 violations",
            details={"violation_count": 10},
        )

        assert result.passed is False
        assert "10" in result.message
