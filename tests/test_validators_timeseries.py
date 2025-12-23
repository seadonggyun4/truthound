"""Tests for time series validators."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from truthound.validators.timeseries import (
    TimeFrequency,
    TimeSeriesStats,
    TimeSeriesGapValidator,
    TimeSeriesIntervalValidator,
    TimeSeriesDuplicateValidator,
    MonotonicityType,
    TimeSeriesMonotonicValidator,
    TimeSeriesOrderValidator,
    SeasonalityValidator,
    SeasonalDecompositionValidator,
    TrendDirection,
    TrendValidator,
    TrendBreakValidator,
    TimeSeriesCompletenessValidator,
    TimeSeriesValueCompletenessValidator,
    TimeSeriesDateRangeValidator,
)


class TestTimeFrequency:
    """Tests for TimeFrequency enum."""

    def test_timedelta_conversion(self):
        """Test timedelta property."""
        assert TimeFrequency.HOURLY.timedelta == timedelta(hours=1)
        assert TimeFrequency.DAILY.timedelta == timedelta(days=1)
        assert TimeFrequency.WEEKLY.timedelta == timedelta(weeks=1)

    def test_seconds_property(self):
        """Test seconds property."""
        assert TimeFrequency.SECONDLY.seconds == 1.0
        assert TimeFrequency.MINUTELY.seconds == 60.0
        assert TimeFrequency.HOURLY.seconds == 3600.0


class TestTimeSeriesGapValidator:
    """Tests for TimeSeriesGapValidator."""

    @pytest.fixture
    def regular_hourly_data(self):
        """Create regular hourly time series."""
        base = datetime(2024, 1, 1)
        timestamps = [base + timedelta(hours=i) for i in range(24)]
        return pl.LazyFrame({"timestamp": timestamps, "value": range(24)})

    @pytest.fixture
    def gapped_data(self):
        """Create time series with gaps."""
        base = datetime(2024, 1, 1)
        timestamps = [
            base,
            base + timedelta(hours=1),
            base + timedelta(hours=2),
            # Gap: missing hours 3, 4
            base + timedelta(hours=5),
            base + timedelta(hours=6),
            # Gap: missing hours 7, 8, 9
            base + timedelta(hours=10),
        ]
        return pl.LazyFrame({"timestamp": timestamps, "value": range(len(timestamps))})

    def test_no_gaps(self, regular_hourly_data):
        """Test validation with no gaps."""
        validator = TimeSeriesGapValidator(
            timestamp_column="timestamp",
            frequency=TimeFrequency.HOURLY,
        )
        issues = validator.validate(regular_hourly_data)
        assert len(issues) == 0

    def test_detects_gaps(self, gapped_data):
        """Test gap detection."""
        validator = TimeSeriesGapValidator(
            timestamp_column="timestamp",
            frequency=TimeFrequency.HOURLY,
            max_gap_ratio=0.01,
        )
        issues = validator.validate(gapped_data)
        assert len(issues) == 1
        assert issues[0].issue_type == "timeseries_gaps_detected"
        assert issues[0].count >= 2  # At least 2 gaps

    def test_auto_infer_frequency(self, gapped_data):
        """Test frequency auto-inference."""
        validator = TimeSeriesGapValidator(
            timestamp_column="timestamp",
            frequency=None,
            max_gap_ratio=0.01,
        )
        issues = validator.validate(gapped_data)
        assert len(issues) == 1


class TestTimeSeriesIntervalValidator:
    """Tests for TimeSeriesIntervalValidator."""

    @pytest.fixture
    def irregular_data(self):
        """Create time series with irregular intervals."""
        base = datetime(2024, 1, 1)
        timestamps = [
            base,
            base + timedelta(seconds=30),  # Too short
            base + timedelta(minutes=5),
            base + timedelta(hours=2),  # Too long
            base + timedelta(hours=3),
        ]
        return pl.LazyFrame({"timestamp": timestamps})

    def test_detects_short_intervals(self, irregular_data):
        """Test detection of too-short intervals."""
        validator = TimeSeriesIntervalValidator(
            timestamp_column="timestamp",
            min_interval=timedelta(minutes=1),
        )
        issues = validator.validate(irregular_data)
        short_issues = [i for i in issues if i.issue_type == "interval_too_short"]
        assert len(short_issues) == 1

    def test_detects_long_intervals(self, irregular_data):
        """Test detection of too-long intervals."""
        validator = TimeSeriesIntervalValidator(
            timestamp_column="timestamp",
            max_interval=timedelta(hours=1),
        )
        issues = validator.validate(irregular_data)
        long_issues = [i for i in issues if i.issue_type == "interval_too_long"]
        assert len(long_issues) == 1


class TestTimeSeriesDuplicateValidator:
    """Tests for TimeSeriesDuplicateValidator."""

    @pytest.fixture
    def data_with_duplicates(self):
        """Create time series with duplicate timestamps."""
        base = datetime(2024, 1, 1)
        timestamps = [
            base,
            base,  # Duplicate
            base + timedelta(hours=1),
            base + timedelta(hours=2),
            base + timedelta(hours=2),  # Duplicate
            base + timedelta(hours=2),  # Duplicate
        ]
        return pl.LazyFrame({"timestamp": timestamps})

    def test_detects_duplicates(self, data_with_duplicates):
        """Test duplicate detection."""
        validator = TimeSeriesDuplicateValidator(
            timestamp_column="timestamp",
            max_duplicates=0,
        )
        issues = validator.validate(data_with_duplicates)
        assert len(issues) == 1
        assert issues[0].issue_type == "duplicate_timestamps"
        assert issues[0].count == 3  # 3 duplicate entries

    def test_allows_duplicates_within_limit(self, data_with_duplicates):
        """Test allowing duplicates up to a limit."""
        validator = TimeSeriesDuplicateValidator(
            timestamp_column="timestamp",
            max_duplicates=5,
            max_duplicate_ratio=1.0,  # Allow any ratio
        )
        issues = validator.validate(data_with_duplicates)
        assert len(issues) == 0


class TestTimeSeriesMonotonicValidator:
    """Tests for TimeSeriesMonotonicValidator."""

    @pytest.fixture
    def increasing_data(self):
        """Create monotonically increasing data."""
        base = datetime(2024, 1, 1)
        return pl.LazyFrame({
            "timestamp": [base + timedelta(hours=i) for i in range(10)],
            "value": [1, 2, 3, 5, 8, 13, 21, 34, 55, 89],
        })

    @pytest.fixture
    def non_monotonic_data(self):
        """Create non-monotonic data."""
        base = datetime(2024, 1, 1)
        return pl.LazyFrame({
            "timestamp": [base + timedelta(hours=i) for i in range(10)],
            "value": [1, 2, 3, 2, 5, 4, 7, 8, 9, 10],  # Violations at index 3, 5
        })

    def test_validates_increasing(self, increasing_data):
        """Test validation of increasing data."""
        validator = TimeSeriesMonotonicValidator(
            timestamp_column="timestamp",
            value_column="value",
            monotonicity=MonotonicityType.STRICTLY_INCREASING,
        )
        issues = validator.validate(increasing_data)
        assert len(issues) == 0

    def test_detects_violations(self, non_monotonic_data):
        """Test detection of monotonicity violations."""
        validator = TimeSeriesMonotonicValidator(
            timestamp_column="timestamp",
            value_column="value",
            monotonicity=MonotonicityType.STRICTLY_INCREASING,
        )
        issues = validator.validate(non_monotonic_data)
        assert len(issues) == 1
        assert issues[0].issue_type == "monotonicity_violation"
        assert issues[0].count == 2  # Two violations

    def test_non_decreasing(self):
        """Test non-decreasing validation."""
        base = datetime(2024, 1, 1)
        data = pl.LazyFrame({
            "timestamp": [base + timedelta(hours=i) for i in range(5)],
            "value": [1, 2, 2, 3, 3],  # Equal values allowed
        })
        validator = TimeSeriesMonotonicValidator(
            timestamp_column="timestamp",
            value_column="value",
            monotonicity=MonotonicityType.NON_DECREASING,
        )
        issues = validator.validate(data)
        assert len(issues) == 0


class TestTimeSeriesOrderValidator:
    """Tests for TimeSeriesOrderValidator."""

    @pytest.fixture
    def unordered_data(self):
        """Create out-of-order time series."""
        base = datetime(2024, 1, 1)
        timestamps = [
            base,
            base + timedelta(hours=2),
            base + timedelta(hours=1),  # Out of order
            base + timedelta(hours=3),
        ]
        return pl.LazyFrame({"timestamp": timestamps})

    def test_detects_unordered(self, unordered_data):
        """Test detection of out-of-order timestamps."""
        validator = TimeSeriesOrderValidator(
            timestamp_column="timestamp",
        )
        issues = validator.validate(unordered_data)
        assert len(issues) == 1
        assert issues[0].issue_type == "timestamp_order_violation"


class TestSeasonalityValidator:
    """Tests for SeasonalityValidator."""

    @pytest.fixture
    def seasonal_data(self):
        """Create data with strong seasonality."""
        base = datetime(2024, 1, 1)
        n = 100
        x = np.arange(n)
        # Create seasonal pattern with period 12
        values = 10 + 5 * np.sin(2 * np.pi * x / 12) + np.random.randn(n) * 0.5
        return pl.LazyFrame({
            "timestamp": [base + timedelta(days=i) for i in range(n)],
            "value": values.tolist(),
        })

    @pytest.fixture
    def non_seasonal_data(self):
        """Create non-seasonal data."""
        base = datetime(2024, 1, 1)
        n = 100
        values = np.random.randn(n) * 10
        return pl.LazyFrame({
            "timestamp": [base + timedelta(days=i) for i in range(n)],
            "value": values.tolist(),
        })

    def test_detects_expected_seasonality(self, seasonal_data):
        """Test detection of expected seasonality."""
        validator = SeasonalityValidator(
            timestamp_column="timestamp",
            value_column="value",
            expected_period=12,
            min_seasonality_strength=0.2,
        )
        issues = validator.validate(seasonal_data)
        # Should not report weak seasonality
        weak_issues = [i for i in issues if i.issue_type == "weak_seasonality"]
        assert len(weak_issues) == 0

    def test_detects_weak_seasonality(self, non_seasonal_data):
        """Test detection of weak/missing seasonality."""
        validator = SeasonalityValidator(
            timestamp_column="timestamp",
            value_column="value",
            expected_period=12,
            min_seasonality_strength=0.5,
        )
        issues = validator.validate(non_seasonal_data)
        assert len(issues) == 1
        assert issues[0].issue_type == "weak_seasonality"


class TestSeasonalDecompositionValidator:
    """Tests for SeasonalDecompositionValidator."""

    @pytest.fixture
    def noisy_data(self):
        """Create noisy data with high residual."""
        base = datetime(2024, 1, 1)
        n = 50
        values = np.random.randn(n) * 100
        return pl.LazyFrame({
            "timestamp": [base + timedelta(days=i) for i in range(n)],
            "value": values.tolist(),
        })

    def test_detects_high_residual(self, noisy_data):
        """Test detection of high residual variance."""
        validator = SeasonalDecompositionValidator(
            timestamp_column="timestamp",
            value_column="value",
            period=7,
            max_residual_ratio=0.1,
        )
        issues = validator.validate(noisy_data)
        residual_issues = [i for i in issues if i.issue_type == "high_residual_variance"]
        assert len(residual_issues) == 1


class TestTrendValidator:
    """Tests for TrendValidator."""

    @pytest.fixture
    def increasing_trend_data(self):
        """Create data with clear increasing trend."""
        base = datetime(2024, 1, 1)
        n = 50
        x = np.arange(n)
        values = 10 + 2 * x + np.random.randn(n) * 0.5
        return pl.LazyFrame({
            "timestamp": [base + timedelta(days=i) for i in range(n)],
            "value": values.tolist(),
        })

    @pytest.fixture
    def decreasing_trend_data(self):
        """Create data with decreasing trend."""
        base = datetime(2024, 1, 1)
        n = 50
        x = np.arange(n)
        values = 100 - 1.5 * x + np.random.randn(n) * 0.5
        return pl.LazyFrame({
            "timestamp": [base + timedelta(days=i) for i in range(n)],
            "value": values.tolist(),
        })

    def test_validates_increasing_trend(self, increasing_trend_data):
        """Test validation of expected increasing trend."""
        validator = TrendValidator(
            timestamp_column="timestamp",
            value_column="value",
            expected_direction=TrendDirection.INCREASING,
        )
        issues = validator.validate(increasing_trend_data)
        direction_issues = [i for i in issues if i.issue_type == "unexpected_trend_direction"]
        assert len(direction_issues) == 0

    def test_detects_wrong_direction(self, decreasing_trend_data):
        """Test detection of wrong trend direction."""
        validator = TrendValidator(
            timestamp_column="timestamp",
            value_column="value",
            expected_direction=TrendDirection.INCREASING,
        )
        issues = validator.validate(decreasing_trend_data)
        assert len(issues) >= 1
        assert any(i.issue_type == "unexpected_trend_direction" for i in issues)

    def test_detects_trend_changes(self, increasing_trend_data):
        """Test trend change detection."""
        validator = TrendValidator(
            timestamp_column="timestamp",
            value_column="value",
            detect_trend_change=True,
            window_size=10,
        )
        # Stable trend should have no changes
        issues = validator.validate(increasing_trend_data)
        change_issues = [i for i in issues if i.issue_type == "trend_direction_change"]
        assert len(change_issues) == 0


class TestTrendBreakValidator:
    """Tests for TrendBreakValidator."""

    @pytest.fixture
    def data_with_break(self):
        """Create data with structural break."""
        base = datetime(2024, 1, 1)
        n = 60
        values = []
        for i in range(n):
            if i < 30:
                values.append(50 + np.random.randn() * 2)
            else:
                values.append(100 + np.random.randn() * 2)  # Level shift
        return pl.LazyFrame({
            "timestamp": [base + timedelta(days=i) for i in range(n)],
            "value": values,
        })

    def test_detects_structural_break(self, data_with_break):
        """Test structural break detection."""
        validator = TrendBreakValidator(
            timestamp_column="timestamp",
            value_column="value",
            min_break_magnitude=0.3,
            window_size=10,
            max_breaks=0,
        )
        issues = validator.validate(data_with_break)
        assert len(issues) >= 1
        assert issues[0].issue_type == "structural_breaks_detected"


class TestTimeSeriesCompletenessValidator:
    """Tests for TimeSeriesCompletenessValidator."""

    @pytest.fixture
    def incomplete_data(self):
        """Create incomplete time series."""
        base = datetime(2024, 1, 1)
        # Only 50 points when we expect 100
        timestamps = [base + timedelta(hours=i * 2) for i in range(50)]
        return pl.LazyFrame({"timestamp": timestamps})

    @pytest.fixture
    def complete_data(self):
        """Create complete time series."""
        base = datetime(2024, 1, 1)
        timestamps = [base + timedelta(hours=i) for i in range(100)]
        return pl.LazyFrame({"timestamp": timestamps})

    def test_detects_incomplete_series(self, incomplete_data):
        """Test detection of incomplete series."""
        validator = TimeSeriesCompletenessValidator(
            timestamp_column="timestamp",
            frequency=TimeFrequency.HOURLY,
            min_coverage=0.9,
        )
        issues = validator.validate(incomplete_data)
        assert len(issues) >= 1
        assert any(i.issue_type == "incomplete_timeseries" for i in issues)

    def test_validates_complete_series(self, complete_data):
        """Test validation of complete series."""
        validator = TimeSeriesCompletenessValidator(
            timestamp_column="timestamp",
            frequency=TimeFrequency.HOURLY,
            min_coverage=0.95,
        )
        issues = validator.validate(complete_data)
        completeness_issues = [i for i in issues if i.issue_type == "incomplete_timeseries"]
        assert len(completeness_issues) == 0


class TestTimeSeriesValueCompletenessValidator:
    """Tests for TimeSeriesValueCompletenessValidator."""

    @pytest.fixture
    def data_with_nulls(self):
        """Create time series with null values."""
        base = datetime(2024, 1, 1)
        values = [1.0, 2.0, None, 4.0, None, None, 7.0, 8.0, None, 10.0]
        return pl.LazyFrame({
            "timestamp": [base + timedelta(hours=i) for i in range(10)],
            "value": values,
        })

    def test_detects_null_ratio(self, data_with_nulls):
        """Test null ratio detection."""
        validator = TimeSeriesValueCompletenessValidator(
            timestamp_column="timestamp",
            value_column="value",
            max_null_ratio=0.2,
        )
        issues = validator.validate(data_with_nulls)
        null_issues = [i for i in issues if i.issue_type == "excessive_null_values"]
        assert len(null_issues) == 1

    def test_detects_consecutive_nulls(self, data_with_nulls):
        """Test consecutive null detection."""
        validator = TimeSeriesValueCompletenessValidator(
            timestamp_column="timestamp",
            value_column="value",
            max_null_ratio=1.0,
            max_consecutive_nulls=1,
        )
        issues = validator.validate(data_with_nulls)
        consecutive_issues = [i for i in issues if i.issue_type == "consecutive_null_values"]
        assert len(consecutive_issues) == 1


class TestTimeSeriesDateRangeValidator:
    """Tests for TimeSeriesDateRangeValidator."""

    @pytest.fixture
    def partial_range_data(self):
        """Create data not covering full expected range."""
        start = datetime(2024, 1, 10)
        return pl.LazyFrame({
            "timestamp": [start + timedelta(days=i) for i in range(20)],
        })

    def test_detects_late_start(self, partial_range_data):
        """Test detection of late start."""
        validator = TimeSeriesDateRangeValidator(
            timestamp_column="timestamp",
            required_start=datetime(2024, 1, 1),
        )
        issues = validator.validate(partial_range_data)
        start_issues = [i for i in issues if "start" in i.issue_type]
        assert len(start_issues) == 1

    def test_detects_early_end(self, partial_range_data):
        """Test detection of early end."""
        validator = TimeSeriesDateRangeValidator(
            timestamp_column="timestamp",
            required_end=datetime(2024, 12, 31),
        )
        issues = validator.validate(partial_range_data)
        end_issues = [i for i in issues if "end" in i.issue_type]
        assert len(end_issues) == 1

    def test_validates_full_range(self):
        """Test validation of data covering full range."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)
        data = pl.LazyFrame({
            "timestamp": [start + timedelta(days=i) for i in range(31)],
        })
        validator = TimeSeriesDateRangeValidator(
            timestamp_column="timestamp",
            required_start=start,
            required_end=end,
        )
        issues = validator.validate(data)
        assert len(issues) == 0


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_dataframe(self):
        """Test handling of empty data."""
        data = pl.LazyFrame({"timestamp": [], "value": []})
        validator = TimeSeriesGapValidator(timestamp_column="timestamp")
        issues = validator.validate(data)
        assert len(issues) == 0

    def test_single_point(self):
        """Test handling of single data point."""
        data = pl.LazyFrame({
            "timestamp": [datetime(2024, 1, 1)],
            "value": [42.0],
        })
        validator = TimeSeriesMonotonicValidator(
            timestamp_column="timestamp",
            value_column="value",
        )
        issues = validator.validate(data)
        assert len(issues) == 0

    def test_two_points(self):
        """Test handling of two data points."""
        base = datetime(2024, 1, 1)
        data = pl.LazyFrame({
            "timestamp": [base, base + timedelta(hours=1)],
            "value": [1.0, 2.0],
        })
        validator = TrendValidator(
            timestamp_column="timestamp",
            value_column="value",
        )
        issues = validator.validate(data)
        # Should not crash, may or may not have issues
        assert isinstance(issues, list)
