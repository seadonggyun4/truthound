"""Tests for analytics aggregations."""

import pytest
from datetime import datetime, timedelta

from truthound.checkpoint.analytics.protocols import (
    TimeSeriesPoint,
    TimeGranularity,
    AggregationFunction,
)
from truthound.checkpoint.analytics.aggregations import (
    TimeBucketAggregation,
    RollupAggregation,
    RollupConfig,
    RollupLevel,
)


class TestTimeBucketAggregation:
    """Tests for TimeBucketAggregation."""

    def test_hourly_buckets(self) -> None:
        """Test hourly bucket aggregation."""
        agg = TimeBucketAggregation(
            granularity=TimeGranularity.HOUR,
            aggregation=AggregationFunction.AVG,
        )

        base_time = datetime(2024, 1, 1, 10, 0, 0)
        points = [
            TimeSeriesPoint(timestamp=base_time + timedelta(minutes=i * 10), value=float(i))
            for i in range(12)  # 2 hours of data
        ]

        buckets = agg.aggregate(points)

        assert len(buckets) == 2
        assert buckets[0].sample_count == 6
        assert buckets[1].sample_count == 6

    def test_sum_aggregation(self) -> None:
        """Test SUM aggregation function."""
        agg = TimeBucketAggregation(
            granularity=TimeGranularity.HOUR,
            aggregation=AggregationFunction.SUM,
        )

        base_time = datetime(2024, 1, 1, 10, 0, 0)
        points = [
            TimeSeriesPoint(timestamp=base_time + timedelta(minutes=i * 10), value=10.0)
            for i in range(6)
        ]

        buckets = agg.aggregate(points)

        assert len(buckets) == 1
        assert buckets[0].value == 60.0  # 6 * 10

    def test_min_max_aggregation(self) -> None:
        """Test MIN and MAX aggregation."""
        base_time = datetime(2024, 1, 1, 10, 0, 0)
        points = [
            TimeSeriesPoint(timestamp=base_time + timedelta(minutes=i * 10), value=float(i * 10))
            for i in range(6)
        ]

        min_agg = TimeBucketAggregation(
            granularity=TimeGranularity.HOUR,
            aggregation=AggregationFunction.MIN,
        )
        max_agg = TimeBucketAggregation(
            granularity=TimeGranularity.HOUR,
            aggregation=AggregationFunction.MAX,
        )

        min_buckets = min_agg.aggregate(points)
        max_buckets = max_agg.aggregate(points)

        assert min_buckets[0].value == 0.0
        assert max_buckets[0].value == 50.0

    def test_count_aggregation(self) -> None:
        """Test COUNT aggregation."""
        agg = TimeBucketAggregation(
            granularity=TimeGranularity.HOUR,
            aggregation=AggregationFunction.COUNT,
        )

        base_time = datetime(2024, 1, 1, 10, 0, 0)
        points = [
            TimeSeriesPoint(timestamp=base_time + timedelta(minutes=i * 10), value=float(i))
            for i in range(6)
        ]

        buckets = agg.aggregate(points)

        assert buckets[0].value == 6.0

    def test_stddev_aggregation(self) -> None:
        """Test STDDEV aggregation."""
        agg = TimeBucketAggregation(
            granularity=TimeGranularity.HOUR,
            aggregation=AggregationFunction.STDDEV,
        )

        base_time = datetime(2024, 1, 1, 10, 0, 0)
        # Values: 0, 10, 20, 30, 40, 50 - mean=25, stddev should be ~17.08
        points = [
            TimeSeriesPoint(timestamp=base_time + timedelta(minutes=i * 10), value=float(i * 10))
            for i in range(6)
        ]

        buckets = agg.aggregate(points)

        assert 17.0 < buckets[0].value < 18.0

    def test_percentile_aggregation(self) -> None:
        """Test percentile aggregation."""
        agg = TimeBucketAggregation(
            granularity=TimeGranularity.HOUR,
            aggregation=AggregationFunction.PERCENTILE_50,
        )

        base_time = datetime(2024, 1, 1, 10, 0, 0)
        # Values: 10, 20, 30, 40, 50 - median should be 30
        points = [
            TimeSeriesPoint(timestamp=base_time + timedelta(minutes=i * 10), value=float((i + 1) * 10))
            for i in range(5)
        ]

        buckets = agg.aggregate(points)

        assert buckets[0].value == 30.0

    def test_fill_gaps(self) -> None:
        """Test gap filling."""
        agg = TimeBucketAggregation(
            granularity=TimeGranularity.HOUR,
            aggregation=AggregationFunction.AVG,
            fill_gaps=True,
            gap_fill_value=0.0,
        )

        base_time = datetime(2024, 1, 1, 10, 0, 0)
        # Points only in hour 10 and hour 12 (skipping hour 11)
        points = [
            TimeSeriesPoint(timestamp=base_time, value=100.0),
            TimeSeriesPoint(timestamp=base_time + timedelta(hours=2), value=200.0),
        ]

        buckets = agg.aggregate(
            points,
            start=base_time,
            end=base_time + timedelta(hours=2),  # End at hour 12 to get exactly 3 buckets
        )

        assert len(buckets) == 3  # Hours 10, 11 (filled), 12
        assert buckets[0].value == 100.0  # Hour 10
        assert buckets[1].value == 0.0  # Hour 11: filled with gap value
        assert buckets[1].sample_count == 0
        assert buckets[2].value == 200.0  # Hour 12

    def test_bucket_to_point(self) -> None:
        """Test converting bucket to point."""
        agg = TimeBucketAggregation(
            granularity=TimeGranularity.HOUR,
            aggregation=AggregationFunction.AVG,
        )

        base_time = datetime(2024, 1, 1, 10, 0, 0)
        points = [
            TimeSeriesPoint(timestamp=base_time, value=100.0),
        ]

        buckets = agg.aggregate(points)
        point = buckets[0].to_point()

        assert point.timestamp == buckets[0].bucket_start
        assert point.value == 100.0
        assert "sample_count" in point.metadata

    def test_common_labels(self) -> None:
        """Test common label extraction."""
        agg = TimeBucketAggregation(
            granularity=TimeGranularity.HOUR,
            aggregation=AggregationFunction.AVG,
        )

        base_time = datetime(2024, 1, 1, 10, 0, 0)
        points = [
            TimeSeriesPoint(
                timestamp=base_time + timedelta(minutes=i * 10),
                value=float(i),
                labels={"env": "prod", "service": "api"},
            )
            for i in range(6)
        ]

        buckets = agg.aggregate(points)

        assert buckets[0].labels == {"env": "prod", "service": "api"}


class TestRollupAggregation:
    """Tests for RollupAggregation."""

    def test_default_levels(self) -> None:
        """Test that default levels are created."""
        rollup = RollupAggregation(use_defaults=True)

        assert RollupLevel.REALTIME in rollup._configs
        assert RollupLevel.SHORT_TERM in rollup._configs
        assert RollupLevel.MEDIUM_TERM in rollup._configs
        assert RollupLevel.LONG_TERM in rollup._configs

    def test_add_custom_level(self) -> None:
        """Test adding custom rollup level."""
        rollup = RollupAggregation(use_defaults=False)

        config = RollupConfig(
            level=RollupLevel.REALTIME,
            granularity=TimeGranularity.MINUTE,
            retention=timedelta(hours=2),
            aggregations=[AggregationFunction.AVG, AggregationFunction.MAX],
        )

        rollup.add_level(config)

        assert RollupLevel.REALTIME in rollup._configs
        assert rollup._configs[RollupLevel.REALTIME].retention == timedelta(hours=2)

    def test_add_points(self) -> None:
        """Test adding points to rollup."""
        rollup = RollupAggregation(use_defaults=False, auto_rollup=False)
        rollup.add_level(RollupConfig(
            level=RollupLevel.REALTIME,
            granularity=TimeGranularity.MINUTE,
            retention=timedelta(hours=1),
        ))

        base_time = datetime(2024, 1, 1, 10, 0, 0)
        points = [
            TimeSeriesPoint(timestamp=base_time + timedelta(seconds=i * 10), value=float(i))
            for i in range(18)  # 3 minutes of data
        ]

        results = rollup.add_points(points, level=RollupLevel.REALTIME)

        assert len(results) == 3  # 3 minute buckets
        for result in results:
            assert result.level == RollupLevel.REALTIME
            assert AggregationFunction.AVG in result.values

    def test_query(self) -> None:
        """Test querying rollup data."""
        rollup = RollupAggregation(use_defaults=False, auto_rollup=False)
        rollup.add_level(RollupConfig(
            level=RollupLevel.REALTIME,
            granularity=TimeGranularity.MINUTE,
            retention=timedelta(hours=1),
        ))

        base_time = datetime(2024, 1, 1, 10, 0, 0)
        points = [
            TimeSeriesPoint(timestamp=base_time + timedelta(seconds=i * 10), value=50.0)
            for i in range(12)
        ]

        rollup.add_points(points, level=RollupLevel.REALTIME)

        results = rollup.query(
            start=base_time,
            end=base_time + timedelta(hours=1),
            aggregation=AggregationFunction.AVG,
        )

        assert len(results) == 2  # 2 minute buckets
        for point in results:
            assert point.value == 50.0

    def test_cleanup_expired(self) -> None:
        """Test cleanup of expired data."""
        rollup = RollupAggregation(use_defaults=False, auto_rollup=False)
        rollup.add_level(RollupConfig(
            level=RollupLevel.REALTIME,
            granularity=TimeGranularity.MINUTE,
            retention=timedelta(minutes=5),
        ))

        now = datetime.now()
        old_time = now - timedelta(minutes=10)

        # Add old data
        old_points = [
            TimeSeriesPoint(timestamp=old_time + timedelta(seconds=i * 10), value=float(i))
            for i in range(6)
        ]
        rollup.add_points(old_points, level=RollupLevel.REALTIME)

        # Cleanup
        removed = rollup.cleanup_expired(now=now)

        assert removed[RollupLevel.REALTIME] > 0

    def test_get_stats(self) -> None:
        """Test getting rollup statistics."""
        rollup = RollupAggregation(use_defaults=False, auto_rollup=False)
        rollup.add_level(RollupConfig(
            level=RollupLevel.REALTIME,
            granularity=TimeGranularity.MINUTE,
            retention=timedelta(hours=1),
        ))

        base_time = datetime(2024, 1, 1, 10, 0, 0)
        points = [
            TimeSeriesPoint(timestamp=base_time + timedelta(seconds=i * 10), value=float(i))
            for i in range(6)
        ]
        rollup.add_points(points, level=RollupLevel.REALTIME)

        stats = rollup.get_stats()

        assert "levels" in stats
        assert "realtime" in stats["levels"]
        assert stats["levels"]["realtime"]["entry_count"] == 1

    def test_clear(self) -> None:
        """Test clearing rollup data."""
        rollup = RollupAggregation(use_defaults=True, auto_rollup=False)

        base_time = datetime.now()
        points = [
            TimeSeriesPoint(timestamp=base_time + timedelta(seconds=i * 10), value=float(i))
            for i in range(6)
        ]
        rollup.add_points(points, level=RollupLevel.REALTIME)

        # Clear specific level
        rollup.clear(RollupLevel.REALTIME)
        assert len(rollup._data[RollupLevel.REALTIME]) == 0

        # Add again and clear all
        rollup.add_points(points, level=RollupLevel.REALTIME)
        rollup.clear()

        for level in rollup._data:
            assert len(rollup._data[level]) == 0


class TestRollupConfig:
    """Tests for RollupConfig."""

    def test_valid_config(self) -> None:
        """Test valid configuration."""
        config = RollupConfig(
            level=RollupLevel.REALTIME,
            granularity=TimeGranularity.MINUTE,
            retention=timedelta(hours=1),
        )

        assert config.level == RollupLevel.REALTIME
        assert len(config.aggregations) > 0

    def test_invalid_retention(self) -> None:
        """Test that negative retention raises error."""
        with pytest.raises(ValueError, match="Retention must be positive"):
            RollupConfig(
                level=RollupLevel.REALTIME,
                granularity=TimeGranularity.MINUTE,
                retention=timedelta(hours=-1),
            )

    def test_custom_aggregations(self) -> None:
        """Test custom aggregation functions."""
        config = RollupConfig(
            level=RollupLevel.REALTIME,
            granularity=TimeGranularity.MINUTE,
            retention=timedelta(hours=1),
            aggregations=[AggregationFunction.SUM, AggregationFunction.COUNT],
        )

        assert len(config.aggregations) == 2
        assert AggregationFunction.SUM in config.aggregations
