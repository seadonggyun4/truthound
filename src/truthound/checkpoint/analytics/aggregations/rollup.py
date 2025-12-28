"""Rollup aggregation for time series data.

Provides multi-level rollup aggregations for efficient
long-term storage and fast querying of historical data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from truthound.checkpoint.analytics.protocols import (
    TimeSeriesPoint,
    TimeGranularity,
    AggregationFunction,
)
from truthound.checkpoint.analytics.aggregations.time_bucket import (
    TimeBucketAggregation,
    BucketResult,
)

logger = logging.getLogger(__name__)


class RollupLevel(Enum):
    """Pre-defined rollup levels for common use cases."""

    # Real-time: 1-minute buckets, keep 1 hour
    REALTIME = "realtime"

    # Short-term: 5-minute buckets, keep 24 hours
    SHORT_TERM = "short_term"

    # Medium-term: 1-hour buckets, keep 7 days
    MEDIUM_TERM = "medium_term"

    # Long-term: 1-day buckets, keep 90 days
    LONG_TERM = "long_term"

    # Archive: 1-week buckets, keep forever
    ARCHIVE = "archive"


@dataclass
class RollupConfig:
    """Configuration for a rollup level.

    Attributes:
        level: Rollup level identifier.
        granularity: Time bucket size.
        retention: How long to keep data at this level.
        aggregations: Aggregation functions to compute.
    """

    level: RollupLevel
    granularity: TimeGranularity
    retention: timedelta
    aggregations: list[AggregationFunction] = field(
        default_factory=lambda: [
            AggregationFunction.AVG,
            AggregationFunction.MIN,
            AggregationFunction.MAX,
            AggregationFunction.COUNT,
        ]
    )

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.retention.total_seconds() <= 0:
            raise ValueError("Retention must be positive")


@dataclass
class RollupResult:
    """Result of a rollup aggregation.

    Attributes:
        level: Rollup level.
        bucket_start: Start of the bucket.
        bucket_end: End of the bucket.
        values: Dictionary of aggregation -> value.
        sample_count: Number of raw samples.
        labels: Common labels.
    """

    level: RollupLevel
    bucket_start: datetime
    bucket_end: datetime
    values: dict[AggregationFunction, float]
    sample_count: int = 0
    labels: dict[str, str] = field(default_factory=dict)

    def get_value(self, aggregation: AggregationFunction) -> float | None:
        """Get value for a specific aggregation."""
        return self.values.get(aggregation)

    def to_point(self, aggregation: AggregationFunction) -> TimeSeriesPoint | None:
        """Convert to TimeSeriesPoint for a specific aggregation."""
        value = self.get_value(aggregation)
        if value is None:
            return None

        return TimeSeriesPoint(
            timestamp=self.bucket_start,
            value=value,
            labels=self.labels,
            metadata={
                "rollup_level": self.level.value,
                "bucket_end": self.bucket_end.isoformat(),
                "sample_count": self.sample_count,
                "aggregation": aggregation.value,
            },
        )


class RollupAggregation:
    """Multi-level rollup aggregation system.

    Maintains multiple levels of aggregation with different granularities
    and retention periods for efficient storage and querying.

    Example:
        >>> rollup = RollupAggregation()
        >>> rollup.add_level(RollupConfig(
        ...     level=RollupLevel.REALTIME,
        ...     granularity=TimeGranularity.MINUTE,
        ...     retention=timedelta(hours=1),
        ... ))
        >>> rollup.add_level(RollupConfig(
        ...     level=RollupLevel.SHORT_TERM,
        ...     granularity=TimeGranularity.HOUR,
        ...     retention=timedelta(days=1),
        ... ))
        >>>
        >>> # Add raw points
        >>> for point in points:
        ...     rollup.add_point(point)
        >>>
        >>> # Query at appropriate level
        >>> results = rollup.query(start, end, preferred_granularity=TimeGranularity.HOUR)
    """

    # Default rollup configurations
    DEFAULT_CONFIGS = [
        RollupConfig(
            level=RollupLevel.REALTIME,
            granularity=TimeGranularity.MINUTE,
            retention=timedelta(hours=1),
        ),
        RollupConfig(
            level=RollupLevel.SHORT_TERM,
            granularity=TimeGranularity.HOUR,
            retention=timedelta(days=1),
        ),
        RollupConfig(
            level=RollupLevel.MEDIUM_TERM,
            granularity=TimeGranularity.DAY,
            retention=timedelta(days=7),
        ),
        RollupConfig(
            level=RollupLevel.LONG_TERM,
            granularity=TimeGranularity.WEEK,
            retention=timedelta(days=90),
        ),
    ]

    def __init__(
        self,
        use_defaults: bool = True,
        auto_rollup: bool = True,
    ) -> None:
        """Initialize rollup aggregation.

        Args:
            use_defaults: Whether to use default configurations.
            auto_rollup: Whether to automatically roll up data.
        """
        self._configs: dict[RollupLevel, RollupConfig] = {}
        self._aggregators: dict[RollupLevel, dict[AggregationFunction, TimeBucketAggregation]] = {}
        self._data: dict[RollupLevel, list[RollupResult]] = {}
        self._auto_rollup = auto_rollup

        if use_defaults:
            for config in self.DEFAULT_CONFIGS:
                self.add_level(config)

    def add_level(self, config: RollupConfig) -> "RollupAggregation":
        """Add a rollup level.

        Args:
            config: Rollup configuration.

        Returns:
            Self for chaining.
        """
        self._configs[config.level] = config
        self._data[config.level] = []

        # Create aggregators for each aggregation function
        self._aggregators[config.level] = {}
        for agg_func in config.aggregations:
            self._aggregators[config.level][agg_func] = TimeBucketAggregation(
                granularity=config.granularity,
                aggregation=agg_func,
            )

        return self

    def add_points(
        self,
        points: list[TimeSeriesPoint],
        level: RollupLevel = RollupLevel.REALTIME,
    ) -> list[RollupResult]:
        """Add points and compute rollup for a level.

        Args:
            points: Time series points.
            level: Target rollup level.

        Returns:
            List of rollup results.
        """
        if level not in self._configs:
            raise ValueError(f"Unknown rollup level: {level}")

        config = self._configs[level]
        aggregators = self._aggregators[level]

        # Compute each aggregation
        agg_results: dict[datetime, dict[AggregationFunction, BucketResult]] = {}

        for agg_func, aggregator in aggregators.items():
            buckets = aggregator.aggregate(points)
            for bucket in buckets:
                if bucket.bucket_start not in agg_results:
                    agg_results[bucket.bucket_start] = {}
                agg_results[bucket.bucket_start][agg_func] = bucket

        # Combine into RollupResults
        results = []
        for bucket_start, agg_buckets in sorted(agg_results.items()):
            # Get bucket end from any aggregation result
            any_bucket = next(iter(agg_buckets.values()))

            values = {
                agg_func: bucket.value
                for agg_func, bucket in agg_buckets.items()
            }

            result = RollupResult(
                level=level,
                bucket_start=bucket_start,
                bucket_end=any_bucket.bucket_end,
                values=values,
                sample_count=any_bucket.sample_count,
                labels=any_bucket.labels,
            )
            results.append(result)

        # Store results
        self._data[level].extend(results)

        # Auto rollup to next level if enabled
        if self._auto_rollup:
            self._cascade_rollup(level)

        return results

    def _cascade_rollup(self, from_level: RollupLevel) -> None:
        """Cascade rollup from one level to the next coarser level."""
        # Determine the next level in the hierarchy
        level_order = [
            RollupLevel.REALTIME,
            RollupLevel.SHORT_TERM,
            RollupLevel.MEDIUM_TERM,
            RollupLevel.LONG_TERM,
            RollupLevel.ARCHIVE,
        ]

        current_index = level_order.index(from_level)
        if current_index >= len(level_order) - 1:
            return

        next_level = level_order[current_index + 1]
        if next_level not in self._configs:
            return

        # Check if we have enough data to roll up
        from_config = self._configs[from_level]
        to_config = self._configs[next_level]

        from_data = self._data[from_level]
        if not from_data:
            return

        # Convert RollupResults back to points for the avg aggregation
        # and roll up to next level
        points = []
        for result in from_data:
            avg_value = result.get_value(AggregationFunction.AVG)
            if avg_value is not None:
                points.append(TimeSeriesPoint(
                    timestamp=result.bucket_start,
                    value=avg_value,
                    labels=result.labels,
                ))

        if points:
            # Roll up to next level (non-recursive call to avoid infinite loop)
            self._add_points_no_cascade(points, next_level)

    def _add_points_no_cascade(
        self,
        points: list[TimeSeriesPoint],
        level: RollupLevel,
    ) -> list[RollupResult]:
        """Add points without cascading (internal use)."""
        if level not in self._configs:
            return []

        aggregators = self._aggregators[level]
        agg_results: dict[datetime, dict[AggregationFunction, BucketResult]] = {}

        for agg_func, aggregator in aggregators.items():
            buckets = aggregator.aggregate(points)
            for bucket in buckets:
                if bucket.bucket_start not in agg_results:
                    agg_results[bucket.bucket_start] = {}
                agg_results[bucket.bucket_start][agg_func] = bucket

        results = []
        for bucket_start, agg_buckets in sorted(agg_results.items()):
            any_bucket = next(iter(agg_buckets.values()))
            values = {
                agg_func: bucket.value
                for agg_func, bucket in agg_buckets.items()
            }
            result = RollupResult(
                level=level,
                bucket_start=bucket_start,
                bucket_end=any_bucket.bucket_end,
                values=values,
                sample_count=any_bucket.sample_count,
                labels=any_bucket.labels,
            )
            results.append(result)

        self._data[level].extend(results)
        return results

    def query(
        self,
        start: datetime,
        end: datetime,
        preferred_granularity: TimeGranularity | None = None,
        aggregation: AggregationFunction = AggregationFunction.AVG,
    ) -> list[TimeSeriesPoint]:
        """Query rollup data.

        Automatically selects the appropriate rollup level based on
        the time range and preferred granularity.

        Args:
            start: Start time.
            end: End time.
            preferred_granularity: Preferred granularity for results.
            aggregation: Aggregation function to retrieve.

        Returns:
            List of time series points.
        """
        # Select appropriate level
        level = self._select_level(start, end, preferred_granularity)

        if level not in self._data:
            return []

        # Filter data within time range
        results = []
        for rollup in self._data[level]:
            if start <= rollup.bucket_start < end:
                point = rollup.to_point(aggregation)
                if point:
                    results.append(point)

        return sorted(results, key=lambda p: p.timestamp)

    def _select_level(
        self,
        start: datetime,
        end: datetime,
        preferred_granularity: TimeGranularity | None,
    ) -> RollupLevel:
        """Select the best rollup level for a query."""
        duration = end - start

        # If preferred granularity is specified, find matching level
        if preferred_granularity:
            for level, config in self._configs.items():
                if config.granularity == preferred_granularity:
                    return level

        # Otherwise, select based on duration
        if duration <= timedelta(hours=1):
            return RollupLevel.REALTIME
        elif duration <= timedelta(days=1):
            return RollupLevel.SHORT_TERM
        elif duration <= timedelta(days=7):
            return RollupLevel.MEDIUM_TERM
        else:
            return RollupLevel.LONG_TERM

    def cleanup_expired(self, now: datetime | None = None) -> dict[RollupLevel, int]:
        """Clean up expired data based on retention policies.

        Args:
            now: Current time (uses datetime.now if not provided).

        Returns:
            Dictionary of level -> number of removed entries.
        """
        now = now or datetime.now()
        removed_counts = {}

        for level, config in self._configs.items():
            cutoff = now - config.retention
            original_count = len(self._data[level])

            self._data[level] = [
                r for r in self._data[level]
                if r.bucket_start >= cutoff
            ]

            removed_counts[level] = original_count - len(self._data[level])

        return removed_counts

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the rollup system.

        Returns:
            Dictionary of statistics.
        """
        stats = {
            "levels": {},
            "total_entries": 0,
        }

        for level, config in self._configs.items():
            data = self._data.get(level, [])
            level_stats = {
                "granularity": config.granularity.value,
                "retention": str(config.retention),
                "entry_count": len(data),
                "aggregations": [a.value for a in config.aggregations],
            }

            if data:
                level_stats["oldest"] = min(r.bucket_start for r in data).isoformat()
                level_stats["newest"] = max(r.bucket_start for r in data).isoformat()

            stats["levels"][level.value] = level_stats
            stats["total_entries"] += len(data)

        return stats

    def clear(self, level: RollupLevel | None = None) -> None:
        """Clear rollup data.

        Args:
            level: Specific level to clear, or None for all.
        """
        if level:
            if level in self._data:
                self._data[level] = []
        else:
            for lvl in self._data:
                self._data[lvl] = []
