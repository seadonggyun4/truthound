"""Time bucket aggregation for time series data.

Provides SQL-like time_bucket functionality for grouping
time series points into fixed-size intervals.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable

from truthound.checkpoint.analytics.protocols import (
    TimeSeriesPoint,
    TimeGranularity,
    AggregationFunction,
)


@dataclass
class BucketResult:
    """Result of a time bucket aggregation.

    Attributes:
        bucket_start: Start of the bucket interval.
        bucket_end: End of the bucket interval.
        value: Aggregated value for the bucket.
        sample_count: Number of samples in the bucket.
        labels: Common labels across samples.
        metadata: Additional metadata.
    """

    bucket_start: datetime
    bucket_end: datetime
    value: float
    sample_count: int = 0
    labels: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_point(self) -> TimeSeriesPoint:
        """Convert to TimeSeriesPoint using bucket start as timestamp."""
        return TimeSeriesPoint(
            timestamp=self.bucket_start,
            value=self.value,
            labels=self.labels,
            metadata={
                **self.metadata,
                "bucket_end": self.bucket_end.isoformat(),
                "sample_count": self.sample_count,
            },
        )


class TimeBucketAggregation:
    """Time bucket aggregation for time series data.

    Groups time series points into fixed-size buckets and applies
    an aggregation function to each bucket.

    Supports various granularities from seconds to months.

    Example:
        >>> agg = TimeBucketAggregation(
        ...     granularity=TimeGranularity.HOUR,
        ...     aggregation=AggregationFunction.AVG,
        ... )
        >>> buckets = agg.aggregate(points)
        >>> for bucket in buckets:
        ...     print(f"{bucket.bucket_start}: {bucket.value}")
    """

    # Granularity to timedelta mapping
    GRANULARITY_DELTAS = {
        TimeGranularity.SECOND: timedelta(seconds=1),
        TimeGranularity.MINUTE: timedelta(minutes=1),
        TimeGranularity.HOUR: timedelta(hours=1),
        TimeGranularity.DAY: timedelta(days=1),
        TimeGranularity.WEEK: timedelta(weeks=1),
        TimeGranularity.MONTH: timedelta(days=30),  # Approximate
    }

    def __init__(
        self,
        granularity: TimeGranularity = TimeGranularity.HOUR,
        aggregation: AggregationFunction = AggregationFunction.AVG,
        origin: datetime | None = None,
        fill_gaps: bool = False,
        gap_fill_value: float = 0.0,
    ) -> None:
        """Initialize time bucket aggregation.

        Args:
            granularity: Size of each bucket.
            aggregation: Aggregation function to apply.
            origin: Origin time for bucket alignment.
            fill_gaps: Whether to fill missing buckets.
            gap_fill_value: Value to use for filled gaps.
        """
        self._granularity = granularity
        self._aggregation = aggregation
        self._origin = origin or datetime(1970, 1, 1)
        self._fill_gaps = fill_gaps
        self._gap_fill_value = gap_fill_value
        self._delta = self.GRANULARITY_DELTAS[granularity]

    def aggregate(
        self,
        points: list[TimeSeriesPoint],
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[BucketResult]:
        """Aggregate points into time buckets.

        Args:
            points: Time series points to aggregate.
            start: Optional start time (uses min point time if not provided).
            end: Optional end time (uses max point time if not provided).

        Returns:
            List of bucket results.
        """
        if not points:
            return []

        # Sort points by timestamp
        sorted_points = sorted(points, key=lambda p: p.timestamp)

        # Determine time range
        range_start = start or sorted_points[0].timestamp
        range_end = end or sorted_points[-1].timestamp

        # Group points into buckets
        buckets: dict[datetime, list[TimeSeriesPoint]] = {}

        for point in sorted_points:
            bucket_start = self._get_bucket_start(point.timestamp)
            if bucket_start not in buckets:
                buckets[bucket_start] = []
            buckets[bucket_start].append(point)

        # Apply aggregation function to each bucket
        results = []
        for bucket_start, bucket_points in sorted(buckets.items()):
            bucket_end = bucket_start + self._delta
            value = self._apply_aggregation(bucket_points)

            # Merge labels (keep only common labels)
            common_labels = self._get_common_labels(bucket_points)

            results.append(BucketResult(
                bucket_start=bucket_start,
                bucket_end=bucket_end,
                value=value,
                sample_count=len(bucket_points),
                labels=common_labels,
            ))

        # Fill gaps if requested
        if self._fill_gaps:
            results = self._fill_bucket_gaps(results, range_start, range_end)

        return results

    def _get_bucket_start(self, timestamp: datetime) -> datetime:
        """Get the start of the bucket containing the timestamp."""
        # Calculate seconds from origin
        delta = timestamp - self._origin
        total_seconds = delta.total_seconds()
        bucket_seconds = self._delta.total_seconds()

        # Floor to bucket boundary
        bucket_index = math.floor(total_seconds / bucket_seconds)
        bucket_start = self._origin + timedelta(seconds=bucket_index * bucket_seconds)

        return bucket_start

    def _apply_aggregation(self, points: list[TimeSeriesPoint]) -> float:
        """Apply aggregation function to bucket points."""
        if not points:
            return 0.0

        values = [p.value for p in points]

        if self._aggregation == AggregationFunction.SUM:
            return sum(values)
        elif self._aggregation == AggregationFunction.AVG:
            return sum(values) / len(values)
        elif self._aggregation == AggregationFunction.MIN:
            return min(values)
        elif self._aggregation == AggregationFunction.MAX:
            return max(values)
        elif self._aggregation == AggregationFunction.COUNT:
            return float(len(values))
        elif self._aggregation == AggregationFunction.FIRST:
            return values[0]
        elif self._aggregation == AggregationFunction.LAST:
            return values[-1]
        elif self._aggregation == AggregationFunction.STDDEV:
            if len(values) < 2:
                return 0.0
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            return math.sqrt(variance)
        elif self._aggregation == AggregationFunction.PERCENTILE_50:
            return self._percentile(values, 50)
        elif self._aggregation == AggregationFunction.PERCENTILE_95:
            return self._percentile(values, 95)
        elif self._aggregation == AggregationFunction.PERCENTILE_99:
            return self._percentile(values, 99)
        else:
            return sum(values) / len(values)

    def _percentile(self, values: list[float], percentile: float) -> float:
        """Compute percentile of values."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = (percentile / 100) * (len(sorted_values) - 1)
        lower = int(index)
        upper = min(lower + 1, len(sorted_values) - 1)
        weight = index - lower
        return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight

    def _get_common_labels(
        self,
        points: list[TimeSeriesPoint],
    ) -> dict[str, str]:
        """Get labels that are common across all points."""
        if not points:
            return {}

        common_labels = dict(points[0].labels)

        for point in points[1:]:
            keys_to_remove = []
            for key, value in common_labels.items():
                if key not in point.labels or point.labels[key] != value:
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                del common_labels[key]

        return common_labels

    def _fill_bucket_gaps(
        self,
        results: list[BucketResult],
        start: datetime,
        end: datetime,
    ) -> list[BucketResult]:
        """Fill missing buckets with gap fill value."""
        if not results:
            return results

        # Create a set of existing bucket starts
        existing_buckets = {r.bucket_start for r in results}

        # Generate all bucket starts in range
        filled_results = list(results)
        current = self._get_bucket_start(start)

        while current <= end:
            if current not in existing_buckets:
                filled_results.append(BucketResult(
                    bucket_start=current,
                    bucket_end=current + self._delta,
                    value=self._gap_fill_value,
                    sample_count=0,
                ))
            current += self._delta

        # Sort by bucket start
        return sorted(filled_results, key=lambda r: r.bucket_start)

    def create_downsampler(
        self,
        target_granularity: TimeGranularity,
    ) -> "TimeBucketAggregation":
        """Create a new aggregation for downsampling to a coarser granularity.

        Args:
            target_granularity: Target granularity (must be coarser).

        Returns:
            New TimeBucketAggregation for downsampling.

        Raises:
            ValueError: If target is not coarser than current.
        """
        current_seconds = self._delta.total_seconds()
        target_delta = self.GRANULARITY_DELTAS[target_granularity]
        target_seconds = target_delta.total_seconds()

        if target_seconds <= current_seconds:
            raise ValueError(
                f"Target granularity {target_granularity} must be coarser than "
                f"current granularity {self._granularity}"
            )

        return TimeBucketAggregation(
            granularity=target_granularity,
            aggregation=self._aggregation,
            origin=self._origin,
            fill_gaps=self._fill_gaps,
            gap_fill_value=self._gap_fill_value,
        )
