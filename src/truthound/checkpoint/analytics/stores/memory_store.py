"""In-memory time series store.

Stores time series data in memory, suitable for development,
testing, and small-scale deployments.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

from truthound.checkpoint.analytics.protocols import (
    TimeGranularity,
    TimeSeriesPoint,
    StoreError,
)
from truthound.checkpoint.analytics.stores.base import BaseTimeSeriesStore

logger = logging.getLogger(__name__)


class InMemoryTimeSeriesStore(BaseTimeSeriesStore):
    """In-memory time series store.

    Stores all data in memory using dictionaries. Thread-safe.
    Supports automatic retention cleanup.

    Example:
        >>> store = InMemoryTimeSeriesStore(retention_days=30)
        >>> await store.connect()
        >>>
        >>> # Write data
        >>> await store.write("cpu_usage", TimeSeriesPoint(
        ...     timestamp=datetime.now(),
        ...     value=75.5,
        ...     labels={"host": "server1"},
        ... ))
        >>>
        >>> # Query data
        >>> points = await store.query(
        ...     "cpu_usage",
        ...     start=datetime.now() - timedelta(hours=1),
        ...     end=datetime.now(),
        ... )
    """

    def __init__(
        self,
        name: str = "in_memory",
        retention_days: int = 90,
        max_points_per_metric: int = 1_000_000,
    ) -> None:
        """Initialize in-memory store.

        Args:
            name: Store name.
            retention_days: Data retention period.
            max_points_per_metric: Maximum points per metric (memory limit).
        """
        super().__init__(name=name, retention_days=retention_days)
        self._max_points = max_points_per_metric
        self._lock = threading.RLock()

        # Storage: metric_name -> list of points (sorted by timestamp)
        self._data: dict[str, list[TimeSeriesPoint]] = defaultdict(list)

        # Pre-aggregated rollups: metric_name -> granularity -> bucket -> value
        self._rollups: dict[str, dict[TimeGranularity, dict[datetime, list[float]]]] = (
            defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        )

    async def write(self, metric_name: str, point: TimeSeriesPoint) -> None:
        """Write a single point."""
        with self._lock:
            self._data[metric_name].append(point)
            self._data[metric_name].sort(key=lambda p: p.timestamp)

            # Update rollups
            self._update_rollups(metric_name, point)

            # Enforce limits
            self._enforce_limits(metric_name)

    async def write_batch(self, metric_name: str, points: list[TimeSeriesPoint]) -> None:
        """Write multiple points."""
        if not points:
            return

        with self._lock:
            self._data[metric_name].extend(points)
            self._data[metric_name].sort(key=lambda p: p.timestamp)

            # Update rollups
            for point in points:
                self._update_rollups(metric_name, point)

            # Enforce limits
            self._enforce_limits(metric_name)

    def _update_rollups(self, metric_name: str, point: TimeSeriesPoint) -> None:
        """Update pre-aggregated rollups with a new point."""
        for granularity in [TimeGranularity.MINUTE, TimeGranularity.HOUR, TimeGranularity.DAY]:
            bucket = self._truncate_to_bucket(point.timestamp, granularity)
            self._rollups[metric_name][granularity][bucket].append(point.value)

    def _enforce_limits(self, metric_name: str) -> None:
        """Enforce storage limits by removing old data."""
        # Remove old data based on retention
        cutoff = datetime.now() - self.retention_period
        self._data[metric_name] = [
            p for p in self._data[metric_name]
            if p.timestamp >= cutoff
        ]

        # Enforce max points limit
        if len(self._data[metric_name]) > self._max_points:
            # Keep the most recent points
            self._data[metric_name] = self._data[metric_name][-self._max_points:]

        # Clean up old rollups
        for granularity in self._rollups[metric_name]:
            rollup = self._rollups[metric_name][granularity]
            old_buckets = [b for b in rollup if b < cutoff]
            for bucket in old_buckets:
                del rollup[bucket]

    async def query(
        self,
        metric_name: str,
        start: datetime,
        end: datetime,
        granularity: TimeGranularity | None = None,
        labels: dict[str, str] | None = None,
    ) -> list[TimeSeriesPoint]:
        """Query points in a time range."""
        with self._lock:
            points = self._data.get(metric_name, [])

            # Filter by time range
            filtered = [
                p for p in points
                if start <= p.timestamp <= end
            ]

            # Filter by labels
            if labels:
                filtered = [
                    p for p in filtered
                    if self._matches_labels(p.labels, labels)
                ]

            # Apply granularity (downsample)
            if granularity:
                filtered = self._downsample(filtered, granularity)

            return filtered

    def _downsample(
        self,
        points: list[TimeSeriesPoint],
        granularity: TimeGranularity,
    ) -> list[TimeSeriesPoint]:
        """Downsample points to the specified granularity."""
        if not points:
            return []

        # Group by bucket
        buckets: dict[datetime, list[TimeSeriesPoint]] = defaultdict(list)
        for point in points:
            bucket = self._truncate_to_bucket(point.timestamp, granularity)
            buckets[bucket].append(point)

        # Average each bucket
        result = []
        for bucket, bucket_points in sorted(buckets.items()):
            avg_value = sum(p.value for p in bucket_points) / len(bucket_points)

            # Merge labels (keep common labels)
            common_labels = bucket_points[0].labels.copy()
            for p in bucket_points[1:]:
                common_labels = {
                    k: v for k, v in common_labels.items()
                    if p.labels.get(k) == v
                }

            result.append(TimeSeriesPoint(
                timestamp=bucket,
                value=avg_value,
                labels=common_labels,
                metadata={"sample_count": len(bucket_points)},
            ))

        return result

    async def aggregate(
        self,
        metric_name: str,
        start: datetime,
        end: datetime,
        aggregation: str,
        granularity: TimeGranularity,
        labels: dict[str, str] | None = None,
    ) -> list[TimeSeriesPoint]:
        """Aggregate points over time buckets."""
        # Try to use pre-computed rollups for common aggregations
        if (
            aggregation in ("avg", "sum", "count")
            and granularity in self._rollups.get(metric_name, {})
            and not labels
        ):
            return self._query_rollup(metric_name, start, end, aggregation, granularity)

        # Fall back to computing from raw data
        points = await self.query(metric_name, start, end, labels=labels)

        if not points:
            return []

        # Group by bucket
        buckets: dict[datetime, list[float]] = defaultdict(list)
        for point in points:
            bucket = self._truncate_to_bucket(point.timestamp, granularity)
            buckets[bucket].append(point.value)

        # Apply aggregation
        result = []
        for bucket in self._generate_buckets(start, end, granularity):
            values = buckets.get(bucket, [])
            if values:
                agg_value = self._aggregate_values(values, aggregation)
                result.append(TimeSeriesPoint(
                    timestamp=bucket,
                    value=agg_value,
                    metadata={
                        "aggregation": aggregation,
                        "sample_count": len(values),
                    },
                ))

        return result

    def _query_rollup(
        self,
        metric_name: str,
        start: datetime,
        end: datetime,
        aggregation: str,
        granularity: TimeGranularity,
    ) -> list[TimeSeriesPoint]:
        """Query from pre-computed rollup."""
        with self._lock:
            rollup = self._rollups.get(metric_name, {}).get(granularity, {})

            result = []
            for bucket in self._generate_buckets(start, end, granularity):
                values = rollup.get(bucket, [])
                if values:
                    agg_value = self._aggregate_values(values, aggregation)
                    result.append(TimeSeriesPoint(
                        timestamp=bucket,
                        value=agg_value,
                        metadata={
                            "aggregation": aggregation,
                            "sample_count": len(values),
                            "source": "rollup",
                        },
                    ))

            return result

    async def delete(
        self,
        metric_name: str,
        before: datetime | None = None,
        labels: dict[str, str] | None = None,
    ) -> int:
        """Delete points."""
        with self._lock:
            if metric_name not in self._data:
                return 0

            original_count = len(self._data[metric_name])

            if before and labels:
                self._data[metric_name] = [
                    p for p in self._data[metric_name]
                    if not (p.timestamp < before and self._matches_labels(p.labels, labels))
                ]
            elif before:
                self._data[metric_name] = [
                    p for p in self._data[metric_name]
                    if p.timestamp >= before
                ]
            elif labels:
                self._data[metric_name] = [
                    p for p in self._data[metric_name]
                    if not self._matches_labels(p.labels, labels)
                ]
            else:
                # Delete all
                self._data[metric_name] = []

            deleted = original_count - len(self._data[metric_name])

            # Clean up rollups if deleting all or by time
            if not labels:
                if before:
                    for granularity in self._rollups.get(metric_name, {}):
                        rollup = self._rollups[metric_name][granularity]
                        old_buckets = [b for b in rollup if b < before]
                        for bucket in old_buckets:
                            del rollup[bucket]
                else:
                    self._rollups.pop(metric_name, None)

            return deleted

    async def get_metrics(self) -> list[str]:
        """Get list of all metric names."""
        with self._lock:
            return list(self._data.keys())

    async def get_metric_stats(self, metric_name: str) -> dict[str, Any]:
        """Get statistics for a metric."""
        with self._lock:
            points = self._data.get(metric_name, [])

            if not points:
                return {
                    "metric_name": metric_name,
                    "point_count": 0,
                }

            values = [p.value for p in points]

            return {
                "metric_name": metric_name,
                "point_count": len(points),
                "first_timestamp": points[0].timestamp.isoformat(),
                "last_timestamp": points[-1].timestamp.isoformat(),
                "min_value": min(values),
                "max_value": max(values),
                "avg_value": sum(values) / len(values),
            }

    def clear(self) -> None:
        """Clear all stored data."""
        with self._lock:
            self._data.clear()
            self._rollups.clear()
