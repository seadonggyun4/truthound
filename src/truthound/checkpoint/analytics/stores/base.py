"""Base time series store implementation.

Provides abstract base class with common functionality for all stores.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any

from truthound.checkpoint.analytics.protocols import (
    TimeSeriesStoreProtocol,
    TimeGranularity,
    TimeSeriesPoint,
    StoreError,
)

logger = logging.getLogger(__name__)


class BaseTimeSeriesStore(ABC):
    """Abstract base class for time series stores.

    Provides common functionality including:
    - Time bucket calculations
    - Aggregation helpers
    - Label filtering

    Subclasses must implement the storage-specific methods.
    """

    def __init__(
        self,
        name: str | None = None,
        retention_days: int = 90,
    ) -> None:
        """Initialize store.

        Args:
            name: Store name.
            retention_days: Data retention period in days.
        """
        self._name = name or self.__class__.__name__
        self._retention_days = retention_days
        self._connected = False

    @property
    def name(self) -> str:
        """Get store name."""
        return self._name

    @property
    def is_connected(self) -> bool:
        """Check if store is connected."""
        return self._connected

    @property
    def retention_period(self) -> timedelta:
        """Get retention period."""
        return timedelta(days=self._retention_days)

    async def connect(self) -> None:
        """Connect to the storage backend.

        Override in subclasses that require connection setup.
        """
        self._connected = True
        logger.info(f"{self._name} connected")

    async def disconnect(self) -> None:
        """Disconnect from the storage backend.

        Override in subclasses that require cleanup.
        """
        self._connected = False
        logger.info(f"{self._name} disconnected")

    @abstractmethod
    async def write(self, metric_name: str, point: TimeSeriesPoint) -> None:
        """Write a single point."""
        pass

    @abstractmethod
    async def write_batch(self, metric_name: str, points: list[TimeSeriesPoint]) -> None:
        """Write multiple points."""
        pass

    @abstractmethod
    async def query(
        self,
        metric_name: str,
        start: datetime,
        end: datetime,
        granularity: TimeGranularity | None = None,
        labels: dict[str, str] | None = None,
    ) -> list[TimeSeriesPoint]:
        """Query points in a time range."""
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    async def delete(
        self,
        metric_name: str,
        before: datetime | None = None,
        labels: dict[str, str] | None = None,
    ) -> int:
        """Delete points."""
        pass

    # Helper methods

    def _truncate_to_bucket(
        self,
        timestamp: datetime,
        granularity: TimeGranularity,
    ) -> datetime:
        """Truncate timestamp to the start of a time bucket.

        Args:
            timestamp: Timestamp to truncate.
            granularity: Bucket granularity.

        Returns:
            Truncated timestamp.
        """
        if granularity == TimeGranularity.SECOND:
            return timestamp.replace(microsecond=0)

        if granularity == TimeGranularity.MINUTE:
            return timestamp.replace(second=0, microsecond=0)

        if granularity == TimeGranularity.HOUR:
            return timestamp.replace(minute=0, second=0, microsecond=0)

        if granularity == TimeGranularity.DAY:
            return timestamp.replace(hour=0, minute=0, second=0, microsecond=0)

        if granularity == TimeGranularity.WEEK:
            # Start of week (Monday)
            days_since_monday = timestamp.weekday()
            start = timestamp - timedelta(days=days_since_monday)
            return start.replace(hour=0, minute=0, second=0, microsecond=0)

        if granularity == TimeGranularity.MONTH:
            return timestamp.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        if granularity == TimeGranularity.QUARTER:
            quarter_month = ((timestamp.month - 1) // 3) * 3 + 1
            return timestamp.replace(
                month=quarter_month, day=1, hour=0, minute=0, second=0, microsecond=0
            )

        if granularity == TimeGranularity.YEAR:
            return timestamp.replace(
                month=1, day=1, hour=0, minute=0, second=0, microsecond=0
            )

        return timestamp

    def _generate_buckets(
        self,
        start: datetime,
        end: datetime,
        granularity: TimeGranularity,
    ) -> list[datetime]:
        """Generate time bucket boundaries.

        Args:
            start: Start time.
            end: End time.
            granularity: Bucket granularity.

        Returns:
            List of bucket start times.
        """
        buckets = []
        current = self._truncate_to_bucket(start, granularity)

        while current <= end:
            buckets.append(current)
            current = self._advance_bucket(current, granularity)

        return buckets

    def _advance_bucket(
        self,
        timestamp: datetime,
        granularity: TimeGranularity,
    ) -> datetime:
        """Advance to the next bucket.

        Args:
            timestamp: Current bucket start.
            granularity: Bucket granularity.

        Returns:
            Next bucket start time.
        """
        delta = granularity.to_timedelta()
        return timestamp + delta

    def _matches_labels(
        self,
        point_labels: dict[str, str],
        filter_labels: dict[str, str],
    ) -> bool:
        """Check if point labels match filter.

        Args:
            point_labels: Labels on the point.
            filter_labels: Labels to filter by.

        Returns:
            True if all filter labels match.
        """
        for key, value in filter_labels.items():
            if point_labels.get(key) != value:
                return False
        return True

    def _aggregate_values(
        self,
        values: list[float],
        aggregation: str,
    ) -> float:
        """Apply aggregation function to values.

        Args:
            values: Values to aggregate.
            aggregation: Aggregation function name.

        Returns:
            Aggregated value.
        """
        if not values:
            return 0.0

        agg = aggregation.lower()

        if agg == "sum":
            return sum(values)
        elif agg == "avg" or agg == "mean":
            return sum(values) / len(values)
        elif agg == "min":
            return min(values)
        elif agg == "max":
            return max(values)
        elif agg == "count":
            return float(len(values))
        elif agg == "first":
            return values[0]
        elif agg == "last":
            return values[-1]
        elif agg.startswith("p"):
            # Percentile (e.g., "p50", "p95", "p99")
            try:
                percentile = int(agg[1:])
                return self._percentile(values, percentile)
            except ValueError:
                pass

        # Default to average
        return sum(values) / len(values)

    def _percentile(self, values: list[float], percentile: int) -> float:
        """Calculate percentile of values.

        Args:
            values: Values to analyze.
            percentile: Percentile (0-100).

        Returns:
            Percentile value.
        """
        if not values:
            return 0.0

        sorted_values = sorted(values)
        n = len(sorted_values)
        k = (n - 1) * (percentile / 100)
        f = math.floor(k)
        c = math.ceil(k)

        if f == c:
            return sorted_values[int(k)]

        return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self._name!r}, connected={self._connected})"
