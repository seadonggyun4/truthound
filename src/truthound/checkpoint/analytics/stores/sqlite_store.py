"""SQLite-based time series store.

Provides persistent storage using SQLite with efficient indexing
for time series queries.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from truthound.checkpoint.analytics.protocols import (
    TimeGranularity,
    TimeSeriesPoint,
    StoreError,
)
from truthound.checkpoint.analytics.stores.base import BaseTimeSeriesStore

logger = logging.getLogger(__name__)


class SQLiteTimeSeriesStore(BaseTimeSeriesStore):
    """SQLite-based time series store.

    Stores time series data in SQLite with efficient indexing.
    Suitable for single-machine deployments with moderate data volumes.

    Example:
        >>> store = SQLiteTimeSeriesStore(db_path="./analytics.db")
        >>> await store.connect()
        >>>
        >>> # Write data
        >>> await store.write("checkpoint.duration", TimeSeriesPoint(
        ...     timestamp=datetime.now(),
        ...     value=1234.5,
        ...     labels={"checkpoint": "daily_check"},
        ... ))
        >>>
        >>> # Query with aggregation
        >>> points = await store.aggregate(
        ...     "checkpoint.duration",
        ...     start=datetime.now() - timedelta(days=7),
        ...     end=datetime.now(),
        ...     aggregation="avg",
        ...     granularity=TimeGranularity.DAY,
        ... )
    """

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        name: str = "sqlite",
        retention_days: int = 90,
        wal_mode: bool = True,
    ) -> None:
        """Initialize SQLite store.

        Args:
            db_path: Path to SQLite database file or ":memory:".
            name: Store name.
            retention_days: Data retention period.
            wal_mode: Enable WAL mode for better concurrency.
        """
        super().__init__(name=name, retention_days=retention_days)
        self._db_path = str(db_path)
        self._wal_mode = wal_mode
        self._conn: sqlite3.Connection | None = None

    async def connect(self) -> None:
        """Connect to SQLite database."""
        try:
            self._conn = sqlite3.connect(
                self._db_path,
                check_same_thread=False,
                isolation_level=None,  # Auto-commit
            )
            self._conn.row_factory = sqlite3.Row

            # Enable optimizations
            if self._wal_mode and self._db_path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")

            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA cache_size=10000")
            self._conn.execute("PRAGMA temp_store=MEMORY")

            # Create tables
            self._create_schema()

            await super().connect()
            logger.info(f"Connected to SQLite database: {self._db_path}")

        except Exception as e:
            raise StoreError(
                f"Failed to connect to SQLite: {e}",
                self.name,
                "connect",
            )

    async def disconnect(self) -> None:
        """Disconnect from SQLite database."""
        if self._conn:
            self._conn.close()
            self._conn = None

        await super().disconnect()

    def _create_schema(self) -> None:
        """Create database schema."""
        if not self._conn:
            return

        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS timeseries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                timestamp_unix REAL NOT NULL,
                value REAL NOT NULL,
                labels TEXT,
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_timeseries_metric_time
            ON timeseries(metric_name, timestamp_unix);

            CREATE INDEX IF NOT EXISTS idx_timeseries_timestamp
            ON timeseries(timestamp_unix);

            CREATE TABLE IF NOT EXISTS rollups (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT NOT NULL,
                granularity TEXT NOT NULL,
                bucket_start TEXT NOT NULL,
                bucket_start_unix REAL NOT NULL,
                sum_value REAL NOT NULL,
                count_value INTEGER NOT NULL,
                min_value REAL NOT NULL,
                max_value REAL NOT NULL,
                labels TEXT,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(metric_name, granularity, bucket_start, labels)
            );

            CREATE INDEX IF NOT EXISTS idx_rollups_metric_granularity
            ON rollups(metric_name, granularity, bucket_start_unix);
        """)

    async def write(self, metric_name: str, point: TimeSeriesPoint) -> None:
        """Write a single point."""
        if not self._conn:
            raise StoreError("Not connected", self.name, "write")

        try:
            labels_json = json.dumps(point.labels) if point.labels else None
            metadata_json = json.dumps(point.metadata) if point.metadata else None

            self._conn.execute("""
                INSERT INTO timeseries
                (metric_name, timestamp, timestamp_unix, value, labels, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                metric_name,
                point.timestamp.isoformat(),
                point.timestamp.timestamp(),
                point.value,
                labels_json,
                metadata_json,
            ))

            # Update rollups
            self._update_rollup(metric_name, point)

        except Exception as e:
            raise StoreError(
                f"Failed to write point: {e}",
                self.name,
                "write",
            )

    async def write_batch(self, metric_name: str, points: list[TimeSeriesPoint]) -> None:
        """Write multiple points."""
        if not points or not self._conn:
            return

        try:
            data = [
                (
                    metric_name,
                    p.timestamp.isoformat(),
                    p.timestamp.timestamp(),
                    p.value,
                    json.dumps(p.labels) if p.labels else None,
                    json.dumps(p.metadata) if p.metadata else None,
                )
                for p in points
            ]

            self._conn.executemany("""
                INSERT INTO timeseries
                (metric_name, timestamp, timestamp_unix, value, labels, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, data)

            # Update rollups
            for point in points:
                self._update_rollup(metric_name, point)

        except Exception as e:
            raise StoreError(
                f"Failed to write batch: {e}",
                self.name,
                "write_batch",
            )

    def _update_rollup(self, metric_name: str, point: TimeSeriesPoint) -> None:
        """Update rollup table with new point."""
        if not self._conn:
            return

        labels_json = json.dumps(point.labels) if point.labels else ""

        for granularity in [TimeGranularity.HOUR, TimeGranularity.DAY]:
            bucket = self._truncate_to_bucket(point.timestamp, granularity)

            self._conn.execute("""
                INSERT INTO rollups
                (metric_name, granularity, bucket_start, bucket_start_unix,
                 sum_value, count_value, min_value, max_value, labels)
                VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?)
                ON CONFLICT(metric_name, granularity, bucket_start, labels)
                DO UPDATE SET
                    sum_value = sum_value + excluded.sum_value,
                    count_value = count_value + 1,
                    min_value = MIN(min_value, excluded.min_value),
                    max_value = MAX(max_value, excluded.max_value),
                    updated_at = CURRENT_TIMESTAMP
            """, (
                metric_name,
                granularity.value,
                bucket.isoformat(),
                bucket.timestamp(),
                point.value,
                point.value,
                point.value,
                labels_json,
            ))

    async def query(
        self,
        metric_name: str,
        start: datetime,
        end: datetime,
        granularity: TimeGranularity | None = None,
        labels: dict[str, str] | None = None,
    ) -> list[TimeSeriesPoint]:
        """Query points in a time range."""
        if not self._conn:
            raise StoreError("Not connected", self.name, "query")

        try:
            query = """
                SELECT timestamp, value, labels, metadata
                FROM timeseries
                WHERE metric_name = ?
                AND timestamp_unix >= ?
                AND timestamp_unix <= ?
            """
            params: list[Any] = [
                metric_name,
                start.timestamp(),
                end.timestamp(),
            ]

            if labels:
                for key, value in labels.items():
                    query += " AND json_extract(labels, ?) = ?"
                    params.extend([f"$.{key}", value])

            query += " ORDER BY timestamp_unix"

            cursor = self._conn.execute(query, params)
            rows = cursor.fetchall()

            points = []
            for row in rows:
                point = TimeSeriesPoint(
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    value=row["value"],
                    labels=json.loads(row["labels"]) if row["labels"] else {},
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                )
                points.append(point)

            # Apply granularity if specified
            if granularity:
                points = self._downsample(points, granularity)

            return points

        except Exception as e:
            raise StoreError(
                f"Failed to query: {e}",
                self.name,
                "query",
            )

    def _downsample(
        self,
        points: list[TimeSeriesPoint],
        granularity: TimeGranularity,
    ) -> list[TimeSeriesPoint]:
        """Downsample points to the specified granularity."""
        if not points:
            return []

        from collections import defaultdict

        # Group by bucket
        buckets: dict[datetime, list[TimeSeriesPoint]] = defaultdict(list)
        for point in points:
            bucket = self._truncate_to_bucket(point.timestamp, granularity)
            buckets[bucket].append(point)

        # Average each bucket
        result = []
        for bucket, bucket_points in sorted(buckets.items()):
            avg_value = sum(p.value for p in bucket_points) / len(bucket_points)

            result.append(TimeSeriesPoint(
                timestamp=bucket,
                value=avg_value,
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
        if not self._conn:
            raise StoreError("Not connected", self.name, "aggregate")

        # Try to use rollups for common aggregations
        if aggregation.lower() in ("avg", "sum", "min", "max", "count") and granularity in (
            TimeGranularity.HOUR, TimeGranularity.DAY
        ):
            return await self._query_rollup(
                metric_name, start, end, aggregation, granularity, labels
            )

        # Fall back to raw query
        try:
            agg_func = {
                "sum": "SUM(value)",
                "avg": "AVG(value)",
                "mean": "AVG(value)",
                "min": "MIN(value)",
                "max": "MAX(value)",
                "count": "COUNT(*)",
            }.get(aggregation.lower(), "AVG(value)")

            # Use strftime for bucketing
            bucket_format = {
                TimeGranularity.MINUTE: "%Y-%m-%dT%H:%M:00",
                TimeGranularity.HOUR: "%Y-%m-%dT%H:00:00",
                TimeGranularity.DAY: "%Y-%m-%d",
                TimeGranularity.WEEK: "%Y-W%W",
                TimeGranularity.MONTH: "%Y-%m-01",
            }.get(granularity, "%Y-%m-%d")

            query = f"""
                SELECT strftime('{bucket_format}', timestamp) as bucket,
                       {agg_func} as agg_value,
                       COUNT(*) as sample_count
                FROM timeseries
                WHERE metric_name = ?
                AND timestamp_unix >= ?
                AND timestamp_unix <= ?
            """
            params: list[Any] = [metric_name, start.timestamp(), end.timestamp()]

            if labels:
                for key, value in labels.items():
                    query += " AND json_extract(labels, ?) = ?"
                    params.extend([f"$.{key}", value])

            query += " GROUP BY bucket ORDER BY bucket"

            cursor = self._conn.execute(query, params)
            rows = cursor.fetchall()

            points = []
            for row in rows:
                # Parse bucket timestamp
                bucket_str = row["bucket"]
                if "T" in bucket_str:
                    bucket = datetime.fromisoformat(bucket_str)
                elif "-W" in bucket_str:
                    year, week = bucket_str.split("-W")
                    bucket = datetime.strptime(f"{year} {week} 1", "%Y %W %w")
                else:
                    bucket = datetime.fromisoformat(bucket_str + "T00:00:00")

                points.append(TimeSeriesPoint(
                    timestamp=bucket,
                    value=row["agg_value"],
                    metadata={
                        "aggregation": aggregation,
                        "sample_count": row["sample_count"],
                    },
                ))

            return points

        except Exception as e:
            raise StoreError(
                f"Failed to aggregate: {e}",
                self.name,
                "aggregate",
            )

    async def _query_rollup(
        self,
        metric_name: str,
        start: datetime,
        end: datetime,
        aggregation: str,
        granularity: TimeGranularity,
        labels: dict[str, str] | None,
    ) -> list[TimeSeriesPoint]:
        """Query from rollup table."""
        if not self._conn:
            raise StoreError("Not connected", self.name, "query_rollup")

        agg = aggregation.lower()
        value_expr = {
            "sum": "sum_value",
            "avg": "sum_value / count_value",
            "min": "min_value",
            "max": "max_value",
            "count": "count_value",
        }.get(agg, "sum_value / count_value")

        query = f"""
            SELECT bucket_start, {value_expr} as value, count_value
            FROM rollups
            WHERE metric_name = ?
            AND granularity = ?
            AND bucket_start_unix >= ?
            AND bucket_start_unix <= ?
        """
        params: list[Any] = [
            metric_name,
            granularity.value,
            start.timestamp(),
            end.timestamp(),
        ]

        if labels:
            labels_json = json.dumps(labels)
            query += " AND labels = ?"
            params.append(labels_json)

        query += " ORDER BY bucket_start_unix"

        cursor = self._conn.execute(query, params)
        rows = cursor.fetchall()

        return [
            TimeSeriesPoint(
                timestamp=datetime.fromisoformat(row["bucket_start"]),
                value=row["value"],
                metadata={
                    "aggregation": aggregation,
                    "sample_count": row["count_value"],
                    "source": "rollup",
                },
            )
            for row in rows
        ]

    async def delete(
        self,
        metric_name: str,
        before: datetime | None = None,
        labels: dict[str, str] | None = None,
    ) -> int:
        """Delete points."""
        if not self._conn:
            raise StoreError("Not connected", self.name, "delete")

        try:
            query = "DELETE FROM timeseries WHERE metric_name = ?"
            params: list[Any] = [metric_name]

            if before:
                query += " AND timestamp_unix < ?"
                params.append(before.timestamp())

            if labels:
                for key, value in labels.items():
                    query += " AND json_extract(labels, ?) = ?"
                    params.extend([f"$.{key}", value])

            cursor = self._conn.execute(query, params)
            deleted = cursor.rowcount

            # Also delete from rollups
            rollup_query = "DELETE FROM rollups WHERE metric_name = ?"
            rollup_params: list[Any] = [metric_name]

            if before:
                rollup_query += " AND bucket_start_unix < ?"
                rollup_params.append(before.timestamp())

            if labels:
                rollup_query += " AND labels = ?"
                rollup_params.append(json.dumps(labels))

            self._conn.execute(rollup_query, rollup_params)

            return deleted

        except Exception as e:
            raise StoreError(
                f"Failed to delete: {e}",
                self.name,
                "delete",
            )

    async def vacuum(self) -> None:
        """Optimize database by running VACUUM."""
        if self._conn:
            self._conn.execute("VACUUM")

    async def get_database_size(self) -> int:
        """Get database size in bytes."""
        if self._db_path == ":memory:":
            return 0

        try:
            return Path(self._db_path).stat().st_size
        except FileNotFoundError:
            return 0
