"""TimescaleDB time series store.

Provides high-performance time series storage using TimescaleDB
(PostgreSQL extension).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from truthound.checkpoint.analytics.protocols import (
    TimeGranularity,
    TimeSeriesPoint,
    StoreError,
)
from truthound.checkpoint.analytics.stores.base import BaseTimeSeriesStore

logger = logging.getLogger(__name__)


class TimescaleDBStore(BaseTimeSeriesStore):
    """TimescaleDB-based time series store.

    High-performance time series storage using TimescaleDB.
    Supports automatic data partitioning, compression, and
    continuous aggregates.

    Example:
        >>> store = TimescaleDBStore(
        ...     connection_string="postgresql://user:pass@localhost/tsdb",
        ... )
        >>> await store.connect()
        >>>
        >>> # Write data
        >>> await store.write("checkpoint.duration", TimeSeriesPoint(
        ...     timestamp=datetime.now(),
        ...     value=1234.5,
        ...     labels={"checkpoint": "daily_check"},
        ... ))
    """

    def __init__(
        self,
        connection_string: str = "postgresql://localhost/timescale",
        name: str = "timescale",
        retention_days: int = 90,
        chunk_interval: str = "1 day",
        compression_after_days: int = 7,
    ) -> None:
        """Initialize TimescaleDB store.

        Args:
            connection_string: PostgreSQL connection string.
            name: Store name.
            retention_days: Data retention period.
            chunk_interval: TimescaleDB chunk interval.
            compression_after_days: Compress chunks after this many days.
        """
        super().__init__(name=name, retention_days=retention_days)
        self._connection_string = connection_string
        self._chunk_interval = chunk_interval
        self._compression_after_days = compression_after_days
        self._pool: Any = None

    async def connect(self) -> None:
        """Connect to TimescaleDB."""
        try:
            import asyncpg
        except ImportError:
            raise StoreError(
                "asyncpg package not installed",
                self.name,
                "Install with: pip install asyncpg",
            )

        try:
            self._pool = await asyncpg.create_pool(
                self._connection_string,
                min_size=2,
                max_size=10,
            )

            # Create schema
            await self._create_schema()

            await super().connect()
            logger.info(f"Connected to TimescaleDB")

        except Exception as e:
            raise StoreError(
                f"Failed to connect to TimescaleDB: {e}",
                self.name,
                "connect",
            )

    async def disconnect(self) -> None:
        """Disconnect from TimescaleDB."""
        if self._pool:
            await self._pool.close()
            self._pool = None

        await super().disconnect()

    async def _create_schema(self) -> None:
        """Create TimescaleDB schema."""
        if not self._pool:
            return

        async with self._pool.acquire() as conn:
            # Enable TimescaleDB extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE")

            # Create metrics table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS timeseries (
                    time TIMESTAMPTZ NOT NULL,
                    metric_name TEXT NOT NULL,
                    value DOUBLE PRECISION NOT NULL,
                    labels JSONB,
                    metadata JSONB
                )
            """)

            # Create hypertable (idempotent with if_not_exists)
            try:
                await conn.execute(f"""
                    SELECT create_hypertable(
                        'timeseries',
                        'time',
                        chunk_time_interval => INTERVAL '{self._chunk_interval}',
                        if_not_exists => TRUE
                    )
                """)
            except Exception:
                pass  # Table might already be a hypertable

            # Create indexes
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timeseries_metric_time
                ON timeseries(metric_name, time DESC)
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timeseries_labels
                ON timeseries USING GIN(labels)
            """)

            # Set up compression (if supported)
            try:
                await conn.execute("""
                    ALTER TABLE timeseries SET (
                        timescaledb.compress,
                        timescaledb.compress_segmentby = 'metric_name'
                    )
                """)

                await conn.execute(f"""
                    SELECT add_compression_policy(
                        'timeseries',
                        INTERVAL '{self._compression_after_days} days',
                        if_not_exists => TRUE
                    )
                """)
            except Exception:
                pass  # Compression might not be available

            # Set up retention policy
            try:
                await conn.execute(f"""
                    SELECT add_retention_policy(
                        'timeseries',
                        INTERVAL '{self._retention_days} days',
                        if_not_exists => TRUE
                    )
                """)
            except Exception:
                pass  # Retention policy might already exist

    async def write(self, metric_name: str, point: TimeSeriesPoint) -> None:
        """Write a single point."""
        if not self._pool:
            raise StoreError("Not connected", self.name, "write")

        try:
            async with self._pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO timeseries (time, metric_name, value, labels, metadata)
                    VALUES ($1, $2, $3, $4, $5)
                """,
                    point.timestamp,
                    metric_name,
                    point.value,
                    json.dumps(point.labels) if point.labels else None,
                    json.dumps(point.metadata) if point.metadata else None,
                )
        except Exception as e:
            raise StoreError(f"Failed to write: {e}", self.name, "write")

    async def write_batch(self, metric_name: str, points: list[TimeSeriesPoint]) -> None:
        """Write multiple points."""
        if not points or not self._pool:
            return

        try:
            async with self._pool.acquire() as conn:
                # Use COPY for best performance
                data = [
                    (
                        p.timestamp,
                        metric_name,
                        p.value,
                        json.dumps(p.labels) if p.labels else None,
                        json.dumps(p.metadata) if p.metadata else None,
                    )
                    for p in points
                ]

                await conn.copy_records_to_table(
                    "timeseries",
                    records=data,
                    columns=["time", "metric_name", "value", "labels", "metadata"],
                )
        except Exception as e:
            raise StoreError(f"Failed to write batch: {e}", self.name, "write_batch")

    async def query(
        self,
        metric_name: str,
        start: datetime,
        end: datetime,
        granularity: TimeGranularity | None = None,
        labels: dict[str, str] | None = None,
    ) -> list[TimeSeriesPoint]:
        """Query points in a time range."""
        if not self._pool:
            raise StoreError("Not connected", self.name, "query")

        try:
            async with self._pool.acquire() as conn:
                if granularity:
                    # Use time_bucket for downsampling
                    bucket_interval = self._granularity_to_interval(granularity)
                    query = f"""
                        SELECT time_bucket('{bucket_interval}', time) AS bucket,
                               AVG(value) AS value,
                               COUNT(*) as sample_count
                        FROM timeseries
                        WHERE metric_name = $1
                        AND time >= $2
                        AND time <= $3
                    """
                    params = [metric_name, start, end]

                    if labels:
                        query += " AND labels @> $4"
                        params.append(json.dumps(labels))

                    query += " GROUP BY bucket ORDER BY bucket"
                else:
                    query = """
                        SELECT time, value, labels, metadata
                        FROM timeseries
                        WHERE metric_name = $1
                        AND time >= $2
                        AND time <= $3
                    """
                    params = [metric_name, start, end]

                    if labels:
                        query += " AND labels @> $4"
                        params.append(json.dumps(labels))

                    query += " ORDER BY time"

                rows = await conn.fetch(query, *params)

                points = []
                for row in rows:
                    if granularity:
                        point = TimeSeriesPoint(
                            timestamp=row["bucket"],
                            value=row["value"],
                            metadata={"sample_count": row["sample_count"]},
                        )
                    else:
                        point = TimeSeriesPoint(
                            timestamp=row["time"],
                            value=row["value"],
                            labels=json.loads(row["labels"]) if row["labels"] else {},
                            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                        )
                    points.append(point)

                return points

        except Exception as e:
            raise StoreError(f"Failed to query: {e}", self.name, "query")

    def _granularity_to_interval(self, granularity: TimeGranularity) -> str:
        """Convert granularity to PostgreSQL interval."""
        mapping = {
            TimeGranularity.SECOND: "1 second",
            TimeGranularity.MINUTE: "1 minute",
            TimeGranularity.HOUR: "1 hour",
            TimeGranularity.DAY: "1 day",
            TimeGranularity.WEEK: "1 week",
            TimeGranularity.MONTH: "1 month",
        }
        return mapping.get(granularity, "1 day")

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
        if not self._pool:
            raise StoreError("Not connected", self.name, "aggregate")

        try:
            bucket_interval = self._granularity_to_interval(granularity)

            agg_func = {
                "sum": "SUM(value)",
                "avg": "AVG(value)",
                "mean": "AVG(value)",
                "min": "MIN(value)",
                "max": "MAX(value)",
                "count": "COUNT(*)",
                "first": "first(value, time)",
                "last": "last(value, time)",
            }.get(aggregation.lower(), "AVG(value)")

            async with self._pool.acquire() as conn:
                query = f"""
                    SELECT time_bucket('{bucket_interval}', time) AS bucket,
                           {agg_func} AS value,
                           COUNT(*) as sample_count
                    FROM timeseries
                    WHERE metric_name = $1
                    AND time >= $2
                    AND time <= $3
                """
                params = [metric_name, start, end]

                if labels:
                    query += " AND labels @> $4"
                    params.append(json.dumps(labels))

                query += " GROUP BY bucket ORDER BY bucket"

                rows = await conn.fetch(query, *params)

                return [
                    TimeSeriesPoint(
                        timestamp=row["bucket"],
                        value=row["value"],
                        metadata={
                            "aggregation": aggregation,
                            "sample_count": row["sample_count"],
                        },
                    )
                    for row in rows
                ]

        except Exception as e:
            raise StoreError(f"Failed to aggregate: {e}", self.name, "aggregate")

    async def delete(
        self,
        metric_name: str,
        before: datetime | None = None,
        labels: dict[str, str] | None = None,
    ) -> int:
        """Delete points."""
        if not self._pool:
            raise StoreError("Not connected", self.name, "delete")

        try:
            async with self._pool.acquire() as conn:
                query = "DELETE FROM timeseries WHERE metric_name = $1"
                params: list[Any] = [metric_name]

                if before:
                    query += f" AND time < ${len(params) + 1}"
                    params.append(before)

                if labels:
                    query += f" AND labels @> ${len(params) + 1}"
                    params.append(json.dumps(labels))

                result = await conn.execute(query, *params)

                # Extract count from result
                count_str = result.split()[-1]
                return int(count_str) if count_str.isdigit() else 0

        except Exception as e:
            raise StoreError(f"Failed to delete: {e}", self.name, "delete")

    async def create_continuous_aggregate(
        self,
        metric_name: str,
        granularity: TimeGranularity,
        aggregations: list[str] = ["avg", "min", "max", "count"],
    ) -> None:
        """Create a continuous aggregate for faster queries.

        Args:
            metric_name: Metric to aggregate.
            granularity: Aggregation granularity.
            aggregations: Aggregation functions to compute.
        """
        if not self._pool:
            return

        view_name = f"agg_{metric_name.replace('.', '_')}_{granularity.value}"
        bucket_interval = self._granularity_to_interval(granularity)

        agg_columns = []
        for agg in aggregations:
            if agg == "avg":
                agg_columns.append(f"AVG(value) AS avg_value")
            elif agg == "min":
                agg_columns.append(f"MIN(value) AS min_value")
            elif agg == "max":
                agg_columns.append(f"MAX(value) AS max_value")
            elif agg == "count":
                agg_columns.append(f"COUNT(*) AS count_value")
            elif agg == "sum":
                agg_columns.append(f"SUM(value) AS sum_value")

        try:
            async with self._pool.acquire() as conn:
                await conn.execute(f"""
                    CREATE MATERIALIZED VIEW IF NOT EXISTS {view_name}
                    WITH (timescaledb.continuous) AS
                    SELECT time_bucket('{bucket_interval}', time) AS bucket,
                           metric_name,
                           {', '.join(agg_columns)}
                    FROM timeseries
                    WHERE metric_name = '{metric_name}'
                    GROUP BY bucket, metric_name
                    WITH NO DATA
                """)

                # Add refresh policy
                await conn.execute(f"""
                    SELECT add_continuous_aggregate_policy(
                        '{view_name}',
                        start_offset => INTERVAL '3 days',
                        end_offset => INTERVAL '1 hour',
                        schedule_interval => INTERVAL '1 hour',
                        if_not_exists => TRUE
                    )
                """)

                logger.info(f"Created continuous aggregate: {view_name}")

        except Exception as e:
            logger.warning(f"Failed to create continuous aggregate: {e}")

    async def get_database_stats(self) -> dict[str, Any]:
        """Get database statistics."""
        if not self._pool:
            return {}

        try:
            async with self._pool.acquire() as conn:
                # Get chunk info
                chunks = await conn.fetch("""
                    SELECT hypertable_name,
                           chunk_name,
                           range_start,
                           range_end,
                           is_compressed
                    FROM timescaledb_information.chunks
                    WHERE hypertable_name = 'timeseries'
                    ORDER BY range_start DESC
                    LIMIT 10
                """)

                # Get table size
                size = await conn.fetchval("""
                    SELECT pg_size_pretty(hypertable_size('timeseries'))
                """)

                return {
                    "table_size": size,
                    "recent_chunks": [dict(c) for c in chunks],
                }

        except Exception as e:
            return {"error": str(e)}
