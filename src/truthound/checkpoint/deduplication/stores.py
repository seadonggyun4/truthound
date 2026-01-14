"""Deduplication Store Implementations.

This module provides storage backends for notification deduplication:
- InMemoryDeduplicationStore: Thread-safe in-memory storage
- RedisStreamsDeduplicationStore: Redis Streams-based distributed storage
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from truthound.checkpoint.deduplication.protocols import (
    DeduplicationRecord,
    DeduplicationStats,
    DeduplicationStore,
    NotificationFingerprint,
    TimeWindow,
)

logger = logging.getLogger(__name__)


@dataclass
class InMemoryDeduplicationStore:
    """Thread-safe in-memory deduplication store.

    Suitable for single-process deployments or testing. Uses
    an OrderedDict for efficient LRU-style cleanup.

    Attributes:
        max_size: Maximum number of records to store.
        auto_cleanup_interval: Interval for automatic cleanup (seconds).

    Example:
        >>> store = InMemoryDeduplicationStore(max_size=10000)
        >>> record = store.put(fingerprint, TimeWindow(minutes=5))
        >>> existing = store.get(fingerprint.key)
    """

    max_size: int = 100000
    auto_cleanup_interval: int = 60

    _records: OrderedDict[str, DeduplicationRecord] = field(
        default_factory=OrderedDict
    )
    _lock: threading.RLock = field(default_factory=threading.RLock)
    _stats: DeduplicationStats = field(default_factory=DeduplicationStats)
    _last_cleanup: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        """Initialize the store."""
        self._records = OrderedDict()
        self._lock = threading.RLock()
        self._stats = DeduplicationStats()
        self._last_cleanup = datetime.now()

    def get(self, key: str) -> DeduplicationRecord | None:
        """Get a record by key.

        Returns None if not found or expired.
        """
        with self._lock:
            record = self._records.get(key)
            if record is None:
                return None

            if record.is_expired:
                del self._records[key]
                return None

            # Move to end (LRU)
            self._records.move_to_end(key)
            return record

    def put(
        self,
        fingerprint: NotificationFingerprint,
        window: TimeWindow,
        metadata: dict[str, Any] | None = None,
    ) -> DeduplicationRecord:
        """Store a deduplication record.

        Creates a new record with expiration based on the time window.
        """
        with self._lock:
            now = datetime.now()
            expires_at = now + window.to_timedelta()

            record = DeduplicationRecord(
                fingerprint=fingerprint,
                sent_at=now,
                expires_at=expires_at,
                count=1,
                metadata=metadata or {},
            )

            # Check if we need to cleanup
            self._maybe_cleanup()

            # Enforce max size (remove oldest)
            while len(self._records) >= self.max_size:
                self._records.popitem(last=False)

            self._records[fingerprint.key] = record
            self._stats.notifications_sent += 1
            self._stats.store_size = len(self._records)

            # Update timestamp stats
            if self._stats.oldest_record is None or now < self._stats.oldest_record:
                self._stats.oldest_record = now
            self._stats.newest_record = now

            return record

    def increment(self, key: str) -> DeduplicationRecord | None:
        """Increment duplicate count for a record."""
        with self._lock:
            record = self._records.get(key)
            if record is None or record.is_expired:
                return None

            record.count += 1
            record.last_duplicate_at = datetime.now()
            self._stats.duplicates_found += 1
            return record

    def delete(self, key: str) -> bool:
        """Delete a record."""
        with self._lock:
            if key in self._records:
                del self._records[key]
                self._stats.store_size = len(self._records)
                return True
            return False

    def cleanup_expired(self) -> int:
        """Remove all expired records."""
        with self._lock:
            now = datetime.now()
            expired_keys = [
                key for key, record in self._records.items() if record.is_expired
            ]

            for key in expired_keys:
                del self._records[key]

            removed = len(expired_keys)
            self._stats.store_size = len(self._records)
            self._last_cleanup = now

            if removed > 0:
                logger.debug(f"Cleaned up {removed} expired deduplication records")

            # Update oldest record
            if self._records:
                self._stats.oldest_record = min(
                    r.sent_at for r in self._records.values()
                )
            else:
                self._stats.oldest_record = None

            return removed

    def _maybe_cleanup(self) -> None:
        """Run cleanup if interval has passed."""
        now = datetime.now()
        if (now - self._last_cleanup).total_seconds() >= self.auto_cleanup_interval:
            self.cleanup_expired()

    def get_stats(self) -> DeduplicationStats:
        """Get store statistics."""
        with self._lock:
            self._stats.store_size = len(self._records)
            return DeduplicationStats(
                total_checked=self._stats.total_checked,
                duplicates_found=self._stats.duplicates_found,
                notifications_sent=self._stats.notifications_sent,
                store_size=self._stats.store_size,
                oldest_record=self._stats.oldest_record,
                newest_record=self._stats.newest_record,
            )

    def clear(self) -> None:
        """Clear all records."""
        with self._lock:
            self._records.clear()
            self._stats = DeduplicationStats()

    def record_check(self) -> None:
        """Record a deduplication check (for stats)."""
        with self._lock:
            self._stats.total_checked += 1

    def get_all_records(self) -> list[DeduplicationRecord]:
        """Get all non-expired records (for debugging)."""
        with self._lock:
            return [r for r in self._records.values() if not r.is_expired]


@dataclass
class RedisStreamsDeduplicationStore:
    """Redis Streams-based distributed deduplication store.

    Uses Redis Streams for distributed notification deduplication
    with automatic expiration via TTL. Supports consumer groups
    for multi-instance coordination.

    Attributes:
        redis_url: Redis connection URL.
        stream_key: Redis Stream key prefix.
        hash_key: Redis Hash key for record lookup.
        consumer_group: Consumer group name.
        max_stream_length: Maximum stream length (MAXLEN).

    Example:
        >>> store = RedisStreamsDeduplicationStore(
        ...     redis_url="redis://localhost:6379",
        ...     stream_key="dedup:notifications",
        ... )
        >>> await store.initialize()
        >>> record = await store.put_async(fingerprint, TimeWindow(minutes=5))
    """

    redis_url: str = "redis://localhost:6379"
    stream_key: str = "truthound:dedup:stream"
    hash_key: str = "truthound:dedup:records"
    consumer_group: str = "truthound-dedup"
    max_stream_length: int = 100000
    connection_timeout: int = 5
    socket_timeout: int = 5

    _redis: Any = field(default=None, repr=False)
    _initialized: bool = field(default=False)
    _stats: DeduplicationStats = field(default_factory=DeduplicationStats)
    _lock: asyncio.Lock = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Initialize fields."""
        self._initialized = False
        self._stats = DeduplicationStats()
        self._redis = None
        self._lock = None

    async def initialize(self) -> None:
        """Initialize Redis connection and consumer group."""
        if self._initialized:
            return

        try:
            import redis.asyncio as aioredis
        except ImportError:
            raise ImportError(
                "redis[asyncio] is required for RedisStreamsDeduplicationStore. "
                "Install with: pip install redis[asyncio]"
            )

        self._lock = asyncio.Lock()

        # Create Redis connection
        self._redis = await aioredis.from_url(
            self.redis_url,
            socket_connect_timeout=self.connection_timeout,
            socket_timeout=self.socket_timeout,
            decode_responses=True,
        )

        # Create consumer group if not exists
        try:
            await self._redis.xgroup_create(
                self.stream_key,
                self.consumer_group,
                id="0",
                mkstream=True,
            )
        except Exception as e:
            # Group already exists
            if "BUSYGROUP" not in str(e):
                raise

        self._initialized = True
        logger.info(f"RedisStreamsDeduplicationStore initialized: {self.stream_key}")

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None
            self._initialized = False

    def _ensure_initialized(self) -> None:
        """Ensure store is initialized."""
        if not self._initialized:
            raise RuntimeError(
                "RedisStreamsDeduplicationStore not initialized. "
                "Call initialize() first."
            )

    def get(self, key: str) -> DeduplicationRecord | None:
        """Synchronous get (wraps async)."""
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.get_async(key))
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(self.get_async(key))

    async def get_async(self, key: str) -> DeduplicationRecord | None:
        """Get a record by key (async)."""
        self._ensure_initialized()

        async with self._lock:
            data = await self._redis.hget(self.hash_key, key)
            if data is None:
                return None

            record = DeduplicationRecord.from_dict(json.loads(data))

            if record.is_expired:
                await self._redis.hdel(self.hash_key, key)
                return None

            return record

    def put(
        self,
        fingerprint: NotificationFingerprint,
        window: TimeWindow,
        metadata: dict[str, Any] | None = None,
    ) -> DeduplicationRecord:
        """Synchronous put (wraps async)."""
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(
                self.put_async(fingerprint, window, metadata)
            )
        except RuntimeError:
            return asyncio.run(self.put_async(fingerprint, window, metadata))

    async def put_async(
        self,
        fingerprint: NotificationFingerprint,
        window: TimeWindow,
        metadata: dict[str, Any] | None = None,
    ) -> DeduplicationRecord:
        """Store a deduplication record (async)."""
        self._ensure_initialized()

        now = datetime.now()
        expires_at = now + window.to_timedelta()

        record = DeduplicationRecord(
            fingerprint=fingerprint,
            sent_at=now,
            expires_at=expires_at,
            count=1,
            metadata=metadata or {},
        )

        async with self._lock:
            # Store in hash
            await self._redis.hset(
                self.hash_key,
                fingerprint.key,
                json.dumps(record.to_dict()),
            )

            # Set TTL on hash field via separate key
            ttl_key = f"{self.hash_key}:ttl:{fingerprint.key}"
            await self._redis.setex(
                ttl_key,
                window.total_seconds,
                fingerprint.key,
            )

            # Add to stream for event tracking
            await self._redis.xadd(
                self.stream_key,
                {
                    "type": "notification_sent",
                    "key": fingerprint.key,
                    "checkpoint": fingerprint.checkpoint_name,
                    "action": fingerprint.action_type,
                    "timestamp": now.isoformat(),
                },
                maxlen=self.max_stream_length,
            )

            self._stats.notifications_sent += 1

        return record

    def increment(self, key: str) -> DeduplicationRecord | None:
        """Synchronous increment (wraps async)."""
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.increment_async(key))
        except RuntimeError:
            return asyncio.run(self.increment_async(key))

    async def increment_async(self, key: str) -> DeduplicationRecord | None:
        """Increment duplicate count (async)."""
        self._ensure_initialized()

        async with self._lock:
            data = await self._redis.hget(self.hash_key, key)
            if data is None:
                return None

            record = DeduplicationRecord.from_dict(json.loads(data))

            if record.is_expired:
                await self._redis.hdel(self.hash_key, key)
                return None

            record.count += 1
            record.last_duplicate_at = datetime.now()

            await self._redis.hset(
                self.hash_key,
                key,
                json.dumps(record.to_dict()),
            )

            # Add to stream
            await self._redis.xadd(
                self.stream_key,
                {
                    "type": "duplicate_suppressed",
                    "key": key,
                    "count": str(record.count),
                    "timestamp": datetime.now().isoformat(),
                },
                maxlen=self.max_stream_length,
            )

            self._stats.duplicates_found += 1

        return record

    def delete(self, key: str) -> bool:
        """Synchronous delete (wraps async)."""
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.delete_async(key))
        except RuntimeError:
            return asyncio.run(self.delete_async(key))

    async def delete_async(self, key: str) -> bool:
        """Delete a record (async)."""
        self._ensure_initialized()

        async with self._lock:
            result = await self._redis.hdel(self.hash_key, key)
            ttl_key = f"{self.hash_key}:ttl:{key}"
            await self._redis.delete(ttl_key)
            return result > 0

    def cleanup_expired(self) -> int:
        """Synchronous cleanup (wraps async)."""
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.cleanup_expired_async())
        except RuntimeError:
            return asyncio.run(self.cleanup_expired_async())

    async def cleanup_expired_async(self) -> int:
        """Remove expired records (async)."""
        self._ensure_initialized()

        removed = 0
        async with self._lock:
            # Get all keys
            all_keys = await self._redis.hkeys(self.hash_key)

            for key in all_keys:
                data = await self._redis.hget(self.hash_key, key)
                if data:
                    record = DeduplicationRecord.from_dict(json.loads(data))
                    if record.is_expired:
                        await self._redis.hdel(self.hash_key, key)
                        removed += 1

        if removed > 0:
            logger.debug(f"Cleaned up {removed} expired Redis dedup records")

        return removed

    def get_stats(self) -> DeduplicationStats:
        """Get store statistics."""
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.get_stats_async())
        except RuntimeError:
            return asyncio.run(self.get_stats_async())

    async def get_stats_async(self) -> DeduplicationStats:
        """Get statistics (async)."""
        self._ensure_initialized()

        async with self._lock:
            store_size = await self._redis.hlen(self.hash_key)

            # Get stream info for timing stats
            try:
                stream_info = await self._redis.xinfo_stream(self.stream_key)
                first_entry = stream_info.get("first-entry")
                last_entry = stream_info.get("last-entry")

                oldest = None
                newest = None
                if first_entry and first_entry[1].get("timestamp"):
                    oldest = datetime.fromisoformat(first_entry[1]["timestamp"])
                if last_entry and last_entry[1].get("timestamp"):
                    newest = datetime.fromisoformat(last_entry[1]["timestamp"])

                return DeduplicationStats(
                    total_checked=self._stats.total_checked,
                    duplicates_found=self._stats.duplicates_found,
                    notifications_sent=self._stats.notifications_sent,
                    store_size=store_size,
                    oldest_record=oldest,
                    newest_record=newest,
                )
            except Exception:
                return DeduplicationStats(
                    total_checked=self._stats.total_checked,
                    duplicates_found=self._stats.duplicates_found,
                    notifications_sent=self._stats.notifications_sent,
                    store_size=store_size,
                )

    def clear(self) -> None:
        """Synchronous clear (wraps async)."""
        try:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self.clear_async())
        except RuntimeError:
            asyncio.run(self.clear_async())

    async def clear_async(self) -> None:
        """Clear all records (async)."""
        self._ensure_initialized()

        async with self._lock:
            await self._redis.delete(self.hash_key)
            await self._redis.delete(self.stream_key)
            self._stats = DeduplicationStats()

    def record_check(self) -> None:
        """Record a deduplication check (for stats)."""
        self._stats.total_checked += 1

    async def read_events(
        self,
        count: int = 100,
        block: int = 0,
        consumer_name: str = "consumer-1",
    ) -> list[dict[str, Any]]:
        """Read events from the stream.

        Args:
            count: Maximum events to read.
            block: Block timeout in milliseconds (0 = no block).
            consumer_name: Consumer name for group reading.

        Returns:
            List of event dictionaries.
        """
        self._ensure_initialized()

        events = []
        result = await self._redis.xreadgroup(
            self.consumer_group,
            consumer_name,
            {self.stream_key: ">"},
            count=count,
            block=block,
        )

        if result:
            for stream_name, messages in result:
                for message_id, data in messages:
                    events.append(
                        {
                            "id": message_id,
                            "stream": stream_name,
                            **data,
                        }
                    )

        return events

    async def acknowledge(self, message_ids: list[str]) -> int:
        """Acknowledge processed messages.

        Args:
            message_ids: List of message IDs to acknowledge.

        Returns:
            Number of acknowledged messages.
        """
        self._ensure_initialized()

        if not message_ids:
            return 0

        return await self._redis.xack(
            self.stream_key,
            self.consumer_group,
            *message_ids,
        )
