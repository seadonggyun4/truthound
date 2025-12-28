"""Base adapter class for streaming platforms.

Provides common functionality for all stream adapters including:
- Connection lifecycle management
- Metrics collection
- Error handling with retries
- State management
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncIterator, Generic, TypeVar
import asyncio
import logging
import time
from collections import deque

from truthound.realtime.protocols import (
    IStreamSource,
    IStreamSink,
    IMetricsCollector,
    IStateStore,
    StreamMessage,
    MessageBatch,
    StreamMetrics,
    StreamSourceConfig,
    StreamSinkConfig,
    StreamingError,
    ConnectionError,
    TimeoutError,
)


logger = logging.getLogger(__name__)

T = TypeVar("T")
ConfigT = TypeVar("ConfigT")


class AdapterState(str, Enum):
    """Adapter connection states."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    DISCONNECTING = "disconnecting"
    ERROR = "error"


@dataclass
class AdapterConfig(StreamSourceConfig, StreamSinkConfig):
    """Combined adapter configuration."""

    # Connection
    connect_timeout_ms: int = 30000
    reconnect_delay_ms: int = 1000
    max_reconnect_attempts: int = 5

    # Health check
    health_check_interval_ms: int = 30000
    enable_health_check: bool = True


# =============================================================================
# Metrics Collector Implementation
# =============================================================================


class DefaultMetricsCollector(IMetricsCollector):
    """Default in-memory metrics collector."""

    def __init__(self, window_size: int = 1000):
        self._consumed = 0
        self._produced = 0
        self._failed = 0
        self._bytes_consumed = 0
        self._bytes_produced = 0
        self._latencies: deque[float] = deque(maxlen=window_size)
        self._last_timestamp: datetime | None = None
        self._start_time = time.monotonic()

    def record_consumed(self, count: int = 1, bytes_size: int = 0) -> None:
        self._consumed += count
        self._bytes_consumed += bytes_size
        self._last_timestamp = datetime.now(timezone.utc)

    def record_produced(self, count: int = 1, bytes_size: int = 0) -> None:
        self._produced += count
        self._bytes_produced += bytes_size

    def record_failed(self, count: int = 1) -> None:
        self._failed += count

    def record_latency(self, latency_ms: float) -> None:
        self._latencies.append(latency_ms)

    def get_metrics(self) -> StreamMetrics:
        latencies = list(self._latencies)
        elapsed = time.monotonic() - self._start_time

        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        max_latency = max(latencies) if latencies else 0.0
        p99_latency = sorted(latencies)[int(len(latencies) * 0.99)] if len(latencies) >= 100 else max_latency
        throughput = self._consumed / elapsed if elapsed > 0 else 0.0

        return StreamMetrics(
            messages_consumed=self._consumed,
            messages_produced=self._produced,
            messages_failed=self._failed,
            bytes_consumed=self._bytes_consumed,
            bytes_produced=self._bytes_produced,
            avg_latency_ms=avg_latency,
            max_latency_ms=max_latency,
            p99_latency_ms=p99_latency,
            throughput_per_second=throughput,
            last_message_timestamp=self._last_timestamp,
        )

    def reset(self) -> None:
        self._consumed = 0
        self._produced = 0
        self._failed = 0
        self._bytes_consumed = 0
        self._bytes_produced = 0
        self._latencies.clear()
        self._last_timestamp = None
        self._start_time = time.monotonic()


# =============================================================================
# No-Op Metrics Collector
# =============================================================================


class NoOpMetricsCollector(IMetricsCollector):
    """No-op metrics collector for when metrics are disabled."""

    def record_consumed(self, count: int = 1, bytes_size: int = 0) -> None:
        pass

    def record_produced(self, count: int = 1, bytes_size: int = 0) -> None:
        pass

    def record_failed(self, count: int = 1) -> None:
        pass

    def record_latency(self, latency_ms: float) -> None:
        pass

    def get_metrics(self) -> StreamMetrics:
        return StreamMetrics()

    def reset(self) -> None:
        pass


# =============================================================================
# In-Memory State Store
# =============================================================================


class InMemoryStateStore(IStateStore[Any], Generic[T]):
    """Thread-safe in-memory state store."""

    def __init__(self):
        self._store: dict[str, tuple[T, float | None]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> T | None:
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None

            value, expires_at = entry
            if expires_at is not None and time.time() > expires_at:
                del self._store[key]
                return None

            return value

    async def put(self, key: str, value: T, ttl: int | None = None) -> None:
        async with self._lock:
            expires_at = time.time() + ttl if ttl is not None else None
            self._store[key] = (value, expires_at)

    async def delete(self, key: str) -> bool:
        async with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    async def get_all(self, prefix: str = "") -> dict[str, T]:
        async with self._lock:
            result = {}
            now = time.time()
            expired_keys = []

            for key, (value, expires_at) in self._store.items():
                if expires_at is not None and now > expires_at:
                    expired_keys.append(key)
                    continue
                if key.startswith(prefix):
                    result[key] = value

            # Clean up expired
            for key in expired_keys:
                del self._store[key]

            return result

    async def clear(self) -> None:
        async with self._lock:
            self._store.clear()


# =============================================================================
# Base Stream Adapter
# =============================================================================


class BaseStreamAdapter(ABC, Generic[T, ConfigT]):
    """Abstract base class for stream adapters.

    Provides common functionality:
    - Connection lifecycle (connect, disconnect, reconnect)
    - Metrics collection
    - Error handling with exponential backoff
    - State management integration

    Subclasses must implement:
    - _do_connect(): Platform-specific connection
    - _do_disconnect(): Platform-specific disconnection
    - _do_consume(): Platform-specific message consumption
    - _do_produce(): Platform-specific message production
    """

    adapter_type: str = "base"

    def __init__(
        self,
        config: ConfigT,
        *,
        state_store: IStateStore | None = None,
        metrics_collector: IMetricsCollector | None = None,
    ):
        """Initialize the adapter.

        Args:
            config: Adapter configuration
            state_store: Optional state store for stateful processing
            metrics_collector: Optional metrics collector
        """
        self._config: ConfigT = config
        self._state = AdapterState.DISCONNECTED
        self._state_store = state_store or InMemoryStateStore()
        self._metrics = metrics_collector or (
            DefaultMetricsCollector() if getattr(config, "enable_metrics", True) else NoOpMetricsCollector()
        )
        self._reconnect_attempts = 0
        self._last_error: Exception | None = None

    @property
    def config(self) -> ConfigT:
        """Get adapter configuration."""
        return self._config

    @property
    def state(self) -> AdapterState:
        """Get current adapter state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Check if adapter is connected."""
        return self._state == AdapterState.CONNECTED

    @property
    def source_name(self) -> str:
        """Get adapter source name."""
        return self.adapter_type

    @property
    def metrics(self) -> StreamMetrics:
        """Get current metrics."""
        return self._metrics.get_metrics()

    @property
    def last_error(self) -> Exception | None:
        """Get last error if any."""
        return self._last_error

    # -------------------------------------------------------------------------
    # Connection Lifecycle
    # -------------------------------------------------------------------------

    async def connect(self) -> None:
        """Connect to the streaming platform."""
        if self._state == AdapterState.CONNECTED:
            return

        self._state = AdapterState.CONNECTING
        try:
            await asyncio.wait_for(
                self._do_connect(),
                timeout=getattr(self._config, "connect_timeout_ms", 30000) / 1000,
            )
            self._state = AdapterState.CONNECTED
            self._reconnect_attempts = 0
            self._last_error = None
            logger.info(f"{self.adapter_type} adapter connected")
        except asyncio.TimeoutError:
            self._state = AdapterState.ERROR
            self._last_error = TimeoutError("Connection timeout")
            raise self._last_error
        except Exception as e:
            self._state = AdapterState.ERROR
            self._last_error = ConnectionError(f"Connection failed: {e}", cause=e)
            raise self._last_error

    async def disconnect(self) -> None:
        """Disconnect from the streaming platform."""
        if self._state == AdapterState.DISCONNECTED:
            return

        self._state = AdapterState.DISCONNECTING
        try:
            await self._do_disconnect()
        except Exception as e:
            logger.warning(f"Error during disconnect: {e}")
        finally:
            self._state = AdapterState.DISCONNECTED
            logger.info(f"{self.adapter_type} adapter disconnected")

    async def reconnect(self) -> None:
        """Reconnect with exponential backoff."""
        max_attempts = getattr(self._config, "max_reconnect_attempts", 5)
        base_delay = getattr(self._config, "reconnect_delay_ms", 1000) / 1000

        self._state = AdapterState.RECONNECTING

        while self._reconnect_attempts < max_attempts:
            self._reconnect_attempts += 1
            delay = base_delay * (2 ** (self._reconnect_attempts - 1))

            logger.info(
                f"Reconnect attempt {self._reconnect_attempts}/{max_attempts} "
                f"in {delay:.1f}s"
            )

            await asyncio.sleep(delay)

            try:
                await self._do_disconnect()
                await self._do_connect()
                self._state = AdapterState.CONNECTED
                self._reconnect_attempts = 0
                logger.info("Reconnection successful")
                return
            except Exception as e:
                logger.warning(f"Reconnect attempt failed: {e}")
                self._last_error = e

        self._state = AdapterState.ERROR
        raise ConnectionError(f"Max reconnection attempts ({max_attempts}) exceeded")

    # -------------------------------------------------------------------------
    # Message Consumption (IStreamSource)
    # -------------------------------------------------------------------------

    async def consume(self) -> AsyncIterator[StreamMessage[T]]:
        """Consume messages from the stream."""
        self._ensure_connected()

        async for message in self._do_consume():
            start_time = time.perf_counter()
            self._metrics.record_consumed(1, len(str(message.value).encode()))
            yield message
            elapsed = (time.perf_counter() - start_time) * 1000
            self._metrics.record_latency(elapsed)

    async def consume_batch(
        self,
        max_messages: int = 100,
        timeout_ms: int = 1000,
    ) -> MessageBatch[T]:
        """Consume a batch of messages."""
        self._ensure_connected()

        messages: list[StreamMessage[T]] = []
        start_time = time.monotonic()
        timeout_sec = timeout_ms / 1000

        async for message in self._do_consume():
            messages.append(message)
            self._metrics.record_consumed(1, len(str(message.value).encode()))

            if len(messages) >= max_messages:
                break
            if time.monotonic() - start_time > timeout_sec:
                break

        return MessageBatch(
            messages=messages,
            start_offset=messages[0].offset if messages else None,
            end_offset=messages[-1].offset if messages else None,
        )

    async def commit(self, message: StreamMessage[T]) -> None:
        """Commit message offset."""
        self._ensure_connected()
        await self._do_commit(message)

    async def commit_batch(self, messages: list[StreamMessage[T]]) -> None:
        """Commit offsets for multiple messages."""
        if not messages:
            return
        self._ensure_connected()
        await self._do_commit_batch(messages)

    async def seek(self, partition: int, offset: int) -> None:
        """Seek to specific offset."""
        self._ensure_connected()
        await self._do_seek(partition, offset)

    # -------------------------------------------------------------------------
    # Message Production (IStreamSink)
    # -------------------------------------------------------------------------

    async def produce(self, message: StreamMessage[T]) -> None:
        """Produce a message."""
        self._ensure_connected()
        try:
            await self._do_produce(message)
            self._metrics.record_produced(1, len(str(message.value).encode()))
        except Exception as e:
            self._metrics.record_failed(1)
            raise StreamingError(f"Produce failed: {e}", cause=e)

    async def produce_batch(self, messages: list[StreamMessage[T]]) -> None:
        """Produce a batch of messages."""
        self._ensure_connected()
        try:
            await self._do_produce_batch(messages)
            total_bytes = sum(len(str(m.value).encode()) for m in messages)
            self._metrics.record_produced(len(messages), total_bytes)
        except Exception as e:
            self._metrics.record_failed(len(messages))
            raise StreamingError(f"Batch produce failed: {e}", cause=e)

    async def flush(self) -> None:
        """Flush any buffered messages."""
        self._ensure_connected()
        await self._do_flush()

    # -------------------------------------------------------------------------
    # Abstract Methods (to be implemented by subclasses)
    # -------------------------------------------------------------------------

    @abstractmethod
    async def _do_connect(self) -> None:
        """Platform-specific connection logic."""
        ...

    @abstractmethod
    async def _do_disconnect(self) -> None:
        """Platform-specific disconnection logic."""
        ...

    @abstractmethod
    async def _do_consume(self) -> AsyncIterator[StreamMessage[T]]:
        """Platform-specific consumption logic."""
        ...

    @abstractmethod
    async def _do_produce(self, message: StreamMessage[T]) -> None:
        """Platform-specific production logic."""
        ...

    async def _do_produce_batch(self, messages: list[StreamMessage[T]]) -> None:
        """Platform-specific batch production logic.

        Default implementation produces messages one by one.
        Override for optimized batch operations.
        """
        for message in messages:
            await self._do_produce(message)

    async def _do_flush(self) -> None:
        """Platform-specific flush logic."""
        pass

    async def _do_commit(self, message: StreamMessage[T]) -> None:
        """Platform-specific commit logic."""
        pass

    async def _do_commit_batch(self, messages: list[StreamMessage[T]]) -> None:
        """Platform-specific batch commit logic.

        Default implementation commits the last message.
        """
        if messages:
            await self._do_commit(messages[-1])

    async def _do_seek(self, partition: int, offset: int) -> None:
        """Platform-specific seek logic."""
        pass

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _ensure_connected(self) -> None:
        """Ensure adapter is connected."""
        if not self.is_connected:
            raise ConnectionError(f"Adapter not connected (state: {self._state})")

    # -------------------------------------------------------------------------
    # Context Manager
    # -------------------------------------------------------------------------

    async def __aenter__(self) -> "BaseStreamAdapter[T, ConfigT]":
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.disconnect()
