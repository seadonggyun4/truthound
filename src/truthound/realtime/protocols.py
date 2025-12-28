"""Core protocols for streaming data processing.

This module defines Protocol-based abstractions for streaming operations,
enabling flexible implementations for different streaming platforms.

Design Principles:
    - Protocol-First: All components are defined as Protocols for loose coupling
    - Generic Types: Full type safety with Generic TypeVars
    - Async-Native: Designed for async/await patterns
    - Composable: Components can be freely combined
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Generic,
    Iterator,
    Protocol,
    TypeVar,
    runtime_checkable,
)

if TYPE_CHECKING:
    import polars as pl


# =============================================================================
# Type Variables
# =============================================================================

T = TypeVar("T")  # Message value type
K = TypeVar("K")  # Key type
R = TypeVar("R")  # Result type
StateT = TypeVar("StateT")  # State type


# =============================================================================
# Enums
# =============================================================================


class DeserializationFormat(str, Enum):
    """Supported message deserialization formats."""

    JSON = "json"
    AVRO = "avro"
    PROTOBUF = "protobuf"
    MSGPACK = "msgpack"
    STRING = "string"
    BYTES = "bytes"


class AckMode(str, Enum):
    """Message acknowledgment modes."""

    AUTO = "auto"  # Automatic acknowledgment
    MANUAL = "manual"  # Manual acknowledgment required
    BATCH = "batch"  # Batch acknowledgment


class OffsetReset(str, Enum):
    """Offset reset policies."""

    EARLIEST = "earliest"
    LATEST = "latest"
    NONE = "none"


# =============================================================================
# Message Types
# =============================================================================


@dataclass(frozen=True)
class MessageHeader:
    """Message header with metadata."""

    key: str
    value: bytes


@dataclass(frozen=True)
class StreamMessage(Generic[T]):
    """Generic stream message with full metadata.

    Attributes:
        key: Message key (optional)
        value: Message payload
        partition: Source partition number
        offset: Message offset within partition
        timestamp: Message timestamp
        headers: Optional message headers
        topic: Source topic/stream name
        metadata: Additional metadata
    """

    key: str | None
    value: T
    partition: int
    offset: int
    timestamp: datetime
    headers: tuple[MessageHeader, ...] = field(default_factory=tuple)
    topic: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def age_ms(self) -> float:
        """Message age in milliseconds."""
        now = datetime.now(timezone.utc)
        ts = self.timestamp.replace(tzinfo=timezone.utc) if self.timestamp.tzinfo is None else self.timestamp
        return (now - ts).total_seconds() * 1000

    def with_value(self, new_value: T) -> "StreamMessage[T]":
        """Create a new message with updated value."""
        return StreamMessage(
            key=self.key,
            value=new_value,
            partition=self.partition,
            offset=self.offset,
            timestamp=self.timestamp,
            headers=self.headers,
            topic=self.topic,
            metadata=self.metadata,
        )


@dataclass
class MessageBatch(Generic[T]):
    """Batch of stream messages."""

    messages: list[StreamMessage[T]]
    partition: int | None = None
    start_offset: int | None = None
    end_offset: int | None = None

    @property
    def size(self) -> int:
        return len(self.messages)

    @property
    def is_empty(self) -> bool:
        return len(self.messages) == 0

    def to_dataframe(self) -> "pl.DataFrame":
        """Convert batch to Polars DataFrame."""
        import polars as pl

        if not self.messages:
            return pl.DataFrame()

        # Extract values - assume dict-like values
        records = []
        for msg in self.messages:
            if isinstance(msg.value, dict):
                record = msg.value.copy()
            else:
                record = {"value": msg.value}
            record["_offset"] = msg.offset
            record["_timestamp"] = msg.timestamp
            record["_partition"] = msg.partition
            records.append(record)

        return pl.DataFrame(records)


# =============================================================================
# Core Protocols
# =============================================================================


@runtime_checkable
class IStreamSource(Protocol[T]):
    """Protocol for stream message sources.

    Implementations should provide message consumption from various
    streaming platforms (Kafka, Kinesis, Pulsar, etc.).
    """

    async def connect(self) -> None:
        """Establish connection to the streaming source.

        Raises:
            ConnectionError: If connection fails
        """
        ...

    async def disconnect(self) -> None:
        """Close connection to the streaming source."""
        ...

    async def consume(self) -> AsyncIterator[StreamMessage[T]]:
        """Consume messages from the source.

        Yields:
            StreamMessage instances as they arrive
        """
        ...

    async def consume_batch(self, max_messages: int = 100, timeout_ms: int = 1000) -> MessageBatch[T]:
        """Consume a batch of messages.

        Args:
            max_messages: Maximum messages to consume
            timeout_ms: Timeout in milliseconds

        Returns:
            MessageBatch containing consumed messages
        """
        ...

    async def commit(self, message: StreamMessage[T]) -> None:
        """Commit message offset.

        Args:
            message: Message to commit offset for
        """
        ...

    async def commit_batch(self, messages: list[StreamMessage[T]]) -> None:
        """Commit offsets for a batch of messages.

        Args:
            messages: Messages to commit offsets for
        """
        ...

    async def seek(self, partition: int, offset: int) -> None:
        """Seek to a specific offset in a partition.

        Args:
            partition: Partition number
            offset: Offset to seek to
        """
        ...

    @property
    def is_connected(self) -> bool:
        """Check if connected to source."""
        ...

    @property
    def source_name(self) -> str:
        """Get source identifier."""
        ...


@runtime_checkable
class IStreamSink(Protocol[T]):
    """Protocol for stream message sinks.

    Implementations should provide message production to various
    streaming platforms.
    """

    async def connect(self) -> None:
        """Establish connection to the sink."""
        ...

    async def disconnect(self) -> None:
        """Close connection to the sink."""
        ...

    async def produce(self, message: StreamMessage[T]) -> None:
        """Produce a single message.

        Args:
            message: Message to produce
        """
        ...

    async def produce_batch(self, messages: list[StreamMessage[T]]) -> None:
        """Produce a batch of messages.

        Args:
            messages: Messages to produce
        """
        ...

    async def flush(self) -> None:
        """Flush any buffered messages."""
        ...

    @property
    def is_connected(self) -> bool:
        """Check if connected to sink."""
        ...


@runtime_checkable
class IStreamProcessor(Protocol[T, R]):
    """Protocol for stream message processors.

    Processes stream messages and produces results.
    Supports stateful processing, windowing, and exactly-once semantics.
    """

    async def process(self, message: StreamMessage[T]) -> StreamMessage[R] | None:
        """Process a single message.

        Args:
            message: Input message

        Returns:
            Processed message or None if filtered
        """
        ...

    async def process_batch(self, batch: MessageBatch[T]) -> MessageBatch[R]:
        """Process a batch of messages.

        Args:
            batch: Input batch

        Returns:
            Batch of processed messages
        """
        ...

    async def checkpoint(self) -> None:
        """Create a checkpoint of processor state."""
        ...

    async def restore(self, checkpoint_id: str) -> None:
        """Restore processor state from checkpoint.

        Args:
            checkpoint_id: Checkpoint to restore from
        """
        ...


# =============================================================================
# State Management Protocols
# =============================================================================


@runtime_checkable
class IStateStore(Protocol[StateT]):
    """Protocol for distributed state storage.

    Provides state management for stateful stream processing.
    """

    async def get(self, key: str) -> StateT | None:
        """Get state value by key.

        Args:
            key: State key

        Returns:
            State value or None if not found
        """
        ...

    async def put(self, key: str, value: StateT, ttl: int | None = None) -> None:
        """Store state value.

        Args:
            key: State key
            value: State value
            ttl: Optional TTL in seconds
        """
        ...

    async def delete(self, key: str) -> bool:
        """Delete state value.

        Args:
            key: State key

        Returns:
            True if deleted, False if not found
        """
        ...

    async def get_all(self, prefix: str = "") -> dict[str, StateT]:
        """Get all state values with optional prefix filter.

        Args:
            prefix: Key prefix filter

        Returns:
            Dictionary of matching key-value pairs
        """
        ...

    async def clear(self) -> None:
        """Clear all state."""
        ...


# =============================================================================
# Metrics and Observability Protocols
# =============================================================================


@dataclass
class StreamMetrics:
    """Streaming metrics snapshot."""

    messages_consumed: int = 0
    messages_produced: int = 0
    messages_failed: int = 0
    bytes_consumed: int = 0
    bytes_produced: int = 0
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    throughput_per_second: float = 0.0
    last_message_timestamp: datetime | None = None
    consumer_lag: int = 0


@runtime_checkable
class IMetricsCollector(Protocol):
    """Protocol for streaming metrics collection."""

    def record_consumed(self, count: int = 1, bytes_size: int = 0) -> None:
        """Record consumed message metrics."""
        ...

    def record_produced(self, count: int = 1, bytes_size: int = 0) -> None:
        """Record produced message metrics."""
        ...

    def record_failed(self, count: int = 1) -> None:
        """Record failed message metrics."""
        ...

    def record_latency(self, latency_ms: float) -> None:
        """Record processing latency."""
        ...

    def get_metrics(self) -> StreamMetrics:
        """Get current metrics snapshot."""
        ...

    def reset(self) -> None:
        """Reset all metrics."""
        ...


# =============================================================================
# Serialization Protocols
# =============================================================================


@runtime_checkable
class ISerializer(Protocol[T]):
    """Protocol for message serialization."""

    def serialize(self, value: T) -> bytes:
        """Serialize value to bytes.

        Args:
            value: Value to serialize

        Returns:
            Serialized bytes
        """
        ...

    def deserialize(self, data: bytes) -> T:
        """Deserialize bytes to value.

        Args:
            data: Bytes to deserialize

        Returns:
            Deserialized value
        """
        ...


# =============================================================================
# Error Handling
# =============================================================================


class StreamingError(Exception):
    """Base exception for streaming errors."""

    def __init__(self, message: str, cause: Exception | None = None):
        super().__init__(message)
        self.cause = cause


class ConnectionError(StreamingError):
    """Connection to streaming platform failed."""

    pass


class SerializationError(StreamingError):
    """Message serialization/deserialization failed."""

    pass


class CommitError(StreamingError):
    """Offset commit failed."""

    pass


class TimeoutError(StreamingError):
    """Operation timed out."""

    pass


# =============================================================================
# Configuration Base
# =============================================================================


@dataclass
class StreamSourceConfig:
    """Base configuration for stream sources."""

    # Consumer settings
    consumer_group: str = "truthound-consumer"
    auto_offset_reset: OffsetReset = OffsetReset.LATEST
    ack_mode: AckMode = AckMode.AUTO
    max_poll_records: int = 500
    poll_timeout_ms: int = 1000

    # Deserialization
    key_deserializer: DeserializationFormat = DeserializationFormat.STRING
    value_deserializer: DeserializationFormat = DeserializationFormat.JSON

    # Error handling
    max_retries: int = 3
    retry_delay_ms: int = 1000
    error_handler: Callable[[Exception, StreamMessage], None] | None = None

    # Metrics
    enable_metrics: bool = True
    metrics_interval_ms: int = 10000


@dataclass
class StreamSinkConfig:
    """Base configuration for stream sinks."""

    # Producer settings
    batch_size: int = 100
    linger_ms: int = 5
    buffer_memory: int = 33554432  # 32MB

    # Serialization
    key_serializer: DeserializationFormat = DeserializationFormat.STRING
    value_serializer: DeserializationFormat = DeserializationFormat.JSON

    # Delivery
    acks: str = "all"  # "0", "1", "all"
    retries: int = 3
    retry_delay_ms: int = 100

    # Metrics
    enable_metrics: bool = True
