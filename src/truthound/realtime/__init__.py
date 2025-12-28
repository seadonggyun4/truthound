"""Real-time validation module for Truthound.

This module provides real-time and streaming validation capabilities:
- Streaming data sources (Kafka, Kinesis, etc.)
- Incremental validation
- Micro-batch processing
- Window-based validation
- Protocol-based adapter abstraction
- State management with Redis support
- Exactly-once processing semantics

Example:
    >>> from truthound import realtime
    >>>
    >>> # Create streaming validator
    >>> validator = realtime.StreamingValidator(
    ...     validators=["null", "range"],
    ...     window_size=1000,
    ... )
    >>>
    >>> # Process streaming data
    >>> async for batch in data_stream:
    ...     result = validator.validate_batch(batch)
    ...     if result.has_issues:
    ...         handle_issues(result)

Example with new adapters:
    >>> from truthound.realtime.adapters import KafkaAdapter, KafkaAdapterConfig
    >>> from truthound.realtime.factory import StreamAdapterFactory
    >>>
    >>> # Create adapter via factory
    >>> adapter = StreamAdapterFactory.create("kafka", {
    ...     "bootstrap_servers": "localhost:9092",
    ...     "topic": "events",
    ... })
    >>>
    >>> async with adapter:
    ...     async for message in adapter.consume():
    ...         process(message)
    ...         await adapter.commit(message)
"""

from truthound.realtime.base import (
    # Enums
    StreamingMode,
    WindowType,
    TriggerType,
    # Configuration
    StreamingConfig,
    WindowConfig,
    # Base classes
    StreamingSource,
    StreamingValidator,
    BatchResult,
    WindowResult,
    # Exceptions
    StreamingError,
    ConnectionError,
    TimeoutError,
)

from truthound.realtime.streaming import (
    MockStreamingSource,
    KafkaSource,
    KinesisSource,
    PubSubSource,
)

from truthound.realtime.incremental import (
    IncrementalValidator,
    StateStore,
    MemoryStateStore,
    CheckpointManager,
)

# New protocol-based streaming components
from truthound.realtime.protocols import (
    # Protocols
    IStreamSource,
    IStreamSink,
    IStreamProcessor,
    IStateStore as IStateStoreProtocol,
    IMetricsCollector,
    # Data classes
    StreamMessage,
    MessageBatch,
    MessageHeader,
    StreamMetrics,
    # Enums
    DeserializationFormat,
    OffsetReset,
    AckMode,
)

from truthound.realtime.factory import StreamAdapterFactory

__all__ = [
    # Enums
    "StreamingMode",
    "WindowType",
    "TriggerType",
    # Configuration
    "StreamingConfig",
    "WindowConfig",
    # Base classes
    "StreamingSource",
    "StreamingValidator",
    "BatchResult",
    "WindowResult",
    # Exceptions
    "StreamingError",
    "ConnectionError",
    "TimeoutError",
    # Streaming sources (legacy)
    "MockStreamingSource",
    "KafkaSource",
    "KinesisSource",
    "PubSubSource",
    # Incremental validation
    "IncrementalValidator",
    "StateStore",
    "MemoryStateStore",
    "CheckpointManager",
    # New protocols
    "IStreamSource",
    "IStreamSink",
    "IStreamProcessor",
    "IStateStoreProtocol",
    "IMetricsCollector",
    # New data classes
    "StreamMessage",
    "MessageBatch",
    "MessageHeader",
    "StreamMetrics",
    # New enums
    "DeserializationFormat",
    "OffsetReset",
    "AckMode",
    # Factory
    "StreamAdapterFactory",
]
