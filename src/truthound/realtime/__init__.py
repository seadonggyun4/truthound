"""Real-time validation module for Truthound.

This module provides real-time and streaming validation capabilities:
- Streaming data sources (Kafka, Kinesis, etc.)
- Incremental validation
- Micro-batch processing
- Window-based validation

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
    # Streaming sources
    "MockStreamingSource",
    "KafkaSource",
    "KinesisSource",
    "PubSubSource",
    # Incremental validation
    "IncrementalValidator",
    "StateStore",
    "MemoryStateStore",
    "CheckpointManager",
]
