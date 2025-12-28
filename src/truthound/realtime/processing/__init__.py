"""Stream processing components.

Provides advanced stream processing capabilities:
- Window processing (Tumbling, Sliding, Session)
- State management (Memory, Redis, RocksDB)
- Exactly-once processing semantics
"""

from truthound.realtime.processing.windows import (
    # Window types
    WindowType,
    WindowConfig,
    WindowResult,
    # Processors
    WindowProcessor,
    TumblingWindowProcessor,
    SlidingWindowProcessor,
    SessionWindowProcessor,
    # Aggregators
    WindowAggregator,
    CountAggregator,
    SumAggregator,
    AvgAggregator,
    MinAggregator,
    MaxAggregator,
)

from truthound.realtime.processing.state import (
    # Protocols
    IStateBackend,
    # Implementations
    MemoryStateBackend,
    RedisStateBackend,
    StateManager,
)

from truthound.realtime.processing.exactly_once import (
    ExactlyOnceProcessor,
    TransactionState,
    ProcessingGuarantee,
)

__all__ = [
    # Windows
    "WindowType",
    "WindowConfig",
    "WindowResult",
    "WindowProcessor",
    "TumblingWindowProcessor",
    "SlidingWindowProcessor",
    "SessionWindowProcessor",
    "WindowAggregator",
    "CountAggregator",
    "SumAggregator",
    "AvgAggregator",
    "MinAggregator",
    "MaxAggregator",
    # State
    "IStateBackend",
    "MemoryStateBackend",
    "RedisStateBackend",
    "StateManager",
    # Exactly-once
    "ExactlyOnceProcessor",
    "TransactionState",
    "ProcessingGuarantee",
]
