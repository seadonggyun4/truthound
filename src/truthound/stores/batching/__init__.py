"""Batch write optimization for storage operations.

This module provides batch writing capabilities to optimize throughput
and reduce I/O overhead when storing large numbers of validation results.

Features:
    - Configurable batch sizes and flush intervals
    - Memory-aware buffer management
    - Parallel batch processing
    - Compression support for batched writes
    - Automatic retry with exponential backoff

Example:
    >>> from truthound.stores.batching import (
    ...     BatchedStore,
    ...     BatchConfig,
    ...     BatchWriter,
    ... )
    >>>
    >>> config = BatchConfig(
    ...     batch_size=1000,
    ...     flush_interval_seconds=5.0,
    ...     max_buffer_memory_mb=100,
    ... )
    >>> store = BatchedStore(underlying_store, config)
    >>>
    >>> async with store:
    ...     for result in results:
    ...         await store.add(result)
    ...     await store.flush()
"""

from truthound.stores.batching.base import (
    BatchConfig,
    BatchMetrics,
    BatchState,
    BatchStatus,
)
from truthound.stores.batching.writer import (
    BatchWriter,
    AsyncBatchWriter,
)
from truthound.stores.batching.store import (
    BatchedStore,
)
from truthound.stores.batching.buffer import (
    BatchBuffer,
    MemoryAwareBuffer,
)

__all__ = [
    # Base
    "BatchConfig",
    "BatchMetrics",
    "BatchState",
    "BatchStatus",
    # Writer
    "BatchWriter",
    "AsyncBatchWriter",
    # Store
    "BatchedStore",
    # Buffer
    "BatchBuffer",
    "MemoryAwareBuffer",
]
