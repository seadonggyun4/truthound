"""Streaming storage module for large-scale validation results.

This module provides streaming storage capabilities for handling validation
results that are too large to fit in memory. It supports:

- Incremental writing of validator results as they complete
- Memory-efficient iteration over stored results
- JSONL (JSON Lines) format for efficient streaming
- Chunked uploads for cloud storage (S3, GCS)
- Cursor-based database iteration
- Optional compression (gzip, zstd, lz4)
- Checkpointing for fault tolerance

Example:
    >>> from truthound.stores.streaming import StreamingFileSystemStore
    >>>
    >>> # Create streaming store
    >>> store = StreamingFileSystemStore(base_path=".truthound/streaming")
    >>>
    >>> # Create a streaming session
    >>> session = store.create_session("run_001", "large_dataset.csv")
    >>>
    >>> # Write results incrementally
    >>> with store.create_writer(session) as writer:
    ...     for result in validation_results:
    ...         writer.write_result(result)
    >>>
    >>> # Finalize and get complete result
    >>> final_result = store.finalize_result(session)
    >>>
    >>> # Read results back efficiently
    >>> for result in store.iter_results("run_001"):
    ...     process(result)

Streaming Store Types:
    - StreamingFileSystemStore: JSONL-based filesystem streaming
    - StreamingS3Store: Multipart upload streaming for S3
    - StreamingDatabaseStore: Cursor-based streaming for databases

Async Support:
    All streaming stores support async operations:

    >>> async with await store.create_async_writer(session) as writer:
    ...     await writer.write_result(result)
    >>>
    >>> async for result in store.aiter_results("run_001"):
    ...     await process(result)
"""

from truthound.stores.streaming.base import (
    ChunkInfo,
    CompressionType,
    StreamingConfig,
    StreamingFormat,
    StreamingMetrics,
    StreamingReader,
    StreamingStore,
    StreamingValidationStore,
    StreamingWriter,
    StreamSession,
    StreamStatus,
)
from truthound.stores.streaming.writer import (
    AsyncStreamWriter,
    BaseStreamWriter,
    BufferedStreamWriter,
    StreamingResultWriter,
    StreamWriteError,
    create_stream_writer,
)
from truthound.stores.streaming.reader import (
    AsyncStreamReader,
    BaseStreamReader,
    ChunkedResultReader,
    StreamingResultReader,
    StreamReadError,
    create_stream_reader,
)
from truthound.stores.streaming.filesystem import (
    StreamingFileSystemConfig,
    StreamingFileSystemStore,
)

# Lazy imports for optional backends
def __getattr__(name: str):
    """Lazy import optional backends."""
    if name == "StreamingS3Store":
        from truthound.stores.streaming.s3 import StreamingS3Store
        return StreamingS3Store
    elif name == "StreamingS3Config":
        from truthound.stores.streaming.s3 import StreamingS3Config
        return StreamingS3Config
    elif name == "StreamingDatabaseStore":
        from truthound.stores.streaming.database import StreamingDatabaseStore
        return StreamingDatabaseStore
    elif name == "StreamingDatabaseConfig":
        from truthound.stores.streaming.database import StreamingDatabaseConfig
        return StreamingDatabaseConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Base types
    "ChunkInfo",
    "CompressionType",
    "StreamingConfig",
    "StreamingFormat",
    "StreamingMetrics",
    "StreamingReader",
    "StreamingStore",
    "StreamingValidationStore",
    "StreamingWriter",
    "StreamSession",
    "StreamStatus",
    # Writers
    "AsyncStreamWriter",
    "BaseStreamWriter",
    "BufferedStreamWriter",
    "StreamingResultWriter",
    "StreamWriteError",
    "create_stream_writer",
    # Readers
    "AsyncStreamReader",
    "BaseStreamReader",
    "ChunkedResultReader",
    "StreamingResultReader",
    "StreamReadError",
    "create_stream_reader",
    # Filesystem store
    "StreamingFileSystemConfig",
    "StreamingFileSystemStore",
    # S3 store (lazy loaded)
    "StreamingS3Store",
    "StreamingS3Config",
    # Database store (lazy loaded)
    "StreamingDatabaseStore",
    "StreamingDatabaseConfig",
]
