"""Base classes and protocols for streaming storage.

This module defines the abstract interfaces and protocols that all streaming
store implementations must follow. Streaming stores enable handling of
validation results that exceed available memory.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Generic,
    Iterator,
    Protocol,
    TypeVar,
    runtime_checkable,
)

if TYPE_CHECKING:
    from truthound.stores.results import ValidationResult, ValidatorResult


# =============================================================================
# Enums
# =============================================================================


class StreamingFormat(str, Enum):
    """Supported streaming formats."""

    JSONL = "jsonl"  # JSON Lines - one JSON object per line
    NDJSON = "ndjson"  # Newline Delimited JSON (same as JSONL)
    CSV = "csv"  # CSV with header
    PARQUET = "parquet"  # Columnar format for analytics


class CompressionType(str, Enum):
    """Supported compression types for streaming."""

    NONE = "none"
    GZIP = "gzip"
    ZSTD = "zstd"
    LZ4 = "lz4"
    SNAPPY = "snappy"


class StreamStatus(str, Enum):
    """Status of a streaming operation."""

    PENDING = "pending"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class StreamingConfig:
    """Configuration for streaming storage operations.

    Attributes:
        format: Output format (jsonl, csv, parquet).
        compression: Compression algorithm to use.
        chunk_size: Number of records per chunk/file.
        buffer_size: In-memory buffer size before flush.
        max_memory_mb: Maximum memory usage in MB.
        flush_interval_seconds: Auto-flush interval.
        enable_checkpoints: Enable periodic checkpoints for recovery.
        checkpoint_interval: Records between checkpoints.
        enable_metrics: Collect streaming metrics.
        max_retries: Maximum retry attempts on failure.
        retry_delay_seconds: Base delay between retries.
    """

    format: StreamingFormat = StreamingFormat.JSONL
    compression: CompressionType = CompressionType.NONE
    chunk_size: int = 10000
    buffer_size: int = 1000
    max_memory_mb: int = 512
    flush_interval_seconds: float = 30.0
    enable_checkpoints: bool = True
    checkpoint_interval: int = 10000
    enable_metrics: bool = True
    max_retries: int = 3
    retry_delay_seconds: float = 1.0

    def validate(self) -> None:
        """Validate configuration values."""
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.buffer_size <= 0:
            raise ValueError("buffer_size must be positive")
        if self.max_memory_mb <= 0:
            raise ValueError("max_memory_mb must be positive")
        if self.flush_interval_seconds < 0:
            raise ValueError("flush_interval_seconds must be non-negative")
        if self.checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be positive")


# =============================================================================
# Metrics and Monitoring
# =============================================================================


@dataclass
class StreamingMetrics:
    """Metrics collected during streaming operations.

    Attributes:
        records_written: Total records written.
        records_read: Total records read.
        bytes_written: Total bytes written (after compression).
        bytes_read: Total bytes read.
        chunks_written: Number of chunks/files written.
        chunks_read: Number of chunks/files read.
        flush_count: Number of buffer flushes.
        retry_count: Number of retry attempts.
        errors: List of errors encountered.
        start_time: When streaming started.
        end_time: When streaming ended.
        peak_memory_mb: Peak memory usage in MB.
        average_throughput: Records per second.
    """

    records_written: int = 0
    records_read: int = 0
    bytes_written: int = 0
    bytes_read: int = 0
    chunks_written: int = 0
    chunks_read: int = 0
    flush_count: int = 0
    retry_count: int = 0
    errors: list[str] = field(default_factory=list)
    start_time: datetime | None = None
    end_time: datetime | None = None
    peak_memory_mb: float = 0.0
    average_throughput: float = 0.0

    def record_write(self, count: int = 1, bytes_count: int = 0) -> None:
        """Record a write operation."""
        self.records_written += count
        self.bytes_written += bytes_count

    def record_read(self, count: int = 1, bytes_count: int = 0) -> None:
        """Record a read operation."""
        self.records_read += count
        self.bytes_read += bytes_count

    def record_chunk(self, is_write: bool = True) -> None:
        """Record a chunk operation."""
        if is_write:
            self.chunks_written += 1
        else:
            self.chunks_read += 1

    def record_error(self, error: str) -> None:
        """Record an error."""
        self.errors.append(error)

    def start(self) -> None:
        """Mark streaming start."""
        self.start_time = datetime.now()

    def finish(self) -> None:
        """Mark streaming end and calculate throughput."""
        self.end_time = datetime.now()
        if self.start_time and self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()
            if duration > 0:
                total_records = self.records_written + self.records_read
                self.average_throughput = total_records / duration

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "records_written": self.records_written,
            "records_read": self.records_read,
            "bytes_written": self.bytes_written,
            "bytes_read": self.bytes_read,
            "chunks_written": self.chunks_written,
            "chunks_read": self.chunks_read,
            "flush_count": self.flush_count,
            "retry_count": self.retry_count,
            "errors": self.errors,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "peak_memory_mb": self.peak_memory_mb,
            "average_throughput": self.average_throughput,
        }


# =============================================================================
# Chunk Management
# =============================================================================


@dataclass
class ChunkInfo:
    """Information about a stored chunk.

    Attributes:
        chunk_id: Unique identifier for the chunk.
        chunk_index: Sequential index of the chunk.
        record_count: Number of records in the chunk.
        byte_size: Size of the chunk in bytes.
        start_offset: Starting record offset.
        end_offset: Ending record offset.
        checksum: Optional checksum for integrity.
        created_at: When the chunk was created.
        path: Storage path/key for the chunk.
    """

    chunk_id: str
    chunk_index: int
    record_count: int
    byte_size: int
    start_offset: int
    end_offset: int
    checksum: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    path: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
            "record_count": self.record_count,
            "byte_size": self.byte_size,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "checksum": self.checksum,
            "created_at": self.created_at.isoformat(),
            "path": self.path,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChunkInfo":
        """Create from dictionary."""
        return cls(
            chunk_id=data["chunk_id"],
            chunk_index=data["chunk_index"],
            record_count=data["record_count"],
            byte_size=data["byte_size"],
            start_offset=data["start_offset"],
            end_offset=data["end_offset"],
            checksum=data.get("checksum"),
            created_at=datetime.fromisoformat(data["created_at"]),
            path=data.get("path", ""),
        )


@dataclass
class StreamSession:
    """Session information for a streaming operation.

    Attributes:
        session_id: Unique identifier for the session.
        run_id: Associated validation run ID.
        data_asset: Name of the data asset being validated.
        status: Current status of the stream.
        config: Streaming configuration.
        metrics: Collected metrics.
        chunks: List of written chunks.
        metadata: Additional session metadata.
        started_at: When the session started.
        updated_at: Last update time.
        checkpoint_offset: Last checkpointed offset.
    """

    session_id: str
    run_id: str
    data_asset: str
    status: StreamStatus = StreamStatus.PENDING
    config: StreamingConfig = field(default_factory=StreamingConfig)
    metrics: StreamingMetrics = field(default_factory=StreamingMetrics)
    chunks: list[ChunkInfo] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    started_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    checkpoint_offset: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "run_id": self.run_id,
            "data_asset": self.data_asset,
            "status": self.status.value,
            "metrics": self.metrics.to_dict(),
            "chunks": [c.to_dict() for c in self.chunks],
            "metadata": self.metadata,
            "started_at": self.started_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "checkpoint_offset": self.checkpoint_offset,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StreamSession":
        """Create from dictionary."""
        return cls(
            session_id=data["session_id"],
            run_id=data["run_id"],
            data_asset=data["data_asset"],
            status=StreamStatus(data.get("status", "pending")),
            metrics=StreamingMetrics(**data.get("metrics", {})),
            chunks=[ChunkInfo.from_dict(c) for c in data.get("chunks", [])],
            metadata=data.get("metadata", {}),
            started_at=datetime.fromisoformat(data["started_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            checkpoint_offset=data.get("checkpoint_offset", 0),
        )


# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class StreamingWriter(Protocol):
    """Protocol for streaming writers."""

    def write(self, record: dict[str, Any]) -> None:
        """Write a single record."""
        ...

    def write_batch(self, records: list[dict[str, Any]]) -> None:
        """Write a batch of records."""
        ...

    def flush(self) -> None:
        """Flush buffered records to storage."""
        ...

    def close(self) -> None:
        """Close the writer and finalize."""
        ...


@runtime_checkable
class StreamingReader(Protocol):
    """Protocol for streaming readers."""

    def read(self) -> dict[str, Any] | None:
        """Read a single record."""
        ...

    def read_batch(self, size: int) -> list[dict[str, Any]]:
        """Read a batch of records."""
        ...

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over records."""
        ...

    def close(self) -> None:
        """Close the reader."""
        ...


@runtime_checkable
class AsyncStreamingWriter(Protocol):
    """Protocol for async streaming writers."""

    async def write(self, record: dict[str, Any]) -> None:
        """Write a single record asynchronously."""
        ...

    async def write_batch(self, records: list[dict[str, Any]]) -> None:
        """Write a batch of records asynchronously."""
        ...

    async def flush(self) -> None:
        """Flush buffered records to storage."""
        ...

    async def close(self) -> None:
        """Close the writer and finalize."""
        ...


@runtime_checkable
class AsyncStreamingReader(Protocol):
    """Protocol for async streaming readers."""

    async def read(self) -> dict[str, Any] | None:
        """Read a single record asynchronously."""
        ...

    async def read_batch(self, size: int) -> list[dict[str, Any]]:
        """Read a batch of records asynchronously."""
        ...

    def __aiter__(self) -> AsyncIterator[dict[str, Any]]:
        """Async iterate over records."""
        ...

    async def close(self) -> None:
        """Close the reader."""
        ...


# =============================================================================
# Abstract Base Classes
# =============================================================================


T = TypeVar("T")
ConfigT = TypeVar("ConfigT", bound=StreamingConfig)


class StreamingStore(ABC, Generic[T, ConfigT]):
    """Abstract base class for streaming stores.

    Streaming stores handle large-scale data that cannot fit in memory.
    They support incremental writing and reading through chunked operations.

    Type Parameters:
        T: The type of objects being stored.
        ConfigT: The configuration type for this store.
    """

    def __init__(self, config: ConfigT | None = None) -> None:
        """Initialize the streaming store.

        Args:
            config: Streaming configuration.
        """
        self._config = config or self._default_config()
        self._config.validate()
        self._initialized = False
        self._active_sessions: dict[str, StreamSession] = {}

    @classmethod
    @abstractmethod
    def _default_config(cls) -> ConfigT:
        """Create default configuration."""
        pass

    @property
    def config(self) -> ConfigT:
        """Get the store configuration."""
        return self._config

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def initialize(self) -> None:
        """Initialize the store."""
        if not self._initialized:
            self._do_initialize()
            self._initialized = True

    @abstractmethod
    def _do_initialize(self) -> None:
        """Perform actual initialization."""
        pass

    def close(self) -> None:
        """Close the store and all active sessions."""
        for session in list(self._active_sessions.values()):
            self._close_session(session)
        self._active_sessions.clear()

    def __enter__(self) -> "StreamingStore[T, ConfigT]":
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()

    # -------------------------------------------------------------------------
    # Session Management
    # -------------------------------------------------------------------------

    @abstractmethod
    def create_session(
        self,
        run_id: str,
        data_asset: str,
        metadata: dict[str, Any] | None = None,
    ) -> StreamSession:
        """Create a new streaming session.

        Args:
            run_id: Validation run identifier.
            data_asset: Name of the data asset.
            metadata: Optional session metadata.

        Returns:
            A new streaming session.
        """
        pass

    @abstractmethod
    def get_session(self, session_id: str) -> StreamSession | None:
        """Get an existing session.

        Args:
            session_id: Session identifier.

        Returns:
            The session if found, None otherwise.
        """
        pass

    @abstractmethod
    def resume_session(self, session_id: str) -> StreamSession:
        """Resume an interrupted session.

        Args:
            session_id: Session identifier.

        Returns:
            The resumed session.

        Raises:
            ValueError: If session cannot be resumed.
        """
        pass

    @abstractmethod
    def _close_session(self, session: StreamSession) -> None:
        """Close and finalize a session."""
        pass

    # -------------------------------------------------------------------------
    # Streaming Write Operations
    # -------------------------------------------------------------------------

    @abstractmethod
    def create_writer(self, session: StreamSession) -> StreamingWriter:
        """Create a writer for the session.

        Args:
            session: The streaming session.

        Returns:
            A streaming writer instance.
        """
        pass

    @abstractmethod
    async def create_async_writer(
        self, session: StreamSession
    ) -> AsyncStreamingWriter:
        """Create an async writer for the session.

        Args:
            session: The streaming session.

        Returns:
            An async streaming writer instance.
        """
        pass

    # -------------------------------------------------------------------------
    # Streaming Read Operations
    # -------------------------------------------------------------------------

    @abstractmethod
    def create_reader(self, run_id: str) -> StreamingReader:
        """Create a reader for a run's results.

        Args:
            run_id: The run ID to read.

        Returns:
            A streaming reader instance.
        """
        pass

    @abstractmethod
    async def create_async_reader(self, run_id: str) -> AsyncStreamingReader:
        """Create an async reader for a run's results.

        Args:
            run_id: The run ID to read.

        Returns:
            An async streaming reader instance.
        """
        pass

    @abstractmethod
    def iter_results(
        self,
        run_id: str,
        batch_size: int = 1000,
    ) -> Iterator[T]:
        """Iterate over results for a run.

        Args:
            run_id: The run ID to iterate.
            batch_size: Number of records per batch.

        Yields:
            Individual result records.
        """
        pass

    @abstractmethod
    async def aiter_results(
        self,
        run_id: str,
        batch_size: int = 1000,
    ) -> AsyncIterator[T]:
        """Async iterate over results for a run.

        Args:
            run_id: The run ID to iterate.
            batch_size: Number of records per batch.

        Yields:
            Individual result records.
        """
        pass

    # -------------------------------------------------------------------------
    # Chunk Management
    # -------------------------------------------------------------------------

    @abstractmethod
    def list_chunks(self, run_id: str) -> list[ChunkInfo]:
        """List all chunks for a run.

        Args:
            run_id: The run ID.

        Returns:
            List of chunk information.
        """
        pass

    @abstractmethod
    def get_chunk(self, chunk_info: ChunkInfo) -> list[T]:
        """Get records from a specific chunk.

        Args:
            chunk_info: The chunk to retrieve.

        Returns:
            Records from the chunk.
        """
        pass

    @abstractmethod
    def delete_chunks(self, run_id: str) -> int:
        """Delete all chunks for a run.

        Args:
            run_id: The run ID.

        Returns:
            Number of chunks deleted.
        """
        pass


class StreamingValidationStore(StreamingStore["ValidatorResult", ConfigT], Generic[ConfigT]):
    """Streaming store specialized for validation results.

    Provides additional methods specific to validation result streaming,
    including aggregation and statistics computation.
    """

    @abstractmethod
    def stream_write_result(
        self,
        session: StreamSession,
        result: "ValidatorResult",
    ) -> None:
        """Write a single validator result to the stream.

        Args:
            session: The streaming session.
            result: The validator result to write.
        """
        pass

    @abstractmethod
    def stream_write_batch(
        self,
        session: StreamSession,
        results: list["ValidatorResult"],
    ) -> None:
        """Write a batch of validator results to the stream.

        Args:
            session: The streaming session.
            results: The validator results to write.
        """
        pass

    @abstractmethod
    def finalize_result(
        self,
        session: StreamSession,
        additional_metadata: dict[str, Any] | None = None,
    ) -> "ValidationResult":
        """Finalize the streaming session and create a ValidationResult.

        This aggregates all streamed results into a single ValidationResult
        with computed statistics.

        Args:
            session: The streaming session.
            additional_metadata: Optional additional metadata.

        Returns:
            The complete ValidationResult.
        """
        pass

    @abstractmethod
    def get_streaming_stats(self, run_id: str) -> dict[str, Any]:
        """Get statistics about a streaming run.

        Args:
            run_id: The run ID.

        Returns:
            Statistics dictionary including record counts, errors, timing.
        """
        pass

    def iter_failed_results(
        self,
        run_id: str,
        batch_size: int = 1000,
    ) -> Iterator["ValidatorResult"]:
        """Iterate over failed results only.

        Args:
            run_id: The run ID.
            batch_size: Number of records per batch.

        Yields:
            Failed validator results.
        """
        for result in self.iter_results(run_id, batch_size):
            if not result.success:
                yield result

    def iter_results_by_column(
        self,
        run_id: str,
        column: str,
        batch_size: int = 1000,
    ) -> Iterator["ValidatorResult"]:
        """Iterate over results for a specific column.

        Args:
            run_id: The run ID.
            column: Column name to filter by.
            batch_size: Number of records per batch.

        Yields:
            Validator results for the specified column.
        """
        for result in self.iter_results(run_id, batch_size):
            if result.column == column:
                yield result

    def iter_results_by_severity(
        self,
        run_id: str,
        severity: str,
        batch_size: int = 1000,
    ) -> Iterator["ValidatorResult"]:
        """Iterate over results with a specific severity.

        Args:
            run_id: The run ID.
            severity: Severity level to filter by.
            batch_size: Number of records per batch.

        Yields:
            Validator results with the specified severity.
        """
        for result in self.iter_results(run_id, batch_size):
            if result.severity == severity:
                yield result
