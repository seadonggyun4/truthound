"""Base classes for real-time validation.

Provides abstractions for streaming data sources and
real-time validation processing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Generic,
    Iterator,
    TypeVar,
)
import threading
from collections import deque

import polars as pl

if TYPE_CHECKING:
    from truthound.validators.base import ValidationIssue


# =============================================================================
# Enums
# =============================================================================


class StreamingMode(str, Enum):
    """Mode of streaming processing."""

    CONTINUOUS = "continuous"  # Process records as they arrive
    MICRO_BATCH = "micro_batch"  # Process in small batches
    WINDOWED = "windowed"  # Process by time/count windows


class WindowType(str, Enum):
    """Type of window for windowed processing."""

    TUMBLING = "tumbling"  # Fixed-size, non-overlapping
    SLIDING = "sliding"  # Fixed-size, overlapping
    SESSION = "session"  # Gap-based
    GLOBAL = "global"  # Single global window


class TriggerType(str, Enum):
    """Type of trigger for batch processing."""

    COUNT = "count"  # Trigger after N records
    TIME = "time"  # Trigger after N seconds
    SIZE = "size"  # Trigger after N bytes
    WATERMARK = "watermark"  # Trigger on watermark


# =============================================================================
# Exceptions
# =============================================================================


class StreamingError(Exception):
    """Base exception for streaming errors."""

    pass


class ConnectionError(StreamingError):
    """Raised when connection to streaming source fails."""

    pass


class TimeoutError(StreamingError):
    """Raised when streaming operation times out."""

    pass


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class StreamingConfig:
    """Configuration for streaming operations.

    Attributes:
        mode: Processing mode
        batch_size: Size of micro-batches
        batch_timeout_ms: Timeout for batch collection
        max_records_per_second: Rate limiting
        checkpoint_interval_ms: Checkpoint interval
        error_handling: Error handling strategy
    """

    mode: StreamingMode = StreamingMode.MICRO_BATCH
    batch_size: int = 1000
    batch_timeout_ms: int = 1000
    max_records_per_second: int | None = None
    checkpoint_interval_ms: int = 10000
    error_handling: str = "skip"  # skip, fail, retry
    max_retries: int = 3
    retry_delay_ms: int = 1000
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WindowConfig:
    """Configuration for window-based processing.

    Attributes:
        window_type: Type of window
        window_size: Size of window (seconds for time, count for count)
        slide_interval: Slide interval for sliding windows
        allowed_lateness: Allowed lateness for late data
    """

    window_type: WindowType = WindowType.TUMBLING
    window_size: int = 60  # seconds or count
    slide_interval: int | None = None  # For sliding windows
    allowed_lateness: int = 0  # seconds
    watermark_delay: int = 0  # seconds


# =============================================================================
# Result Classes
# =============================================================================


@dataclass
class BatchResult:
    """Result of validating a single batch.

    Attributes:
        batch_id: Unique batch identifier
        record_count: Number of records in batch
        issue_count: Number of issues found
        issues: List of validation issues
        processing_time_ms: Time to process batch
        timestamp: When batch was processed
    """

    batch_id: str
    record_count: int = 0
    issue_count: int = 0
    issues: list["ValidationIssue"] = field(default_factory=list)
    processing_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_issues(self) -> bool:
        return self.issue_count > 0

    @property
    def issue_ratio(self) -> float:
        if self.record_count == 0:
            return 0.0
        return self.issue_count / self.record_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "record_count": self.record_count,
            "issue_count": self.issue_count,
            "issue_ratio": round(self.issue_ratio, 4),
            "processing_time_ms": round(self.processing_time_ms, 2),
            "timestamp": self.timestamp.isoformat(),
            "issues": [i.to_dict() for i in self.issues],
            "metadata": self.metadata,
        }


@dataclass
class WindowResult:
    """Result of validating a window of data.

    Attributes:
        window_id: Unique window identifier
        window_start: Window start time
        window_end: Window end time
        total_records: Total records in window
        batch_results: Results from individual batches
        aggregate_issues: Aggregated issue counts
    """

    window_id: str
    window_start: datetime
    window_end: datetime
    total_records: int = 0
    total_issues: int = 0
    batch_results: list[BatchResult] = field(default_factory=list)
    aggregate_issues: dict[str, int] = field(default_factory=dict)
    processing_time_ms: float = 0.0

    @property
    def has_issues(self) -> bool:
        return self.total_issues > 0

    @property
    def batch_count(self) -> int:
        return len(self.batch_results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
            "total_records": self.total_records,
            "total_issues": self.total_issues,
            "batch_count": self.batch_count,
            "aggregate_issues": self.aggregate_issues,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }


# =============================================================================
# Base Classes
# =============================================================================


ConfigT = TypeVar("ConfigT", bound=StreamingConfig)


class StreamingSource(ABC, Generic[ConfigT]):
    """Abstract base class for streaming data sources.

    Provides interface for connecting to and reading from
    streaming data sources like Kafka, Kinesis, etc.

    Example:
        class KafkaSource(StreamingSource):
            def connect(self) -> None:
                self._consumer = KafkaConsumer(self._topic, **self._kafka_config)

            def read_batch(self, max_records: int) -> pl.DataFrame:
                records = self._consumer.poll(max_records=max_records)
                return self._to_dataframe(records)
    """

    source_type: str = "base"

    def __init__(self, config: ConfigT | None = None, **kwargs: Any):
        """Initialize the streaming source.

        Args:
            config: Source configuration
            **kwargs: Additional parameters
        """
        self._config: ConfigT = config or self._default_config()  # type: ignore
        self._connected = False
        self._lock = threading.RLock()

        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)

    @abstractmethod
    def connect(self) -> None:
        """Connect to the streaming source.

        Raises:
            ConnectionError: If connection fails
        """
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the streaming source."""
        ...

    @abstractmethod
    def read_batch(self, max_records: int | None = None) -> pl.DataFrame:
        """Read a batch of records.

        Args:
            max_records: Maximum records to read

        Returns:
            DataFrame with batch data
        """
        ...

    def read_batches(
        self,
        max_batches: int | None = None,
    ) -> Iterator[pl.DataFrame]:
        """Iterate over batches.

        Args:
            max_batches: Maximum number of batches

        Yields:
            DataFrame for each batch
        """
        batch_count = 0
        while max_batches is None or batch_count < max_batches:
            batch = self.read_batch()
            if batch is None or len(batch) == 0:
                break
            yield batch
            batch_count += 1

    async def read_batches_async(
        self,
        max_batches: int | None = None,
    ) -> AsyncIterator[pl.DataFrame]:
        """Async iterator over batches.

        Args:
            max_batches: Maximum number of batches

        Yields:
            DataFrame for each batch
        """
        import asyncio

        batch_count = 0
        while max_batches is None or batch_count < max_batches:
            batch = await asyncio.get_event_loop().run_in_executor(
                None, self.read_batch
            )
            if batch is None or len(batch) == 0:
                break
            yield batch
            batch_count += 1

    def commit(self) -> None:
        """Commit current position (for sources that support it)."""
        pass

    def seek(self, position: Any) -> None:
        """Seek to a specific position.

        Args:
            position: Position to seek to (type depends on source)
        """
        pass

    @property
    def is_connected(self) -> bool:
        """Check if connected to source."""
        return self._connected

    @property
    def config(self) -> ConfigT:
        """Get source configuration."""
        return self._config

    def _default_config(self) -> StreamingConfig:
        """Return default configuration."""
        return StreamingConfig()

    def __enter__(self) -> "StreamingSource":
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.disconnect()


class StreamingValidator:
    """Validate streaming data in real-time.

    Provides methods to:
    - Validate individual batches
    - Accumulate window statistics
    - Handle errors gracefully
    - Track validation metrics

    Example:
        >>> validator = StreamingValidator(
        ...     validators=["null", "range"],
        ...     config=StreamingConfig(batch_size=1000),
        ... )
        >>>
        >>> with KafkaSource(topic="events") as source:
        ...     for batch in source.read_batches():
        ...         result = validator.validate_batch(batch)
        ...         if result.has_issues:
        ...             alert(result)
    """

    def __init__(
        self,
        validators: list[str] | None = None,
        config: StreamingConfig | None = None,
        on_issue: Callable[["ValidationIssue"], None] | None = None,
        on_batch_complete: Callable[[BatchResult], None] | None = None,
    ):
        """Initialize the streaming validator.

        Args:
            validators: List of validator names to use
            config: Streaming configuration
            on_issue: Callback for each issue found
            on_batch_complete: Callback after each batch
        """
        self._validators = validators or []
        self._config = config or StreamingConfig()
        self._on_issue = on_issue
        self._on_batch_complete = on_batch_complete

        self._batch_count = 0
        self._total_records = 0
        self._total_issues = 0
        self._recent_results: deque[BatchResult] = deque(maxlen=100)
        self._lock = threading.RLock()

    def validate_batch(
        self,
        batch: pl.DataFrame,
        batch_id: str | None = None,
    ) -> BatchResult:
        """Validate a single batch of data.

        Args:
            batch: DataFrame batch to validate
            batch_id: Optional batch identifier

        Returns:
            BatchResult with validation results
        """
        import time
        import uuid

        start = time.perf_counter()
        batch_id = batch_id or str(uuid.uuid4())[:8]

        # Run validation
        from truthound.api import check

        issues = []
        try:
            report = check(batch, validators=self._validators)
            issues = report.issues
        except Exception as e:
            if self._config.error_handling == "fail":
                raise
            # Log error but continue

        # Process issues
        for issue in issues:
            if self._on_issue:
                self._on_issue(issue)

        elapsed = (time.perf_counter() - start) * 1000

        result = BatchResult(
            batch_id=batch_id,
            record_count=len(batch),
            issue_count=len(issues),
            issues=issues,
            processing_time_ms=elapsed,
        )

        # Update stats
        with self._lock:
            self._batch_count += 1
            self._total_records += len(batch)
            self._total_issues += len(issues)
            self._recent_results.append(result)

        # Callback
        if self._on_batch_complete:
            self._on_batch_complete(result)

        return result

    def validate_stream(
        self,
        source: StreamingSource,
        max_batches: int | None = None,
    ) -> Iterator[BatchResult]:
        """Validate batches from a streaming source.

        Args:
            source: Streaming source to read from
            max_batches: Maximum batches to process

        Yields:
            BatchResult for each batch
        """
        for batch in source.read_batches(max_batches):
            yield self.validate_batch(batch)

    async def validate_stream_async(
        self,
        source: StreamingSource,
        max_batches: int | None = None,
    ) -> AsyncIterator[BatchResult]:
        """Async validation of streaming data.

        Args:
            source: Streaming source
            max_batches: Maximum batches

        Yields:
            BatchResult for each batch
        """
        import asyncio

        async for batch in source.read_batches_async(max_batches):
            result = await asyncio.get_event_loop().run_in_executor(
                None, self.validate_batch, batch
            )
            yield result

    def get_stats(self) -> dict[str, Any]:
        """Get validation statistics."""
        with self._lock:
            recent = list(self._recent_results)

        avg_time = 0.0
        avg_issues = 0.0
        if recent:
            avg_time = sum(r.processing_time_ms for r in recent) / len(recent)
            avg_issues = sum(r.issue_count for r in recent) / len(recent)

        return {
            "batch_count": self._batch_count,
            "total_records": self._total_records,
            "total_issues": self._total_issues,
            "issue_rate": self._total_issues / self._total_records if self._total_records > 0 else 0,
            "avg_processing_time_ms": round(avg_time, 2),
            "avg_issues_per_batch": round(avg_issues, 2),
            "recent_batches": len(recent),
        }

    def reset_stats(self) -> None:
        """Reset validation statistics."""
        with self._lock:
            self._batch_count = 0
            self._total_records = 0
            self._total_issues = 0
            self._recent_results.clear()

    @property
    def config(self) -> StreamingConfig:
        """Get validator configuration."""
        return self._config

    @property
    def validators(self) -> list[str]:
        """Get list of validators."""
        return list(self._validators)
