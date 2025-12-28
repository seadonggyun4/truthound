"""Audit logging for store operations.

This module provides comprehensive audit logging for compliance requirements
(SOC2, GDPR, HIPAA). It supports multiple backends and is designed for
high-throughput, non-blocking operation.

Features:
- Multiple backend support (memory, file, JSON, Elasticsearch, Kafka)
- Async batching for high-throughput
- Automatic sensitive data redaction
- Structured event format
- Query capability for audit trails
"""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
import os
import re
import threading
import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Callable, Generator, TextIO

from truthound.stores.observability.config import AuditConfig, AuditLogLevel
from truthound.stores.observability.protocols import (
    AuditBackend,
    ObservabilityContext,
)

logger = logging.getLogger(__name__)


class AuditEventType(str, Enum):
    """Types of audit events."""

    # CRUD operations
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"

    # Query operations
    QUERY = "query"
    LIST = "list"
    COUNT = "count"

    # Lifecycle operations
    INITIALIZE = "initialize"
    CLOSE = "close"
    FLUSH = "flush"

    # Batch operations
    BATCH_CREATE = "batch_create"
    BATCH_DELETE = "batch_delete"

    # Replication
    REPLICATE = "replicate"
    SYNC = "sync"

    # Migration
    MIGRATE = "migrate"
    ROLLBACK = "rollback"

    # Access control
    ACCESS_DENIED = "access_denied"
    ACCESS_GRANTED = "access_granted"

    # Errors
    ERROR = "error"
    VALIDATION_ERROR = "validation_error"


class AuditStatus(str, Enum):
    """Status of audited operation."""

    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    DENIED = "denied"


@dataclass
class AuditEvent:
    """An audit event record.

    Attributes:
        event_id: Unique identifier for this event.
        event_type: Type of the event.
        timestamp: When the event occurred.
        status: Status of the operation.
        store_type: Type of store (e.g., "filesystem", "s3").
        store_id: Unique identifier for the store instance.
        operation: Specific operation performed.
        resource_type: Type of resource affected.
        resource_id: ID of the affected resource.
        user_id: ID of the user performing the operation.
        tenant_id: ID of the tenant.
        correlation_id: Correlation ID for request tracing.
        trace_id: Distributed trace ID.
        span_id: Span ID within the trace.
        source_ip: Source IP address.
        user_agent: User agent string.
        duration_ms: Duration of the operation in milliseconds.
        input_summary: Summary of input (redacted if sensitive).
        output_summary: Summary of output (redacted if sensitive).
        error_message: Error message if operation failed.
        error_type: Type of error if operation failed.
        metadata: Additional metadata.
    """

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: AuditEventType = AuditEventType.READ
    timestamp: datetime = field(default_factory=datetime.utcnow)
    status: AuditStatus = AuditStatus.SUCCESS
    store_type: str = ""
    store_id: str = ""
    operation: str = ""
    resource_type: str = ""
    resource_id: str | None = None
    user_id: str | None = None
    tenant_id: str | None = None
    correlation_id: str | None = None
    trace_id: str | None = None
    span_id: str | None = None
    source_ip: str | None = None
    user_agent: str | None = None
    duration_ms: float | None = None
    input_summary: str | None = None
    output_summary: str | None = None
    error_message: str | None = None
    error_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary."""
        data = asdict(self)
        data["event_type"] = self.event_type.value
        data["status"] = self.status.value
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AuditEvent":
        """Create event from dictionary."""
        data = data.copy()
        data["event_type"] = AuditEventType(data["event_type"])
        data["status"] = AuditStatus(data["status"])
        if isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)

    @classmethod
    def from_context(
        cls,
        context: ObservabilityContext,
        event_type: AuditEventType,
        operation: str,
        **kwargs: Any,
    ) -> "AuditEvent":
        """Create an audit event from an observability context."""
        return cls(
            event_type=event_type,
            operation=operation,
            correlation_id=context.correlation_id,
            trace_id=context.trace_id,
            span_id=context.span_id,
            user_id=context.user_id,
            tenant_id=context.tenant_id,
            source_ip=context.source_ip,
            user_agent=context.user_agent,
            **kwargs,
        )


class DataRedactor:
    """Redacts sensitive data from audit logs."""

    def __init__(
        self,
        sensitive_fields: list[str] | None = None,
        patterns: list[re.Pattern[str]] | None = None,
    ) -> None:
        self.sensitive_fields = set(sensitive_fields or [])
        self.patterns = patterns or [
            re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),  # Email
            re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN
            re.compile(r"\b\d{16}\b"),  # Credit card
            re.compile(r"(?i)(password|secret|token|api_key)\s*[:=]\s*\S+"),  # Key-value
        ]
        self.redacted_value = "[REDACTED]"

    def redact(self, data: Any, path: str = "") -> Any:
        """Recursively redact sensitive data."""
        if isinstance(data, dict):
            return {
                k: self._redact_value(k, v, f"{path}.{k}")
                for k, v in data.items()
            }
        elif isinstance(data, list):
            return [self.redact(item, f"{path}[{i}]") for i, item in enumerate(data)]
        elif isinstance(data, str):
            return self._redact_string(data)
        return data

    def _redact_value(self, key: str, value: Any, path: str) -> Any:
        """Redact a single value based on key or content."""
        key_lower = key.lower()
        for sensitive in self.sensitive_fields:
            if sensitive.lower() in key_lower:
                return self.redacted_value
        return self.redact(value, path)

    def _redact_string(self, text: str) -> str:
        """Redact patterns from a string."""
        result = text
        for pattern in self.patterns:
            result = pattern.sub(self.redacted_value, result)
        return result


class BaseAuditBackend(ABC):
    """Base class for audit backends."""

    def __init__(self, config: AuditConfig) -> None:
        self.config = config
        self.redactor = DataRedactor(config.sensitive_fields) if config.redact_sensitive else None

    def _prepare_event(self, event: AuditEvent) -> AuditEvent:
        """Prepare event for logging (redaction, etc.)."""
        if self.redactor and event.input_summary:
            event.input_summary = self.redactor._redact_string(event.input_summary)
        if self.redactor and event.output_summary:
            event.output_summary = self.redactor._redact_string(event.output_summary)
        return event

    def _should_log(self, event: AuditEvent) -> bool:
        """Check if event should be logged based on level."""
        level = self.config.level

        if level == AuditLogLevel.MINIMAL:
            return event.event_type in {
                AuditEventType.CREATE,
                AuditEventType.UPDATE,
                AuditEventType.DELETE,
                AuditEventType.BATCH_CREATE,
                AuditEventType.BATCH_DELETE,
                AuditEventType.ERROR,
            }
        elif level == AuditLogLevel.STANDARD:
            return event.event_type not in {
                AuditEventType.INITIALIZE,
                AuditEventType.CLOSE,
            }
        elif level == AuditLogLevel.VERBOSE:
            return True
        elif level == AuditLogLevel.DEBUG:
            return True

        return True


class InMemoryAuditBackend(BaseAuditBackend):
    """In-memory audit backend for testing and development."""

    def __init__(self, config: AuditConfig | None = None) -> None:
        super().__init__(config or AuditConfig())
        self._events: list[AuditEvent] = []
        self._lock = threading.Lock()

    def log(self, event: AuditEvent) -> None:
        if not self._should_log(event):
            return
        event = self._prepare_event(event)
        with self._lock:
            self._events.append(event)

    def query(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        event_type: str | None = None,
        user_id: str | None = None,
        resource_id: str | None = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        with self._lock:
            results = []
            for event in reversed(self._events):
                if start_time and event.timestamp < start_time:
                    continue
                if end_time and event.timestamp > end_time:
                    continue
                if event_type and event.event_type.value != event_type:
                    continue
                if user_id and event.user_id != user_id:
                    continue
                if resource_id and event.resource_id != resource_id:
                    continue
                results.append(event)
                if len(results) >= limit:
                    break
            return results

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass

    def clear(self) -> None:
        """Clear all events (for testing)."""
        with self._lock:
            self._events.clear()

    @property
    def events(self) -> list[AuditEvent]:
        """Get all events (for testing)."""
        with self._lock:
            return list(self._events)


class FileAuditBackend(BaseAuditBackend):
    """File-based audit backend with rotation support."""

    def __init__(
        self,
        config: AuditConfig,
        file_path: Path | str | None = None,
    ) -> None:
        super().__init__(config)
        self.file_path = Path(file_path or config.file_path or "./audit.log")
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._file: TextIO | None = None
        self._lock = threading.Lock()
        self._open_file()

    def _open_file(self) -> None:
        self._file = open(self.file_path, "a", encoding="utf-8")

    def log(self, event: AuditEvent) -> None:
        if not self._should_log(event):
            return
        event = self._prepare_event(event)
        line = self._format_event(event)
        with self._lock:
            if self._file:
                self._file.write(line + "\n")

    def _format_event(self, event: AuditEvent) -> str:
        """Format event as log line."""
        return (
            f"{event.timestamp.isoformat()} "
            f"[{event.event_type.value.upper()}] "
            f"[{event.status.value}] "
            f"store={event.store_type} "
            f"op={event.operation} "
            f"resource={event.resource_id or 'N/A'} "
            f"user={event.user_id or 'N/A'} "
            f"correlation_id={event.correlation_id or 'N/A'} "
            f"duration_ms={event.duration_ms or 0:.2f}"
        )

    def query(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        event_type: str | None = None,
        user_id: str | None = None,
        resource_id: str | None = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        # File backend doesn't support efficient querying
        # Return empty list - use JsonAuditBackend for queryable audit logs
        logger.warning("FileAuditBackend does not support querying. Use JsonAuditBackend.")
        return []

    def flush(self) -> None:
        with self._lock:
            if self._file:
                self._file.flush()

    def close(self) -> None:
        with self._lock:
            if self._file:
                self._file.close()
                self._file = None


class JsonAuditBackend(BaseAuditBackend):
    """JSON-based audit backend with querying support."""

    def __init__(
        self,
        config: AuditConfig,
        file_path: Path | str | None = None,
        compress: bool = False,
    ) -> None:
        super().__init__(config)
        self.file_path = Path(file_path or config.file_path or "./audit.jsonl")
        self.compress = compress
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._buffer: list[AuditEvent] = []
        self._last_flush = datetime.utcnow()

    def log(self, event: AuditEvent) -> None:
        if not self._should_log(event):
            return
        event = self._prepare_event(event)
        with self._lock:
            self._buffer.append(event)
            if len(self._buffer) >= self.config.batch_size:
                self._flush_buffer()

    def _flush_buffer(self) -> None:
        """Flush buffer to file."""
        if not self._buffer:
            return

        if self.compress:
            with gzip.open(self.file_path.with_suffix(".jsonl.gz"), "at") as f:
                for event in self._buffer:
                    f.write(json.dumps(event.to_dict()) + "\n")
        else:
            with open(self.file_path, "a", encoding="utf-8") as f:
                for event in self._buffer:
                    f.write(json.dumps(event.to_dict()) + "\n")

        self._buffer.clear()
        self._last_flush = datetime.utcnow()

    def query(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        event_type: str | None = None,
        user_id: str | None = None,
        resource_id: str | None = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        results: list[AuditEvent] = []

        # Check buffered events first
        with self._lock:
            for event in self._buffer:
                if self._matches_filter(
                    event, start_time, end_time, event_type, user_id, resource_id
                ):
                    results.append(event)

        # Then check file
        if self.file_path.exists():
            open_func = gzip.open if self.compress else open
            file_path = (
                self.file_path.with_suffix(".jsonl.gz")
                if self.compress
                else self.file_path
            )
            if file_path.exists():
                with open_func(file_path, "rt") as f:  # type: ignore
                    for line in f:
                        if len(results) >= limit:
                            break
                        try:
                            event = AuditEvent.from_dict(json.loads(line))
                            if self._matches_filter(
                                event,
                                start_time,
                                end_time,
                                event_type,
                                user_id,
                                resource_id,
                            ):
                                results.append(event)
                        except (json.JSONDecodeError, KeyError):
                            continue

        return results[:limit]

    def _matches_filter(
        self,
        event: AuditEvent,
        start_time: datetime | None,
        end_time: datetime | None,
        event_type: str | None,
        user_id: str | None,
        resource_id: str | None,
    ) -> bool:
        if start_time and event.timestamp < start_time:
            return False
        if end_time and event.timestamp > end_time:
            return False
        if event_type and event.event_type.value != event_type:
            return False
        if user_id and event.user_id != user_id:
            return False
        if resource_id and event.resource_id != resource_id:
            return False
        return True

    def flush(self) -> None:
        with self._lock:
            self._flush_buffer()

    def close(self) -> None:
        self.flush()


class CompositeAuditBackend(BaseAuditBackend):
    """Composite backend that writes to multiple backends."""

    def __init__(
        self,
        backends: list[AuditBackend],
        config: AuditConfig | None = None,
    ) -> None:
        super().__init__(config or AuditConfig())
        self.backends = backends

    def log(self, event: AuditEvent) -> None:
        for backend in self.backends:
            try:
                backend.log(event)
            except Exception as e:
                logger.error(f"Failed to log to backend {backend}: {e}")

    def query(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        event_type: str | None = None,
        user_id: str | None = None,
        resource_id: str | None = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        # Query from first backend that supports it
        for backend in self.backends:
            try:
                results = backend.query(
                    start_time, end_time, event_type, user_id, resource_id, limit
                )
                if results:
                    return results
            except Exception as e:
                logger.error(f"Failed to query backend {backend}: {e}")
        return []

    def flush(self) -> None:
        for backend in self.backends:
            try:
                backend.flush()
            except Exception as e:
                logger.error(f"Failed to flush backend {backend}: {e}")

    def close(self) -> None:
        for backend in self.backends:
            try:
                backend.close()
            except Exception as e:
                logger.error(f"Failed to close backend {backend}: {e}")


class AsyncAuditBackend(BaseAuditBackend):
    """Async wrapper for audit backends using a background thread."""

    def __init__(
        self,
        backend: AuditBackend,
        config: AuditConfig | None = None,
        queue_size: int = 10000,
    ) -> None:
        super().__init__(config or AuditConfig())
        self._backend = backend
        self._queue: Queue[AuditEvent | None] = Queue(maxsize=queue_size)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._running = True
        self._thread.start()

    def _worker(self) -> None:
        """Background worker that processes audit events."""
        while self._running:
            try:
                event = self._queue.get(timeout=1.0)
                if event is None:
                    break
                self._backend.log(event)
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing audit event: {e}")

    def log(self, event: AuditEvent) -> None:
        try:
            self._queue.put_nowait(event)
        except Exception:
            logger.warning("Audit queue full, dropping event")

    def query(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        event_type: str | None = None,
        user_id: str | None = None,
        resource_id: str | None = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        return self._backend.query(
            start_time, end_time, event_type, user_id, resource_id, limit
        )

    def flush(self) -> None:
        # Wait for queue to drain
        while not self._queue.empty():
            pass
        self._backend.flush()

    def close(self) -> None:
        self._running = False
        self._queue.put(None)
        self._thread.join(timeout=5.0)
        self._backend.close()


class AuditLogger:
    """High-level audit logger with context management."""

    def __init__(
        self,
        backend: AuditBackend,
        store_type: str = "unknown",
        store_id: str | None = None,
    ) -> None:
        self.backend = backend
        self.store_type = store_type
        self.store_id = store_id or str(uuid.uuid4())[:8]
        self._context_stack: list[ObservabilityContext] = []

    @contextmanager
    def operation(
        self,
        event_type: AuditEventType,
        operation: str,
        resource_type: str = "",
        resource_id: str | None = None,
        context: ObservabilityContext | None = None,
        **metadata: Any,
    ) -> Generator[AuditEvent, None, None]:
        """Context manager for auditing an operation."""
        start_time = datetime.utcnow()

        event = AuditEvent(
            event_type=event_type,
            timestamp=start_time,
            store_type=self.store_type,
            store_id=self.store_id,
            operation=operation,
            resource_type=resource_type,
            resource_id=resource_id,
            metadata=metadata,
        )

        if context:
            event.correlation_id = context.correlation_id
            event.trace_id = context.trace_id
            event.span_id = context.span_id
            event.user_id = context.user_id
            event.tenant_id = context.tenant_id
            event.source_ip = context.source_ip
            event.user_agent = context.user_agent

        try:
            yield event
            event.status = AuditStatus.SUCCESS
        except Exception as e:
            event.status = AuditStatus.FAILURE
            event.error_type = type(e).__name__
            event.error_message = str(e)
            raise
        finally:
            end_time = datetime.utcnow()
            event.duration_ms = (end_time - start_time).total_seconds() * 1000
            self.backend.log(event)

    def log_event(
        self,
        event_type: AuditEventType,
        operation: str,
        status: AuditStatus = AuditStatus.SUCCESS,
        **kwargs: Any,
    ) -> None:
        """Log a single audit event."""
        event = AuditEvent(
            event_type=event_type,
            status=status,
            store_type=self.store_type,
            store_id=self.store_id,
            operation=operation,
            **kwargs,
        )
        self.backend.log(event)

    def flush(self) -> None:
        """Flush any buffered events."""
        self.backend.flush()

    def close(self) -> None:
        """Close the audit logger."""
        self.backend.close()
