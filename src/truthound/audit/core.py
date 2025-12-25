"""Core types, configuration, and interfaces for audit logging.

This module provides the foundational types and interfaces for the
audit logging system, following enterprise security and compliance standards.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Generic, Iterator, Mapping, TypeVar


# =============================================================================
# Enums
# =============================================================================


class AuditEventType(Enum):
    """Types of audit events."""

    # Authentication & Authorization
    LOGIN = "login"
    LOGOUT = "logout"
    LOGIN_FAILED = "login_failed"
    PASSWORD_CHANGE = "password_change"
    PERMISSION_CHANGE = "permission_change"
    ACCESS_DENIED = "access_denied"

    # Data Operations
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    EXPORT = "export"
    IMPORT = "import"

    # System Operations
    CONFIG_CHANGE = "config_change"
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    BACKUP = "backup"
    RESTORE = "restore"

    # Validation & Checkpoint
    VALIDATION_START = "validation_start"
    VALIDATION_COMPLETE = "validation_complete"
    VALIDATION_FAILED = "validation_failed"
    CHECKPOINT_RUN = "checkpoint_run"

    # Security Events
    SECURITY_ALERT = "security_alert"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"

    # Custom
    CUSTOM = "custom"


class AuditSeverity(Enum):
    """Severity levels for audit events."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AuditOutcome(Enum):
    """Outcome of an audited action."""

    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


class AuditCategory(Enum):
    """Categories for audit events (for filtering and reporting)."""

    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    SYSTEM = "system"
    SECURITY = "security"
    COMPLIANCE = "compliance"
    VALIDATION = "validation"
    CUSTOM = "custom"


# =============================================================================
# Core Data Types
# =============================================================================


@dataclass
class AuditActor:
    """Represents the entity performing an action.

    Example:
        >>> actor = AuditActor(
        ...     id="user:123",
        ...     type="user",
        ...     name="john.doe@example.com",
        ...     ip_address="192.168.1.100",
        ... )
    """

    id: str
    type: str = "user"  # user, service, system, api_key
    name: str = ""
    email: str = ""
    ip_address: str = ""
    user_agent: str = ""
    session_id: str = ""
    roles: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def system(cls) -> "AuditActor":
        """Create a system actor."""
        return cls(id="system", type="system", name="System")

    @classmethod
    def anonymous(cls, ip_address: str = "") -> "AuditActor":
        """Create an anonymous actor."""
        return cls(
            id="anonymous",
            type="anonymous",
            name="Anonymous",
            ip_address=ip_address,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "email": self.email,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "session_id": self.session_id,
            "roles": self.roles,
            "metadata": self.metadata,
        }


@dataclass
class AuditResource:
    """Represents the resource being acted upon.

    Example:
        >>> resource = AuditResource(
        ...     id="dataset:456",
        ...     type="dataset",
        ...     name="user_transactions",
        ...     path="/data/transactions.csv",
        ... )
    """

    id: str
    type: str
    name: str = ""
    path: str = ""
    owner: str = ""
    sensitivity: str = ""  # public, internal, confidential, restricted
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "path": self.path,
            "owner": self.owner,
            "sensitivity": self.sensitivity,
            "metadata": self.metadata,
        }


@dataclass
class AuditContext:
    """Contextual information about an audit event.

    Example:
        >>> context = AuditContext(
        ...     request_id="req-abc123",
        ...     trace_id="trace-xyz789",
        ...     environment="production",
        ... )
    """

    request_id: str = ""
    trace_id: str = ""
    span_id: str = ""
    correlation_id: str = ""
    environment: str = ""
    service_name: str = ""
    service_version: str = ""
    host: str = ""
    tags: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "correlation_id": self.correlation_id,
            "environment": self.environment,
            "service_name": self.service_name,
            "service_version": self.service_version,
            "host": self.host,
            "tags": self.tags,
            "metadata": self.metadata,
        }


@dataclass
class AuditEvent:
    """Represents a single audit log event.

    This is the core data structure for audit logging, containing
    all information about an audited action.

    Example:
        >>> event = AuditEvent(
        ...     event_type=AuditEventType.UPDATE,
        ...     actor=AuditActor(id="user:123", name="john"),
        ...     resource=AuditResource(id="doc:456", type="document"),
        ...     action="update_document",
        ...     outcome=AuditOutcome.SUCCESS,
        ... )
    """

    # Event identification
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Event classification
    event_type: AuditEventType = AuditEventType.CUSTOM
    category: AuditCategory = AuditCategory.CUSTOM
    severity: AuditSeverity = AuditSeverity.INFO

    # Action details
    action: str = ""
    outcome: AuditOutcome = AuditOutcome.UNKNOWN
    message: str = ""
    reason: str = ""

    # Entities
    actor: AuditActor | None = None
    resource: AuditResource | None = None
    target: AuditResource | None = None  # For operations involving two resources

    # Context
    context: AuditContext = field(default_factory=AuditContext)

    # Data changes
    old_value: Any = None
    new_value: Any = None
    changes: dict[str, Any] = field(default_factory=dict)

    # Additional data
    data: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    # Timing
    duration_ms: float | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None

    # Integrity
    checksum: str = ""
    signed_by: str = ""
    signature: str = ""

    def __post_init__(self) -> None:
        """Post-initialization processing."""
        if self.actor is None:
            self.actor = AuditActor.system()

    @property
    def timestamp_iso(self) -> str:
        """Get ISO formatted timestamp."""
        return self.timestamp.isoformat()

    @property
    def timestamp_unix(self) -> float:
        """Get Unix timestamp."""
        return self.timestamp.timestamp()

    def compute_checksum(self) -> str:
        """Compute checksum for integrity verification."""
        data = json.dumps(
            {
                "id": self.id,
                "timestamp": self.timestamp_iso,
                "event_type": self.event_type.value,
                "action": self.action,
                "outcome": self.outcome.value,
                "actor_id": self.actor.id if self.actor else None,
                "resource_id": self.resource.id if self.resource else None,
            },
            sort_keys=True,
        )
        return hashlib.sha256(data.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "timestamp": self.timestamp_iso,
            "timestamp_unix": self.timestamp_unix,
            "event_type": self.event_type.value,
            "category": self.category.value,
            "severity": self.severity.value,
            "action": self.action,
            "outcome": self.outcome.value,
            "message": self.message,
            "reason": self.reason,
            "actor": self.actor.to_dict() if self.actor else None,
            "resource": self.resource.to_dict() if self.resource else None,
            "target": self.target.to_dict() if self.target else None,
            "context": self.context.to_dict(),
            "old_value": self.old_value,
            "new_value": self.new_value,
            "changes": self.changes,
            "data": self.data,
            "tags": self.tags,
            "duration_ms": self.duration_ms,
            "checksum": self.checksum or self.compute_checksum(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AuditEvent":
        """Create from dictionary."""
        # Parse nested objects
        actor = None
        if data.get("actor"):
            actor = AuditActor(**data["actor"])

        resource = None
        if data.get("resource"):
            resource = AuditResource(**data["resource"])

        target = None
        if data.get("target"):
            target = AuditResource(**data["target"])

        context = AuditContext()
        if data.get("context"):
            context = AuditContext(**data["context"])

        # Parse timestamp
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        elif timestamp is None:
            timestamp = datetime.now(timezone.utc)

        return cls(
            id=data.get("id", str(uuid.uuid4())),
            timestamp=timestamp,
            event_type=AuditEventType(data.get("event_type", "custom")),
            category=AuditCategory(data.get("category", "custom")),
            severity=AuditSeverity(data.get("severity", "info")),
            action=data.get("action", ""),
            outcome=AuditOutcome(data.get("outcome", "unknown")),
            message=data.get("message", ""),
            reason=data.get("reason", ""),
            actor=actor,
            resource=resource,
            target=target,
            context=context,
            old_value=data.get("old_value"),
            new_value=data.get("new_value"),
            changes=data.get("changes", {}),
            data=data.get("data", {}),
            tags=data.get("tags", []),
            duration_ms=data.get("duration_ms"),
            checksum=data.get("checksum", ""),
        )


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class AuditConfig:
    """Configuration for audit logging.

    Example:
        >>> config = AuditConfig(
        ...     enabled=True,
        ...     service_name="my-service",
        ...     include_read_events=False,
        ...     retention_days=365,
        ... )
    """

    # Basic settings
    enabled: bool = True
    service_name: str = ""
    service_version: str = ""
    environment: str = ""

    # Event filtering
    include_read_events: bool = True
    include_debug_events: bool = False
    min_severity: AuditSeverity = AuditSeverity.INFO
    excluded_actions: list[str] = field(default_factory=list)
    excluded_actors: list[str] = field(default_factory=list)

    # Privacy settings
    mask_sensitive_data: bool = True
    sensitive_fields: list[str] = field(
        default_factory=lambda: [
            "password",
            "token",
            "secret",
            "api_key",
            "credit_card",
            "ssn",
        ]
    )
    anonymize_ip: bool = False

    # Retention
    retention_days: int = 90
    archive_after_days: int = 30

    # Integrity
    compute_checksums: bool = True
    sign_events: bool = False
    signing_key: str = ""

    # Performance
    async_write: bool = True
    batch_size: int = 100
    flush_interval_seconds: float = 5.0
    max_queue_size: int = 10000

    # Storage
    storage_backend: str = "memory"
    storage_config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AuditConfig":
        """Create config from dictionary."""
        if "min_severity" in data and isinstance(data["min_severity"], str):
            data["min_severity"] = AuditSeverity(data["min_severity"])
        return cls(**data)

    @classmethod
    def development(cls) -> "AuditConfig":
        """Create development configuration."""
        return cls(
            enabled=True,
            include_debug_events=True,
            min_severity=AuditSeverity.DEBUG,
            async_write=False,
            storage_backend="memory",
        )

    @classmethod
    def production(cls, service_name: str) -> "AuditConfig":
        """Create production configuration."""
        return cls(
            enabled=True,
            service_name=service_name,
            include_debug_events=False,
            min_severity=AuditSeverity.INFO,
            compute_checksums=True,
            async_write=True,
            retention_days=365,
        )


# =============================================================================
# Exceptions
# =============================================================================


class AuditError(Exception):
    """Base exception for audit errors."""

    pass


class AuditStorageError(AuditError):
    """Error during audit storage operations."""

    pass


class AuditValidationError(AuditError):
    """Error during audit event validation."""

    pass


class AuditConfigError(AuditError):
    """Error in audit configuration."""

    pass


# =============================================================================
# Abstract Interfaces
# =============================================================================


class AuditStorage(ABC):
    """Abstract base class for audit log storage.

    Implementations provide persistence for audit events.
    """

    @abstractmethod
    def write(self, event: AuditEvent) -> None:
        """Write a single audit event.

        Args:
            event: Audit event to write.
        """
        pass

    @abstractmethod
    def write_batch(self, events: list[AuditEvent]) -> None:
        """Write multiple audit events.

        Args:
            events: List of events to write.
        """
        pass

    @abstractmethod
    def read(self, event_id: str) -> AuditEvent | None:
        """Read a single audit event by ID.

        Args:
            event_id: Event ID.

        Returns:
            AuditEvent or None if not found.
        """
        pass

    @abstractmethod
    def query(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        event_types: list[AuditEventType] | None = None,
        actor_id: str | None = None,
        resource_id: str | None = None,
        outcome: AuditOutcome | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AuditEvent]:
        """Query audit events.

        Args:
            start_time: Start of time range.
            end_time: End of time range.
            event_types: Filter by event types.
            actor_id: Filter by actor ID.
            resource_id: Filter by resource ID.
            outcome: Filter by outcome.
            limit: Maximum results.
            offset: Skip first N results.

        Returns:
            List of matching events.
        """
        pass

    @abstractmethod
    def count(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        event_types: list[AuditEventType] | None = None,
    ) -> int:
        """Count matching audit events.

        Args:
            start_time: Start of time range.
            end_time: End of time range.
            event_types: Filter by event types.

        Returns:
            Count of matching events.
        """
        pass

    @abstractmethod
    def delete_before(self, before: datetime) -> int:
        """Delete events before a given time.

        Args:
            before: Delete events before this time.

        Returns:
            Number of deleted events.
        """
        pass

    def close(self) -> None:
        """Close storage connection."""
        pass

    def flush(self) -> None:
        """Flush any buffered events."""
        pass


class AuditFormatter(ABC):
    """Abstract base class for audit event formatters."""

    @abstractmethod
    def format(self, event: AuditEvent) -> str:
        """Format an audit event to string.

        Args:
            event: Event to format.

        Returns:
            Formatted string.
        """
        pass

    @abstractmethod
    def parse(self, data: str) -> AuditEvent:
        """Parse string back to audit event.

        Args:
            data: Formatted string.

        Returns:
            Parsed AuditEvent.
        """
        pass


class AuditFilter(ABC):
    """Abstract base class for audit event filters."""

    @abstractmethod
    def should_log(self, event: AuditEvent) -> bool:
        """Determine if event should be logged.

        Args:
            event: Event to check.

        Returns:
            True if should be logged.
        """
        pass


class AuditProcessor(ABC):
    """Abstract base class for audit event processors.

    Processors can modify or enrich events before storage.
    """

    @abstractmethod
    def process(self, event: AuditEvent) -> AuditEvent:
        """Process an audit event.

        Args:
            event: Event to process.

        Returns:
            Processed event.
        """
        pass


# =============================================================================
# Event Builder
# =============================================================================


class AuditEventBuilder:
    """Builder for creating audit events with fluent interface.

    Example:
        >>> event = (
        ...     AuditEventBuilder()
        ...     .set_type(AuditEventType.UPDATE)
        ...     .set_actor(user_id="123", name="john")
        ...     .set_resource(id="doc:456", type="document")
        ...     .set_action("update_document")
        ...     .set_outcome(AuditOutcome.SUCCESS)
        ...     .build()
        ... )
    """

    def __init__(self) -> None:
        self._event_type = AuditEventType.CUSTOM
        self._category = AuditCategory.CUSTOM
        self._severity = AuditSeverity.INFO
        self._action = ""
        self._outcome = AuditOutcome.UNKNOWN
        self._message = ""
        self._reason = ""
        self._actor: AuditActor | None = None
        self._resource: AuditResource | None = None
        self._target: AuditResource | None = None
        self._context = AuditContext()
        self._old_value: Any = None
        self._new_value: Any = None
        self._changes: dict[str, Any] = {}
        self._data: dict[str, Any] = {}
        self._tags: list[str] = []
        self._duration_ms: float | None = None

    def set_type(self, event_type: AuditEventType) -> "AuditEventBuilder":
        """Set event type."""
        self._event_type = event_type
        return self

    def set_category(self, category: AuditCategory) -> "AuditEventBuilder":
        """Set category."""
        self._category = category
        return self

    def set_severity(self, severity: AuditSeverity) -> "AuditEventBuilder":
        """Set severity."""
        self._severity = severity
        return self

    def set_action(self, action: str) -> "AuditEventBuilder":
        """Set action."""
        self._action = action
        return self

    def set_outcome(self, outcome: AuditOutcome) -> "AuditEventBuilder":
        """Set outcome."""
        self._outcome = outcome
        return self

    def set_message(self, message: str) -> "AuditEventBuilder":
        """Set message."""
        self._message = message
        return self

    def set_reason(self, reason: str) -> "AuditEventBuilder":
        """Set reason."""
        self._reason = reason
        return self

    def set_actor(
        self,
        id: str | None = None,  # noqa: A002
        type: str = "user",  # noqa: A002
        name: str = "",
        email: str = "",
        ip_address: str = "",
        **kwargs: Any,
    ) -> "AuditEventBuilder":
        """Set actor."""
        if id:
            self._actor = AuditActor(
                id=id,
                type=type,
                name=name,
                email=email,
                ip_address=ip_address,
                **kwargs,
            )
        return self

    def set_actor_object(self, actor: AuditActor) -> "AuditEventBuilder":
        """Set actor from object."""
        self._actor = actor
        return self

    def set_resource(
        self,
        id: str | None = None,  # noqa: A002
        type: str = "",  # noqa: A002
        name: str = "",
        path: str = "",
        **kwargs: Any,
    ) -> "AuditEventBuilder":
        """Set resource."""
        if id:
            self._resource = AuditResource(
                id=id,
                type=type,
                name=name,
                path=path,
                **kwargs,
            )
        return self

    def set_resource_object(self, resource: AuditResource) -> "AuditEventBuilder":
        """Set resource from object."""
        self._resource = resource
        return self

    def set_target(
        self,
        id: str | None = None,  # noqa: A002
        type: str = "",  # noqa: A002
        name: str = "",
        **kwargs: Any,
    ) -> "AuditEventBuilder":
        """Set target resource."""
        if id:
            self._target = AuditResource(id=id, type=type, name=name, **kwargs)
        return self

    def set_context(
        self,
        request_id: str = "",
        trace_id: str = "",
        environment: str = "",
        **kwargs: Any,
    ) -> "AuditEventBuilder":
        """Set context."""
        self._context = AuditContext(
            request_id=request_id,
            trace_id=trace_id,
            environment=environment,
            **kwargs,
        )
        return self

    def set_changes(
        self,
        old_value: Any = None,
        new_value: Any = None,
        changes: dict[str, Any] | None = None,
    ) -> "AuditEventBuilder":
        """Set change data."""
        self._old_value = old_value
        self._new_value = new_value
        if changes:
            self._changes = changes
        return self

    def add_data(self, key: str, value: Any) -> "AuditEventBuilder":
        """Add data field."""
        self._data[key] = value
        return self

    def set_data(self, data: dict[str, Any]) -> "AuditEventBuilder":
        """Set all data."""
        self._data = data
        return self

    def add_tag(self, tag: str) -> "AuditEventBuilder":
        """Add tag."""
        self._tags.append(tag)
        return self

    def set_tags(self, tags: list[str]) -> "AuditEventBuilder":
        """Set all tags."""
        self._tags = tags
        return self

    def set_duration(self, duration_ms: float) -> "AuditEventBuilder":
        """Set duration in milliseconds."""
        self._duration_ms = duration_ms
        return self

    def build(self) -> AuditEvent:
        """Build the audit event."""
        return AuditEvent(
            event_type=self._event_type,
            category=self._category,
            severity=self._severity,
            action=self._action,
            outcome=self._outcome,
            message=self._message,
            reason=self._reason,
            actor=self._actor,
            resource=self._resource,
            target=self._target,
            context=self._context,
            old_value=self._old_value,
            new_value=self._new_value,
            changes=self._changes,
            data=self._data,
            tags=self._tags,
            duration_ms=self._duration_ms,
        )


# =============================================================================
# Utility Functions
# =============================================================================


def current_timestamp() -> datetime:
    """Get current UTC timestamp."""
    return datetime.now(timezone.utc)


def generate_event_id() -> str:
    """Generate unique event ID."""
    return str(uuid.uuid4())


def mask_sensitive_value(value: Any, mask_char: str = "*") -> str:
    """Mask a sensitive value.

    Args:
        value: Value to mask.
        mask_char: Character to use for masking.

    Returns:
        Masked string.
    """
    if value is None:
        return ""

    s = str(value)
    if len(s) <= 4:
        return mask_char * len(s)

    # Show first and last 2 characters
    return s[:2] + mask_char * (len(s) - 4) + s[-2:]


def anonymize_ip_address(ip: str) -> str:
    """Anonymize an IP address.

    Args:
        ip: IP address to anonymize.

    Returns:
        Anonymized IP.
    """
    if not ip:
        return ""

    parts = ip.split(".")
    if len(parts) == 4:
        # IPv4: mask last octet
        return f"{parts[0]}.{parts[1]}.{parts[2]}.0"

    # IPv6: mask last 80 bits
    if ":" in ip:
        parts = ip.split(":")
        return ":".join(parts[:3]) + "::0"

    return ip
