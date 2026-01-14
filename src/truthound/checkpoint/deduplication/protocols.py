"""Deduplication Protocols and Core Types.

This module defines the core protocols and data types for the
notification deduplication system.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Protocol, Sequence, runtime_checkable


class WindowUnit(str, Enum):
    """Time window unit."""

    SECONDS = "seconds"
    MINUTES = "minutes"
    HOURS = "hours"
    DAYS = "days"

    def to_seconds(self, value: int) -> int:
        """Convert value to seconds."""
        multipliers = {
            WindowUnit.SECONDS: 1,
            WindowUnit.MINUTES: 60,
            WindowUnit.HOURS: 3600,
            WindowUnit.DAYS: 86400,
        }
        return value * multipliers[self]


@dataclass(frozen=True)
class TimeWindow:
    """Time window configuration for deduplication.

    Defines the duration within which duplicate notifications
    are suppressed.

    Attributes:
        value: Numeric value of the window.
        unit: Unit of the window (seconds, minutes, hours, days).

    Example:
        >>> window = TimeWindow(5, WindowUnit.MINUTES)
        >>> window.to_timedelta()
        datetime.timedelta(seconds=300)
    """

    value: int = 300
    unit: WindowUnit = WindowUnit.SECONDS

    def __init__(
        self,
        value: int | None = None,
        unit: WindowUnit | str = WindowUnit.SECONDS,
        *,
        seconds: int | None = None,
        minutes: int | None = None,
        hours: int | None = None,
        days: int | None = None,
    ) -> None:
        """Initialize time window.

        Can be initialized with value/unit or with named parameters.

        Args:
            value: Numeric value (used with unit).
            unit: Time unit.
            seconds: Shorthand for seconds.
            minutes: Shorthand for minutes.
            hours: Shorthand for hours.
            days: Shorthand for days.
        """
        # Handle named parameters
        if seconds is not None:
            object.__setattr__(self, "value", seconds)
            object.__setattr__(self, "unit", WindowUnit.SECONDS)
        elif minutes is not None:
            object.__setattr__(self, "value", minutes)
            object.__setattr__(self, "unit", WindowUnit.MINUTES)
        elif hours is not None:
            object.__setattr__(self, "value", hours)
            object.__setattr__(self, "unit", WindowUnit.HOURS)
        elif days is not None:
            object.__setattr__(self, "value", days)
            object.__setattr__(self, "unit", WindowUnit.DAYS)
        else:
            # Use value/unit parameters
            object.__setattr__(self, "value", value if value is not None else 300)
            if isinstance(unit, str):
                unit = WindowUnit(unit.lower())
            object.__setattr__(self, "unit", unit)

    @property
    def total_seconds(self) -> int:
        """Get total seconds in this window."""
        return self.unit.to_seconds(self.value)

    def to_timedelta(self) -> timedelta:
        """Convert to timedelta."""
        return timedelta(seconds=self.total_seconds)

    def __str__(self) -> str:
        return f"{self.value} {self.unit.value}"

    def __repr__(self) -> str:
        return f"TimeWindow({self.value}, {self.unit.value})"


@dataclass
class NotificationFingerprint:
    """Unique fingerprint for a notification.

    The fingerprint identifies a specific notification based on
    its key attributes, enabling duplicate detection.

    Attributes:
        key: The hash key (unique identifier).
        checkpoint_name: Name of the checkpoint.
        action_type: Type of notification action (slack, email, etc.).
        components: Components used to generate the fingerprint.
        created_at: When the fingerprint was created.
        metadata: Additional metadata.
    """

    key: str
    checkpoint_name: str
    action_type: str
    components: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def generate(
        cls,
        checkpoint_name: str,
        action_type: str,
        *,
        severity: str | None = None,
        data_asset: str | None = None,
        issue_types: Sequence[str] | None = None,
        custom_key: str | None = None,
        algorithm: str = "sha256",
        **extra_components: Any,
    ) -> NotificationFingerprint:
        """Generate a fingerprint from notification attributes.

        Args:
            checkpoint_name: Name of the checkpoint.
            action_type: Type of action (slack, email, etc.).
            severity: Issue severity level.
            data_asset: Data asset name.
            issue_types: Types of issues detected.
            custom_key: Custom key override.
            algorithm: Hash algorithm.
            **extra_components: Additional components to include.

        Returns:
            Generated NotificationFingerprint.

        Example:
            >>> fp = NotificationFingerprint.generate(
            ...     checkpoint_name="data_quality",
            ...     action_type="slack",
            ...     severity="high",
            ...     data_asset="orders",
            ... )
        """
        # Build components dictionary
        components: dict[str, Any] = {
            "checkpoint_name": checkpoint_name,
            "action_type": action_type,
        }

        if severity:
            components["severity"] = severity
        if data_asset:
            components["data_asset"] = data_asset
        if issue_types:
            components["issue_types"] = sorted(issue_types)

        # Add extra components
        components.update(extra_components)

        # Generate key
        if custom_key:
            key = custom_key
        else:
            canonical = json.dumps(components, sort_keys=True, separators=(",", ":"))
            hasher = hashlib.new(algorithm)
            hasher.update(canonical.encode("utf-8"))
            key = hasher.hexdigest()[:32]  # Use first 32 chars

        return cls(
            key=key,
            checkpoint_name=checkpoint_name,
            action_type=action_type,
            components=components,
        )

    @classmethod
    def from_checkpoint_result(
        cls,
        checkpoint_result: Any,
        action_type: str,
        *,
        include_issues: bool = True,
        include_metadata: bool = False,
    ) -> NotificationFingerprint:
        """Generate fingerprint from a CheckpointResult.

        Args:
            checkpoint_result: The checkpoint result object.
            action_type: Type of notification action.
            include_issues: Include issue types in fingerprint.
            include_metadata: Include metadata in fingerprint.

        Returns:
            Generated fingerprint.
        """
        components: dict[str, Any] = {
            "checkpoint_name": checkpoint_result.checkpoint_name,
            "action_type": action_type,
            "status": str(checkpoint_result.status),
        }

        if checkpoint_result.data_asset:
            components["data_asset"] = checkpoint_result.data_asset

        if include_issues and checkpoint_result.validation_result:
            issue_types = set()
            for issue in checkpoint_result.validation_result.issues:
                issue_types.add(issue.validator_name)
            if issue_types:
                components["issue_types"] = sorted(issue_types)

        if include_metadata:
            components["metadata"] = checkpoint_result.metadata

        return cls.generate(
            checkpoint_name=checkpoint_result.checkpoint_name,
            action_type=action_type,
            **components,
        )

    def with_window(self, window: TimeWindow) -> NotificationFingerprint:
        """Create fingerprint with time window encoded.

        This creates a time-bucketed fingerprint for window-based
        deduplication.

        Args:
            window: Time window for bucketing.

        Returns:
            New fingerprint with window bucket.
        """
        # Calculate window bucket
        window_seconds = window.total_seconds
        bucket = int(self.created_at.timestamp() // window_seconds)

        new_components = {
            **self.components,
            "_window_bucket": bucket,
            "_window_seconds": window_seconds,
        }

        canonical = json.dumps(new_components, sort_keys=True, separators=(",", ":"))
        hasher = hashlib.sha256()
        hasher.update(canonical.encode("utf-8"))
        key = hasher.hexdigest()[:32]

        return NotificationFingerprint(
            key=key,
            checkpoint_name=self.checkpoint_name,
            action_type=self.action_type,
            components=new_components,
            created_at=self.created_at,
            metadata=self.metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key": self.key,
            "checkpoint_name": self.checkpoint_name,
            "action_type": self.action_type,
            "components": self.components,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NotificationFingerprint:
        """Create from dictionary."""
        return cls(
            key=data["key"],
            checkpoint_name=data["checkpoint_name"],
            action_type=data["action_type"],
            components=data.get("components", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            metadata=data.get("metadata", {}),
        )

    def __hash__(self) -> int:
        return hash(self.key)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, NotificationFingerprint):
            return self.key == other.key
        return False


@dataclass
class DeduplicationRecord:
    """Record of a sent notification for deduplication.

    Attributes:
        fingerprint: The notification fingerprint.
        sent_at: When the notification was sent.
        expires_at: When this record expires.
        count: Number of duplicates suppressed.
        last_duplicate_at: When the last duplicate was detected.
        metadata: Additional record metadata.
    """

    fingerprint: NotificationFingerprint
    sent_at: datetime
    expires_at: datetime
    count: int = 1
    last_duplicate_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Check if this record has expired."""
        return datetime.now() > self.expires_at

    @property
    def remaining_ttl(self) -> timedelta:
        """Get remaining time-to-live."""
        remaining = self.expires_at - datetime.now()
        return max(remaining, timedelta(0))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "fingerprint": self.fingerprint.to_dict(),
            "sent_at": self.sent_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "count": self.count,
            "last_duplicate_at": self.last_duplicate_at.isoformat()
            if self.last_duplicate_at
            else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeduplicationRecord:
        """Create from dictionary."""
        return cls(
            fingerprint=NotificationFingerprint.from_dict(data["fingerprint"]),
            sent_at=datetime.fromisoformat(data["sent_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]),
            count=data.get("count", 1),
            last_duplicate_at=datetime.fromisoformat(data["last_duplicate_at"])
            if data.get("last_duplicate_at")
            else None,
            metadata=data.get("metadata", {}),
        )


@dataclass
class DeduplicationResult:
    """Result of a deduplication check.

    Attributes:
        is_duplicate: Whether the notification is a duplicate.
        fingerprint: The notification fingerprint.
        original_record: The original record if duplicate.
        suppressed_count: Number of times this has been suppressed.
        message: Human-readable result message.
    """

    is_duplicate: bool
    fingerprint: NotificationFingerprint
    original_record: DeduplicationRecord | None = None
    suppressed_count: int = 0
    message: str = ""

    @property
    def should_send(self) -> bool:
        """Check if notification should be sent."""
        return not self.is_duplicate


@dataclass
class DeduplicationStats:
    """Statistics for the deduplication system.

    Attributes:
        total_checked: Total notifications checked.
        duplicates_found: Number of duplicates detected.
        notifications_sent: Number of notifications sent.
        store_size: Current size of the deduplication store.
        oldest_record: Timestamp of oldest record.
        newest_record: Timestamp of newest record.
    """

    total_checked: int = 0
    duplicates_found: int = 0
    notifications_sent: int = 0
    store_size: int = 0
    oldest_record: datetime | None = None
    newest_record: datetime | None = None

    @property
    def deduplication_rate(self) -> float:
        """Calculate deduplication rate as percentage."""
        if self.total_checked == 0:
            return 0.0
        return (self.duplicates_found / self.total_checked) * 100

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_checked": self.total_checked,
            "duplicates_found": self.duplicates_found,
            "notifications_sent": self.notifications_sent,
            "store_size": self.store_size,
            "deduplication_rate": round(self.deduplication_rate, 2),
            "oldest_record": self.oldest_record.isoformat()
            if self.oldest_record
            else None,
            "newest_record": self.newest_record.isoformat()
            if self.newest_record
            else None,
        }


@runtime_checkable
class DeduplicationStore(Protocol):
    """Protocol for deduplication state storage backends.

    Implementations must provide thread-safe storage for
    deduplication records with TTL support.
    """

    def get(self, key: str) -> DeduplicationRecord | None:
        """Get a deduplication record by key.

        Args:
            key: The fingerprint key.

        Returns:
            The record if found and not expired, None otherwise.
        """
        ...

    def put(
        self,
        fingerprint: NotificationFingerprint,
        window: TimeWindow,
        metadata: dict[str, Any] | None = None,
    ) -> DeduplicationRecord:
        """Store a deduplication record.

        Args:
            fingerprint: The notification fingerprint.
            window: Time window for expiration.
            metadata: Additional metadata.

        Returns:
            The created record.
        """
        ...

    def increment(self, key: str) -> DeduplicationRecord | None:
        """Increment the duplicate count for a record.

        Args:
            key: The fingerprint key.

        Returns:
            The updated record if found, None otherwise.
        """
        ...

    def delete(self, key: str) -> bool:
        """Delete a record.

        Args:
            key: The fingerprint key.

        Returns:
            True if deleted, False if not found.
        """
        ...

    def cleanup_expired(self) -> int:
        """Remove all expired records.

        Returns:
            Number of records removed.
        """
        ...

    def get_stats(self) -> DeduplicationStats:
        """Get store statistics.

        Returns:
            Current statistics.
        """
        ...

    def clear(self) -> None:
        """Clear all records."""
        ...


class WindowStrategy(ABC):
    """Abstract base class for time window strategies.

    Different strategies determine how duplicates are detected
    within time windows.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Get strategy name."""
        pass

    @abstractmethod
    def get_window_key(
        self,
        fingerprint: NotificationFingerprint,
        window: TimeWindow,
        timestamp: datetime | None = None,
    ) -> str:
        """Get the window-specific key for a fingerprint.

        Args:
            fingerprint: The notification fingerprint.
            window: Time window configuration.
            timestamp: Timestamp for window calculation.

        Returns:
            Window-specific key.
        """
        pass

    @abstractmethod
    def is_in_window(
        self,
        record: DeduplicationRecord,
        window: TimeWindow,
        timestamp: datetime | None = None,
    ) -> bool:
        """Check if a record is still within the window.

        Args:
            record: The deduplication record.
            window: Time window configuration.
            timestamp: Current timestamp.

        Returns:
            True if record is within window.
        """
        pass
