"""Base classes and interfaces for retention policies.

This module defines the abstract base classes and protocols that all
retention policy implementations must follow.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Protocol, runtime_checkable


# =============================================================================
# Enums
# =============================================================================


class RetentionAction(Enum):
    """Action to take when retention policy is triggered."""

    DELETE = "delete"  # Permanently delete
    ARCHIVE = "archive"  # Move to archive storage
    COMPRESS = "compress"  # Compress data
    TIER_DOWN = "tier_down"  # Move to cheaper storage tier


class PolicyMode(Enum):
    """How to combine multiple policies."""

    ALL = "all"  # All policies must agree to keep
    ANY = "any"  # Any policy can veto deletion


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class RetentionSchedule:
    """Schedule for automatic retention cleanup.

    Attributes:
        enabled: Whether automatic cleanup is enabled.
        interval_hours: Hours between cleanup runs.
        run_at_hour: Optional specific hour to run (0-23).
        run_on_days: Optional specific days to run (0=Mon, 6=Sun).
        max_duration_seconds: Maximum cleanup duration.
        batch_size: Items to process per batch.
    """

    enabled: bool = True
    interval_hours: int = 24
    run_at_hour: int | None = None
    run_on_days: list[int] | None = None
    max_duration_seconds: int = 3600
    batch_size: int = 100


@dataclass
class RetentionConfig:
    """Configuration for retention behavior.

    Attributes:
        policies: List of retention policies to apply.
        mode: How to combine multiple policies.
        default_action: Default action when policy triggers.
        schedule: Automatic cleanup schedule.
        dry_run: If True, only report what would be deleted.
        archive_store_name: Name of store for archived items.
        preserve_latest: Always keep at least one item per data_asset.
        excluded_tags: Tags that exempt items from cleanup.
        excluded_assets: Data assets exempt from cleanup.
    """

    policies: list["RetentionPolicy"] = field(default_factory=list)
    mode: PolicyMode = PolicyMode.ALL
    default_action: RetentionAction = RetentionAction.DELETE
    schedule: RetentionSchedule = field(default_factory=RetentionSchedule)
    dry_run: bool = False
    archive_store_name: str | None = None
    preserve_latest: bool = True
    excluded_tags: dict[str, str] = field(default_factory=dict)
    excluded_assets: list[str] = field(default_factory=list)


@dataclass
class RetentionResult:
    """Result of a retention cleanup run.

    Attributes:
        start_time: When cleanup started.
        end_time: When cleanup finished.
        items_scanned: Total items checked.
        items_deleted: Items permanently deleted.
        items_archived: Items moved to archive.
        items_compressed: Items compressed.
        items_tiered: Items moved to different tier.
        items_preserved: Items kept (not deleted).
        items_excluded: Items skipped due to exclusions.
        bytes_freed: Storage bytes freed.
        errors: List of errors encountered.
        dry_run: Whether this was a dry run.
    """

    start_time: datetime
    end_time: datetime | None = None
    items_scanned: int = 0
    items_deleted: int = 0
    items_archived: int = 0
    items_compressed: int = 0
    items_tiered: int = 0
    items_preserved: int = 0
    items_excluded: int = 0
    bytes_freed: int = 0
    errors: list[str] = field(default_factory=list)
    dry_run: bool = False

    @property
    def duration_seconds(self) -> float:
        """Get cleanup duration in seconds."""
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time).total_seconds()

    @property
    def total_processed(self) -> int:
        """Get total items processed (deleted + preserved)."""
        return self.items_deleted + self.items_preserved + self.items_archived

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "items_scanned": self.items_scanned,
            "items_deleted": self.items_deleted,
            "items_archived": self.items_archived,
            "items_compressed": self.items_compressed,
            "items_tiered": self.items_tiered,
            "items_preserved": self.items_preserved,
            "items_excluded": self.items_excluded,
            "bytes_freed": self.bytes_freed,
            "errors": self.errors,
            "dry_run": self.dry_run,
        }


@dataclass
class ItemMetadata:
    """Metadata about an item for retention evaluation.

    Attributes:
        item_id: Unique identifier.
        data_asset: Associated data asset.
        created_at: When item was created.
        size_bytes: Size in bytes.
        status: Validation status.
        tags: Item tags.
        access_count: Number of times accessed.
        last_accessed: Last access time.
    """

    item_id: str
    data_asset: str
    created_at: datetime
    size_bytes: int = 0
    status: str = ""
    tags: dict[str, str] = field(default_factory=dict)
    access_count: int = 0
    last_accessed: datetime | None = None


# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class Retainable(Protocol):
    """Protocol for objects that can have retention policies applied."""

    @property
    def id(self) -> str:
        """Unique identifier for the item."""
        ...

    @property
    def data_asset(self) -> str:
        """Associated data asset."""
        ...

    @property
    def run_time(self) -> datetime:
        """When the item was created."""
        ...

    @property
    def tags(self) -> dict[str, str]:
        """Item tags."""
        ...


# =============================================================================
# Abstract Base Classes
# =============================================================================


class RetentionPolicy(ABC):
    """Abstract base class for retention policies.

    Retention policies determine which items should be kept and which
    should be deleted/archived. They can be combined to create complex
    retention rules.

    Example:
        >>> class KeepRecentPolicy(RetentionPolicy):
        ...     def __init__(self, days: int):
        ...         self.days = days
        ...
        ...     def should_retain(self, item: ItemMetadata) -> bool:
        ...         age = datetime.now() - item.created_at
        ...         return age.days <= self.days
        ...
        ...     @property
        ...     def description(self) -> str:
        ...         return f"Keep items newer than {self.days} days"
    """

    @abstractmethod
    def should_retain(self, item: ItemMetadata) -> bool:
        """Determine if an item should be retained.

        Args:
            item: Metadata about the item.

        Returns:
            True if the item should be kept, False if it can be deleted.
        """
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of the policy."""
        pass

    @property
    def name(self) -> str:
        """Policy name (defaults to class name)."""
        return self.__class__.__name__

    @property
    def action(self) -> RetentionAction:
        """Action to take when policy triggers (default: DELETE)."""
        return RetentionAction.DELETE

    def get_expiry_time(self, item: ItemMetadata) -> datetime | None:
        """Get when an item will expire under this policy.

        Args:
            item: Metadata about the item.

        Returns:
            Expected expiry time, or None if not applicable.
        """
        return None

    def get_priority(self, item: ItemMetadata) -> int:
        """Get deletion priority for an item (higher = delete first).

        Args:
            item: Metadata about the item.

        Returns:
            Priority score (default 0).
        """
        return 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize policy to dictionary."""
        return {
            "type": self.__class__.__name__,
            "name": self.name,
            "description": self.description,
            "action": self.action.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RetentionPolicy":
        """Deserialize policy from dictionary.

        Override in subclasses for proper deserialization.
        """
        raise NotImplementedError("Subclasses must implement from_dict")


class PolicyEvaluator:
    """Evaluates multiple policies against an item.

    This class handles combining multiple policies according to
    the configured mode (ALL or ANY).
    """

    def __init__(
        self,
        policies: list[RetentionPolicy],
        mode: PolicyMode = PolicyMode.ALL,
    ) -> None:
        """Initialize the evaluator.

        Args:
            policies: List of policies to evaluate.
            mode: How to combine policy results.
        """
        self.policies = policies
        self.mode = mode

    def evaluate(self, item: ItemMetadata) -> tuple[bool, list[RetentionPolicy]]:
        """Evaluate all policies for an item.

        Args:
            item: Metadata about the item.

        Returns:
            Tuple of (should_retain, list of policies that said "delete").
        """
        if not self.policies:
            return True, []  # No policies = keep everything

        retain_votes: list[bool] = []
        delete_policies: list[RetentionPolicy] = []

        for policy in self.policies:
            should_retain = policy.should_retain(item)
            retain_votes.append(should_retain)
            if not should_retain:
                delete_policies.append(policy)

        if self.mode == PolicyMode.ALL:
            # All policies must agree to keep
            should_retain = all(retain_votes)
        else:
            # Any policy can keep the item
            should_retain = any(retain_votes)

        return should_retain, delete_policies

    def get_deletion_priority(self, item: ItemMetadata) -> int:
        """Get combined deletion priority from all policies.

        Args:
            item: Metadata about the item.

        Returns:
            Combined priority score.
        """
        return sum(p.get_priority(item) for p in self.policies)

    def get_earliest_expiry(self, item: ItemMetadata) -> datetime | None:
        """Get the earliest expiry time from all policies.

        Args:
            item: Metadata about the item.

        Returns:
            Earliest expiry time, or None if none apply.
        """
        expiry_times = [
            p.get_expiry_time(item)
            for p in self.policies
            if p.get_expiry_time(item) is not None
        ]
        return min(expiry_times) if expiry_times else None
