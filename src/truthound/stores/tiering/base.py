"""Base classes and interfaces for tiered storage.

This module defines the abstract base classes and data structures that all
tiered storage implementations must follow.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from truthound.stores.base import ValidationStore


# =============================================================================
# Exceptions
# =============================================================================


class TieringError(Exception):
    """Base exception for tiering-related errors."""

    pass


class TierNotFoundError(TieringError):
    """Raised when a specified tier doesn't exist."""

    def __init__(self, tier_name: str) -> None:
        self.tier_name = tier_name
        super().__init__(f"Storage tier not found: {tier_name}")


class TierMigrationError(TieringError):
    """Raised when data migration between tiers fails."""

    def __init__(
        self,
        item_id: str,
        from_tier: str,
        to_tier: str,
        message: str,
    ) -> None:
        self.item_id = item_id
        self.from_tier = from_tier
        self.to_tier = to_tier
        super().__init__(
            f"Failed to migrate {item_id} from {from_tier} to {to_tier}: {message}"
        )


class TierAccessError(TieringError):
    """Raised when accessing a tier fails."""

    def __init__(self, tier_name: str, message: str) -> None:
        self.tier_name = tier_name
        super().__init__(f"Failed to access tier {tier_name}: {message}")


# =============================================================================
# Enums
# =============================================================================


class TierType(Enum):
    """Type of storage tier."""

    HOT = "hot"  # Frequently accessed, fast, expensive
    WARM = "warm"  # Occasionally accessed, moderate speed/cost
    COLD = "cold"  # Rarely accessed, slow, cheap
    ARCHIVE = "archive"  # Very rarely accessed, very slow, cheapest


class MigrationDirection(Enum):
    """Direction of data migration."""

    DEMOTE = "demote"  # Move to cheaper/slower tier
    PROMOTE = "promote"  # Move to more expensive/faster tier


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class StorageTier:
    """Definition of a storage tier.

    Attributes:
        name: Unique identifier for the tier.
        store: The store backend for this tier.
        tier_type: Type of tier (hot, warm, cold, archive).
        priority: Lower number = higher priority (checked first for reads).
        cost_per_gb: Optional cost per GB for capacity planning.
        retrieval_time_ms: Expected retrieval time in milliseconds.
        metadata: Additional tier-specific metadata.
    """

    name: str
    store: "ValidationStore[Any]"
    tier_type: TierType = TierType.HOT
    priority: int = 1
    cost_per_gb: float = 0.0
    retrieval_time_ms: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate tier configuration."""
        if self.priority < 1:
            raise ValueError("Priority must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (without store reference)."""
        return {
            "name": self.name,
            "tier_type": self.tier_type.value,
            "priority": self.priority,
            "cost_per_gb": self.cost_per_gb,
            "retrieval_time_ms": self.retrieval_time_ms,
            "metadata": self.metadata,
        }


@dataclass
class TierInfo:
    """Information about an item's tier placement.

    Attributes:
        item_id: The item identifier.
        tier_name: Current tier name.
        created_at: When item was created in this tier.
        migrated_at: When item was last migrated.
        access_count: Number of accesses.
        last_accessed: Last access time.
        size_bytes: Size in bytes.
        next_migration: When scheduled for next migration.
    """

    item_id: str
    tier_name: str
    created_at: datetime
    migrated_at: datetime | None = None
    access_count: int = 0
    last_accessed: datetime | None = None
    size_bytes: int = 0
    next_migration: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "item_id": self.item_id,
            "tier_name": self.tier_name,
            "created_at": self.created_at.isoformat(),
            "migrated_at": self.migrated_at.isoformat() if self.migrated_at else None,
            "access_count": self.access_count,
            "last_accessed": (
                self.last_accessed.isoformat() if self.last_accessed else None
            ),
            "size_bytes": self.size_bytes,
            "next_migration": (
                self.next_migration.isoformat() if self.next_migration else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TierInfo":
        """Create from dictionary."""
        return cls(
            item_id=data["item_id"],
            tier_name=data["tier_name"],
            created_at=datetime.fromisoformat(data["created_at"]),
            migrated_at=(
                datetime.fromisoformat(data["migrated_at"])
                if data.get("migrated_at")
                else None
            ),
            access_count=data.get("access_count", 0),
            last_accessed=(
                datetime.fromisoformat(data["last_accessed"])
                if data.get("last_accessed")
                else None
            ),
            size_bytes=data.get("size_bytes", 0),
            next_migration=(
                datetime.fromisoformat(data["next_migration"])
                if data.get("next_migration")
                else None
            ),
        )


@dataclass
class TieringResult:
    """Result of a tiering operation (migration run).

    Attributes:
        start_time: When operation started.
        end_time: When operation finished.
        items_scanned: Total items checked.
        items_migrated: Items moved between tiers.
        bytes_migrated: Total bytes moved.
        migrations: Details of each migration.
        errors: Errors encountered.
        dry_run: Whether this was a dry run.
    """

    start_time: datetime
    end_time: datetime | None = None
    items_scanned: int = 0
    items_migrated: int = 0
    bytes_migrated: int = 0
    migrations: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    dry_run: bool = False

    @property
    def duration_seconds(self) -> float:
        """Get operation duration in seconds."""
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "items_scanned": self.items_scanned,
            "items_migrated": self.items_migrated,
            "bytes_migrated": self.bytes_migrated,
            "migrations": self.migrations,
            "errors": self.errors,
            "dry_run": self.dry_run,
        }


@dataclass
class TieringConfig:
    """Configuration for tiered storage behavior.

    Attributes:
        policies: List of tier migration policies.
        default_tier: Name of the default tier for new items.
        enable_promotion: Whether to promote on frequent access.
        promotion_threshold: Access count to trigger promotion.
        check_interval_hours: Hours between automatic tier checks.
        batch_size: Items to process per migration batch.
        enable_parallel_migration: Whether to migrate items in parallel.
        max_parallel_migrations: Maximum concurrent migrations.
    """

    policies: list["TierPolicy"] = field(default_factory=list)
    default_tier: str = "hot"
    enable_promotion: bool = True
    promotion_threshold: int = 10
    check_interval_hours: int = 24
    batch_size: int = 100
    enable_parallel_migration: bool = False
    max_parallel_migrations: int = 4


# =============================================================================
# Abstract Base Classes
# =============================================================================


class TierPolicy(ABC):
    """Abstract base class for tier migration policies.

    Tier policies determine when items should be migrated between tiers.
    They are evaluated periodically to identify items for migration.

    Example:
        >>> class OldDataPolicy(TierPolicy):
        ...     def __init__(self, days: int):
        ...         self.days = days
        ...
        ...     def should_migrate(self, info: TierInfo) -> bool:
        ...         age = datetime.now() - info.created_at
        ...         return age.days > self.days
        ...
        ...     @property
        ...     def from_tier(self) -> str:
        ...         return "hot"
        ...
        ...     @property
        ...     def to_tier(self) -> str:
        ...         return "warm"
    """

    @abstractmethod
    def should_migrate(self, info: TierInfo) -> bool:
        """Determine if an item should be migrated.

        Args:
            info: Information about the item.

        Returns:
            True if the item should be migrated, False otherwise.
        """
        pass

    @property
    @abstractmethod
    def from_tier(self) -> str:
        """Source tier name."""
        pass

    @property
    @abstractmethod
    def to_tier(self) -> str:
        """Destination tier name."""
        pass

    @property
    def direction(self) -> MigrationDirection:
        """Migration direction (default: demote)."""
        return MigrationDirection.DEMOTE

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description."""
        pass

    def get_priority(self, info: TierInfo) -> int:
        """Get migration priority (higher = migrate first).

        Args:
            info: Information about the item.

        Returns:
            Priority score.
        """
        return 0

    def get_scheduled_time(self, info: TierInfo) -> datetime | None:
        """Get when item should be migrated.

        Args:
            info: Information about the item.

        Returns:
            Scheduled migration time, or None if not scheduled.
        """
        return None

    def to_dict(self) -> dict[str, Any]:
        """Serialize policy to dictionary."""
        return {
            "type": self.__class__.__name__,
            "from_tier": self.from_tier,
            "to_tier": self.to_tier,
            "direction": self.direction.value,
            "description": self.description,
        }


class TierMetadataStore(ABC):
    """Abstract base class for storing tier placement metadata.

    This tracks which tier each item is in and related metadata.
    """

    @abstractmethod
    def save_info(self, info: TierInfo) -> None:
        """Save tier placement info.

        Args:
            info: Tier info to save.
        """
        pass

    @abstractmethod
    def get_info(self, item_id: str) -> TierInfo | None:
        """Get tier placement info.

        Args:
            item_id: The item identifier.

        Returns:
            Tier info, or None if not found.
        """
        pass

    @abstractmethod
    def delete_info(self, item_id: str) -> bool:
        """Delete tier placement info.

        Args:
            item_id: The item identifier.

        Returns:
            True if deleted, False if not found.
        """
        pass

    @abstractmethod
    def list_by_tier(self, tier_name: str) -> list[TierInfo]:
        """List all items in a tier.

        Args:
            tier_name: The tier name.

        Returns:
            List of tier info for items in the tier.
        """
        pass

    @abstractmethod
    def update_access(self, item_id: str) -> None:
        """Update access statistics for an item.

        Args:
            item_id: The item identifier.
        """
        pass


class InMemoryTierMetadataStore(TierMetadataStore):
    """In-memory implementation of tier metadata store.

    Suitable for testing and single-process usage.
    """

    def __init__(self) -> None:
        """Initialize the store."""
        self._items: dict[str, TierInfo] = {}

    def save_info(self, info: TierInfo) -> None:
        """Save tier placement info."""
        self._items[info.item_id] = info

    def get_info(self, item_id: str) -> TierInfo | None:
        """Get tier placement info."""
        return self._items.get(item_id)

    def delete_info(self, item_id: str) -> bool:
        """Delete tier placement info."""
        if item_id in self._items:
            del self._items[item_id]
            return True
        return False

    def list_by_tier(self, tier_name: str) -> list[TierInfo]:
        """List all items in a tier."""
        return [
            info for info in self._items.values() if info.tier_name == tier_name
        ]

    def update_access(self, item_id: str) -> None:
        """Update access statistics for an item."""
        info = self._items.get(item_id)
        if info:
            info.access_count += 1
            info.last_accessed = datetime.now()
