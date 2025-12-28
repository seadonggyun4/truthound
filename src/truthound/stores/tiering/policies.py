"""Tier migration policy implementations.

This module provides various policies for determining when data
should be migrated between storage tiers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable

from truthound.stores.tiering.base import (
    MigrationDirection,
    TierInfo,
    TierPolicy,
)


class AgeBasedTierPolicy(TierPolicy):
    """Migrate items based on age.

    Items older than the specified threshold are migrated to a different tier.

    Example:
        >>> # Move items older than 7 days from hot to warm
        >>> policy = AgeBasedTierPolicy(
        ...     from_tier="hot",
        ...     to_tier="warm",
        ...     after_days=7,
        ... )
    """

    def __init__(
        self,
        from_tier: str,
        to_tier: str,
        after_days: int = 0,
        after_hours: int = 0,
        direction: MigrationDirection = MigrationDirection.DEMOTE,
    ) -> None:
        """Initialize the policy.

        Args:
            from_tier: Source tier name.
            to_tier: Destination tier name.
            after_days: Days before migration.
            after_hours: Additional hours before migration.
            direction: Migration direction.
        """
        self._from_tier = from_tier
        self._to_tier = to_tier
        self._threshold = timedelta(days=after_days, hours=after_hours)
        self._direction = direction

        if self._threshold.total_seconds() <= 0:
            raise ValueError("Threshold must be positive")

    def should_migrate(self, info: TierInfo) -> bool:
        """Check if item is old enough to migrate."""
        if info.tier_name != self._from_tier:
            return False

        age = datetime.now() - info.created_at
        return age >= self._threshold

    @property
    def from_tier(self) -> str:
        """Source tier name."""
        return self._from_tier

    @property
    def to_tier(self) -> str:
        """Destination tier name."""
        return self._to_tier

    @property
    def direction(self) -> MigrationDirection:
        """Migration direction."""
        return self._direction

    @property
    def description(self) -> str:
        """Human-readable description."""
        days = self._threshold.days
        hours = self._threshold.seconds // 3600

        time_str = ""
        if days:
            time_str = f"{days} day{'s' if days != 1 else ''}"
        if hours:
            time_str += f" {hours} hour{'s' if hours != 1 else ''}"

        return (
            f"Migrate from {self._from_tier} to {self._to_tier} "
            f"after {time_str.strip()}"
        )

    def get_scheduled_time(self, info: TierInfo) -> datetime | None:
        """Get when item will be migrated."""
        if info.tier_name != self._from_tier:
            return None
        return info.created_at + self._threshold

    def get_priority(self, info: TierInfo) -> int:
        """Older items have higher priority."""
        age = datetime.now() - info.created_at
        return int(age.total_seconds())

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        result = super().to_dict()
        result["threshold_seconds"] = self._threshold.total_seconds()
        return result


class AccessBasedTierPolicy(TierPolicy):
    """Migrate items based on access patterns.

    Items that haven't been accessed recently can be demoted.
    Frequently accessed items can be promoted.

    Example:
        >>> # Demote items not accessed in 30 days
        >>> demote = AccessBasedTierPolicy(
        ...     from_tier="hot",
        ...     to_tier="warm",
        ...     inactive_days=30,
        ... )
        >>>
        >>> # Promote frequently accessed items
        >>> promote = AccessBasedTierPolicy(
        ...     from_tier="warm",
        ...     to_tier="hot",
        ...     min_access_count=100,
        ...     access_window_days=7,
        ...     direction=MigrationDirection.PROMOTE,
        ... )
    """

    def __init__(
        self,
        from_tier: str,
        to_tier: str,
        inactive_days: int | None = None,
        min_access_count: int | None = None,
        access_window_days: int = 7,
        direction: MigrationDirection = MigrationDirection.DEMOTE,
    ) -> None:
        """Initialize the policy.

        Args:
            from_tier: Source tier name.
            to_tier: Destination tier name.
            inactive_days: Days without access to trigger demotion.
            min_access_count: Minimum accesses in window for promotion.
            access_window_days: Window for counting accesses.
            direction: Migration direction.
        """
        self._from_tier = from_tier
        self._to_tier = to_tier
        self._inactive_days = inactive_days
        self._min_access_count = min_access_count
        self._access_window_days = access_window_days
        self._direction = direction

        if inactive_days is None and min_access_count is None:
            raise ValueError(
                "Either inactive_days or min_access_count must be specified"
            )

    def should_migrate(self, info: TierInfo) -> bool:
        """Check if item should be migrated based on access patterns."""
        if info.tier_name != self._from_tier:
            return False

        now = datetime.now()

        # Check for inactivity (demotion)
        if self._inactive_days is not None:
            last_access = info.last_accessed or info.created_at
            inactive = now - last_access
            if inactive.days >= self._inactive_days:
                return True

        # Check for high activity (promotion)
        if self._min_access_count is not None:
            if info.access_count >= self._min_access_count:
                # Check if accesses are recent
                if info.last_accessed:
                    access_age = now - info.last_accessed
                    if access_age.days <= self._access_window_days:
                        return True

        return False

    @property
    def from_tier(self) -> str:
        """Source tier name."""
        return self._from_tier

    @property
    def to_tier(self) -> str:
        """Destination tier name."""
        return self._to_tier

    @property
    def direction(self) -> MigrationDirection:
        """Migration direction."""
        return self._direction

    @property
    def description(self) -> str:
        """Human-readable description."""
        conditions = []

        if self._inactive_days is not None:
            conditions.append(f"inactive for {self._inactive_days} days")

        if self._min_access_count is not None:
            conditions.append(
                f"{self._min_access_count}+ accesses in {self._access_window_days} days"
            )

        action = "Demote" if self._direction == MigrationDirection.DEMOTE else "Promote"
        return (
            f"{action} from {self._from_tier} to {self._to_tier} "
            f"when {' or '.join(conditions)}"
        )

    def get_priority(self, info: TierInfo) -> int:
        """Less active items have higher demotion priority."""
        if self._direction == MigrationDirection.DEMOTE:
            # Older last access = higher priority
            last_access = info.last_accessed or info.created_at
            return int((datetime.now() - last_access).total_seconds())
        else:
            # Higher access count = higher priority for promotion
            return info.access_count

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        result = super().to_dict()
        result.update(
            {
                "inactive_days": self._inactive_days,
                "min_access_count": self._min_access_count,
                "access_window_days": self._access_window_days,
            }
        )
        return result


class SizeBasedTierPolicy(TierPolicy):
    """Migrate items based on size.

    Large items can be moved to cheaper storage, or total tier
    size can be limited.

    Example:
        >>> # Move large items to cold storage
        >>> policy = SizeBasedTierPolicy(
        ...     from_tier="hot",
        ...     to_tier="cold",
        ...     min_size_mb=100,  # Items > 100MB
        ... )
        >>>
        >>> # Limit hot tier to 10GB
        >>> policy = SizeBasedTierPolicy(
        ...     from_tier="hot",
        ...     to_tier="warm",
        ...     tier_max_size_gb=10,
        ... )
    """

    def __init__(
        self,
        from_tier: str,
        to_tier: str,
        min_size_bytes: int = 0,
        min_size_kb: int = 0,
        min_size_mb: int = 0,
        min_size_gb: int = 0,
        tier_max_size_bytes: int = 0,
        tier_max_size_gb: int = 0,
        direction: MigrationDirection = MigrationDirection.DEMOTE,
    ) -> None:
        """Initialize the policy.

        Args:
            from_tier: Source tier name.
            to_tier: Destination tier name.
            min_size_*: Minimum item size to migrate.
            tier_max_size_*: Maximum total tier size.
            direction: Migration direction.
        """
        self._from_tier = from_tier
        self._to_tier = to_tier
        self._direction = direction

        self._min_size = (
            min_size_bytes
            + min_size_kb * 1024
            + min_size_mb * 1024 * 1024
            + min_size_gb * 1024 * 1024 * 1024
        )

        self._tier_max_size = (
            tier_max_size_bytes + tier_max_size_gb * 1024 * 1024 * 1024
        )

        # Batch state for tier-wide limits
        self._tier_current_size = 0
        self._items_in_tier: list[TierInfo] = []

    def should_migrate(self, info: TierInfo) -> bool:
        """Check if item should be migrated based on size."""
        if info.tier_name != self._from_tier:
            return False

        # Individual item size check
        if self._min_size > 0 and info.size_bytes >= self._min_size:
            return True

        # Tier-wide size limit check (requires batch context)
        if self._tier_max_size > 0:
            if self._tier_current_size > self._tier_max_size:
                # Need to migrate some items - check if this one is marked
                return info in self._items_in_tier

        return False

    def prepare_batch(self, tier_items: list[TierInfo]) -> None:
        """Prepare for batch evaluation with tier-wide context.

        Args:
            tier_items: All items in the source tier.
        """
        if self._tier_max_size <= 0:
            return

        # Calculate total tier size
        self._tier_current_size = sum(i.size_bytes for i in tier_items)

        if self._tier_current_size <= self._tier_max_size:
            self._items_in_tier = []
            return

        # Sort by priority (oldest/largest first) and mark items for migration
        excess = self._tier_current_size - self._tier_max_size
        sorted_items = sorted(tier_items, key=lambda x: x.created_at)

        migrate_size = 0
        self._items_in_tier = []
        for item in sorted_items:
            if migrate_size >= excess:
                break
            self._items_in_tier.append(item)
            migrate_size += item.size_bytes

    @property
    def from_tier(self) -> str:
        """Source tier name."""
        return self._from_tier

    @property
    def to_tier(self) -> str:
        """Destination tier name."""
        return self._to_tier

    @property
    def direction(self) -> MigrationDirection:
        """Migration direction."""
        return self._direction

    @property
    def description(self) -> str:
        """Human-readable description."""
        conditions = []

        if self._min_size > 0:
            if self._min_size >= 1024 * 1024 * 1024:
                size_str = f"{self._min_size / (1024**3):.1f} GB"
            elif self._min_size >= 1024 * 1024:
                size_str = f"{self._min_size / (1024**2):.1f} MB"
            else:
                size_str = f"{self._min_size / 1024:.1f} KB"
            conditions.append(f"item size >= {size_str}")

        if self._tier_max_size > 0:
            if self._tier_max_size >= 1024 * 1024 * 1024:
                size_str = f"{self._tier_max_size / (1024**3):.1f} GB"
            else:
                size_str = f"{self._tier_max_size / (1024**2):.1f} MB"
            conditions.append(f"tier size exceeds {size_str}")

        return (
            f"Migrate from {self._from_tier} to {self._to_tier} "
            f"when {' or '.join(conditions)}"
        )

    def get_priority(self, info: TierInfo) -> int:
        """Larger items have higher migration priority."""
        return info.size_bytes

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        result = super().to_dict()
        result.update(
            {
                "min_size_bytes": self._min_size,
                "tier_max_size_bytes": self._tier_max_size,
            }
        )
        return result


class ScheduledTierPolicy(TierPolicy):
    """Migrate items based on a schedule.

    Items are migrated at specific times or on specific days.

    Example:
        >>> # Migrate to cold storage on weekends
        >>> policy = ScheduledTierPolicy(
        ...     from_tier="warm",
        ...     to_tier="cold",
        ...     on_days=[5, 6],  # Saturday, Sunday
        ...     at_hour=2,  # 2 AM
        ... )
    """

    def __init__(
        self,
        from_tier: str,
        to_tier: str,
        on_days: list[int] | None = None,
        at_hour: int | None = None,
        min_age_days: int = 0,
        direction: MigrationDirection = MigrationDirection.DEMOTE,
    ) -> None:
        """Initialize the policy.

        Args:
            from_tier: Source tier name.
            to_tier: Destination tier name.
            on_days: Days of week to run (0=Monday, 6=Sunday).
            at_hour: Hour to run (0-23).
            min_age_days: Minimum item age to consider.
            direction: Migration direction.
        """
        self._from_tier = from_tier
        self._to_tier = to_tier
        self._on_days = on_days
        self._at_hour = at_hour
        self._min_age_days = min_age_days
        self._direction = direction

    def should_migrate(self, info: TierInfo) -> bool:
        """Check if item should be migrated based on schedule."""
        if info.tier_name != self._from_tier:
            return False

        now = datetime.now()

        # Check day constraint
        if self._on_days is not None:
            if now.weekday() not in self._on_days:
                return False

        # Check hour constraint
        if self._at_hour is not None:
            if now.hour != self._at_hour:
                return False

        # Check age constraint
        if self._min_age_days > 0:
            age = now - info.created_at
            if age.days < self._min_age_days:
                return False

        return True

    @property
    def from_tier(self) -> str:
        """Source tier name."""
        return self._from_tier

    @property
    def to_tier(self) -> str:
        """Destination tier name."""
        return self._to_tier

    @property
    def direction(self) -> MigrationDirection:
        """Migration direction."""
        return self._direction

    @property
    def description(self) -> str:
        """Human-readable description."""
        conditions = []

        if self._on_days is not None:
            day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            days = ", ".join(day_names[d] for d in self._on_days)
            conditions.append(f"on {days}")

        if self._at_hour is not None:
            conditions.append(f"at {self._at_hour:02d}:00")

        if self._min_age_days > 0:
            conditions.append(f"items older than {self._min_age_days} days")

        return (
            f"Migrate from {self._from_tier} to {self._to_tier} "
            f"{' '.join(conditions)}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        result = super().to_dict()
        result.update(
            {
                "on_days": self._on_days,
                "at_hour": self._at_hour,
                "min_age_days": self._min_age_days,
            }
        )
        return result


class CompositeTierPolicy(TierPolicy):
    """Combine multiple policies with custom logic.

    Example:
        >>> # Migrate if old AND large
        >>> policy = CompositeTierPolicy(
        ...     from_tier="hot",
        ...     to_tier="cold",
        ...     policies=[
        ...         AgeBasedTierPolicy("hot", "cold", after_days=30),
        ...         SizeBasedTierPolicy("hot", "cold", min_size_mb=100),
        ...     ],
        ...     require_all=True,
        ... )
    """

    def __init__(
        self,
        from_tier: str,
        to_tier: str,
        policies: list[TierPolicy],
        require_all: bool = True,
        direction: MigrationDirection = MigrationDirection.DEMOTE,
    ) -> None:
        """Initialize the policy.

        Args:
            from_tier: Source tier name.
            to_tier: Destination tier name.
            policies: Child policies.
            require_all: If True, all must agree. If False, any triggers migration.
            direction: Migration direction.
        """
        self._from_tier = from_tier
        self._to_tier = to_tier
        self._policies = policies
        self._require_all = require_all
        self._direction = direction

    def should_migrate(self, info: TierInfo) -> bool:
        """Check if item should be migrated based on all policies."""
        if info.tier_name != self._from_tier:
            return False

        results = [p.should_migrate(info) for p in self._policies]

        if self._require_all:
            return all(results)
        return any(results)

    def prepare_batch(self, tier_items: list[TierInfo]) -> None:
        """Prepare child policies for batch evaluation."""
        for policy in self._policies:
            if hasattr(policy, "prepare_batch"):
                policy.prepare_batch(tier_items)

    @property
    def from_tier(self) -> str:
        """Source tier name."""
        return self._from_tier

    @property
    def to_tier(self) -> str:
        """Destination tier name."""
        return self._to_tier

    @property
    def direction(self) -> MigrationDirection:
        """Migration direction."""
        return self._direction

    @property
    def description(self) -> str:
        """Human-readable description."""
        logic = "all of" if self._require_all else "any of"
        child_descs = [f"  - {p.description}" for p in self._policies]
        return (
            f"Migrate from {self._from_tier} to {self._to_tier} when {logic}:\n"
            + "\n".join(child_descs)
        )

    def get_priority(self, info: TierInfo) -> int:
        """Get combined priority from all policies."""
        return sum(p.get_priority(info) for p in self._policies)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        result = super().to_dict()
        result.update(
            {
                "policies": [p.to_dict() for p in self._policies],
                "require_all": self._require_all,
            }
        )
        return result


class CustomTierPolicy(TierPolicy):
    """Custom tier policy using a callable.

    Example:
        >>> def my_rule(info: TierInfo) -> bool:
        ...     return info.size_bytes > 1000 and info.access_count < 5
        >>>
        >>> policy = CustomTierPolicy(
        ...     from_tier="hot",
        ...     to_tier="warm",
        ...     predicate=my_rule,
        ...     description="Large but rarely accessed items",
        ... )
    """

    def __init__(
        self,
        from_tier: str,
        to_tier: str,
        predicate: Callable[[TierInfo], bool],
        description: str = "Custom policy",
        direction: MigrationDirection = MigrationDirection.DEMOTE,
    ) -> None:
        """Initialize the policy.

        Args:
            from_tier: Source tier name.
            to_tier: Destination tier name.
            predicate: Function that returns True to migrate.
            description: Human-readable description.
            direction: Migration direction.
        """
        self._from_tier = from_tier
        self._to_tier = to_tier
        self._predicate = predicate
        self._description = description
        self._direction = direction

    def should_migrate(self, info: TierInfo) -> bool:
        """Check if item should be migrated using custom predicate."""
        if info.tier_name != self._from_tier:
            return False
        return self._predicate(info)

    @property
    def from_tier(self) -> str:
        """Source tier name."""
        return self._from_tier

    @property
    def to_tier(self) -> str:
        """Destination tier name."""
        return self._to_tier

    @property
    def direction(self) -> MigrationDirection:
        """Migration direction."""
        return self._direction

    @property
    def description(self) -> str:
        """Human-readable description."""
        return self._description
