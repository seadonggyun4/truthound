"""Retention policy implementations.

This module provides various retention policies for different use cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable

from truthound.stores.retention.base import (
    ItemMetadata,
    PolicyMode,
    RetentionAction,
    RetentionPolicy,
)


class TimeBasedPolicy(RetentionPolicy):
    """Retain items based on age.

    Items older than max_age will be marked for deletion.

    Example:
        >>> policy = TimeBasedPolicy(max_age_days=30)  # Keep 30 days
        >>> policy = TimeBasedPolicy(max_age_hours=72)  # Keep 72 hours
    """

    def __init__(
        self,
        max_age_days: int = 0,
        max_age_hours: int = 0,
        max_age_minutes: int = 0,
        action: RetentionAction = RetentionAction.DELETE,
    ) -> None:
        """Initialize the policy.

        Args:
            max_age_days: Maximum age in days.
            max_age_hours: Additional hours.
            max_age_minutes: Additional minutes.
            action: Action to take on expired items.
        """
        self.max_age = timedelta(
            days=max_age_days,
            hours=max_age_hours,
            minutes=max_age_minutes,
        )
        self._action = action

        if self.max_age.total_seconds() <= 0:
            raise ValueError("max_age must be positive")

    def should_retain(self, item: ItemMetadata) -> bool:
        """Check if item is younger than max_age."""
        age = datetime.now() - item.created_at
        return age <= self.max_age

    @property
    def description(self) -> str:
        """Human-readable description."""
        days = self.max_age.days
        hours = self.max_age.seconds // 3600
        minutes = (self.max_age.seconds % 3600) // 60

        parts = []
        if days:
            parts.append(f"{days} day{'s' if days != 1 else ''}")
        if hours:
            parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
        if minutes:
            parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")

        return f"Keep items younger than {', '.join(parts)}"

    @property
    def action(self) -> RetentionAction:
        """Action to take on expired items."""
        return self._action

    def get_expiry_time(self, item: ItemMetadata) -> datetime:
        """Get when the item will expire."""
        return item.created_at + self.max_age

    def get_priority(self, item: ItemMetadata) -> int:
        """Older items have higher deletion priority."""
        age = datetime.now() - item.created_at
        return int(age.total_seconds())

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        result = super().to_dict()
        result.update(
            {
                "max_age_seconds": self.max_age.total_seconds(),
            }
        )
        return result


class CountBasedPolicy(RetentionPolicy):
    """Retain only a maximum number of items.

    When the count exceeds max_count, oldest items are marked for deletion.

    Example:
        >>> policy = CountBasedPolicy(max_count=1000)  # Keep max 1000
        >>> policy = CountBasedPolicy(max_count=100, per_asset=True)  # Per data asset
    """

    def __init__(
        self,
        max_count: int,
        per_asset: bool = False,
        action: RetentionAction = RetentionAction.DELETE,
    ) -> None:
        """Initialize the policy.

        Args:
            max_count: Maximum number of items to keep.
            per_asset: Apply limit per data_asset (vs globally).
            action: Action to take on excess items.
        """
        if max_count <= 0:
            raise ValueError("max_count must be positive")

        self.max_count = max_count
        self.per_asset = per_asset
        self._action = action

        # These are set during batch evaluation
        self._current_counts: dict[str, int] = {}
        self._item_ranks: dict[str, int] = {}

    def should_retain(self, item: ItemMetadata) -> bool:
        """Check if item should be retained based on count.

        Note: This requires batch context to work properly.
        Use RetentionStore.run_cleanup() which provides proper context.
        """
        # Check if we have rank information
        if item.item_id in self._item_ranks:
            rank = self._item_ranks[item.item_id]
            return rank <= self.max_count

        # Without batch context, we can't determine count-based retention
        return True

    def prepare_batch(self, items: list[ItemMetadata]) -> None:
        """Prepare for batch evaluation.

        Args:
            items: All items to consider for retention.
        """
        # Reset state
        self._current_counts = {}
        self._item_ranks = {}

        # Group by asset if needed
        if self.per_asset:
            asset_items: dict[str, list[ItemMetadata]] = {}
            for item in items:
                if item.data_asset not in asset_items:
                    asset_items[item.data_asset] = []
                asset_items[item.data_asset].append(item)

            for asset, asset_items_list in asset_items.items():
                # Sort by creation time (newest first)
                sorted_items = sorted(
                    asset_items_list,
                    key=lambda x: x.created_at,
                    reverse=True,
                )
                for rank, item in enumerate(sorted_items, 1):
                    self._item_ranks[item.item_id] = rank
                self._current_counts[asset] = len(asset_items_list)
        else:
            # Global count
            sorted_items = sorted(items, key=lambda x: x.created_at, reverse=True)
            for rank, item in enumerate(sorted_items, 1):
                self._item_ranks[item.item_id] = rank
            self._current_counts["__global__"] = len(items)

    @property
    def description(self) -> str:
        """Human-readable description."""
        scope = "per data asset" if self.per_asset else "globally"
        return f"Keep at most {self.max_count} items {scope}"

    @property
    def action(self) -> RetentionAction:
        """Action to take on excess items."""
        return self._action

    def get_priority(self, item: ItemMetadata) -> int:
        """Lower ranks (older) have higher priority for deletion."""
        rank = self._item_ranks.get(item.item_id, 0)
        if rank > self.max_count:
            return rank
        return 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        result = super().to_dict()
        result.update(
            {
                "max_count": self.max_count,
                "per_asset": self.per_asset,
            }
        )
        return result


class SizeBasedPolicy(RetentionPolicy):
    """Retain items until total size exceeds limit.

    When total size exceeds max_size, oldest items are marked for deletion.

    Example:
        >>> policy = SizeBasedPolicy(max_size_mb=500)  # Max 500 MB
        >>> policy = SizeBasedPolicy(max_size_gb=10, per_asset=True)  # 10 GB per asset
    """

    def __init__(
        self,
        max_size_bytes: int = 0,
        max_size_kb: int = 0,
        max_size_mb: int = 0,
        max_size_gb: int = 0,
        per_asset: bool = False,
        action: RetentionAction = RetentionAction.DELETE,
    ) -> None:
        """Initialize the policy.

        Args:
            max_size_bytes: Maximum size in bytes.
            max_size_kb: Maximum size in kilobytes.
            max_size_mb: Maximum size in megabytes.
            max_size_gb: Maximum size in gigabytes.
            per_asset: Apply limit per data_asset (vs globally).
            action: Action to take on excess items.
        """
        self.max_size = (
            max_size_bytes
            + max_size_kb * 1024
            + max_size_mb * 1024 * 1024
            + max_size_gb * 1024 * 1024 * 1024
        )
        self.per_asset = per_asset
        self._action = action

        if self.max_size <= 0:
            raise ValueError("max_size must be positive")

        # Batch state
        self._cumulative_sizes: dict[str, int] = {}
        self._item_cumulative: dict[str, int] = {}

    def should_retain(self, item: ItemMetadata) -> bool:
        """Check if item should be retained based on size."""
        if item.item_id in self._item_cumulative:
            cumulative = self._item_cumulative[item.item_id]
            return cumulative <= self.max_size

        return True

    def prepare_batch(self, items: list[ItemMetadata]) -> None:
        """Prepare for batch evaluation.

        Args:
            items: All items to consider for retention.
        """
        self._cumulative_sizes = {}
        self._item_cumulative = {}

        if self.per_asset:
            asset_items: dict[str, list[ItemMetadata]] = {}
            for item in items:
                if item.data_asset not in asset_items:
                    asset_items[item.data_asset] = []
                asset_items[item.data_asset].append(item)

            for asset, asset_items_list in asset_items.items():
                # Sort by creation time (newest first)
                sorted_items = sorted(
                    asset_items_list,
                    key=lambda x: x.created_at,
                    reverse=True,
                )
                cumulative = 0
                for item in sorted_items:
                    cumulative += item.size_bytes
                    self._item_cumulative[item.item_id] = cumulative
                self._cumulative_sizes[asset] = cumulative
        else:
            sorted_items = sorted(items, key=lambda x: x.created_at, reverse=True)
            cumulative = 0
            for item in sorted_items:
                cumulative += item.size_bytes
                self._item_cumulative[item.item_id] = cumulative
            self._cumulative_sizes["__global__"] = cumulative

    @property
    def description(self) -> str:
        """Human-readable description."""
        # Format size nicely
        if self.max_size >= 1024 * 1024 * 1024:
            size_str = f"{self.max_size / (1024 * 1024 * 1024):.1f} GB"
        elif self.max_size >= 1024 * 1024:
            size_str = f"{self.max_size / (1024 * 1024):.1f} MB"
        elif self.max_size >= 1024:
            size_str = f"{self.max_size / 1024:.1f} KB"
        else:
            size_str = f"{self.max_size} bytes"

        scope = "per data asset" if self.per_asset else "total"
        return f"Keep items within {size_str} {scope}"

    @property
    def action(self) -> RetentionAction:
        """Action to take on excess items."""
        return self._action

    def get_priority(self, item: ItemMetadata) -> int:
        """Items that push over limit have higher priority."""
        cumulative = self._item_cumulative.get(item.item_id, 0)
        if cumulative > self.max_size:
            return cumulative - self.max_size
        return 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        result = super().to_dict()
        result.update(
            {
                "max_size_bytes": self.max_size,
                "per_asset": self.per_asset,
            }
        )
        return result


class StatusBasedPolicy(RetentionPolicy):
    """Retain items based on validation status.

    Example:
        >>> # Delete failed results after 7 days, keep successes 30 days
        >>> policy = StatusBasedPolicy(
        ...     status="failure",
        ...     max_age_days=7,
        ... )
    """

    def __init__(
        self,
        status: str,
        max_age_days: int | None = None,
        retain: bool = True,
        action: RetentionAction = RetentionAction.DELETE,
    ) -> None:
        """Initialize the policy.

        Args:
            status: Status to match.
            max_age_days: Optional max age for matching items.
            retain: If True, keep matching items. If False, delete them.
            action: Action to take on non-retained items.
        """
        self.status = status
        self.max_age_days = max_age_days
        self.retain = retain
        self._action = action

    def should_retain(self, item: ItemMetadata) -> bool:
        """Check if item should be retained based on status."""
        matches_status = item.status == self.status

        if not matches_status:
            # Different status, use default retention
            return True

        # Status matches
        if self.max_age_days is not None:
            age = datetime.now() - item.created_at
            if age.days > self.max_age_days:
                return not self.retain

        return self.retain

    @property
    def description(self) -> str:
        """Human-readable description."""
        action = "Keep" if self.retain else "Delete"
        age_str = ""
        if self.max_age_days is not None:
            age_str = f" older than {self.max_age_days} days"
        return f"{action} items with status '{self.status}'{age_str}"

    @property
    def action(self) -> RetentionAction:
        """Action to take on non-retained items."""
        return self._action

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        result = super().to_dict()
        result.update(
            {
                "status": self.status,
                "max_age_days": self.max_age_days,
                "retain": self.retain,
            }
        )
        return result


class TagBasedPolicy(RetentionPolicy):
    """Retain items based on tags.

    Example:
        >>> # Keep items tagged as 'production'
        >>> policy = TagBasedPolicy(required_tags={"env": "production"})
        >>>
        >>> # Delete items tagged as 'temp'
        >>> policy = TagBasedPolicy(delete_tags={"type": "temp"})
    """

    def __init__(
        self,
        required_tags: dict[str, str] | None = None,
        delete_tags: dict[str, str] | None = None,
        any_match: bool = False,
        action: RetentionAction = RetentionAction.DELETE,
    ) -> None:
        """Initialize the policy.

        Args:
            required_tags: Tags that must be present to retain.
            delete_tags: Tags that trigger deletion if present.
            any_match: If True, match any tag. If False, match all.
            action: Action to take on non-retained items.
        """
        self.required_tags = required_tags or {}
        self.delete_tags = delete_tags or {}
        self.any_match = any_match
        self._action = action

    def should_retain(self, item: ItemMetadata) -> bool:
        """Check if item should be retained based on tags."""
        # Check delete tags first
        if self.delete_tags:
            matches = []
            for key, value in self.delete_tags.items():
                matches.append(item.tags.get(key) == value)

            if self.any_match and any(matches):
                return False
            if not self.any_match and all(matches):
                return False

        # Check required tags
        if self.required_tags:
            matches = []
            for key, value in self.required_tags.items():
                matches.append(item.tags.get(key) == value)

            if self.any_match:
                return any(matches)
            return all(matches)

        return True

    @property
    def description(self) -> str:
        """Human-readable description."""
        parts = []
        if self.required_tags:
            tags_str = ", ".join(f"{k}={v}" for k, v in self.required_tags.items())
            match_str = "any of" if self.any_match else "all of"
            parts.append(f"Keep items with {match_str} tags: {tags_str}")
        if self.delete_tags:
            tags_str = ", ".join(f"{k}={v}" for k, v in self.delete_tags.items())
            match_str = "any of" if self.any_match else "all of"
            parts.append(f"Delete items with {match_str} tags: {tags_str}")
        return "; ".join(parts) or "No tag rules"

    @property
    def action(self) -> RetentionAction:
        """Action to take on non-retained items."""
        return self._action

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        result = super().to_dict()
        result.update(
            {
                "required_tags": self.required_tags,
                "delete_tags": self.delete_tags,
                "any_match": self.any_match,
            }
        )
        return result


class CompositePolicy(RetentionPolicy):
    """Combine multiple policies with custom logic.

    Example:
        >>> # Keep if (age < 30 days) AND (count < 1000)
        >>> policy = CompositePolicy(
        ...     policies=[
        ...         TimeBasedPolicy(max_age_days=30),
        ...         CountBasedPolicy(max_count=1000),
        ...     ],
        ...     mode=PolicyMode.ALL,
        ... )
    """

    def __init__(
        self,
        policies: list[RetentionPolicy],
        mode: PolicyMode = PolicyMode.ALL,
        action: RetentionAction = RetentionAction.DELETE,
    ) -> None:
        """Initialize the policy.

        Args:
            policies: Child policies to combine.
            mode: How to combine results (ALL = all must agree, ANY = any can keep).
            action: Action to take on non-retained items.
        """
        self.policies = policies
        self.mode = mode
        self._action = action

    def should_retain(self, item: ItemMetadata) -> bool:
        """Check if item should be retained based on all child policies."""
        results = [p.should_retain(item) for p in self.policies]

        if self.mode == PolicyMode.ALL:
            return all(results)
        return any(results)

    def prepare_batch(self, items: list[ItemMetadata]) -> None:
        """Prepare child policies for batch evaluation."""
        for policy in self.policies:
            if hasattr(policy, "prepare_batch"):
                policy.prepare_batch(items)

    @property
    def description(self) -> str:
        """Human-readable description."""
        mode_str = "all of" if self.mode == PolicyMode.ALL else "any of"
        child_descs = [f"  - {p.description}" for p in self.policies]
        return f"Retain if {mode_str}:\n" + "\n".join(child_descs)

    @property
    def action(self) -> RetentionAction:
        """Action to take on non-retained items."""
        return self._action

    def get_priority(self, item: ItemMetadata) -> int:
        """Get combined priority from all policies."""
        return sum(p.get_priority(item) for p in self.policies)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        result = super().to_dict()
        result.update(
            {
                "policies": [p.to_dict() for p in self.policies],
                "mode": self.mode.value,
            }
        )
        return result


class CustomPolicy(RetentionPolicy):
    """Custom retention policy using a callable.

    Example:
        >>> def keep_weekday_only(item: ItemMetadata) -> bool:
        ...     return item.created_at.weekday() < 5  # Mon-Fri
        >>>
        >>> policy = CustomPolicy(
        ...     predicate=keep_weekday_only,
        ...     description="Keep items created on weekdays",
        ... )
    """

    def __init__(
        self,
        predicate: Callable[[ItemMetadata], bool],
        description: str = "Custom policy",
        action: RetentionAction = RetentionAction.DELETE,
    ) -> None:
        """Initialize the policy.

        Args:
            predicate: Function that returns True to retain, False to delete.
            description: Human-readable description.
            action: Action to take on non-retained items.
        """
        self._predicate = predicate
        self._description = description
        self._action = action

    def should_retain(self, item: ItemMetadata) -> bool:
        """Check if item should be retained using custom predicate."""
        return self._predicate(item)

    @property
    def description(self) -> str:
        """Human-readable description."""
        return self._description

    @property
    def action(self) -> RetentionAction:
        """Action to take on non-retained items."""
        return self._action
