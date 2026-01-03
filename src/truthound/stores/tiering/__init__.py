"""Tiered storage system for validation stores.

This module provides a tiered storage layer that automatically moves data
between different storage tiers based on access patterns, age, and policies.

Example:
    >>> from truthound.stores import get_store
    >>> from truthound.stores.tiering import (
    ...     TieredStore,
    ...     TieringConfig,
    ...     StorageTier,
    ...     TierPolicy,
    ... )
    >>>
    >>> # Define storage tiers
    >>> tiers = [
    ...     StorageTier(
    ...         name="hot",
    ...         store=get_store("filesystem", base_path=".truthound/hot"),
    ...         priority=1,  # Highest priority, fastest access
    ...     ),
    ...     StorageTier(
    ...         name="warm",
    ...         store=get_store("s3", bucket="my-bucket", storage_class="STANDARD_IA"),
    ...         priority=2,
    ...     ),
    ...     StorageTier(
    ...         name="cold",
    ...         store=get_store("s3", bucket="my-bucket", storage_class="GLACIER"),
    ...         priority=3,  # Lowest priority, cheapest
    ...     ),
    ... ]
    >>>
    >>> # Define tiering policies
    >>> policies = [
    ...     TierPolicy(
    ...         from_tier="hot",
    ...         to_tier="warm",
    ...         after_days=7,
    ...     ),
    ...     TierPolicy(
    ...         from_tier="warm",
    ...         to_tier="cold",
    ...         after_days=30,
    ...     ),
    ... ]
    >>>
    >>> # Create tiered store
    >>> store = TieredStore(tiers, TieringConfig(policies=policies))
    >>>
    >>> # Use like any other store
    >>> store.save(result)  # Saved to hot tier
    >>> store.get(run_id)  # Fetched from appropriate tier
"""

from truthound.stores.tiering.base import (
    StorageTier,
    TierInfo,
    TierType,
    TieringConfig,
    TierPolicy,
    TieringResult,
    TieringError,
    TierNotFoundError,
    TierMigrationError,
    TierAccessError,
    MigrationDirection,
    TierMetadataStore,
    InMemoryTierMetadataStore,
)
from truthound.stores.tiering.store import TieredStore
from truthound.stores.tiering.policies import (
    AgeBasedTierPolicy,
    AccessBasedTierPolicy,
    SizeBasedTierPolicy,
    ScheduledTierPolicy,
    CompositeTierPolicy,
)
from truthound.stores.tiering.manager import TieringManager

__all__ = [
    # Base types
    "StorageTier",
    "TierInfo",
    "TierType",
    "TieringConfig",
    "TierPolicy",
    "TieringResult",
    "MigrationDirection",
    # Errors
    "TieringError",
    "TierNotFoundError",
    "TierMigrationError",
    "TierAccessError",
    # Metadata stores
    "TierMetadataStore",
    "InMemoryTierMetadataStore",
    # Policies
    "AgeBasedTierPolicy",
    "AccessBasedTierPolicy",
    "SizeBasedTierPolicy",
    "ScheduledTierPolicy",
    "CompositeTierPolicy",
    # Main store
    "TieredStore",
    # Manager
    "TieringManager",
]
