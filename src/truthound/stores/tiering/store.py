"""Tiered storage store implementation.

This module provides the main TieredStore class that manages multiple
storage tiers with automatic data migration.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Generic, TypeVar

from truthound.stores.base import (
    BaseStore,
    StoreConfig,
    StoreNotFoundError,
    StoreQuery,
    StoreReadError,
    StoreWriteError,
    ValidationStore,
)
from truthound.stores.results import ValidationResult
from truthound.stores.tiering.base import (
    InMemoryTierMetadataStore,
    StorageTier,
    TierInfo,
    TieringConfig,
    TieringResult,
    TierMetadataStore,
    TierMigrationError,
    TierNotFoundError,
)

ConfigT = TypeVar("ConfigT", bound=StoreConfig)

logger = logging.getLogger(__name__)


class TieredStore(ValidationStore[ConfigT], Generic[ConfigT]):
    """A store that manages multiple storage tiers.

    TieredStore provides automatic data placement and migration across
    multiple storage backends based on configurable policies.

    Example:
        >>> from truthound.stores import get_store
        >>> from truthound.stores.tiering import (
        ...     TieredStore,
        ...     StorageTier,
        ...     TieringConfig,
        ...     TierType,
        ...     AgeBasedTierPolicy,
        ... )
        >>>
        >>> # Define tiers
        >>> tiers = [
        ...     StorageTier(
        ...         name="hot",
        ...         store=get_store("filesystem", base_path=".truthound/hot"),
        ...         tier_type=TierType.HOT,
        ...         priority=1,
        ...     ),
        ...     StorageTier(
        ...         name="cold",
        ...         store=get_store("s3", bucket="archive-bucket"),
        ...         tier_type=TierType.COLD,
        ...         priority=2,
        ...     ),
        ... ]
        >>>
        >>> # Configure tiering
        >>> config = TieringConfig(
        ...     policies=[
        ...         AgeBasedTierPolicy("hot", "cold", after_days=30),
        ...     ],
        ... )
        >>>
        >>> # Create tiered store
        >>> store = TieredStore(tiers, config)
        >>>
        >>> # Save (goes to default tier)
        >>> store.save(result)
        >>>
        >>> # Get (searches all tiers)
        >>> result = store.get(run_id)
    """

    def __init__(
        self,
        tiers: list[StorageTier],
        config: TieringConfig | None = None,
        metadata_store: TierMetadataStore | None = None,
    ) -> None:
        """Initialize the tiered store.

        Args:
            tiers: List of storage tiers (must have at least one).
            config: Tiering configuration.
            metadata_store: Store for tier placement metadata.
        """
        if not tiers:
            raise ValueError("At least one tier must be provided")

        self._tiers = {t.name: t for t in tiers}
        self._tier_list = sorted(tiers, key=lambda t: t.priority)
        self._config = config or TieringConfig()
        self._metadata_store = metadata_store or InMemoryTierMetadataStore()
        self._initialized = False

        # Validate default tier exists
        if self._config.default_tier not in self._tiers:
            raise ValueError(
                f"Default tier '{self._config.default_tier}' not in tiers: "
                f"{list(self._tiers.keys())}"
            )

    @classmethod
    def _default_config(cls) -> ConfigT:
        """Create default configuration."""
        return StoreConfig()  # type: ignore

    def _do_initialize(self) -> None:
        """Initialize all tier stores."""
        for tier in self._tier_list:
            tier.store.initialize()

    def _get_tier(self, name: str) -> StorageTier:
        """Get a tier by name."""
        if name not in self._tiers:
            raise TierNotFoundError(name)
        return self._tiers[name]

    def _find_item_tier(self, item_id: str) -> StorageTier | None:
        """Find which tier contains an item.

        Searches tiers in priority order (fastest first).
        """
        # First check metadata
        info = self._metadata_store.get_info(item_id)
        if info and info.tier_name in self._tiers:
            tier = self._tiers[info.tier_name]
            if tier.store.exists(item_id):
                return tier

        # Fall back to searching all tiers
        for tier in self._tier_list:
            if tier.store.exists(item_id):
                return tier

        return None

    def _record_access(self, item_id: str) -> None:
        """Record an access for promotion consideration."""
        self._metadata_store.update_access(item_id)

        # Check if item should be promoted
        if self._config.enable_promotion:
            info = self._metadata_store.get_info(item_id)
            if info and info.access_count >= self._config.promotion_threshold:
                # Find higher-priority tier
                current_tier = self._tiers.get(info.tier_name)
                if current_tier:
                    for tier in self._tier_list:
                        if tier.priority < current_tier.priority:
                            # Promote to this tier
                            try:
                                self._migrate_item(
                                    item_id, info.tier_name, tier.name
                                )
                            except TierMigrationError as e:
                                logger.warning(f"Promotion failed: {e}")
                            break

    # -------------------------------------------------------------------------
    # CRUD Operations
    # -------------------------------------------------------------------------

    def save(
        self,
        item: ValidationResult,
        tier: str | None = None,
    ) -> str:
        """Save a validation result to a tier.

        Args:
            item: The validation result to save.
            tier: Target tier name (uses default if None).

        Returns:
            The run ID of the saved result.

        Raises:
            TierNotFoundError: If specified tier doesn't exist.
            StoreWriteError: If saving fails.
        """
        self.initialize()

        tier_name = tier or self._config.default_tier
        target_tier = self._get_tier(tier_name)

        # Save to tier
        item_id = target_tier.store.save(item)

        # Save metadata
        info = TierInfo(
            item_id=item_id,
            tier_name=tier_name,
            created_at=datetime.now(),
            size_bytes=len(json.dumps(item.to_dict(), default=str).encode()),
        )
        self._metadata_store.save_info(info)

        return item_id

    def get(self, item_id: str) -> ValidationResult:
        """Retrieve a validation result from any tier.

        Args:
            item_id: The run ID of the result.

        Returns:
            The validation result.

        Raises:
            StoreNotFoundError: If the result doesn't exist in any tier.
        """
        self.initialize()

        tier = self._find_item_tier(item_id)
        if not tier:
            raise StoreNotFoundError("ValidationResult", item_id)

        result = tier.store.get(item_id)

        # Record access for promotion
        self._record_access(item_id)

        return result

    def exists(self, item_id: str) -> bool:
        """Check if a validation result exists in any tier.

        Args:
            item_id: The run ID to check.

        Returns:
            True if the result exists.
        """
        self.initialize()
        return self._find_item_tier(item_id) is not None

    def delete(self, item_id: str) -> bool:
        """Delete a validation result from all tiers.

        Args:
            item_id: The run ID of the result to delete.

        Returns:
            True if something was deleted.
        """
        self.initialize()

        deleted = False

        # Delete from all tiers
        for tier in self._tier_list:
            if tier.store.delete(item_id):
                deleted = True

        # Delete metadata
        self._metadata_store.delete_info(item_id)

        return deleted

    def list_ids(self, query: StoreQuery | None = None) -> list[str]:
        """List validation result IDs matching the query.

        Combines results from all tiers, removing duplicates.
        """
        self.initialize()

        all_ids = set()
        for tier in self._tier_list:
            tier_ids = tier.store.list_ids(query)
            all_ids.update(tier_ids)

        return list(all_ids)

    def query(self, query: StoreQuery) -> list[ValidationResult]:
        """Query validation results from all tiers."""
        ids = self.list_ids(query)
        results = []

        for item_id in ids:
            try:
                result = self.get(item_id)
                results.append(result)
            except (StoreNotFoundError, StoreReadError):
                continue

        return results

    # -------------------------------------------------------------------------
    # Tier Operations
    # -------------------------------------------------------------------------

    def get_item_tier(self, item_id: str) -> str | None:
        """Get the current tier for an item.

        Args:
            item_id: The item identifier.

        Returns:
            Tier name, or None if not found.
        """
        self.initialize()

        info = self._metadata_store.get_info(item_id)
        if info:
            return info.tier_name

        tier = self._find_item_tier(item_id)
        return tier.name if tier else None

    def get_item_info(self, item_id: str) -> TierInfo | None:
        """Get tier information for an item.

        Args:
            item_id: The item identifier.

        Returns:
            Tier info, or None if not found.
        """
        self.initialize()
        return self._metadata_store.get_info(item_id)

    def move_to_tier(self, item_id: str, target_tier: str) -> bool:
        """Manually move an item to a specific tier.

        Args:
            item_id: The item identifier.
            target_tier: Name of the target tier.

        Returns:
            True if moved successfully.

        Raises:
            StoreNotFoundError: If item doesn't exist.
            TierNotFoundError: If target tier doesn't exist.
            TierMigrationError: If migration fails.
        """
        self.initialize()

        current_tier = self._find_item_tier(item_id)
        if not current_tier:
            raise StoreNotFoundError("ValidationResult", item_id)

        if current_tier.name == target_tier:
            return True  # Already in target tier

        self._get_tier(target_tier)  # Validate target exists

        self._migrate_item(item_id, current_tier.name, target_tier)
        return True

    def _migrate_item(
        self,
        item_id: str,
        from_tier: str,
        to_tier: str,
    ) -> None:
        """Migrate an item between tiers.

        Args:
            item_id: The item identifier.
            from_tier: Source tier name.
            to_tier: Destination tier name.

        Raises:
            TierMigrationError: If migration fails.
        """
        source = self._get_tier(from_tier)
        target = self._get_tier(to_tier)

        try:
            # Read from source
            item = source.store.get(item_id)

            # Write to target
            target.store.save(item)

            # Delete from source
            source.store.delete(item_id)

            # Update metadata
            info = self._metadata_store.get_info(item_id)
            if info:
                info.tier_name = to_tier
                info.migrated_at = datetime.now()
                self._metadata_store.save_info(info)
            else:
                new_info = TierInfo(
                    item_id=item_id,
                    tier_name=to_tier,
                    created_at=datetime.now(),
                    migrated_at=datetime.now(),
                )
                self._metadata_store.save_info(new_info)

            logger.info(f"Migrated {item_id} from {from_tier} to {to_tier}")

        except Exception as e:
            raise TierMigrationError(item_id, from_tier, to_tier, str(e))

    def run_tiering(self, dry_run: bool = False) -> TieringResult:
        """Run tier migration based on policies.

        Args:
            dry_run: If True, only report what would be migrated.

        Returns:
            Result of the tiering operation.
        """
        self.initialize()

        start_time = datetime.now()
        result = TieringResult(start_time=start_time, dry_run=dry_run)

        try:
            # Process each policy
            for policy in self._config.policies:
                from_tier = self._get_tier(policy.from_tier)

                # Get all items in source tier
                tier_items = self._metadata_store.list_by_tier(from_tier.name)
                result.items_scanned += len(tier_items)

                # Prepare batch context if needed
                if hasattr(policy, "prepare_batch"):
                    policy.prepare_batch(tier_items)

                # Evaluate each item
                for info in tier_items:
                    if policy.should_migrate(info):
                        migration = {
                            "item_id": info.item_id,
                            "from_tier": policy.from_tier,
                            "to_tier": policy.to_tier,
                            "size_bytes": info.size_bytes,
                        }

                        if dry_run:
                            result.migrations.append(migration)
                            result.items_migrated += 1
                            result.bytes_migrated += info.size_bytes
                        else:
                            try:
                                self._migrate_item(
                                    info.item_id,
                                    policy.from_tier,
                                    policy.to_tier,
                                )
                                result.migrations.append(migration)
                                result.items_migrated += 1
                                result.bytes_migrated += info.size_bytes
                            except TierMigrationError as e:
                                result.errors.append(str(e))

        except Exception as e:
            result.errors.append(f"Tiering failed: {str(e)}")
            logger.error(f"Tiering operation failed: {e}")

        result.end_time = datetime.now()
        return result

    def get_tier_stats(self) -> dict[str, Any]:
        """Get statistics for all tiers.

        Returns:
            Dictionary with tier statistics.
        """
        self.initialize()

        stats = {}
        for tier in self._tier_list:
            tier_items = self._metadata_store.list_by_tier(tier.name)
            total_size = sum(i.size_bytes for i in tier_items)
            total_accesses = sum(i.access_count for i in tier_items)

            stats[tier.name] = {
                "type": tier.tier_type.value,
                "priority": tier.priority,
                "item_count": len(tier_items),
                "total_size_bytes": total_size,
                "total_accesses": total_accesses,
                "cost_per_gb": tier.cost_per_gb,
                "estimated_monthly_cost": (
                    total_size / (1024**3) * tier.cost_per_gb
                ),
            }

        return stats

    def list_tiers(self) -> list[dict[str, Any]]:
        """List all configured tiers.

        Returns:
            List of tier information dictionaries.
        """
        return [tier.to_dict() for tier in self._tier_list]

    def close(self) -> None:
        """Close all tier stores."""
        for tier in self._tier_list:
            tier.store.close()

    @property
    def default_tier(self) -> str:
        """Get the default tier name."""
        return self._config.default_tier

    @property
    def tiers(self) -> dict[str, StorageTier]:
        """Get all tiers."""
        return self._tiers.copy()
