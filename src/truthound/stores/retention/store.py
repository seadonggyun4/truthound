"""Retention-enabled store wrapper implementation.

This module provides the main RetentionStore class that wraps any
BaseStore with automatic retention and cleanup capabilities.
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
from truthound.stores.retention.base import (
    ItemMetadata,
    PolicyEvaluator,
    RetentionAction,
    RetentionConfig,
    RetentionResult,
)
from truthound.stores.retention.policies import (
    CountBasedPolicy,
    SizeBasedPolicy,
)

ConfigT = TypeVar("ConfigT", bound=StoreConfig)

logger = logging.getLogger(__name__)


class RetentionStore(ValidationStore[ConfigT], Generic[ConfigT]):
    """A store wrapper that adds retention policies to any base store.

    This class wraps an existing store and adds automatic cleanup based
    on configurable retention policies.

    Example:
        >>> from truthound.stores import get_store
        >>> from truthound.stores.retention import (
        ...     RetentionStore,
        ...     RetentionConfig,
        ...     TimeBasedPolicy,
        ...     CountBasedPolicy,
        ... )
        >>>
        >>> # Create base store
        >>> base = get_store("filesystem", base_path=".truthound/results")
        >>>
        >>> # Configure retention
        >>> config = RetentionConfig(
        ...     policies=[
        ...         TimeBasedPolicy(max_age_days=30),
        ...         CountBasedPolicy(max_count=1000),
        ...     ],
        ...     schedule=RetentionSchedule(interval_hours=24),
        ... )
        >>>
        >>> # Create retention-enabled store
        >>> store = RetentionStore(base, config)
        >>>
        >>> # Manual cleanup
        >>> result = store.run_cleanup()
        >>> print(f"Deleted {result.items_deleted} items")
    """

    def __init__(
        self,
        base_store: ValidationStore[Any],
        config: RetentionConfig | None = None,
        archive_store: ValidationStore[Any] | None = None,
    ) -> None:
        """Initialize the retention store.

        Args:
            base_store: The underlying store to wrap.
            config: Retention configuration.
            archive_store: Optional store for archived items.
        """
        self._base_store = base_store
        self._retention_config = config or RetentionConfig()
        self._archive_store = archive_store
        self._evaluator = PolicyEvaluator(
            self._retention_config.policies,
            self._retention_config.mode,
        )
        self._last_cleanup: datetime | None = None
        self._cleanup_history: list[RetentionResult] = []
        self._initialized = False

    @classmethod
    def _default_config(cls) -> ConfigT:
        """Create default configuration."""
        return StoreConfig()  # type: ignore

    def _do_initialize(self) -> None:
        """Initialize the store."""
        self._base_store.initialize()
        if self._archive_store:
            self._archive_store.initialize()

    # -------------------------------------------------------------------------
    # Delegated CRUD Operations
    # -------------------------------------------------------------------------

    def save(self, item: ValidationResult) -> str:
        """Save a validation result.

        Args:
            item: The validation result to save.

        Returns:
            The run ID of the saved result.
        """
        self.initialize()
        result = self._base_store.save(item)

        # Check if cleanup is needed based on schedule
        if self._should_auto_cleanup():
            try:
                self.run_cleanup()
            except Exception as e:
                logger.warning(f"Auto cleanup failed: {e}")

        return result

    def get(self, item_id: str) -> ValidationResult:
        """Retrieve a validation result."""
        self.initialize()
        return self._base_store.get(item_id)

    def exists(self, item_id: str) -> bool:
        """Check if a validation result exists."""
        self.initialize()
        return self._base_store.exists(item_id)

    def delete(self, item_id: str) -> bool:
        """Delete a validation result."""
        self.initialize()
        return self._base_store.delete(item_id)

    def list_ids(self, query: StoreQuery | None = None) -> list[str]:
        """List validation result IDs matching the query."""
        self.initialize()
        return self._base_store.list_ids(query)

    def query(self, query: StoreQuery) -> list[ValidationResult]:
        """Query validation results."""
        self.initialize()
        return self._base_store.query(query)

    # -------------------------------------------------------------------------
    # Retention Operations
    # -------------------------------------------------------------------------

    def _should_auto_cleanup(self) -> bool:
        """Check if automatic cleanup should run."""
        if not self._retention_config.schedule.enabled:
            return False

        if self._last_cleanup is None:
            return True

        elapsed = datetime.now() - self._last_cleanup
        interval_hours = self._retention_config.schedule.interval_hours
        return elapsed.total_seconds() >= interval_hours * 3600

    def _get_item_metadata(self, item_id: str) -> ItemMetadata | None:
        """Get metadata for an item without loading full content."""
        try:
            result = self._base_store.get(item_id)
            return ItemMetadata(
                item_id=result.run_id,
                data_asset=result.data_asset,
                created_at=result.run_time,
                size_bytes=len(json.dumps(result.to_dict(), default=str).encode()),
                status=result.status.value,
                tags=result.tags,
            )
        except (StoreNotFoundError, StoreReadError):
            return None

    def _is_excluded(self, metadata: ItemMetadata) -> bool:
        """Check if an item is excluded from cleanup."""
        # Check excluded assets
        if metadata.data_asset in self._retention_config.excluded_assets:
            return True

        # Check excluded tags
        for tag_key, tag_value in self._retention_config.excluded_tags.items():
            if metadata.tags.get(tag_key) == tag_value:
                return True

        return False

    def _prepare_policies(self, items: list[ItemMetadata]) -> None:
        """Prepare policies that need batch context."""
        for policy in self._retention_config.policies:
            if hasattr(policy, "prepare_batch"):
                policy.prepare_batch(items)

    def run_cleanup(
        self,
        dry_run: bool | None = None,
        batch_size: int | None = None,
    ) -> RetentionResult:
        """Run retention cleanup.

        Args:
            dry_run: If True, only report what would be deleted.
                     Uses config value if None.
            batch_size: Items to process per batch.
                       Uses config value if None.

        Returns:
            Result of the cleanup operation.
        """
        self.initialize()

        start_time = datetime.now()
        dry_run = dry_run if dry_run is not None else self._retention_config.dry_run
        batch_size = batch_size or self._retention_config.schedule.batch_size

        result = RetentionResult(
            start_time=start_time,
            dry_run=dry_run,
        )

        try:
            # Get all item IDs
            all_ids = self._base_store.list_ids()
            result.items_scanned = len(all_ids)

            # Collect metadata for all items
            all_metadata: list[ItemMetadata] = []
            for item_id in all_ids:
                metadata = self._get_item_metadata(item_id)
                if metadata:
                    all_metadata.append(metadata)

            # Prepare policies that need batch context
            self._prepare_policies(all_metadata)

            # Track latest item per asset for preserve_latest
            latest_per_asset: dict[str, str] = {}
            if self._retention_config.preserve_latest:
                for metadata in all_metadata:
                    asset = metadata.data_asset
                    if asset not in latest_per_asset:
                        latest_per_asset[asset] = metadata.item_id
                    else:
                        existing = next(
                            m
                            for m in all_metadata
                            if m.item_id == latest_per_asset[asset]
                        )
                        if metadata.created_at > existing.created_at:
                            latest_per_asset[asset] = metadata.item_id

            # Evaluate each item
            items_to_process: list[tuple[ItemMetadata, RetentionAction]] = []

            for metadata in all_metadata:
                # Check exclusions
                if self._is_excluded(metadata):
                    result.items_excluded += 1
                    continue

                # Check preserve_latest
                if (
                    self._retention_config.preserve_latest
                    and metadata.item_id == latest_per_asset.get(metadata.data_asset)
                ):
                    result.items_preserved += 1
                    continue

                # Evaluate policies
                should_retain, triggered_policies = self._evaluator.evaluate(metadata)

                if should_retain:
                    result.items_preserved += 1
                else:
                    # Determine action from first triggered policy
                    action = (
                        triggered_policies[0].action
                        if triggered_policies
                        else self._retention_config.default_action
                    )
                    items_to_process.append((metadata, action))

            # Process items in batches
            for i in range(0, len(items_to_process), batch_size):
                batch = items_to_process[i : i + batch_size]

                for metadata, action in batch:
                    try:
                        if dry_run:
                            # Just count
                            if action == RetentionAction.DELETE:
                                result.items_deleted += 1
                            elif action == RetentionAction.ARCHIVE:
                                result.items_archived += 1
                            elif action == RetentionAction.COMPRESS:
                                result.items_compressed += 1
                            elif action == RetentionAction.TIER_DOWN:
                                result.items_tiered += 1
                            result.bytes_freed += metadata.size_bytes
                        else:
                            # Actually perform the action
                            success = self._perform_action(metadata, action)
                            if success:
                                if action == RetentionAction.DELETE:
                                    result.items_deleted += 1
                                elif action == RetentionAction.ARCHIVE:
                                    result.items_archived += 1
                                elif action == RetentionAction.COMPRESS:
                                    result.items_compressed += 1
                                elif action == RetentionAction.TIER_DOWN:
                                    result.items_tiered += 1
                                result.bytes_freed += metadata.size_bytes

                    except Exception as e:
                        result.errors.append(
                            f"Error processing {metadata.item_id}: {str(e)}"
                        )

                # Check timeout
                elapsed = (datetime.now() - start_time).total_seconds()
                max_duration = self._retention_config.schedule.max_duration_seconds
                if elapsed >= max_duration:
                    result.errors.append(
                        f"Cleanup stopped: exceeded max duration of {max_duration}s"
                    )
                    break

        except Exception as e:
            result.errors.append(f"Cleanup failed: {str(e)}")
            logger.error(f"Retention cleanup failed: {e}")

        result.end_time = datetime.now()
        self._last_cleanup = result.end_time
        self._cleanup_history.append(result)

        # Keep only last 100 results in history
        if len(self._cleanup_history) > 100:
            self._cleanup_history = self._cleanup_history[-100:]

        return result

    def _perform_action(
        self,
        metadata: ItemMetadata,
        action: RetentionAction,
    ) -> bool:
        """Perform a retention action on an item.

        Args:
            metadata: Item metadata.
            action: Action to perform.

        Returns:
            True if action was successful.
        """
        if action == RetentionAction.DELETE:
            return self._base_store.delete(metadata.item_id)

        elif action == RetentionAction.ARCHIVE:
            if self._archive_store:
                # Move to archive
                try:
                    item = self._base_store.get(metadata.item_id)
                    self._archive_store.save(item)
                    self._base_store.delete(metadata.item_id)
                    return True
                except Exception as e:
                    logger.error(f"Archive failed for {metadata.item_id}: {e}")
                    return False
            else:
                # No archive store, just delete
                return self._base_store.delete(metadata.item_id)

        elif action == RetentionAction.COMPRESS:
            # Compression is handled at the store level
            # This is a no-op if store doesn't support it
            return True

        elif action == RetentionAction.TIER_DOWN:
            # Tier changes are cloud-specific
            # Check if store supports set_access_tier
            if hasattr(self._base_store, "set_access_tier"):
                try:
                    self._base_store.set_access_tier(metadata.item_id, "Cool")
                    return True
                except Exception as e:
                    logger.error(f"Tier change failed for {metadata.item_id}: {e}")
                    return False
            return True

        return False

    # -------------------------------------------------------------------------
    # Retention Info and Stats
    # -------------------------------------------------------------------------

    def get_retention_info(self, item_id: str) -> dict[str, Any]:
        """Get retention information for an item.

        Args:
            item_id: The item ID.

        Returns:
            Dictionary with retention info.
        """
        self.initialize()

        metadata = self._get_item_metadata(item_id)
        if not metadata:
            raise StoreNotFoundError("ValidationResult", item_id)

        # Collect all metadata for batch policies
        all_ids = self._base_store.list_ids()
        all_metadata = []
        for id_ in all_ids:
            m = self._get_item_metadata(id_)
            if m:
                all_metadata.append(m)

        self._prepare_policies(all_metadata)

        # Evaluate
        should_retain, triggered_policies = self._evaluator.evaluate(metadata)
        expiry = self._evaluator.get_earliest_expiry(metadata)

        return {
            "item_id": item_id,
            "will_be_retained": should_retain,
            "excluded": self._is_excluded(metadata),
            "expiry_time": expiry.isoformat() if expiry else None,
            "triggered_policies": [
                {"name": p.name, "description": p.description}
                for p in triggered_policies
            ],
            "metadata": {
                "data_asset": metadata.data_asset,
                "created_at": metadata.created_at.isoformat(),
                "age_days": (datetime.now() - metadata.created_at).days,
                "size_bytes": metadata.size_bytes,
                "status": metadata.status,
            },
        }

    def get_cleanup_stats(self) -> dict[str, Any]:
        """Get cleanup statistics.

        Returns:
            Dictionary with cleanup stats.
        """
        if not self._cleanup_history:
            return {
                "total_cleanups": 0,
                "last_cleanup": None,
                "total_deleted": 0,
                "total_bytes_freed": 0,
            }

        return {
            "total_cleanups": len(self._cleanup_history),
            "last_cleanup": (
                self._cleanup_history[-1].end_time.isoformat()
                if self._cleanup_history[-1].end_time
                else None
            ),
            "total_deleted": sum(r.items_deleted for r in self._cleanup_history),
            "total_archived": sum(r.items_archived for r in self._cleanup_history),
            "total_bytes_freed": sum(r.bytes_freed for r in self._cleanup_history),
            "average_items_per_cleanup": (
                sum(r.items_scanned for r in self._cleanup_history)
                / len(self._cleanup_history)
            ),
            "error_count": sum(len(r.errors) for r in self._cleanup_history),
        }

    def get_cleanup_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent cleanup history.

        Args:
            limit: Maximum results to return.

        Returns:
            List of cleanup results.
        """
        return [r.to_dict() for r in self._cleanup_history[-limit:]]

    def estimate_cleanup(self) -> dict[str, Any]:
        """Estimate what a cleanup would delete without running it.

        Returns:
            Dictionary with estimates.
        """
        result = self.run_cleanup(dry_run=True)
        return {
            "items_would_delete": result.items_deleted,
            "items_would_archive": result.items_archived,
            "bytes_would_free": result.bytes_freed,
            "items_would_preserve": result.items_preserved,
        }

    def close(self) -> None:
        """Close the store."""
        self._base_store.close()
        if self._archive_store:
            self._archive_store.close()

    @property
    def retention_config(self) -> RetentionConfig:
        """Get the retention configuration."""
        return self._retention_config
