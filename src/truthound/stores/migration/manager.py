"""Migration manager for coordinating schema migrations.

This module provides a manager for planning and executing
schema migrations across stores.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from truthound.stores.migration.base import (
    MigrationConfig,
    MigrationResult,
    SchemaVersion,
)
from truthound.stores.migration.registry import MigrationRegistry
from truthound.stores.migration.store import MigratableStore

logger = logging.getLogger(__name__)


class MigrationManager:
    """Manager for coordinating schema migrations.

    The MigrationManager provides a high-level interface for planning
    and executing migrations across one or more stores.

    Example:
        >>> from truthound.stores.migration import MigrationManager
        >>>
        >>> # Create manager
        >>> manager = MigrationManager()
        >>>
        >>> # Add stores
        >>> manager.add_store("results", results_store)
        >>> manager.add_store("expectations", expectations_store)
        >>>
        >>> # Plan migration
        >>> plan = manager.plan_migration("2.0.0")
        >>>
        >>> # Execute migration
        >>> results = manager.execute_migration("2.0.0")
    """

    def __init__(
        self,
        registry: MigrationRegistry | None = None,
    ) -> None:
        """Initialize the manager.

        Args:
            registry: Shared migration registry for all stores.
        """
        self._registry = registry
        self._stores: dict[str, MigratableStore[Any]] = {}
        self._migration_history: list[dict[str, Any]] = []

    def add_store(
        self,
        name: str,
        store: MigratableStore[Any],
    ) -> None:
        """Add a store to manage.

        Args:
            name: Name to identify the store.
            store: The migratable store.
        """
        self._stores[name] = store

    def remove_store(self, name: str) -> bool:
        """Remove a store from management.

        Args:
            name: Name of the store.

        Returns:
            True if removed, False if not found.
        """
        if name in self._stores:
            del self._stores[name]
            return True
        return False

    def plan_migration(
        self,
        target_version: str,
        store_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Plan a migration to a target version.

        This analyzes all stores and creates a migration plan
        without actually executing migrations.

        Args:
            target_version: Target schema version.
            store_names: Specific stores to migrate (all if None).

        Returns:
            Migration plan dictionary.
        """
        stores_to_check = store_names or list(self._stores.keys())
        plan: dict[str, Any] = {
            "target_version": target_version,
            "timestamp": datetime.now().isoformat(),
            "stores": {},
            "total_items": 0,
            "items_need_migration": 0,
            "estimated_duration_seconds": 0,
        }

        for name in stores_to_check:
            if name not in self._stores:
                plan["stores"][name] = {"error": "Store not found"}
                continue

            store = self._stores[name]
            status = store.get_migration_status()

            # Estimate based on current stats
            items_to_migrate = status.get("needs_migration_estimated", 0)

            store_plan = {
                "current_version": status.get("current_version"),
                "total_items": status.get("total_items", 0),
                "items_need_migration": items_to_migrate,
                "version_distribution": status.get("version_distribution", {}),
            }

            # Check if migration is possible
            registry = store.registry
            try:
                for from_v in status.get("version_distribution", {}).keys():
                    if from_v != target_version:
                        path = registry.find_path(from_v, target_version)
                        store_plan["migration_steps"] = [
                            f"{m.from_version} -> {m.to_version}" for m in path
                        ]
            except Exception as e:
                store_plan["error"] = str(e)

            plan["stores"][name] = store_plan
            plan["total_items"] += status.get("total_items", 0)
            plan["items_need_migration"] += items_to_migrate

        # Rough estimate: 10 items/second
        plan["estimated_duration_seconds"] = plan["items_need_migration"] / 10

        return plan

    def execute_migration(
        self,
        target_version: str,
        store_names: list[str] | None = None,
        dry_run: bool = False,
        batch_size: int = 100,
    ) -> dict[str, MigrationResult]:
        """Execute migration to target version.

        Args:
            target_version: Target schema version.
            store_names: Specific stores to migrate (all if None).
            dry_run: If True, only simulate migration.
            batch_size: Items to process per batch.

        Returns:
            Dictionary mapping store name to migration result.
        """
        stores_to_migrate = store_names or list(self._stores.keys())
        results: dict[str, MigrationResult] = {}

        logger.info(f"Starting migration to version {target_version}")

        for name in stores_to_migrate:
            if name not in self._stores:
                results[name] = MigrationResult(
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    to_version=target_version,
                    errors=[f"Store not found: {name}"],
                )
                continue

            store = self._stores[name]

            # Update store's target version
            store._current_version = SchemaVersion.parse(target_version)

            logger.info(f"Migrating store: {name}")

            try:
                result = store.migrate_all(dry_run=dry_run, batch_size=batch_size)
                results[name] = result

                logger.info(
                    f"Store {name}: {result.items_migrated} migrated, "
                    f"{result.items_failed} failed"
                )

            except Exception as e:
                results[name] = MigrationResult(
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    to_version=target_version,
                    errors=[str(e)],
                )
                logger.error(f"Migration failed for store {name}: {e}")

        # Record in history
        self._migration_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "target_version": target_version,
                "dry_run": dry_run,
                "results": {
                    name: result.to_dict() for name, result in results.items()
                },
            }
        )

        return results

    def validate_migration(
        self,
        target_version: str,
        sample_size: int = 10,
    ) -> dict[str, Any]:
        """Validate migration with sample data from each store.

        Args:
            target_version: Target schema version.
            sample_size: Number of items to test per store.

        Returns:
            Validation results.
        """
        validation: dict[str, Any] = {
            "target_version": target_version,
            "timestamp": datetime.now().isoformat(),
            "stores": {},
            "overall_valid": True,
        }

        for name, store in self._stores.items():
            store_validation: dict[str, Any] = {
                "items_tested": 0,
                "items_valid": 0,
                "items_invalid": 0,
                "errors": [],
            }

            # Get sample items
            all_ids = store.list_ids()
            sample_ids = all_ids[:sample_size]

            for item_id in sample_ids:
                result = store.validate_item(item_id)
                store_validation["items_tested"] += 1

                if result.get("can_migrate", False):
                    store_validation["items_valid"] += 1
                else:
                    store_validation["items_invalid"] += 1
                    if "error" in result:
                        store_validation["errors"].append(
                            f"{item_id}: {result['error']}"
                        )

            validation["stores"][name] = store_validation

            if store_validation["items_invalid"] > 0:
                validation["overall_valid"] = False

        return validation

    def rollback_migration(
        self,
        previous_version: str,
        store_names: list[str] | None = None,
    ) -> dict[str, MigrationResult]:
        """Rollback to a previous version.

        Note: This requires reversible migrations.

        Args:
            previous_version: Version to rollback to.
            store_names: Specific stores to rollback (all if None).

        Returns:
            Dictionary mapping store name to migration result.
        """
        # Rollback is just a migration to an older version
        return self.execute_migration(
            target_version=previous_version,
            store_names=store_names,
        )

    def get_status(self) -> dict[str, Any]:
        """Get overall migration status.

        Returns:
            Status dictionary.
        """
        status: dict[str, Any] = {
            "stores": {},
            "migration_history_count": len(self._migration_history),
            "last_migration": (
                self._migration_history[-1] if self._migration_history else None
            ),
        }

        for name, store in self._stores.items():
            status["stores"][name] = store.get_migration_status()

        return status

    def get_migration_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent migration history.

        Args:
            limit: Maximum entries to return.

        Returns:
            List of migration history entries.
        """
        return self._migration_history[-limit:]

    def list_stores(self) -> list[str]:
        """List managed store names.

        Returns:
            List of store names.
        """
        return list(self._stores.keys())

    def get_store(self, name: str) -> MigratableStore[Any] | None:
        """Get a managed store by name.

        Args:
            name: Store name.

        Returns:
            The store, or None if not found.
        """
        return self._stores.get(name)
