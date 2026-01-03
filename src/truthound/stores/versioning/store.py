"""Versioned store wrapper implementation.

This module provides the main VersionedStore class that wraps any
BaseStore with versioning capabilities.
"""

from __future__ import annotations

import hashlib
import json
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
from truthound.stores.versioning.base import (
    DiffType,
    VersionConflictError,
    VersionDiff,
    VersionInfo,
    VersioningConfig,
    VersioningMode,
    VersionNotFoundError,
    VersionStore,
)
from truthound.stores.versioning.strategies import (
    GitLikeStrategy,
    IncrementalStrategy,
    SemanticStrategy,
    TimestampStrategy,
    VersioningStrategy,
    get_strategy,
)

ConfigT = TypeVar("ConfigT", bound=StoreConfig)


class InMemoryVersionStore(VersionStore):
    """In-memory version metadata store.

    Suitable for testing and single-process usage.
    For production, use a persistent store.
    """

    def __init__(self) -> None:
        """Initialize the store."""
        self._versions: dict[str, dict[int, VersionInfo]] = {}

    def save_version_info(self, info: VersionInfo) -> None:
        """Save version information."""
        if info.item_id not in self._versions:
            self._versions[info.item_id] = {}
        self._versions[info.item_id][info.version] = info

    def get_version_info(self, item_id: str, version: int) -> VersionInfo:
        """Get version information."""
        if item_id not in self._versions:
            raise VersionNotFoundError(item_id, version)
        if version not in self._versions[item_id]:
            raise VersionNotFoundError(item_id, version)
        return self._versions[item_id][version]

    def list_versions(
        self,
        item_id: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[VersionInfo]:
        """List all versions for an item."""
        if item_id not in self._versions:
            return []

        versions = sorted(
            self._versions[item_id].values(),
            key=lambda v: v.version,
            reverse=True,
        )

        if offset:
            versions = versions[offset:]
        if limit:
            versions = versions[:limit]

        return versions

    def get_latest_version(self, item_id: str) -> VersionInfo | None:
        """Get the latest version for an item."""
        if item_id not in self._versions:
            return None
        if not self._versions[item_id]:
            return None

        return max(
            self._versions[item_id].values(),
            key=lambda v: v.version,
        )

    def delete_version(self, item_id: str, version: int) -> bool:
        """Delete a specific version."""
        if item_id not in self._versions:
            return False
        if version not in self._versions[item_id]:
            return False

        del self._versions[item_id][version]
        return True

    def count_versions(self, item_id: str) -> int:
        """Count versions for an item."""
        if item_id not in self._versions:
            return 0
        return len(self._versions[item_id])


class VersionedStore(ValidationStore[ConfigT], Generic[ConfigT]):
    """A store wrapper that adds versioning to any base store.

    This class wraps an existing store and adds version tracking,
    history management, rollback, and diff capabilities.

    Example:
        >>> from truthound.stores import get_store
        >>> from truthound.stores.versioning import VersionedStore, VersioningConfig
        >>>
        >>> base = get_store("filesystem", base_path=".truthound/results")
        >>> store = VersionedStore(base, VersioningConfig(max_versions=10))
        >>>
        >>> # Save with versioning
        >>> version_id = store.save(result, message="Initial validation")
        >>>
        >>> # Get history
        >>> history = store.get_version_history(result.run_id)
        >>> for info in history:
        ...     print(f"v{info.version}: {info.message}")
        >>>
        >>> # Rollback
        >>> store.rollback(result.run_id, version=1)
    """

    def __init__(
        self,
        base_store: ValidationStore[Any],
        config: VersioningConfig | None = None,
        version_store: VersionStore | None = None,
    ) -> None:
        """Initialize the versioned store.

        Args:
            base_store: The underlying store to wrap.
            config: Versioning configuration.
            version_store: Store for version metadata (uses in-memory if None).
        """
        self._base_store = base_store
        self._versioning_config = config or VersioningConfig()
        self._version_store = version_store or InMemoryVersionStore()
        self._strategy = self._create_strategy()
        self._initialized = False

    @classmethod
    def _default_config(cls) -> ConfigT:
        """Create default configuration."""
        # This is a wrapper, so we return a minimal config
        return StoreConfig()  # type: ignore

    def _create_strategy(self) -> VersioningStrategy:
        """Create the versioning strategy based on config."""
        return get_strategy(self._versioning_config.mode.value)

    def _do_initialize(self) -> None:
        """Initialize the store."""
        self._base_store.initialize()

    def _get_versioned_id(self, item_id: str, version: int) -> str:
        """Get the versioned item ID for storage."""
        return f"{item_id}__v{version}"

    def _compute_checksum(self, data: dict[str, Any]) -> str:
        """Compute checksum for content."""
        content = json.dumps(data, sort_keys=True, default=str)
        algo = self._versioning_config.checksum_algorithm
        if algo == "sha256":
            return hashlib.sha256(content.encode()).hexdigest()
        elif algo == "sha1":
            return hashlib.sha1(content.encode()).hexdigest()
        elif algo == "md5":
            return hashlib.md5(content.encode()).hexdigest()
        else:
            return hashlib.sha256(content.encode()).hexdigest()

    def _cleanup_old_versions(self, item_id: str) -> int:
        """Remove old versions exceeding max_versions limit.

        Args:
            item_id: The item ID to clean up.

        Returns:
            Number of versions deleted.
        """
        if self._versioning_config.max_versions <= 0:
            return 0

        versions = self._version_store.list_versions(item_id)
        if len(versions) <= self._versioning_config.max_versions:
            return 0

        # Delete oldest versions
        deleted = 0
        versions_to_delete = versions[self._versioning_config.max_versions :]

        for info in versions_to_delete:
            versioned_id = self._get_versioned_id(item_id, info.version)
            try:
                self._base_store.delete(versioned_id)
            except Exception:
                pass
            self._version_store.delete_version(item_id, info.version)
            deleted += 1

        return deleted

    # -------------------------------------------------------------------------
    # Core CRUD Operations with Versioning
    # -------------------------------------------------------------------------

    def save(
        self,
        item: ValidationResult,
        message: str | None = None,
        created_by: str | None = None,
        expected_version: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Save a validation result with versioning.

        Args:
            item: The validation result to save.
            message: Optional commit-like message.
            created_by: Who is creating this version.
            expected_version: For optimistic locking - fail if current version differs.
            metadata: Additional metadata for the version.

        Returns:
            The run ID of the saved result.

        Raises:
            VersionConflictError: If expected_version doesn't match current.
            StoreWriteError: If saving fails.
        """
        self.initialize()

        item_id = item.run_id

        # Check for version conflict
        latest = self._version_store.get_latest_version(item_id)
        current_version = latest.version if latest else None

        if expected_version is not None and current_version != expected_version:
            raise VersionConflictError(
                item_id,
                expected_version,
                current_version or 0,
            )

        # Check for required message
        if self._versioning_config.require_message and not message:
            raise StoreWriteError("Commit message is required")

        # Get next version
        strategy_metadata = metadata or {}
        if self._versioning_config.track_changes and isinstance(
            self._strategy, GitLikeStrategy
        ):
            strategy_metadata["content"] = item.to_dict()

        new_version = self._strategy.get_next_version(
            item_id, current_version, strategy_metadata
        )

        # Save versioned item
        versioned_id = self._get_versioned_id(item_id, new_version)
        data = item.to_dict()
        checksum = self._compute_checksum(data)

        # Create a copy with versioned ID for storage
        versioned_data = data.copy()
        versioned_data["run_id"] = versioned_id
        versioned_item = ValidationResult.from_dict(versioned_data)

        try:
            self._base_store.save(versioned_item)
        except Exception as e:
            raise StoreWriteError(f"Failed to save version: {e}")

        # Save version info
        version_info = VersionInfo(
            version=new_version,
            item_id=item_id,
            created_at=datetime.now(),
            created_by=created_by,
            message=message,
            parent_version=current_version,
            metadata=metadata or {},
            checksum=checksum,
            size_bytes=len(json.dumps(data, default=str).encode()),
        )
        self._version_store.save_version_info(version_info)

        # Auto cleanup if enabled
        if self._versioning_config.auto_cleanup:
            self._cleanup_old_versions(item_id)

        # Also save as "current" version for easy access
        self._base_store.save(item)

        return item_id

    def get(self, item_id: str, version: int | None = None) -> ValidationResult:
        """Retrieve a validation result, optionally at a specific version.

        Args:
            item_id: The run ID of the result.
            version: Optional version to retrieve (latest if None).

        Returns:
            The validation result.

        Raises:
            StoreNotFoundError: If the result doesn't exist.
            VersionNotFoundError: If the specified version doesn't exist.
        """
        self.initialize()

        if version is None:
            # Get latest/current version
            return self._base_store.get(item_id)

        # Get specific version
        versioned_id = self._get_versioned_id(item_id, version)

        # Verify version exists in metadata
        try:
            self._version_store.get_version_info(item_id, version)
        except VersionNotFoundError:
            raise VersionNotFoundError(item_id, version)

        result = self._base_store.get(versioned_id)

        # Restore original ID
        result.run_id = item_id
        return result

    def exists(self, item_id: str, version: int | None = None) -> bool:
        """Check if a validation result exists.

        Args:
            item_id: The run ID to check.
            version: Optional version to check.

        Returns:
            True if the result exists.
        """
        self.initialize()

        if version is None:
            return self._base_store.exists(item_id)

        versioned_id = self._get_versioned_id(item_id, version)
        return self._base_store.exists(versioned_id)

    def delete(
        self,
        item_id: str,
        version: int | None = None,
        delete_all_versions: bool = False,
    ) -> bool:
        """Delete a validation result.

        Args:
            item_id: The run ID of the result to delete.
            version: Specific version to delete (all if None and delete_all_versions).
            delete_all_versions: Delete all versions if True.

        Returns:
            True if something was deleted.
        """
        self.initialize()

        if version is not None:
            # Delete specific version
            versioned_id = self._get_versioned_id(item_id, version)
            deleted = self._base_store.delete(versioned_id)
            self._version_store.delete_version(item_id, version)
            return deleted

        if delete_all_versions:
            # Delete all versions
            versions = self._version_store.list_versions(item_id)
            deleted_any = False

            for info in versions:
                versioned_id = self._get_versioned_id(item_id, info.version)
                if self._base_store.delete(versioned_id):
                    deleted_any = True
                self._version_store.delete_version(item_id, info.version)

            # Delete current version
            if self._base_store.delete(item_id):
                deleted_any = True

            return deleted_any

        # Delete only current (latest) version
        return self._base_store.delete(item_id)

    def list_ids(self, query: StoreQuery | None = None) -> list[str]:
        """List validation result IDs matching the query.

        Returns only the main IDs, not versioned IDs.
        """
        self.initialize()

        all_ids = self._base_store.list_ids(query)

        # Filter out versioned IDs (those with __v suffix)
        return [id_ for id_ in all_ids if "__v" not in id_]

    def query(self, query: StoreQuery) -> list[ValidationResult]:
        """Query validation results."""
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
    # Versioning-Specific Operations
    # -------------------------------------------------------------------------

    def get_version_history(
        self,
        item_id: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[VersionInfo]:
        """Get the version history for an item.

        Args:
            item_id: The item ID.
            limit: Maximum versions to return.
            offset: Versions to skip.

        Returns:
            List of version info, newest first.
        """
        self.initialize()
        return self._version_store.list_versions(item_id, limit, offset)

    def get_version_info(self, item_id: str, version: int) -> VersionInfo:
        """Get information about a specific version.

        Args:
            item_id: The item ID.
            version: The version number.

        Returns:
            Version information.

        Raises:
            VersionNotFoundError: If version doesn't exist.
        """
        self.initialize()
        return self._version_store.get_version_info(item_id, version)

    def get_latest_version_info(self, item_id: str) -> VersionInfo | None:
        """Get the latest version info for an item.

        Args:
            item_id: The item ID.

        Returns:
            Latest version info, or None if no versions.
        """
        self.initialize()
        return self._version_store.get_latest_version(item_id)

    def count_versions(self, item_id: str) -> int:
        """Count versions for an item.

        Args:
            item_id: The item ID.

        Returns:
            Number of versions.
        """
        self.initialize()
        return self._version_store.count_versions(item_id)

    def rollback(
        self,
        item_id: str,
        version: int,
        message: str | None = None,
        created_by: str | None = None,
    ) -> str:
        """Rollback to a previous version.

        This creates a new version that's a copy of the specified version.

        Args:
            item_id: The item ID.
            version: The version to rollback to.
            message: Optional commit message for the rollback.
            created_by: Who is performing the rollback.

        Returns:
            The new version number.

        Raises:
            VersionNotFoundError: If the specified version doesn't exist.
        """
        self.initialize()

        # Get the version to rollback to
        old_result = self.get(item_id, version)

        # Create rollback message if not provided
        if message is None:
            message = f"Rollback to version {version}"

        # Save as new version
        return self.save(
            old_result,
            message=message,
            created_by=created_by,
            metadata={"rollback_from": version},
        )

    def diff(
        self,
        item_id: str,
        version_a: int,
        version_b: int | None = None,
    ) -> VersionDiff:
        """Compare two versions.

        Args:
            item_id: The item ID.
            version_a: First version to compare.
            version_b: Second version (latest if None).

        Returns:
            Diff between the versions.

        Raises:
            VersionNotFoundError: If a specified version doesn't exist.
        """
        self.initialize()

        # Get versions
        result_a = self.get(item_id, version_a)
        result_b = self.get(item_id, version_b) if version_b else self.get(item_id)

        dict_a = result_a.to_dict()
        dict_b = result_b.to_dict()

        # Calculate diff
        changes = self._calculate_diff(dict_a, dict_b)

        # Build summary
        added = len([c for c in changes if c["type"] == DiffType.ADDED.value])
        removed = len([c for c in changes if c["type"] == DiffType.REMOVED.value])
        modified = len([c for c in changes if c["type"] == DiffType.MODIFIED.value])

        summary = f"{added} added, {removed} removed, {modified} modified"

        return VersionDiff(
            item_id=item_id,
            version_a=version_a,
            version_b=version_b or self.get_latest_version_info(item_id).version,
            changes=changes,
            summary=summary,
        )

    def _calculate_diff(
        self,
        dict_a: dict[str, Any],
        dict_b: dict[str, Any],
        path: str = "",
    ) -> list[dict[str, Any]]:
        """Calculate differences between two dictionaries."""
        changes = []

        all_keys = set(dict_a.keys()) | set(dict_b.keys())

        for key in all_keys:
            current_path = f"{path}.{key}" if path else key

            if key not in dict_a:
                changes.append(
                    {
                        "path": current_path,
                        "type": DiffType.ADDED.value,
                        "old_value": None,
                        "new_value": dict_b[key],
                    }
                )
            elif key not in dict_b:
                changes.append(
                    {
                        "path": current_path,
                        "type": DiffType.REMOVED.value,
                        "old_value": dict_a[key],
                        "new_value": None,
                    }
                )
            elif dict_a[key] != dict_b[key]:
                if isinstance(dict_a[key], dict) and isinstance(dict_b[key], dict):
                    # Recurse into nested dicts
                    changes.extend(
                        self._calculate_diff(dict_a[key], dict_b[key], current_path)
                    )
                else:
                    changes.append(
                        {
                            "path": current_path,
                            "type": DiffType.MODIFIED.value,
                            "old_value": dict_a[key],
                            "new_value": dict_b[key],
                        }
                    )

        return changes

    def close(self) -> None:
        """Close the store."""
        self._base_store.close()

    @property
    def strategy(self) -> VersioningStrategy:
        """Get the versioning strategy."""
        return self._strategy

    @property
    def versioning_config(self) -> VersioningConfig:
        """Get the versioning configuration."""
        return self._versioning_config
