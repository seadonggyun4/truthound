"""In-memory store backend.

This module provides a store implementation that keeps data in memory.
Useful for testing and development. Data is not persisted between sessions.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import Any, TypeVar

from truthound.stores.base import (
    StoreConfig,
    StoreNotFoundError,
    StoreQuery,
    ValidationStore,
    ExpectationStore,
)
from truthound.stores.results import ValidationResult
from truthound.stores.expectations import ExpectationSuite

T = TypeVar("T")


@dataclass
class MemoryConfig(StoreConfig):
    """Configuration for memory store.

    Attributes:
        max_items: Maximum number of items to store (0 for unlimited).
        deep_copy: Whether to deep copy items on save/retrieve.
    """

    max_items: int = 0
    deep_copy: bool = True


class MemoryStore(ValidationStore[MemoryConfig]):
    """In-memory validation store.

    Stores validation results in memory. Data is not persisted between
    sessions. Useful for testing and development.

    Example:
        >>> store = MemoryStore()
        >>> result = ValidationResult.from_report(report, "customers.csv")
        >>> run_id = store.save(result)
        >>> assert store.exists(run_id)
    """

    def __init__(
        self,
        max_items: int = 0,
        deep_copy: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the memory store.

        Args:
            max_items: Maximum number of items to store (0 for unlimited).
            deep_copy: Whether to deep copy items on save/retrieve.
            **kwargs: Additional configuration options.
        """
        config = MemoryConfig(max_items=max_items, deep_copy=deep_copy)
        super().__init__(config)
        self._data: dict[str, dict[str, Any]] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

    @classmethod
    def _default_config(cls) -> MemoryConfig:
        """Create default configuration."""
        return MemoryConfig()

    def _do_initialize(self) -> None:
        """Initialize the store (no-op for memory store)."""
        pass

    def save(self, item: ValidationResult) -> str:
        """Save a validation result to memory.

        Args:
            item: The validation result to save.

        Returns:
            The run ID of the saved result.
        """
        self.initialize()

        item_id = item.run_id
        data = item.to_dict()

        if self._config.deep_copy:
            data = deepcopy(data)

        # Enforce max items limit (remove oldest)
        if self._config.max_items > 0 and len(self._data) >= self._config.max_items:
            # Remove oldest item
            oldest_id = min(
                self._metadata.keys(),
                key=lambda k: self._metadata[k].get("run_time", ""),
            )
            del self._data[oldest_id]
            del self._metadata[oldest_id]

        self._data[item_id] = data
        self._metadata[item_id] = {
            "data_asset": item.data_asset,
            "run_time": item.run_time.isoformat(),
            "status": item.status.value,
            "tags": item.tags,
        }

        return item_id

    def get(self, item_id: str) -> ValidationResult:
        """Retrieve a validation result by run ID.

        Args:
            item_id: The run ID of the result to retrieve.

        Returns:
            The validation result.

        Raises:
            StoreNotFoundError: If the result doesn't exist.
        """
        self.initialize()

        if item_id not in self._data:
            raise StoreNotFoundError("ValidationResult", item_id)

        data = self._data[item_id]
        if self._config.deep_copy:
            data = deepcopy(data)

        return ValidationResult.from_dict(data)

    def exists(self, item_id: str) -> bool:
        """Check if a validation result exists.

        Args:
            item_id: The run ID to check.

        Returns:
            True if the result exists.
        """
        self.initialize()
        return item_id in self._data

    def delete(self, item_id: str) -> bool:
        """Delete a validation result.

        Args:
            item_id: The run ID of the result to delete.

        Returns:
            True if the result was deleted, False if it didn't exist.
        """
        self.initialize()

        if item_id not in self._data:
            return False

        del self._data[item_id]
        del self._metadata[item_id]
        return True

    def list_ids(self, query: StoreQuery | None = None) -> list[str]:
        """List validation result IDs matching the query.

        Args:
            query: Optional query to filter results.

        Returns:
            List of matching run IDs.
        """
        self.initialize()

        if not query:
            return list(self._data.keys())

        # Filter by query
        matching_ids: list[tuple[str, datetime]] = []

        for item_id, meta in self._metadata.items():
            if query.matches(meta):
                run_time = datetime.fromisoformat(meta["run_time"])
                matching_ids.append((item_id, run_time))

        # Sort
        reverse = not query.ascending
        matching_ids.sort(key=lambda x: x[1], reverse=reverse)

        # Apply offset and limit
        ids = [item_id for item_id, _ in matching_ids]

        if query.offset:
            ids = ids[query.offset :]
        if query.limit:
            ids = ids[: query.limit]

        return ids

    def query(self, query: StoreQuery) -> list[ValidationResult]:
        """Query validation results.

        Args:
            query: Query parameters for filtering.

        Returns:
            List of matching validation results.
        """
        ids = self.list_ids(query)
        return [self.get(item_id) for item_id in ids]

    def clear_all(self) -> int:
        """Clear all stored items.

        Returns:
            Number of items cleared.
        """
        count = len(self._data)
        self._data.clear()
        self._metadata.clear()
        return count


class MemoryExpectationStore(ExpectationStore[MemoryConfig]):
    """In-memory expectation store.

    Stores expectation suites in memory. Useful for testing.
    """

    def __init__(self, deep_copy: bool = True, **kwargs: Any) -> None:
        """Initialize the memory expectation store.

        Args:
            deep_copy: Whether to deep copy items on save/retrieve.
        """
        config = MemoryConfig(deep_copy=deep_copy)
        super().__init__(config)
        self._data: dict[str, dict[str, Any]] = {}

    @classmethod
    def _default_config(cls) -> MemoryConfig:
        """Create default configuration."""
        return MemoryConfig()

    def _do_initialize(self) -> None:
        """Initialize the store (no-op for memory store)."""
        pass

    def save(self, item: ExpectationSuite) -> str:
        """Save an expectation suite to memory.

        Args:
            item: The suite to save.

        Returns:
            The suite name.
        """
        self.initialize()

        data = item.to_dict()
        if self._config.deep_copy:
            data = deepcopy(data)

        self._data[item.name] = data
        return item.name

    def get(self, item_id: str) -> ExpectationSuite:
        """Retrieve an expectation suite by name.

        Args:
            item_id: The suite name.

        Returns:
            The expectation suite.

        Raises:
            StoreNotFoundError: If the suite doesn't exist.
        """
        self.initialize()

        if item_id not in self._data:
            raise StoreNotFoundError("ExpectationSuite", item_id)

        data = self._data[item_id]
        if self._config.deep_copy:
            data = deepcopy(data)

        return ExpectationSuite.from_dict(data)

    def exists(self, item_id: str) -> bool:
        """Check if a suite exists."""
        self.initialize()
        return item_id in self._data

    def delete(self, item_id: str) -> bool:
        """Delete an expectation suite.

        Args:
            item_id: The suite name.

        Returns:
            True if deleted, False if it didn't exist.
        """
        self.initialize()

        if item_id not in self._data:
            return False

        del self._data[item_id]
        return True

    def list_ids(self, query: StoreQuery | None = None) -> list[str]:
        """List all suite names."""
        self.initialize()

        if not query or not query.data_asset:
            return list(self._data.keys())

        # Filter by data_asset
        names: list[str] = []
        for name, data in self._data.items():
            if data.get("data_asset") == query.data_asset:
                names.append(name)

        return names

    def query(self, query: StoreQuery) -> list[ExpectationSuite]:
        """Query expectation suites."""
        names = self.list_ids(query)
        suites = [self.get(name) for name in names]

        # Apply limit
        if query.limit:
            suites = suites[query.offset : query.offset + query.limit]
        elif query.offset:
            suites = suites[query.offset :]

        return suites

    def clear_all(self) -> int:
        """Clear all stored items.

        Returns:
            Number of items cleared.
        """
        count = len(self._data)
        self._data.clear()
        return count
