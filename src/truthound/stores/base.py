"""Base classes and interfaces for validation stores.

This module defines the abstract base classes and protocols that all store
implementations must follow. The design follows the Repository pattern with
a focus on extensibility and testability.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Iterator,
    Protocol,
    TypeVar,
    runtime_checkable,
)

if TYPE_CHECKING:
    from truthound.stores.results import ValidationResult
    from truthound.stores.expectations import ExpectationSuite


# =============================================================================
# Exceptions
# =============================================================================


class StoreError(Exception):
    """Base exception for all store-related errors."""

    pass


class StoreNotFoundError(StoreError):
    """Raised when a requested item is not found in the store."""

    def __init__(self, item_type: str, identifier: str) -> None:
        self.item_type = item_type
        self.identifier = identifier
        super().__init__(f"{item_type} not found: {identifier}")


class StoreConnectionError(StoreError):
    """Raised when connection to store backend fails."""

    def __init__(self, backend: str, message: str) -> None:
        self.backend = backend
        super().__init__(f"Failed to connect to {backend}: {message}")


class StoreWriteError(StoreError):
    """Raised when writing to store fails."""

    pass


class StoreReadError(StoreError):
    """Raised when reading from store fails."""

    pass


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class StoreConfig:
    """Base configuration for all stores.

    Subclasses can extend this with backend-specific options.

    Attributes:
        namespace: Optional namespace to isolate different projects/environments.
        prefix: Path prefix for organizing stored data.
        serialization_format: Format for serializing data ("json", "yaml", "pickle").
        compression: Optional compression ("gzip", "zstd", None).
        metadata: Additional metadata to include with stored items.
    """

    namespace: str = "default"
    prefix: str = ""
    serialization_format: str = "json"
    compression: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_full_prefix(self) -> str:
        """Get the full path prefix including namespace."""
        parts = [p for p in [self.namespace, self.prefix] if p]
        return "/".join(parts)


# =============================================================================
# Protocols (Structural Typing)
# =============================================================================


@runtime_checkable
class Serializable(Protocol):
    """Protocol for objects that can be serialized to/from dict."""

    def to_dict(self) -> dict[str, Any]: ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Serializable": ...


@runtime_checkable
class Identifiable(Protocol):
    """Protocol for objects with a unique identifier."""

    @property
    def id(self) -> str: ...


# =============================================================================
# Type Variables
# =============================================================================

T = TypeVar("T", bound=Serializable)
ConfigT = TypeVar("ConfigT", bound=StoreConfig)


# =============================================================================
# Query and Filter Classes
# =============================================================================


@dataclass
class StoreQuery:
    """Query parameters for filtering stored items.

    Attributes:
        data_asset: Filter by data asset name (exact or pattern).
        start_time: Filter results after this time.
        end_time: Filter results before this time.
        status: Filter by result status ("success", "failure", "error").
        tags: Filter by tags (all must match).
        limit: Maximum number of results to return.
        offset: Number of results to skip (for pagination).
        order_by: Field to order by ("run_time", "data_asset", "status").
        ascending: Sort order (True for ascending, False for descending).
    """

    data_asset: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    status: str | None = None
    tags: dict[str, str] | None = None
    limit: int | None = None
    offset: int = 0
    order_by: str = "run_time"
    ascending: bool = False

    def matches(self, item: dict[str, Any]) -> bool:
        """Check if an item matches this query's filters."""
        if self.data_asset and item.get("data_asset") != self.data_asset:
            return False

        if self.status and item.get("status") != self.status:
            return False

        run_time = item.get("run_time")
        if run_time:
            if isinstance(run_time, str):
                run_time = datetime.fromisoformat(run_time)
            if self.start_time and run_time < self.start_time:
                return False
            if self.end_time and run_time > self.end_time:
                return False

        if self.tags:
            item_tags = item.get("tags", {})
            for key, value in self.tags.items():
                if item_tags.get(key) != value:
                    return False

        return True


# =============================================================================
# Abstract Base Store
# =============================================================================


class BaseStore(ABC, Generic[T, ConfigT]):
    """Abstract base class for all validation stores.

    This class defines the interface that all store implementations must follow.
    It uses generics to allow type-safe operations with different stored types.

    Type Parameters:
        T: The type of objects being stored (must be Serializable).
        ConfigT: The configuration type for this store.

    Example:
        >>> class MyStore(BaseStore[ValidationResult, MyStoreConfig]):
        ...     def save(self, item: ValidationResult) -> str:
        ...         # Implementation
        ...         pass
    """

    def __init__(self, config: ConfigT | None = None) -> None:
        """Initialize the store with optional configuration.

        Args:
            config: Store configuration. If None, uses default configuration.
        """
        self._config = config or self._default_config()
        self._initialized = False

    @classmethod
    @abstractmethod
    def _default_config(cls) -> ConfigT:
        """Create default configuration for this store type."""
        pass

    @property
    def config(self) -> ConfigT:
        """Get the store configuration."""
        return self._config

    # -------------------------------------------------------------------------
    # Lifecycle Methods
    # -------------------------------------------------------------------------

    def initialize(self) -> None:
        """Initialize the store (create directories, connect to database, etc.).

        This method is called automatically on first use, but can be called
        explicitly for early initialization or connection testing.
        """
        if not self._initialized:
            self._do_initialize()
            self._initialized = True

    @abstractmethod
    def _do_initialize(self) -> None:
        """Perform actual initialization. Override in subclasses."""
        pass

    def close(self) -> None:
        """Close any open connections or resources.

        Override in subclasses that need cleanup (e.g., database connections).
        """
        pass

    def __enter__(self) -> "BaseStore[T, ConfigT]":
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()

    # -------------------------------------------------------------------------
    # CRUD Operations
    # -------------------------------------------------------------------------

    @abstractmethod
    def save(self, item: T) -> str:
        """Save an item to the store.

        Args:
            item: The item to save.

        Returns:
            The unique identifier for the saved item.

        Raises:
            StoreWriteError: If saving fails.
        """
        pass

    @abstractmethod
    def get(self, item_id: str) -> T:
        """Retrieve an item by its identifier.

        Args:
            item_id: The unique identifier of the item.

        Returns:
            The retrieved item.

        Raises:
            StoreNotFoundError: If the item doesn't exist.
            StoreReadError: If reading fails.
        """
        pass

    @abstractmethod
    def exists(self, item_id: str) -> bool:
        """Check if an item exists in the store.

        Args:
            item_id: The unique identifier to check.

        Returns:
            True if the item exists, False otherwise.
        """
        pass

    @abstractmethod
    def delete(self, item_id: str) -> bool:
        """Delete an item from the store.

        Args:
            item_id: The unique identifier of the item to delete.

        Returns:
            True if the item was deleted, False if it didn't exist.

        Raises:
            StoreWriteError: If deletion fails.
        """
        pass

    # -------------------------------------------------------------------------
    # Query Operations
    # -------------------------------------------------------------------------

    @abstractmethod
    def list_ids(self, query: StoreQuery | None = None) -> list[str]:
        """List item identifiers matching the query.

        Args:
            query: Optional query to filter results.

        Returns:
            List of matching item identifiers.
        """
        pass

    @abstractmethod
    def query(self, query: StoreQuery) -> list[T]:
        """Query items matching the given criteria.

        Args:
            query: Query parameters for filtering.

        Returns:
            List of matching items.
        """
        pass

    def iter_query(self, query: StoreQuery, batch_size: int = 100) -> Iterator[T]:
        """Iterate over items matching the query in batches.

        This is more memory-efficient for large result sets.

        Args:
            query: Query parameters for filtering.
            batch_size: Number of items to fetch per batch.

        Yields:
            Items matching the query.
        """
        offset = query.offset
        while True:
            batch_query = StoreQuery(
                data_asset=query.data_asset,
                start_time=query.start_time,
                end_time=query.end_time,
                status=query.status,
                tags=query.tags,
                limit=batch_size,
                offset=offset,
                order_by=query.order_by,
                ascending=query.ascending,
            )
            batch = self.query(batch_query)
            if not batch:
                break
            yield from batch
            offset += len(batch)
            if query.limit and offset >= query.offset + query.limit:
                break

    # -------------------------------------------------------------------------
    # Convenience Methods
    # -------------------------------------------------------------------------

    def get_latest(self, data_asset: str) -> T | None:
        """Get the most recent item for a data asset.

        Args:
            data_asset: The data asset to get the latest result for.

        Returns:
            The latest item, or None if no items exist.
        """
        query = StoreQuery(
            data_asset=data_asset,
            limit=1,
            order_by="run_time",
            ascending=False,
        )
        results = self.query(query)
        return results[0] if results else None

    def count(self, query: StoreQuery | None = None) -> int:
        """Count items matching the query.

        Args:
            query: Optional query to filter items.

        Returns:
            Number of matching items.
        """
        return len(self.list_ids(query))

    def clear(self, query: StoreQuery | None = None) -> int:
        """Delete all items matching the query.

        Args:
            query: Optional query to filter items. If None, deletes all items.

        Returns:
            Number of items deleted.
        """
        ids = self.list_ids(query)
        deleted = 0
        for item_id in ids:
            if self.delete(item_id):
                deleted += 1
        return deleted


# =============================================================================
# Specialized Store Types
# =============================================================================


class ValidationStore(BaseStore["ValidationResult", ConfigT], Generic[ConfigT]):
    """Store specialized for validation results.

    Provides additional methods specific to validation result storage.
    """

    def list_runs(self, data_asset: str) -> list[str]:
        """List all run IDs for a data asset.

        Args:
            data_asset: The data asset to list runs for.

        Returns:
            List of run IDs, ordered by time (newest first).
        """
        query = StoreQuery(
            data_asset=data_asset,
            order_by="run_time",
            ascending=False,
        )
        return self.list_ids(query)

    def get_history(
        self,
        data_asset: str,
        limit: int = 10,
    ) -> list["ValidationResult"]:
        """Get validation history for a data asset.

        Args:
            data_asset: The data asset to get history for.
            limit: Maximum number of results to return.

        Returns:
            List of validation results, ordered by time (newest first).
        """
        query = StoreQuery(
            data_asset=data_asset,
            limit=limit,
            order_by="run_time",
            ascending=False,
        )
        return self.query(query)

    def get_failures(
        self,
        data_asset: str | None = None,
        limit: int = 100,
    ) -> list["ValidationResult"]:
        """Get failed validation results.

        Args:
            data_asset: Optional data asset to filter by.
            limit: Maximum number of results to return.

        Returns:
            List of failed validation results.
        """
        query = StoreQuery(
            data_asset=data_asset,
            status="failure",
            limit=limit,
            order_by="run_time",
            ascending=False,
        )
        return self.query(query)


class ExpectationStore(BaseStore["ExpectationSuite", ConfigT], Generic[ConfigT]):
    """Store specialized for expectation suites.

    Provides additional methods specific to expectation storage.
    """

    def list_suites(self, data_asset: str | None = None) -> list[str]:
        """List all expectation suite names.

        Args:
            data_asset: Optional data asset to filter by.

        Returns:
            List of suite names.
        """
        query = StoreQuery(data_asset=data_asset) if data_asset else None
        return self.list_ids(query)

    def get_suite(self, suite_name: str) -> "ExpectationSuite":
        """Get an expectation suite by name.

        Args:
            suite_name: The name of the suite to retrieve.

        Returns:
            The expectation suite.

        Raises:
            StoreNotFoundError: If the suite doesn't exist.
        """
        return self.get(suite_name)

    def save_suite(self, suite: "ExpectationSuite") -> str:
        """Save an expectation suite.

        Args:
            suite: The suite to save.

        Returns:
            The suite name.
        """
        return self.save(suite)
