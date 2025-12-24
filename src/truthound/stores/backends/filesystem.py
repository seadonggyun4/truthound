"""Filesystem-based store backend.

This module provides a store implementation that persists data to the local
filesystem. It requires no external dependencies and is the default backend.
"""

from __future__ import annotations

import gzip
import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

from truthound.stores.base import (
    BaseStore,
    StoreConfig,
    StoreNotFoundError,
    StoreQuery,
    StoreReadError,
    StoreWriteError,
    ValidationStore,
    ExpectationStore,
)
from truthound.stores.results import ValidationResult
from truthound.stores.expectations import ExpectationSuite

T = TypeVar("T")


@dataclass
class FileSystemConfig(StoreConfig):
    """Configuration for filesystem store.

    Attributes:
        base_path: Base directory for storing files.
        file_extension: File extension to use (.json, .yaml).
        create_dirs: Whether to create directories if they don't exist.
        pretty_print: Whether to format JSON with indentation.
        use_compression: Whether to compress stored files.
    """

    base_path: str = ".truthound/store"
    file_extension: str = ".json"
    create_dirs: bool = True
    pretty_print: bool = True
    use_compression: bool = False

    def get_full_path(self) -> Path:
        """Get the full storage path including namespace and prefix."""
        path = Path(self.base_path)
        if self.namespace:
            path = path / self.namespace
        if self.prefix:
            path = path / self.prefix
        return path


class FileSystemStore(ValidationStore[FileSystemConfig]):
    """Filesystem-based validation store.

    Stores validation results as JSON files on the local filesystem.
    This is the default store backend and requires no external dependencies.

    Example:
        >>> store = FileSystemStore(base_path=".truthound/results")
        >>> result = ValidationResult.from_report(report, "customers.csv")
        >>> run_id = store.save(result)
        >>> retrieved = store.get(run_id)
    """

    def __init__(
        self,
        base_path: str = ".truthound/store",
        namespace: str = "default",
        prefix: str = "validations",
        compression: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the filesystem store.

        Args:
            base_path: Base directory for storing files.
            namespace: Namespace for organizing data.
            prefix: Additional path prefix.
            compression: Whether to compress stored files.
            **kwargs: Additional configuration options.
        """
        config = FileSystemConfig(
            base_path=base_path,
            namespace=namespace,
            prefix=prefix,
            use_compression=compression,
            **{k: v for k, v in kwargs.items() if hasattr(FileSystemConfig, k)},
        )
        super().__init__(config)
        self._index: dict[str, dict[str, Any]] = {}
        self._index_path: Path | None = None

    @classmethod
    def _default_config(cls) -> FileSystemConfig:
        """Create default configuration."""
        return FileSystemConfig()

    def _do_initialize(self) -> None:
        """Initialize the store directory and index."""
        path = self._config.get_full_path()

        if self._config.create_dirs:
            path.mkdir(parents=True, exist_ok=True)

        # Load or create index
        self._index_path = path / "_index.json"
        if self._index_path.exists():
            try:
                with open(self._index_path) as f:
                    self._index = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._index = {}
        else:
            self._index = {}

    def _save_index(self) -> None:
        """Persist the index to disk."""
        if self._index_path:
            with open(self._index_path, "w") as f:
                json.dump(self._index, f, indent=2, default=str)

    def _get_file_path(self, item_id: str) -> Path:
        """Get the file path for an item."""
        ext = self._config.file_extension
        if self._config.use_compression:
            ext += ".gz"
        return self._config.get_full_path() / f"{item_id}{ext}"

    def _serialize(self, data: dict[str, Any]) -> bytes:
        """Serialize data to bytes."""
        indent = 2 if self._config.pretty_print else None
        json_str = json.dumps(data, indent=indent, default=str)
        content = json_str.encode("utf-8")

        if self._config.use_compression:
            content = gzip.compress(content)

        return content

    def _deserialize(self, content: bytes) -> dict[str, Any]:
        """Deserialize bytes to data."""
        if self._config.use_compression:
            content = gzip.decompress(content)

        return json.loads(content.decode("utf-8"))

    def save(self, item: ValidationResult) -> str:
        """Save a validation result to the filesystem.

        Args:
            item: The validation result to save.

        Returns:
            The run ID of the saved result.

        Raises:
            StoreWriteError: If saving fails.
        """
        self.initialize()

        item_id = item.run_id
        file_path = self._get_file_path(item_id)

        try:
            content = self._serialize(item.to_dict())
            with open(file_path, "wb") as f:
                f.write(content)

            # Update index
            self._index[item_id] = {
                "data_asset": item.data_asset,
                "run_time": item.run_time.isoformat(),
                "status": item.status.value,
                "file": file_path.name,
                "tags": item.tags,
            }
            self._save_index()

            return item_id

        except OSError as e:
            raise StoreWriteError(f"Failed to write {file_path}: {e}")

    def get(self, item_id: str) -> ValidationResult:
        """Retrieve a validation result by run ID.

        Args:
            item_id: The run ID of the result to retrieve.

        Returns:
            The validation result.

        Raises:
            StoreNotFoundError: If the result doesn't exist.
            StoreReadError: If reading fails.
        """
        self.initialize()

        file_path = self._get_file_path(item_id)

        if not file_path.exists():
            raise StoreNotFoundError("ValidationResult", item_id)

        try:
            with open(file_path, "rb") as f:
                content = f.read()
            data = self._deserialize(content)
            return ValidationResult.from_dict(data)

        except (json.JSONDecodeError, KeyError) as e:
            raise StoreReadError(f"Failed to parse {file_path}: {e}")
        except OSError as e:
            raise StoreReadError(f"Failed to read {file_path}: {e}")

    def exists(self, item_id: str) -> bool:
        """Check if a validation result exists.

        Args:
            item_id: The run ID to check.

        Returns:
            True if the result exists.
        """
        self.initialize()
        return item_id in self._index or self._get_file_path(item_id).exists()

    def delete(self, item_id: str) -> bool:
        """Delete a validation result.

        Args:
            item_id: The run ID of the result to delete.

        Returns:
            True if the result was deleted, False if it didn't exist.

        Raises:
            StoreWriteError: If deletion fails.
        """
        self.initialize()

        file_path = self._get_file_path(item_id)

        if not file_path.exists():
            # Also remove from index if present
            if item_id in self._index:
                del self._index[item_id]
                self._save_index()
            return False

        try:
            file_path.unlink()
            if item_id in self._index:
                del self._index[item_id]
                self._save_index()
            return True

        except OSError as e:
            raise StoreWriteError(f"Failed to delete {file_path}: {e}")

    def list_ids(self, query: StoreQuery | None = None) -> list[str]:
        """List validation result IDs matching the query.

        Args:
            query: Optional query to filter results.

        Returns:
            List of matching run IDs.
        """
        self.initialize()

        # Get all IDs from index
        if not query:
            return list(self._index.keys())

        # Filter by query
        matching_ids: list[tuple[str, datetime]] = []

        for item_id, meta in self._index.items():
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
        results: list[ValidationResult] = []

        for item_id in ids:
            try:
                result = self.get(item_id)
                results.append(result)
            except (StoreNotFoundError, StoreReadError):
                # Skip corrupted or deleted entries
                continue

        return results

    def rebuild_index(self) -> int:
        """Rebuild the index from stored files.

        This is useful if the index gets corrupted or out of sync.

        Returns:
            Number of items indexed.
        """
        self.initialize()

        self._index = {}
        path = self._config.get_full_path()

        for file_path in path.glob(f"*{self._config.file_extension}*"):
            if file_path.name.startswith("_"):
                continue

            try:
                with open(file_path, "rb") as f:
                    content = f.read()
                data = self._deserialize(content)

                item_id = data.get("run_id")
                if item_id:
                    self._index[item_id] = {
                        "data_asset": data.get("data_asset", "unknown"),
                        "run_time": data.get("run_time"),
                        "status": data.get("status"),
                        "file": file_path.name,
                        "tags": data.get("tags", {}),
                    }
            except (json.JSONDecodeError, OSError, gzip.BadGzipFile):
                continue

        self._save_index()
        return len(self._index)


class FileSystemExpectationStore(ExpectationStore[FileSystemConfig]):
    """Filesystem-based expectation store.

    Stores expectation suites as JSON files on the local filesystem.

    Example:
        >>> store = FileSystemExpectationStore(base_path=".truthound/expectations")
        >>> suite = ExpectationSuite.create("my_suite", "customers.csv")
        >>> store.save(suite)
    """

    def __init__(
        self,
        base_path: str = ".truthound/store",
        namespace: str = "default",
        prefix: str = "expectations",
        **kwargs: Any,
    ) -> None:
        """Initialize the expectation store.

        Args:
            base_path: Base directory for storing files.
            namespace: Namespace for organizing data.
            prefix: Additional path prefix.
            **kwargs: Additional configuration options.
        """
        config = FileSystemConfig(
            base_path=base_path,
            namespace=namespace,
            prefix=prefix,
            **{k: v for k, v in kwargs.items() if hasattr(FileSystemConfig, k)},
        )
        super().__init__(config)

    @classmethod
    def _default_config(cls) -> FileSystemConfig:
        """Create default configuration."""
        return FileSystemConfig(prefix="expectations")

    def _do_initialize(self) -> None:
        """Initialize the store directory."""
        path = self._config.get_full_path()
        if self._config.create_dirs:
            path.mkdir(parents=True, exist_ok=True)

    def _get_file_path(self, suite_name: str) -> Path:
        """Get the file path for a suite."""
        safe_name = suite_name.replace("/", "_").replace("\\", "_")
        return self._config.get_full_path() / f"{safe_name}{self._config.file_extension}"

    def save(self, item: ExpectationSuite) -> str:
        """Save an expectation suite.

        Args:
            item: The suite to save.

        Returns:
            The suite name.

        Raises:
            StoreWriteError: If saving fails.
        """
        self.initialize()

        file_path = self._get_file_path(item.name)

        try:
            indent = 2 if self._config.pretty_print else None
            with open(file_path, "w") as f:
                json.dump(item.to_dict(), f, indent=indent, default=str)
            return item.name

        except OSError as e:
            raise StoreWriteError(f"Failed to write {file_path}: {e}")

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

        file_path = self._get_file_path(item_id)

        if not file_path.exists():
            raise StoreNotFoundError("ExpectationSuite", item_id)

        try:
            with open(file_path) as f:
                data = json.load(f)
            return ExpectationSuite.from_dict(data)

        except (json.JSONDecodeError, KeyError) as e:
            raise StoreReadError(f"Failed to parse {file_path}: {e}")
        except OSError as e:
            raise StoreReadError(f"Failed to read {file_path}: {e}")

    def exists(self, item_id: str) -> bool:
        """Check if a suite exists."""
        self.initialize()
        return self._get_file_path(item_id).exists()

    def delete(self, item_id: str) -> bool:
        """Delete an expectation suite.

        Args:
            item_id: The suite name.

        Returns:
            True if deleted, False if it didn't exist.
        """
        self.initialize()

        file_path = self._get_file_path(item_id)

        if not file_path.exists():
            return False

        try:
            file_path.unlink()
            return True
        except OSError as e:
            raise StoreWriteError(f"Failed to delete {file_path}: {e}")

    def list_ids(self, query: StoreQuery | None = None) -> list[str]:
        """List all suite names."""
        self.initialize()

        path = self._config.get_full_path()
        ext = self._config.file_extension

        names: list[str] = []
        for file_path in path.glob(f"*{ext}"):
            name = file_path.stem
            if query and query.data_asset:
                # Filter by data_asset if specified
                try:
                    suite = self.get(name)
                    if suite.data_asset != query.data_asset:
                        continue
                except (StoreNotFoundError, StoreReadError):
                    continue
            names.append(name)

        return sorted(names)

    def query(self, query: StoreQuery) -> list[ExpectationSuite]:
        """Query expectation suites."""
        names = self.list_ids(query)
        suites: list[ExpectationSuite] = []

        for name in names:
            try:
                suite = self.get(name)
                suites.append(suite)
            except (StoreNotFoundError, StoreReadError):
                continue

        # Apply limit
        if query.limit:
            suites = suites[query.offset : query.offset + query.limit]
        elif query.offset:
            suites = suites[query.offset :]

        return suites
