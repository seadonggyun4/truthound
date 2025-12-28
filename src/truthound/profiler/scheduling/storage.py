"""Storage implementations for profile history.

This module provides storage backends for managing profile history,
enabling incremental profiling and trend analysis.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

from truthound.profiler.scheduling.protocols import ProfileStorage

if TYPE_CHECKING:
    from truthound.profiler.base import TableProfile

logger = logging.getLogger(__name__)


@dataclass
class ProfileHistoryEntry:
    """Entry in the profile history.

    Attributes:
        profile_id: Unique identifier.
        timestamp: When the profile was created.
        source: Data source identifier.
        row_count: Number of rows profiled.
        column_count: Number of columns profiled.
        metadata: Additional metadata.
        profile: The actual profile (may be None if not loaded).
    """

    profile_id: str
    timestamp: datetime
    source: str = ""
    row_count: int = 0
    column_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    profile: Any = None  # TableProfile when loaded

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "profile_id": self.profile_id,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProfileHistoryEntry":
        """Create from dictionary."""
        return cls(
            profile_id=data["profile_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            source=data.get("source", ""),
            row_count=data.get("row_count", 0),
            column_count=data.get("column_count", 0),
            metadata=data.get("metadata", {}),
        )


class ProfileHistoryStorage(ProfileStorage, ABC):
    """Abstract base class for profile storage implementations."""

    @abstractmethod
    def get_history(
        self,
        limit: int | None = None,
        since: datetime | None = None,
    ) -> list[ProfileHistoryEntry]:
        """Get profile history entries.

        Args:
            limit: Maximum number of entries.
            since: Only include entries after this time.

        Returns:
            List of history entries.
        """
        ...


class InMemoryProfileStorage(ProfileHistoryStorage):
    """In-memory profile storage for testing and development.

    Stores profiles in memory with no persistence.

    Example:
        storage = InMemoryProfileStorage(max_profiles=10)
        storage.save(profile)

        last = storage.get_last_profile()
    """

    def __init__(self, max_profiles: int = 100):
        """Initialize storage.

        Args:
            max_profiles: Maximum number of profiles to keep.
        """
        self._max_profiles = max_profiles
        self._profiles: dict[str, "TableProfile"] = {}
        self._history: list[ProfileHistoryEntry] = []

    def get_last_profile(self) -> "TableProfile | None":
        """Get the most recent profile."""
        if not self._history:
            return None
        latest = max(self._history, key=lambda e: e.timestamp)
        return self._profiles.get(latest.profile_id)

    def get_last_run_time(self) -> datetime | None:
        """Get timestamp of last profiling run."""
        if not self._history:
            return None
        return max(e.timestamp for e in self._history)

    def save(
        self,
        profile: "TableProfile",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Save a profile to storage."""
        profile_id = str(uuid.uuid4())
        now = datetime.now()

        entry = ProfileHistoryEntry(
            profile_id=profile_id,
            timestamp=now,
            source=getattr(profile, "source", ""),
            row_count=getattr(profile, "row_count", 0),
            column_count=len(getattr(profile, "columns", [])),
            metadata=metadata or {},
        )

        self._profiles[profile_id] = profile
        self._history.append(entry)

        # Trim if over limit
        while len(self._history) > self._max_profiles:
            oldest = min(self._history, key=lambda e: e.timestamp)
            self._history.remove(oldest)
            self._profiles.pop(oldest.profile_id, None)

        return profile_id

    def get_profile(self, profile_id: str) -> "TableProfile | None":
        """Get a specific profile by ID."""
        return self._profiles.get(profile_id)

    def list_profiles(
        self,
        limit: int | None = None,
        since: datetime | None = None,
    ) -> list[tuple[str, datetime]]:
        """List available profiles."""
        entries = self._history

        if since:
            entries = [e for e in entries if e.timestamp > since]

        # Sort by timestamp descending
        entries = sorted(entries, key=lambda e: e.timestamp, reverse=True)

        if limit:
            entries = entries[:limit]

        return [(e.profile_id, e.timestamp) for e in entries]

    def delete_profile(self, profile_id: str) -> bool:
        """Delete a profile."""
        if profile_id in self._profiles:
            del self._profiles[profile_id]
            self._history = [e for e in self._history if e.profile_id != profile_id]
            return True
        return False

    def get_baseline_schema(self) -> Any:
        """Get the baseline schema from the first profile."""
        if not self._history:
            return None
        oldest = min(self._history, key=lambda e: e.timestamp)
        profile = self._profiles.get(oldest.profile_id)
        if profile:
            return getattr(profile, "schema", None)
        return None

    def get_history(
        self,
        limit: int | None = None,
        since: datetime | None = None,
    ) -> list[ProfileHistoryEntry]:
        """Get profile history entries."""
        entries = self._history

        if since:
            entries = [e for e in entries if e.timestamp > since]

        entries = sorted(entries, key=lambda e: e.timestamp, reverse=True)

        if limit:
            entries = entries[:limit]

        return entries


class FileProfileStorage(ProfileHistoryStorage):
    """File-based profile storage with persistence.

    Stores profiles as JSON files in a directory structure.

    Example:
        storage = FileProfileStorage("./profiles")
        storage.save(profile)

        # Profiles are persisted across restarts
        last = storage.get_last_profile()
    """

    def __init__(
        self,
        base_path: str | Path,
        max_profiles: int = 100,
        compress: bool = False,
    ):
        """Initialize storage.

        Args:
            base_path: Base directory for profile storage.
            max_profiles: Maximum number of profiles to keep.
            compress: Whether to compress stored profiles.
        """
        self._base_path = Path(base_path)
        self._max_profiles = max_profiles
        self._compress = compress

        # Create directories
        self._profiles_dir = self._base_path / "profiles"
        self._profiles_dir.mkdir(parents=True, exist_ok=True)

        self._index_file = self._base_path / "index.json"
        self._index: list[ProfileHistoryEntry] = self._load_index()

    def _load_index(self) -> list[ProfileHistoryEntry]:
        """Load the profile index."""
        if not self._index_file.exists():
            return []

        try:
            with open(self._index_file, "r") as f:
                data = json.load(f)
                return [ProfileHistoryEntry.from_dict(e) for e in data]
        except Exception as e:
            logger.warning(f"Failed to load profile index: {e}")
            return []

    def _save_index(self) -> None:
        """Save the profile index."""
        try:
            with open(self._index_file, "w") as f:
                json.dump([e.to_dict() for e in self._index], f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save profile index: {e}")

    def _get_profile_path(self, profile_id: str) -> Path:
        """Get path for a profile file."""
        suffix = ".json.gz" if self._compress else ".json"
        return self._profiles_dir / f"{profile_id}{suffix}"

    def get_last_profile(self) -> "TableProfile | None":
        """Get the most recent profile."""
        if not self._index:
            return None
        latest = max(self._index, key=lambda e: e.timestamp)
        return self.get_profile(latest.profile_id)

    def get_last_run_time(self) -> datetime | None:
        """Get timestamp of last profiling run."""
        if not self._index:
            return None
        return max(e.timestamp for e in self._index)

    def save(
        self,
        profile: "TableProfile",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Save a profile to storage."""
        profile_id = str(uuid.uuid4())
        now = datetime.now()

        # Create history entry
        entry = ProfileHistoryEntry(
            profile_id=profile_id,
            timestamp=now,
            source=getattr(profile, "source", ""),
            row_count=getattr(profile, "row_count", 0),
            column_count=len(getattr(profile, "columns", [])),
            metadata=metadata or {},
        )

        # Save profile to file
        profile_path = self._get_profile_path(profile_id)
        try:
            profile_data = self._serialize_profile(profile)

            if self._compress:
                import gzip
                with gzip.open(profile_path, "wt", encoding="utf-8") as f:
                    json.dump(profile_data, f)
            else:
                with open(profile_path, "w") as f:
                    json.dump(profile_data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save profile: {e}")
            raise

        # Update index
        self._index.append(entry)
        self._save_index()

        # Cleanup old profiles
        self._cleanup_old_profiles()

        return profile_id

    def get_profile(self, profile_id: str) -> "TableProfile | None":
        """Get a specific profile by ID."""
        profile_path = self._get_profile_path(profile_id)

        if not profile_path.exists():
            return None

        try:
            if self._compress:
                import gzip
                with gzip.open(profile_path, "rt", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                with open(profile_path, "r") as f:
                    data = json.load(f)

            return self._deserialize_profile(data)

        except Exception as e:
            logger.error(f"Failed to load profile {profile_id}: {e}")
            return None

    def list_profiles(
        self,
        limit: int | None = None,
        since: datetime | None = None,
    ) -> list[tuple[str, datetime]]:
        """List available profiles."""
        entries = self._index

        if since:
            entries = [e for e in entries if e.timestamp > since]

        entries = sorted(entries, key=lambda e: e.timestamp, reverse=True)

        if limit:
            entries = entries[:limit]

        return [(e.profile_id, e.timestamp) for e in entries]

    def delete_profile(self, profile_id: str) -> bool:
        """Delete a profile."""
        profile_path = self._get_profile_path(profile_id)

        # Remove file
        if profile_path.exists():
            try:
                profile_path.unlink()
            except Exception as e:
                logger.error(f"Failed to delete profile file: {e}")
                return False

        # Update index
        self._index = [e for e in self._index if e.profile_id != profile_id]
        self._save_index()

        return True

    def get_baseline_schema(self) -> Any:
        """Get the baseline schema from the first profile."""
        if not self._index:
            return None

        oldest = min(self._index, key=lambda e: e.timestamp)
        profile = self.get_profile(oldest.profile_id)

        if profile:
            return getattr(profile, "schema", None)
        return None

    def get_history(
        self,
        limit: int | None = None,
        since: datetime | None = None,
    ) -> list[ProfileHistoryEntry]:
        """Get profile history entries."""
        entries = self._index

        if since:
            entries = [e for e in entries if e.timestamp > since]

        entries = sorted(entries, key=lambda e: e.timestamp, reverse=True)

        if limit:
            entries = entries[:limit]

        return entries

    def _cleanup_old_profiles(self) -> None:
        """Remove old profiles beyond the limit."""
        while len(self._index) > self._max_profiles:
            oldest = min(self._index, key=lambda e: e.timestamp)
            self.delete_profile(oldest.profile_id)

    def _serialize_profile(self, profile: "TableProfile") -> dict[str, Any]:
        """Serialize a profile to a dictionary."""
        # Use profile's own serialization if available
        if hasattr(profile, "to_dict"):
            return profile.to_dict()

        # Fallback to basic serialization
        return {
            "type": type(profile).__name__,
            "data": str(profile),
        }

    def _deserialize_profile(self, data: dict[str, Any]) -> "TableProfile":
        """Deserialize a profile from a dictionary."""
        # Import here to avoid circular imports
        from truthound.profiler.base import TableProfile

        # Use profile's own deserialization if available
        if hasattr(TableProfile, "from_dict"):
            return TableProfile.from_dict(data)

        # This is a simplified fallback
        raise NotImplementedError(
            "Profile deserialization requires TableProfile.from_dict method"
        )
