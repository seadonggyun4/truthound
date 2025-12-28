"""Version storage backends for report versioning.

This module provides storage backends for versioned reports.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

from truthound.datadocs.versioning.version import (
    ReportVersion,
    VersionInfo,
    VersioningStrategy,
    IncrementalStrategy,
)


class VersionStorage(ABC):
    """Abstract base class for version storage.

    Provides methods for saving, loading, and managing versioned reports.
    """

    def __init__(
        self,
        strategy: VersioningStrategy | None = None,
    ) -> None:
        """Initialize version storage.

        Args:
            strategy: Versioning strategy to use.
        """
        self._strategy = strategy or IncrementalStrategy()

    @abstractmethod
    def save(
        self,
        report_id: str,
        content: bytes | str,
        format: str = "html",
        message: str | None = None,
        created_by: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ReportVersion:
        """Save a new version of a report.

        Args:
            report_id: Unique report identifier.
            content: Report content.
            format: Output format.
            message: Version message.
            created_by: Author.
            metadata: Additional metadata.

        Returns:
            The created ReportVersion.
        """
        pass

    @abstractmethod
    def load(
        self,
        report_id: str,
        version: int | None = None,
    ) -> ReportVersion | None:
        """Load a report version.

        Args:
            report_id: Report identifier.
            version: Version to load (latest if None).

        Returns:
            ReportVersion or None if not found.
        """
        pass

    @abstractmethod
    def list_versions(
        self,
        report_id: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[VersionInfo]:
        """List all versions of a report.

        Args:
            report_id: Report identifier.
            limit: Maximum versions to return.
            offset: Versions to skip.

        Returns:
            List of VersionInfo, newest first.
        """
        pass

    @abstractmethod
    def get_latest_version(self, report_id: str) -> int | None:
        """Get the latest version number for a report.

        Args:
            report_id: Report identifier.

        Returns:
            Latest version number or None.
        """
        pass

    @abstractmethod
    def delete_version(self, report_id: str, version: int) -> bool:
        """Delete a specific version.

        Args:
            report_id: Report identifier.
            version: Version to delete.

        Returns:
            True if deleted, False if not found.
        """
        pass

    @abstractmethod
    def count_versions(self, report_id: str) -> int:
        """Count versions for a report.

        Args:
            report_id: Report identifier.

        Returns:
            Number of versions.
        """
        pass

    def _compute_checksum(self, content: bytes | str) -> str:
        """Compute checksum for content.

        Args:
            content: Report content.

        Returns:
            SHA256 checksum.
        """
        if isinstance(content, str):
            content = content.encode("utf-8")
        return hashlib.sha256(content).hexdigest()

    def _get_size(self, content: bytes | str) -> int:
        """Get content size in bytes.

        Args:
            content: Report content.

        Returns:
            Size in bytes.
        """
        if isinstance(content, bytes):
            return len(content)
        return len(content.encode("utf-8"))


class InMemoryVersionStorage(VersionStorage):
    """In-memory version storage for testing and development.

    All data is lost when the process exits.
    """

    def __init__(
        self,
        strategy: VersioningStrategy | None = None,
    ) -> None:
        super().__init__(strategy)
        # report_id -> version -> ReportVersion
        self._versions: dict[str, dict[int, ReportVersion]] = {}

    def save(
        self,
        report_id: str,
        content: bytes | str,
        format: str = "html",
        message: str | None = None,
        created_by: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ReportVersion:
        # Get current version
        current_version = self.get_latest_version(report_id)

        # Generate next version
        next_version = self._strategy.next_version(
            current_version,
            metadata={"content": content, **(metadata or {})},
        )

        # Create version info
        info = VersionInfo(
            version=next_version,
            report_id=report_id,
            created_at=datetime.now(),
            created_by=created_by,
            message=message,
            parent_version=current_version,
            checksum=self._compute_checksum(content),
            size_bytes=self._get_size(content),
            metadata=metadata or {},
        )

        # Create report version
        report_version = ReportVersion(
            info=info,
            content=content,
            format=format,
        )

        # Store
        if report_id not in self._versions:
            self._versions[report_id] = {}
        self._versions[report_id][next_version] = report_version

        return report_version

    def load(
        self,
        report_id: str,
        version: int | None = None,
    ) -> ReportVersion | None:
        if report_id not in self._versions:
            return None

        versions = self._versions[report_id]

        if version is None:
            version = self.get_latest_version(report_id)

        return versions.get(version)

    def list_versions(
        self,
        report_id: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[VersionInfo]:
        if report_id not in self._versions:
            return []

        versions = sorted(
            self._versions[report_id].values(),
            key=lambda v: v.version,
            reverse=True,
        )

        # Apply offset
        versions = versions[offset:]

        # Apply limit
        if limit is not None:
            versions = versions[:limit]

        return [v.info for v in versions]

    def get_latest_version(self, report_id: str) -> int | None:
        if report_id not in self._versions or not self._versions[report_id]:
            return None
        return max(self._versions[report_id].keys())

    def delete_version(self, report_id: str, version: int) -> bool:
        if report_id not in self._versions:
            return False
        if version not in self._versions[report_id]:
            return False
        del self._versions[report_id][version]
        return True

    def count_versions(self, report_id: str) -> int:
        if report_id not in self._versions:
            return 0
        return len(self._versions[report_id])

    def clear(self) -> None:
        """Clear all stored versions."""
        self._versions.clear()


class FileVersionStorage(VersionStorage):
    """File-based version storage.

    Stores versions as files in a directory structure:
    base_dir/
        report_id/
            versions.json    # Version metadata
            v1.html          # Version 1 content
            v2.html          # Version 2 content
    """

    def __init__(
        self,
        base_dir: Path | str,
        strategy: VersioningStrategy | None = None,
    ) -> None:
        """Initialize file version storage.

        Args:
            base_dir: Base directory for storing versions.
            strategy: Versioning strategy.
        """
        super().__init__(strategy)
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def _report_dir(self, report_id: str) -> Path:
        """Get directory for a report."""
        safe_id = report_id.replace("/", "_").replace("\\", "_")
        return self._base_dir / safe_id

    def _versions_file(self, report_id: str) -> Path:
        """Get versions metadata file."""
        return self._report_dir(report_id) / "versions.json"

    def _version_file(self, report_id: str, version: int, format: str) -> Path:
        """Get version content file."""
        return self._report_dir(report_id) / f"v{version}.{format}"

    def _load_versions_metadata(self, report_id: str) -> dict[int, dict]:
        """Load versions metadata from file."""
        versions_file = self._versions_file(report_id)
        if not versions_file.exists():
            return {}

        with open(versions_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        return {int(k): v for k, v in data.items()}

    def _save_versions_metadata(
        self,
        report_id: str,
        metadata: dict[int, dict],
    ) -> None:
        """Save versions metadata to file."""
        report_dir = self._report_dir(report_id)
        report_dir.mkdir(parents=True, exist_ok=True)

        versions_file = self._versions_file(report_id)
        with open(versions_file, "w", encoding="utf-8") as f:
            json.dump(
                {str(k): v for k, v in metadata.items()},
                f,
                indent=2,
                ensure_ascii=False,
            )

    def save(
        self,
        report_id: str,
        content: bytes | str,
        format: str = "html",
        message: str | None = None,
        created_by: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ReportVersion:
        # Load existing metadata
        versions_metadata = self._load_versions_metadata(report_id)

        # Get current version
        current_version = max(versions_metadata.keys()) if versions_metadata else None

        # Generate next version
        next_version = self._strategy.next_version(
            current_version,
            metadata={"content": content, **(metadata or {})},
        )

        # Create version info
        info = VersionInfo(
            version=next_version,
            report_id=report_id,
            created_at=datetime.now(),
            created_by=created_by,
            message=message,
            parent_version=current_version,
            checksum=self._compute_checksum(content),
            size_bytes=self._get_size(content),
            metadata=metadata or {},
        )

        # Save content
        version_file = self._version_file(report_id, next_version, format)
        version_file.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(content, bytes):
            version_file.write_bytes(content)
        else:
            version_file.write_text(content, encoding="utf-8")

        # Update metadata
        versions_metadata[next_version] = {
            **info.to_dict(),
            "format": format,
        }
        self._save_versions_metadata(report_id, versions_metadata)

        return ReportVersion(
            info=info,
            content=content,
            format=format,
        )

    def load(
        self,
        report_id: str,
        version: int | None = None,
    ) -> ReportVersion | None:
        versions_metadata = self._load_versions_metadata(report_id)
        if not versions_metadata:
            return None

        if version is None:
            version = max(versions_metadata.keys())

        if version not in versions_metadata:
            return None

        meta = versions_metadata[version]
        info = VersionInfo.from_dict(meta)
        format = meta.get("format", "html")

        version_file = self._version_file(report_id, version, format)
        if not version_file.exists():
            return None

        # Determine if binary
        if format in ("pdf", "png", "jpg"):
            content = version_file.read_bytes()
        else:
            content = version_file.read_text(encoding="utf-8")

        return ReportVersion(
            info=info,
            content=content,
            format=format,
        )

    def list_versions(
        self,
        report_id: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[VersionInfo]:
        versions_metadata = self._load_versions_metadata(report_id)
        if not versions_metadata:
            return []

        # Sort by version descending
        sorted_versions = sorted(
            versions_metadata.items(),
            key=lambda x: x[0],
            reverse=True,
        )

        # Apply offset
        sorted_versions = sorted_versions[offset:]

        # Apply limit
        if limit is not None:
            sorted_versions = sorted_versions[:limit]

        return [VersionInfo.from_dict(meta) for _, meta in sorted_versions]

    def get_latest_version(self, report_id: str) -> int | None:
        versions_metadata = self._load_versions_metadata(report_id)
        if not versions_metadata:
            return None
        return max(versions_metadata.keys())

    def delete_version(self, report_id: str, version: int) -> bool:
        versions_metadata = self._load_versions_metadata(report_id)
        if version not in versions_metadata:
            return False

        format = versions_metadata[version].get("format", "html")

        # Delete content file
        version_file = self._version_file(report_id, version, format)
        if version_file.exists():
            version_file.unlink()

        # Update metadata
        del versions_metadata[version]
        self._save_versions_metadata(report_id, versions_metadata)

        return True

    def count_versions(self, report_id: str) -> int:
        versions_metadata = self._load_versions_metadata(report_id)
        return len(versions_metadata)
