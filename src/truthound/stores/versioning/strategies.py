"""Versioning strategy implementations.

This module provides different strategies for generating and managing
version numbers.
"""

from __future__ import annotations

import hashlib
import time
from datetime import datetime
from typing import Any

from truthound.stores.versioning.base import VersioningStrategy


class IncrementalStrategy(VersioningStrategy):
    """Simple incremental versioning: 1, 2, 3, ...

    This is the default strategy and is suitable for most use cases.
    Versions are sequential integers starting from 1.
    """

    def get_next_version(
        self,
        item_id: str,
        current_version: int | None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Generate the next version number.

        Args:
            item_id: ID of the item being versioned.
            current_version: Current version number, or None if first version.
            metadata: Optional metadata (unused).

        Returns:
            The next version number.
        """
        if current_version is None:
            return 1
        return current_version + 1

    def format_version(self, version: int) -> str:
        """Format a version number for display.

        Args:
            version: The version number.

        Returns:
            Formatted version string (e.g., "v1", "v2").
        """
        return f"v{version}"

    def parse_version(self, version_str: str) -> int:
        """Parse a formatted version string.

        Args:
            version_str: The formatted version string.

        Returns:
            The version number.
        """
        if version_str.startswith("v"):
            return int(version_str[1:])
        return int(version_str)


class SemanticStrategy(VersioningStrategy):
    """Semantic versioning-like strategy.

    Versions are stored as integers but displayed as X.Y.Z format.
    Internal version = major * 10000 + minor * 100 + patch

    Example:
        1.0.0 -> 10000
        1.1.0 -> 10100
        1.1.1 -> 10101
        2.0.0 -> 20000
    """

    def get_next_version(
        self,
        item_id: str,
        current_version: int | None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Generate the next version number.

        Args:
            item_id: ID of the item being versioned.
            current_version: Current version number, or None if first version.
            metadata: Optional metadata with 'bump' key ('major', 'minor', 'patch').

        Returns:
            The next version number.
        """
        if current_version is None:
            return 10000  # 1.0.0

        bump_type = (metadata or {}).get("bump", "patch")

        major, minor, patch = self._decompose_version(current_version)

        if bump_type == "major":
            major += 1
            minor = 0
            patch = 0
        elif bump_type == "minor":
            minor += 1
            patch = 0
        else:  # patch
            patch += 1

        return major * 10000 + minor * 100 + patch

    def format_version(self, version: int) -> str:
        """Format a version number for display.

        Args:
            version: The version number.

        Returns:
            Formatted version string (e.g., "1.0.0", "1.2.3").
        """
        major, minor, patch = self._decompose_version(version)
        return f"{major}.{minor}.{patch}"

    def parse_version(self, version_str: str) -> int:
        """Parse a formatted version string.

        Args:
            version_str: The formatted version string.

        Returns:
            The version number.
        """
        if version_str.startswith("v"):
            version_str = version_str[1:]

        parts = version_str.split(".")
        if len(parts) != 3:
            raise ValueError(f"Invalid semantic version: {version_str}")

        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
        return major * 10000 + minor * 100 + patch

    def _decompose_version(self, version: int) -> tuple[int, int, int]:
        """Decompose version number into major, minor, patch."""
        major = version // 10000
        minor = (version % 10000) // 100
        patch = version % 100
        return major, minor, patch


class TimestampStrategy(VersioningStrategy):
    """Timestamp-based versioning.

    Versions are Unix timestamps (milliseconds) stored as integers.
    Displayed as ISO format timestamps.
    """

    def get_next_version(
        self,
        item_id: str,
        current_version: int | None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Generate the next version number.

        Args:
            item_id: ID of the item being versioned.
            current_version: Current version number (unused).
            metadata: Optional metadata (unused).

        Returns:
            Current timestamp in milliseconds.
        """
        return int(time.time() * 1000)

    def format_version(self, version: int) -> str:
        """Format a version number for display.

        Args:
            version: The version number (timestamp in ms).

        Returns:
            ISO format timestamp string.
        """
        dt = datetime.fromtimestamp(version / 1000)
        return dt.isoformat()

    def parse_version(self, version_str: str) -> int:
        """Parse a formatted version string.

        Args:
            version_str: The formatted version string (ISO timestamp).

        Returns:
            The version number (timestamp in ms).
        """
        dt = datetime.fromisoformat(version_str)
        return int(dt.timestamp() * 1000)

    def compare_versions(self, version_a: int, version_b: int) -> int:
        """Compare two versions (timestamps).

        Args:
            version_a: First version (timestamp).
            version_b: Second version (timestamp).

        Returns:
            -1 if a < b, 0 if a == b, 1 if a > b.
        """
        # Timestamps are naturally ordered
        if version_a < version_b:
            return -1
        elif version_a > version_b:
            return 1
        return 0


class GitLikeStrategy(VersioningStrategy):
    """Git-like hash-based versioning.

    Versions are stored as integers (sequential for ordering)
    but have associated content hashes for identification.
    Display format is a short hash prefix.
    """

    def __init__(self, hash_length: int = 7) -> None:
        """Initialize the strategy.

        Args:
            hash_length: Length of displayed hash (default 7 like git).
        """
        self.hash_length = hash_length
        self._hash_cache: dict[int, str] = {}

    def get_next_version(
        self,
        item_id: str,
        current_version: int | None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Generate the next version number.

        Args:
            item_id: ID of the item being versioned.
            current_version: Current version number, or None if first version.
            metadata: Optional metadata with 'content' for hash generation.

        Returns:
            The next version number.
        """
        version = 1 if current_version is None else current_version + 1

        # Generate and cache content hash if content provided
        if metadata and "content" in metadata:
            content = metadata["content"]
            if isinstance(content, dict):
                import json

                content = json.dumps(content, sort_keys=True)
            if isinstance(content, str):
                content = content.encode()
            content_hash = hashlib.sha256(content).hexdigest()
            self._hash_cache[version] = content_hash

        return version

    def format_version(self, version: int) -> str:
        """Format a version number for display.

        Args:
            version: The version number.

        Returns:
            Short hash string or version number.
        """
        if version in self._hash_cache:
            return self._hash_cache[version][: self.hash_length]
        return f"{version:07d}"

    def parse_version(self, version_str: str) -> int:
        """Parse a formatted version string.

        Args:
            version_str: The formatted version string.

        Returns:
            The version number.

        Note:
            For hash strings, this returns -1 as hashes can't be
            directly converted back to version numbers.
        """
        try:
            return int(version_str)
        except ValueError:
            # Try to find in cache (reverse lookup)
            for ver, hash_val in self._hash_cache.items():
                if hash_val.startswith(version_str):
                    return ver
            return -1

    def get_content_hash(self, version: int) -> str | None:
        """Get the content hash for a version.

        Args:
            version: The version number.

        Returns:
            Full content hash, or None if not cached.
        """
        return self._hash_cache.get(version)

    def set_content_hash(self, version: int, content_hash: str) -> None:
        """Set the content hash for a version.

        Args:
            version: The version number.
            content_hash: The content hash.
        """
        self._hash_cache[version] = content_hash


def get_strategy(mode: str) -> VersioningStrategy:
    """Get a versioning strategy by mode name.

    Args:
        mode: Strategy mode name ('incremental', 'semantic', 'timestamp', 'git_like').

    Returns:
        The versioning strategy instance.

    Raises:
        ValueError: If mode is unknown.
    """
    strategies: dict[str, type[VersioningStrategy]] = {
        "incremental": IncrementalStrategy,
        "semantic": SemanticStrategy,
        "timestamp": TimestampStrategy,
        "git_like": GitLikeStrategy,
    }

    if mode not in strategies:
        raise ValueError(
            f"Unknown versioning mode: {mode}. "
            f"Available: {', '.join(strategies.keys())}"
        )

    return strategies[mode]()
