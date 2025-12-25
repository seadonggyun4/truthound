"""Pattern-based cache invalidation system.

This module provides advanced cache invalidation capabilities using
glob-style pattern matching, supporting flexible cache management.

Key Features:
- Glob pattern matching (*, **, ?)
- Regex pattern support
- Tag-based invalidation
- Dependency tracking
- Cascading invalidation
- Batch operations

Example:
    from truthound.profiler.cache_patterns import (
        PatternInvalidator,
        InvalidationPattern,
        PatternType,
    )

    # Create invalidator
    invalidator = PatternInvalidator(cache_backend)

    # Invalidate by glob pattern
    count = invalidator.invalidate("profile:user:*")

    # Invalidate by regex
    count = invalidator.invalidate(r"profile:user:\d+", pattern_type=PatternType.REGEX)

    # Invalidate by tags
    count = invalidator.invalidate_by_tags(["user", "session"])
"""

from __future__ import annotations

import fnmatch
import re
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Iterator, Protocol


# =============================================================================
# Types and Enums
# =============================================================================


class PatternType(str, Enum):
    """Types of invalidation patterns."""

    GLOB = "glob"  # fnmatch-style: *, **, ?
    REGEX = "regex"  # Regular expressions
    PREFIX = "prefix"  # Simple prefix matching
    EXACT = "exact"  # Exact key match
    TAG = "tag"  # Tag-based matching


@dataclass(frozen=True)
class InvalidationPattern:
    """Represents a cache invalidation pattern.

    Attributes:
        pattern: The pattern string
        pattern_type: Type of pattern matching
        compiled: Pre-compiled pattern (for regex)
    """

    pattern: str
    pattern_type: PatternType = PatternType.GLOB
    compiled: re.Pattern | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        """Compile regex patterns."""
        if self.pattern_type == PatternType.REGEX and self.compiled is None:
            try:
                object.__setattr__(self, "compiled", re.compile(self.pattern))
            except re.error as e:
                raise ValueError(f"Invalid regex pattern: {self.pattern}") from e

    def matches(self, key: str) -> bool:
        """Check if key matches this pattern.

        Args:
            key: Cache key to check

        Returns:
            True if key matches the pattern
        """
        if self.pattern_type == PatternType.EXACT:
            return key == self.pattern

        elif self.pattern_type == PatternType.PREFIX:
            return key.startswith(self.pattern)

        elif self.pattern_type == PatternType.GLOB:
            return self._glob_match(key)

        elif self.pattern_type == PatternType.REGEX:
            if self.compiled:
                return bool(self.compiled.match(key))
            return bool(re.match(self.pattern, key))

        return False

    def _glob_match(self, key: str) -> bool:
        """Match using glob-style patterns.

        Supports:
        - * : matches any characters except :
        - ** : matches any characters including :
        - ? : matches single character
        """
        # Convert glob pattern to regex
        regex = self._glob_to_regex(self.pattern)
        return bool(re.match(regex, key))

    def _glob_to_regex(self, pattern: str) -> str:
        """Convert glob pattern to regex.

        Args:
            pattern: Glob pattern

        Returns:
            Equivalent regex pattern
        """
        # Escape special regex chars except * and ?
        regex = ""
        i = 0

        while i < len(pattern):
            c = pattern[i]

            if c == "*":
                # Check for **
                if i + 1 < len(pattern) and pattern[i + 1] == "*":
                    regex += ".*"  # Match everything
                    i += 2
                    continue
                else:
                    regex += "[^:]*"  # Match except separator

            elif c == "?":
                regex += "."  # Match single char

            elif c in ".^$+{}[]|()":
                regex += "\\" + c  # Escape

            else:
                regex += c

            i += 1

        return f"^{regex}$"


@dataclass
class InvalidationResult:
    """Result of cache invalidation operation.

    Attributes:
        pattern: Pattern used for invalidation
        keys_matched: Number of keys that matched
        keys_invalidated: Number of keys successfully invalidated
        duration_ms: Operation duration in milliseconds
        errors: List of error messages
    """

    pattern: str
    keys_matched: int = 0
    keys_invalidated: int = 0
    duration_ms: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Check if invalidation was successful."""
        return len(self.errors) == 0 and self.keys_matched == self.keys_invalidated

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pattern": self.pattern,
            "keys_matched": self.keys_matched,
            "keys_invalidated": self.keys_invalidated,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "errors": self.errors,
        }


# =============================================================================
# Cache Key Protocol
# =============================================================================


class CacheKeyProvider(Protocol):
    """Protocol for cache backends that can list keys."""

    def keys(self, pattern: str = "*") -> Iterator[str]:
        """Iterate over cache keys matching pattern."""
        ...

    def delete(self, key: str) -> bool:
        """Delete a single key."""
        ...

    def delete_many(self, keys: list[str]) -> int:
        """Delete multiple keys."""
        ...


# =============================================================================
# Tag Registry
# =============================================================================


@dataclass
class TagEntry:
    """Entry in tag registry.

    Attributes:
        key: Cache key
        tags: Set of tags
        created_at: When entry was created
    """

    key: str
    tags: set[str]
    created_at: datetime = field(default_factory=datetime.now)


class TagRegistry:
    """Registry for cache key tags.

    Tracks which cache keys are associated with which tags,
    enabling tag-based invalidation.

    Example:
        registry = TagRegistry()

        # Register tags for a key
        registry.register("user:123:profile", {"user", "profile"})

        # Find keys by tag
        keys = registry.find_by_tag("user")

        # Find by multiple tags (intersection)
        keys = registry.find_by_tags(["user", "active"])
    """

    def __init__(self, max_entries: int = 100000):
        """Initialize registry.

        Args:
            max_entries: Maximum number of entries to track
        """
        self.max_entries = max_entries
        self._entries: dict[str, TagEntry] = {}
        self._tag_index: dict[str, set[str]] = {}  # tag -> set of keys
        self._lock = threading.RLock()

    def register(self, key: str, tags: set[str] | list[str]) -> None:
        """Register tags for a cache key.

        Args:
            key: Cache key
            tags: Tags to associate with key
        """
        tags = set(tags)

        with self._lock:
            # Update or create entry
            if key in self._entries:
                old_tags = self._entries[key].tags
                self._entries[key].tags = tags

                # Update tag index for removed tags
                for tag in old_tags - tags:
                    if tag in self._tag_index:
                        self._tag_index[tag].discard(key)
            else:
                # Check capacity
                if len(self._entries) >= self.max_entries:
                    self._evict_oldest()

                self._entries[key] = TagEntry(key=key, tags=tags)

            # Update tag index for added tags
            for tag in tags:
                if tag not in self._tag_index:
                    self._tag_index[tag] = set()
                self._tag_index[tag].add(key)

    def unregister(self, key: str) -> set[str]:
        """Unregister a cache key.

        Args:
            key: Cache key to unregister

        Returns:
            Tags that were associated with the key
        """
        with self._lock:
            if key not in self._entries:
                return set()

            entry = self._entries.pop(key)

            # Update tag index
            for tag in entry.tags:
                if tag in self._tag_index:
                    self._tag_index[tag].discard(key)
                    if not self._tag_index[tag]:
                        del self._tag_index[tag]

            return entry.tags

    def get_tags(self, key: str) -> set[str]:
        """Get tags for a cache key.

        Args:
            key: Cache key

        Returns:
            Set of tags
        """
        with self._lock:
            if key in self._entries:
                return self._entries[key].tags.copy()
            return set()

    def find_by_tag(self, tag: str) -> set[str]:
        """Find all keys with a specific tag.

        Args:
            tag: Tag to search for

        Returns:
            Set of cache keys
        """
        with self._lock:
            return self._tag_index.get(tag, set()).copy()

    def find_by_tags(
        self,
        tags: list[str],
        match_all: bool = True,
    ) -> set[str]:
        """Find keys matching multiple tags.

        Args:
            tags: Tags to search for
            match_all: If True, keys must have all tags (AND)
                      If False, keys must have any tag (OR)

        Returns:
            Set of cache keys
        """
        if not tags:
            return set()

        with self._lock:
            tag_sets = [
                self._tag_index.get(tag, set())
                for tag in tags
            ]

            if match_all:
                # Intersection (AND)
                result = tag_sets[0].copy()
                for s in tag_sets[1:]:
                    result &= s
            else:
                # Union (OR)
                result = set()
                for s in tag_sets:
                    result |= s

            return result

    def list_tags(self) -> list[str]:
        """List all known tags.

        Returns:
            List of tag names
        """
        with self._lock:
            return list(self._tag_index.keys())

    def tag_counts(self) -> dict[str, int]:
        """Get count of keys for each tag.

        Returns:
            Dictionary mapping tags to key counts
        """
        with self._lock:
            return {
                tag: len(keys)
                for tag, keys in self._tag_index.items()
            }

    def clear(self) -> int:
        """Clear all entries.

        Returns:
            Number of entries cleared
        """
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
            self._tag_index.clear()
            return count

    def _evict_oldest(self) -> None:
        """Evict oldest entries when at capacity."""
        if not self._entries:
            return

        # Find oldest entry
        oldest_key = min(
            self._entries.keys(),
            key=lambda k: self._entries[k].created_at,
        )
        self.unregister(oldest_key)


# =============================================================================
# Dependency Tracker
# =============================================================================


class DependencyTracker:
    """Tracks cache key dependencies for cascading invalidation.

    When a key is invalidated, all keys that depend on it
    are also invalidated.

    Example:
        tracker = DependencyTracker()

        # Key B depends on key A
        tracker.add_dependency("keyB", "keyA")

        # When A is invalidated, B should also be invalidated
        deps = tracker.get_dependents("keyA")  # Returns {"keyB"}
    """

    def __init__(self):
        """Initialize tracker."""
        self._dependencies: dict[str, set[str]] = {}  # key -> keys it depends on
        self._dependents: dict[str, set[str]] = {}  # key -> keys that depend on it
        self._lock = threading.RLock()

    def add_dependency(self, key: str, depends_on: str | list[str]) -> None:
        """Add a dependency.

        Args:
            key: Cache key that has the dependency
            depends_on: Key(s) that this key depends on
        """
        if isinstance(depends_on, str):
            depends_on = [depends_on]

        with self._lock:
            if key not in self._dependencies:
                self._dependencies[key] = set()

            for dep in depends_on:
                self._dependencies[key].add(dep)

                if dep not in self._dependents:
                    self._dependents[dep] = set()
                self._dependents[dep].add(key)

    def remove_dependency(self, key: str, depends_on: str | None = None) -> None:
        """Remove dependency.

        Args:
            key: Cache key
            depends_on: Specific dependency to remove, or None for all
        """
        with self._lock:
            if key not in self._dependencies:
                return

            if depends_on:
                self._dependencies[key].discard(depends_on)
                if depends_on in self._dependents:
                    self._dependents[depends_on].discard(key)
            else:
                # Remove all dependencies
                for dep in self._dependencies[key]:
                    if dep in self._dependents:
                        self._dependents[dep].discard(key)
                del self._dependencies[key]

    def get_dependencies(self, key: str) -> set[str]:
        """Get keys that this key depends on.

        Args:
            key: Cache key

        Returns:
            Set of dependency keys
        """
        with self._lock:
            return self._dependencies.get(key, set()).copy()

    def get_dependents(self, key: str, recursive: bool = True) -> set[str]:
        """Get keys that depend on this key.

        Args:
            key: Cache key
            recursive: If True, include transitive dependents

        Returns:
            Set of dependent keys
        """
        with self._lock:
            direct = self._dependents.get(key, set()).copy()

            if not recursive:
                return direct

            # Find transitive dependents
            all_dependents = direct.copy()
            to_process = list(direct)

            while to_process:
                current = to_process.pop()
                new_deps = self._dependents.get(current, set()) - all_dependents
                all_dependents |= new_deps
                to_process.extend(new_deps)

            return all_dependents

    def unregister(self, key: str) -> None:
        """Unregister a key and all its relationships.

        Args:
            key: Cache key to unregister
        """
        with self._lock:
            # Remove dependencies
            self.remove_dependency(key)

            # Remove from dependents index
            if key in self._dependents:
                del self._dependents[key]

            # Remove from other keys' dependencies
            for k in list(self._dependencies.keys()):
                self._dependencies[k].discard(key)

    def clear(self) -> None:
        """Clear all dependencies."""
        with self._lock:
            self._dependencies.clear()
            self._dependents.clear()


# =============================================================================
# Pattern Invalidator
# =============================================================================


class PatternInvalidator:
    """Pattern-based cache invalidation.

    Provides flexible cache invalidation using various pattern types
    including glob, regex, prefix, and tag-based matching.

    Example:
        from truthound.profiler.caching import ProfileCache

        cache = ProfileCache()
        invalidator = PatternInvalidator(cache.backend)

        # Invalidate all user profiles
        result = invalidator.invalidate("profile:user:*")
        print(f"Invalidated {result.keys_invalidated} keys")

        # Invalidate by regex
        result = invalidator.invalidate(
            r"profile:user:\d+:session",
            pattern_type=PatternType.REGEX,
        )

        # Invalidate with cascade
        result = invalidator.invalidate_with_cascade("user:123")

    Attributes:
        backend: Cache backend that supports key listing
        tag_registry: Optional tag registry for tag-based invalidation
        dependency_tracker: Optional dependency tracker for cascading
        batch_size: Batch size for delete operations
    """

    def __init__(
        self,
        backend: Any,
        tag_registry: TagRegistry | None = None,
        dependency_tracker: DependencyTracker | None = None,
        batch_size: int = 1000,
    ):
        """Initialize invalidator.

        Args:
            backend: Cache backend
            tag_registry: Tag registry instance
            dependency_tracker: Dependency tracker instance
            batch_size: Batch size for deletions
        """
        self.backend = backend
        self.tag_registry = tag_registry or TagRegistry()
        self.dependency_tracker = dependency_tracker or DependencyTracker()
        self.batch_size = batch_size
        self._lock = threading.RLock()

    def invalidate(
        self,
        pattern: str,
        pattern_type: PatternType = PatternType.GLOB,
        dry_run: bool = False,
    ) -> InvalidationResult:
        """Invalidate cache entries matching pattern.

        Args:
            pattern: Invalidation pattern
            pattern_type: Type of pattern matching
            dry_run: If True, only count matches without deleting

        Returns:
            InvalidationResult with statistics
        """
        start_time = time.time()
        result = InvalidationResult(pattern=pattern)

        try:
            inv_pattern = InvalidationPattern(pattern, pattern_type)

            # Find matching keys
            matching_keys = self._find_matching_keys(inv_pattern)
            result.keys_matched = len(matching_keys)

            if not dry_run:
                # Delete in batches
                deleted = self._delete_keys(list(matching_keys))
                result.keys_invalidated = deleted

                # Unregister from tag registry
                for key in matching_keys:
                    self.tag_registry.unregister(key)
                    self.dependency_tracker.unregister(key)

        except Exception as e:
            result.errors.append(str(e))

        result.duration_ms = (time.time() - start_time) * 1000
        return result

    def invalidate_by_tags(
        self,
        tags: list[str],
        match_all: bool = True,
        dry_run: bool = False,
    ) -> InvalidationResult:
        """Invalidate cache entries by tags.

        Args:
            tags: Tags to match
            match_all: If True, keys must have all tags
            dry_run: If True, only count matches

        Returns:
            InvalidationResult
        """
        start_time = time.time()
        pattern_str = f"tags:[{','.join(tags)}]"
        result = InvalidationResult(pattern=pattern_str)

        try:
            matching_keys = self.tag_registry.find_by_tags(tags, match_all)
            result.keys_matched = len(matching_keys)

            if not dry_run:
                deleted = self._delete_keys(list(matching_keys))
                result.keys_invalidated = deleted

                for key in matching_keys:
                    self.tag_registry.unregister(key)
                    self.dependency_tracker.unregister(key)

        except Exception as e:
            result.errors.append(str(e))

        result.duration_ms = (time.time() - start_time) * 1000
        return result

    def invalidate_with_cascade(
        self,
        key: str,
        dry_run: bool = False,
    ) -> InvalidationResult:
        """Invalidate a key and all its dependents.

        Args:
            key: Cache key to invalidate
            dry_run: If True, only count matches

        Returns:
            InvalidationResult
        """
        start_time = time.time()
        result = InvalidationResult(pattern=f"cascade:{key}")

        try:
            # Get all dependents
            dependents = self.dependency_tracker.get_dependents(key, recursive=True)
            all_keys = {key} | dependents
            result.keys_matched = len(all_keys)

            if not dry_run:
                deleted = self._delete_keys(list(all_keys))
                result.keys_invalidated = deleted

                for k in all_keys:
                    self.tag_registry.unregister(k)
                    self.dependency_tracker.unregister(k)

        except Exception as e:
            result.errors.append(str(e))

        result.duration_ms = (time.time() - start_time) * 1000
        return result

    def invalidate_multiple(
        self,
        patterns: list[str],
        pattern_type: PatternType = PatternType.GLOB,
        dry_run: bool = False,
    ) -> list[InvalidationResult]:
        """Invalidate multiple patterns.

        Args:
            patterns: List of patterns to invalidate
            pattern_type: Type of pattern matching
            dry_run: If True, only count matches

        Returns:
            List of InvalidationResult for each pattern
        """
        return [
            self.invalidate(pattern, pattern_type, dry_run)
            for pattern in patterns
        ]

    def _find_matching_keys(self, pattern: InvalidationPattern) -> set[str]:
        """Find all keys matching pattern.

        Args:
            pattern: Invalidation pattern

        Returns:
            Set of matching keys
        """
        matching = set()

        # Try to use backend's native pattern support
        if hasattr(self.backend, "keys"):
            try:
                if pattern.pattern_type == PatternType.GLOB:
                    # Some backends support glob patterns natively
                    for key in self.backend.keys(pattern.pattern):
                        matching.add(key)
                    return matching
                elif pattern.pattern_type == PatternType.PREFIX:
                    for key in self.backend.keys(f"{pattern.pattern}*"):
                        matching.add(key)
                    return matching
            except (TypeError, NotImplementedError):
                pass

            # Fall back to scanning all keys
            try:
                for key in self.backend.keys("*"):
                    if pattern.matches(key):
                        matching.add(key)
            except (TypeError, NotImplementedError):
                pass

        # For memory backend, scan internal cache
        if hasattr(self.backend, "_cache"):
            for key in list(self.backend._cache.keys()):
                if pattern.matches(key):
                    matching.add(key)

        return matching

    def _delete_keys(self, keys: list[str]) -> int:
        """Delete keys from cache.

        Args:
            keys: Keys to delete

        Returns:
            Number of keys deleted
        """
        if not keys:
            return 0

        # Try batch delete
        if hasattr(self.backend, "delete_many"):
            try:
                total = 0
                for i in range(0, len(keys), self.batch_size):
                    batch = keys[i:i + self.batch_size]
                    total += self.backend.delete_many(batch)
                return total
            except (TypeError, NotImplementedError):
                pass

        # Fall back to individual deletes
        deleted = 0
        for key in keys:
            try:
                if self.backend.delete(key):
                    deleted += 1
            except Exception:
                pass

        return deleted

    def register_with_tags(
        self,
        key: str,
        tags: set[str] | list[str],
    ) -> None:
        """Register a cache key with tags.

        Args:
            key: Cache key
            tags: Tags to associate
        """
        self.tag_registry.register(key, tags)

    def register_dependency(
        self,
        key: str,
        depends_on: str | list[str],
    ) -> None:
        """Register a cache key dependency.

        Args:
            key: Cache key
            depends_on: Key(s) this key depends on
        """
        self.dependency_tracker.add_dependency(key, depends_on)

    def get_stats(self) -> dict[str, Any]:
        """Get invalidator statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "tag_count": len(self.tag_registry.list_tags()),
            "tag_entries": len(self.tag_registry._entries),
            "tag_counts": self.tag_registry.tag_counts(),
            "dependency_count": len(self.dependency_tracker._dependencies),
        }


# =============================================================================
# Convenience Functions
# =============================================================================


def create_pattern(
    pattern: str,
    pattern_type: PatternType | str = PatternType.GLOB,
) -> InvalidationPattern:
    """Create an invalidation pattern.

    Args:
        pattern: Pattern string
        pattern_type: Type of pattern

    Returns:
        InvalidationPattern instance
    """
    if isinstance(pattern_type, str):
        pattern_type = PatternType(pattern_type.lower())

    return InvalidationPattern(pattern=pattern, pattern_type=pattern_type)


def glob_to_regex(pattern: str) -> str:
    """Convert glob pattern to regex pattern.

    Args:
        pattern: Glob pattern

    Returns:
        Equivalent regex pattern
    """
    return InvalidationPattern(pattern, PatternType.GLOB)._glob_to_regex(pattern)
