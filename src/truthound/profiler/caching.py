"""Caching layer for data profiling with file hash based caching.

This module provides a flexible caching system for profile results:
- File hash based cache key generation
- Multiple backend support (memory, file, Redis)
- TTL-based expiration
- Lazy evaluation with cache-through pattern

Key features:
- Pluggable backend architecture
- Content-based cache invalidation
- Compression support for large profiles
- Thread-safe operations

Example:
    from truthound.profiler.caching import ProfileCache, FileHashCacheKey

    # Create cache with memory backend
    cache = ProfileCache()

    # Generate cache key from file
    key = FileHashCacheKey.from_file("data.parquet")

    # Cache-through pattern
    profile = cache.get_or_compute(key, lambda: expensive_profile())
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import pickle
import shutil
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Generic, Protocol, TypeVar

from truthound.profiler.base import TableProfile
from truthound.profiler.schema import ProfileSerializer


# =============================================================================
# Cache Key Protocol
# =============================================================================


class CacheKeyProtocol(Protocol):
    """Protocol for cache keys."""

    def to_string(self) -> str:
        """Convert key to string representation."""
        ...

    def __hash__(self) -> int:
        ...

    def __eq__(self, other: object) -> bool:
        ...


@dataclass(frozen=True)
class CacheKey:
    """Base cache key implementation."""

    key: str
    namespace: str = "default"
    version: str = "1"

    def to_string(self) -> str:
        """Create unique string representation."""
        return f"{self.namespace}:{self.version}:{self.key}"

    def __hash__(self) -> int:
        return hash(self.to_string())


@dataclass(frozen=True)
class FileHashCacheKey(CacheKey):
    """Cache key based on file content hash.

    Uses SHA-256 to create a content-based cache key that
    automatically invalidates when file contents change.

    Attributes:
        file_path: Original file path
        file_hash: SHA-256 hash of file contents
        file_size: File size in bytes
        file_mtime: File modification time
    """

    file_path: str = ""
    file_hash: str = ""
    file_size: int = 0
    file_mtime: float = 0.0

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        namespace: str = "profile",
        version: str = "1",
        quick_hash: bool = False,
        sample_size: int = 1024 * 1024,  # 1MB sample for quick hash
    ) -> "FileHashCacheKey":
        """Create cache key from file.

        Args:
            path: Path to the file
            namespace: Cache namespace
            version: Cache version
            quick_hash: If True, only hash first/last portions for speed
            sample_size: Bytes to sample when using quick hash

        Returns:
            FileHashCacheKey instance
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        stat = path.stat()
        file_size = stat.st_size
        file_mtime = stat.st_mtime

        # Calculate hash
        if quick_hash and file_size > sample_size * 2:
            # For large files, hash beginning, end, and size
            file_hash = cls._quick_hash(path, sample_size)
        else:
            file_hash = cls._full_hash(path)

        return cls(
            key=file_hash,
            namespace=namespace,
            version=version,
            file_path=str(path),
            file_hash=file_hash,
            file_size=file_size,
            file_mtime=file_mtime,
        )

    @staticmethod
    def _full_hash(path: Path, chunk_size: int = 8192) -> str:
        """Calculate full file hash."""
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def _quick_hash(path: Path, sample_size: int) -> str:
        """Calculate quick hash from file samples."""
        hasher = hashlib.sha256()
        file_size = path.stat().st_size

        with open(path, "rb") as f:
            # Hash beginning
            hasher.update(f.read(sample_size))

            # Hash end
            f.seek(-sample_size, 2)
            hasher.update(f.read(sample_size))

        # Include size in hash
        hasher.update(str(file_size).encode())

        return hasher.hexdigest()

    def to_string(self) -> str:
        """Create unique string representation including file info."""
        return f"{self.namespace}:{self.version}:{self.file_hash}:{self.file_size}"


@dataclass(frozen=True)
class DataFrameHashCacheKey(CacheKey):
    """Cache key based on DataFrame content hash.

    Creates a hash based on DataFrame schema and sample data.
    """

    schema_hash: str = ""
    sample_hash: str = ""
    row_count: int = 0
    column_count: int = 0

    @classmethod
    def from_dataframe(
        cls,
        df: Any,  # pl.DataFrame or similar
        *,
        namespace: str = "profile",
        version: str = "1",
        sample_rows: int = 1000,
    ) -> "DataFrameHashCacheKey":
        """Create cache key from DataFrame.

        Args:
            df: Polars DataFrame
            namespace: Cache namespace
            version: Cache version
            sample_rows: Number of rows to sample for hash

        Returns:
            DataFrameHashCacheKey instance
        """
        import polars as pl

        if not isinstance(df, (pl.DataFrame, pl.LazyFrame)):
            raise TypeError(f"Expected Polars DataFrame, got {type(df)}")

        if isinstance(df, pl.LazyFrame):
            schema = df.collect_schema()
            sample_df = df.head(sample_rows).collect()
        else:
            schema = df.schema
            sample_df = df.head(sample_rows)

        # Hash schema
        schema_str = str(sorted(schema.items()))
        schema_hash = hashlib.sha256(schema_str.encode()).hexdigest()[:16]

        # Hash sample data
        sample_bytes = sample_df.to_pandas().to_csv().encode()
        sample_hash = hashlib.sha256(sample_bytes).hexdigest()[:16]

        # Combined key
        key = f"{schema_hash}:{sample_hash}:{len(sample_df)}:{len(schema)}"

        return cls(
            key=key,
            namespace=namespace,
            version=version,
            schema_hash=schema_hash,
            sample_hash=sample_hash,
            row_count=len(sample_df),
            column_count=len(schema),
        )


# =============================================================================
# Cache Entry
# =============================================================================


@dataclass
class CacheEntry:
    """Cached profile entry with metadata."""

    profile: TableProfile
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime | None = None
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    compressed: bool = False
    size_bytes: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def touch(self) -> None:
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = datetime.now()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        serializer = ProfileSerializer()
        return {
            "profile": serializer.serialize(self.profile),
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CacheEntry":
        """Deserialize from dictionary."""
        serializer = ProfileSerializer()
        profile = serializer.deserialize(data["profile"])

        expires_at = None
        if data.get("expires_at"):
            expires_at = datetime.fromisoformat(data["expires_at"])

        return cls(
            profile=profile,
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=expires_at,
            access_count=data.get("access_count", 0),
            last_accessed=datetime.fromisoformat(data.get("last_accessed", data["created_at"])),
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# Cache Backend Protocol
# =============================================================================


class CacheBackend(ABC):
    """Abstract base class for cache backends.

    Implement this to create custom cache backends (Redis, Memcached, etc.)
    """

    @abstractmethod
    def get(self, key: str) -> CacheEntry | None:
        """Retrieve entry from cache.

        Args:
            key: Cache key string

        Returns:
            CacheEntry if found, None otherwise
        """
        pass

    @abstractmethod
    def set(
        self,
        key: str,
        entry: CacheEntry,
        ttl: timedelta | None = None,
    ) -> None:
        """Store entry in cache.

        Args:
            key: Cache key string
            entry: Entry to cache
            ttl: Time-to-live for entry
        """
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete entry from cache.

        Args:
            key: Cache key string

        Returns:
            True if entry was deleted, False if not found
        """
        pass

    @abstractmethod
    def clear(self) -> int:
        """Clear all entries from cache.

        Returns:
            Number of entries cleared
        """
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists in cache.

        Args:
            key: Cache key string

        Returns:
            True if key exists
        """
        pass

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        return {}


class MemoryCacheBackend(CacheBackend):
    """In-memory cache backend with LRU eviction.

    Thread-safe implementation using locks.

    Attributes:
        max_size: Maximum number of entries
        max_memory_bytes: Maximum memory usage in bytes (0 = unlimited)
    """

    def __init__(
        self,
        *,
        max_size: int = 1000,
        max_memory_bytes: int = 0,
    ):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_bytes
        self._cache: dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> CacheEntry | None:
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None

            if entry.is_expired():
                del self._cache[key]
                self._misses += 1
                return None

            entry.touch()
            self._hits += 1
            return entry

    def set(
        self,
        key: str,
        entry: CacheEntry,
        ttl: timedelta | None = None,
    ) -> None:
        with self._lock:
            if ttl:
                entry.expires_at = datetime.now() + ttl

            self._cache[key] = entry

            # Evict if over size
            if len(self._cache) > self.max_size:
                self._evict_lru()

    def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> int:
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count

    def exists(self, key: str) -> bool:
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False
            if entry.is_expired():
                del self._cache[key]
                return False
            return True

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            return {
                "type": "memory",
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_ratio": self._hits / total if total > 0 else 0.0,
            }

    def _evict_lru(self) -> None:
        """Evict least recently used entries."""
        if not self._cache:
            return

        # Find LRU entry
        lru_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].last_accessed,
        )
        del self._cache[lru_key]


class FileCacheBackend(CacheBackend):
    """File-based cache backend with optional compression.

    Stores cached profiles as JSON files with gzip compression.

    Attributes:
        cache_dir: Directory for cache files
        compress: Whether to compress cache files
    """

    def __init__(
        self,
        cache_dir: str | Path = ".truthound_cache",
        *,
        compress: bool = True,
        max_size_mb: int = 1000,
    ):
        self.cache_dir = Path(cache_dir)
        self.compress = compress
        self.max_size_mb = max_size_mb
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, key: str) -> Path:
        """Get file path for cache key."""
        # Use hash to avoid filesystem issues with long keys
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        suffix = ".json.gz" if self.compress else ".json"
        return self.cache_dir / f"{key_hash}{suffix}"

    def get(self, key: str) -> CacheEntry | None:
        path = self._get_path(key)

        with self._lock:
            if not path.exists():
                self._misses += 1
                return None

            try:
                if self.compress:
                    with gzip.open(path, "rt", encoding="utf-8") as f:
                        data = json.load(f)
                else:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                entry = CacheEntry.from_dict(data)

                if entry.is_expired():
                    path.unlink(missing_ok=True)
                    self._misses += 1
                    return None

                entry.touch()
                self._hits += 1

                # Update file with new access stats
                self._save_entry(path, entry)

                return entry

            except (json.JSONDecodeError, KeyError, OSError):
                path.unlink(missing_ok=True)
                self._misses += 1
                return None

    def set(
        self,
        key: str,
        entry: CacheEntry,
        ttl: timedelta | None = None,
    ) -> None:
        if ttl:
            entry.expires_at = datetime.now() + ttl

        path = self._get_path(key)

        with self._lock:
            self._save_entry(path, entry)

            # Check cache size and cleanup if needed
            self._maybe_cleanup()

    def _save_entry(self, path: Path, entry: CacheEntry) -> None:
        """Save entry to file."""
        data = entry.to_dict()

        if self.compress:
            with gzip.open(path, "wt", encoding="utf-8") as f:
                json.dump(data, f)
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f)

    def delete(self, key: str) -> bool:
        path = self._get_path(key)

        with self._lock:
            if path.exists():
                path.unlink()
                return True
            return False

    def clear(self) -> int:
        with self._lock:
            count = 0
            for path in self.cache_dir.glob("*.json*"):
                path.unlink()
                count += 1
            return count

    def exists(self, key: str) -> bool:
        path = self._get_path(key)
        return path.exists()

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            cache_files = list(self.cache_dir.glob("*.json*"))
            total_size = sum(f.stat().st_size for f in cache_files)
            total = self._hits + self._misses

            return {
                "type": "file",
                "cache_dir": str(self.cache_dir),
                "file_count": len(cache_files),
                "total_size_mb": total_size / (1024 * 1024),
                "hits": self._hits,
                "misses": self._misses,
                "hit_ratio": self._hits / total if total > 0 else 0.0,
            }

    def _maybe_cleanup(self) -> None:
        """Clean up cache if over size limit."""
        cache_files = list(self.cache_dir.glob("*.json*"))
        total_size = sum(f.stat().st_size for f in cache_files)
        max_bytes = self.max_size_mb * 1024 * 1024

        if total_size <= max_bytes:
            return

        # Sort by modification time, delete oldest
        cache_files.sort(key=lambda f: f.stat().st_mtime)

        for path in cache_files:
            if total_size <= max_bytes * 0.8:  # Clean to 80%
                break
            size = path.stat().st_size
            path.unlink()
            total_size -= size


class RedisConnectionError(Exception):
    """Raised when Redis connection fails."""

    pass


class RedisCacheBackend(CacheBackend):
    """Redis-based cache backend for distributed caching.

    Requires redis package to be installed. Includes proper error
    handling for connection failures and timeouts.

    For production use with automatic fallback, consider using
    `ResilientCacheBackend` from `truthound.profiler.resilience`.

    Example:
        backend = RedisCacheBackend(
            host="localhost",
            port=6379,
            prefix="truthound:cache:",
            connect_timeout=5.0,
            socket_timeout=2.0,
        )

    Attributes:
        host: Redis server hostname
        port: Redis server port
        prefix: Key prefix for namespace isolation
        connection_info: Connection details for diagnostics
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: str | None = None,
        prefix: str = "truthound:cache:",
        connect_timeout: float = 5.0,
        socket_timeout: float = 2.0,
        retry_on_timeout: bool = True,
        max_connections: int = 10,
        health_check_interval: int = 30,
        lazy_connect: bool = True,
        **kwargs: Any,
    ):
        self.host = host
        self.port = port
        self.prefix = prefix
        self._hits = 0
        self._misses = 0
        self._errors = 0
        self._lock = threading.RLock()
        self._connected = False
        self._last_error: str | None = None
        self._last_error_time: datetime | None = None

        try:
            import redis
            from redis.exceptions import RedisError
            self._redis_module = redis
            self._RedisError = RedisError
        except ImportError:
            raise ImportError(
                "Redis support requires the 'redis' package. "
                "Install with: pip install redis"
            )

        # Create connection pool with timeout settings
        try:
            self._pool = redis.ConnectionPool(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=False,
                socket_connect_timeout=connect_timeout,
                socket_timeout=socket_timeout,
                retry_on_timeout=retry_on_timeout,
                max_connections=max_connections,
                health_check_interval=health_check_interval,
                **kwargs,
            )
            self._client = redis.Redis(connection_pool=self._pool)

            # Test connection unless lazy
            if not lazy_connect:
                self._client.ping()
                self._connected = True

        except Exception as e:
            self._connected = False
            self._last_error = str(e)
            self._last_error_time = datetime.now()
            if not lazy_connect:
                raise RedisConnectionError(
                    f"Failed to connect to Redis at {host}:{port}: {e}"
                ) from e

    @property
    def connection_info(self) -> dict[str, Any]:
        """Get connection information."""
        return {
            "host": self.host,
            "port": self.port,
            "prefix": self.prefix,
            "connected": self._connected,
            "last_error": self._last_error,
            "last_error_time": (
                self._last_error_time.isoformat()
                if self._last_error_time else None
            ),
        }

    def _make_key(self, key: str) -> str:
        """Create Redis key with prefix."""
        return f"{self.prefix}{key}"

    def _handle_error(self, e: Exception, operation: str) -> None:
        """Handle and record errors."""
        with self._lock:
            self._errors += 1
            self._last_error = f"{operation}: {e}"
            self._last_error_time = datetime.now()

            # Check if it's a connection error
            if "Connection" in str(type(e).__name__) or "Timeout" in str(type(e).__name__):
                self._connected = False

    def ping(self) -> bool:
        """Check if Redis is reachable.

        Returns:
            True if Redis responds to ping
        """
        try:
            self._client.ping()
            self._connected = True
            return True
        except Exception as e:
            self._handle_error(e, "ping")
            return False

    def get(self, key: str) -> CacheEntry | None:
        redis_key = self._make_key(key)

        try:
            data = self._client.get(redis_key)
            self._connected = True
        except self._RedisError as e:
            self._handle_error(e, "get")
            raise RedisConnectionError(f"Redis get failed: {e}") from e

        if data is None:
            with self._lock:
                self._misses += 1
            return None

        try:
            entry_dict = json.loads(data.decode("utf-8"))
            entry = CacheEntry.from_dict(entry_dict)

            if entry.is_expired():
                try:
                    self._client.delete(redis_key)
                except self._RedisError:
                    pass  # Ignore delete errors for expired entries
                with self._lock:
                    self._misses += 1
                return None

            entry.touch()
            with self._lock:
                self._hits += 1

            return entry

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # Corrupted entry - try to delete
            try:
                self._client.delete(redis_key)
            except self._RedisError:
                pass
            with self._lock:
                self._misses += 1
            return None

    def set(
        self,
        key: str,
        entry: CacheEntry,
        ttl: timedelta | None = None,
    ) -> None:
        if ttl:
            entry.expires_at = datetime.now() + ttl

        redis_key = self._make_key(key)

        try:
            data = json.dumps(entry.to_dict()).encode("utf-8")
        except (TypeError, ValueError) as e:
            raise ValueError(f"Failed to serialize cache entry: {e}") from e

        try:
            if ttl:
                self._client.setex(redis_key, ttl, data)
            else:
                self._client.set(redis_key, data)
            self._connected = True
        except self._RedisError as e:
            self._handle_error(e, "set")
            raise RedisConnectionError(f"Redis set failed: {e}") from e

    def delete(self, key: str) -> bool:
        redis_key = self._make_key(key)
        try:
            result = self._client.delete(redis_key) > 0
            self._connected = True
            return result
        except self._RedisError as e:
            self._handle_error(e, "delete")
            raise RedisConnectionError(f"Redis delete failed: {e}") from e

    def clear(self) -> int:
        pattern = f"{self.prefix}*"
        try:
            keys = self._client.keys(pattern)
            if keys:
                result = self._client.delete(*keys)
                self._connected = True
                return result
            return 0
        except self._RedisError as e:
            self._handle_error(e, "clear")
            raise RedisConnectionError(f"Redis clear failed: {e}") from e

    def exists(self, key: str) -> bool:
        redis_key = self._make_key(key)
        try:
            result = self._client.exists(redis_key) > 0
            self._connected = True
            return result
        except self._RedisError as e:
            self._handle_error(e, "exists")
            raise RedisConnectionError(f"Redis exists failed: {e}") from e

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses

            stats = {
                "type": "redis",
                "host": self.host,
                "port": self.port,
                "prefix": self.prefix,
                "connected": self._connected,
                "hits": self._hits,
                "misses": self._misses,
                "errors": self._errors,
                "hit_ratio": self._hits / total if total > 0 else 0.0,
            }

            # Try to get key count
            try:
                pattern = f"{self.prefix}*"
                keys = self._client.keys(pattern)
                stats["key_count"] = len(keys)
            except self._RedisError:
                stats["key_count"] = -1

            if self._last_error:
                stats["last_error"] = self._last_error
                stats["last_error_time"] = (
                    self._last_error_time.isoformat()
                    if self._last_error_time else None
                )

            return stats

    def close(self) -> None:
        """Close the connection pool."""
        try:
            self._pool.disconnect()
            self._connected = False
        except Exception:
            pass


# =============================================================================
# Cache Backend Registry
# =============================================================================


class CacheBackendRegistry:
    """Registry for cache backend factories.

    Allows registration of custom backend types.

    Example:
        registry = CacheBackendRegistry()
        registry.register("custom", CustomBackend)
        backend = registry.create("custom", **kwargs)
    """

    def __init__(self) -> None:
        self._backends: dict[str, type[CacheBackend]] = {}

    def register(
        self,
        name: str,
        backend_class: type[CacheBackend],
    ) -> None:
        """Register a backend class."""
        self._backends[name] = backend_class

    def create(self, name: str, **kwargs: Any) -> CacheBackend:
        """Create a backend instance."""
        if name not in self._backends:
            raise KeyError(
                f"Unknown cache backend: {name}. "
                f"Available: {list(self._backends.keys())}"
            )
        return self._backends[name](**kwargs)

    def list_backends(self) -> list[str]:
        """List registered backend names."""
        return list(self._backends.keys())


# Global registry with default backends
cache_backend_registry = CacheBackendRegistry()
cache_backend_registry.register("memory", MemoryCacheBackend)
cache_backend_registry.register("file", FileCacheBackend)
cache_backend_registry.register("redis", RedisCacheBackend)


# =============================================================================
# Profile Cache
# =============================================================================


@dataclass
class CacheConfig:
    """Configuration for profile caching."""

    backend: str = "memory"
    backend_options: dict[str, Any] = field(default_factory=dict)
    default_ttl: timedelta | None = None
    enabled: bool = True
    compression: bool = True


class ProfileCache:
    """High-level profile caching with cache-through pattern.

    This is the main interface for caching profile results.
    It wraps a cache backend and provides convenience methods.

    Example:
        # Create cache with default memory backend
        cache = ProfileCache()

        # Or with file backend
        cache = ProfileCache(
            backend="file",
            backend_options={"cache_dir": ".cache"}
        )

        # Cache-through pattern
        key = FileHashCacheKey.from_file("data.parquet")
        profile = cache.get_or_compute(
            key,
            compute_fn=lambda: profile_file("data.parquet")
        )
    """

    def __init__(
        self,
        backend: str | CacheBackend = "memory",
        backend_options: dict[str, Any] | None = None,
        default_ttl: timedelta | None = None,
        enabled: bool = True,
    ):
        """Initialize profile cache.

        Args:
            backend: Backend name or instance
            backend_options: Options for backend construction
            default_ttl: Default time-to-live for entries
            enabled: Whether caching is enabled
        """
        self.enabled = enabled
        self.default_ttl = default_ttl

        if isinstance(backend, CacheBackend):
            self._backend = backend
        else:
            options = backend_options or {}
            self._backend = cache_backend_registry.create(backend, **options)

    @property
    def backend(self) -> CacheBackend:
        """Access the underlying backend."""
        return self._backend

    def get(self, key: CacheKeyProtocol) -> TableProfile | None:
        """Get profile from cache.

        Args:
            key: Cache key

        Returns:
            Cached profile or None
        """
        if not self.enabled:
            return None

        entry = self._backend.get(key.to_string())
        return entry.profile if entry else None

    def set(
        self,
        key: CacheKeyProtocol,
        profile: TableProfile,
        ttl: timedelta | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store profile in cache.

        Args:
            key: Cache key
            profile: Profile to cache
            ttl: Time-to-live (uses default if not specified)
            metadata: Additional metadata to store
        """
        if not self.enabled:
            return

        entry = CacheEntry(
            profile=profile,
            metadata=metadata or {},
        )

        self._backend.set(
            key.to_string(),
            entry,
            ttl=ttl or self.default_ttl,
        )

    def get_or_compute(
        self,
        key: CacheKeyProtocol,
        compute_fn: Callable[[], TableProfile],
        ttl: timedelta | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TableProfile:
        """Get from cache or compute and cache.

        This implements the cache-through pattern:
        1. Try to get from cache
        2. If miss, compute the profile
        3. Store in cache
        4. Return the profile

        Args:
            key: Cache key
            compute_fn: Function to compute profile on cache miss
            ttl: Time-to-live for cached entry
            metadata: Additional metadata to store

        Returns:
            Cached or computed profile
        """
        # Try cache first
        cached = self.get(key)
        if cached is not None:
            return cached

        # Compute profile
        profile = compute_fn()

        # Store in cache
        self.set(key, profile, ttl=ttl, metadata=metadata)

        return profile

    def invalidate(self, key: CacheKeyProtocol) -> bool:
        """Invalidate a cache entry.

        Args:
            key: Cache key

        Returns:
            True if entry was invalidated
        """
        return self._backend.delete(key.to_string())

    def invalidate_by_pattern(self, pattern: str) -> int:
        """Invalidate entries matching a pattern.

        Note: Only supported by some backends.

        Args:
            pattern: Pattern to match (glob-style)

        Returns:
            Number of entries invalidated
        """
        # This is a simplified implementation
        # Full pattern matching would require backend support
        return 0

    def clear(self) -> int:
        """Clear all cache entries.

        Returns:
            Number of entries cleared
        """
        return self._backend.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        stats = self._backend.get_stats()
        stats["enabled"] = self.enabled
        stats["default_ttl_seconds"] = (
            self.default_ttl.total_seconds() if self.default_ttl else None
        )
        return stats

    def warm(
        self,
        keys: list[CacheKeyProtocol],
        compute_fn: Callable[[CacheKeyProtocol], TableProfile],
        *,
        parallel: bool = False,
    ) -> dict[str, bool]:
        """Warm cache with multiple entries.

        Args:
            keys: Cache keys to warm
            compute_fn: Function to compute each profile
            parallel: Whether to compute in parallel

        Returns:
            Dictionary mapping key strings to success status
        """
        results: dict[str, bool] = {}

        for key in keys:
            key_str = key.to_string()
            try:
                if not self._backend.exists(key_str):
                    profile = compute_fn(key)
                    self.set(key, profile)
                results[key_str] = True
            except Exception:
                results[key_str] = False

        return results


# =============================================================================
# Caching Decorator
# =============================================================================


def cached_profile(
    cache: ProfileCache | None = None,
    ttl: timedelta | None = None,
    key_fn: Callable[..., CacheKeyProtocol] | None = None,
) -> Callable:
    """Decorator to cache profile function results.

    Example:
        cache = ProfileCache()

        @cached_profile(cache, ttl=timedelta(hours=1))
        def profile_file(path: str) -> TableProfile:
            # expensive profiling...
            return profile

    Args:
        cache: ProfileCache instance (creates default if not provided)
        ttl: Time-to-live for cached entries
        key_fn: Function to generate cache key from arguments

    Returns:
        Decorated function
    """
    _cache = cache or ProfileCache()

    def decorator(func: Callable[..., TableProfile]) -> Callable[..., TableProfile]:
        def wrapper(*args: Any, **kwargs: Any) -> TableProfile:
            # Generate cache key
            if key_fn:
                key = key_fn(*args, **kwargs)
            else:
                # Default: use first argument as file path
                if args and isinstance(args[0], (str, Path)):
                    key = FileHashCacheKey.from_file(args[0])
                else:
                    # Fallback to function call hash
                    call_hash = hashlib.sha256(
                        f"{func.__name__}:{args}:{kwargs}".encode()
                    ).hexdigest()
                    key = CacheKey(key=call_hash)

            return _cache.get_or_compute(
                key,
                compute_fn=lambda: func(*args, **kwargs),
                ttl=ttl,
            )

        return wrapper

    return decorator


# =============================================================================
# Convenience Functions
# =============================================================================


def create_cache(
    backend: str = "memory",
    **kwargs: Any,
) -> ProfileCache:
    """Create a ProfileCache with the specified backend.

    Args:
        backend: Backend type ("memory", "file", "redis")
        **kwargs: Backend-specific options

    Returns:
        Configured ProfileCache instance
    """
    return ProfileCache(backend=backend, backend_options=kwargs)


def hash_file(path: str | Path, quick: bool = False) -> str:
    """Calculate file content hash.

    Args:
        path: Path to file
        quick: Use quick hash for large files

    Returns:
        SHA-256 hash string
    """
    key = FileHashCacheKey.from_file(path, quick_hash=quick)
    return key.file_hash
