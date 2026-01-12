"""Auto schema caching system for zero-config validation."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import xxhash

    _HAS_XXHASH = True
except ImportError:
    _HAS_XXHASH = False

from truthound.schema import Schema, learn


def _fast_hash(content: str) -> str:
    """Compute a fast hash of content string.

    Uses xxhash (xxh64) if available for ~10x faster hashing,
    falls back to SHA256 if xxhash is not installed.

    Args:
        content: String content to hash.

    Returns:
        16-character hexadecimal hash digest.
    """
    if _HAS_XXHASH:
        return xxhash.xxh64(content.encode()).hexdigest()[:16]
    return hashlib.sha256(content.encode()).hexdigest()[:16]


# Default cache directory
CACHE_DIR = Path(".truthound")
SCHEMA_CACHE_FILE = "schemas.json"


def get_cache_dir(base_path: Path | None = None) -> Path:
    """Get the cache directory path.

    Args:
        base_path: Optional base path. Defaults to current directory.

    Returns:
        Path to the .truthound cache directory.
    """
    base = base_path or Path.cwd()
    return base / CACHE_DIR


def ensure_cache_dir(base_path: Path | None = None) -> Path:
    """Ensure cache directory exists.

    Args:
        base_path: Optional base path.

    Returns:
        Path to the cache directory.
    """
    cache_dir = get_cache_dir(base_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_data_fingerprint(data: Any) -> str:
    """Generate a fingerprint for data source.

    For files: Uses path + modification time + size.
    For in-memory data: Uses hash of column names + row count.

    Args:
        data: Input data source.

    Returns:
        String fingerprint for the data.
    """
    if isinstance(data, str):
        path = Path(data)
        if path.exists():
            stat = path.stat()
            content = f"{path.absolute()}:{stat.st_mtime}:{stat.st_size}"
            return _fast_hash(content)
        return _fast_hash(data)
    elif isinstance(data, dict):
        # For dict, use keys (already deterministic order in Python 3.7+) and approximate size
        keys_str = ":".join(data.keys())
        size = sum(len(v) for v in data.values() if hasattr(v, "__len__"))
        content = f"dict:{keys_str}:{size}"
        return _fast_hash(content)
    else:
        # For DataFrames, use shape and column names (already ordered)
        try:
            if hasattr(data, "columns"):
                # Polars/Pandas DataFrames preserve column order
                cols_str = ":".join(data.columns)
                rows = len(data) if hasattr(data, "__len__") else 0
                content = f"{data.shape}:{cols_str}"
                return _fast_hash(content)
            else:
                rows = len(data) if hasattr(data, "__len__") else 0
                content = f"{type(data).__name__}:{rows}"
                return _fast_hash(content)
        except Exception:
            return _fast_hash(str(type(data)))


def get_source_key(data: Any) -> str:
    """Get a stable key for the data source.

    Args:
        data: Input data source.

    Returns:
        String key identifying the source.
    """
    if isinstance(data, str):
        path = Path(data)
        if path.exists():
            return str(path.absolute())
        return data
    elif isinstance(data, dict):
        # Python 3.7+ dicts maintain insertion order
        return f"dict:{':'.join(data.keys())}"
    else:
        return f"{type(data).__name__}"


class SchemaCache:
    """Manages cached schemas for automatic validation."""

    def __init__(self, cache_dir: Path | None = None):
        """Initialize schema cache.

        Args:
            cache_dir: Optional cache directory path.
        """
        self.cache_dir = cache_dir or get_cache_dir()
        self.cache_file = self.cache_dir / SCHEMA_CACHE_FILE
        self._cache: dict = {}
        self._load()

    def _load(self) -> None:
        """Load cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file) as f:
                    self._cache = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._cache = {}

    def _save(self) -> None:
        """Save cache to disk."""
        ensure_cache_dir(self.cache_dir.parent if self.cache_dir.name == CACHE_DIR else None)
        with open(self.cache_file, "w") as f:
            json.dump(self._cache, f, indent=2)

    def get_schema(self, data: Any) -> Schema | None:
        """Get cached schema for data source.

        Args:
            data: Input data source.

        Returns:
            Cached Schema if valid, None otherwise.
        """
        source_key = get_source_key(data)
        fingerprint = get_data_fingerprint(data)

        if source_key not in self._cache:
            return None

        entry = self._cache[source_key]

        # Check if fingerprint matches (data hasn't changed)
        if entry.get("fingerprint") != fingerprint:
            return None

        # Load schema from file
        schema_path = self.cache_dir / entry.get("schema_file", "")
        if not schema_path.exists():
            return None

        try:
            return Schema.load(schema_path)
        except Exception:
            return None

    def save_schema(self, data: Any, schema: Schema) -> Path:
        """Save schema to cache.

        Args:
            data: Input data source.
            schema: Schema to cache.

        Returns:
            Path to the saved schema file.
        """
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        source_key = get_source_key(data)
        fingerprint = get_data_fingerprint(data)

        # Generate schema filename
        schema_hash = hashlib.sha256(source_key.encode()).hexdigest()[:12]
        schema_file = f"schema_{schema_hash}.yaml"
        schema_path = self.cache_dir / schema_file

        # Save schema
        schema.save(schema_path)

        # Update cache index
        self._cache[source_key] = {
            "fingerprint": fingerprint,
            "schema_file": schema_file,
            "created_at": datetime.now().isoformat(),
            "row_count": schema.row_count,
            "column_count": len(schema.columns),
        }
        self._save()

        return schema_path

    def clear(self) -> None:
        """Clear all cached schemas."""
        if self.cache_dir.exists():
            for f in self.cache_dir.glob("schema_*.yaml"):
                f.unlink()
            if self.cache_file.exists():
                self.cache_file.unlink()
        self._cache = {}

    def list_cached(self) -> list[dict]:
        """List all cached schema entries.

        Returns:
            List of cache entry information.
        """
        return [
            {"source": k, **v}
            for k, v in self._cache.items()
        ]


# Global cache instance
_global_cache: SchemaCache | None = None


def get_global_cache() -> SchemaCache:
    """Get the global schema cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = SchemaCache()
    return _global_cache


def get_or_learn_schema(data: Any, force_learn: bool = False) -> tuple[Schema, bool]:
    """Get cached schema or learn new one.

    This is the core of "zero-config" - automatically caches and retrieves
    schemas based on data source.

    Args:
        data: Input data source.
        force_learn: If True, always learn new schema.

    Returns:
        Tuple of (Schema, was_cached: bool).
    """
    cache = get_global_cache()

    if not force_learn:
        cached = cache.get_schema(data)
        if cached is not None:
            return cached, True

    # Learn new schema
    schema = learn(data)
    cache.save_schema(data, schema)
    return schema, False
