"""Distributed state management for stream processing.

Provides state backends for stateful stream processing:
- Memory: In-memory state (single node)
- Redis: Distributed state using Redis
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar
import asyncio
import json
import logging
import time
from collections import defaultdict

from truthound.realtime.protocols import IStateStore


logger = logging.getLogger(__name__)

T = TypeVar("T")


class IStateBackend(ABC, Generic[T]):
    """Protocol for state backends.

    State backends provide persistent state storage for
    stateful stream processing operations.
    """

    @abstractmethod
    async def get(self, namespace: str, key: str) -> T | None:
        """Get value by key.

        Args:
            namespace: State namespace
            key: State key

        Returns:
            Value or None if not found
        """
        ...

    @abstractmethod
    async def put(
        self,
        namespace: str,
        key: str,
        value: T,
        ttl: int | None = None,
    ) -> None:
        """Store value.

        Args:
            namespace: State namespace
            key: State key
            value: Value to store
            ttl: Optional TTL in seconds
        """
        ...

    @abstractmethod
    async def delete(self, namespace: str, key: str) -> bool:
        """Delete value.

        Args:
            namespace: State namespace
            key: State key

        Returns:
            True if deleted, False if not found
        """
        ...

    @abstractmethod
    async def get_all(self, namespace: str, prefix: str = "") -> dict[str, T]:
        """Get all values in namespace.

        Args:
            namespace: State namespace
            prefix: Optional key prefix filter

        Returns:
            Dictionary of key-value pairs
        """
        ...

    @abstractmethod
    async def clear(self, namespace: str) -> None:
        """Clear all state in namespace.

        Args:
            namespace: State namespace
        """
        ...

    @abstractmethod
    async def keys(self, namespace: str, pattern: str = "*") -> list[str]:
        """Get all keys matching pattern.

        Args:
            namespace: State namespace
            pattern: Key pattern (supports * wildcard)

        Returns:
            List of matching keys
        """
        ...


# =============================================================================
# Memory State Backend
# =============================================================================


@dataclass
class StateEntry(Generic[T]):
    """Entry in memory state store."""

    value: T
    expires_at: float | None = None


class MemoryStateBackend(IStateBackend[Any]):
    """In-memory state backend.

    Thread-safe in-memory state storage for single-node deployments.
    Supports TTL-based expiration.

    Example:
        >>> backend = MemoryStateBackend()
        >>> await backend.put("my-namespace", "key1", {"count": 42})
        >>> value = await backend.get("my-namespace", "key1")
    """

    def __init__(self):
        self._store: dict[str, dict[str, StateEntry]] = defaultdict(dict)
        self._lock = asyncio.Lock()

    async def get(self, namespace: str, key: str) -> Any | None:
        async with self._lock:
            ns_store = self._store.get(namespace, {})
            entry = ns_store.get(key)

            if entry is None:
                return None

            # Check expiration
            if entry.expires_at is not None and time.time() > entry.expires_at:
                del ns_store[key]
                return None

            return entry.value

    async def put(
        self,
        namespace: str,
        key: str,
        value: Any,
        ttl: int | None = None,
    ) -> None:
        async with self._lock:
            expires_at = time.time() + ttl if ttl else None
            self._store[namespace][key] = StateEntry(value=value, expires_at=expires_at)

    async def delete(self, namespace: str, key: str) -> bool:
        async with self._lock:
            ns_store = self._store.get(namespace, {})
            if key in ns_store:
                del ns_store[key]
                return True
            return False

    async def get_all(self, namespace: str, prefix: str = "") -> dict[str, Any]:
        async with self._lock:
            ns_store = self._store.get(namespace, {})
            result = {}
            now = time.time()
            expired = []

            for key, entry in ns_store.items():
                if entry.expires_at is not None and now > entry.expires_at:
                    expired.append(key)
                    continue
                if key.startswith(prefix):
                    result[key] = entry.value

            # Clean up expired
            for key in expired:
                del ns_store[key]

            return result

    async def clear(self, namespace: str) -> None:
        async with self._lock:
            if namespace in self._store:
                self._store[namespace].clear()

    async def keys(self, namespace: str, pattern: str = "*") -> list[str]:
        async with self._lock:
            ns_store = self._store.get(namespace, {})
            now = time.time()
            result = []
            expired = []

            for key, entry in ns_store.items():
                if entry.expires_at is not None and now > entry.expires_at:
                    expired.append(key)
                    continue
                if self._match_pattern(key, pattern):
                    result.append(key)

            # Clean up expired
            for key in expired:
                del ns_store[key]

            return result

    def _match_pattern(self, key: str, pattern: str) -> bool:
        """Simple pattern matching with * wildcard."""
        if pattern == "*":
            return True
        if "*" not in pattern:
            return key == pattern

        # Simple prefix/suffix matching
        parts = pattern.split("*")
        if len(parts) == 2:
            prefix, suffix = parts
            return key.startswith(prefix) and key.endswith(suffix)

        return True


# =============================================================================
# Redis State Backend
# =============================================================================


@dataclass
class RedisStateConfig:
    """Configuration for Redis state backend."""

    url: str = "redis://localhost:6379"
    db: int = 0
    password: str | None = None
    key_prefix: str = "truthound:state:"
    socket_timeout: float = 5.0
    connection_pool_size: int = 10


class RedisStateBackend(IStateBackend[Any]):
    """Redis-based distributed state backend.

    Uses Redis for distributed state storage across multiple nodes.
    Supports TTL-based expiration and pattern-based key queries.

    Example:
        >>> config = RedisStateConfig(url="redis://localhost:6379")
        >>> backend = RedisStateBackend(config)
        >>> await backend.connect()
        >>> await backend.put("my-namespace", "key1", {"count": 42}, ttl=3600)

    Requires:
        pip install redis
    """

    def __init__(self, config: RedisStateConfig | None = None):
        self._config = config or RedisStateConfig()
        self._client: Any = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        """Connect to Redis."""
        try:
            import redis.asyncio as redis
        except ImportError:
            raise ImportError(
                "redis package is required for Redis state backend. "
                "Install with: pip install redis"
            )

        self._client = redis.from_url(
            self._config.url,
            db=self._config.db,
            password=self._config.password,
            socket_timeout=self._config.socket_timeout,
            max_connections=self._config.connection_pool_size,
            decode_responses=True,
        )

        # Test connection
        await self._client.ping()
        self._connected = True
        logger.info(f"Connected to Redis at {self._config.url}")

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self._client:
            await self._client.close()
            self._client = None
        self._connected = False

    def _full_key(self, namespace: str, key: str) -> str:
        """Build full Redis key."""
        return f"{self._config.key_prefix}{namespace}:{key}"

    async def get(self, namespace: str, key: str) -> Any | None:
        if not self._client:
            raise RuntimeError("Not connected to Redis")

        full_key = self._full_key(namespace, key)
        value = await self._client.get(full_key)

        if value is None:
            return None

        return json.loads(value)

    async def put(
        self,
        namespace: str,
        key: str,
        value: Any,
        ttl: int | None = None,
    ) -> None:
        if not self._client:
            raise RuntimeError("Not connected to Redis")

        full_key = self._full_key(namespace, key)
        serialized = json.dumps(value)

        if ttl:
            await self._client.setex(full_key, ttl, serialized)
        else:
            await self._client.set(full_key, serialized)

    async def delete(self, namespace: str, key: str) -> bool:
        if not self._client:
            raise RuntimeError("Not connected to Redis")

        full_key = self._full_key(namespace, key)
        result = await self._client.delete(full_key)
        return result > 0

    async def get_all(self, namespace: str, prefix: str = "") -> dict[str, Any]:
        if not self._client:
            raise RuntimeError("Not connected to Redis")

        pattern = self._full_key(namespace, f"{prefix}*")
        result = {}

        async for key in self._client.scan_iter(match=pattern):
            value = await self._client.get(key)
            if value:
                # Extract the original key
                original_key = key.replace(self._full_key(namespace, ""), "")
                result[original_key] = json.loads(value)

        return result

    async def clear(self, namespace: str) -> None:
        if not self._client:
            raise RuntimeError("Not connected to Redis")

        pattern = self._full_key(namespace, "*")
        keys = []

        async for key in self._client.scan_iter(match=pattern):
            keys.append(key)

        if keys:
            await self._client.delete(*keys)

    async def keys(self, namespace: str, pattern: str = "*") -> list[str]:
        if not self._client:
            raise RuntimeError("Not connected to Redis")

        redis_pattern = self._full_key(namespace, pattern)
        prefix = self._full_key(namespace, "")
        result = []

        async for key in self._client.scan_iter(match=redis_pattern):
            original_key = key.replace(prefix, "")
            result.append(original_key)

        return result

    async def __aenter__(self) -> "RedisStateBackend":
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.disconnect()


# =============================================================================
# State Manager
# =============================================================================


class StateManager:
    """High-level state manager for stream processing.

    Provides a unified interface for state operations with
    support for multiple namespaces and backends.

    Example:
        >>> manager = StateManager(backend=MemoryStateBackend())
        >>> state = manager.get_state("my-processor")
        >>> await state.put("counter", 0)
        >>> value = await state.get("counter")
    """

    def __init__(
        self,
        backend: IStateBackend | None = None,
        default_namespace: str = "default",
    ):
        """Initialize state manager.

        Args:
            backend: State backend (defaults to memory)
            default_namespace: Default namespace for state operations
        """
        self._backend = backend or MemoryStateBackend()
        self._default_namespace = default_namespace

    @property
    def backend(self) -> IStateBackend:
        """Get state backend."""
        return self._backend

    def get_state(self, namespace: str | None = None) -> "NamespacedState":
        """Get namespaced state accessor.

        Args:
            namespace: State namespace (uses default if not specified)

        Returns:
            NamespacedState accessor
        """
        ns = namespace or self._default_namespace
        return NamespacedState(self._backend, ns)

    async def checkpoint(self, checkpoint_id: str, state: dict[str, Any]) -> None:
        """Save checkpoint.

        Args:
            checkpoint_id: Unique checkpoint identifier
            state: State to checkpoint
        """
        await self._backend.put("checkpoints", checkpoint_id, state)

    async def restore_checkpoint(self, checkpoint_id: str) -> dict[str, Any] | None:
        """Restore checkpoint.

        Args:
            checkpoint_id: Checkpoint to restore

        Returns:
            Checkpointed state or None if not found
        """
        return await self._backend.get("checkpoints", checkpoint_id)

    async def list_checkpoints(self) -> list[str]:
        """List all checkpoints.

        Returns:
            List of checkpoint IDs
        """
        return await self._backend.keys("checkpoints")


class NamespacedState:
    """Namespaced state accessor.

    Provides convenient access to state within a specific namespace.
    """

    def __init__(self, backend: IStateBackend, namespace: str):
        self._backend = backend
        self._namespace = namespace

    @property
    def namespace(self) -> str:
        """Get namespace."""
        return self._namespace

    async def get(self, key: str) -> Any | None:
        """Get value."""
        return await self._backend.get(self._namespace, key)

    async def put(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Put value."""
        await self._backend.put(self._namespace, key, value, ttl)

    async def delete(self, key: str) -> bool:
        """Delete value."""
        return await self._backend.delete(self._namespace, key)

    async def get_all(self, prefix: str = "") -> dict[str, Any]:
        """Get all values."""
        return await self._backend.get_all(self._namespace, prefix)

    async def clear(self) -> None:
        """Clear all state."""
        await self._backend.clear(self._namespace)

    async def keys(self, pattern: str = "*") -> list[str]:
        """Get all keys."""
        return await self._backend.keys(self._namespace, pattern)

    async def increment(self, key: str, delta: int = 1) -> int:
        """Increment counter.

        Args:
            key: Counter key
            delta: Amount to increment

        Returns:
            New counter value
        """
        current = await self.get(key) or 0
        new_value = current + delta
        await self.put(key, new_value)
        return new_value

    async def get_or_default(self, key: str, default: T) -> T:
        """Get value or return default.

        Args:
            key: State key
            default: Default value

        Returns:
            Stored value or default
        """
        value = await self.get(key)
        return value if value is not None else default
