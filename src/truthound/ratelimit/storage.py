"""Storage backends for rate limiting.

This module provides various storage implementations for persisting
rate limit state:
- Memory: Fast, single-process storage
- Redis: Distributed storage for multi-instance deployments
- Distributed: Generic distributed storage interface
"""

from __future__ import annotations

import hashlib
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from truthound.ratelimit.core import (
    RateLimitStorage,
    TokenBucketState,
    WindowState,
    current_time,
)


T = TypeVar("T")


# =============================================================================
# Memory Storage
# =============================================================================


@dataclass
class _StorageEntry(Generic[T]):
    """Entry in memory storage with TTL."""

    value: T
    expires_at: float | None = None

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.expires_at is None:
            return False
        return current_time() > self.expires_at


class MemoryStorage(RateLimitStorage[T]):
    """In-memory storage for rate limiting.

    Thread-safe implementation using locks.
    Suitable for single-process deployments.

    Example:
        >>> storage = MemoryStorage[TokenBucketState]()
        >>> storage.set("key", TokenBucketState(tokens=10, last_update=time.time()))
        >>> state = storage.get("key")
    """

    def __init__(
        self,
        cleanup_interval: float = 60.0,
        max_entries: int = 10000,
    ) -> None:
        """Initialize memory storage.

        Args:
            cleanup_interval: Seconds between cleanup runs.
            max_entries: Maximum number of entries before forced cleanup.
        """
        self._storage: dict[str, _StorageEntry[T]] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()
        self._cleanup_interval = cleanup_interval
        self._max_entries = max_entries
        self._last_cleanup = current_time()

    def get(self, key: str) -> T | None:
        """Get state for a key."""
        with self._global_lock:
            entry = self._storage.get(key)
            if entry is None:
                return None
            if entry.is_expired():
                del self._storage[key]
                return None
            return entry.value

    def set(self, key: str, state: T, ttl: float | None = None) -> None:
        """Set state for a key."""
        expires_at = None
        if ttl is not None:
            expires_at = current_time() + ttl

        with self._global_lock:
            self._storage[key] = _StorageEntry(value=state, expires_at=expires_at)
            self._maybe_cleanup()

    def delete(self, key: str) -> bool:
        """Delete state for a key."""
        with self._global_lock:
            if key in self._storage:
                del self._storage[key]
                return True
            return False

    def increment(
        self,
        key: str,
        amount: int = 1,
        ttl: float | None = None,
    ) -> int:
        """Atomically increment a counter."""
        with self._global_lock:
            entry = self._storage.get(key)

            if entry is None or entry.is_expired():
                new_value = amount
            else:
                # Extract count from state
                if isinstance(entry.value, WindowState):
                    new_value = entry.value.count + amount
                else:
                    # Generic increment (assume numeric-like)
                    new_value = amount

            expires_at = None
            if ttl is not None:
                expires_at = current_time() + ttl

            # Store as WindowState for counter operations
            new_state = WindowState(count=new_value, window_start=current_time())
            self._storage[key] = _StorageEntry(
                value=new_state,  # type: ignore
                expires_at=expires_at,
            )

            return new_value

    def get_with_lock(
        self,
        key: str,
        timeout: float = 1.0,
    ) -> tuple[T | None, Any]:
        """Get state with a lock."""
        lock = self._get_or_create_lock(key)
        acquired = lock.acquire(timeout=timeout)

        if not acquired:
            return None, None

        value = self.get(key)
        return value, lock

    def set_with_lock(
        self,
        key: str,
        state: T,
        lock_token: Any,
        ttl: float | None = None,
    ) -> bool:
        """Set state with a lock."""
        if lock_token is None:
            return False

        try:
            self.set(key, state, ttl)
            return True
        finally:
            if isinstance(lock_token, threading.Lock):
                try:
                    lock_token.release()
                except RuntimeError:
                    pass  # Already released

    def clear(self) -> None:
        """Clear all entries."""
        with self._global_lock:
            self._storage.clear()
            self._locks.clear()

    def size(self) -> int:
        """Get number of entries."""
        with self._global_lock:
            return len(self._storage)

    def _get_or_create_lock(self, key: str) -> threading.Lock:
        """Get or create a lock for a key."""
        with self._global_lock:
            if key not in self._locks:
                self._locks[key] = threading.Lock()
            return self._locks[key]

    def _maybe_cleanup(self) -> None:
        """Cleanup expired entries if needed."""
        now = current_time()

        if now - self._last_cleanup < self._cleanup_interval:
            if len(self._storage) < self._max_entries:
                return

        self._cleanup()
        self._last_cleanup = now

    def _cleanup(self) -> None:
        """Remove expired entries."""
        expired = [
            key for key, entry in self._storage.items()
            if entry.is_expired()
        ]
        for key in expired:
            del self._storage[key]
            if key in self._locks:
                del self._locks[key]


# =============================================================================
# Redis Storage
# =============================================================================


class RedisStorage(RateLimitStorage[T]):
    """Redis-based storage for distributed rate limiting.

    Provides atomic operations using Redis Lua scripts.
    Suitable for multi-instance deployments.

    Example:
        >>> import redis
        >>> client = redis.Redis(host='localhost', port=6379)
        >>> storage = RedisStorage(client, prefix="ratelimit:")
    """

    # Lua script for atomic token bucket operations
    TOKEN_BUCKET_SCRIPT = """
    local key = KEYS[1]
    local capacity = tonumber(ARGV[1])
    local refill_rate = tonumber(ARGV[2])
    local tokens_requested = tonumber(ARGV[3])
    local now = tonumber(ARGV[4])
    local ttl = tonumber(ARGV[5])

    local data = redis.call('HMGET', key, 'tokens', 'last_update')
    local tokens = tonumber(data[1]) or capacity
    local last_update = tonumber(data[2]) or now

    -- Refill tokens
    local elapsed = now - last_update
    tokens = math.min(capacity, tokens + elapsed * refill_rate)

    -- Check if we can acquire
    local allowed = 0
    if tokens >= tokens_requested then
        tokens = tokens - tokens_requested
        allowed = 1
    end

    -- Update state
    redis.call('HMSET', key, 'tokens', tokens, 'last_update', now)
    if ttl > 0 then
        redis.call('EXPIRE', key, ttl)
    end

    return {allowed, tokens}
    """

    # Lua script for sliding window counter
    SLIDING_WINDOW_SCRIPT = """
    local key_prefix = KEYS[1]
    local window_size = tonumber(ARGV[1])
    local sub_window_count = tonumber(ARGV[2])
    local limit = tonumber(ARGV[3])
    local tokens = tonumber(ARGV[4])
    local now = tonumber(ARGV[5])
    local ttl = tonumber(ARGV[6])

    local sub_window_size = window_size / sub_window_count
    local current_sub = math.floor(now / sub_window_size)
    local oldest_valid = current_sub - sub_window_count + 1

    -- Count requests in valid sub-windows
    local total = 0
    for i = oldest_valid, current_sub do
        local count = redis.call('GET', key_prefix .. ':' .. i)
        if count then
            total = total + tonumber(count)
        end
    end

    -- Check limit
    local allowed = 0
    if total + tokens <= limit then
        local sub_key = key_prefix .. ':' .. current_sub
        redis.call('INCRBY', sub_key, tokens)
        if ttl > 0 then
            redis.call('EXPIRE', sub_key, ttl)
        end
        allowed = 1
        total = total + tokens
    end

    return {allowed, limit - total}
    """

    def __init__(
        self,
        client: Any,
        prefix: str = "rl:",
        default_ttl: float = 3600.0,
    ) -> None:
        """Initialize Redis storage.

        Args:
            client: Redis client instance.
            prefix: Key prefix for rate limit entries.
            default_ttl: Default TTL for entries.
        """
        self._client = client
        self._prefix = prefix
        self._default_ttl = default_ttl
        self._scripts: dict[str, Any] = {}

    def _get_script(self, name: str, script: str) -> Any:
        """Get or register a Lua script."""
        if name not in self._scripts:
            self._scripts[name] = self._client.register_script(script)
        return self._scripts[name]

    def _make_key(self, key: str) -> str:
        """Create full Redis key."""
        return f"{self._prefix}{key}"

    def get(self, key: str) -> T | None:
        """Get state for a key."""
        full_key = self._make_key(key)
        data = self._client.hgetall(full_key)

        if not data:
            return None

        # Parse based on stored type
        if b"tokens" in data:
            return TokenBucketState(
                tokens=float(data[b"tokens"]),
                last_update=float(data[b"last_update"]),
                bucket_key=key,
            )  # type: ignore
        elif b"count" in data:
            return WindowState(
                count=int(data[b"count"]),
                window_start=float(data.get(b"window_start", 0)),
                bucket_key=key,
            )  # type: ignore

        return None

    def set(self, key: str, state: T, ttl: float | None = None) -> None:
        """Set state for a key."""
        full_key = self._make_key(key)
        ttl = ttl or self._default_ttl

        if isinstance(state, TokenBucketState):
            self._client.hmset(full_key, {
                "tokens": state.tokens,
                "last_update": state.last_update,
            })
        elif isinstance(state, WindowState):
            self._client.hmset(full_key, {
                "count": state.count,
                "window_start": state.window_start,
            })
        else:
            # Generic serialization
            import json
            self._client.set(full_key, json.dumps(state))

        if ttl > 0:
            self._client.expire(full_key, int(ttl))

    def delete(self, key: str) -> bool:
        """Delete state for a key."""
        full_key = self._make_key(key)
        return self._client.delete(full_key) > 0

    def increment(
        self,
        key: str,
        amount: int = 1,
        ttl: float | None = None,
    ) -> int:
        """Atomically increment a counter."""
        full_key = self._make_key(key)
        new_value = self._client.incrby(full_key, amount)

        if ttl is not None and ttl > 0:
            self._client.expire(full_key, int(ttl))

        return new_value

    def get_with_lock(
        self,
        key: str,
        timeout: float = 1.0,
    ) -> tuple[T | None, Any]:
        """Get state with a distributed lock."""
        lock_key = f"{self._make_key(key)}:lock"
        lock_value = hashlib.sha256(
            f"{time.time()}{threading.current_thread().ident}".encode()
        ).hexdigest()[:16]

        # Try to acquire lock
        acquired = self._client.set(
            lock_key,
            lock_value,
            nx=True,
            ex=int(timeout) or 1,
        )

        if not acquired:
            return None, None

        value = self.get(key)
        return value, (lock_key, lock_value)

    def set_with_lock(
        self,
        key: str,
        state: T,
        lock_token: Any,
        ttl: float | None = None,
    ) -> bool:
        """Set state and release lock atomically."""
        if lock_token is None:
            return False

        lock_key, lock_value = lock_token

        # Verify we still hold the lock
        current = self._client.get(lock_key)
        if current != lock_value.encode():
            return False

        try:
            self.set(key, state, ttl)
            return True
        finally:
            # Release lock
            self._client.delete(lock_key)

    def token_bucket_acquire(
        self,
        key: str,
        capacity: int,
        refill_rate: float,
        tokens: int,
        ttl: float | None = None,
    ) -> tuple[bool, float]:
        """Atomic token bucket acquire operation.

        Args:
            key: Bucket key.
            capacity: Maximum tokens.
            refill_rate: Tokens per second.
            tokens: Tokens to acquire.
            ttl: Entry TTL.

        Returns:
            Tuple of (allowed, remaining_tokens).
        """
        script = self._get_script("token_bucket", self.TOKEN_BUCKET_SCRIPT)
        result = script(
            keys=[self._make_key(key)],
            args=[
                capacity,
                refill_rate,
                tokens,
                current_time(),
                int(ttl or self._default_ttl),
            ],
        )
        return bool(result[0]), float(result[1])

    def sliding_window_acquire(
        self,
        key: str,
        window_size: float,
        sub_window_count: int,
        limit: int,
        tokens: int,
        ttl: float | None = None,
    ) -> tuple[bool, int]:
        """Atomic sliding window acquire operation.

        Args:
            key: Window key.
            window_size: Window size in seconds.
            sub_window_count: Number of sub-windows.
            limit: Request limit.
            tokens: Tokens to acquire.
            ttl: Entry TTL.

        Returns:
            Tuple of (allowed, remaining).
        """
        script = self._get_script("sliding_window", self.SLIDING_WINDOW_SCRIPT)
        result = script(
            keys=[self._make_key(key)],
            args=[
                window_size,
                sub_window_count,
                limit,
                tokens,
                current_time(),
                int(ttl or window_size * 2),
            ],
        )
        return bool(result[0]), int(result[1])


# =============================================================================
# Distributed Storage Interface
# =============================================================================


class DistributedStorage(RateLimitStorage[T], ABC):
    """Abstract base for distributed storage backends.

    Provides common interface for various distributed storage systems.
    """

    @abstractmethod
    def cas(
        self,
        key: str,
        expected: T | None,
        new_value: T,
        ttl: float | None = None,
    ) -> bool:
        """Compare-and-swap operation.

        Args:
            key: Storage key.
            expected: Expected current value (None for create).
            new_value: New value to set.
            ttl: Time-to-live.

        Returns:
            True if swap succeeded.
        """
        pass

    @abstractmethod
    def watch(self, key: str, callback: Any) -> None:
        """Watch key for changes.

        Args:
            key: Key to watch.
            callback: Callback on change.
        """
        pass


# =============================================================================
# Memcached Storage
# =============================================================================


class MemcachedStorage(RateLimitStorage[T]):
    """Memcached-based storage for distributed rate limiting.

    Uses CAS operations for atomic updates.

    Example:
        >>> from pymemcache.client import Client
        >>> client = Client(('localhost', 11211))
        >>> storage = MemcachedStorage(client)
    """

    def __init__(
        self,
        client: Any,
        prefix: str = "rl:",
        default_ttl: float = 3600.0,
    ) -> None:
        """Initialize Memcached storage.

        Args:
            client: Memcached client instance.
            prefix: Key prefix.
            default_ttl: Default TTL.
        """
        self._client = client
        self._prefix = prefix
        self._default_ttl = default_ttl

    def _make_key(self, key: str) -> str:
        """Create full Memcached key."""
        # Memcached has key length limits, hash long keys
        full_key = f"{self._prefix}{key}"
        if len(full_key) > 250:
            key_hash = hashlib.sha256(key.encode()).hexdigest()[:32]
            full_key = f"{self._prefix}h:{key_hash}"
        return full_key

    def get(self, key: str) -> T | None:
        """Get state for a key."""
        import json

        full_key = self._make_key(key)
        data = self._client.get(full_key)

        if data is None:
            return None

        try:
            parsed = json.loads(data)
            if "tokens" in parsed:
                return TokenBucketState(**parsed)  # type: ignore
            elif "count" in parsed:
                return WindowState(**parsed)  # type: ignore
            return parsed  # type: ignore
        except (json.JSONDecodeError, TypeError):
            return None

    def set(self, key: str, state: T, ttl: float | None = None) -> None:
        """Set state for a key."""
        import json

        full_key = self._make_key(key)
        ttl = int(ttl or self._default_ttl)

        if isinstance(state, (TokenBucketState, WindowState)):
            data = json.dumps({
                k: v for k, v in state.__dict__.items()
                if not k.startswith("_")
            })
        else:
            data = json.dumps(state)

        self._client.set(full_key, data, expire=ttl)

    def delete(self, key: str) -> bool:
        """Delete state for a key."""
        full_key = self._make_key(key)
        return self._client.delete(full_key)

    def increment(
        self,
        key: str,
        amount: int = 1,
        ttl: float | None = None,
    ) -> int:
        """Atomically increment a counter."""
        full_key = self._make_key(key)

        try:
            result = self._client.incr(full_key, amount)
            if result is None:
                # Key doesn't exist, initialize
                self._client.set(full_key, str(amount), expire=int(ttl or self._default_ttl))
                return amount
            return result
        except Exception:
            # Key doesn't exist or not a number
            self._client.set(full_key, str(amount), expire=int(ttl or self._default_ttl))
            return amount

    def get_with_lock(
        self,
        key: str,
        timeout: float = 1.0,
    ) -> tuple[T | None, Any]:
        """Get with CAS token."""
        import json

        full_key = self._make_key(key)
        result = self._client.gets(full_key)

        if result is None:
            return None, None

        data, cas_token = result
        try:
            parsed = json.loads(data)
            if "tokens" in parsed:
                state = TokenBucketState(**parsed)
            elif "count" in parsed:
                state = WindowState(**parsed)
            else:
                state = parsed
            return state, cas_token  # type: ignore
        except (json.JSONDecodeError, TypeError):
            return None, cas_token

    def set_with_lock(
        self,
        key: str,
        state: T,
        lock_token: Any,
        ttl: float | None = None,
    ) -> bool:
        """Set with CAS."""
        import json

        if lock_token is None:
            return False

        full_key = self._make_key(key)

        if isinstance(state, (TokenBucketState, WindowState)):
            data = json.dumps({
                k: v for k, v in state.__dict__.items()
                if not k.startswith("_")
            })
        else:
            data = json.dumps(state)

        try:
            return self._client.cas(
                full_key,
                data,
                lock_token,
                expire=int(ttl or self._default_ttl),
            )
        except Exception:
            return False


# =============================================================================
# Storage Factory
# =============================================================================


def create_storage(
    backend: str = "memory",
    **kwargs: Any,
) -> RateLimitStorage:
    """Create storage backend from configuration.

    Args:
        backend: Storage backend type ("memory", "redis", "memcached").
        **kwargs: Backend-specific configuration.

    Returns:
        Storage instance.

    Example:
        >>> storage = create_storage("memory")
        >>> storage = create_storage("redis", client=redis_client)
    """
    if backend == "memory":
        return MemoryStorage(**kwargs)
    elif backend == "redis":
        if "client" not in kwargs:
            raise ValueError("Redis storage requires 'client' argument")
        return RedisStorage(**kwargs)
    elif backend == "memcached":
        if "client" not in kwargs:
            raise ValueError("Memcached storage requires 'client' argument")
        return MemcachedStorage(**kwargs)
    else:
        raise ValueError(f"Unknown storage backend: {backend}")
