"""Redis integration for distributed timeout coordination.

This module provides Redis-backed coordination for distributed timeout
management, enabling:
- Distributed locks
- Deadline sharing across nodes
- Leader election
- Heartbeat coordination

Note: Requires the 'redis' package for actual Redis connectivity.
This module provides the interface and mock implementations for testing.

Example:
    from truthound.validators.timeout.advanced.redis_backend import (
        RedisCoordinator,
        RedisConfig,
        create_redis_coordinator,
    )

    # Create coordinator
    config = RedisConfig(
        host="localhost",
        port=6379,
        db=0,
    )
    coordinator = create_redis_coordinator(config)

    # Use distributed lock
    async with coordinator.lock("my_operation") as lock:
        if lock.acquired:
            result = await process()

    # Create distributed deadline
    deadline = await coordinator.create_deadline(
        timeout_seconds=60,
        operation_id="batch_validation",
    )
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import threading
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, AsyncGenerator

logger = logging.getLogger(__name__)


def is_redis_available() -> bool:
    """Check if redis package is available.

    Returns:
        True if redis package is installed
    """
    try:
        import redis
        return True
    except ImportError:
        return False


class LockStatus(str, Enum):
    """Status of a distributed lock."""

    ACQUIRED = "acquired"
    WAITING = "waiting"
    FAILED = "failed"
    RELEASED = "released"


@dataclass
class RedisConfig:
    """Configuration for Redis connection.

    Attributes:
        host: Redis host
        port: Redis port
        db: Redis database number
        password: Optional password
        ssl: Whether to use SSL
        socket_timeout: Socket timeout in seconds
        connection_pool_size: Maximum connections
        key_prefix: Prefix for all keys
        lock_timeout: Default lock timeout in seconds
        heartbeat_interval: Heartbeat interval in seconds
    """

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str | None = None
    ssl: bool = False
    socket_timeout: float = 5.0
    connection_pool_size: int = 10
    key_prefix: str = "truthound:"
    lock_timeout: float = 30.0
    heartbeat_interval: float = 5.0

    def get_url(self) -> str:
        """Get Redis URL.

        Returns:
            Redis connection URL
        """
        protocol = "rediss" if self.ssl else "redis"
        auth = f":{self.password}@" if self.password else ""
        return f"{protocol}://{auth}{self.host}:{self.port}/{self.db}"


@dataclass
class RedisLock:
    """Distributed lock using Redis.

    Attributes:
        name: Lock name
        token: Unique token for this lock holder
        acquired: Whether lock is acquired
        acquired_at: When lock was acquired
        expires_at: When lock expires
    """

    name: str
    token: str = field(default_factory=lambda: uuid.uuid4().hex)
    acquired: bool = False
    acquired_at: datetime | None = None
    expires_at: datetime | None = None

    @property
    def is_valid(self) -> bool:
        """Check if lock is still valid."""
        if not self.acquired or not self.expires_at:
            return False
        return datetime.now(timezone.utc) < self.expires_at

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "token": self.token,
            "acquired": self.acquired,
            "acquired_at": self.acquired_at.isoformat() if self.acquired_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "is_valid": self.is_valid,
        }


@dataclass
class RedisDeadline:
    """Distributed deadline stored in Redis.

    Attributes:
        deadline_id: Unique deadline identifier
        deadline_utc: Absolute deadline
        owner_node: Node that created the deadline
        operation_id: Operation being coordinated
        status: Current status
        created_at: Creation timestamp
    """

    deadline_id: str
    deadline_utc: datetime
    owner_node: str
    operation_id: str = ""
    status: str = "active"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def remaining_seconds(self) -> float:
        """Get remaining time until deadline."""
        now = datetime.now(timezone.utc)
        remaining = (self.deadline_utc - now).total_seconds()
        return max(0.0, remaining)

    @property
    def is_expired(self) -> bool:
        """Check if deadline has passed."""
        return self.remaining_seconds <= 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "deadline_id": self.deadline_id,
            "deadline_utc": self.deadline_utc.isoformat(),
            "owner_node": self.owner_node,
            "operation_id": self.operation_id,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
            "remaining_seconds": self.remaining_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RedisDeadline":
        """Create from dictionary."""
        deadline_utc = datetime.fromisoformat(data["deadline_utc"])
        if deadline_utc.tzinfo is None:
            deadline_utc = deadline_utc.replace(tzinfo=timezone.utc)

        created = data.get("created_at")
        if created:
            created = datetime.fromisoformat(created)
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
        else:
            created = datetime.now(timezone.utc)

        return cls(
            deadline_id=data["deadline_id"],
            deadline_utc=deadline_utc,
            owner_node=data["owner_node"],
            operation_id=data.get("operation_id", ""),
            status=data.get("status", "active"),
            created_at=created,
            metadata=data.get("metadata", {}),
        )


class RedisBackend(ABC):
    """Abstract base for Redis backend implementations."""

    @abstractmethod
    async def set(
        self,
        key: str,
        value: str,
        ex: int | None = None,
        nx: bool = False,
    ) -> bool:
        """Set a key.

        Args:
            key: Key name
            value: Value to set
            ex: Expiration in seconds
            nx: Only set if not exists

        Returns:
            True if set was successful
        """
        pass

    @abstractmethod
    async def get(self, key: str) -> str | None:
        """Get a key.

        Args:
            key: Key name

        Returns:
            Value or None
        """
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete a key.

        Args:
            key: Key name

        Returns:
            True if deleted
        """
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists.

        Args:
            key: Key name

        Returns:
            True if exists
        """
        pass

    @abstractmethod
    async def expire(self, key: str, seconds: int) -> bool:
        """Set key expiration.

        Args:
            key: Key name
            seconds: TTL in seconds

        Returns:
            True if expiration was set
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the connection."""
        pass


class InMemoryRedisBackend(RedisBackend):
    """In-memory implementation for testing.

    This provides a mock Redis backend for testing without
    requiring an actual Redis server.
    """

    def __init__(self) -> None:
        """Initialize in-memory backend."""
        self._store: dict[str, tuple[str, float | None]] = {}
        self._lock = threading.Lock()

    async def set(
        self,
        key: str,
        value: str,
        ex: int | None = None,
        nx: bool = False,
    ) -> bool:
        """Set a key in memory."""
        with self._lock:
            if nx and key in self._store:
                # Check if not expired
                _, expires = self._store[key]
                if expires is None or expires > time.time():
                    return False

            expires_at = time.time() + ex if ex else None
            self._store[key] = (value, expires_at)
            return True

    async def get(self, key: str) -> str | None:
        """Get a key from memory."""
        with self._lock:
            if key not in self._store:
                return None

            value, expires = self._store[key]
            if expires and expires <= time.time():
                del self._store[key]
                return None

            return value

    async def delete(self, key: str) -> bool:
        """Delete a key from memory."""
        with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in memory."""
        with self._lock:
            if key not in self._store:
                return False

            _, expires = self._store[key]
            if expires and expires <= time.time():
                del self._store[key]
                return False

            return True

    async def expire(self, key: str, seconds: int) -> bool:
        """Set expiration in memory."""
        with self._lock:
            if key not in self._store:
                return False

            value, _ = self._store[key]
            self._store[key] = (value, time.time() + seconds)
            return True

    async def close(self) -> None:
        """Clear memory."""
        with self._lock:
            self._store.clear()


class RealRedisBackend(RedisBackend):
    """Real Redis backend using redis-py.

    Requires the 'redis' package to be installed.
    """

    def __init__(self, config: RedisConfig):
        """Initialize Redis backend.

        Args:
            config: Redis configuration
        """
        self.config = config
        self._client: Any = None
        self._initialized = False

    async def _ensure_connected(self) -> None:
        """Ensure Redis client is connected."""
        if self._initialized:
            return

        try:
            import redis.asyncio as aioredis
        except ImportError:
            raise ImportError(
                "redis package is required for Redis backend. "
                "Install with: pip install redis"
            )

        self._client = aioredis.Redis(
            host=self.config.host,
            port=self.config.port,
            db=self.config.db,
            password=self.config.password,
            ssl=self.config.ssl,
            socket_timeout=self.config.socket_timeout,
            max_connections=self.config.connection_pool_size,
        )
        self._initialized = True

    async def set(
        self,
        key: str,
        value: str,
        ex: int | None = None,
        nx: bool = False,
    ) -> bool:
        """Set a key in Redis."""
        await self._ensure_connected()
        result = await self._client.set(key, value, ex=ex, nx=nx)
        return result is True or result == "OK"

    async def get(self, key: str) -> str | None:
        """Get a key from Redis."""
        await self._ensure_connected()
        result = await self._client.get(key)
        if result is None:
            return None
        return result.decode() if isinstance(result, bytes) else result

    async def delete(self, key: str) -> bool:
        """Delete a key from Redis."""
        await self._ensure_connected()
        result = await self._client.delete(key)
        return result > 0

    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis."""
        await self._ensure_connected()
        result = await self._client.exists(key)
        return result > 0

    async def expire(self, key: str, seconds: int) -> bool:
        """Set expiration in Redis."""
        await self._ensure_connected()
        result = await self._client.expire(key, seconds)
        return result is True

    async def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._initialized = False


class RedisCoordinator:
    """Coordinator for distributed timeout using Redis.

    Provides distributed coordination primitives:
    - Distributed locks
    - Deadline sharing
    - Node registration
    - Heartbeat monitoring

    Example:
        coordinator = RedisCoordinator(config)
        await coordinator.start()

        async with coordinator.lock("operation") as lock:
            if lock.acquired:
                await do_work()

        await coordinator.stop()
    """

    def __init__(
        self,
        config: RedisConfig | None = None,
        backend: RedisBackend | None = None,
    ):
        """Initialize coordinator.

        Args:
            config: Redis configuration
            backend: Optional backend (for testing)
        """
        self.config = config or RedisConfig()
        self._backend = backend
        self._node_id = f"node-{uuid.uuid4().hex[:8]}"
        self._running = False
        self._heartbeat_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the coordinator."""
        if self._backend is None:
            if is_redis_available():
                self._backend = RealRedisBackend(self.config)
            else:
                logger.warning("Redis package not available, using in-memory backend")
                self._backend = InMemoryRedisBackend()

        # Register node
        node_key = f"{self.config.key_prefix}nodes:{self._node_id}"
        await self._backend.set(
            node_key,
            datetime.now(timezone.utc).isoformat(),
            ex=int(self.config.lock_timeout),
        )

        self._running = True
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        logger.info(f"Redis coordinator started: {self._node_id}")

    async def stop(self) -> None:
        """Stop the coordinator."""
        self._running = False

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        # Unregister node
        if self._backend:
            node_key = f"{self.config.key_prefix}nodes:{self._node_id}"
            await self._backend.delete(node_key)
            await self._backend.close()

        logger.info(f"Redis coordinator stopped: {self._node_id}")

    async def _heartbeat_loop(self) -> None:
        """Background heartbeat loop."""
        while self._running:
            try:
                node_key = f"{self.config.key_prefix}nodes:{self._node_id}"
                await self._backend.set(
                    node_key,
                    datetime.now(timezone.utc).isoformat(),
                    ex=int(self.config.lock_timeout),
                )
            except Exception as e:
                logger.warning(f"Heartbeat failed: {e}")

            await asyncio.sleep(self.config.heartbeat_interval)

    @asynccontextmanager
    async def lock(
        self,
        name: str,
        timeout_seconds: float | None = None,
        blocking: bool = True,
        retry_interval: float = 0.1,
    ) -> AsyncGenerator[RedisLock, None]:
        """Acquire a distributed lock.

        Args:
            name: Lock name
            timeout_seconds: Lock timeout (None = config default)
            blocking: Whether to block waiting for lock
            retry_interval: Seconds between retry attempts

        Yields:
            RedisLock
        """
        lock = RedisLock(name=name)
        lock_key = f"{self.config.key_prefix}locks:{name}"
        timeout = timeout_seconds or self.config.lock_timeout

        try:
            # Try to acquire lock
            deadline = time.time() + timeout if blocking else time.time()

            while time.time() < deadline:
                acquired = await self._backend.set(
                    lock_key,
                    lock.token,
                    ex=int(timeout),
                    nx=True,
                )

                if acquired:
                    lock.acquired = True
                    lock.acquired_at = datetime.now(timezone.utc)
                    lock.expires_at = lock.acquired_at + timedelta(seconds=timeout)
                    break

                if not blocking:
                    break

                await asyncio.sleep(retry_interval)

            yield lock

        finally:
            # Release lock if we acquired it
            if lock.acquired:
                # Verify we still own the lock before deleting
                current = await self._backend.get(lock_key)
                if current == lock.token:
                    await self._backend.delete(lock_key)

    async def create_deadline(
        self,
        timeout_seconds: float,
        operation_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> RedisDeadline:
        """Create a distributed deadline.

        Args:
            timeout_seconds: Timeout in seconds
            operation_id: Operation identifier
            metadata: Additional metadata

        Returns:
            RedisDeadline
        """
        import json

        deadline_id = f"deadline-{uuid.uuid4().hex[:12]}"
        deadline_utc = datetime.now(timezone.utc) + timedelta(seconds=timeout_seconds)

        deadline = RedisDeadline(
            deadline_id=deadline_id,
            deadline_utc=deadline_utc,
            owner_node=self._node_id,
            operation_id=operation_id,
            metadata=metadata or {},
        )

        deadline_key = f"{self.config.key_prefix}deadlines:{deadline_id}"
        await self._backend.set(
            deadline_key,
            json.dumps(deadline.to_dict()),
            ex=int(timeout_seconds) + 60,  # Keep a bit longer than deadline
        )

        return deadline

    async def get_deadline(self, deadline_id: str) -> RedisDeadline | None:
        """Get a deadline by ID.

        Args:
            deadline_id: Deadline identifier

        Returns:
            RedisDeadline or None
        """
        import json

        deadline_key = f"{self.config.key_prefix}deadlines:{deadline_id}"
        data = await self._backend.get(deadline_key)

        if data is None:
            return None

        return RedisDeadline.from_dict(json.loads(data))

    async def complete_deadline(
        self,
        deadline_id: str,
        status: str = "completed",
    ) -> bool:
        """Mark a deadline as completed.

        Args:
            deadline_id: Deadline identifier
            status: Completion status

        Returns:
            True if updated
        """
        import json

        deadline_key = f"{self.config.key_prefix}deadlines:{deadline_id}"
        data = await self._backend.get(deadline_key)

        if data is None:
            return False

        deadline = RedisDeadline.from_dict(json.loads(data))
        deadline.status = status

        # Update with short TTL
        await self._backend.set(
            deadline_key,
            json.dumps(deadline.to_dict()),
            ex=60,  # Keep for 1 minute after completion
        )

        return True

    async def extend_deadline(
        self,
        deadline_id: str,
        additional_seconds: float,
    ) -> RedisDeadline | None:
        """Extend a deadline.

        Args:
            deadline_id: Deadline identifier
            additional_seconds: Seconds to add

        Returns:
            Updated deadline or None
        """
        import json

        deadline_key = f"{self.config.key_prefix}deadlines:{deadline_id}"
        data = await self._backend.get(deadline_key)

        if data is None:
            return None

        deadline = RedisDeadline.from_dict(json.loads(data))
        deadline.deadline_utc += timedelta(seconds=additional_seconds)

        new_ttl = int(deadline.remaining_seconds) + 60
        await self._backend.set(
            deadline_key,
            json.dumps(deadline.to_dict()),
            ex=new_ttl,
        )

        return deadline

    def get_node_id(self) -> str:
        """Get this node's ID.

        Returns:
            Node ID
        """
        return self._node_id


def create_redis_coordinator(
    config: RedisConfig | None = None,
    use_memory: bool = False,
) -> RedisCoordinator:
    """Create a Redis coordinator.

    Args:
        config: Redis configuration
        use_memory: Force in-memory backend

    Returns:
        RedisCoordinator
    """
    backend = InMemoryRedisBackend() if use_memory else None
    return RedisCoordinator(config, backend)
