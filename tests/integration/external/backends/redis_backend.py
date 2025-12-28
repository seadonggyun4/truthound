"""Redis backend for integration tests.

This module provides Redis integration testing with support for:
- Docker containers (default)
- Local Redis instances
- Redis Cluster
- Redis Sentinel

Features:
    - Full Redis API testing
    - Pub/Sub testing
    - Cluster operations
    - Distributed locking

Usage:
    >>> config = RedisConfig.from_env()
    >>> with RedisBackend(config) as backend:
    ...     backend.client.set("key", "value")
    ...     assert backend.client.get("key") == "value"
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, ClassVar, TYPE_CHECKING

from tests.integration.external.base import (
    ExternalServiceBackend,
    HealthCheckResult,
    ProviderType,
    ServiceCategory,
    ServiceConfig,
)
from tests.integration.external.providers.docker_provider import DockerContainerConfig

if TYPE_CHECKING:
    from redis import Redis

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class RedisConfig(DockerContainerConfig):
    """Redis-specific configuration.

    Attributes:
        password: Redis password (optional)
        db: Database number (default: 0)
        decode_responses: Decode responses to strings
        ssl: Use SSL/TLS
        cluster_mode: Enable cluster mode
        sentinel_master: Sentinel master name
        sentinels: List of sentinel addresses
        max_connections: Maximum pool connections
    """

    password: str | None = None
    db: int = 0
    decode_responses: bool = True
    ssl: bool = False
    cluster_mode: bool = False
    sentinel_master: str | None = None
    sentinels: list[tuple[str, int]] = field(default_factory=list)
    max_connections: int = 10

    def __post_init__(self) -> None:
        """Set Redis-specific defaults."""
        self.name = self.name or "redis"
        self.category = ServiceCategory.CACHE
        self.image = self.image or "redis"
        self.tag = self.tag or "7-alpine"
        self.ports = self.ports or {"6379/tcp": None}
        self.health_cmd = self.health_cmd or "redis-cli ping"

        # Add password to environment if set
        if self.password:
            self.environment["REDIS_PASSWORD"] = self.password

    @classmethod
    def from_env(cls, name: str = "redis") -> "RedisConfig":
        """Create configuration from environment variables.

        Environment variables:
            TRUTHOUND_TEST_REDIS_HOST
            TRUTHOUND_TEST_REDIS_PORT
            TRUTHOUND_TEST_REDIS_PASSWORD
            TRUTHOUND_TEST_REDIS_DB
            TRUTHOUND_TEST_REDIS_SSL
        """
        prefix = "TRUTHOUND_TEST_REDIS"

        return cls(
            name=name,
            category=ServiceCategory.CACHE,
            host=os.getenv(f"{prefix}_HOST"),
            port=int(os.getenv(f"{prefix}_PORT", "0")) or None,
            password=os.getenv(f"{prefix}_PASSWORD"),
            db=int(os.getenv(f"{prefix}_DB", "0")),
            ssl=os.getenv(f"{prefix}_SSL", "false").lower() == "true",
            timeout_seconds=int(os.getenv(f"{prefix}_TIMEOUT", "30")),
        )


# =============================================================================
# Redis Backend
# =============================================================================


class RedisBackend(ExternalServiceBackend[RedisConfig, "Redis"]):
    """Redis test backend.

    Provides Redis connection and operations for integration testing.

    Features:
        - Automatic Docker container management
        - Connection pooling
        - Health checking
        - Support for standalone, cluster, and sentinel modes
    """

    service_name: ClassVar[str] = "redis"
    service_category: ClassVar[ServiceCategory] = ServiceCategory.CACHE
    default_port: ClassVar[int] = 6379
    default_image: ClassVar[str] = "redis:7-alpine"

    def __init__(
        self,
        config: RedisConfig | None = None,
        provider: Any = None,
    ) -> None:
        """Initialize Redis backend.

        Args:
            config: Redis configuration (uses defaults if not provided)
            provider: Service provider (auto-detected if not provided)
        """
        if config is None:
            config = RedisConfig.from_env()
        super().__init__(config, provider)

    def _create_client(self) -> "Redis":
        """Create Redis client."""
        try:
            import redis
        except ImportError:
            raise ImportError(
                "redis package not installed. Run: pip install redis"
            )

        config = self.config

        # Check for cluster mode
        if config.cluster_mode:
            return self._create_cluster_client(redis)

        # Check for sentinel mode
        if config.sentinel_master and config.sentinels:
            return self._create_sentinel_client(redis)

        # Standard client
        client = redis.Redis(
            host=self.host,
            port=self.port,
            password=config.password,
            db=config.db,
            decode_responses=config.decode_responses,
            ssl=config.ssl,
            socket_timeout=config.timeout_seconds,
            socket_connect_timeout=config.timeout_seconds,
            max_connections=config.max_connections,
        )

        # Test connection
        client.ping()
        return client

    def _create_cluster_client(self, redis_module: Any) -> "Redis":
        """Create Redis Cluster client."""
        from redis.cluster import RedisCluster

        client = RedisCluster(
            host=self.host,
            port=self.port,
            password=self.config.password,
            decode_responses=self.config.decode_responses,
        )
        return client

    def _create_sentinel_client(self, redis_module: Any) -> "Redis":
        """Create Redis Sentinel client."""
        from redis.sentinel import Sentinel

        sentinel = Sentinel(
            self.config.sentinels,
            socket_timeout=self.config.timeout_seconds,
        )
        master = sentinel.master_for(
            self.config.sentinel_master,
            password=self.config.password,
            decode_responses=self.config.decode_responses,
        )
        return master

    def _close_client(self) -> None:
        """Close Redis client."""
        if self._client is not None:
            self._client.close()

    def _perform_health_check(self) -> HealthCheckResult:
        """Perform Redis health check."""
        if self._client is None:
            return HealthCheckResult.failure("Client not connected")

        try:
            response = self._client.ping()
            if response:
                info = self._client.info("server")
                return HealthCheckResult.success(
                    "Redis healthy",
                    redis_version=info.get("redis_version"),
                    connected_clients=info.get("connected_clients"),
                )
            return HealthCheckResult.failure("Ping returned False")

        except Exception as e:
            return HealthCheckResult.failure(str(e))

    # -------------------------------------------------------------------------
    # Redis-Specific Operations
    # -------------------------------------------------------------------------

    def flush_all(self) -> bool:
        """Flush all keys in the current database."""
        if self._client is None:
            return False
        self._client.flushdb()
        return True

    def get_info(self, section: str | None = None) -> dict[str, Any]:
        """Get Redis server info."""
        if self._client is None:
            return {}
        return self._client.info(section)

    def get_key_count(self) -> int:
        """Get number of keys in current database."""
        if self._client is None:
            return 0
        return self._client.dbsize()

    def execute_command(self, *args: Any) -> Any:
        """Execute a raw Redis command."""
        if self._client is None:
            raise RuntimeError("Client not connected")
        return self._client.execute_command(*args)


# =============================================================================
# Test Helpers
# =============================================================================


def create_redis_backend(
    provider_type: ProviderType = ProviderType.DOCKER,
) -> RedisBackend:
    """Create a Redis backend with specified provider.

    Convenience function for creating Redis backends in tests.

    Args:
        provider_type: Provider to use (DOCKER, LOCAL, MOCK)

    Returns:
        Configured RedisBackend
    """
    config = RedisConfig.from_env()
    config.provider = provider_type

    from tests.integration.external.providers import get_provider
    provider = get_provider(provider_type, config)

    return RedisBackend(config, provider)
