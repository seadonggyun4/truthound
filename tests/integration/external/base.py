"""Base classes for external service integration tests.

This module provides the abstract base classes and protocols that define
the interface for external service integration testing.

Architecture Overview:

    ServiceProvider (Protocol)
        - Abstracts service provisioning (Docker, Local, Cloud, Mock)
        - Handles lifecycle: start, stop, health check
        - Provider-specific configuration

    ExternalServiceBackend (ABC)
        - Platform-specific client creation
        - Connection management with retry logic
        - Metrics collection and cost tracking

    ServiceRegistry (Singleton)
        - Central registry of available services
        - Provider discovery and caching
        - Service dependency resolution

Design Decisions:
    1. Provider Pattern: Same test can run against different providers
    2. Health Check Protocol: Uniform health checking across all services
    3. Context Managers: Automatic cleanup on test completion
    4. Cost Tracking: Cloud services track estimated costs
"""

from __future__ import annotations

import abc
import atexit
import logging
import os
import threading
import time
import uuid
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Iterator,
    Protocol,
    TypeVar,
    runtime_checkable,
)

if TYPE_CHECKING:
    from docker import DockerClient

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================


class ServiceStatus(Enum):
    """Status of an external service."""

    UNKNOWN = auto()
    STARTING = auto()
    RUNNING = auto()
    HEALTHY = auto()
    UNHEALTHY = auto()
    STOPPING = auto()
    STOPPED = auto()
    FAILED = auto()
    UNAVAILABLE = auto()


class ServiceCategory(Enum):
    """Categories of external services."""

    CACHE = "cache"  # Redis, Memcached
    MESSAGE_QUEUE = "message_queue"  # Kafka, RabbitMQ
    SEARCH = "search"  # Elasticsearch
    LOGGING = "logging"  # Loki, Fluentd
    SECRETS = "secrets"  # Vault, Cloud KMS
    STORAGE = "storage"  # S3, GCS, MinIO
    DATABASE = "database"  # PostgreSQL, MySQL
    TMS = "tms"  # Translation Management Systems
    MONITORING = "monitoring"  # Prometheus


class ProviderType(Enum):
    """Types of service providers."""

    DOCKER = "docker"
    LOCAL = "local"
    CLOUD = "cloud"
    MOCK = "mock"


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class ServiceConfig:
    """Configuration for an external service.

    Attributes:
        name: Service name (e.g., "redis", "elasticsearch")
        category: Service category
        provider: Preferred provider type
        host: Service host (override)
        port: Service port (override)
        timeout_seconds: Connection timeout
        retry_attempts: Number of retry attempts
        retry_delay_seconds: Delay between retries
        health_check_interval: Health check interval in seconds
        cleanup_on_exit: Whether to clean up on process exit
        environment: Additional environment variables
        labels: Metadata labels for the service
    """

    name: str
    category: ServiceCategory
    provider: ProviderType = ProviderType.DOCKER
    host: str | None = None
    port: int | None = None
    timeout_seconds: int = 30
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.0
    health_check_interval: float = 5.0
    cleanup_on_exit: bool = True
    environment: dict[str, str] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)

    # Cost tracking for cloud services
    max_cost_usd: float = 1.0
    track_cost: bool = True

    @classmethod
    def from_env(cls, name: str, category: ServiceCategory) -> "ServiceConfig":
        """Create configuration from environment variables.

        Environment variable naming convention:
            TRUTHOUND_TEST_{NAME}_{SETTING}

        Example:
            TRUTHOUND_TEST_REDIS_HOST=localhost
            TRUTHOUND_TEST_REDIS_PORT=6379
        """
        prefix = f"TRUTHOUND_TEST_{name.upper()}"

        provider_str = os.getenv(f"{prefix}_PROVIDER", "docker")
        provider = ProviderType(provider_str.lower())

        return cls(
            name=name,
            category=category,
            provider=provider,
            host=os.getenv(f"{prefix}_HOST"),
            port=int(os.getenv(f"{prefix}_PORT", "0")) or None,
            timeout_seconds=int(os.getenv(f"{prefix}_TIMEOUT", "30")),
            retry_attempts=int(os.getenv(f"{prefix}_RETRIES", "3")),
            cleanup_on_exit=os.getenv(f"{prefix}_CLEANUP", "true").lower() == "true",
            max_cost_usd=float(os.getenv(f"{prefix}_MAX_COST", "1.0")),
        )


@dataclass
class ServiceMetrics:
    """Metrics collected for a service during testing.

    Attributes:
        operation_count: Number of operations performed
        total_duration_seconds: Total time spent on operations
        error_count: Number of errors encountered
        bytes_transferred: Total bytes transferred (if applicable)
        estimated_cost_usd: Estimated cost for cloud services
        operations: List of operation details
    """

    operation_count: int = 0
    total_duration_seconds: float = 0.0
    error_count: int = 0
    bytes_transferred: int = 0
    estimated_cost_usd: float = 0.0
    operations: list[dict[str, Any]] = field(default_factory=list)

    def record_operation(
        self,
        operation: str,
        duration_seconds: float,
        success: bool = True,
        bytes_count: int = 0,
        cost_usd: float = 0.0,
        error: str | None = None,
        **metadata: Any,
    ) -> None:
        """Record an operation."""
        self.operation_count += 1
        self.total_duration_seconds += duration_seconds
        self.bytes_transferred += bytes_count
        self.estimated_cost_usd += cost_usd

        if not success:
            self.error_count += 1

        self.operations.append(
            {
                "operation": operation,
                "duration_seconds": duration_seconds,
                "success": success,
                "bytes_count": bytes_count,
                "cost_usd": cost_usd,
                "error": error,
                "timestamp": datetime.utcnow().isoformat(),
                **metadata,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "operation_count": self.operation_count,
            "total_duration_seconds": self.total_duration_seconds,
            "error_count": self.error_count,
            "bytes_transferred": self.bytes_transferred,
            "estimated_cost_usd": self.estimated_cost_usd,
            "avg_operation_duration": (
                self.total_duration_seconds / self.operation_count
                if self.operation_count > 0
                else 0
            ),
            "error_rate": (
                self.error_count / self.operation_count
                if self.operation_count > 0
                else 0
            ),
        }


# =============================================================================
# Health Check Protocol
# =============================================================================


@dataclass
class HealthCheckResult:
    """Result of a health check.

    Attributes:
        healthy: Whether the service is healthy
        message: Status message
        latency_ms: Health check latency in milliseconds
        details: Additional health details
        timestamp: When the check was performed
    """

    healthy: bool
    message: str = ""
    latency_ms: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @classmethod
    def success(cls, message: str = "OK", latency_ms: float = 0.0, **details: Any) -> "HealthCheckResult":
        """Create a successful health check result."""
        return cls(healthy=True, message=message, latency_ms=latency_ms, details=details)

    @classmethod
    def failure(cls, message: str, latency_ms: float = 0.0, **details: Any) -> "HealthCheckResult":
        """Create a failed health check result."""
        return cls(healthy=False, message=message, latency_ms=latency_ms, details=details)


@runtime_checkable
class HealthCheckable(Protocol):
    """Protocol for services that support health checking."""

    def health_check(self) -> HealthCheckResult:
        """Perform a health check.

        Returns:
            HealthCheckResult with status and details.
        """
        ...


# =============================================================================
# Service Provider Protocol
# =============================================================================


ConfigT = TypeVar("ConfigT", bound=ServiceConfig)
ClientT = TypeVar("ClientT")


@runtime_checkable
class ServiceProvider(Protocol[ConfigT]):
    """Protocol for service providers.

    Service providers handle the lifecycle of external services,
    abstracting the difference between Docker, local, cloud, and mock services.
    """

    provider_type: ClassVar[ProviderType]

    def is_available(self) -> bool:
        """Check if this provider is available."""
        ...

    def start_service(self, config: ConfigT) -> dict[str, Any]:
        """Start the service.

        Returns:
            Connection info dictionary with host, port, etc.
        """
        ...

    def stop_service(self, config: ConfigT) -> None:
        """Stop the service."""
        ...

    def get_connection_info(self, config: ConfigT) -> dict[str, Any]:
        """Get connection information for the service.

        Returns:
            Dictionary with host, port, and other connection details.
        """
        ...

    def health_check(self, config: ConfigT) -> HealthCheckResult:
        """Check if the service is healthy."""
        ...


# =============================================================================
# Abstract Service Backend
# =============================================================================


class ExternalServiceBackend(abc.ABC, Generic[ConfigT, ClientT]):
    """Abstract base class for external service test backends.

    This class defines the interface that all external service backends
    must implement. It provides:
    - Connection management with retry logic
    - Health checking and wait-for-ready
    - Metrics collection
    - Automatic cleanup

    Type Parameters:
        ConfigT: Service configuration type
        ClientT: Client type for the service

    Example:
        >>> class RedisBackend(ExternalServiceBackend[RedisConfig, Redis]):
        ...     service_name = "redis"
        ...     service_category = ServiceCategory.CACHE
        ...
        ...     def _create_client(self) -> Redis:
        ...         return Redis(host=self.host, port=self.port)
    """

    # Class-level attributes (must be overridden)
    service_name: ClassVar[str] = "unknown"
    service_category: ClassVar[ServiceCategory] = ServiceCategory.CACHE
    default_port: ClassVar[int] = 0
    default_image: ClassVar[str] = ""

    def __init__(
        self,
        config: ConfigT,
        provider: ServiceProvider[ConfigT] | None = None,
    ) -> None:
        """Initialize the backend.

        Args:
            config: Service configuration
            provider: Optional service provider (auto-detected if not provided)
        """
        self.config = config
        self._provider = provider
        self._client: ClientT | None = None
        self._status = ServiceStatus.UNKNOWN
        self._metrics = ServiceMetrics()
        self._connection_info: dict[str, Any] = {}
        self._instance_id = uuid.uuid4().hex[:8]

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def status(self) -> ServiceStatus:
        """Get current service status."""
        return self._status

    @property
    def is_running(self) -> bool:
        """Check if service is running."""
        return self._status in (ServiceStatus.RUNNING, ServiceStatus.HEALTHY)

    @property
    def is_healthy(self) -> bool:
        """Check if service is healthy."""
        return self._status == ServiceStatus.HEALTHY

    @property
    def host(self) -> str:
        """Get service host."""
        return self._connection_info.get("host", self.config.host or "localhost")

    @property
    def port(self) -> int:
        """Get service port."""
        return self._connection_info.get("port", self.config.port or self.default_port)

    @property
    def client(self) -> ClientT:
        """Get the service client.

        Raises:
            RuntimeError: If not connected.
        """
        if self._client is None:
            raise RuntimeError(f"{self.service_name} client not connected")
        return self._client

    @property
    def metrics(self) -> ServiceMetrics:
        """Get collected metrics."""
        return self._metrics

    # -------------------------------------------------------------------------
    # Provider Management
    # -------------------------------------------------------------------------

    def get_provider(self) -> ServiceProvider[ConfigT]:
        """Get or create the service provider."""
        if self._provider is not None:
            return self._provider

        # Auto-detect provider
        from tests.integration.external.providers import get_provider

        self._provider = get_provider(self.config.provider, self.config)
        return self._provider

    # -------------------------------------------------------------------------
    # Lifecycle Management
    # -------------------------------------------------------------------------

    def start(self) -> bool:
        """Start the service and connect.

        Returns:
            True if successful, False otherwise.
        """
        if self.is_running:
            return True

        try:
            self._status = ServiceStatus.STARTING
            logger.info(f"[{self.service_name}] Starting service...")

            # Start the service via provider
            provider = self.get_provider()
            self._connection_info = provider.start_service(self.config)

            self._status = ServiceStatus.RUNNING
            logger.info(
                f"[{self.service_name}] Service started at "
                f"{self.host}:{self.port}"
            )

            # Connect client
            return self.connect()

        except Exception as e:
            self._status = ServiceStatus.FAILED
            logger.error(f"[{self.service_name}] Failed to start: {e}")
            return False

    def stop(self) -> None:
        """Stop the service."""
        if self._status in (ServiceStatus.STOPPED, ServiceStatus.UNKNOWN):
            return

        try:
            self._status = ServiceStatus.STOPPING
            logger.info(f"[{self.service_name}] Stopping service...")

            # Disconnect client
            self.disconnect()

            # Stop service via provider
            if self._provider is not None:
                self._provider.stop_service(self.config)

            self._status = ServiceStatus.STOPPED
            logger.info(f"[{self.service_name}] Service stopped")

        except Exception as e:
            logger.warning(f"[{self.service_name}] Error stopping: {e}")
            self._status = ServiceStatus.STOPPED

    def connect(self) -> bool:
        """Connect to the service.

        Returns:
            True if connected successfully.
        """
        if self._client is not None:
            return True

        try:
            self._client = self._create_client()
            logger.info(f"[{self.service_name}] Connected")
            return True

        except Exception as e:
            logger.error(f"[{self.service_name}] Connection failed: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from the service."""
        if self._client is not None:
            try:
                self._close_client()
            except Exception as e:
                logger.warning(f"[{self.service_name}] Error disconnecting: {e}")
            finally:
                self._client = None
                logger.info(f"[{self.service_name}] Disconnected")

    def wait_until_ready(
        self,
        timeout_seconds: float | None = None,
        check_interval: float | None = None,
    ) -> bool:
        """Wait until the service is ready.

        Args:
            timeout_seconds: Maximum time to wait
            check_interval: Time between checks

        Returns:
            True if service became ready, False if timeout.
        """
        timeout = timeout_seconds or self.config.timeout_seconds
        interval = check_interval or self.config.health_check_interval

        start_time = time.time()
        while time.time() - start_time < timeout:
            result = self.health_check()
            if result.healthy:
                self._status = ServiceStatus.HEALTHY
                logger.info(
                    f"[{self.service_name}] Service ready "
                    f"(latency: {result.latency_ms:.1f}ms)"
                )
                return True

            time.sleep(interval)

        logger.warning(f"[{self.service_name}] Timeout waiting for service")
        self._status = ServiceStatus.UNHEALTHY
        return False

    # -------------------------------------------------------------------------
    # Health Checking
    # -------------------------------------------------------------------------

    def health_check(self) -> HealthCheckResult:
        """Perform a health check.

        Returns:
            HealthCheckResult with status and details.
        """
        start_time = time.time()
        try:
            result = self._perform_health_check()
            result.latency_ms = (time.time() - start_time) * 1000
            return result
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return HealthCheckResult.failure(str(e), latency_ms=latency_ms)

    # -------------------------------------------------------------------------
    # Abstract Methods
    # -------------------------------------------------------------------------

    @abc.abstractmethod
    def _create_client(self) -> ClientT:
        """Create the service client.

        Returns:
            Connected client instance.
        """
        ...

    @abc.abstractmethod
    def _close_client(self) -> None:
        """Close the service client."""
        ...

    @abc.abstractmethod
    def _perform_health_check(self) -> HealthCheckResult:
        """Perform service-specific health check.

        Returns:
            HealthCheckResult with status.
        """
        ...

    # -------------------------------------------------------------------------
    # Context Manager
    # -------------------------------------------------------------------------

    @contextmanager
    def session(self) -> Iterator["ExternalServiceBackend[ConfigT, ClientT]"]:
        """Context manager for a test session.

        Automatically starts and stops the service.

        Example:
            >>> with backend.session() as svc:
            ...     svc.client.set("key", "value")
        """
        try:
            self.start()
            self.wait_until_ready()
            yield self
        finally:
            if self.config.cleanup_on_exit:
                self.stop()

    def __enter__(self) -> "ExternalServiceBackend[ConfigT, ClientT]":
        """Context manager entry."""
        self.start()
        self.wait_until_ready()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        if self.config.cleanup_on_exit or exc_type is None:
            self.stop()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"service={self.service_name!r}, "
            f"status={self._status.name}, "
            f"host={self.host}:{self.port})"
        )


# =============================================================================
# Service Registry
# =============================================================================


class ServiceRegistry:
    """Central registry for external services.

    Provides singleton access to service backends and handles
    service discovery, caching, and dependency resolution.

    Example:
        >>> registry = ServiceRegistry.instance()
        >>> redis = registry.get_backend("redis")
        >>> if redis.is_available():
        ...     redis.start()
    """

    _instance: ClassVar["ServiceRegistry | None"] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self) -> None:
        """Initialize the registry."""
        self._backends: dict[str, ExternalServiceBackend[Any, Any]] = {}
        self._providers: dict[ProviderType, ServiceProvider[Any]] = {}
        self._started_services: list[str] = []

        # Register cleanup on exit
        atexit.register(self._cleanup)

    @classmethod
    def instance(cls) -> "ServiceRegistry":
        """Get the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the registry (for testing)."""
        if cls._instance is not None:
            cls._instance._cleanup()
            cls._instance = None

    def register_backend(
        self,
        name: str,
        backend: ExternalServiceBackend[Any, Any],
    ) -> None:
        """Register a service backend."""
        self._backends[name] = backend
        logger.debug(f"Registered backend: {name}")

    def get_backend(self, name: str) -> ExternalServiceBackend[Any, Any] | None:
        """Get a registered backend."""
        return self._backends.get(name)

    def get_or_create_backend(
        self,
        name: str,
        factory: type[ExternalServiceBackend[Any, Any]],
        config: ServiceConfig,
    ) -> ExternalServiceBackend[Any, Any]:
        """Get or create a backend."""
        if name not in self._backends:
            backend = factory(config)
            self.register_backend(name, backend)
        return self._backends[name]

    def list_backends(self) -> list[str]:
        """List all registered backend names."""
        return list(self._backends.keys())

    def start_service(self, name: str) -> bool:
        """Start a service by name."""
        backend = self.get_backend(name)
        if backend is None:
            logger.error(f"Unknown service: {name}")
            return False

        if backend.start():
            self._started_services.append(name)
            return True
        return False

    def stop_service(self, name: str) -> None:
        """Stop a service by name."""
        backend = self.get_backend(name)
        if backend is not None:
            backend.stop()
            if name in self._started_services:
                self._started_services.remove(name)

    def stop_all(self) -> None:
        """Stop all started services."""
        for name in reversed(self._started_services[:]):
            self.stop_service(name)

    def _cleanup(self) -> None:
        """Cleanup on exit."""
        logger.info("Cleaning up external services...")
        self.stop_all()
