"""Testcontainers integration for streaming tests.

Provides managed containers for integration testing:
- KafkaTestContainer: Apache Kafka with Zookeeper
- RedisTestContainer: Redis for state management
- LocalStackTestContainer: AWS Kinesis emulation
- StreamTestEnvironment: Multi-container orchestration
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator
import asyncio
import logging
import os


logger = logging.getLogger(__name__)


# =============================================================================
# Container Protocols
# =============================================================================


class ITestContainer(ABC):
    """Protocol for test containers."""

    @abstractmethod
    async def start(self) -> None:
        """Start the container."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Stop the container."""
        ...

    @abstractmethod
    def get_connection_url(self) -> str:
        """Get connection URL for the service."""
        ...

    @property
    @abstractmethod
    def is_running(self) -> bool:
        """Check if container is running."""
        ...


# =============================================================================
# Kafka Test Container
# =============================================================================


@dataclass
class KafkaTestContainerConfig:
    """Configuration for Kafka test container.

    Attributes:
        image: Docker image to use
        kafka_port: Kafka broker port
        zookeeper_port: Zookeeper port
        startup_timeout: Container startup timeout in seconds
        auto_create_topics: Whether to auto-create topics
        num_partitions: Default number of partitions for topics
        replication_factor: Default replication factor
    """

    image: str = "confluentinc/cp-kafka:7.5.0"
    kafka_port: int = 9092
    zookeeper_port: int = 2181
    startup_timeout: int = 60
    auto_create_topics: bool = True
    num_partitions: int = 3
    replication_factor: int = 1


class KafkaTestContainer(ITestContainer):
    """Kafka test container using testcontainers-python.

    Provides a real Kafka broker for integration testing.

    Example:
        >>> async with KafkaTestContainer() as kafka:
        ...     url = kafka.get_connection_url()
        ...     # Use with KafkaAdapter
        ...     adapter = KafkaAdapter(KafkaAdapterConfig(
        ...         bootstrap_servers=url,
        ...         topic="test-topic",
        ...     ))

    Requires:
        pip install testcontainers[kafka]
    """

    def __init__(self, config: KafkaTestContainerConfig | None = None):
        self._config = config or KafkaTestContainerConfig()
        self._container: Any = None
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    async def start(self) -> None:
        """Start Kafka container."""
        if self._running:
            return

        try:
            from testcontainers.kafka import KafkaContainer
        except ImportError:
            raise ImportError(
                "testcontainers is required for integration testing. "
                "Install with: pip install testcontainers[kafka]"
            )

        def _start_sync():
            self._container = KafkaContainer(self._config.image)
            self._container.start()
            return self._container

        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _start_sync)
        self._running = True

        logger.info(f"Kafka container started: {self.get_connection_url()}")

    async def stop(self) -> None:
        """Stop Kafka container."""
        if not self._running or not self._container:
            return

        def _stop_sync():
            self._container.stop()

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _stop_sync)
        self._running = False

        logger.info("Kafka container stopped")

    def get_connection_url(self) -> str:
        """Get Kafka bootstrap servers URL."""
        if not self._container:
            raise RuntimeError("Container not started")
        return self._container.get_bootstrap_server()

    async def create_topic(
        self,
        topic: str,
        num_partitions: int | None = None,
        replication_factor: int | None = None,
    ) -> None:
        """Create a Kafka topic.

        Args:
            topic: Topic name
            num_partitions: Number of partitions
            replication_factor: Replication factor
        """
        if not self._running:
            raise RuntimeError("Container not running")

        try:
            from kafka.admin import KafkaAdminClient, NewTopic
        except ImportError:
            raise ImportError(
                "kafka-python is required for topic management. "
                "Install with: pip install kafka-python"
            )

        def _create_topic_sync():
            admin = KafkaAdminClient(
                bootstrap_servers=self.get_connection_url(),
            )
            topic_obj = NewTopic(
                name=topic,
                num_partitions=num_partitions or self._config.num_partitions,
                replication_factor=replication_factor or self._config.replication_factor,
            )
            admin.create_topics([topic_obj])
            admin.close()

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _create_topic_sync)

    async def __aenter__(self) -> "KafkaTestContainer":
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.stop()


# =============================================================================
# Redis Test Container
# =============================================================================


@dataclass
class RedisTestContainerConfig:
    """Configuration for Redis test container.

    Attributes:
        image: Docker image to use
        port: Redis port
        startup_timeout: Container startup timeout in seconds
    """

    image: str = "redis:7-alpine"
    port: int = 6379
    startup_timeout: int = 30


class RedisTestContainer(ITestContainer):
    """Redis test container for state management testing.

    Example:
        >>> async with RedisTestContainer() as redis:
        ...     url = redis.get_connection_url()
        ...     # Use with RedisStateBackend

    Requires:
        pip install testcontainers[redis]
    """

    def __init__(self, config: RedisTestContainerConfig | None = None):
        self._config = config or RedisTestContainerConfig()
        self._container: Any = None
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    async def start(self) -> None:
        """Start Redis container."""
        if self._running:
            return

        try:
            from testcontainers.redis import RedisContainer
        except ImportError:
            raise ImportError(
                "testcontainers is required for integration testing. "
                "Install with: pip install testcontainers[redis]"
            )

        def _start_sync():
            self._container = RedisContainer(self._config.image)
            self._container.start()
            return self._container

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _start_sync)
        self._running = True

        logger.info(f"Redis container started: {self.get_connection_url()}")

    async def stop(self) -> None:
        """Stop Redis container."""
        if not self._running or not self._container:
            return

        def _stop_sync():
            self._container.stop()

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _stop_sync)
        self._running = False

        logger.info("Redis container stopped")

    def get_connection_url(self) -> str:
        """Get Redis connection URL."""
        if not self._container:
            raise RuntimeError("Container not started")
        return self._container.get_connection_url()

    def get_host(self) -> str:
        """Get Redis host."""
        if not self._container:
            raise RuntimeError("Container not started")
        return self._container.get_container_host_ip()

    def get_port(self) -> int:
        """Get Redis port."""
        if not self._container:
            raise RuntimeError("Container not started")
        return int(self._container.get_exposed_port(self._config.port))

    async def __aenter__(self) -> "RedisTestContainer":
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.stop()


# =============================================================================
# LocalStack Test Container (for AWS Kinesis)
# =============================================================================


@dataclass
class LocalStackTestContainerConfig:
    """Configuration for LocalStack test container.

    Attributes:
        image: Docker image to use
        services: AWS services to enable
        port: LocalStack port
        startup_timeout: Container startup timeout in seconds
    """

    image: str = "localstack/localstack:3"
    services: list[str] = field(default_factory=lambda: ["kinesis"])
    port: int = 4566
    startup_timeout: int = 60


class LocalStackTestContainer(ITestContainer):
    """LocalStack test container for AWS service emulation.

    Provides local AWS Kinesis for integration testing.

    Example:
        >>> async with LocalStackTestContainer() as localstack:
        ...     endpoint = localstack.get_connection_url()
        ...     # Use with KinesisAdapter
        ...     adapter = KinesisAdapter(KinesisAdapterConfig(
        ...         stream_name="test-stream",
        ...         endpoint_url=endpoint,
        ...     ))

    Requires:
        pip install testcontainers[localstack]
    """

    def __init__(self, config: LocalStackTestContainerConfig | None = None):
        self._config = config or LocalStackTestContainerConfig()
        self._container: Any = None
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    async def start(self) -> None:
        """Start LocalStack container."""
        if self._running:
            return

        try:
            from testcontainers.localstack import LocalStackContainer
        except ImportError:
            raise ImportError(
                "testcontainers is required for integration testing. "
                "Install with: pip install testcontainers[localstack]"
            )

        def _start_sync():
            self._container = LocalStackContainer(self._config.image)
            for service in self._config.services:
                self._container.with_services(service)
            self._container.start()
            return self._container

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _start_sync)
        self._running = True

        logger.info(f"LocalStack container started: {self.get_connection_url()}")

    async def stop(self) -> None:
        """Stop LocalStack container."""
        if not self._running or not self._container:
            return

        def _stop_sync():
            self._container.stop()

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _stop_sync)
        self._running = False

        logger.info("LocalStack container stopped")

    def get_connection_url(self) -> str:
        """Get LocalStack endpoint URL."""
        if not self._container:
            raise RuntimeError("Container not started")
        host = self._container.get_container_host_ip()
        port = self._container.get_exposed_port(self._config.port)
        return f"http://{host}:{port}"

    async def create_kinesis_stream(
        self,
        stream_name: str,
        shard_count: int = 1,
    ) -> None:
        """Create a Kinesis stream.

        Args:
            stream_name: Stream name
            shard_count: Number of shards
        """
        if not self._running:
            raise RuntimeError("Container not running")

        try:
            import boto3
        except ImportError:
            raise ImportError(
                "boto3 is required for Kinesis operations. "
                "Install with: pip install boto3"
            )

        def _create_stream_sync():
            client = boto3.client(
                "kinesis",
                endpoint_url=self.get_connection_url(),
                region_name="us-east-1",
                aws_access_key_id="test",
                aws_secret_access_key="test",
            )
            client.create_stream(
                StreamName=stream_name,
                ShardCount=shard_count,
            )
            # Wait for stream to become active
            waiter = client.get_waiter("stream_exists")
            waiter.wait(StreamName=stream_name)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _create_stream_sync)

    async def __aenter__(self) -> "LocalStackTestContainer":
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.stop()


# =============================================================================
# Stream Test Environment
# =============================================================================


class StreamTestEnvironment:
    """Multi-container test environment for streaming tests.

    Orchestrates multiple test containers for comprehensive
    integration testing.

    Example:
        >>> async with StreamTestEnvironment(["kafka", "redis"]) as env:
        ...     kafka_url = env.get_service_url("kafka")
        ...     redis_url = env.get_service_url("redis")
        ...     # Run integration tests
    """

    _container_types: dict[str, type[ITestContainer]] = {
        "kafka": KafkaTestContainer,
        "redis": RedisTestContainer,
        "localstack": LocalStackTestContainer,
        "kinesis": LocalStackTestContainer,  # Alias for Kinesis via LocalStack
    }

    def __init__(
        self,
        services: list[str],
        configs: dict[str, Any] | None = None,
    ):
        """Initialize test environment.

        Args:
            services: List of services to start (kafka, redis, localstack, kinesis)
            configs: Optional configurations for each service
        """
        self._services = services
        self._configs = configs or {}
        self._containers: dict[str, ITestContainer] = {}

    async def start(self) -> None:
        """Start all containers."""
        tasks = []

        for service in self._services:
            if service not in self._container_types:
                raise ValueError(f"Unknown service: {service}")

            container_cls = self._container_types[service]
            config = self._configs.get(service)
            container = container_cls(config) if config else container_cls()
            self._containers[service] = container
            tasks.append(container.start())

        # Start all containers in parallel
        await asyncio.gather(*tasks)

        logger.info(f"Test environment started with services: {self._services}")

    async def stop(self) -> None:
        """Stop all containers."""
        tasks = [c.stop() for c in self._containers.values()]
        await asyncio.gather(*tasks, return_exceptions=True)
        self._containers.clear()

        logger.info("Test environment stopped")

    def get_container(self, service: str) -> ITestContainer:
        """Get container by service name.

        Args:
            service: Service name

        Returns:
            Container instance
        """
        if service not in self._containers:
            raise KeyError(f"Service not found: {service}")
        return self._containers[service]

    def get_service_url(self, service: str) -> str:
        """Get connection URL for a service.

        Args:
            service: Service name

        Returns:
            Connection URL
        """
        return self.get_container(service).get_connection_url()

    async def create_kafka_topic(
        self,
        topic: str,
        num_partitions: int = 3,
    ) -> None:
        """Create a Kafka topic.

        Args:
            topic: Topic name
            num_partitions: Number of partitions
        """
        kafka = self._containers.get("kafka")
        if not kafka or not isinstance(kafka, KafkaTestContainer):
            raise RuntimeError("Kafka not available in environment")
        await kafka.create_topic(topic, num_partitions)

    async def create_kinesis_stream(
        self,
        stream_name: str,
        shard_count: int = 1,
    ) -> None:
        """Create a Kinesis stream.

        Args:
            stream_name: Stream name
            shard_count: Number of shards
        """
        localstack = self._containers.get("localstack") or self._containers.get("kinesis")
        if not localstack or not isinstance(localstack, LocalStackTestContainer):
            raise RuntimeError("LocalStack/Kinesis not available in environment")
        await localstack.create_kinesis_stream(stream_name, shard_count)

    async def __aenter__(self) -> "StreamTestEnvironment":
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.stop()


# =============================================================================
# Utility Functions
# =============================================================================


def skip_if_no_docker() -> bool:
    """Check if Docker is available for testing.

    Returns:
        True if Docker is available
    """
    import shutil

    return shutil.which("docker") is not None


def require_testcontainers():
    """Decorator to skip tests if testcontainers is not available."""
    import functools

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                import testcontainers
            except ImportError:
                import pytest

                pytest.skip("testcontainers not installed")
            return await func(*args, **kwargs)

        return wrapper

    return decorator
