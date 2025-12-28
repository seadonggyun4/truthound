"""Docker-based service provider for integration tests.

This module provides Docker container management for external services,
enabling isolated and reproducible testing environments.

Features:
    - Container lifecycle management (create, start, stop, remove)
    - Health check with readiness probes
    - Port mapping and network configuration
    - Volume mounting for persistence
    - Resource limits (CPU, memory)
    - Container cleanup on exit

Architecture:
    DockerServiceProvider
        |
        +---> DockerClient (docker-py)
        |         |
        |         +---> Container management
        |         +---> Network management
        |         +---> Volume management
        |
        +---> ContainerRegistry (tracks started containers)

Usage:
    >>> provider = DockerServiceProvider()
    >>> if provider.is_available():
    ...     config = DockerContainerConfig(
    ...         name="redis",
    ...         image="redis:7-alpine",
    ...         ports={"6379/tcp": 6379},
    ...     )
    ...     info = provider.start_service(config)
    ...     # Use service at info["host"]:info["port"]
    ...     provider.stop_service(config)
"""

from __future__ import annotations

import atexit
import logging
import os
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Any, ClassVar

from tests.integration.external.base import (
    HealthCheckResult,
    ProviderType,
    ServiceConfig,
    ServiceStatus,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Docker Container Configuration
# =============================================================================


@dataclass
class DockerContainerConfig(ServiceConfig):
    """Extended configuration for Docker containers.

    Attributes:
        image: Docker image to use
        tag: Image tag (default: latest)
        ports: Port mappings {container_port: host_port}
        environment: Environment variables
        volumes: Volume mounts {host_path: container_path}
        command: Override container command
        entrypoint: Override container entrypoint
        network: Docker network name
        cpu_limit: CPU limit (e.g., "1.0" for 1 CPU)
        memory_limit: Memory limit (e.g., "512m")
        health_cmd: Health check command
        health_interval: Health check interval
        health_timeout: Health check timeout
        health_retries: Health check retry count
        remove_on_stop: Remove container when stopped
        auto_remove: Auto-remove container on exit
    """

    image: str = ""
    tag: str = "latest"
    ports: dict[str, int | None] = field(default_factory=dict)
    volumes: dict[str, str] = field(default_factory=dict)
    command: str | list[str] | None = None
    entrypoint: str | list[str] | None = None
    network: str | None = None
    cpu_limit: str | None = None
    memory_limit: str | None = None
    health_cmd: str | list[str] | None = None
    health_interval: int = 5  # seconds
    health_timeout: int = 3  # seconds
    health_retries: int = 3
    remove_on_stop: bool = True
    auto_remove: bool = False

    @property
    def full_image(self) -> str:
        """Get full image name with tag."""
        return f"{self.image}:{self.tag}"


# =============================================================================
# Container Registry
# =============================================================================


class ContainerRegistry:
    """Registry of started containers for cleanup.

    Thread-safe singleton that tracks all started containers
    and ensures cleanup on exit.
    """

    _instance: ClassVar["ContainerRegistry | None"] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self) -> None:
        self._containers: dict[str, str] = {}  # name -> container_id
        self._client: Any = None
        atexit.register(self._cleanup_all)

    @classmethod
    def instance(cls) -> "ContainerRegistry":
        """Get singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def register(self, name: str, container_id: str) -> None:
        """Register a container."""
        self._containers[name] = container_id
        logger.debug(f"Registered container: {name} ({container_id[:12]})")

    def unregister(self, name: str) -> str | None:
        """Unregister a container."""
        return self._containers.pop(name, None)

    def get(self, name: str) -> str | None:
        """Get container ID by name."""
        return self._containers.get(name)

    def list_all(self) -> list[str]:
        """List all registered container names."""
        return list(self._containers.keys())

    def _get_client(self) -> Any:
        """Get Docker client."""
        if self._client is None:
            try:
                import docker
                self._client = docker.from_env()
            except ImportError:
                logger.error("docker package not installed")
                return None
            except Exception as e:
                logger.error(f"Failed to create Docker client: {e}")
                return None
        return self._client

    def _cleanup_all(self) -> None:
        """Cleanup all registered containers."""
        if not self._containers:
            return

        client = self._get_client()
        if client is None:
            return

        logger.info(f"Cleaning up {len(self._containers)} containers...")
        for name, container_id in list(self._containers.items()):
            try:
                container = client.containers.get(container_id)
                container.stop(timeout=5)
                container.remove(force=True)
                logger.info(f"Removed container: {name}")
            except Exception as e:
                logger.warning(f"Failed to cleanup {name}: {e}")

        self._containers.clear()


# =============================================================================
# Docker Service Provider
# =============================================================================


class DockerServiceProvider:
    """Service provider using Docker containers.

    Manages container lifecycle for integration testing with:
    - Automatic port allocation
    - Health check waiting
    - Resource management
    - Cleanup on exit
    """

    provider_type: ClassVar[ProviderType] = ProviderType.DOCKER

    def __init__(self) -> None:
        """Initialize the Docker provider."""
        self._client: Any = None
        self._registry = ContainerRegistry.instance()
        self._started: dict[str, Any] = {}  # config.name -> container

    def is_available(self) -> bool:
        """Check if Docker is available."""
        try:
            client = self._get_client()
            if client is None:
                return False
            client.ping()
            return True
        except Exception as e:
            logger.debug(f"Docker not available: {e}")
            return False

    def _get_client(self) -> Any:
        """Get Docker client."""
        if self._client is None:
            try:
                import docker
                self._client = docker.from_env()
            except ImportError:
                logger.error("docker package not installed. Run: pip install docker")
                return None
            except Exception as e:
                logger.error(f"Failed to connect to Docker: {e}")
                return None
        return self._client

    def start_service(self, config: DockerContainerConfig) -> dict[str, Any]:
        """Start a Docker container.

        Args:
            config: Container configuration

        Returns:
            Connection info with host, port, container_id
        """
        client = self._get_client()
        if client is None:
            raise RuntimeError("Docker client not available")

        container_name = self._get_container_name(config)

        # Check if container already exists
        try:
            existing = client.containers.get(container_name)
            if existing.status == "running":
                logger.info(f"Container {container_name} already running")
                return self._get_connection_info(config, existing)
            else:
                logger.info(f"Removing stopped container: {container_name}")
                existing.remove(force=True)
        except Exception:
            pass  # Container doesn't exist

        # Pull image if needed
        self._pull_image(client, config)

        # Prepare container options
        run_kwargs = self._prepare_run_kwargs(config, container_name)

        # Start container
        logger.info(f"Starting container: {container_name} ({config.full_image})")
        container = client.containers.run(**run_kwargs)

        # Register for cleanup
        self._registry.register(container_name, container.id)
        self._started[config.name] = container

        # Wait for container to be ready
        if not self._wait_for_ready(container, config):
            logger.warning(f"Container {container_name} not ready, continuing anyway")

        return self._get_connection_info(config, container)

    def stop_service(self, config: DockerContainerConfig) -> None:
        """Stop and optionally remove a container."""
        container_name = self._get_container_name(config)
        container = self._started.pop(config.name, None)

        if container is None:
            client = self._get_client()
            if client is None:
                return
            try:
                container = client.containers.get(container_name)
            except Exception:
                logger.debug(f"Container {container_name} not found")
                return

        try:
            logger.info(f"Stopping container: {container_name}")
            container.stop(timeout=10)

            if config.remove_on_stop:
                container.remove(force=True)
                logger.info(f"Removed container: {container_name}")

            self._registry.unregister(container_name)

        except Exception as e:
            logger.warning(f"Error stopping container {container_name}: {e}")

    def get_connection_info(self, config: DockerContainerConfig) -> dict[str, Any]:
        """Get connection info for a running container."""
        container = self._started.get(config.name)
        if container is None:
            client = self._get_client()
            if client is None:
                return {}
            try:
                container_name = self._get_container_name(config)
                container = client.containers.get(container_name)
            except Exception:
                return {}

        return self._get_connection_info(config, container)

    def health_check(self, config: DockerContainerConfig) -> HealthCheckResult:
        """Check container health."""
        container = self._started.get(config.name)
        if container is None:
            return HealthCheckResult.failure("Container not started")

        try:
            container.reload()
            status = container.status
            health = container.attrs.get("State", {}).get("Health", {})
            health_status = health.get("Status", "unknown")

            if status != "running":
                return HealthCheckResult.failure(f"Container status: {status}")

            if health_status == "healthy":
                return HealthCheckResult.success("Container healthy")
            elif health_status == "starting":
                return HealthCheckResult.failure("Container starting")
            elif health_status == "unhealthy":
                return HealthCheckResult.failure("Container unhealthy")
            else:
                # No health check defined, check if running
                return HealthCheckResult.success("Container running (no health check)")

        except Exception as e:
            return HealthCheckResult.failure(str(e))

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _get_container_name(self, config: DockerContainerConfig) -> str:
        """Generate container name."""
        return f"truthound_test_{config.name}"

    def _pull_image(self, client: Any, config: DockerContainerConfig) -> None:
        """Pull Docker image if not present."""
        try:
            client.images.get(config.full_image)
            logger.debug(f"Image already exists: {config.full_image}")
        except Exception:
            logger.info(f"Pulling image: {config.full_image}")
            client.images.pull(config.image, tag=config.tag)

    def _prepare_run_kwargs(
        self,
        config: DockerContainerConfig,
        container_name: str,
    ) -> dict[str, Any]:
        """Prepare kwargs for container.run()."""
        kwargs: dict[str, Any] = {
            "image": config.full_image,
            "name": container_name,
            "detach": True,
            "auto_remove": config.auto_remove,
            "environment": config.environment,
            "labels": {
                "truthound.test": "true",
                "truthound.service": config.name,
                **config.labels,
            },
        }

        # Port mapping
        if config.ports:
            kwargs["ports"] = {}
            for container_port, host_port in config.ports.items():
                if host_port is None:
                    # Dynamic port allocation
                    host_port = self._find_free_port()
                kwargs["ports"][container_port] = host_port

        # Volume mounts
        if config.volumes:
            kwargs["volumes"] = {
                host: {"bind": container, "mode": "rw"}
                for host, container in config.volumes.items()
            }

        # Command and entrypoint
        if config.command:
            kwargs["command"] = config.command
        if config.entrypoint:
            kwargs["entrypoint"] = config.entrypoint

        # Network
        if config.network:
            kwargs["network"] = config.network

        # Resource limits
        if config.cpu_limit or config.memory_limit:
            kwargs["cpu_period"] = 100000
            if config.cpu_limit:
                kwargs["cpu_quota"] = int(float(config.cpu_limit) * 100000)
            if config.memory_limit:
                kwargs["mem_limit"] = config.memory_limit

        # Health check
        if config.health_cmd:
            kwargs["healthcheck"] = {
                "test": config.health_cmd if isinstance(config.health_cmd, list)
                        else ["CMD-SHELL", config.health_cmd],
                "interval": config.health_interval * 1000000000,  # nanoseconds
                "timeout": config.health_timeout * 1000000000,
                "retries": config.health_retries,
                "start_period": 5 * 1000000000,
            }

        return kwargs

    def _get_connection_info(
        self,
        config: DockerContainerConfig,
        container: Any,
    ) -> dict[str, Any]:
        """Extract connection info from container."""
        container.reload()

        # Get host
        host = os.getenv("DOCKER_HOST_IP", "localhost")

        # Get ports
        ports = {}
        network_settings = container.attrs.get("NetworkSettings", {})
        port_bindings = network_settings.get("Ports", {})

        for container_port, bindings in port_bindings.items():
            if bindings:
                # Use first binding
                host_port = int(bindings[0]["HostPort"])
                port_key = container_port.split("/")[0]  # Remove /tcp or /udp
                ports[port_key] = host_port

        # Determine primary port
        primary_port = None
        if ports:
            # Use first configured port or first available
            if config.ports:
                first_key = list(config.ports.keys())[0].split("/")[0]
                primary_port = ports.get(first_key)
            if primary_port is None:
                primary_port = list(ports.values())[0]

        return {
            "host": host,
            "port": primary_port,
            "ports": ports,
            "container_id": container.id,
            "container_name": container.name,
            "status": container.status,
        }

    def _wait_for_ready(
        self,
        container: Any,
        config: DockerContainerConfig,
    ) -> bool:
        """Wait for container to be ready."""
        timeout = config.timeout_seconds
        interval = config.health_check_interval

        start_time = time.time()
        while time.time() - start_time < timeout:
            container.reload()

            # Check container is running
            if container.status != "running":
                logger.debug(f"Container status: {container.status}")
                time.sleep(interval)
                continue

            # Check health status if defined
            health = container.attrs.get("State", {}).get("Health", {})
            health_status = health.get("Status")

            if health_status is None:
                # No health check, just ensure running
                return True
            elif health_status == "healthy":
                return True
            elif health_status == "unhealthy":
                logger.warning("Container health check failed")
                return False

            time.sleep(interval)

        return False

    def _find_free_port(self) -> int:
        """Find a free port on the host."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port
