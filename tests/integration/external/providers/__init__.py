"""Service providers for external integration tests.

This package contains provider implementations for different environments:
- Docker: Containerized services for local testing
- Local: Locally installed services
- Cloud: Cloud-managed services (AWS, GCP, Azure)
- Mock: In-memory mocks for unit testing

Provider Selection:
    Providers are selected based on configuration and availability:
    1. Check if preferred provider is available
    2. Fall back to Docker if available
    3. Fall back to Mock for testing without dependencies

Usage:
    >>> from tests.integration.external.providers import get_provider
    >>> provider = get_provider(ProviderType.DOCKER, config)
    >>> if provider.is_available():
    ...     connection_info = provider.start_service(config)
"""

from tests.integration.external.providers.docker_provider import (
    DockerServiceProvider,
    DockerContainerConfig,
)
from tests.integration.external.providers.mock_provider import (
    MockServiceProvider,
    MockService,
)
from tests.integration.external.providers.registry import (
    get_provider,
    get_available_providers,
    get_best_provider,
    ProviderRegistry,
)

__all__ = [
    # Docker
    "DockerServiceProvider",
    "DockerContainerConfig",
    # Mock
    "MockServiceProvider",
    "MockService",
    # Registry
    "get_provider",
    "get_available_providers",
    "get_best_provider",
    "ProviderRegistry",
]
