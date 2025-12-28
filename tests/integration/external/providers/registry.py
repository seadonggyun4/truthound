"""Provider registry for external service integration tests.

This module manages provider discovery, selection, and caching for
external service integration testing.

Provider Selection Logic:
    1. Use provider specified in config
    2. Fall back to Docker if available
    3. Fall back to Mock as last resort

Usage:
    >>> from tests.integration.external.providers import get_provider
    >>> provider = get_provider(ProviderType.DOCKER, config)
    >>> if provider.is_available():
    ...     info = provider.start_service(config)
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from tests.integration.external.base import ProviderType, ServiceConfig

logger = logging.getLogger(__name__)


class ProviderRegistry:
    """Central registry for service providers.

    Manages provider instances and provides automatic fallback
    when preferred providers are unavailable.
    """

    _instance: ClassVar["ProviderRegistry | None"] = None
    _providers: ClassVar[dict[ProviderType, Any]] = {}

    def __init__(self) -> None:
        """Initialize the registry."""
        self._cached_providers: dict[ProviderType, Any] = {}

    @classmethod
    def instance(cls) -> "ProviderRegistry":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the registry (for testing)."""
        cls._instance = None
        cls._providers.clear()

    def get_provider(
        self,
        provider_type: ProviderType,
        config: ServiceConfig | None = None,
    ) -> Any:
        """Get a provider by type.

        Args:
            provider_type: Type of provider to get
            config: Optional config for provider-specific initialization

        Returns:
            Provider instance

        Raises:
            ValueError: If provider type is unknown
        """
        # Check cache
        if provider_type in self._cached_providers:
            return self._cached_providers[provider_type]

        # Create provider
        provider = self._create_provider(provider_type)
        self._cached_providers[provider_type] = provider
        return provider

    def _create_provider(self, provider_type: ProviderType) -> Any:
        """Create a provider instance."""
        if provider_type == ProviderType.DOCKER:
            from tests.integration.external.providers.docker_provider import (
                DockerServiceProvider,
            )
            return DockerServiceProvider()

        elif provider_type == ProviderType.MOCK:
            from tests.integration.external.providers.mock_provider import (
                MockServiceProvider,
            )
            return MockServiceProvider()

        elif provider_type == ProviderType.LOCAL:
            # Local provider for locally installed services
            return LocalServiceProvider()

        elif provider_type == ProviderType.CLOUD:
            # Cloud provider for managed services
            return CloudServiceProvider()

        else:
            raise ValueError(f"Unknown provider type: {provider_type}")

    def get_available_providers(self) -> list[ProviderType]:
        """Get list of available provider types."""
        available = []
        for provider_type in ProviderType:
            try:
                provider = self.get_provider(provider_type)
                if provider.is_available():
                    available.append(provider_type)
            except Exception as e:
                logger.debug(f"Provider {provider_type} not available: {e}")
        return available

    def get_best_provider(
        self,
        preferred: ProviderType | None = None,
    ) -> tuple[ProviderType, Any]:
        """Get the best available provider.

        Selection priority:
        1. Preferred provider if available
        2. Docker if available
        3. Mock as fallback

        Returns:
            Tuple of (provider_type, provider_instance)
        """
        # Try preferred provider
        if preferred is not None:
            try:
                provider = self.get_provider(preferred)
                if provider.is_available():
                    return preferred, provider
            except Exception:
                pass

        # Try Docker
        try:
            from tests.integration.external.providers.docker_provider import (
                DockerServiceProvider,
            )
            docker_provider = DockerServiceProvider()
            if docker_provider.is_available():
                self._cached_providers[ProviderType.DOCKER] = docker_provider
                return ProviderType.DOCKER, docker_provider
        except Exception:
            pass

        # Fall back to Mock
        from tests.integration.external.providers.mock_provider import (
            MockServiceProvider,
        )
        mock_provider = MockServiceProvider()
        self._cached_providers[ProviderType.MOCK] = mock_provider
        return ProviderType.MOCK, mock_provider


class LocalServiceProvider:
    """Provider for locally installed services.

    Connects to services running on the local machine,
    typically for development environments.
    """

    provider_type: ClassVar[ProviderType] = ProviderType.LOCAL

    def is_available(self) -> bool:
        """Check if local services are available."""
        # Local provider requires explicit configuration
        return False

    def start_service(self, config: ServiceConfig) -> dict[str, Any]:
        """Local services are expected to be already running."""
        return {
            "host": config.host or "localhost",
            "port": config.port or 0,
            "local": True,
        }

    def stop_service(self, config: ServiceConfig) -> None:
        """No-op for local services."""
        pass

    def get_connection_info(self, config: ServiceConfig) -> dict[str, Any]:
        """Get connection info for local service."""
        return {
            "host": config.host or "localhost",
            "port": config.port or 0,
            "local": True,
        }

    def health_check(self, config: ServiceConfig) -> Any:
        """Health check for local service."""
        from tests.integration.external.base import HealthCheckResult
        # Would need to implement service-specific health checks
        return HealthCheckResult.failure("Local health check not implemented")


class CloudServiceProvider:
    """Provider for cloud-managed services.

    Connects to cloud services (AWS, GCP, Azure) using
    appropriate credentials and configuration.
    """

    provider_type: ClassVar[ProviderType] = ProviderType.CLOUD

    def is_available(self) -> bool:
        """Check if cloud provider is configured."""
        # Cloud provider requires explicit credentials
        return False

    def start_service(self, config: ServiceConfig) -> dict[str, Any]:
        """Cloud services are externally managed."""
        return {
            "host": config.host,
            "port": config.port,
            "cloud": True,
        }

    def stop_service(self, config: ServiceConfig) -> None:
        """No-op for cloud services (externally managed)."""
        pass

    def get_connection_info(self, config: ServiceConfig) -> dict[str, Any]:
        """Get connection info for cloud service."""
        return {
            "host": config.host,
            "port": config.port,
            "cloud": True,
        }

    def health_check(self, config: ServiceConfig) -> Any:
        """Health check for cloud service."""
        from tests.integration.external.base import HealthCheckResult
        return HealthCheckResult.failure("Cloud health check not implemented")


# =============================================================================
# Module-level convenience functions
# =============================================================================


def get_provider(
    provider_type: ProviderType,
    config: ServiceConfig | None = None,
) -> Any:
    """Get a provider by type.

    Args:
        provider_type: Type of provider
        config: Optional configuration

    Returns:
        Provider instance
    """
    return ProviderRegistry.instance().get_provider(provider_type, config)


def get_available_providers() -> list[ProviderType]:
    """Get list of available provider types."""
    return ProviderRegistry.instance().get_available_providers()


def get_best_provider(
    preferred: ProviderType | None = None,
) -> tuple[ProviderType, Any]:
    """Get the best available provider.

    Args:
        preferred: Preferred provider type

    Returns:
        Tuple of (provider_type, provider_instance)
    """
    return ProviderRegistry.instance().get_best_provider(preferred)
