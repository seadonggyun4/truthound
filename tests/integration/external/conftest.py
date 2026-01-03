"""Pytest configuration for external service integration tests.

This module provides pytest fixtures, markers, and hooks for
running integration tests against external services.

Usage:
    # Run all external integration tests
    pytest tests/integration/external/ -v

    # Run with specific provider
    pytest tests/integration/external/ -v --provider=docker
    pytest tests/integration/external/ -v --provider=mock

    # Run specific service tests
    pytest tests/integration/external/ -v -m redis
    pytest tests/integration/external/ -v -m elasticsearch
    pytest tests/integration/external/ -v -m vault
    pytest tests/integration/external/ -v -m kms
    pytest tests/integration/external/ -v -m tms

Configuration:
    Tests are configured via environment variables:
    - TRUTHOUND_TEST_REDIS_HOST: Redis host
    - TRUTHOUND_TEST_REDIS_PORT: Redis port
    - TRUTHOUND_TEST_ELASTICSEARCH_HOST: Elasticsearch host
    - etc.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Generator

import pytest

from tests.integration.external.base import (
    ProviderType,
    ServiceCategory,
    ServiceConfig,
    ServiceRegistry,
)
from tests.integration.external.providers import (
    get_provider,
    get_best_provider,
    get_available_providers,
)

if TYPE_CHECKING:
    from tests.integration.external.backends.redis_backend import RedisBackend
    from tests.integration.external.backends.elasticsearch_backend import ElasticsearchBackend
    from tests.integration.external.backends.vault_backend import VaultBackend
    from tests.integration.external.backends.kms_backend import KMSBackend
    from tests.integration.external.backends.tms_backend import TMSBackend

logger = logging.getLogger(__name__)


# =============================================================================
# Pytest Markers
# =============================================================================


def pytest_configure(config: pytest.Config) -> None:
    """Configure custom pytest markers."""
    # Service markers
    config.addinivalue_line(
        "markers",
        "redis: marks test to run only with Redis",
    )
    config.addinivalue_line(
        "markers",
        "elasticsearch: marks test to run only with Elasticsearch",
    )
    config.addinivalue_line(
        "markers",
        "vault: marks test to run only with Vault",
    )
    config.addinivalue_line(
        "markers",
        "kms: marks test to run only with Cloud KMS",
    )
    config.addinivalue_line(
        "markers",
        "tms: marks test to run only with TMS APIs",
    )
    config.addinivalue_line(
        "markers",
        "loki: marks test to run only with Loki",
    )
    config.addinivalue_line(
        "markers",
        "fluentd: marks test to run only with Fluentd",
    )

    # Provider markers
    config.addinivalue_line(
        "markers",
        "docker: marks test to require Docker provider",
    )
    config.addinivalue_line(
        "markers",
        "mock: marks test to use Mock provider only",
    )
    config.addinivalue_line(
        "markers",
        "cloud: marks test to require Cloud provider",
    )

    # Feature markers
    config.addinivalue_line(
        "markers",
        "slow: marks test as slow (> 30 seconds)",
    )
    config.addinivalue_line(
        "markers",
        "expensive: marks test as potentially expensive (cloud costs)",
    )
    config.addinivalue_line(
        "markers",
        "integration: marks test as requiring external services",
    )


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add custom command line options.

    Note: --skip-slow and --skip-expensive are defined in the parent
    conftest.py (tests/integration/conftest.py) to avoid duplication.
    """
    parser.addoption(
        "--provider",
        action="store",
        default=None,
        choices=["docker", "mock", "local", "cloud"],
        help="Service provider to use (docker, mock, local, cloud)",
    )
    parser.addoption(
        "--service",
        action="append",
        default=None,
        help="Specific service(s) to test (can be specified multiple times)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Modify collected tests based on options and markers."""
    # Skip slow tests if requested
    if config.getoption("--skip-slow"):
        skip_slow = pytest.mark.skip(reason="--skip-slow option set")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    # Skip expensive tests if requested
    if config.getoption("--skip-expensive"):
        skip_expensive = pytest.mark.skip(reason="--skip-expensive option set")
        for item in items:
            if "expensive" in item.keywords:
                item.add_marker(skip_expensive)

    # Filter by provider if specified
    provider = config.getoption("--provider")
    if provider:
        provider_type = ProviderType(provider)
        from tests.integration.external.providers import get_provider as _get_provider

        try:
            p = _get_provider(provider_type)
            if not p.is_available():
                skip_provider = pytest.mark.skip(
                    reason=f"Provider {provider} not available"
                )
                for item in items:
                    item.add_marker(skip_provider)
        except Exception:
            pass

    # Filter by service if specified
    services = config.getoption("--service")
    if services:
        skip_other = pytest.mark.skip(reason=f"Only running services: {services}")
        for item in items:
            item_services = []
            for marker in ["redis", "elasticsearch", "vault", "kms", "tms", "loki", "fluentd"]:
                if marker in item.keywords:
                    item_services.append(marker)

            if item_services and not any(s in services for s in item_services):
                item.add_marker(skip_other)


# =============================================================================
# Provider Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def provider_type(request: pytest.FixtureRequest) -> ProviderType:
    """Get the provider type to use for tests."""
    provider_str = request.config.getoption("--provider")
    if provider_str:
        return ProviderType(provider_str)

    # Auto-detect best provider
    best_type, _ = get_best_provider()
    return best_type


@pytest.fixture(scope="session")
def available_providers() -> list[ProviderType]:
    """Get list of available provider types."""
    return get_available_providers()


@pytest.fixture(scope="session")
def docker_available() -> bool:
    """Check if Docker is available."""
    try:
        provider = get_provider(ProviderType.DOCKER)
        return provider.is_available()
    except Exception:
        return False


@pytest.fixture(scope="session")
def service_registry() -> Generator[ServiceRegistry, None, None]:
    """Get service registry with automatic cleanup."""
    registry = ServiceRegistry.instance()
    yield registry
    registry.stop_all()


# =============================================================================
# Redis Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def redis_config() -> "RedisConfig":
    """Get Redis configuration."""
    from tests.integration.external.backends.redis_backend import RedisConfig
    return RedisConfig.from_env()


@pytest.fixture(scope="session")
def redis_backend(
    provider_type: ProviderType,
    docker_available: bool,
) -> Generator["RedisBackend | None", None, None]:
    """Create Redis backend for tests."""
    from tests.integration.external.backends.redis_backend import (
        RedisBackend,
        RedisConfig,
    )

    # Skip if Docker not available and not using mock
    if provider_type == ProviderType.DOCKER and not docker_available:
        pytest.skip("Docker not available for Redis")

    config = RedisConfig.from_env()
    config.provider = provider_type

    backend = RedisBackend(config)

    try:
        if backend.start() and backend.wait_until_ready():
            yield backend
        else:
            pytest.skip("Redis backend failed to start")
            yield None
    finally:
        backend.stop()


@pytest.fixture
def redis_client(redis_backend: "RedisBackend") -> Generator[Any, None, None]:
    """Get Redis client for tests."""
    if redis_backend is None:
        pytest.skip("Redis backend not available")

    # Flush database before test
    redis_backend.flush_all()
    yield redis_backend.client


# =============================================================================
# Elasticsearch Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def elasticsearch_config() -> "ElasticsearchConfig":
    """Get Elasticsearch configuration."""
    from tests.integration.external.backends.elasticsearch_backend import ElasticsearchConfig
    return ElasticsearchConfig.from_env()


@pytest.fixture(scope="session")
def elasticsearch_backend(
    provider_type: ProviderType,
    docker_available: bool,
) -> Generator["ElasticsearchBackend | None", None, None]:
    """Create Elasticsearch backend for tests."""
    from tests.integration.external.backends.elasticsearch_backend import (
        ElasticsearchBackend,
        ElasticsearchConfig,
    )

    if provider_type == ProviderType.DOCKER and not docker_available:
        pytest.skip("Docker not available for Elasticsearch")

    config = ElasticsearchConfig.from_env()
    config.provider = provider_type
    # Elasticsearch takes longer to start
    config.timeout_seconds = 120

    backend = ElasticsearchBackend(config)

    try:
        if backend.start() and backend.wait_until_ready():
            yield backend
        else:
            pytest.skip("Elasticsearch backend failed to start")
            yield None
    finally:
        # Cleanup test indices
        if backend.is_running:
            backend.cleanup_test_indices()
        backend.stop()


@pytest.fixture
def elasticsearch_client(
    elasticsearch_backend: "ElasticsearchBackend",
) -> Generator[Any, None, None]:
    """Get Elasticsearch client for tests."""
    if elasticsearch_backend is None:
        pytest.skip("Elasticsearch backend not available")

    yield elasticsearch_backend.client


# =============================================================================
# Vault Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def vault_config() -> "VaultConfig":
    """Get Vault configuration."""
    from tests.integration.external.backends.vault_backend import VaultConfig
    return VaultConfig.from_env()


@pytest.fixture(scope="session")
def vault_backend(
    provider_type: ProviderType,
    docker_available: bool,
) -> Generator["VaultBackend | None", None, None]:
    """Create Vault backend for tests."""
    from tests.integration.external.backends.vault_backend import (
        VaultBackend,
        VaultConfig,
    )

    if provider_type == ProviderType.DOCKER and not docker_available:
        pytest.skip("Docker not available for Vault")

    config = VaultConfig.from_env()
    config.provider = provider_type

    backend = VaultBackend(config)

    try:
        if backend.start() and backend.wait_until_ready():
            yield backend
        else:
            pytest.skip("Vault backend failed to start")
            yield None
    finally:
        backend.stop()


@pytest.fixture
def vault_client(vault_backend: "VaultBackend") -> Generator[Any, None, None]:
    """Get Vault client for tests."""
    if vault_backend is None:
        pytest.skip("Vault backend not available")

    yield vault_backend.client


# =============================================================================
# KMS Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def kms_config() -> "KMSConfig":
    """Get KMS configuration."""
    from tests.integration.external.backends.kms_backend import KMSConfig
    return KMSConfig.from_env()


@pytest.fixture(scope="session")
def kms_backend(
    provider_type: ProviderType,
    docker_available: bool,
) -> Generator["KMSBackend | None", None, None]:
    """Create KMS backend for tests."""
    from tests.integration.external.backends.kms_backend import (
        KMSBackend,
        KMSConfig,
        KMSProvider,
    )

    # Use mock if Docker not available
    if provider_type == ProviderType.DOCKER and not docker_available:
        config = KMSConfig.from_env()
        config.provider = ProviderType.MOCK
        config.kms_provider = KMSProvider.MOCK
    else:
        config = KMSConfig.from_env()
        config.provider = provider_type

    backend = KMSBackend(config)

    try:
        if backend.start() and backend.wait_until_ready():
            yield backend
        else:
            pytest.skip("KMS backend failed to start")
            yield None
    finally:
        backend.stop()


# =============================================================================
# TMS Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def tms_config() -> "TMSConfig":
    """Get TMS configuration."""
    from tests.integration.external.backends.tms_backend import TMSConfig
    return TMSConfig.from_env()


@pytest.fixture(scope="session")
def tms_backend(
    provider_type: ProviderType,
) -> Generator["TMSBackend | None", None, None]:
    """Create TMS backend for tests."""
    from tests.integration.external.backends.tms_backend import (
        TMSBackend,
        TMSConfig,
        TMSProvider,
    )

    # TMS uses mock by default (real APIs require credentials)
    config = TMSConfig.from_env()
    if config.tms_provider == TMSProvider.MOCK or not config.api_key:
        config.tms_provider = TMSProvider.MOCK

    backend = TMSBackend(config)

    try:
        if backend.start() and backend.wait_until_ready():
            yield backend
        else:
            pytest.skip("TMS backend failed to start")
            yield None
    finally:
        backend.stop()


# =============================================================================
# Mock Service Fixtures
# =============================================================================


@pytest.fixture
def mock_redis() -> Generator["MockRedisService", None, None]:
    """Get mock Redis service for fast testing."""
    from tests.integration.external.providers.mock_provider import MockRedisService

    service = MockRedisService()
    service.start()
    yield service
    service.stop()


@pytest.fixture
def mock_elasticsearch() -> Generator["MockElasticsearchService", None, None]:
    """Get mock Elasticsearch service for fast testing."""
    from tests.integration.external.providers.mock_provider import MockElasticsearchService

    service = MockElasticsearchService()
    service.start()
    yield service
    service.stop()


@pytest.fixture
def mock_vault() -> Generator["MockVaultService", None, None]:
    """Get mock Vault service for fast testing."""
    from tests.integration.external.providers.mock_provider import MockVaultService

    service = MockVaultService()
    service.start()
    yield service
    service.stop()


@pytest.fixture
def mock_tms() -> Generator["MockTMSService", None, None]:
    """Get mock TMS service for fast testing."""
    from tests.integration.external.providers.mock_provider import MockTMSService

    service = MockTMSService()
    service.start()
    yield service
    service.stop()


# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture
def test_project_id() -> str:
    """Generate a unique test project ID."""
    import uuid
    return f"test-project-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def test_key_id() -> str:
    """Generate a unique test key ID."""
    import uuid
    return f"test-key-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def sample_secret_data() -> dict[str, str]:
    """Sample secret data for testing."""
    return {
        "username": "test_user",
        "password": "test_password_123",
        "api_key": "sk-test-1234567890",
    }


@pytest.fixture
def sample_translation_data() -> list[dict[str, str]]:
    """Sample translation data for testing."""
    return [
        {"key": "hello", "text": "Hello", "context": "Greeting"},
        {"key": "goodbye", "text": "Goodbye", "context": "Farewell"},
        {"key": "welcome", "text": "Welcome to our app", "context": "Landing page"},
    ]


# =============================================================================
# Hooks
# =============================================================================


def pytest_terminal_summary(
    terminalreporter: Any,
    exitstatus: int,
    config: pytest.Config,
) -> None:
    """Add integration test summary to terminal output."""
    stats = terminalreporter.stats

    # Check if we ran external integration tests
    if not any(
        "external" in str(getattr(item, "fspath", ""))
        for items in stats.values()
        for item in items
        if hasattr(item, "fspath")
    ):
        return

    terminalreporter.write_sep("=", "External Service Integration Summary")

    # Report available providers
    try:
        available = get_available_providers()
        terminalreporter.write_line("\nAvailable Providers:")
        for ptype in ProviderType:
            symbol = "✓" if ptype in available else "✗"
            terminalreporter.write_line(f"  {symbol} {ptype.value}")
    except Exception:
        pass

    # Report provider used
    provider = config.getoption("--provider", default=None)
    if provider:
        terminalreporter.write_line(f"\nProvider used: {provider}")
