"""Service backends for external integration tests.

This package contains backend implementations for external services:
- Redis: In-memory cache and distributed coordination
- Elasticsearch: Search and logging
- Vault: Secrets management
- Cloud KMS: Key management (AWS, GCP, Azure)
- TMS: Translation management systems

Each backend implements the ExternalServiceBackend interface and provides:
- Connection management
- Health checking
- Client creation
- Service-specific operations

Usage:
    >>> from tests.integration.external.backends import RedisBackend
    >>> config = RedisConfig.from_env()
    >>> with RedisBackend(config) as redis:
    ...     redis.client.set("key", "value")
"""

from tests.integration.external.backends.redis_backend import (
    RedisBackend,
    RedisConfig,
)
from tests.integration.external.backends.elasticsearch_backend import (
    ElasticsearchBackend,
    ElasticsearchConfig,
)
from tests.integration.external.backends.vault_backend import (
    VaultBackend,
    VaultConfig,
)
from tests.integration.external.backends.kms_backend import (
    KMSBackend,
    KMSConfig,
    KMSProvider,
)
from tests.integration.external.backends.tms_backend import (
    TMSBackend,
    TMSConfig,
    TMSProvider,
)

__all__ = [
    # Redis
    "RedisBackend",
    "RedisConfig",
    # Elasticsearch
    "ElasticsearchBackend",
    "ElasticsearchConfig",
    # Vault
    "VaultBackend",
    "VaultConfig",
    # KMS
    "KMSBackend",
    "KMSConfig",
    "KMSProvider",
    # TMS
    "TMSBackend",
    "TMSConfig",
    "TMSProvider",
]
