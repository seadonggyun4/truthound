"""Elasticsearch backend for integration tests.

This module provides Elasticsearch integration testing with support for:
- Docker containers
- Local Elasticsearch instances
- Elasticsearch Cloud

Features:
    - Index management
    - Document operations
    - Search testing
    - Logging sink validation

Usage:
    >>> config = ElasticsearchConfig.from_env()
    >>> with ElasticsearchBackend(config) as backend:
    ...     backend.client.index(index="test", body={"message": "hello"})
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
    from elasticsearch import Elasticsearch

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class ElasticsearchConfig(DockerContainerConfig):
    """Elasticsearch-specific configuration.

    Attributes:
        username: Elasticsearch username
        password: Elasticsearch password
        api_key: API key authentication
        cloud_id: Elastic Cloud ID
        ssl: Use SSL/TLS
        verify_certs: Verify SSL certificates
        ca_certs: Path to CA certificates
    """

    username: str | None = None
    password: str | None = None
    api_key: str | None = None
    cloud_id: str | None = None
    ssl: bool = False
    verify_certs: bool = True
    ca_certs: str | None = None

    def __post_init__(self) -> None:
        """Set Elasticsearch-specific defaults."""
        self.name = self.name or "elasticsearch"
        self.category = ServiceCategory.SEARCH
        self.image = self.image or "docker.elastic.co/elasticsearch/elasticsearch"
        self.tag = self.tag or "8.11.0"
        self.ports = self.ports or {"9200/tcp": None}
        self.health_cmd = self.health_cmd or "curl -s http://localhost:9200/_cluster/health"

        # Single-node setup for testing
        if "discovery.type" not in self.environment:
            self.environment["discovery.type"] = "single-node"
        if "xpack.security.enabled" not in self.environment:
            self.environment["xpack.security.enabled"] = "false"
        if "ES_JAVA_OPTS" not in self.environment:
            self.environment["ES_JAVA_OPTS"] = "-Xms512m -Xmx512m"

    @classmethod
    def from_env(cls, name: str = "elasticsearch") -> "ElasticsearchConfig":
        """Create configuration from environment variables."""
        prefix = "TRUTHOUND_TEST_ELASTICSEARCH"

        return cls(
            name=name,
            category=ServiceCategory.SEARCH,
            host=os.getenv(f"{prefix}_HOST"),
            port=int(os.getenv(f"{prefix}_PORT", "0")) or None,
            username=os.getenv(f"{prefix}_USERNAME"),
            password=os.getenv(f"{prefix}_PASSWORD"),
            api_key=os.getenv(f"{prefix}_API_KEY"),
            cloud_id=os.getenv(f"{prefix}_CLOUD_ID"),
            ssl=os.getenv(f"{prefix}_SSL", "false").lower() == "true",
            verify_certs=os.getenv(f"{prefix}_VERIFY_CERTS", "true").lower() == "true",
            timeout_seconds=int(os.getenv(f"{prefix}_TIMEOUT", "60")),
        )


# =============================================================================
# Elasticsearch Backend
# =============================================================================


class ElasticsearchBackend(ExternalServiceBackend[ElasticsearchConfig, "Elasticsearch"]):
    """Elasticsearch test backend.

    Provides Elasticsearch connection and operations for integration testing.

    Features:
        - Automatic Docker container management
        - Index lifecycle management
        - Document CRUD operations
        - Search operations
    """

    service_name: ClassVar[str] = "elasticsearch"
    service_category: ClassVar[ServiceCategory] = ServiceCategory.SEARCH
    default_port: ClassVar[int] = 9200
    default_image: ClassVar[str] = "docker.elastic.co/elasticsearch/elasticsearch:8.11.0"

    def __init__(
        self,
        config: ElasticsearchConfig | None = None,
        provider: Any = None,
    ) -> None:
        """Initialize Elasticsearch backend."""
        if config is None:
            config = ElasticsearchConfig.from_env()
        super().__init__(config, provider)

    def _create_client(self) -> "Elasticsearch":
        """Create Elasticsearch client."""
        try:
            from elasticsearch import Elasticsearch
        except ImportError:
            raise ImportError(
                "elasticsearch package not installed. Run: pip install elasticsearch"
            )

        config = self.config

        # Build client kwargs
        kwargs: dict[str, Any] = {}

        # Host configuration
        if config.cloud_id:
            kwargs["cloud_id"] = config.cloud_id
        else:
            scheme = "https" if config.ssl else "http"
            kwargs["hosts"] = [f"{scheme}://{self.host}:{self.port}"]

        # Authentication
        if config.api_key:
            kwargs["api_key"] = config.api_key
        elif config.username and config.password:
            kwargs["basic_auth"] = (config.username, config.password)

        # SSL configuration
        if config.ssl:
            kwargs["verify_certs"] = config.verify_certs
            if config.ca_certs:
                kwargs["ca_certs"] = config.ca_certs

        # Timeouts
        kwargs["request_timeout"] = config.timeout_seconds

        client = Elasticsearch(**kwargs)

        # Test connection
        if not client.ping():
            raise ConnectionError("Failed to connect to Elasticsearch")

        return client

    def _close_client(self) -> None:
        """Close Elasticsearch client."""
        if self._client is not None:
            self._client.close()

    def _perform_health_check(self) -> HealthCheckResult:
        """Perform Elasticsearch health check."""
        if self._client is None:
            return HealthCheckResult.failure("Client not connected")

        try:
            health = self._client.cluster.health()
            status = health.get("status", "unknown")

            if status == "green":
                return HealthCheckResult.success(
                    "Cluster healthy",
                    cluster_name=health.get("cluster_name"),
                    number_of_nodes=health.get("number_of_nodes"),
                    active_shards=health.get("active_shards"),
                )
            elif status == "yellow":
                return HealthCheckResult.success(
                    "Cluster degraded (yellow)",
                    cluster_name=health.get("cluster_name"),
                    unassigned_shards=health.get("unassigned_shards"),
                )
            else:
                return HealthCheckResult.failure(
                    f"Cluster unhealthy: {status}",
                    cluster_name=health.get("cluster_name"),
                )

        except Exception as e:
            return HealthCheckResult.failure(str(e))

    # -------------------------------------------------------------------------
    # Elasticsearch-Specific Operations
    # -------------------------------------------------------------------------

    def create_index(
        self,
        index: str,
        mappings: dict[str, Any] | None = None,
        settings: dict[str, Any] | None = None,
    ) -> bool:
        """Create an index."""
        if self._client is None:
            return False

        body: dict[str, Any] = {}
        if mappings:
            body["mappings"] = mappings
        if settings:
            body["settings"] = settings

        self._client.indices.create(index=index, body=body if body else None)
        return True

    def delete_index(self, index: str) -> bool:
        """Delete an index."""
        if self._client is None:
            return False

        self._client.indices.delete(index=index, ignore=[404])
        return True

    def index_exists(self, index: str) -> bool:
        """Check if index exists."""
        if self._client is None:
            return False
        return self._client.indices.exists(index=index)

    def index_document(
        self,
        index: str,
        document: dict[str, Any],
        doc_id: str | None = None,
    ) -> str:
        """Index a document."""
        if self._client is None:
            raise RuntimeError("Client not connected")

        result = self._client.index(index=index, document=document, id=doc_id)
        return result["_id"]

    def get_document(self, index: str, doc_id: str) -> dict[str, Any]:
        """Get a document by ID."""
        if self._client is None:
            raise RuntimeError("Client not connected")

        result = self._client.get(index=index, id=doc_id)
        return result["_source"]

    def search(
        self,
        index: str | None = None,
        query: dict[str, Any] | None = None,
        size: int = 10,
    ) -> list[dict[str, Any]]:
        """Search documents."""
        if self._client is None:
            raise RuntimeError("Client not connected")

        body = {"query": query} if query else {"query": {"match_all": {}}}
        result = self._client.search(index=index, body=body, size=size)

        return [hit["_source"] for hit in result["hits"]["hits"]]

    def bulk_index(
        self,
        index: str,
        documents: list[dict[str, Any]],
    ) -> int:
        """Bulk index documents."""
        if self._client is None:
            raise RuntimeError("Client not connected")

        from elasticsearch.helpers import bulk

        actions = [
            {"_index": index, "_source": doc}
            for doc in documents
        ]

        success, _ = bulk(self._client, actions)
        return success

    def cleanup_test_indices(self, prefix: str = "truthound_test_") -> int:
        """Delete all indices with the given prefix."""
        if self._client is None:
            return 0

        indices = self._client.indices.get(index=f"{prefix}*", ignore=[404])
        count = len(indices)

        if count > 0:
            self._client.indices.delete(index=f"{prefix}*", ignore=[404])

        return count


# =============================================================================
# Test Helpers
# =============================================================================


def create_elasticsearch_backend(
    provider_type: ProviderType = ProviderType.DOCKER,
) -> ElasticsearchBackend:
    """Create an Elasticsearch backend with specified provider."""
    config = ElasticsearchConfig.from_env()
    config.provider = provider_type

    from tests.integration.external.providers import get_provider
    provider = get_provider(provider_type, config)

    return ElasticsearchBackend(config, provider)
