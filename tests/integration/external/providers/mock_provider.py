"""Mock service provider for testing without external dependencies.

This module provides in-memory mock implementations of external services,
enabling tests to run without Docker or cloud services.

Features:
    - In-memory Redis mock with full API compatibility
    - Mock Elasticsearch for logging tests
    - Mock Vault for secrets tests
    - Mock TMS APIs for translation tests

Usage:
    >>> provider = MockServiceProvider()
    >>> config = ServiceConfig(name="redis", category=ServiceCategory.CACHE)
    >>> info = provider.start_service(config)
    >>> # info contains mock connection details
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar, Dict, List, Optional

from tests.integration.external.base import (
    HealthCheckResult,
    ProviderType,
    ServiceCategory,
    ServiceConfig,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Base Mock Service
# =============================================================================


@dataclass
class MockService:
    """Base class for mock services.

    Provides common functionality for all mock services including
    state management, health checking, and lifecycle hooks.
    """

    name: str
    category: ServiceCategory
    started: bool = False
    healthy: bool = True
    start_time: datetime | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def start(self) -> dict[str, Any]:
        """Start the mock service."""
        with self._lock:
            self.started = True
            self.healthy = True
            self.start_time = datetime.utcnow()
            logger.debug(f"Mock service started: {self.name}")
            return {"host": "localhost", "port": 0, "mock": True}

    def stop(self) -> None:
        """Stop the mock service."""
        with self._lock:
            self.started = False
            logger.debug(f"Mock service stopped: {self.name}")

    def health_check(self) -> HealthCheckResult:
        """Check service health."""
        if not self.started:
            return HealthCheckResult.failure("Service not started")
        if not self.healthy:
            return HealthCheckResult.failure("Service unhealthy")
        return HealthCheckResult.success("Mock service healthy")


# =============================================================================
# Redis Mock
# =============================================================================


class MockRedisService(MockService):
    """In-memory Redis mock with key-value operations.

    Supports:
    - String operations (get, set, delete, exists)
    - TTL (time-to-live) for keys
    - Hash operations (hget, hset, hgetall)
    - List operations (lpush, rpush, lrange)
    - Set operations (sadd, smembers, sismember)
    - Pub/Sub basics

    Example:
        >>> redis = MockRedisService("redis", ServiceCategory.CACHE)
        >>> redis.start()
        >>> redis.set("key", "value")
        >>> redis.get("key")
        'value'
    """

    def __init__(self, name: str = "redis", category: ServiceCategory = ServiceCategory.CACHE):
        super().__init__(name=name, category=category)
        self._data: dict[str, Any] = {}
        self._expiry: dict[str, float] = {}
        self._hashes: dict[str, dict[str, str]] = defaultdict(dict)
        self._lists: dict[str, list[str]] = defaultdict(list)
        self._sets: dict[str, set[str]] = defaultdict(set)
        self._pubsub_channels: dict[str, list[Any]] = defaultdict(list)

    def start(self) -> dict[str, Any]:
        """Start mock Redis."""
        result = super().start()
        self._data.clear()
        self._expiry.clear()
        self._hashes.clear()
        self._lists.clear()
        self._sets.clear()
        return {**result, "port": 6379}

    def stop(self) -> None:
        """Stop mock Redis."""
        super().stop()
        self._data.clear()

    # String operations
    def get(self, key: str) -> str | None:
        """Get a key value."""
        self._check_expiry(key)
        return self._data.get(key)

    def set(
        self,
        key: str,
        value: str,
        ex: int | None = None,
        px: int | None = None,
    ) -> bool:
        """Set a key value with optional TTL."""
        self._data[key] = value
        if ex is not None:
            self._expiry[key] = time.time() + ex
        elif px is not None:
            self._expiry[key] = time.time() + px / 1000
        return True

    def delete(self, *keys: str) -> int:
        """Delete keys."""
        count = 0
        for key in keys:
            if key in self._data:
                del self._data[key]
                self._expiry.pop(key, None)
                count += 1
        return count

    def exists(self, *keys: str) -> int:
        """Check if keys exist."""
        count = 0
        for key in keys:
            self._check_expiry(key)
            if key in self._data:
                count += 1
        return count

    def keys(self, pattern: str = "*") -> list[str]:
        """Get keys matching pattern."""
        import fnmatch
        result = []
        for key in list(self._data.keys()):
            self._check_expiry(key)
            if key in self._data and fnmatch.fnmatch(key, pattern):
                result.append(key)
        return result

    def expire(self, key: str, seconds: int) -> bool:
        """Set key expiry."""
        if key in self._data:
            self._expiry[key] = time.time() + seconds
            return True
        return False

    def ttl(self, key: str) -> int:
        """Get time-to-live for key."""
        if key not in self._data:
            return -2
        if key not in self._expiry:
            return -1
        remaining = self._expiry[key] - time.time()
        if remaining <= 0:
            self._check_expiry(key)
            return -2
        return int(remaining)

    # Hash operations
    def hset(self, name: str, key: str, value: str) -> int:
        """Set hash field."""
        is_new = key not in self._hashes[name]
        self._hashes[name][key] = value
        return 1 if is_new else 0

    def hget(self, name: str, key: str) -> str | None:
        """Get hash field."""
        return self._hashes.get(name, {}).get(key)

    def hgetall(self, name: str) -> dict[str, str]:
        """Get all hash fields."""
        return dict(self._hashes.get(name, {}))

    def hdel(self, name: str, *keys: str) -> int:
        """Delete hash fields."""
        count = 0
        if name in self._hashes:
            for key in keys:
                if key in self._hashes[name]:
                    del self._hashes[name][key]
                    count += 1
        return count

    # List operations
    def lpush(self, name: str, *values: str) -> int:
        """Push values to list head."""
        for value in values:
            self._lists[name].insert(0, value)
        return len(self._lists[name])

    def rpush(self, name: str, *values: str) -> int:
        """Push values to list tail."""
        self._lists[name].extend(values)
        return len(self._lists[name])

    def lrange(self, name: str, start: int, end: int) -> list[str]:
        """Get list range."""
        lst = self._lists.get(name, [])
        if end < 0:
            end = len(lst) + end + 1
        else:
            end = end + 1
        return lst[start:end]

    def llen(self, name: str) -> int:
        """Get list length."""
        return len(self._lists.get(name, []))

    # Set operations
    def sadd(self, name: str, *values: str) -> int:
        """Add values to set."""
        count = 0
        for value in values:
            if value not in self._sets[name]:
                self._sets[name].add(value)
                count += 1
        return count

    def smembers(self, name: str) -> set[str]:
        """Get set members."""
        return set(self._sets.get(name, set()))

    def sismember(self, name: str, value: str) -> bool:
        """Check if value is in set."""
        return value in self._sets.get(name, set())

    def scard(self, name: str) -> int:
        """Get set cardinality."""
        return len(self._sets.get(name, set()))

    # Pub/Sub
    def publish(self, channel: str, message: str) -> int:
        """Publish message to channel."""
        subscribers = len(self._pubsub_channels.get(channel, []))
        for callback in self._pubsub_channels.get(channel, []):
            callback({"channel": channel, "data": message})
        return subscribers

    def subscribe(self, channel: str, callback: Any) -> None:
        """Subscribe to channel."""
        self._pubsub_channels[channel].append(callback)

    # Utility
    def ping(self) -> str:
        """Ping the server."""
        return "PONG"

    def flushdb(self) -> bool:
        """Flush the database."""
        self._data.clear()
        self._expiry.clear()
        self._hashes.clear()
        self._lists.clear()
        self._sets.clear()
        return True

    def _check_expiry(self, key: str) -> None:
        """Check and clean expired keys."""
        if key in self._expiry:
            if time.time() > self._expiry[key]:
                self._data.pop(key, None)
                self._expiry.pop(key, None)


# =============================================================================
# Elasticsearch Mock
# =============================================================================


class MockElasticsearchService(MockService):
    """In-memory Elasticsearch mock for logging tests.

    Supports:
    - Index operations (create, delete, exists)
    - Document CRUD (index, get, delete)
    - Basic search
    - Bulk operations

    Example:
        >>> es = MockElasticsearchService("elasticsearch", ServiceCategory.SEARCH)
        >>> es.start()
        >>> es.index("logs", {"message": "test"})
        >>> es.search("logs", {"query": {"match_all": {}}})
    """

    def __init__(
        self,
        name: str = "elasticsearch",
        category: ServiceCategory = ServiceCategory.SEARCH,
    ):
        super().__init__(name=name, category=category)
        self._indices: dict[str, dict[str, Any]] = {}  # index -> id -> doc
        self._id_counter = 0
        self._lock = threading.Lock()

    def start(self) -> dict[str, Any]:
        """Start mock Elasticsearch."""
        result = super().start()
        self._indices.clear()
        self._id_counter = 0
        return {**result, "port": 9200}

    def stop(self) -> None:
        """Stop mock Elasticsearch."""
        super().stop()
        self._indices.clear()

    # Index operations
    def indices_create(self, index: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        """Create an index."""
        if index not in self._indices:
            self._indices[index] = {}
        return {"acknowledged": True, "index": index}

    def indices_delete(self, index: str) -> dict[str, Any]:
        """Delete an index."""
        if index in self._indices:
            del self._indices[index]
        return {"acknowledged": True}

    def indices_exists(self, index: str) -> bool:
        """Check if index exists."""
        return index in self._indices

    # Document operations
    def index(
        self,
        index: str,
        body: dict[str, Any],
        id: str | None = None,
    ) -> dict[str, Any]:
        """Index a document."""
        if index not in self._indices:
            self._indices[index] = {}

        with self._lock:
            if id is None:
                self._id_counter += 1
                id = str(self._id_counter)

        self._indices[index][id] = {
            "_source": body,
            "_id": id,
            "_index": index,
            "_version": 1,
        }

        return {
            "_index": index,
            "_id": id,
            "_version": 1,
            "result": "created",
        }

    def get(self, index: str, id: str) -> dict[str, Any]:
        """Get a document."""
        if index not in self._indices or id not in self._indices[index]:
            raise Exception(f"Document not found: {index}/{id}")
        return self._indices[index][id]

    def delete(self, index: str, id: str) -> dict[str, Any]:
        """Delete a document."""
        if index in self._indices and id in self._indices[index]:
            del self._indices[index][id]
            return {"result": "deleted"}
        return {"result": "not_found"}

    def search(
        self,
        index: str | None = None,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Search documents."""
        hits = []

        indices_to_search = [index] if index else list(self._indices.keys())

        for idx in indices_to_search:
            if idx not in self._indices:
                continue
            for doc_id, doc in self._indices[idx].items():
                # Simple matching (in production, implement query DSL)
                if self._matches_query(doc["_source"], body):
                    hits.append({
                        "_index": idx,
                        "_id": doc_id,
                        "_source": doc["_source"],
                        "_score": 1.0,
                    })

        return {
            "hits": {
                "total": {"value": len(hits), "relation": "eq"},
                "hits": hits,
            },
            "took": 1,
        }

    def bulk(self, body: list[dict[str, Any]]) -> dict[str, Any]:
        """Bulk operations."""
        items = []
        errors = False

        i = 0
        while i < len(body):
            action_line = body[i]
            action_type = list(action_line.keys())[0]
            action_meta = action_line[action_type]

            index = action_meta.get("_index")
            doc_id = action_meta.get("_id")

            if action_type in ("index", "create"):
                doc = body[i + 1] if i + 1 < len(body) else {}
                result = self.index(index, doc, id=doc_id)
                items.append({action_type: result})
                i += 2
            elif action_type == "delete":
                result = self.delete(index, doc_id)
                items.append({action_type: result})
                i += 1
            else:
                i += 1

        return {"items": items, "errors": errors, "took": 1}

    def _matches_query(self, doc: dict[str, Any], query: dict[str, Any] | None) -> bool:
        """Check if document matches query."""
        if query is None:
            return True
        query_body = query.get("query", {})
        if "match_all" in query_body:
            return True
        # Add more query type support as needed
        return True


# =============================================================================
# Vault Mock
# =============================================================================


class MockVaultService(MockService):
    """In-memory HashiCorp Vault mock for secrets tests.

    Supports:
    - KV secrets engine (v1 and v2)
    - Token authentication
    - Secret CRUD operations
    - Transit secrets engine (encryption/decryption)

    Example:
        >>> vault = MockVaultService("vault", ServiceCategory.SECRETS)
        >>> vault.start()
        >>> vault.write_secret("secret/data/myapp", {"password": "secret123"})
        >>> vault.read_secret("secret/data/myapp")
    """

    def __init__(
        self,
        name: str = "vault",
        category: ServiceCategory = ServiceCategory.SECRETS,
    ):
        super().__init__(name=name, category=category)
        self._secrets: dict[str, dict[str, Any]] = {}
        self._tokens: dict[str, dict[str, Any]] = {}
        self._transit_keys: dict[str, bytes] = {}
        self._root_token = "s.mock-root-token"

    def start(self) -> dict[str, Any]:
        """Start mock Vault."""
        result = super().start()
        self._secrets.clear()
        self._tokens.clear()
        self._tokens[self._root_token] = {
            "policies": ["root"],
            "renewable": False,
        }
        return {**result, "port": 8200, "token": self._root_token}

    def stop(self) -> None:
        """Stop mock Vault."""
        super().stop()
        self._secrets.clear()
        self._tokens.clear()

    # Authentication
    def authenticate(self, token: str) -> bool:
        """Check if token is valid."""
        return token in self._tokens

    def create_token(self, policies: list[str] | None = None) -> str:
        """Create a new token."""
        import uuid
        token = f"s.{uuid.uuid4().hex}"
        self._tokens[token] = {
            "policies": policies or ["default"],
            "renewable": True,
        }
        return token

    def revoke_token(self, token: str) -> bool:
        """Revoke a token."""
        if token in self._tokens:
            del self._tokens[token]
            return True
        return False

    # KV Secrets
    def read_secret(self, path: str) -> dict[str, Any] | None:
        """Read a secret."""
        return self._secrets.get(path)

    def write_secret(self, path: str, data: dict[str, Any]) -> bool:
        """Write a secret."""
        self._secrets[path] = {
            "data": data,
            "metadata": {
                "created_time": datetime.utcnow().isoformat(),
                "version": 1,
            },
        }
        return True

    def delete_secret(self, path: str) -> bool:
        """Delete a secret."""
        if path in self._secrets:
            del self._secrets[path]
            return True
        return False

    def list_secrets(self, path: str) -> list[str]:
        """List secrets under a path."""
        keys = []
        prefix = path.rstrip("/") + "/"
        for key in self._secrets.keys():
            if key.startswith(prefix):
                remainder = key[len(prefix):]
                if "/" in remainder:
                    keys.append(remainder.split("/")[0] + "/")
                else:
                    keys.append(remainder)
        return list(set(keys))

    # Transit Engine
    def transit_create_key(self, name: str) -> bool:
        """Create a transit encryption key."""
        import os
        self._transit_keys[name] = os.urandom(32)
        return True

    def transit_encrypt(self, key_name: str, plaintext: str) -> str:
        """Encrypt data using transit engine."""
        if key_name not in self._transit_keys:
            raise ValueError(f"Key not found: {key_name}")

        import base64
        import hashlib
        key = self._transit_keys[key_name]
        # Simple XOR encryption for mock purposes
        plaintext_bytes = plaintext.encode()
        key_stream = hashlib.sha256(key + b"encrypt").digest()
        encrypted = bytes(p ^ key_stream[i % len(key_stream)]
                         for i, p in enumerate(plaintext_bytes))
        return f"vault:v1:{base64.b64encode(encrypted).decode()}"

    def transit_decrypt(self, key_name: str, ciphertext: str) -> str:
        """Decrypt data using transit engine."""
        if key_name not in self._transit_keys:
            raise ValueError(f"Key not found: {key_name}")

        import base64
        import hashlib
        key = self._transit_keys[key_name]
        # Parse ciphertext
        if not ciphertext.startswith("vault:v1:"):
            raise ValueError("Invalid ciphertext format")
        encrypted = base64.b64decode(ciphertext[9:])
        # Simple XOR decryption
        key_stream = hashlib.sha256(key + b"encrypt").digest()
        decrypted = bytes(c ^ key_stream[i % len(key_stream)]
                         for i, c in enumerate(encrypted))
        return decrypted.decode()


# =============================================================================
# TMS Mock
# =============================================================================


class MockTMSService(MockService):
    """In-memory Translation Management System mock.

    Supports:
    - Project management
    - String management
    - Translation CRUD
    - Webhook simulation

    Example:
        >>> tms = MockTMSService("crowdin", ServiceCategory.TMS)
        >>> tms.start()
        >>> tms.create_project("my-project")
        >>> tms.add_string("my-project", "hello", "Hello, World!")
    """

    def __init__(
        self,
        name: str = "tms",
        category: ServiceCategory = ServiceCategory.TMS,
    ):
        super().__init__(name=name, category=category)
        self._projects: dict[str, dict[str, Any]] = {}
        self._strings: dict[str, dict[str, dict[str, Any]]] = {}  # project -> key -> data
        self._translations: dict[str, dict[str, dict[str, str]]] = {}  # project -> key -> locale -> translation

    def start(self) -> dict[str, Any]:
        """Start mock TMS."""
        result = super().start()
        self._projects.clear()
        self._strings.clear()
        self._translations.clear()
        return {**result, "port": 8080, "api_key": "mock-api-key"}

    def stop(self) -> None:
        """Stop mock TMS."""
        super().stop()
        self._projects.clear()

    # Project operations
    def create_project(self, project_id: str, name: str | None = None) -> dict[str, Any]:
        """Create a project."""
        self._projects[project_id] = {
            "id": project_id,
            "name": name or project_id,
            "created": datetime.utcnow().isoformat(),
        }
        self._strings[project_id] = {}
        self._translations[project_id] = {}
        return self._projects[project_id]

    def get_project(self, project_id: str) -> dict[str, Any] | None:
        """Get project details."""
        return self._projects.get(project_id)

    def list_projects(self) -> list[dict[str, Any]]:
        """List all projects."""
        return list(self._projects.values())

    def delete_project(self, project_id: str) -> bool:
        """Delete a project."""
        if project_id in self._projects:
            del self._projects[project_id]
            self._strings.pop(project_id, None)
            self._translations.pop(project_id, None)
            return True
        return False

    # String operations
    def add_string(
        self,
        project_id: str,
        key: str,
        source_text: str,
        context: str | None = None,
    ) -> dict[str, Any]:
        """Add a source string."""
        if project_id not in self._projects:
            raise ValueError(f"Project not found: {project_id}")

        self._strings[project_id][key] = {
            "key": key,
            "text": source_text,
            "context": context,
            "created": datetime.utcnow().isoformat(),
        }
        self._translations[project_id][key] = {}
        return self._strings[project_id][key]

    def get_string(self, project_id: str, key: str) -> dict[str, Any] | None:
        """Get a source string."""
        return self._strings.get(project_id, {}).get(key)

    def list_strings(self, project_id: str) -> list[dict[str, Any]]:
        """List all strings in a project."""
        return list(self._strings.get(project_id, {}).values())

    def delete_string(self, project_id: str, key: str) -> bool:
        """Delete a source string."""
        if project_id in self._strings and key in self._strings[project_id]:
            del self._strings[project_id][key]
            self._translations[project_id].pop(key, None)
            return True
        return False

    # Translation operations
    def add_translation(
        self,
        project_id: str,
        key: str,
        locale: str,
        translation: str,
    ) -> dict[str, Any]:
        """Add a translation."""
        if project_id not in self._strings or key not in self._strings[project_id]:
            raise ValueError(f"String not found: {project_id}/{key}")

        self._translations[project_id][key][locale] = translation
        return {
            "key": key,
            "locale": locale,
            "translation": translation,
        }

    def get_translation(
        self,
        project_id: str,
        key: str,
        locale: str,
    ) -> str | None:
        """Get a translation."""
        return self._translations.get(project_id, {}).get(key, {}).get(locale)

    def list_translations(
        self,
        project_id: str,
        locale: str | None = None,
    ) -> dict[str, dict[str, str]]:
        """List translations, optionally filtered by locale."""
        result = {}
        for key, locales in self._translations.get(project_id, {}).items():
            if locale:
                if locale in locales:
                    result[key] = {locale: locales[locale]}
            else:
                result[key] = dict(locales)
        return result

    def export_translations(
        self,
        project_id: str,
        locale: str,
    ) -> dict[str, str]:
        """Export translations for a locale."""
        result = {}
        for key, locales in self._translations.get(project_id, {}).items():
            if locale in locales:
                result[key] = locales[locale]
        return result


# =============================================================================
# Mock Service Provider
# =============================================================================


class MockServiceProvider:
    """Service provider using in-memory mocks.

    No external dependencies required - all services are simulated
    in memory for fast, isolated testing.
    """

    provider_type: ClassVar[ProviderType] = ProviderType.MOCK

    _services: ClassVar[dict[str, type[MockService]]] = {
        "redis": MockRedisService,
        "elasticsearch": MockElasticsearchService,
        "vault": MockVaultService,
        "tms": MockTMSService,
    }

    def __init__(self) -> None:
        """Initialize the mock provider."""
        self._instances: dict[str, MockService] = {}

    def is_available(self) -> bool:
        """Mock provider is always available."""
        return True

    def get_service_class(self, name: str) -> type[MockService] | None:
        """Get mock service class by name."""
        return self._services.get(name)

    def register_service(self, name: str, service_class: type[MockService]) -> None:
        """Register a custom mock service."""
        self._services[name] = service_class

    def start_service(self, config: ServiceConfig) -> dict[str, Any]:
        """Start a mock service."""
        service_class = self.get_service_class(config.name)
        if service_class is None:
            # Use base mock service
            service = MockService(name=config.name, category=config.category)
        else:
            service = service_class(name=config.name, category=config.category)

        self._instances[config.name] = service
        return service.start()

    def stop_service(self, config: ServiceConfig) -> None:
        """Stop a mock service."""
        service = self._instances.pop(config.name, None)
        if service is not None:
            service.stop()

    def get_connection_info(self, config: ServiceConfig) -> dict[str, Any]:
        """Get mock connection info."""
        service = self._instances.get(config.name)
        if service is None:
            return {}
        return {"host": "localhost", "port": 0, "mock": True}

    def health_check(self, config: ServiceConfig) -> HealthCheckResult:
        """Check mock service health."""
        service = self._instances.get(config.name)
        if service is None:
            return HealthCheckResult.failure("Service not started")
        return service.health_check()

    def get_instance(self, name: str) -> MockService | None:
        """Get a mock service instance for direct access."""
        return self._instances.get(name)
