"""Translation Management System backend for integration tests.

This module provides TMS integration testing with support for:
- Crowdin
- Lokalise
- Phrase
- Transifex
- POEditor
- Mock TMS (for testing)

Features:
    - Project management
    - String/translation management
    - Webhook testing
    - Export/import operations

Usage:
    >>> config = TMSConfig.from_env()
    >>> with TMSBackend(config) as backend:
    ...     backend.create_project("my-project")
    ...     backend.add_string("my-project", "hello", "Hello, World!")
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Protocol

from tests.integration.external.base import (
    ExternalServiceBackend,
    HealthCheckResult,
    ProviderType,
    ServiceCategory,
    ServiceConfig,
)
from tests.integration.external.providers.docker_provider import DockerContainerConfig
from tests.integration.external.providers.mock_provider import MockTMSService

logger = logging.getLogger(__name__)


# =============================================================================
# TMS Provider Enum
# =============================================================================


class TMSProvider(Enum):
    """Supported TMS providers."""

    CROWDIN = "crowdin"
    LOKALISE = "lokalise"
    PHRASE = "phrase"
    TRANSIFEX = "transifex"
    POEDITOR = "poeditor"
    MOCK = "mock"


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class TMSConfig(ServiceConfig):
    """TMS-specific configuration.

    Attributes:
        tms_provider: TMS provider to use
        api_key: API key for authentication
        project_id: Default project ID
        base_url: API base URL (for custom endpoints)
        organization_id: Organization ID (for some providers)
        rate_limit: Rate limit (requests per second)
        retry_attempts: Number of retry attempts
        webhook_secret: Secret for webhook verification
    """

    tms_provider: TMSProvider = TMSProvider.MOCK
    api_key: str | None = None
    project_id: str | None = None
    base_url: str | None = None
    organization_id: str | None = None
    rate_limit: float = 10.0  # requests per second
    retry_attempts: int = 3
    webhook_secret: str | None = None

    def __post_init__(self) -> None:
        """Set TMS-specific defaults."""
        self.name = self.name or "tms"
        self.category = ServiceCategory.TMS

        # Set default base URLs
        if self.base_url is None:
            self.base_url = self._get_default_base_url()

    def _get_default_base_url(self) -> str:
        """Get default base URL for provider."""
        urls = {
            TMSProvider.CROWDIN: "https://api.crowdin.com/api/v2",
            TMSProvider.LOKALISE: "https://api.lokalise.com/api2",
            TMSProvider.PHRASE: "https://api.phrase.com/v2",
            TMSProvider.TRANSIFEX: "https://rest.api.transifex.com",
            TMSProvider.POEDITOR: "https://api.poeditor.com/v2",
            TMSProvider.MOCK: "http://localhost:8080",
        }
        return urls.get(self.tms_provider, "")

    @classmethod
    def from_env(cls, name: str = "tms") -> "TMSConfig":
        """Create configuration from environment variables."""
        prefix = "TRUTHOUND_TEST_TMS"

        provider_str = os.getenv(f"{prefix}_PROVIDER", "mock")
        try:
            tms_provider = TMSProvider(provider_str.lower())
        except ValueError:
            tms_provider = TMSProvider.MOCK

        return cls(
            name=name,
            category=ServiceCategory.TMS,
            tms_provider=tms_provider,
            api_key=os.getenv(f"{prefix}_API_KEY"),
            project_id=os.getenv(f"{prefix}_PROJECT_ID"),
            base_url=os.getenv(f"{prefix}_BASE_URL"),
            organization_id=os.getenv(f"{prefix}_ORGANIZATION_ID"),
            rate_limit=float(os.getenv(f"{prefix}_RATE_LIMIT", "10.0")),
            webhook_secret=os.getenv(f"{prefix}_WEBHOOK_SECRET"),
            timeout_seconds=int(os.getenv(f"{prefix}_TIMEOUT", "30")),
        )


# =============================================================================
# TMS Client Protocol
# =============================================================================


class TMSClientProtocol(Protocol):
    """Protocol for TMS client implementations."""

    def create_project(self, project_id: str, name: str | None = None) -> dict[str, Any]:
        """Create a project."""
        ...

    def get_project(self, project_id: str) -> dict[str, Any] | None:
        """Get project details."""
        ...

    def list_projects(self) -> list[dict[str, Any]]:
        """List all projects."""
        ...

    def delete_project(self, project_id: str) -> bool:
        """Delete a project."""
        ...

    def add_string(
        self,
        project_id: str,
        key: str,
        source_text: str,
        context: str | None = None,
    ) -> dict[str, Any]:
        """Add a source string."""
        ...

    def get_string(self, project_id: str, key: str) -> dict[str, Any] | None:
        """Get a source string."""
        ...

    def list_strings(self, project_id: str) -> list[dict[str, Any]]:
        """List all strings in a project."""
        ...

    def add_translation(
        self,
        project_id: str,
        key: str,
        locale: str,
        translation: str,
    ) -> dict[str, Any]:
        """Add a translation."""
        ...

    def get_translation(self, project_id: str, key: str, locale: str) -> str | None:
        """Get a translation."""
        ...

    def export_translations(self, project_id: str, locale: str) -> dict[str, str]:
        """Export translations for a locale."""
        ...


# =============================================================================
# HTTP-based TMS Client
# =============================================================================


class HTTPTMSClient:
    """Base HTTP client for TMS APIs."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        rate_limit: float = 10.0,
        timeout: int = 30,
    ):
        """Initialize HTTP client."""
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._rate_limit = rate_limit
        self._timeout = timeout
        self._last_request_time = 0.0

    def _request(
        self,
        method: str,
        endpoint: str,
        data: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make an HTTP request with rate limiting."""
        import time
        import requests

        # Rate limiting
        elapsed = time.time() - self._last_request_time
        min_interval = 1.0 / self._rate_limit
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

        url = f"{self._base_url}{endpoint}"
        headers = self._get_headers()

        response = requests.request(
            method=method,
            url=url,
            headers=headers,
            json=data,
            params=params,
            timeout=self._timeout,
        )

        self._last_request_time = time.time()

        response.raise_for_status()
        return response.json() if response.content else {}

    def _get_headers(self) -> dict[str, str]:
        """Get request headers."""
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }


# =============================================================================
# Crowdin Client
# =============================================================================


class CrowdinClient(HTTPTMSClient):
    """Crowdin API client."""

    def create_project(self, project_id: str, name: str | None = None) -> dict[str, Any]:
        """Create a Crowdin project."""
        data = {
            "name": name or project_id,
            "sourceLanguageId": "en",
        }
        return self._request("POST", "/projects", data)

    def get_project(self, project_id: str) -> dict[str, Any] | None:
        """Get project details."""
        try:
            return self._request("GET", f"/projects/{project_id}")
        except Exception:
            return None

    def list_projects(self) -> list[dict[str, Any]]:
        """List all projects."""
        response = self._request("GET", "/projects")
        return response.get("data", [])

    def delete_project(self, project_id: str) -> bool:
        """Delete a project."""
        try:
            self._request("DELETE", f"/projects/{project_id}")
            return True
        except Exception:
            return False

    def add_string(
        self,
        project_id: str,
        key: str,
        source_text: str,
        context: str | None = None,
    ) -> dict[str, Any]:
        """Add a source string."""
        data = {
            "text": source_text,
            "identifier": key,
            "context": context,
        }
        return self._request("POST", f"/projects/{project_id}/strings", data)

    def get_string(self, project_id: str, key: str) -> dict[str, Any] | None:
        """Get a source string."""
        params = {"filter": key}
        response = self._request("GET", f"/projects/{project_id}/strings", params=params)
        strings = response.get("data", [])
        for s in strings:
            if s.get("data", {}).get("identifier") == key:
                return s["data"]
        return None

    def list_strings(self, project_id: str) -> list[dict[str, Any]]:
        """List all strings."""
        response = self._request("GET", f"/projects/{project_id}/strings")
        return [s["data"] for s in response.get("data", [])]

    def add_translation(
        self,
        project_id: str,
        key: str,
        locale: str,
        translation: str,
    ) -> dict[str, Any]:
        """Add a translation."""
        # First get string ID
        string_info = self.get_string(project_id, key)
        if not string_info:
            raise ValueError(f"String not found: {key}")

        string_id = string_info["id"]
        data = {
            "stringId": string_id,
            "languageId": locale,
            "text": translation,
        }
        return self._request("POST", f"/projects/{project_id}/translations", data)

    def get_translation(self, project_id: str, key: str, locale: str) -> str | None:
        """Get a translation."""
        # Implementation depends on Crowdin API structure
        return None

    def export_translations(self, project_id: str, locale: str) -> dict[str, str]:
        """Export translations."""
        # Crowdin uses async build process
        return {}


# =============================================================================
# TMS Backend
# =============================================================================


class TMSBackend(ExternalServiceBackend[TMSConfig, TMSClientProtocol]):
    """TMS test backend.

    Provides unified interface for testing with various TMS providers.

    Features:
        - Multi-provider support (Crowdin, Lokalise, Phrase, etc.)
        - Project management
        - String/translation operations
        - Export/import testing
    """

    service_name: ClassVar[str] = "tms"
    service_category: ClassVar[ServiceCategory] = ServiceCategory.TMS
    default_port: ClassVar[int] = 8080
    default_image: ClassVar[str] = ""

    def __init__(
        self,
        config: TMSConfig | None = None,
        provider: Any = None,
    ) -> None:
        """Initialize TMS backend."""
        if config is None:
            config = TMSConfig.from_env()
        super().__init__(config, provider)
        self._mock_service: MockTMSService | None = None

    def _create_client(self) -> TMSClientProtocol:
        """Create TMS client based on provider."""
        config = self.config

        if config.tms_provider == TMSProvider.MOCK:
            return self._create_mock_client()

        elif config.tms_provider == TMSProvider.CROWDIN:
            if not config.api_key:
                raise ValueError("Crowdin API key required")
            return CrowdinClient(
                base_url=config.base_url or "",
                api_key=config.api_key,
                rate_limit=config.rate_limit,
                timeout=config.timeout_seconds,
            )

        elif config.tms_provider == TMSProvider.LOKALISE:
            return self._create_lokalise_client(config)

        elif config.tms_provider == TMSProvider.PHRASE:
            return self._create_phrase_client(config)

        elif config.tms_provider == TMSProvider.TRANSIFEX:
            return self._create_transifex_client(config)

        elif config.tms_provider == TMSProvider.POEDITOR:
            return self._create_poeditor_client(config)

        else:
            raise ValueError(f"Unsupported TMS provider: {config.tms_provider}")

    def _create_mock_client(self) -> TMSClientProtocol:
        """Create mock TMS client."""
        self._mock_service = MockTMSService("tms", ServiceCategory.TMS)
        self._mock_service.start()
        return self._mock_service  # type: ignore

    def _create_lokalise_client(self, config: TMSConfig) -> TMSClientProtocol:
        """Create Lokalise client."""
        raise NotImplementedError("Lokalise client not implemented")

    def _create_phrase_client(self, config: TMSConfig) -> TMSClientProtocol:
        """Create Phrase client."""
        raise NotImplementedError("Phrase client not implemented")

    def _create_transifex_client(self, config: TMSConfig) -> TMSClientProtocol:
        """Create Transifex client."""
        raise NotImplementedError("Transifex client not implemented")

    def _create_poeditor_client(self, config: TMSConfig) -> TMSClientProtocol:
        """Create POEditor client."""
        raise NotImplementedError("POEditor client not implemented")

    def _close_client(self) -> None:
        """Close TMS client."""
        if self._mock_service is not None:
            self._mock_service.stop()
            self._mock_service = None

    def _perform_health_check(self) -> HealthCheckResult:
        """Perform TMS health check."""
        if self._client is None:
            return HealthCheckResult.failure("Client not connected")

        try:
            projects = self._client.list_projects()
            return HealthCheckResult.success(
                "TMS healthy",
                provider=self.config.tms_provider.value,
                project_count=len(projects),
            )
        except Exception as e:
            return HealthCheckResult.failure(str(e))

    # -------------------------------------------------------------------------
    # Project Operations
    # -------------------------------------------------------------------------

    def create_project(
        self,
        project_id: str,
        name: str | None = None,
    ) -> dict[str, Any]:
        """Create a project.

        Args:
            project_id: Project identifier
            name: Project name (defaults to project_id)

        Returns:
            Project data
        """
        if self._client is None:
            raise RuntimeError("Client not connected")
        return self._client.create_project(project_id, name)

    def get_project(self, project_id: str) -> dict[str, Any] | None:
        """Get project details."""
        if self._client is None:
            return None
        return self._client.get_project(project_id)

    def list_projects(self) -> list[dict[str, Any]]:
        """List all projects."""
        if self._client is None:
            return []
        return self._client.list_projects()

    def delete_project(self, project_id: str) -> bool:
        """Delete a project."""
        if self._client is None:
            return False
        return self._client.delete_project(project_id)

    # -------------------------------------------------------------------------
    # String Operations
    # -------------------------------------------------------------------------

    def add_string(
        self,
        project_id: str,
        key: str,
        source_text: str,
        context: str | None = None,
    ) -> dict[str, Any]:
        """Add a source string.

        Args:
            project_id: Project identifier
            key: String key/identifier
            source_text: Source language text
            context: Optional context for translators

        Returns:
            String data
        """
        if self._client is None:
            raise RuntimeError("Client not connected")
        return self._client.add_string(project_id, key, source_text, context)

    def get_string(self, project_id: str, key: str) -> dict[str, Any] | None:
        """Get a source string."""
        if self._client is None:
            return None
        return self._client.get_string(project_id, key)

    def list_strings(self, project_id: str) -> list[dict[str, Any]]:
        """List all strings in a project."""
        if self._client is None:
            return []
        return self._client.list_strings(project_id)

    # -------------------------------------------------------------------------
    # Translation Operations
    # -------------------------------------------------------------------------

    def add_translation(
        self,
        project_id: str,
        key: str,
        locale: str,
        translation: str,
    ) -> dict[str, Any]:
        """Add a translation.

        Args:
            project_id: Project identifier
            key: String key
            locale: Target locale code
            translation: Translated text

        Returns:
            Translation data
        """
        if self._client is None:
            raise RuntimeError("Client not connected")
        return self._client.add_translation(project_id, key, locale, translation)

    def get_translation(
        self,
        project_id: str,
        key: str,
        locale: str,
    ) -> str | None:
        """Get a translation."""
        if self._client is None:
            return None
        return self._client.get_translation(project_id, key, locale)

    def export_translations(
        self,
        project_id: str,
        locale: str,
    ) -> dict[str, str]:
        """Export translations for a locale.

        Args:
            project_id: Project identifier
            locale: Locale code

        Returns:
            Dictionary of key -> translation
        """
        if self._client is None:
            return {}
        return self._client.export_translations(project_id, locale)

    # -------------------------------------------------------------------------
    # Bulk Operations
    # -------------------------------------------------------------------------

    def bulk_add_strings(
        self,
        project_id: str,
        strings: list[dict[str, str]],
    ) -> list[dict[str, Any]]:
        """Add multiple strings at once.

        Args:
            project_id: Project identifier
            strings: List of {"key": ..., "text": ..., "context": ...}

        Returns:
            List of created string data
        """
        results = []
        for s in strings:
            result = self.add_string(
                project_id,
                s["key"],
                s["text"],
                s.get("context"),
            )
            results.append(result)
        return results

    def bulk_add_translations(
        self,
        project_id: str,
        translations: list[dict[str, str]],
    ) -> list[dict[str, Any]]:
        """Add multiple translations at once.

        Args:
            project_id: Project identifier
            translations: List of {"key": ..., "locale": ..., "translation": ...}

        Returns:
            List of created translation data
        """
        results = []
        for t in translations:
            result = self.add_translation(
                project_id,
                t["key"],
                t["locale"],
                t["translation"],
            )
            results.append(result)
        return results


# =============================================================================
# Test Helpers
# =============================================================================


def create_tms_backend(
    provider_type: ProviderType = ProviderType.MOCK,
    tms_provider: TMSProvider = TMSProvider.MOCK,
) -> TMSBackend:
    """Create a TMS backend with specified providers."""
    config = TMSConfig.from_env()
    config.provider = provider_type
    config.tms_provider = tms_provider

    return TMSBackend(config)
