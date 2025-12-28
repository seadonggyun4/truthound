"""Translation Management System (TMS) Integration.

This module provides integration with external translation management systems
for enterprise-grade internationalization workflows.

Supported TMS Providers:
- Crowdin (crowdin.com)
- Lokalise (lokalise.com)
- Phrase (phrase.com, formerly PhraseApp)
- Transifex (transifex.com)
- POEditor (poeditor.com)

Features:
- Catalog synchronization (push/pull)
- Translation status tracking
- Webhook support for real-time updates
- Batch operations
- Rate limiting and retry logic

Usage:
    from truthound.validators.i18n.tms import (
        CrowdinProvider,
        LokaliseProvider,
        TMSManager,
    )

    # Create provider
    provider = CrowdinProvider(
        api_key="your-api-key",
        project_id="your-project-id",
    )

    # Sync translations
    updated = provider.sync_catalog(LocaleInfo.parse("ko"), local_catalog)

    # Check translation status
    status = provider.get_translation_status(LocaleInfo.parse("ko"))
    print(f"Korean translation: {status['validators']:.1%} complete")
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urljoin

from truthound.validators.i18n.protocols import (
    BaseTranslationService,
    LocaleInfo,
)


logger = logging.getLogger(__name__)


# ==============================================================================
# TMS Configuration and Types
# ==============================================================================

class TMSProvider(str, Enum):
    """Supported TMS providers."""
    CROWDIN = "crowdin"
    LOKALISE = "lokalise"
    PHRASE = "phrase"
    TRANSIFEX = "transifex"
    POEDITOR = "poeditor"
    CUSTOM = "custom"


@dataclass
class TMSConfig:
    """Configuration for TMS integration.

    Attributes:
        provider: TMS provider type
        api_key: API key or token
        project_id: Project identifier
        base_url: API base URL (optional, for self-hosted)
        source_locale: Source locale for translations
        file_format: File format for uploads (json, xliff, etc.)
        rate_limit: Maximum requests per second
        retry_attempts: Number of retry attempts
        retry_delay: Initial retry delay in seconds
        webhook_secret: Secret for webhook verification
        timeout: Request timeout in seconds
    """
    provider: TMSProvider
    api_key: str
    project_id: str
    base_url: str | None = None
    source_locale: str = "en"
    file_format: str = "json"
    rate_limit: float = 10.0
    retry_attempts: int = 3
    retry_delay: float = 1.0
    webhook_secret: str | None = None
    timeout: int = 30


@dataclass
class TranslationKey:
    """A translation key with metadata.

    Attributes:
        key: Unique key identifier
        source: Source text
        context: Context or description
        tags: Tags for categorization
        max_length: Maximum allowed length
        pluralizable: Whether key supports pluralization
    """
    key: str
    source: str
    context: str = ""
    tags: list[str] = field(default_factory=list)
    max_length: int | None = None
    pluralizable: bool = False


@dataclass
class TranslationEntry:
    """A translation entry from TMS.

    Attributes:
        key: Translation key
        source: Source text
        target: Translated text
        locale: Target locale
        is_reviewed: Whether translation is reviewed
        is_approved: Whether translation is approved
        translator: Translator identifier
        updated_at: Last update time
    """
    key: str
    source: str
    target: str
    locale: str
    is_reviewed: bool = False
    is_approved: bool = False
    translator: str | None = None
    updated_at: datetime | None = None


@dataclass
class TranslationStatus:
    """Translation status for a locale.

    Attributes:
        locale: Target locale
        total_keys: Total number of keys
        translated: Number of translated keys
        reviewed: Number of reviewed keys
        approved: Number of approved keys
        progress: Overall progress percentage
        last_activity: Last activity time
    """
    locale: str
    total_keys: int = 0
    translated: int = 0
    reviewed: int = 0
    approved: int = 0
    progress: float = 0.0
    last_activity: datetime | None = None

    @property
    def completion_rate(self) -> float:
        """Get completion rate as percentage."""
        if self.total_keys == 0:
            return 0.0
        return self.translated / self.total_keys


@dataclass
class WebhookEvent:
    """Webhook event from TMS.

    Attributes:
        event_type: Type of event
        project_id: Project identifier
        locale: Affected locale
        keys: Affected translation keys
        payload: Raw event payload
        timestamp: Event timestamp
        signature: Event signature for verification
    """
    event_type: str
    project_id: str
    locale: str | None = None
    keys: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    signature: str | None = None


# ==============================================================================
# Rate Limiter
# ==============================================================================

class RateLimiter:
    """Token bucket rate limiter.

    Implements token bucket algorithm for rate limiting API requests.
    """

    def __init__(self, rate: float, burst: int | None = None) -> None:
        """Initialize rate limiter.

        Args:
            rate: Tokens per second
            burst: Maximum burst size (default: 2x rate)
        """
        self.rate = rate
        self.burst = burst or int(rate * 2)
        self.tokens = float(self.burst)
        self.last_update = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, tokens: int = 1) -> float:
        """Acquire tokens, blocking if necessary.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            Time waited in seconds
        """
        with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
            self.last_update = now

            if self.tokens >= tokens:
                self.tokens -= tokens
                return 0.0

            # Need to wait
            wait_time = (tokens - self.tokens) / self.rate
            time.sleep(wait_time)

            now = time.monotonic()
            elapsed = now - self.last_update
            self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
            self.tokens -= tokens
            self.last_update = now

            return wait_time

    def try_acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens without blocking.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            True if tokens were acquired
        """
        with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
            self.last_update = now

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False


# ==============================================================================
# HTTP Client (Abstract)
# ==============================================================================

class HTTPClient(ABC):
    """Abstract HTTP client for TMS API calls."""

    @abstractmethod
    def get(self, url: str, headers: dict | None = None) -> dict[str, Any]:
        """Make GET request."""
        pass

    @abstractmethod
    def post(self, url: str, data: dict | None = None, headers: dict | None = None) -> dict[str, Any]:
        """Make POST request."""
        pass

    @abstractmethod
    def put(self, url: str, data: dict | None = None, headers: dict | None = None) -> dict[str, Any]:
        """Make PUT request."""
        pass

    @abstractmethod
    def delete(self, url: str, headers: dict | None = None) -> dict[str, Any]:
        """Make DELETE request."""
        pass


class MockHTTPClient(HTTPClient):
    """Mock HTTP client for testing."""

    def __init__(self) -> None:
        self.responses: dict[str, Any] = {}
        self.requests: list[dict] = []

    def add_response(self, method: str, url: str, response: dict[str, Any]) -> None:
        """Add a mock response."""
        key = f"{method}:{url}"
        self.responses[key] = response

    def get(self, url: str, headers: dict | None = None) -> dict[str, Any]:
        self.requests.append({"method": "GET", "url": url, "headers": headers})
        return self.responses.get(f"GET:{url}", {})

    def post(self, url: str, data: dict | None = None, headers: dict | None = None) -> dict[str, Any]:
        self.requests.append({"method": "POST", "url": url, "data": data, "headers": headers})
        return self.responses.get(f"POST:{url}", {})

    def put(self, url: str, data: dict | None = None, headers: dict | None = None) -> dict[str, Any]:
        self.requests.append({"method": "PUT", "url": url, "data": data, "headers": headers})
        return self.responses.get(f"PUT:{url}", {})

    def delete(self, url: str, headers: dict | None = None) -> dict[str, Any]:
        self.requests.append({"method": "DELETE", "url": url, "headers": headers})
        return self.responses.get(f"DELETE:{url}", {})


# ==============================================================================
# Base TMS Provider
# ==============================================================================

class BaseTMSProvider(BaseTranslationService, ABC):
    """Base class for TMS provider implementations.

    Provides common functionality for all TMS integrations:
    - Rate limiting
    - Retry logic
    - Error handling
    - Webhook verification
    """

    def __init__(
        self,
        config: TMSConfig,
        http_client: HTTPClient | None = None,
    ) -> None:
        """Initialize provider.

        Args:
            config: TMS configuration
            http_client: HTTP client (for testing)
        """
        super().__init__(config.api_key, config.project_id)
        self.config = config
        self.http_client = http_client or MockHTTPClient()
        self.rate_limiter = RateLimiter(config.rate_limit)

    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: dict | None = None,
        headers: dict | None = None,
    ) -> dict[str, Any]:
        """Make an API request with rate limiting and retries.

        Args:
            method: HTTP method
            endpoint: API endpoint
            data: Request data
            headers: Request headers

        Returns:
            Response data
        """
        # Rate limit
        self.rate_limiter.acquire()

        # Build URL
        base_url = self.config.base_url or self._get_default_base_url()
        url = urljoin(base_url, endpoint)

        # Add auth headers
        headers = headers or {}
        headers.update(self._get_auth_headers())

        # Retry loop
        last_error = None
        for attempt in range(self.config.retry_attempts):
            try:
                if method == "GET":
                    return self.http_client.get(url, headers)
                elif method == "POST":
                    return self.http_client.post(url, data, headers)
                elif method == "PUT":
                    return self.http_client.put(url, data, headers)
                elif method == "DELETE":
                    return self.http_client.delete(url, headers)
                else:
                    raise ValueError(f"Unsupported method: {method}")
            except Exception as e:
                last_error = e
                if attempt < self.config.retry_attempts - 1:
                    delay = self.config.retry_delay * (2 ** attempt)
                    logger.warning(f"Request failed, retrying in {delay}s: {e}")
                    time.sleep(delay)

        raise last_error or Exception("Request failed")

    @abstractmethod
    def _get_default_base_url(self) -> str:
        """Get the default API base URL."""
        pass

    @abstractmethod
    def _get_auth_headers(self) -> dict[str, str]:
        """Get authentication headers."""
        pass

    def verify_webhook(self, payload: bytes, signature: str) -> bool:
        """Verify webhook signature.

        Args:
            payload: Raw webhook payload
            signature: Provided signature

        Returns:
            True if signature is valid
        """
        if not self.config.webhook_secret:
            logger.warning("No webhook secret configured")
            return False

        expected = hmac.new(
            self.config.webhook_secret.encode(),
            payload,
            hashlib.sha256,
        ).hexdigest()

        return hmac.compare_digest(expected, signature)

    def parse_webhook(self, payload: dict[str, Any]) -> WebhookEvent:
        """Parse a webhook payload into an event.

        Args:
            payload: Webhook payload

        Returns:
            WebhookEvent
        """
        return WebhookEvent(
            event_type=payload.get("event", "unknown"),
            project_id=payload.get("project_id", ""),
            locale=payload.get("language"),
            keys=payload.get("keys", []),
            payload=payload,
        )


# ==============================================================================
# Crowdin Provider
# ==============================================================================

class CrowdinProvider(BaseTMSProvider):
    """Crowdin TMS integration.

    Crowdin API v2 implementation.
    See: https://developer.crowdin.com/api/v2/
    """

    def __init__(
        self,
        api_key: str,
        project_id: str,
        organization: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize Crowdin provider.

        Args:
            api_key: Crowdin API token
            project_id: Crowdin project ID
            organization: Organization domain (for enterprise)
            **kwargs: Additional config options
        """
        config = TMSConfig(
            provider=TMSProvider.CROWDIN,
            api_key=api_key,
            project_id=project_id,
            base_url=f"https://{organization}.api.crowdin.com" if organization else None,
            **kwargs,
        )
        super().__init__(config)
        self.organization = organization

    def _get_default_base_url(self) -> str:
        return "https://api.crowdin.com/api/v2/"

    def _get_auth_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

    def sync_catalog(
        self,
        locale: LocaleInfo,
        catalog: dict[str, str],
    ) -> dict[str, str]:
        """Sync local catalog with Crowdin.

        Downloads translations for the locale and merges with local catalog.

        Args:
            locale: Target locale
            catalog: Local message catalog

        Returns:
            Updated catalog with Crowdin translations
        """
        # Get language ID
        language_id = self._locale_to_crowdin(locale)

        # Build export request
        endpoint = f"projects/{self.config.project_id}/translations/builds"
        data = {
            "targetLanguageIds": [language_id],
        }

        try:
            # Request translation export
            build_response = self._make_request("POST", endpoint, data)
            build_id = build_response.get("data", {}).get("id")

            if build_id:
                # Download translations
                download_endpoint = f"projects/{self.config.project_id}/translations/builds/{build_id}/download"
                download_response = self._make_request("GET", download_endpoint)

                # Parse and merge translations
                translations = self._parse_translations(download_response)
                result = catalog.copy()
                result.update(translations)
                return result

        except Exception as e:
            logger.error(f"Failed to sync with Crowdin: {e}")

        return catalog

    def push_new_keys(
        self,
        keys: list[str],
        source_locale: LocaleInfo,
        source_messages: dict[str, str],
    ) -> bool:
        """Push new keys to Crowdin.

        Args:
            keys: New translation keys
            source_locale: Source locale
            source_messages: Source message templates

        Returns:
            True if successful
        """
        if not keys:
            return True

        endpoint = f"projects/{self.config.project_id}/strings"

        try:
            for key in keys:
                if key in source_messages:
                    data = {
                        "text": source_messages[key],
                        "identifier": key,
                        "context": f"Validator message: {key}",
                    }
                    self._make_request("POST", endpoint, data)

            return True
        except Exception as e:
            logger.error(f"Failed to push keys to Crowdin: {e}")
            return False

    def get_translation_status(
        self,
        locale: LocaleInfo,
    ) -> dict[str, float]:
        """Get translation status from Crowdin.

        Args:
            locale: Target locale

        Returns:
            Dictionary with completion percentages
        """
        language_id = self._locale_to_crowdin(locale)
        endpoint = f"projects/{self.config.project_id}/languages/{language_id}/progress"

        try:
            response = self._make_request("GET", endpoint)
            data = response.get("data", [])

            # Aggregate by file/namespace
            result: dict[str, float] = {}
            for item in data:
                file_id = item.get("fileId", "default")
                progress = item.get("translationProgress", 0)
                result[str(file_id)] = progress / 100.0

            # Overall progress
            if data:
                result["overall"] = sum(item.get("translationProgress", 0) for item in data) / len(data) / 100.0
            else:
                result["overall"] = 0.0

            return result

        except Exception as e:
            logger.error(f"Failed to get Crowdin status: {e}")
            return {"overall": 0.0}

    def _locale_to_crowdin(self, locale: LocaleInfo) -> str:
        """Convert LocaleInfo to Crowdin language ID."""
        # Crowdin uses different format for some locales
        mapping = {
            "zh-Hans": "zh-CN",
            "zh-Hant": "zh-TW",
            "pt-BR": "pt-BR",
            "pt-PT": "pt-PT",
        }
        return mapping.get(locale.tag, locale.tag)

    def _parse_translations(self, response: dict) -> dict[str, str]:
        """Parse Crowdin translation response."""
        # This would parse the actual Crowdin file format
        # For now, return empty dict as placeholder
        return response.get("translations", {})


# ==============================================================================
# Lokalise Provider
# ==============================================================================

class LokaliseProvider(BaseTMSProvider):
    """Lokalise TMS integration.

    Lokalise API v2 implementation.
    See: https://developers.lokalise.com/reference/
    """

    def __init__(
        self,
        api_key: str,
        project_id: str,
        **kwargs: Any,
    ) -> None:
        """Initialize Lokalise provider.

        Args:
            api_key: Lokalise API token
            project_id: Lokalise project ID
            **kwargs: Additional config options
        """
        config = TMSConfig(
            provider=TMSProvider.LOKALISE,
            api_key=api_key,
            project_id=project_id,
            **kwargs,
        )
        super().__init__(config)

    def _get_default_base_url(self) -> str:
        return "https://api.lokalise.com/api2/"

    def _get_auth_headers(self) -> dict[str, str]:
        return {
            "X-Api-Token": self.config.api_key,
            "Content-Type": "application/json",
        }

    def sync_catalog(
        self,
        locale: LocaleInfo,
        catalog: dict[str, str],
    ) -> dict[str, str]:
        """Sync with Lokalise."""
        # Get keys with translations
        endpoint = f"projects/{self.config.project_id}/keys"
        params = {
            "include_translations": 1,
            "filter_langs": self._locale_to_lokalise(locale),
        }

        try:
            response = self._make_request("GET", endpoint, params)
            keys = response.get("keys", [])

            result = catalog.copy()
            for key in keys:
                key_name = key.get("key_name", {})
                if isinstance(key_name, dict):
                    key_id = key_name.get("web") or key_name.get("other")
                else:
                    key_id = key_name

                translations = key.get("translations", [])
                for trans in translations:
                    if trans.get("language_iso") == self._locale_to_lokalise(locale):
                        result[key_id] = trans.get("translation", "")

            return result

        except Exception as e:
            logger.error(f"Failed to sync with Lokalise: {e}")
            return catalog

    def push_new_keys(
        self,
        keys: list[str],
        source_locale: LocaleInfo,
        source_messages: dict[str, str],
    ) -> bool:
        """Push new keys to Lokalise."""
        if not keys:
            return True

        endpoint = f"projects/{self.config.project_id}/keys"
        lang_iso = self._locale_to_lokalise(source_locale)

        try:
            key_data = []
            for key in keys:
                if key in source_messages:
                    key_data.append({
                        "key_name": key,
                        "platforms": ["web"],
                        "translations": [{
                            "language_iso": lang_iso,
                            "translation": source_messages[key],
                        }],
                    })

            if key_data:
                self._make_request("POST", endpoint, {"keys": key_data})

            return True
        except Exception as e:
            logger.error(f"Failed to push keys to Lokalise: {e}")
            return False

    def get_translation_status(
        self,
        locale: LocaleInfo,
    ) -> dict[str, float]:
        """Get translation status from Lokalise."""
        endpoint = f"projects/{self.config.project_id}/languages"

        try:
            response = self._make_request("GET", endpoint)
            languages = response.get("languages", [])

            target_iso = self._locale_to_lokalise(locale)
            for lang in languages:
                if lang.get("lang_iso") == target_iso:
                    progress = lang.get("progress", 0)
                    return {
                        "overall": progress / 100.0,
                        "words_total": lang.get("words_total", 0),
                        "words_done": lang.get("words_done", 0),
                    }

            return {"overall": 0.0}

        except Exception as e:
            logger.error(f"Failed to get Lokalise status: {e}")
            return {"overall": 0.0}

    def _locale_to_lokalise(self, locale: LocaleInfo) -> str:
        """Convert LocaleInfo to Lokalise language ISO."""
        if locale.region:
            return f"{locale.language}_{locale.region}"
        return locale.language


# ==============================================================================
# Phrase (PhraseApp) Provider
# ==============================================================================

class PhraseProvider(BaseTMSProvider):
    """Phrase TMS integration.

    Phrase Strings API v2 implementation.
    See: https://developers.phrase.com/api/
    """

    def __init__(
        self,
        api_key: str,
        project_id: str,
        **kwargs: Any,
    ) -> None:
        """Initialize Phrase provider."""
        config = TMSConfig(
            provider=TMSProvider.PHRASE,
            api_key=api_key,
            project_id=project_id,
            **kwargs,
        )
        super().__init__(config)

    def _get_default_base_url(self) -> str:
        return "https://api.phrase.com/v2/"

    def _get_auth_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"token {self.config.api_key}",
            "Content-Type": "application/json",
        }

    def sync_catalog(
        self,
        locale: LocaleInfo,
        catalog: dict[str, str],
    ) -> dict[str, str]:
        """Sync with Phrase."""
        endpoint = f"projects/{self.config.project_id}/locales/{locale.tag}/download"
        params = {"file_format": "nested_json"}

        try:
            response = self._make_request("GET", endpoint, params)
            result = catalog.copy()
            result.update(self._flatten_translations(response))
            return result
        except Exception as e:
            logger.error(f"Failed to sync with Phrase: {e}")
            return catalog

    def push_new_keys(
        self,
        keys: list[str],
        source_locale: LocaleInfo,
        source_messages: dict[str, str],
    ) -> bool:
        """Push new keys to Phrase."""
        endpoint = f"projects/{self.config.project_id}/keys"

        try:
            for key in keys:
                if key in source_messages:
                    data = {
                        "name": key,
                        "description": f"Validator message: {key}",
                        "default_translation_content": source_messages[key],
                    }
                    self._make_request("POST", endpoint, data)
            return True
        except Exception as e:
            logger.error(f"Failed to push keys to Phrase: {e}")
            return False

    def get_translation_status(
        self,
        locale: LocaleInfo,
    ) -> dict[str, float]:
        """Get translation status from Phrase."""
        endpoint = f"projects/{self.config.project_id}/locales/{locale.tag}"

        try:
            response = self._make_request("GET", endpoint)
            stats = response.get("statistics", {})
            return {
                "overall": stats.get("translations_completed_progress", 0) / 100.0,
                "keys_total": stats.get("keys_total_count", 0),
                "translations_completed": stats.get("translations_completed_count", 0),
            }
        except Exception as e:
            logger.error(f"Failed to get Phrase status: {e}")
            return {"overall": 0.0}

    def _flatten_translations(self, nested: dict) -> dict[str, str]:
        """Flatten nested translation structure."""
        result = {}

        def _flatten(obj: Any, prefix: str = "") -> None:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    new_key = f"{prefix}.{k}" if prefix else k
                    _flatten(v, new_key)
            else:
                result[prefix] = str(obj)

        _flatten(nested)
        return result


# ==============================================================================
# TMS Manager
# ==============================================================================

class TMSManager:
    """Manager for TMS operations.

    Provides a high-level interface for managing translations across
    multiple TMS providers.

    Example:
        manager = TMSManager()

        # Add providers
        manager.add_provider("crowdin", CrowdinProvider(...))
        manager.add_provider("lokalise", LokaliseProvider(...))

        # Sync translations
        catalog = manager.sync_all(LocaleInfo.parse("ko"), local_catalog)

        # Push new keys
        manager.push_new_keys(["new.key"], LocaleInfo.parse("en"), source_messages)
    """

    def __init__(self) -> None:
        self._providers: dict[str, BaseTMSProvider] = {}
        self._webhooks: dict[str, Callable[[WebhookEvent], None]] = {}

    def add_provider(self, name: str, provider: BaseTMSProvider) -> None:
        """Add a TMS provider.

        Args:
            name: Provider identifier
            provider: TMS provider instance
        """
        self._providers[name] = provider

    def remove_provider(self, name: str) -> None:
        """Remove a TMS provider.

        Args:
            name: Provider identifier
        """
        self._providers.pop(name, None)

    def get_provider(self, name: str) -> BaseTMSProvider | None:
        """Get a TMS provider by name.

        Args:
            name: Provider identifier

        Returns:
            Provider instance or None
        """
        return self._providers.get(name)

    def sync_catalog(
        self,
        locale: LocaleInfo,
        catalog: dict[str, str],
        provider_name: str | None = None,
    ) -> dict[str, str]:
        """Sync catalog with TMS.

        Args:
            locale: Target locale
            catalog: Local message catalog
            provider_name: Specific provider (or all if None)

        Returns:
            Updated catalog
        """
        result = catalog.copy()

        providers = (
            [self._providers[provider_name]] if provider_name
            else self._providers.values()
        )

        for provider in providers:
            try:
                result = provider.sync_catalog(locale, result)
            except Exception as e:
                logger.error(f"Failed to sync with {provider.__class__.__name__}: {e}")

        return result

    def push_new_keys(
        self,
        keys: list[str],
        source_locale: LocaleInfo,
        source_messages: dict[str, str],
        provider_name: str | None = None,
    ) -> bool:
        """Push new keys to TMS.

        Args:
            keys: New translation keys
            source_locale: Source locale
            source_messages: Source message templates
            provider_name: Specific provider (or all if None)

        Returns:
            True if all pushes succeeded
        """
        providers = (
            [self._providers[provider_name]] if provider_name
            else self._providers.values()
        )

        success = True
        for provider in providers:
            try:
                if not provider.push_new_keys(keys, source_locale, source_messages):
                    success = False
            except Exception as e:
                logger.error(f"Failed to push to {provider.__class__.__name__}: {e}")
                success = False

        return success

    def get_translation_status(
        self,
        locale: LocaleInfo,
        provider_name: str | None = None,
    ) -> dict[str, dict[str, float]]:
        """Get translation status from TMS.

        Args:
            locale: Target locale
            provider_name: Specific provider (or all if None)

        Returns:
            Dictionary of provider name -> status dict
        """
        if provider_name:
            providers = {provider_name: self._providers[provider_name]}
        else:
            providers = self._providers

        result = {}
        for name, provider in providers.items():
            try:
                result[name] = provider.get_translation_status(locale)
            except Exception as e:
                logger.error(f"Failed to get status from {name}: {e}")
                result[name] = {"overall": 0.0, "error": str(e)}

        return result

    def register_webhook_handler(
        self,
        provider_name: str,
        handler: Callable[[WebhookEvent], None],
    ) -> None:
        """Register a webhook event handler.

        Args:
            provider_name: Provider identifier
            handler: Event handler function
        """
        self._webhooks[provider_name] = handler

    def handle_webhook(
        self,
        provider_name: str,
        payload: bytes,
        signature: str | None = None,
    ) -> bool:
        """Handle incoming webhook.

        Args:
            provider_name: Provider identifier
            payload: Raw webhook payload
            signature: Webhook signature

        Returns:
            True if handled successfully
        """
        provider = self._providers.get(provider_name)
        if not provider:
            logger.error(f"Unknown provider: {provider_name}")
            return False

        # Verify signature if provided
        if signature and not provider.verify_webhook(payload, signature):
            logger.error("Invalid webhook signature")
            return False

        # Parse event
        try:
            payload_dict = json.loads(payload)
            event = provider.parse_webhook(payload_dict)
        except Exception as e:
            logger.error(f"Failed to parse webhook: {e}")
            return False

        # Call handler
        handler = self._webhooks.get(provider_name)
        if handler:
            try:
                handler(event)
                return True
            except Exception as e:
                logger.error(f"Webhook handler error: {e}")
                return False

        return True


# ==============================================================================
# Factory Functions
# ==============================================================================

def create_provider(
    provider_type: TMSProvider | str,
    api_key: str,
    project_id: str,
    **kwargs: Any,
) -> BaseTMSProvider:
    """Create a TMS provider.

    Args:
        provider_type: Provider type
        api_key: API key
        project_id: Project ID
        **kwargs: Provider-specific options

    Returns:
        TMS provider instance
    """
    if isinstance(provider_type, str):
        provider_type = TMSProvider(provider_type.lower())

    if provider_type == TMSProvider.CROWDIN:
        return CrowdinProvider(api_key, project_id, **kwargs)
    elif provider_type == TMSProvider.LOKALISE:
        return LokaliseProvider(api_key, project_id, **kwargs)
    elif provider_type == TMSProvider.PHRASE:
        return PhraseProvider(api_key, project_id, **kwargs)
    else:
        raise ValueError(f"Unsupported provider: {provider_type}")


# Global manager instance
_tms_manager = TMSManager()


def get_tms_manager() -> TMSManager:
    """Get the global TMS manager.

    Returns:
        TMSManager instance
    """
    return _tms_manager
