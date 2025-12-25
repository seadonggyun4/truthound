"""Base classes and protocols for secret management.

This module defines the core abstractions for secret providers,
enabling a pluggable architecture for different secret backends.

Design Principles:
    1. Protocol-based: Duck typing for flexibility
    2. Immutable values: SecretValue is immutable after creation
    3. Lazy evaluation: Secrets are fetched on demand
    4. Secure by default: Values are redacted in repr/str
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Iterator,
    Protocol,
    TypeVar,
    runtime_checkable,
)

if TYPE_CHECKING:
    pass


# =============================================================================
# Exceptions
# =============================================================================


class SecretError(Exception):
    """Base exception for secret-related errors."""

    pass


class SecretNotFoundError(SecretError):
    """Raised when a secret is not found."""

    def __init__(self, key: str, provider: str | None = None) -> None:
        self.key = key
        self.provider = provider
        msg = f"Secret not found: {key}"
        if provider:
            msg += f" (provider: {provider})"
        super().__init__(msg)


class SecretAccessError(SecretError):
    """Raised when access to a secret is denied."""

    def __init__(
        self, key: str, reason: str, provider: str | None = None
    ) -> None:
        self.key = key
        self.reason = reason
        self.provider = provider
        msg = f"Access denied for secret '{key}': {reason}"
        if provider:
            msg += f" (provider: {provider})"
        super().__init__(msg)


class SecretProviderError(SecretError):
    """Raised when a secret provider encounters an error."""

    def __init__(
        self, provider: str, message: str, cause: Exception | None = None
    ) -> None:
        self.provider = provider
        self.cause = cause
        msg = f"Provider '{provider}' error: {message}"
        super().__init__(msg)
        if cause:
            self.__cause__ = cause


# =============================================================================
# Secret Value
# =============================================================================


class SecretValue:
    """Immutable, secure container for secret values.

    SecretValue wraps a secret string and provides security features:
    - Value is not exposed in repr/str
    - Comparison is constant-time to prevent timing attacks
    - Hash is computed for change detection
    - Metadata tracks provenance

    Example:
        >>> secret = SecretValue("my-api-key", provider="env", key="API_KEY")
        >>> print(secret)  # "SecretValue(***)"
        >>> secret.get_value()  # "my-api-key"
        >>> secret == "my-api-key"  # True (constant-time comparison)
    """

    __slots__ = (
        "_value",
        "_hash",
        "_provider",
        "_key",
        "_created_at",
        "_expires_at",
        "_metadata",
    )

    def __init__(
        self,
        value: str,
        *,
        provider: str = "unknown",
        key: str = "",
        expires_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize secret value.

        Args:
            value: The secret string.
            provider: Name of the provider that supplied this secret.
            key: The key/path used to retrieve this secret.
            expires_at: When this cached value expires.
            metadata: Additional metadata about the secret.
        """
        self._value = value
        self._hash = hashlib.sha256(value.encode()).hexdigest()[:16]
        self._provider = provider
        self._key = key
        self._created_at = datetime.now()
        self._expires_at = expires_at
        self._metadata = metadata or {}

    def get_value(self) -> str:
        """Get the actual secret value.

        This is the only way to access the underlying secret.
        """
        return self._value

    def expose(self) -> str:
        """Alias for get_value() for explicit intent."""
        return self._value

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return self._provider

    @property
    def key(self) -> str:
        """Get the key used to retrieve this secret."""
        return self._key

    @property
    def created_at(self) -> datetime:
        """Get creation timestamp."""
        return self._created_at

    @property
    def expires_at(self) -> datetime | None:
        """Get expiration timestamp."""
        return self._expires_at

    @property
    def is_expired(self) -> bool:
        """Check if this cached value has expired."""
        if self._expires_at is None:
            return False
        return datetime.now() > self._expires_at

    @property
    def metadata(self) -> dict[str, Any]:
        """Get metadata (read-only copy)."""
        return self._metadata.copy()

    @property
    def hash(self) -> str:
        """Get hash for change detection (not cryptographically secure)."""
        return self._hash

    def __eq__(self, other: object) -> bool:
        """Constant-time comparison to prevent timing attacks."""
        if isinstance(other, SecretValue):
            other_value = other._value
        elif isinstance(other, str):
            other_value = other
        else:
            return NotImplemented

        # Constant-time comparison
        if len(self._value) != len(other_value):
            # Still do comparison to maintain constant time
            _ = sum(a != b for a, b in zip(self._value, other_value))
            return False

        result = 0
        for a, b in zip(self._value, other_value):
            result |= ord(a) ^ ord(b)
        return result == 0

    def __hash__(self) -> int:
        """Hash based on the value hash, not the actual value."""
        return hash(self._hash)

    def __repr__(self) -> str:
        """Safe representation that doesn't expose the value."""
        return f"SecretValue(key={self._key!r}, provider={self._provider!r})"

    def __str__(self) -> str:
        """String representation shows masked value."""
        return "***"

    def __len__(self) -> int:
        """Return length of the secret value."""
        return len(self._value)

    def __bool__(self) -> bool:
        """Check if value is non-empty."""
        return bool(self._value)


# =============================================================================
# Secret Reference
# =============================================================================


@dataclass(frozen=True)
class SecretReference:
    """Reference to a secret that can be resolved later.

    SecretReference represents a pointer to a secret without containing
    the actual value. It supports various reference formats:

    Formats:
        - ${secrets:KEY} - Default provider
        - ${secrets:provider/KEY} - Specific provider
        - ${env:VAR_NAME} - Environment variable
        - ${vault:path/to/secret} - HashiCorp Vault
        - ${aws:secret-name} - AWS Secrets Manager
        - ${azure:vault/secret} - Azure Key Vault
        - ${gcp:project/secret} - GCP Secret Manager
        - ${file:/path/to/file} - Read from file

    Attributes:
        key: The secret key or path.
        provider: The provider to use (None = default).
        version: Specific version to retrieve (optional).
        field: Specific field for structured secrets (optional).
        default: Default value if secret not found (optional).
    """

    key: str
    provider: str | None = None
    version: str | None = None
    field: str | None = None
    default: str | None = None

    # Regex patterns for parsing references
    _PATTERNS = {
        # ${secrets:key} or ${secrets:provider/key}
        "secrets": re.compile(
            r"\$\{secrets:(?:([a-zA-Z0-9_-]+)/)?([^}|]+)(?:\|([^}]*))?\}"
        ),
        # ${env:VAR_NAME}
        "env": re.compile(r"\$\{env:([^}|]+)(?:\|([^}]*))?\}"),
        # ${vault:path/to/secret#field}
        "vault": re.compile(r"\$\{vault:([^}#|]+)(?:#([^}|]+))?(?:\|([^}]*))?\}"),
        # ${aws:secret-name#field}
        "aws": re.compile(r"\$\{aws:([^}#|]+)(?:#([^}|]+))?(?:\|([^}]*))?\}"),
        # ${azure:vault/secret}
        "azure": re.compile(r"\$\{azure:([^}|]+)(?:\|([^}]*))?\}"),
        # ${gcp:project/secret}
        "gcp": re.compile(r"\$\{gcp:([^}|]+)(?:\|([^}]*))?\}"),
        # ${file:/path/to/file}
        "file": re.compile(r"\$\{file:([^}|]+)(?:\|([^}]*))?\}"),
    }

    # Generic pattern to detect any secret reference
    _GENERIC_PATTERN = re.compile(r"\$\{([a-zA-Z_]+):([^}]+)\}")

    @classmethod
    def parse(cls, value: str) -> "SecretReference | None":
        """Parse a string into a SecretReference.

        Args:
            value: String potentially containing a secret reference.

        Returns:
            SecretReference if the string is a valid reference, None otherwise.
        """
        value = value.strip()

        # Try secrets pattern first
        match = cls._PATTERNS["secrets"].fullmatch(value)
        if match:
            provider, key, default = match.groups()
            return cls(key=key, provider=provider, default=default)

        # Try env pattern
        match = cls._PATTERNS["env"].fullmatch(value)
        if match:
            key, default = match.groups()
            return cls(key=key, provider="env", default=default)

        # Try vault pattern
        match = cls._PATTERNS["vault"].fullmatch(value)
        if match:
            key, field, default = match.groups()
            return cls(key=key, provider="vault", field=field, default=default)

        # Try AWS pattern
        match = cls._PATTERNS["aws"].fullmatch(value)
        if match:
            key, field, default = match.groups()
            return cls(key=key, provider="aws", field=field, default=default)

        # Try Azure pattern
        match = cls._PATTERNS["azure"].fullmatch(value)
        if match:
            key, default = match.groups()
            return cls(key=key, provider="azure", default=default)

        # Try GCP pattern
        match = cls._PATTERNS["gcp"].fullmatch(value)
        if match:
            key, default = match.groups()
            return cls(key=key, provider="gcp", default=default)

        # Try file pattern
        match = cls._PATTERNS["file"].fullmatch(value)
        if match:
            key, default = match.groups()
            return cls(key=key, provider="file", default=default)

        return None

    @classmethod
    def is_reference(cls, value: str) -> bool:
        """Check if a string is a secret reference."""
        return cls._GENERIC_PATTERN.search(value) is not None

    @classmethod
    def find_all(cls, text: str) -> list["SecretReference"]:
        """Find all secret references in a text string."""
        refs = []
        for match in cls._GENERIC_PATTERN.finditer(text):
            ref = cls.parse(match.group(0))
            if ref:
                refs.append(ref)
        return refs

    def to_string(self) -> str:
        """Convert back to reference string format."""
        if self.provider:
            base = f"${{secrets:{self.provider}/{self.key}}}"
        else:
            base = f"${{secrets:{self.key}}}"

        if self.default:
            base = base[:-1] + f"|{self.default}}}"

        return base


# =============================================================================
# Secret Provider Protocol
# =============================================================================


@runtime_checkable
class SecretProvider(Protocol):
    """Protocol for secret providers.

    Implementations must provide get() and optionally supports_key().
    This uses Python's Protocol for duck typing, allowing any class
    with the right methods to be used as a provider.

    Example:
        >>> class MyProvider:
        ...     def get(self, key: str) -> SecretValue:
        ...         return SecretValue(fetch_from_my_backend(key))
        ...
        ...     def supports_key(self, key: str) -> bool:
        ...         return key.startswith("my-prefix/")
    """

    @property
    def name(self) -> str:
        """Provider name for identification."""
        ...

    def get(
        self,
        key: str,
        version: str | None = None,
        field: str | None = None,
    ) -> SecretValue:
        """Retrieve a secret by key.

        Args:
            key: Secret key or path.
            version: Specific version to retrieve (optional).
            field: Specific field for structured secrets (optional).

        Returns:
            SecretValue containing the secret.

        Raises:
            SecretNotFoundError: If the secret doesn't exist.
            SecretAccessError: If access is denied.
            SecretProviderError: If the provider encounters an error.
        """
        ...

    def supports_key(self, key: str) -> bool:
        """Check if this provider can handle the given key.

        Default implementation returns True for all keys.
        Override for providers that only handle specific key patterns.
        """
        ...


# =============================================================================
# Base Provider Implementation
# =============================================================================


class BaseSecretProvider(ABC):
    """Abstract base class for secret providers.

    Provides common functionality for secret providers including:
    - Caching with configurable TTL
    - Key validation and normalization
    - Error handling and logging
    - Metrics collection

    Subclasses must implement:
    - _fetch(): Retrieve secret from backend
    - name property: Provider identifier

    Optionally override:
    - _normalize_key(): Customize key format
    - _validate_key(): Add key validation
    - supports_key(): Filter handled keys
    """

    def __init__(
        self,
        *,
        cache_ttl_seconds: int = 300,
        enable_cache: bool = True,
        prefix: str = "",
        audit_callback: Callable[[str, str, bool], None] | None = None,
    ) -> None:
        """Initialize the provider.

        Args:
            cache_ttl_seconds: How long to cache secrets (default: 5 minutes).
            enable_cache: Whether to enable caching.
            prefix: Prefix to add to all keys.
            audit_callback: Callback for audit logging (key, action, success).
        """
        self._cache: dict[str, SecretValue] = {}
        self._cache_ttl = timedelta(seconds=cache_ttl_seconds)
        self._enable_cache = enable_cache
        self._prefix = prefix
        self._audit_callback = audit_callback

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for identification."""
        pass

    @abstractmethod
    def _fetch(
        self,
        key: str,
        version: str | None = None,
        field: str | None = None,
    ) -> str:
        """Fetch secret value from backend.

        Args:
            key: Normalized key.
            version: Specific version.
            field: Specific field.

        Returns:
            The secret value as a string.

        Raises:
            SecretNotFoundError: If not found.
            SecretAccessError: If access denied.
            SecretProviderError: For other errors.
        """
        pass

    def get(
        self,
        key: str,
        version: str | None = None,
        field: str | None = None,
    ) -> SecretValue:
        """Retrieve a secret with caching.

        Args:
            key: Secret key.
            version: Specific version.
            field: Specific field.

        Returns:
            SecretValue containing the secret.
        """
        normalized_key = self._normalize_key(key)
        cache_key = self._cache_key(normalized_key, version, field)

        # Check cache
        if self._enable_cache and cache_key in self._cache:
            cached = self._cache[cache_key]
            if not cached.is_expired:
                self._audit("cache_hit", key, True)
                return cached

        # Fetch from backend
        try:
            value = self._fetch(normalized_key, version, field)
            expires_at = datetime.now() + self._cache_ttl if self._enable_cache else None

            secret = SecretValue(
                value=value,
                provider=self.name,
                key=key,
                expires_at=expires_at,
                metadata={
                    "version": version,
                    "field": field,
                    "normalized_key": normalized_key,
                },
            )

            # Cache the value
            if self._enable_cache:
                self._cache[cache_key] = secret

            self._audit("fetch", key, True)
            return secret

        except SecretError:
            self._audit("fetch", key, False)
            raise
        except Exception as e:
            self._audit("fetch", key, False)
            raise SecretProviderError(self.name, str(e), e) from e

    def supports_key(self, key: str) -> bool:
        """Check if this provider can handle the given key."""
        return True

    def clear_cache(self, key: str | None = None) -> None:
        """Clear cached secrets.

        Args:
            key: Specific key to clear, or None to clear all.
        """
        if key is None:
            self._cache.clear()
        else:
            normalized = self._normalize_key(key)
            keys_to_remove = [
                k for k in self._cache if k.startswith(normalized)
            ]
            for k in keys_to_remove:
                del self._cache[k]

    def _normalize_key(self, key: str) -> str:
        """Normalize the key format."""
        if self._prefix and not key.startswith(self._prefix):
            key = f"{self._prefix}{key}"
        return key

    def _cache_key(
        self,
        key: str,
        version: str | None,
        field: str | None,
    ) -> str:
        """Generate cache key."""
        parts = [key]
        if version:
            parts.append(f"v:{version}")
        if field:
            parts.append(f"f:{field}")
        return ":".join(parts)

    def _audit(self, action: str, key: str, success: bool) -> None:
        """Log audit event."""
        if self._audit_callback:
            try:
                self._audit_callback(key, action, success)
            except Exception:
                pass  # Don't fail on audit errors

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
