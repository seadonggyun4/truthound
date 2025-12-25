"""Tenant resolvers for multi-tenancy.

This module provides various strategies for resolving the current tenant
from request context, supporting different deployment patterns.

Resolution Strategies:
    - Header-based: X-Tenant-ID header
    - Subdomain-based: acme.truthound.io
    - Path-based: /tenants/acme/...
    - API key-based: API key includes tenant info
    - JWT claim-based: Tenant ID in JWT token
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import time
from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from truthound.multitenancy.core import (
    TenantResolver,
    TenantError,
    TenantAccessDeniedError,
)


# =============================================================================
# Header-Based Resolution
# =============================================================================


@dataclass
class HeaderResolverConfig:
    """Configuration for header-based tenant resolution."""

    header_name: str = "X-Tenant-ID"
    header_alternatives: list[str] = field(
        default_factory=lambda: ["X-Tenant", "Tenant-ID"]
    )
    required: bool = False
    validate_format: bool = True
    allowed_pattern: str = r"^[a-zA-Z0-9_-]+$"


class HeaderResolver(TenantResolver):
    """Resolve tenant from HTTP header.

    The most common approach for API-based multi-tenancy.

    Example:
        >>> resolver = HeaderResolver(config=HeaderResolverConfig(
        ...     header_name="X-Tenant-ID",
        ... ))
        >>> tenant_id = resolver.resolve({"headers": {"X-Tenant-ID": "acme"}})
    """

    def __init__(self, config: HeaderResolverConfig | None = None) -> None:
        self._config = config or HeaderResolverConfig()
        self._pattern = re.compile(self._config.allowed_pattern)

    @property
    def name(self) -> str:
        return "header"

    def resolve(self, context: Mapping[str, Any]) -> str | None:
        """Resolve tenant ID from headers."""
        headers = context.get("headers", {})

        # Normalize header names to lowercase for case-insensitive matching
        normalized_headers = {k.lower(): v for k, v in headers.items()}

        # Try primary header
        tenant_id = normalized_headers.get(self._config.header_name.lower())

        # Try alternatives
        if not tenant_id:
            for alt in self._config.header_alternatives:
                tenant_id = normalized_headers.get(alt.lower())
                if tenant_id:
                    break

        if not tenant_id:
            return None

        # Validate format
        if self._config.validate_format:
            if not self._pattern.match(tenant_id):
                raise TenantAccessDeniedError(
                    f"Invalid tenant ID format: {tenant_id}",
                    tenant_id=tenant_id,
                )

        return tenant_id


# =============================================================================
# Subdomain-Based Resolution
# =============================================================================


@dataclass
class SubdomainResolverConfig:
    """Configuration for subdomain-based tenant resolution."""

    base_domain: str = "truthound.io"
    subdomain_position: int = 0  # 0 = first subdomain (acme.truthound.io)
    exclude_subdomains: list[str] = field(
        default_factory=lambda: ["www", "api", "app", "admin"]
    )
    validate_format: bool = True
    allowed_pattern: str = r"^[a-z0-9-]+$"


class SubdomainResolver(TenantResolver):
    """Resolve tenant from subdomain.

    Common for SaaS applications where each tenant has their own subdomain.

    Example:
        >>> resolver = SubdomainResolver(config=SubdomainResolverConfig(
        ...     base_domain="truthound.io",
        ... ))
        >>> tenant_id = resolver.resolve({"host": "acme.truthound.io"})
        >>> # Returns: "acme"
    """

    def __init__(self, config: SubdomainResolverConfig | None = None) -> None:
        self._config = config or SubdomainResolverConfig()
        self._pattern = re.compile(self._config.allowed_pattern)

    @property
    def name(self) -> str:
        return "subdomain"

    def resolve(self, context: Mapping[str, Any]) -> str | None:
        """Resolve tenant ID from subdomain."""
        host = context.get("host", "")
        if not host:
            headers = context.get("headers", {})
            host = headers.get("Host", headers.get("host", ""))

        if not host:
            return None

        # Remove port if present
        host = host.split(":")[0]

        # Check if it's the base domain
        if not host.endswith(f".{self._config.base_domain}"):
            if host == self._config.base_domain:
                return None
            return None

        # Extract subdomain
        subdomain_part = host[: -len(self._config.base_domain) - 1]
        subdomains = subdomain_part.split(".")

        if len(subdomains) <= self._config.subdomain_position:
            return None

        tenant_slug = subdomains[self._config.subdomain_position]

        # Check exclusions
        if tenant_slug in self._config.exclude_subdomains:
            return None

        # Validate format
        if self._config.validate_format:
            if not self._pattern.match(tenant_slug):
                raise TenantAccessDeniedError(
                    f"Invalid tenant subdomain: {tenant_slug}",
                    tenant_id=tenant_slug,
                )

        return tenant_slug


# =============================================================================
# Path-Based Resolution
# =============================================================================


@dataclass
class PathResolverConfig:
    """Configuration for path-based tenant resolution."""

    path_prefix: str = "/tenants/"
    path_pattern: str = r"^/tenants/([a-zA-Z0-9_-]+)"
    tenant_position: int = 1  # Capture group position
    strip_prefix: bool = True


class PathResolver(TenantResolver):
    """Resolve tenant from URL path.

    Useful for APIs where tenant is part of the URL structure.

    Example:
        >>> resolver = PathResolver(config=PathResolverConfig(
        ...     path_prefix="/tenants/",
        ... ))
        >>> tenant_id = resolver.resolve({"path": "/tenants/acme/data"})
        >>> # Returns: "acme"
    """

    def __init__(self, config: PathResolverConfig | None = None) -> None:
        self._config = config or PathResolverConfig()
        self._pattern = re.compile(self._config.path_pattern)

    @property
    def name(self) -> str:
        return "path"

    def resolve(self, context: Mapping[str, Any]) -> str | None:
        """Resolve tenant ID from path."""
        path = context.get("path", "")
        if not path:
            return None

        match = self._pattern.match(path)
        if not match:
            return None

        try:
            return match.group(self._config.tenant_position)
        except IndexError:
            return None

    def strip_tenant_from_path(self, path: str, tenant_id: str) -> str:
        """Remove tenant prefix from path for routing."""
        if self._config.strip_prefix:
            prefix = f"{self._config.path_prefix}{tenant_id}"
            if path.startswith(prefix):
                return path[len(prefix) :] or "/"
        return path


# =============================================================================
# API Key-Based Resolution
# =============================================================================


@dataclass
class APIKeyResolverConfig:
    """Configuration for API key-based tenant resolution."""

    header_name: str = "X-API-Key"
    header_alternatives: list[str] = field(
        default_factory=lambda: ["Authorization", "Api-Key"]
    )
    prefix: str = "Bearer "  # For Authorization header
    key_format: str = "prefixed"  # "prefixed" (tenant_xxx_key) or "lookup"
    tenant_separator: str = "_"
    secret_key: str = ""  # For HMAC validation
    validate_signature: bool = False


class APIKeyResolver(TenantResolver):
    """Resolve tenant from API key.

    API keys can encode tenant information directly or require lookup.

    Key Formats:
        - Prefixed: tenant_acme_abc123... (tenant ID is embedded)
        - Lookup: abc123... (requires database lookup)

    Example:
        >>> resolver = APIKeyResolver(config=APIKeyResolverConfig(
        ...     key_format="prefixed",
        ... ))
        >>> tenant_id = resolver.resolve({
        ...     "headers": {"X-API-Key": "tenant_acme_abc123def456"}
        ... })
        >>> # Returns: "acme"
    """

    def __init__(
        self,
        config: APIKeyResolverConfig | None = None,
        key_lookup: Callable[[str], str | None] | None = None,
    ) -> None:
        self._config = config or APIKeyResolverConfig()
        self._key_lookup = key_lookup

    @property
    def name(self) -> str:
        return "api_key"

    def resolve(self, context: Mapping[str, Any]) -> str | None:
        """Resolve tenant ID from API key."""
        headers = context.get("headers", {})
        normalized_headers = {k.lower(): v for k, v in headers.items()}

        # Get API key from headers
        api_key = normalized_headers.get(self._config.header_name.lower())

        if not api_key:
            for alt in self._config.header_alternatives:
                api_key = normalized_headers.get(alt.lower())
                if api_key:
                    break

        if not api_key:
            return None

        # Remove prefix if present (e.g., "Bearer ")
        if self._config.prefix and api_key.startswith(self._config.prefix):
            api_key = api_key[len(self._config.prefix) :]

        # Extract tenant based on format
        if self._config.key_format == "prefixed":
            return self._extract_from_prefixed_key(api_key)
        elif self._config.key_format == "lookup" and self._key_lookup:
            return self._key_lookup(api_key)

        return None

    def _extract_from_prefixed_key(self, api_key: str) -> str | None:
        """Extract tenant ID from a prefixed API key."""
        # Format: tenant_{tenant_id}_{random_part}
        parts = api_key.split(self._config.tenant_separator)
        if len(parts) >= 3 and parts[0] == "tenant":
            return parts[1]
        return None


# =============================================================================
# JWT-Based Resolution
# =============================================================================


@dataclass
class JWTResolverConfig:
    """Configuration for JWT-based tenant resolution."""

    header_name: str = "Authorization"
    prefix: str = "Bearer "
    tenant_claim: str = "tenant_id"
    tenant_claim_alternatives: list[str] = field(
        default_factory=lambda: ["tenantId", "tid", "org_id"]
    )
    verify_signature: bool = True
    secret_key: str = ""
    algorithms: list[str] = field(default_factory=lambda: ["HS256"])


class JWTResolver(TenantResolver):
    """Resolve tenant from JWT token claims.

    Extracts tenant ID from a claim in the JWT token.

    Example:
        >>> resolver = JWTResolver(config=JWTResolverConfig(
        ...     tenant_claim="tenant_id",
        ...     verify_signature=False,  # For demo only
        ... ))
        >>> tenant_id = resolver.resolve({
        ...     "headers": {"Authorization": "Bearer eyJ..."}
        ... })
    """

    def __init__(self, config: JWTResolverConfig | None = None) -> None:
        self._config = config or JWTResolverConfig()

    @property
    def name(self) -> str:
        return "jwt"

    def resolve(self, context: Mapping[str, Any]) -> str | None:
        """Resolve tenant ID from JWT claims."""
        headers = context.get("headers", {})
        normalized_headers = {k.lower(): v for k, v in headers.items()}

        auth_header = normalized_headers.get(self._config.header_name.lower())
        if not auth_header:
            return None

        # Remove Bearer prefix
        if auth_header.startswith(self._config.prefix):
            token = auth_header[len(self._config.prefix) :]
        else:
            token = auth_header

        # Decode JWT
        try:
            claims = self._decode_jwt(token)
        except Exception:
            return None

        # Extract tenant claim
        tenant_id = claims.get(self._config.tenant_claim)
        if tenant_id:
            return str(tenant_id)

        # Try alternatives
        for alt in self._config.tenant_claim_alternatives:
            tenant_id = claims.get(alt)
            if tenant_id:
                return str(tenant_id)

        return None

    def _decode_jwt(self, token: str) -> dict[str, Any]:
        """Decode JWT token.

        This is a simplified implementation. Production code should
        use a proper JWT library like PyJWT.
        """
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("Invalid JWT format")

        # Decode payload (second part)
        payload = parts[1]
        # Add padding if needed
        padding = 4 - len(payload) % 4
        if padding != 4:
            payload += "=" * padding

        decoded = base64.urlsafe_b64decode(payload)
        return json.loads(decoded)


# =============================================================================
# Composite Resolver (Chain of Responsibility)
# =============================================================================


class CompositeResolver(TenantResolver):
    """Composite resolver that tries multiple strategies.

    Follows the Chain of Responsibility pattern to try resolvers
    in order until one succeeds.

    Example:
        >>> resolver = CompositeResolver([
        ...     HeaderResolver(),
        ...     SubdomainResolver(),
        ...     PathResolver(),
        ... ])
        >>> tenant_id = resolver.resolve(context)
    """

    def __init__(
        self,
        resolvers: list[TenantResolver],
        require_resolution: bool = False,
    ) -> None:
        if not resolvers:
            raise ValueError("At least one resolver is required")
        self._resolvers = resolvers
        self._require_resolution = require_resolution

    @property
    def name(self) -> str:
        return "composite"

    def resolve(self, context: Mapping[str, Any]) -> str | None:
        """Try each resolver in order."""
        for resolver in self._resolvers:
            try:
                tenant_id = resolver.resolve(context)
                if tenant_id:
                    return tenant_id
            except TenantError:
                # Re-raise tenant errors
                raise
            except Exception:
                # Skip failed resolvers
                continue

        if self._require_resolution:
            raise TenantAccessDeniedError("Could not resolve tenant from context")

        return None

    def add_resolver(self, resolver: TenantResolver, priority: int = -1) -> None:
        """Add a resolver at the specified priority."""
        if priority < 0 or priority >= len(self._resolvers):
            self._resolvers.append(resolver)
        else:
            self._resolvers.insert(priority, resolver)


# =============================================================================
# Context-Based Resolution
# =============================================================================


class ContextResolver(TenantResolver):
    """Resolve tenant from thread-local or async context.

    Useful when tenant has already been resolved earlier in the
    request lifecycle.

    Example:
        >>> resolver = ContextResolver()
        >>> # Tenant already set in TenantContext
        >>> tenant_id = resolver.resolve({})
    """

    def __init__(self, fallback: TenantResolver | None = None) -> None:
        self._fallback = fallback

    @property
    def name(self) -> str:
        return "context"

    def resolve(self, context: Mapping[str, Any]) -> str | None:
        """Resolve tenant from context variable."""
        from truthound.multitenancy.core import TenantContext

        tenant_id = TenantContext.get_current_tenant_id()
        if tenant_id:
            return tenant_id

        if self._fallback:
            return self._fallback.resolve(context)

        return None


# =============================================================================
# Callable Resolver
# =============================================================================


class CallableResolver(TenantResolver):
    """Resolver that uses a custom callable.

    Provides maximum flexibility for custom resolution logic.

    Example:
        >>> resolver = CallableResolver(
        ...     resolver_func=lambda ctx: ctx.get("custom_tenant"),
        ...     name="custom",
        ... )
    """

    def __init__(
        self,
        resolver_func: Callable[[Mapping[str, Any]], str | None],
        name: str = "callable",
    ) -> None:
        self._resolver_func = resolver_func
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def resolve(self, context: Mapping[str, Any]) -> str | None:
        """Resolve using the custom callable."""
        return self._resolver_func(context)


# =============================================================================
# Factory Function
# =============================================================================


def create_resolver(
    strategy: str,
    **kwargs: Any,
) -> TenantResolver:
    """Create a tenant resolver.

    Args:
        strategy: Resolution strategy ("header", "subdomain", "path", "jwt", "api_key")
        **kwargs: Strategy-specific configuration

    Returns:
        Configured TenantResolver instance.

    Example:
        >>> resolver = create_resolver("header", header_name="X-Tenant")
        >>> resolver = create_resolver("subdomain", base_domain="myapp.io")
    """
    if strategy == "header":
        config = HeaderResolverConfig(**kwargs)
        return HeaderResolver(config=config)
    elif strategy == "subdomain":
        config = SubdomainResolverConfig(**kwargs)
        return SubdomainResolver(config=config)
    elif strategy == "path":
        config = PathResolverConfig(**kwargs)
        return PathResolver(config=config)
    elif strategy == "jwt":
        config = JWTResolverConfig(**kwargs)
        return JWTResolver(config=config)
    elif strategy == "api_key":
        config = APIKeyResolverConfig(**kwargs)
        return APIKeyResolver(config=config)
    elif strategy == "context":
        return ContextResolver()
    else:
        raise ValueError(f"Unknown resolver strategy: {strategy}")
