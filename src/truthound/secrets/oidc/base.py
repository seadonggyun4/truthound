"""Base classes and protocols for OIDC authentication.

This module defines the core abstractions for OIDC-based authentication,
providing a pluggable architecture for different identity providers and
cloud platforms.

Design Principles:
    1. Protocol-based: Duck typing for flexibility
    2. Token immutability: OIDCToken is immutable after creation
    3. Secure by default: Tokens are redacted in repr/str
    4. Extensible: Easy to add new providers and exchangers
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Protocol,
    TypeVar,
    runtime_checkable,
)

if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class OIDCError(Exception):
    """Base exception for OIDC-related errors."""

    pass


class OIDCTokenError(OIDCError):
    """Raised when there's an error with the OIDC token."""

    def __init__(self, message: str, provider: str | None = None) -> None:
        self.provider = provider
        msg = message
        if provider:
            msg = f"[{provider}] {message}"
        super().__init__(msg)


class OIDCExchangeError(OIDCError):
    """Raised when token exchange fails."""

    def __init__(
        self,
        message: str,
        cloud_provider: str,
        status_code: int | None = None,
        response: str | None = None,
    ) -> None:
        self.cloud_provider = cloud_provider
        self.status_code = status_code
        self.response = response
        msg = f"[{cloud_provider}] Token exchange failed: {message}"
        if status_code:
            msg += f" (status: {status_code})"
        super().__init__(msg)


class OIDCConfigurationError(OIDCError):
    """Raised when OIDC configuration is invalid."""

    def __init__(self, message: str, field: str | None = None) -> None:
        self.field = field
        msg = f"OIDC configuration error: {message}"
        if field:
            msg += f" (field: {field})"
        super().__init__(msg)


class OIDCProviderNotAvailableError(OIDCError):
    """Raised when OIDC provider is not available in the current environment."""

    def __init__(self, provider: str, reason: str) -> None:
        self.provider = provider
        self.reason = reason
        super().__init__(
            f"OIDC provider '{provider}' is not available: {reason}"
        )


# =============================================================================
# Cloud Provider Enum
# =============================================================================


class CloudProvider(str, Enum):
    """Supported cloud providers for token exchange."""

    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    VAULT = "vault"

    def __str__(self) -> str:
        return self.value


class CIProvider(str, Enum):
    """Supported CI/CD providers with OIDC support."""

    GITHUB_ACTIONS = "github_actions"
    GITLAB_CI = "gitlab_ci"
    CIRCLECI = "circleci"
    BITBUCKET = "bitbucket"
    JENKINS = "jenkins"
    UNKNOWN = "unknown"

    def __str__(self) -> str:
        return self.value


# =============================================================================
# OIDC Token and Claims
# =============================================================================


@dataclass(frozen=True)
class OIDCClaims:
    """Parsed OIDC token claims.

    Common claims across different CI providers with optional
    provider-specific extensions.

    Attributes:
        issuer: Token issuer (iss claim).
        subject: Subject identifier (sub claim).
        audience: Token audience (aud claim).
        expiration: Token expiration time (exp claim).
        issued_at: Token issue time (iat claim).
        repository: Repository identifier (if available).
        ref: Git ref (branch/tag) if available.
        sha: Git commit SHA if available.
        actor: User/actor who triggered the workflow.
        workflow: Workflow name if available.
        job: Job name if available.
        run_id: Run/pipeline ID if available.
        environment: Deployment environment if available.
        extra: Additional provider-specific claims.
    """

    issuer: str
    subject: str
    audience: str | list[str]
    expiration: datetime
    issued_at: datetime
    repository: str | None = None
    ref: str | None = None
    sha: str | None = None
    actor: str | None = None
    workflow: str | None = None
    job: str | None = None
    run_id: str | None = None
    environment: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_jwt_payload(cls, payload: dict[str, Any]) -> "OIDCClaims":
        """Parse claims from JWT payload.

        Args:
            payload: Decoded JWT payload.

        Returns:
            OIDCClaims instance.
        """
        # Required claims
        issuer = payload.get("iss", "")
        subject = payload.get("sub", "")
        audience = payload.get("aud", "")

        # Time claims
        exp = payload.get("exp", 0)
        iat = payload.get("iat", 0)
        expiration = datetime.fromtimestamp(exp) if exp else datetime.now()
        issued_at = datetime.fromtimestamp(iat) if iat else datetime.now()

        # Common optional claims (varies by provider)
        repository = payload.get("repository") or payload.get("project_path")
        ref = payload.get("ref") or payload.get("ref_path")
        sha = payload.get("sha") or payload.get("commit_sha")
        actor = payload.get("actor") or payload.get("user_login")
        workflow = payload.get("workflow") or payload.get("pipeline_source")
        job = payload.get("job") or payload.get("job_id")
        run_id = payload.get("run_id") or payload.get("pipeline_id")
        environment = payload.get("environment") or payload.get("environment_scope")

        # Collect extra claims
        known_claims = {
            "iss", "sub", "aud", "exp", "iat", "nbf", "jti",
            "repository", "project_path", "ref", "ref_path",
            "sha", "commit_sha", "actor", "user_login",
            "workflow", "pipeline_source", "job", "job_id",
            "run_id", "pipeline_id", "environment", "environment_scope",
        }
        extra = {k: v for k, v in payload.items() if k not in known_claims}

        return cls(
            issuer=issuer,
            subject=subject,
            audience=audience,
            expiration=expiration,
            issued_at=issued_at,
            repository=repository,
            ref=ref,
            sha=sha,
            actor=actor,
            workflow=workflow,
            job=job,
            run_id=run_id,
            environment=environment,
            extra=extra,
        )

    @property
    def is_expired(self) -> bool:
        """Check if the token is expired."""
        return datetime.now() >= self.expiration

    @property
    def time_until_expiry(self) -> timedelta:
        """Get time remaining until expiration."""
        return self.expiration - datetime.now()

    def get_audience_list(self) -> list[str]:
        """Get audience as a list."""
        if isinstance(self.audience, list):
            return self.audience
        return [self.audience] if self.audience else []


class OIDCToken:
    """Container for OIDC JWT token.

    Provides secure handling of the raw JWT token with:
    - Lazy claim parsing
    - Token not exposed in repr/str
    - Expiration checking
    - Hash for change detection

    Example:
        >>> token = OIDCToken(jwt_string, provider="github")
        >>> print(token)  # "OIDCToken(provider=github, expires_in=...)"
        >>> token.get_token()  # Returns raw JWT
        >>> token.claims.subject  # Parsed claims
    """

    __slots__ = (
        "_token",
        "_hash",
        "_provider",
        "_claims",
        "_created_at",
    )

    def __init__(
        self,
        token: str,
        *,
        provider: str = "unknown",
    ) -> None:
        """Initialize OIDC token.

        Args:
            token: Raw JWT token string.
            provider: Identity provider name.
        """
        self._token = token
        self._hash = hashlib.sha256(token.encode()).hexdigest()[:16]
        self._provider = provider
        self._claims: OIDCClaims | None = None
        self._created_at = datetime.now()

    def get_token(self) -> str:
        """Get the raw JWT token.

        This is the only way to access the underlying token.
        """
        return self._token

    @property
    def provider(self) -> str:
        """Get the identity provider name."""
        return self._provider

    @property
    def claims(self) -> OIDCClaims:
        """Get parsed token claims (lazy parsing)."""
        if self._claims is None:
            self._claims = self._parse_claims()
        return self._claims

    @property
    def is_expired(self) -> bool:
        """Check if the token is expired."""
        return self.claims.is_expired

    @property
    def hash(self) -> str:
        """Get hash for change detection."""
        return self._hash

    def _parse_claims(self) -> OIDCClaims:
        """Parse JWT claims without verification.

        Note: This only decodes the payload, it does NOT verify the signature.
        Signature verification is done by the token exchanger (e.g., AWS STS).
        """
        try:
            parts = self._token.split(".")
            if len(parts) != 3:
                raise OIDCTokenError(
                    "Invalid JWT format", provider=self._provider
                )

            # Decode payload (add padding if needed)
            payload_b64 = parts[1]
            padding = 4 - len(payload_b64) % 4
            if padding != 4:
                payload_b64 += "=" * padding

            payload_json = base64.urlsafe_b64decode(payload_b64)
            payload = json.loads(payload_json)

            return OIDCClaims.from_jwt_payload(payload)

        except json.JSONDecodeError as e:
            raise OIDCTokenError(
                f"Invalid JWT payload: {e}", provider=self._provider
            ) from e
        except Exception as e:
            if isinstance(e, OIDCTokenError):
                raise
            raise OIDCTokenError(
                f"Failed to parse JWT: {e}", provider=self._provider
            ) from e

    def __repr__(self) -> str:
        """Safe representation that doesn't expose the token."""
        try:
            expires_in = self.claims.time_until_expiry
            expires_str = f"{expires_in.total_seconds():.0f}s"
        except Exception:
            expires_str = "unknown"

        return (
            f"OIDCToken(provider={self._provider!r}, "
            f"expires_in={expires_str})"
        )

    def __str__(self) -> str:
        """String representation shows masked token."""
        return f"OIDCToken(***)"

    def __bool__(self) -> bool:
        """Check if token is valid and not expired."""
        return bool(self._token) and not self.is_expired


# =============================================================================
# Cloud Credentials
# =============================================================================


@dataclass
class CloudCredentials:
    """Base class for cloud provider credentials.

    Attributes:
        provider: Cloud provider name.
        expires_at: When credentials expire.
        metadata: Additional metadata.
    """

    provider: CloudProvider = field(default=CloudProvider.AWS)
    expires_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Check if credentials are expired."""
        if self.expires_at is None:
            return False
        return datetime.now() >= self.expires_at

    @property
    def time_until_expiry(self) -> timedelta | None:
        """Get time remaining until expiration."""
        if self.expires_at is None:
            return None
        return self.expires_at - datetime.now()


@dataclass
class AWSCredentials(CloudCredentials):
    """AWS STS credentials from OIDC token exchange.

    Attributes:
        access_key_id: AWS access key ID.
        secret_access_key: AWS secret access key.
        session_token: AWS session token.
        assumed_role_arn: ARN of the assumed role.
    """

    access_key_id: str = ""
    secret_access_key: str = ""
    session_token: str = ""
    assumed_role_arn: str = ""

    def __post_init__(self) -> None:
        self.provider = CloudProvider.AWS

    def to_boto3_session_credentials(self) -> dict[str, str]:
        """Convert to boto3 session credentials format."""
        return {
            "aws_access_key_id": self.access_key_id,
            "aws_secret_access_key": self.secret_access_key,
            "aws_session_token": self.session_token,
        }

    def to_environment_variables(self) -> dict[str, str]:
        """Convert to environment variable format."""
        return {
            "AWS_ACCESS_KEY_ID": self.access_key_id,
            "AWS_SECRET_ACCESS_KEY": self.secret_access_key,
            "AWS_SESSION_TOKEN": self.session_token,
        }

    def __repr__(self) -> str:
        """Safe representation."""
        return (
            f"AWSCredentials(access_key_id={self.access_key_id[:8]}..., "
            f"role={self.assumed_role_arn})"
        )


@dataclass
class GCPCredentials(CloudCredentials):
    """GCP Workload Identity credentials from OIDC token exchange.

    Attributes:
        access_token: OAuth2 access token.
        token_type: Token type (usually "Bearer").
        service_account: Service account email.
        project_id: GCP project ID.
    """

    access_token: str = ""
    token_type: str = "Bearer"
    service_account: str = ""
    project_id: str = ""

    def __post_init__(self) -> None:
        self.provider = CloudProvider.GCP

    def to_google_credentials(self) -> dict[str, Any]:
        """Convert to google-auth credentials format."""
        return {
            "token": self.access_token,
            "expiry": self.expires_at,
        }

    def to_environment_variables(self) -> dict[str, str]:
        """Convert to environment variable format."""
        return {
            "GOOGLE_OAUTH_ACCESS_TOKEN": self.access_token,
        }

    def get_authorization_header(self) -> dict[str, str]:
        """Get HTTP authorization header."""
        return {"Authorization": f"{self.token_type} {self.access_token}"}

    def __repr__(self) -> str:
        """Safe representation."""
        return (
            f"GCPCredentials(service_account={self.service_account}, "
            f"project={self.project_id})"
        )


@dataclass
class AzureCredentials(CloudCredentials):
    """Azure federated credentials from OIDC token exchange.

    Attributes:
        access_token: OAuth2 access token.
        token_type: Token type (usually "Bearer").
        tenant_id: Azure tenant ID.
        client_id: Azure client/app ID.
        subscription_id: Azure subscription ID.
    """

    access_token: str = ""
    token_type: str = "Bearer"
    tenant_id: str = ""
    client_id: str = ""
    subscription_id: str = ""

    def __post_init__(self) -> None:
        self.provider = CloudProvider.AZURE

    def to_environment_variables(self) -> dict[str, str]:
        """Convert to environment variable format."""
        return {
            "AZURE_ACCESS_TOKEN": self.access_token,
            "AZURE_TENANT_ID": self.tenant_id,
            "AZURE_CLIENT_ID": self.client_id,
            "AZURE_SUBSCRIPTION_ID": self.subscription_id,
        }

    def get_authorization_header(self) -> dict[str, str]:
        """Get HTTP authorization header."""
        return {"Authorization": f"{self.token_type} {self.access_token}"}

    def __repr__(self) -> str:
        """Safe representation."""
        return (
            f"AzureCredentials(tenant={self.tenant_id}, "
            f"client={self.client_id})"
        )


# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class OIDCProvider(Protocol):
    """Protocol for OIDC identity providers.

    Implementations must provide get_token() to retrieve the OIDC JWT token
    from the CI environment.

    Example:
        >>> class MyOIDCProvider:
        ...     @property
        ...     def name(self) -> str:
        ...         return "my-ci"
        ...
        ...     def get_token(self, audience: str) -> OIDCToken:
        ...         jwt = fetch_from_my_ci_environment(audience)
        ...         return OIDCToken(jwt, provider=self.name)
        ...
        ...     def is_available(self) -> bool:
        ...         return "MY_CI_TOKEN_URL" in os.environ
    """

    @property
    def name(self) -> str:
        """Provider name for identification."""
        ...

    def get_token(self, audience: str | None = None) -> OIDCToken:
        """Retrieve OIDC token for the given audience.

        Args:
            audience: Token audience (required by some providers).

        Returns:
            OIDCToken containing the JWT.

        Raises:
            OIDCTokenError: If token retrieval fails.
            OIDCProviderNotAvailableError: If provider is not available.
        """
        ...

    def is_available(self) -> bool:
        """Check if this provider is available in the current environment."""
        ...


@runtime_checkable
class TokenExchanger(Protocol):
    """Protocol for cloud token exchangers.

    Implementations exchange an OIDC token for cloud provider credentials.

    Example:
        >>> class MyExchanger:
        ...     @property
        ...     def cloud_provider(self) -> CloudProvider:
        ...         return CloudProvider.AWS
        ...
        ...     def exchange(self, token: OIDCToken) -> CloudCredentials:
        ...         creds = call_sts_api(token.get_token())
        ...         return AWSCredentials(...)
    """

    @property
    def cloud_provider(self) -> CloudProvider:
        """Cloud provider this exchanger targets."""
        ...

    def exchange(self, token: OIDCToken) -> CloudCredentials:
        """Exchange OIDC token for cloud credentials.

        Args:
            token: OIDC token from CI provider.

        Returns:
            Cloud provider credentials.

        Raises:
            OIDCExchangeError: If exchange fails.
        """
        ...


# =============================================================================
# Base Implementations
# =============================================================================


class BaseOIDCProvider(ABC):
    """Abstract base class for OIDC providers.

    Provides common functionality for OIDC identity providers:
    - Token caching with configurable TTL
    - Retry logic for token requests
    - Logging and metrics

    Subclasses must implement:
    - _fetch_token(): Retrieve token from the CI environment
    - name property: Provider identifier
    - is_available(): Check if provider is usable
    """

    def __init__(
        self,
        *,
        cache_ttl_seconds: int = 300,
        enable_cache: bool = True,
        retry_attempts: int = 3,
        retry_delay_seconds: float = 1.0,
    ) -> None:
        """Initialize the provider.

        Args:
            cache_ttl_seconds: How long to cache tokens.
            enable_cache: Whether to enable caching.
            retry_attempts: Number of retry attempts for token fetch.
            retry_delay_seconds: Delay between retries.
        """
        self._cache: dict[str, OIDCToken] = {}
        self._cache_ttl = timedelta(seconds=cache_ttl_seconds)
        self._enable_cache = enable_cache
        self._retry_attempts = retry_attempts
        self._retry_delay = retry_delay_seconds

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for identification."""
        pass

    @abstractmethod
    def _fetch_token(self, audience: str | None = None) -> str:
        """Fetch token from the CI environment.

        Args:
            audience: Token audience.

        Returns:
            Raw JWT token string.

        Raises:
            OIDCTokenError: If token fetch fails.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is available in the current environment."""
        pass

    def get_token(self, audience: str | None = None) -> OIDCToken:
        """Get OIDC token with caching.

        Args:
            audience: Token audience.

        Returns:
            OIDCToken containing the JWT.
        """
        if not self.is_available():
            raise OIDCProviderNotAvailableError(
                self.name,
                "Required environment variables not found",
            )

        cache_key = audience or "_default_"

        # Check cache
        if self._enable_cache and cache_key in self._cache:
            cached = self._cache[cache_key]
            # Use token if not expired and has >30s remaining
            if not cached.is_expired:
                remaining = cached.claims.time_until_expiry
                if remaining.total_seconds() > 30:
                    logger.debug(
                        f"Using cached OIDC token for {self.name} "
                        f"(expires in {remaining.total_seconds():.0f}s)"
                    )
                    return cached

        # Fetch new token with retry
        import time
        last_error: Exception | None = None

        for attempt in range(self._retry_attempts):
            try:
                jwt = self._fetch_token(audience)
                token = OIDCToken(jwt, provider=self.name)

                # Cache the token
                if self._enable_cache:
                    self._cache[cache_key] = token

                logger.debug(
                    f"Fetched new OIDC token from {self.name} "
                    f"(expires in {token.claims.time_until_expiry.total_seconds():.0f}s)"
                )
                return token

            except Exception as e:
                last_error = e
                if attempt < self._retry_attempts - 1:
                    time.sleep(self._retry_delay * (2 ** attempt))
                    logger.warning(
                        f"Retry {attempt + 1}/{self._retry_attempts} "
                        f"for OIDC token from {self.name}: {e}"
                    )

        if last_error:
            if isinstance(last_error, OIDCError):
                raise last_error
            raise OIDCTokenError(str(last_error), provider=self.name) from last_error
        raise OIDCTokenError("Token fetch failed", provider=self.name)

    def clear_cache(self, audience: str | None = None) -> None:
        """Clear cached tokens.

        Args:
            audience: Specific audience to clear, or None to clear all.
        """
        if audience is None:
            self._cache.clear()
        else:
            self._cache.pop(audience, None)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


class BaseTokenExchanger(ABC):
    """Abstract base class for token exchangers.

    Provides common functionality for cloud token exchange:
    - Credential caching with configurable TTL
    - Retry logic for exchange requests
    - Logging and metrics

    Subclasses must implement:
    - _exchange(): Perform the actual token exchange
    - cloud_provider property: Target cloud provider
    """

    def __init__(
        self,
        *,
        cache_ttl_seconds: int = 3600,
        enable_cache: bool = True,
        retry_attempts: int = 3,
        retry_delay_seconds: float = 1.0,
    ) -> None:
        """Initialize the exchanger.

        Args:
            cache_ttl_seconds: How long to cache credentials.
            enable_cache: Whether to enable caching.
            retry_attempts: Number of retry attempts.
            retry_delay_seconds: Delay between retries.
        """
        self._cache: dict[str, CloudCredentials] = {}
        self._cache_ttl = timedelta(seconds=cache_ttl_seconds)
        self._enable_cache = enable_cache
        self._retry_attempts = retry_attempts
        self._retry_delay = retry_delay_seconds

    @property
    @abstractmethod
    def cloud_provider(self) -> CloudProvider:
        """Cloud provider this exchanger targets."""
        pass

    @abstractmethod
    def _exchange(self, token: OIDCToken) -> CloudCredentials:
        """Perform the actual token exchange.

        Args:
            token: OIDC token to exchange.

        Returns:
            Cloud provider credentials.

        Raises:
            OIDCExchangeError: If exchange fails.
        """
        pass

    def exchange(self, token: OIDCToken) -> CloudCredentials:
        """Exchange OIDC token for cloud credentials with caching.

        Args:
            token: OIDC token from CI provider.

        Returns:
            Cloud provider credentials.
        """
        cache_key = token.hash

        # Check cache
        if self._enable_cache and cache_key in self._cache:
            cached = self._cache[cache_key]
            # Use credentials if not expired and has >60s remaining
            if not cached.is_expired:
                remaining = cached.time_until_expiry
                if remaining and remaining.total_seconds() > 60:
                    logger.debug(
                        f"Using cached {self.cloud_provider} credentials "
                        f"(expires in {remaining.total_seconds():.0f}s)"
                    )
                    return cached

        # Exchange token with retry
        import time
        last_error: Exception | None = None

        for attempt in range(self._retry_attempts):
            try:
                credentials = self._exchange(token)

                # Cache the credentials
                if self._enable_cache:
                    self._cache[cache_key] = credentials

                logger.debug(
                    f"Exchanged OIDC token for {self.cloud_provider} credentials"
                )
                return credentials

            except Exception as e:
                last_error = e
                if attempt < self._retry_attempts - 1:
                    time.sleep(self._retry_delay * (2 ** attempt))
                    logger.warning(
                        f"Retry {attempt + 1}/{self._retry_attempts} "
                        f"for {self.cloud_provider} exchange: {e}"
                    )

        if last_error:
            if isinstance(last_error, OIDCError):
                raise last_error
            raise OIDCExchangeError(
                str(last_error),
                cloud_provider=str(self.cloud_provider),
            ) from last_error
        raise OIDCExchangeError(
            "Exchange failed",
            cloud_provider=str(self.cloud_provider),
        )

    def clear_cache(self) -> None:
        """Clear cached credentials."""
        self._cache.clear()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(provider={self.cloud_provider!r})"
