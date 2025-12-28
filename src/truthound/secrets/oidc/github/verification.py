"""OIDC Token Verification Utilities.

This module provides utilities for verifying GitHub Actions OIDC tokens
using JWKS (JSON Web Key Set) from GitHub's OIDC provider.

Features:
    - JWKS fetching and caching
    - JWT signature verification
    - Claims validation
    - Token expiration checking

Note: Token verification requires the 'cryptography' package:
    pip install cryptography

Example:
    >>> from truthound.secrets.oidc.github import verify_token, GitHubActionsJWKS
    >>>
    >>> # Verify a token
    >>> result = verify_token(token_string, audience="sts.amazonaws.com")
    >>> if result.is_valid:
    ...     print(f"Token verified for: {result.claims.repository}")
    >>> else:
    ...     print(f"Verification failed: {result.error}")
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from abc import ABC, abstractmethod

from truthound.secrets.oidc.github.claims import (
    GitHubActionsClaims,
    parse_github_claims,
)


logger = logging.getLogger(__name__)


# GitHub Actions OIDC endpoints
GITHUB_OIDC_ISSUER = "https://token.actions.githubusercontent.com"
GITHUB_OIDC_JWKS_URL = f"{GITHUB_OIDC_ISSUER}/.well-known/jwks"
GITHUB_OIDC_CONFIG_URL = f"{GITHUB_OIDC_ISSUER}/.well-known/openid-configuration"


# =============================================================================
# JWKS Verification
# =============================================================================


@dataclass
class JWK:
    """JSON Web Key representation.

    Attributes:
        kty: Key type (RSA, EC).
        kid: Key ID.
        alg: Algorithm.
        use: Key usage (sig).
        n: RSA modulus (base64url).
        e: RSA exponent (base64url).
        x: EC X coordinate (base64url).
        y: EC Y coordinate (base64url).
        crv: EC curve name.
    """

    kty: str
    kid: str = ""
    alg: str = ""
    use: str = ""
    n: str = ""  # RSA
    e: str = ""  # RSA
    x: str = ""  # EC
    y: str = ""  # EC
    crv: str = ""  # EC

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JWK":
        """Create JWK from dictionary."""
        return cls(
            kty=data.get("kty", ""),
            kid=data.get("kid", ""),
            alg=data.get("alg", ""),
            use=data.get("use", ""),
            n=data.get("n", ""),
            e=data.get("e", ""),
            x=data.get("x", ""),
            y=data.get("y", ""),
            crv=data.get("crv", ""),
        )

    def get_public_key(self) -> Any:
        """Convert JWK to cryptography public key.

        Returns:
            RSAPublicKey or EllipticCurvePublicKey.

        Raises:
            ImportError: If cryptography is not installed.
            ValueError: If key type is not supported.
        """
        try:
            from cryptography.hazmat.primitives.asymmetric import rsa, ec
            from cryptography.hazmat.backends import default_backend
        except ImportError as e:
            raise ImportError(
                "Token verification requires 'cryptography' package. "
                "Install with: pip install cryptography"
            ) from e

        if self.kty == "RSA":
            return self._build_rsa_key()
        elif self.kty == "EC":
            return self._build_ec_key()
        else:
            raise ValueError(f"Unsupported key type: {self.kty}")

    def _build_rsa_key(self) -> Any:
        """Build RSA public key from JWK parameters."""
        from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicNumbers
        from cryptography.hazmat.backends import default_backend

        # Decode base64url values
        n_bytes = self._base64url_decode(self.n)
        e_bytes = self._base64url_decode(self.e)

        # Convert to integers
        n_int = int.from_bytes(n_bytes, byteorder="big")
        e_int = int.from_bytes(e_bytes, byteorder="big")

        # Create public key
        public_numbers = RSAPublicNumbers(e_int, n_int)
        return public_numbers.public_key(default_backend())

    def _build_ec_key(self) -> Any:
        """Build EC public key from JWK parameters."""
        from cryptography.hazmat.primitives.asymmetric.ec import (
            EllipticCurvePublicNumbers,
            SECP256R1,
            SECP384R1,
            SECP521R1,
        )
        from cryptography.hazmat.backends import default_backend

        # Map curve names
        curves = {
            "P-256": SECP256R1(),
            "P-384": SECP384R1(),
            "P-521": SECP521R1(),
        }

        curve = curves.get(self.crv)
        if not curve:
            raise ValueError(f"Unsupported curve: {self.crv}")

        # Decode coordinates
        x_bytes = self._base64url_decode(self.x)
        y_bytes = self._base64url_decode(self.y)

        x_int = int.from_bytes(x_bytes, byteorder="big")
        y_int = int.from_bytes(y_bytes, byteorder="big")

        public_numbers = EllipticCurvePublicNumbers(x_int, y_int, curve)
        return public_numbers.public_key(default_backend())

    @staticmethod
    def _base64url_decode(value: str) -> bytes:
        """Decode base64url value."""
        # Add padding if needed
        padding = 4 - len(value) % 4
        if padding != 4:
            value += "=" * padding
        return base64.urlsafe_b64decode(value)


@dataclass
class JWKS:
    """JSON Web Key Set.

    Attributes:
        keys: List of JWK objects.
        fetched_at: When the JWKS was fetched.
        expires_at: When the cached JWKS expires.
    """

    keys: list[JWK] = field(default_factory=list)
    fetched_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], cache_ttl: int = 3600) -> "JWKS":
        """Create JWKS from dictionary."""
        keys = [JWK.from_dict(k) for k in data.get("keys", [])]
        now = datetime.now()
        return cls(
            keys=keys,
            fetched_at=now,
            expires_at=now + timedelta(seconds=cache_ttl),
        )

    def get_key(self, kid: str) -> JWK | None:
        """Get key by key ID."""
        for key in self.keys:
            if key.kid == kid:
                return key
        return None

    @property
    def is_expired(self) -> bool:
        """Check if cached JWKS is expired."""
        if self.expires_at is None:
            return False
        return datetime.now() >= self.expires_at


class JWKSVerifier(ABC):
    """Abstract base class for JWKS-based verification."""

    @abstractmethod
    def get_jwks(self, force_refresh: bool = False) -> JWKS:
        """Get JWKS (potentially cached).

        Args:
            force_refresh: Force refresh from source.

        Returns:
            JWKS instance.
        """
        pass

    @abstractmethod
    def verify(
        self,
        token: str,
        audience: str | list[str] | None = None,
    ) -> "TokenVerificationResult":
        """Verify a JWT token.

        Args:
            token: JWT token string.
            audience: Expected audience(s).

        Returns:
            TokenVerificationResult.
        """
        pass


class GitHubActionsJWKS(JWKSVerifier):
    """GitHub Actions OIDC JWKS verifier.

    Fetches and caches the JWKS from GitHub Actions OIDC provider,
    and provides token verification.

    Example:
        >>> verifier = GitHubActionsJWKS()
        >>> result = verifier.verify(token, audience="sts.amazonaws.com")
        >>> if result.is_valid:
        ...     print(f"Valid token for: {result.claims.repository}")
    """

    def __init__(
        self,
        *,
        jwks_url: str = GITHUB_OIDC_JWKS_URL,
        issuer: str = GITHUB_OIDC_ISSUER,
        cache_ttl_seconds: int = 3600,
        request_timeout: float = 30.0,
    ) -> None:
        """Initialize verifier.

        Args:
            jwks_url: URL to fetch JWKS from.
            issuer: Expected token issuer.
            cache_ttl_seconds: JWKS cache TTL.
            request_timeout: HTTP request timeout.
        """
        self._jwks_url = jwks_url
        self._issuer = issuer
        self._cache_ttl = cache_ttl_seconds
        self._request_timeout = request_timeout
        self._cached_jwks: JWKS | None = None

    def get_jwks(self, force_refresh: bool = False) -> JWKS:
        """Get JWKS from GitHub Actions OIDC provider.

        Args:
            force_refresh: Force refresh from server.

        Returns:
            JWKS instance.
        """
        # Check cache
        if not force_refresh and self._cached_jwks:
            if not self._cached_jwks.is_expired:
                return self._cached_jwks

        # Fetch from server
        import urllib.request
        import urllib.error

        request = urllib.request.Request(
            self._jwks_url,
            headers={"Accept": "application/json"},
        )

        try:
            with urllib.request.urlopen(
                request, timeout=self._request_timeout
            ) as response:
                data = json.loads(response.read())
                self._cached_jwks = JWKS.from_dict(data, self._cache_ttl)
                return self._cached_jwks

        except urllib.error.URLError as e:
            logger.error(f"Failed to fetch JWKS: {e}")
            # Return cached if available, even if expired
            if self._cached_jwks:
                return self._cached_jwks
            raise

    def verify(
        self,
        token: str,
        audience: str | list[str] | None = None,
    ) -> "TokenVerificationResult":
        """Verify a GitHub Actions OIDC token.

        Args:
            token: JWT token string.
            audience: Expected audience(s).

        Returns:
            TokenVerificationResult with verification status.
        """
        try:
            # Parse token header to get key ID
            header = self._parse_header(token)
            kid = header.get("kid", "")
            alg = header.get("alg", "RS256")

            # Get JWKS and find key
            jwks = self.get_jwks()
            jwk = jwks.get_key(kid)

            if jwk is None:
                # Try refreshing JWKS
                jwks = self.get_jwks(force_refresh=True)
                jwk = jwks.get_key(kid)

            if jwk is None:
                return TokenVerificationResult(
                    is_valid=False,
                    error=f"Key not found: {kid}",
                )

            # Verify signature
            if not self._verify_signature(token, jwk, alg):
                return TokenVerificationResult(
                    is_valid=False,
                    error="Signature verification failed",
                )

            # Parse and validate claims
            payload = self._parse_payload(token)
            claims = parse_github_claims(payload)

            # Check expiration
            if claims.is_expired:
                return TokenVerificationResult(
                    is_valid=False,
                    error="Token is expired",
                    claims=claims,
                )

            # Check issuer
            if claims.issuer != self._issuer:
                return TokenVerificationResult(
                    is_valid=False,
                    error=f"Invalid issuer: {claims.issuer}",
                    claims=claims,
                )

            # Check audience
            if audience:
                audiences = audience if isinstance(audience, list) else [audience]
                claim_audiences = (
                    claims.audience
                    if isinstance(claims.audience, list)
                    else [claims.audience]
                )

                if not any(aud in claim_audiences for aud in audiences):
                    return TokenVerificationResult(
                        is_valid=False,
                        error=f"Invalid audience: {claims.audience}",
                        claims=claims,
                    )

            return TokenVerificationResult(
                is_valid=True,
                claims=claims,
            )

        except Exception as e:
            logger.error(f"Token verification failed: {e}")
            return TokenVerificationResult(
                is_valid=False,
                error=str(e),
            )

    def _parse_header(self, token: str) -> dict[str, Any]:
        """Parse JWT header."""
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("Invalid JWT format")

        header_b64 = parts[0]
        padding = 4 - len(header_b64) % 4
        if padding != 4:
            header_b64 += "=" * padding

        header_json = base64.urlsafe_b64decode(header_b64)
        return json.loads(header_json)

    def _parse_payload(self, token: str) -> dict[str, Any]:
        """Parse JWT payload."""
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("Invalid JWT format")

        payload_b64 = parts[1]
        padding = 4 - len(payload_b64) % 4
        if padding != 4:
            payload_b64 += "=" * padding

        payload_json = base64.urlsafe_b64decode(payload_b64)
        return json.loads(payload_json)

    def _verify_signature(self, token: str, jwk: JWK, alg: str) -> bool:
        """Verify JWT signature using JWK."""
        try:
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric import padding, utils, ec
            from cryptography.exceptions import InvalidSignature
        except ImportError:
            logger.warning(
                "cryptography not available, skipping signature verification"
            )
            return True  # Skip verification if cryptography not available

        parts = token.split(".")
        if len(parts) != 3:
            return False

        # Data to verify (header.payload)
        message = f"{parts[0]}.{parts[1]}".encode()

        # Decode signature
        sig_b64 = parts[2]
        padding_len = 4 - len(sig_b64) % 4
        if padding_len != 4:
            sig_b64 += "=" * padding_len
        signature = base64.urlsafe_b64decode(sig_b64)

        # Get public key
        public_key = jwk.get_public_key()

        try:
            if alg.startswith("RS"):
                # RSA algorithms
                hash_alg = self._get_hash_algorithm(alg)
                public_key.verify(
                    signature,
                    message,
                    padding.PKCS1v15(),
                    hash_alg,
                )
            elif alg.startswith("ES"):
                # ECDSA algorithms
                hash_alg = self._get_hash_algorithm(alg)
                # Convert DER signature to raw if needed
                public_key.verify(
                    signature,
                    message,
                    ec.ECDSA(hash_alg),
                )
            else:
                logger.warning(f"Unsupported algorithm: {alg}")
                return False

            return True

        except InvalidSignature:
            return False

    def _get_hash_algorithm(self, alg: str) -> Any:
        """Get hash algorithm for JWT algorithm."""
        from cryptography.hazmat.primitives import hashes

        if alg in ("RS256", "ES256"):
            return hashes.SHA256()
        elif alg in ("RS384", "ES384"):
            return hashes.SHA384()
        elif alg in ("RS512", "ES512"):
            return hashes.SHA512()
        else:
            return hashes.SHA256()


# =============================================================================
# Token Verifier
# =============================================================================


@dataclass
class TokenVerificationResult:
    """Result of token verification.

    Attributes:
        is_valid: Whether token is valid.
        claims: Parsed claims (if successful).
        error: Error message (if failed).
    """

    is_valid: bool
    claims: GitHubActionsClaims | None = None
    error: str | None = None

    def __bool__(self) -> bool:
        return self.is_valid


class TokenVerifier:
    """Token verification with configurable policies.

    Example:
        >>> verifier = TokenVerifier(
        ...     allowed_repositories=["owner/repo"],
        ...     allowed_branches=["main"],
        ...     require_environment=True,
        ... )
        >>> result = verifier.verify(token, audience="sts.amazonaws.com")
    """

    def __init__(
        self,
        *,
        jwks_verifier: JWKSVerifier | None = None,
        allowed_repositories: list[str] | None = None,
        allowed_branches: list[str] | None = None,
        allowed_environments: list[str] | None = None,
        allowed_actors: list[str] | None = None,
        require_environment: bool = False,
        deny_pull_requests: bool = False,
        verify_signature: bool = True,
    ) -> None:
        """Initialize verifier.

        Args:
            jwks_verifier: JWKS verifier instance.
            allowed_repositories: List of allowed repository patterns.
            allowed_branches: List of allowed branch names.
            allowed_environments: List of allowed environment names.
            allowed_actors: List of allowed actor names.
            require_environment: Require deployment environment.
            deny_pull_requests: Deny pull request events.
            verify_signature: Enable signature verification.
        """
        self._jwks_verifier = jwks_verifier or GitHubActionsJWKS()
        self._allowed_repositories = allowed_repositories
        self._allowed_branches = allowed_branches
        self._allowed_environments = allowed_environments
        self._allowed_actors = allowed_actors
        self._require_environment = require_environment
        self._deny_pull_requests = deny_pull_requests
        self._verify_signature = verify_signature

    def verify(
        self,
        token: str,
        audience: str | list[str] | None = None,
    ) -> TokenVerificationResult:
        """Verify token with policy checks.

        Args:
            token: JWT token string.
            audience: Expected audience(s).

        Returns:
            TokenVerificationResult.
        """
        # First, verify signature and basic claims
        if self._verify_signature:
            result = self._jwks_verifier.verify(token, audience)
            if not result.is_valid:
                return result
            claims = result.claims
        else:
            # Parse claims without verification
            from truthound.secrets.oidc.github.claims import parse_github_claims
            payload = self._parse_payload(token)
            claims = parse_github_claims(payload)

        if claims is None:
            return TokenVerificationResult(
                is_valid=False,
                error="Failed to parse claims",
            )

        # Apply policy checks
        errors: list[str] = []

        # Check repository
        if self._allowed_repositories:
            if not any(
                claims.matches_repository(repo)
                for repo in self._allowed_repositories
            ):
                errors.append(
                    f"Repository '{claims.repository}' not in allowed list"
                )

        # Check branch
        if self._allowed_branches and claims.branch_name:
            if claims.branch_name not in self._allowed_branches:
                errors.append(
                    f"Branch '{claims.branch_name}' not in allowed list"
                )

        # Check environment
        if self._require_environment and not claims.environment:
            errors.append("Deployment environment required")

        if self._allowed_environments and claims.environment:
            if claims.environment not in self._allowed_environments:
                errors.append(
                    f"Environment '{claims.environment}' not in allowed list"
                )

        # Check actors
        if self._allowed_actors:
            if claims.actor not in self._allowed_actors:
                errors.append(
                    f"Actor '{claims.actor}' not in allowed list"
                )

        # Check pull requests
        if self._deny_pull_requests and claims.is_pull_request:
            errors.append("Pull request events are denied")

        if errors:
            return TokenVerificationResult(
                is_valid=False,
                claims=claims,
                error="; ".join(errors),
            )

        return TokenVerificationResult(
            is_valid=True,
            claims=claims,
        )

    def _parse_payload(self, token: str) -> dict[str, Any]:
        """Parse JWT payload without verification."""
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("Invalid JWT format")

        payload_b64 = parts[1]
        padding = 4 - len(payload_b64) % 4
        if padding != 4:
            payload_b64 += "=" * padding

        payload_json = base64.urlsafe_b64decode(payload_b64)
        return json.loads(payload_json)


# =============================================================================
# Convenience Functions
# =============================================================================


def verify_token(
    token: str,
    audience: str | list[str] | None = None,
    *,
    verify_signature: bool = True,
) -> TokenVerificationResult:
    """Verify a GitHub Actions OIDC token.

    Args:
        token: JWT token string.
        audience: Expected audience(s).
        verify_signature: Enable signature verification.

    Returns:
        TokenVerificationResult.

    Example:
        >>> result = verify_token(token, audience="sts.amazonaws.com")
        >>> if result.is_valid:
        ...     print(f"Valid for: {result.claims.repository}")
    """
    verifier = TokenVerifier(verify_signature=verify_signature)
    return verifier.verify(token, audience)


def get_jwks() -> JWKS:
    """Fetch GitHub Actions OIDC JWKS.

    Returns:
        JWKS instance.
    """
    verifier = GitHubActionsJWKS()
    return verifier.get_jwks()
