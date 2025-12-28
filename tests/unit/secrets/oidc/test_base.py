"""Tests for OIDC base classes and protocols.

This module tests the core abstractions for OIDC authentication:
- OIDCToken and OIDCClaims
- CloudCredentials (AWS, GCP, Azure)
- Exception hierarchy
- Protocols (OIDCProvider, TokenExchanger)
- BaseOIDCProvider and BaseTokenExchanger
"""

from __future__ import annotations

import base64
import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from truthound.secrets.oidc.base import (
    # Core types
    OIDCToken,
    OIDCClaims,
    CloudCredentials,
    AWSCredentials,
    GCPCredentials,
    AzureCredentials,
    CloudProvider,
    CIProvider,
    # Protocols
    OIDCProvider,
    TokenExchanger,
    # Base classes
    BaseOIDCProvider,
    BaseTokenExchanger,
    # Exceptions
    OIDCError,
    OIDCTokenError,
    OIDCExchangeError,
    OIDCConfigurationError,
    OIDCProviderNotAvailableError,
)


# =============================================================================
# Test Data
# =============================================================================


def create_jwt_token(
    claims: dict,
    header: dict | None = None,
) -> str:
    """Create a test JWT token (not cryptographically valid)."""
    if header is None:
        header = {"alg": "RS256", "typ": "JWT"}

    def encode_part(data: dict) -> str:
        json_bytes = json.dumps(data).encode()
        b64 = base64.urlsafe_b64encode(json_bytes).decode()
        return b64.rstrip("=")

    header_b64 = encode_part(header)
    payload_b64 = encode_part(claims)
    signature_b64 = encode_part({"sig": "dummy"})

    return f"{header_b64}.{payload_b64}.{signature_b64}"


def create_github_token_claims(
    repository: str = "owner/repo",
    ref: str = "refs/heads/main",
    sha: str = "abc123",
    actor: str = "github-user",
    workflow: str = "CI",
    run_id: str = "12345",
    environment: str | None = None,
    exp_offset_seconds: int = 3600,
) -> dict:
    """Create GitHub Actions-like token claims."""
    now = datetime.now()
    exp = now + timedelta(seconds=exp_offset_seconds)
    iat = now

    claims = {
        "iss": "https://token.actions.githubusercontent.com",
        "sub": f"repo:{repository}:ref:{ref}",
        "aud": "sts.amazonaws.com",
        "exp": int(exp.timestamp()),
        "iat": int(iat.timestamp()),
        "repository": repository,
        "repository_owner": repository.split("/")[0],
        "ref": ref,
        "sha": sha,
        "actor": actor,
        "workflow": workflow,
        "run_id": run_id,
    }
    if environment:
        claims["environment"] = environment
    return claims


# =============================================================================
# OIDCClaims Tests
# =============================================================================


class TestOIDCClaims:
    """Tests for OIDCClaims dataclass."""

    def test_from_jwt_payload_basic(self):
        """Test parsing basic JWT claims."""
        now = datetime.now()
        exp = now + timedelta(hours=1)

        payload = {
            "iss": "https://example.com",
            "sub": "user123",
            "aud": "api.example.com",
            "exp": int(exp.timestamp()),
            "iat": int(now.timestamp()),
        }

        claims = OIDCClaims.from_jwt_payload(payload)

        assert claims.issuer == "https://example.com"
        assert claims.subject == "user123"
        assert claims.audience == "api.example.com"
        assert not claims.is_expired

    def test_from_jwt_payload_github_actions(self):
        """Test parsing GitHub Actions token claims."""
        payload = create_github_token_claims(
            repository="myorg/myrepo",
            ref="refs/heads/feature",
            sha="deadbeef",
            actor="developer",
            workflow="Build",
            run_id="999",
            environment="production",
        )

        claims = OIDCClaims.from_jwt_payload(payload)

        assert claims.issuer == "https://token.actions.githubusercontent.com"
        assert claims.repository == "myorg/myrepo"
        assert claims.ref == "refs/heads/feature"
        assert claims.sha == "deadbeef"
        assert claims.actor == "developer"
        assert claims.workflow == "Build"
        assert claims.run_id == "999"
        assert claims.environment == "production"

    def test_from_jwt_payload_gitlab_ci(self):
        """Test parsing GitLab CI token claims."""
        now = datetime.now()
        payload = {
            "iss": "https://gitlab.com",
            "sub": "project_path:group/project:ref_type:branch:ref:main",
            "aud": "https://aws.amazon.com",
            "exp": int((now + timedelta(hours=1)).timestamp()),
            "iat": int(now.timestamp()),
            "project_path": "group/project",
            "ref_path": "refs/heads/main",
            "commit_sha": "abc123",
            "user_login": "gitlab-user",
            "pipeline_source": "push",
            "pipeline_id": "12345",
            "job_id": "67890",
            "environment_scope": "production",
        }

        claims = OIDCClaims.from_jwt_payload(payload)

        assert claims.issuer == "https://gitlab.com"
        assert claims.repository == "group/project"  # from project_path
        assert claims.ref == "refs/heads/main"  # from ref_path
        assert claims.sha == "abc123"  # from commit_sha
        assert claims.actor == "gitlab-user"  # from user_login
        assert claims.workflow == "push"  # from pipeline_source
        assert claims.job == "67890"  # from job_id
        assert claims.run_id == "12345"  # from pipeline_id
        assert claims.environment == "production"  # from environment_scope

    def test_is_expired_true(self):
        """Test expired token detection."""
        payload = create_github_token_claims(exp_offset_seconds=-60)  # Expired 1 min ago
        claims = OIDCClaims.from_jwt_payload(payload)

        assert claims.is_expired is True

    def test_is_expired_false(self):
        """Test non-expired token detection."""
        payload = create_github_token_claims(exp_offset_seconds=3600)  # Expires in 1 hour
        claims = OIDCClaims.from_jwt_payload(payload)

        assert claims.is_expired is False

    def test_time_until_expiry(self):
        """Test time until expiration calculation."""
        payload = create_github_token_claims(exp_offset_seconds=600)  # 10 minutes
        claims = OIDCClaims.from_jwt_payload(payload)

        remaining = claims.time_until_expiry
        assert 590 < remaining.total_seconds() < 610  # Allow some variance

    def test_get_audience_list_string(self):
        """Test audience list extraction from string."""
        now = datetime.now()
        payload = {
            "iss": "https://example.com",
            "sub": "user",
            "aud": "single-audience",
            "exp": int((now + timedelta(hours=1)).timestamp()),
            "iat": int(now.timestamp()),
        }
        claims = OIDCClaims.from_jwt_payload(payload)

        assert claims.get_audience_list() == ["single-audience"]

    def test_get_audience_list_list(self):
        """Test audience list extraction from list."""
        now = datetime.now()
        payload = {
            "iss": "https://example.com",
            "sub": "user",
            "aud": ["audience1", "audience2"],
            "exp": int((now + timedelta(hours=1)).timestamp()),
            "iat": int(now.timestamp()),
        }
        claims = OIDCClaims.from_jwt_payload(payload)

        assert claims.get_audience_list() == ["audience1", "audience2"]

    def test_extra_claims_preserved(self):
        """Test that unknown claims are preserved in extra."""
        now = datetime.now()
        payload = {
            "iss": "https://example.com",
            "sub": "user",
            "aud": "api",
            "exp": int((now + timedelta(hours=1)).timestamp()),
            "iat": int(now.timestamp()),
            "custom_claim": "custom_value",
            "another_claim": 123,
        }
        claims = OIDCClaims.from_jwt_payload(payload)

        assert claims.extra["custom_claim"] == "custom_value"
        assert claims.extra["another_claim"] == 123


# =============================================================================
# OIDCToken Tests
# =============================================================================


class TestOIDCToken:
    """Tests for OIDCToken container."""

    def test_token_creation(self):
        """Test basic token creation."""
        claims = create_github_token_claims()
        jwt = create_jwt_token(claims)

        token = OIDCToken(jwt, provider="github_actions")

        assert token.provider == "github_actions"
        assert token.get_token() == jwt

    def test_claims_lazy_parsing(self):
        """Test that claims are parsed lazily."""
        claims = create_github_token_claims()
        jwt = create_jwt_token(claims)

        token = OIDCToken(jwt, provider="github_actions")

        # Claims not parsed yet
        assert token._claims is None

        # Access claims
        _ = token.claims

        # Now parsed
        assert token._claims is not None

    def test_claims_parsing(self):
        """Test claims are correctly parsed."""
        claims = create_github_token_claims(
            repository="test/repo",
            actor="test-user",
        )
        jwt = create_jwt_token(claims)

        token = OIDCToken(jwt, provider="github_actions")

        assert token.claims.repository == "test/repo"
        assert token.claims.actor == "test-user"
        assert token.claims.issuer == "https://token.actions.githubusercontent.com"

    def test_is_expired(self):
        """Test token expiration check."""
        # Expired token
        expired_claims = create_github_token_claims(exp_offset_seconds=-60)
        expired_jwt = create_jwt_token(expired_claims)
        expired_token = OIDCToken(expired_jwt, provider="test")

        assert expired_token.is_expired is True

        # Valid token
        valid_claims = create_github_token_claims(exp_offset_seconds=3600)
        valid_jwt = create_jwt_token(valid_claims)
        valid_token = OIDCToken(valid_jwt, provider="test")

        assert valid_token.is_expired is False

    def test_hash_for_change_detection(self):
        """Test token hash for caching."""
        claims = create_github_token_claims()
        jwt = create_jwt_token(claims)

        token = OIDCToken(jwt, provider="test")

        assert len(token.hash) == 16  # SHA256[:16]

        # Same token should have same hash
        token2 = OIDCToken(jwt, provider="test")
        assert token.hash == token2.hash

        # Different token should have different hash
        claims2 = create_github_token_claims(sha="different")
        jwt2 = create_jwt_token(claims2)
        token3 = OIDCToken(jwt2, provider="test")
        assert token.hash != token3.hash

    def test_repr_does_not_expose_token(self):
        """Test that repr doesn't expose the raw token."""
        claims = create_github_token_claims()
        jwt = create_jwt_token(claims)

        token = OIDCToken(jwt, provider="github_actions")

        repr_str = repr(token)
        str_str = str(token)

        # Should not contain the actual JWT
        assert jwt not in repr_str
        assert jwt not in str_str

        # Should show provider info
        assert "github_actions" in repr_str
        assert "OIDCToken" in str_str

    def test_bool_valid_token(self):
        """Test bool conversion for valid token."""
        claims = create_github_token_claims(exp_offset_seconds=3600)
        jwt = create_jwt_token(claims)

        token = OIDCToken(jwt, provider="test")

        assert bool(token) is True

    def test_bool_expired_token(self):
        """Test bool conversion for expired token."""
        claims = create_github_token_claims(exp_offset_seconds=-60)
        jwt = create_jwt_token(claims)

        token = OIDCToken(jwt, provider="test")

        assert bool(token) is False

    def test_invalid_jwt_format(self):
        """Test error handling for invalid JWT format."""
        token = OIDCToken("not.a.valid.jwt.with.too.many.parts", provider="test")

        with pytest.raises(OIDCTokenError) as exc_info:
            _ = token.claims

        assert "Invalid JWT format" in str(exc_info.value)
        assert exc_info.value.provider == "test"

    def test_invalid_jwt_payload(self):
        """Test error handling for invalid JWT payload."""
        # Create token with invalid base64 payload
        token = OIDCToken("header.not_valid_base64!!!.signature", provider="test")

        with pytest.raises(OIDCTokenError) as exc_info:
            _ = token.claims

        assert exc_info.value.provider == "test"


# =============================================================================
# CloudCredentials Tests
# =============================================================================


class TestCloudCredentials:
    """Tests for cloud credential dataclasses."""

    def test_aws_credentials_creation(self):
        """Test AWS credentials creation."""
        creds = AWSCredentials(
            access_key_id="AKIAIOSFODNN7EXAMPLE",
            secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            session_token="session-token",
            assumed_role_arn="arn:aws:sts::123456789012:assumed-role/role/session",
            expires_at=datetime.now() + timedelta(hours=1),
        )

        assert creds.provider == CloudProvider.AWS
        assert creds.access_key_id == "AKIAIOSFODNN7EXAMPLE"
        assert not creds.is_expired

    def test_aws_credentials_to_boto3(self):
        """Test AWS credentials conversion to boto3 format."""
        creds = AWSCredentials(
            access_key_id="AKID",
            secret_access_key="SECRET",
            session_token="TOKEN",
        )

        boto3_creds = creds.to_boto3_session_credentials()

        assert boto3_creds == {
            "aws_access_key_id": "AKID",
            "aws_secret_access_key": "SECRET",
            "aws_session_token": "TOKEN",
        }

    def test_aws_credentials_to_env(self):
        """Test AWS credentials conversion to environment variables."""
        creds = AWSCredentials(
            access_key_id="AKID",
            secret_access_key="SECRET",
            session_token="TOKEN",
        )

        env_vars = creds.to_environment_variables()

        assert env_vars == {
            "AWS_ACCESS_KEY_ID": "AKID",
            "AWS_SECRET_ACCESS_KEY": "SECRET",
            "AWS_SESSION_TOKEN": "TOKEN",
        }

    def test_aws_credentials_repr_safe(self):
        """Test AWS credentials repr doesn't expose secrets."""
        creds = AWSCredentials(
            access_key_id="AKIAIOSFODNN7EXAMPLE",
            secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            session_token="very-long-session-token",
            assumed_role_arn="arn:aws:sts::123456789012:assumed-role/role/session",
        )

        repr_str = repr(creds)

        assert "wJalrXUtnFEMI" not in repr_str  # Secret key
        assert "very-long-session-token" not in repr_str
        assert "AKIAIOSF" in repr_str  # Truncated access key shown (first 8 chars)

    def test_gcp_credentials_creation(self):
        """Test GCP credentials creation."""
        creds = GCPCredentials(
            access_token="ya29.example-token",
            service_account="sa@project.iam.gserviceaccount.com",
            project_id="my-project",
            expires_at=datetime.now() + timedelta(hours=1),
        )

        assert creds.provider == CloudProvider.GCP
        assert creds.token_type == "Bearer"
        assert not creds.is_expired

    def test_gcp_credentials_to_google_format(self):
        """Test GCP credentials conversion to google-auth format."""
        expires_at = datetime.now() + timedelta(hours=1)
        creds = GCPCredentials(
            access_token="token123",
            expires_at=expires_at,
        )

        google_creds = creds.to_google_credentials()

        assert google_creds["token"] == "token123"
        assert google_creds["expiry"] == expires_at

    def test_gcp_credentials_authorization_header(self):
        """Test GCP credentials authorization header."""
        creds = GCPCredentials(
            access_token="token123",
            token_type="Bearer",
        )

        header = creds.get_authorization_header()

        assert header == {"Authorization": "Bearer token123"}

    def test_azure_credentials_creation(self):
        """Test Azure credentials creation."""
        creds = AzureCredentials(
            access_token="eyJ0eXAiOiJKV1QiLCJhbGciOiJS...",
            tenant_id="12345678-1234-1234-1234-123456789012",
            client_id="87654321-4321-4321-4321-210987654321",
            subscription_id="abcdef12-3456-7890-abcd-ef1234567890",
        )

        assert creds.provider == CloudProvider.AZURE
        assert creds.token_type == "Bearer"

    def test_azure_credentials_to_env(self):
        """Test Azure credentials conversion to environment variables."""
        creds = AzureCredentials(
            access_token="token",
            tenant_id="tenant",
            client_id="client",
            subscription_id="subscription",
        )

        env_vars = creds.to_environment_variables()

        assert env_vars == {
            "AZURE_ACCESS_TOKEN": "token",
            "AZURE_TENANT_ID": "tenant",
            "AZURE_CLIENT_ID": "client",
            "AZURE_SUBSCRIPTION_ID": "subscription",
        }

    def test_credentials_is_expired(self):
        """Test credential expiration check."""
        # Expired
        expired_creds = AWSCredentials(
            access_key_id="AKID",
            secret_access_key="SECRET",
            session_token="TOKEN",
            expires_at=datetime.now() - timedelta(minutes=1),
        )
        assert expired_creds.is_expired is True

        # Not expired
        valid_creds = AWSCredentials(
            access_key_id="AKID",
            secret_access_key="SECRET",
            session_token="TOKEN",
            expires_at=datetime.now() + timedelta(hours=1),
        )
        assert valid_creds.is_expired is False

        # No expiration set
        no_exp_creds = AWSCredentials(
            access_key_id="AKID",
            secret_access_key="SECRET",
            session_token="TOKEN",
        )
        assert no_exp_creds.is_expired is False

    def test_credentials_time_until_expiry(self):
        """Test time until expiration calculation."""
        expires_in = timedelta(minutes=30)
        creds = AWSCredentials(
            access_key_id="AKID",
            secret_access_key="SECRET",
            session_token="TOKEN",
            expires_at=datetime.now() + expires_in,
        )

        remaining = creds.time_until_expiry
        assert remaining is not None
        assert 1700 < remaining.total_seconds() < 1810  # ~30 min with variance


# =============================================================================
# Enum Tests
# =============================================================================


class TestEnums:
    """Tests for CloudProvider and CIProvider enums."""

    def test_cloud_provider_values(self):
        """Test CloudProvider enum values."""
        assert CloudProvider.AWS.value == "aws"
        assert CloudProvider.GCP.value == "gcp"
        assert CloudProvider.AZURE.value == "azure"
        assert CloudProvider.VAULT.value == "vault"

    def test_cloud_provider_str(self):
        """Test CloudProvider string conversion."""
        assert str(CloudProvider.AWS) == "aws"
        assert str(CloudProvider.GCP) == "gcp"

    def test_ci_provider_values(self):
        """Test CIProvider enum values."""
        assert CIProvider.GITHUB_ACTIONS.value == "github_actions"
        assert CIProvider.GITLAB_CI.value == "gitlab_ci"
        assert CIProvider.CIRCLECI.value == "circleci"
        assert CIProvider.BITBUCKET.value == "bitbucket"
        assert CIProvider.JENKINS.value == "jenkins"
        assert CIProvider.UNKNOWN.value == "unknown"


# =============================================================================
# Exception Tests
# =============================================================================


class TestExceptions:
    """Tests for OIDC exception hierarchy."""

    def test_oidc_error_base(self):
        """Test base OIDC error."""
        error = OIDCError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert isinstance(error, Exception)

    def test_oidc_token_error(self):
        """Test OIDC token error."""
        error = OIDCTokenError("Token invalid", provider="github")

        assert error.provider == "github"
        assert "[github]" in str(error)
        assert "Token invalid" in str(error)

    def test_oidc_token_error_without_provider(self):
        """Test OIDC token error without provider."""
        error = OIDCTokenError("Token expired")

        assert error.provider is None
        assert "Token expired" in str(error)

    def test_oidc_exchange_error(self):
        """Test OIDC exchange error."""
        error = OIDCExchangeError(
            "Authentication failed",
            cloud_provider="aws",
            status_code=403,
            response='{"error": "access_denied"}',
        )

        assert error.cloud_provider == "aws"
        assert error.status_code == 403
        assert error.response == '{"error": "access_denied"}'
        assert "[aws]" in str(error)
        assert "403" in str(error)

    def test_oidc_configuration_error(self):
        """Test OIDC configuration error."""
        error = OIDCConfigurationError(
            "Missing required field",
            field="role_arn",
        )

        assert error.field == "role_arn"
        assert "role_arn" in str(error)

    def test_oidc_provider_not_available_error(self):
        """Test OIDC provider not available error."""
        error = OIDCProviderNotAvailableError(
            "github_actions",
            "Missing ACTIONS_ID_TOKEN_REQUEST_URL",
        )

        assert error.provider == "github_actions"
        assert error.reason == "Missing ACTIONS_ID_TOKEN_REQUEST_URL"
        assert "github_actions" in str(error)
        assert "not available" in str(error)


# =============================================================================
# Protocol Tests
# =============================================================================


class TestProtocols:
    """Tests for OIDCProvider and TokenExchanger protocols."""

    def test_oidc_provider_protocol(self):
        """Test that classes implementing OIDCProvider are recognized."""

        class MockOIDCProvider:
            @property
            def name(self) -> str:
                return "mock"

            def get_token(self, audience: str | None = None) -> OIDCToken:
                claims = create_github_token_claims()
                return OIDCToken(create_jwt_token(claims), provider="mock")

            def is_available(self) -> bool:
                return True

        provider = MockOIDCProvider()
        assert isinstance(provider, OIDCProvider)

    def test_token_exchanger_protocol(self):
        """Test that classes implementing TokenExchanger are recognized."""

        class MockTokenExchanger:
            @property
            def cloud_provider(self) -> CloudProvider:
                return CloudProvider.AWS

            def exchange(self, token: OIDCToken) -> CloudCredentials:
                return AWSCredentials(
                    access_key_id="AKID",
                    secret_access_key="SECRET",
                    session_token="TOKEN",
                )

        exchanger = MockTokenExchanger()
        assert isinstance(exchanger, TokenExchanger)


# =============================================================================
# BaseOIDCProvider Tests
# =============================================================================


class TestBaseOIDCProvider:
    """Tests for BaseOIDCProvider base class."""

    def test_concrete_provider(self):
        """Test concrete provider implementation."""

        class ConcreteProvider(BaseOIDCProvider):
            @property
            def name(self) -> str:
                return "concrete"

            def _fetch_token(self, audience: str | None = None) -> str:
                claims = create_github_token_claims()
                return create_jwt_token(claims)

            def is_available(self) -> bool:
                return True

        provider = ConcreteProvider()
        token = provider.get_token()

        assert token.provider == "concrete"
        assert not token.is_expired

    def test_token_caching(self):
        """Test token caching behavior."""
        fetch_count = 0

        class CachingProvider(BaseOIDCProvider):
            @property
            def name(self) -> str:
                return "caching"

            def _fetch_token(self, audience: str | None = None) -> str:
                nonlocal fetch_count
                fetch_count += 1
                claims = create_github_token_claims()
                return create_jwt_token(claims)

            def is_available(self) -> bool:
                return True

        provider = CachingProvider(enable_cache=True)

        # First call - fetches
        token1 = provider.get_token("aud1")
        assert fetch_count == 1

        # Second call with same audience - uses cache
        token2 = provider.get_token("aud1")
        assert fetch_count == 1
        assert token1.hash == token2.hash

        # Third call with different audience - fetches
        token3 = provider.get_token("aud2")
        assert fetch_count == 2

    def test_token_caching_disabled(self):
        """Test behavior with caching disabled."""
        fetch_count = 0

        class NoCacheProvider(BaseOIDCProvider):
            @property
            def name(self) -> str:
                return "no-cache"

            def _fetch_token(self, audience: str | None = None) -> str:
                nonlocal fetch_count
                fetch_count += 1
                claims = create_github_token_claims()
                return create_jwt_token(claims)

            def is_available(self) -> bool:
                return True

        provider = NoCacheProvider(enable_cache=False)

        provider.get_token()
        provider.get_token()
        provider.get_token()

        assert fetch_count == 3

    def test_clear_cache(self):
        """Test cache clearing."""
        fetch_count = 0

        class ClearableProvider(BaseOIDCProvider):
            @property
            def name(self) -> str:
                return "clearable"

            def _fetch_token(self, audience: str | None = None) -> str:
                nonlocal fetch_count
                fetch_count += 1
                claims = create_github_token_claims()
                return create_jwt_token(claims)

            def is_available(self) -> bool:
                return True

        provider = ClearableProvider()

        provider.get_token()
        provider.get_token()
        assert fetch_count == 1

        provider.clear_cache()

        provider.get_token()
        assert fetch_count == 2

    def test_provider_not_available(self):
        """Test error when provider is not available."""

        class UnavailableProvider(BaseOIDCProvider):
            @property
            def name(self) -> str:
                return "unavailable"

            def _fetch_token(self, audience: str | None = None) -> str:
                return ""

            def is_available(self) -> bool:
                return False

        provider = UnavailableProvider()

        with pytest.raises(OIDCProviderNotAvailableError) as exc_info:
            provider.get_token()

        assert exc_info.value.provider == "unavailable"

    def test_retry_on_failure(self):
        """Test retry behavior on token fetch failure."""
        attempt_count = 0

        class RetryProvider(BaseOIDCProvider):
            @property
            def name(self) -> str:
                return "retry"

            def _fetch_token(self, audience: str | None = None) -> str:
                nonlocal attempt_count
                attempt_count += 1
                if attempt_count < 3:
                    raise Exception("Transient error")
                claims = create_github_token_claims()
                return create_jwt_token(claims)

            def is_available(self) -> bool:
                return True

        provider = RetryProvider(retry_attempts=3, retry_delay_seconds=0.01)
        token = provider.get_token()

        assert attempt_count == 3
        assert token is not None


# =============================================================================
# BaseTokenExchanger Tests
# =============================================================================


class TestBaseTokenExchanger:
    """Tests for BaseTokenExchanger base class."""

    def test_concrete_exchanger(self):
        """Test concrete exchanger implementation."""

        class ConcreteExchanger(BaseTokenExchanger):
            @property
            def cloud_provider(self) -> CloudProvider:
                return CloudProvider.AWS

            def _exchange(self, token: OIDCToken) -> AWSCredentials:
                return AWSCredentials(
                    access_key_id="AKID",
                    secret_access_key="SECRET",
                    session_token="TOKEN",
                    expires_at=datetime.now() + timedelta(hours=1),
                )

        exchanger = ConcreteExchanger()
        claims = create_github_token_claims()
        jwt = create_jwt_token(claims)
        token = OIDCToken(jwt, provider="test")

        creds = exchanger.exchange(token)

        assert creds.access_key_id == "AKID"
        assert creds.provider == CloudProvider.AWS

    def test_credential_caching(self):
        """Test credential caching behavior."""
        exchange_count = 0

        class CachingExchanger(BaseTokenExchanger):
            @property
            def cloud_provider(self) -> CloudProvider:
                return CloudProvider.AWS

            def _exchange(self, token: OIDCToken) -> AWSCredentials:
                nonlocal exchange_count
                exchange_count += 1
                return AWSCredentials(
                    access_key_id="AKID",
                    secret_access_key="SECRET",
                    session_token="TOKEN",
                    expires_at=datetime.now() + timedelta(hours=1),
                )

        exchanger = CachingExchanger(enable_cache=True)
        claims = create_github_token_claims()
        jwt = create_jwt_token(claims)
        token = OIDCToken(jwt, provider="test")

        # First exchange
        creds1 = exchanger.exchange(token)
        assert exchange_count == 1

        # Second with same token - uses cache
        creds2 = exchanger.exchange(token)
        assert exchange_count == 1

        # Different token - exchanges again
        jwt2 = create_jwt_token(create_github_token_claims(sha="different"))
        token2 = OIDCToken(jwt2, provider="test")
        creds3 = exchanger.exchange(token2)
        assert exchange_count == 2

    def test_retry_on_exchange_failure(self):
        """Test retry behavior on exchange failure."""
        attempt_count = 0

        class RetryExchanger(BaseTokenExchanger):
            @property
            def cloud_provider(self) -> CloudProvider:
                return CloudProvider.AWS

            def _exchange(self, token: OIDCToken) -> AWSCredentials:
                nonlocal attempt_count
                attempt_count += 1
                if attempt_count < 2:
                    raise Exception("Transient error")
                return AWSCredentials(
                    access_key_id="AKID",
                    secret_access_key="SECRET",
                    session_token="TOKEN",
                )

        exchanger = RetryExchanger(retry_attempts=3, retry_delay_seconds=0.01)
        claims = create_github_token_claims()
        jwt = create_jwt_token(claims)
        token = OIDCToken(jwt, provider="test")

        creds = exchanger.exchange(token)

        assert attempt_count == 2
        assert creds.access_key_id == "AKID"

    def test_exchange_failure_after_retries(self):
        """Test exchange failure after all retries exhausted."""

        class FailingExchanger(BaseTokenExchanger):
            @property
            def cloud_provider(self) -> CloudProvider:
                return CloudProvider.AWS

            def _exchange(self, token: OIDCToken) -> AWSCredentials:
                raise Exception("Permanent error")

        exchanger = FailingExchanger(retry_attempts=2, retry_delay_seconds=0.01)
        claims = create_github_token_claims()
        jwt = create_jwt_token(claims)
        token = OIDCToken(jwt, provider="test")

        with pytest.raises(OIDCExchangeError):
            exchanger.exchange(token)
