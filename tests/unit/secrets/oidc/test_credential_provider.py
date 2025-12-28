"""Tests for OIDC credential provider.

This module tests the high-level credential provider API:
- OIDCCredentialProvider
- OIDCCredentialConfig
- OIDCSecretProvider
- Utility functions (get_oidc_credentials, with_oidc_credentials, oidc_credentials)
"""

from __future__ import annotations

import base64
import json
import os
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import pytest

from truthound.secrets.oidc.credential_provider import (
    OIDCCredentialProvider,
    OIDCCredentialConfig,
    OIDCSecretProvider,
    get_oidc_credentials,
    with_oidc_credentials,
    oidc_credentials,
)
from truthound.secrets.oidc.base import (
    CloudProvider,
    OIDCToken,
    OIDCError,
    OIDCProviderNotAvailableError,
    AWSCredentials,
    GCPCredentials,
    AzureCredentials,
)
from truthound.secrets.oidc.providers import (
    BaseOIDCProvider,
    GitHubActionsOIDCProvider,
)
from truthound.secrets.oidc.exchangers import (
    BaseTokenExchanger,
    AWSTokenExchanger,
)


# =============================================================================
# Test Utilities
# =============================================================================


def create_jwt_token(claims: dict) -> str:
    """Create a test JWT token."""
    header = {"alg": "RS256", "typ": "JWT"}

    def encode_part(data: dict) -> str:
        json_bytes = json.dumps(data).encode()
        b64 = base64.urlsafe_b64encode(json_bytes).decode()
        return b64.rstrip("=")

    header_b64 = encode_part(header)
    payload_b64 = encode_part(claims)
    signature_b64 = encode_part({"sig": "dummy"})

    return f"{header_b64}.{payload_b64}.{signature_b64}"


def create_test_claims(exp_offset: int = 3600) -> dict:
    """Create test token claims."""
    now = datetime.now()
    return {
        "iss": "https://token.actions.githubusercontent.com",
        "sub": "repo:owner/repo:ref:refs/heads/main",
        "aud": "sts.amazonaws.com",
        "exp": int((now + timedelta(seconds=exp_offset)).timestamp()),
        "iat": int(now.timestamp()),
        "repository": "owner/repo",
    }


class MockOIDCProvider(BaseOIDCProvider):
    """Mock OIDC provider for testing."""

    def __init__(self, jwt: str | None = None, available: bool = True, **kwargs):
        self._jwt = jwt or create_jwt_token(create_test_claims())
        self._available = available
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        return "mock"

    def is_available(self) -> bool:
        return self._available

    def _fetch_token(self, audience: str | None = None) -> str:
        return self._jwt


class MockTokenExchanger(BaseTokenExchanger):
    """Mock token exchanger for testing."""

    def __init__(
        self,
        cloud: CloudProvider = CloudProvider.AWS,
        credentials: AWSCredentials | None = None,
        **kwargs,
    ):
        self._cloud = cloud
        self._credentials = credentials or AWSCredentials(
            access_key_id="AKID",
            secret_access_key="SECRET",
            session_token="TOKEN",
            expires_at=datetime.now() + timedelta(hours=1),
        )
        super().__init__(**kwargs)

    @property
    def cloud_provider(self) -> CloudProvider:
        return self._cloud

    def _exchange(self, token: OIDCToken) -> AWSCredentials:
        return self._credentials


# =============================================================================
# OIDCCredentialConfig Tests
# =============================================================================


class TestOIDCCredentialConfig:
    """Tests for OIDCCredentialConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = OIDCCredentialConfig()

        assert config.cloud_provider == CloudProvider.AWS
        assert config.auto_detect_ci is True
        assert config.enable_cache is True
        assert config.cache_ttl_seconds == 3600
        assert config.fallback_to_env is True

    def test_cloud_provider_from_string(self):
        """Test cloud provider conversion from string."""
        config = OIDCCredentialConfig(cloud_provider="gcp")

        assert config.cloud_provider == CloudProvider.GCP

    def test_aws_specific_config(self):
        """Test AWS-specific configuration."""
        config = OIDCCredentialConfig(
            cloud_provider="aws",
            aws_role_arn="arn:aws:iam::123:role/test",
            aws_session_name="my-session",
            aws_region="us-west-2",
        )

        assert config.aws_role_arn == "arn:aws:iam::123:role/test"
        assert config.aws_session_name == "my-session"
        assert config.aws_region == "us-west-2"

    def test_gcp_specific_config(self):
        """Test GCP-specific configuration."""
        config = OIDCCredentialConfig(
            cloud_provider="gcp",
            gcp_project_number="123456789",
            gcp_pool_id="my-pool",
            gcp_provider_id="github",
            gcp_service_account="sa@project.iam.gserviceaccount.com",
        )

        assert config.gcp_project_number == "123456789"
        assert config.gcp_pool_id == "my-pool"
        assert config.gcp_provider_id == "github"

    def test_azure_specific_config(self):
        """Test Azure-specific configuration."""
        config = OIDCCredentialConfig(
            cloud_provider="azure",
            azure_tenant_id="tenant-id",
            azure_client_id="client-id",
            azure_subscription_id="sub-id",
        )

        assert config.azure_tenant_id == "tenant-id"
        assert config.azure_client_id == "client-id"

    def test_vault_specific_config(self):
        """Test Vault-specific configuration."""
        config = OIDCCredentialConfig(
            cloud_provider="vault",
            vault_url="https://vault.example.com",
            vault_role="my-role",
            vault_jwt_path="custom-jwt",
        )

        assert config.vault_url == "https://vault.example.com"
        assert config.vault_role == "my-role"
        assert config.vault_jwt_path == "custom-jwt"


# =============================================================================
# OIDCCredentialProvider Tests
# =============================================================================


class TestOIDCCredentialProvider:
    """Tests for OIDCCredentialProvider."""

    def test_init_with_explicit_providers(self):
        """Test initialization with explicit providers."""
        oidc_provider = MockOIDCProvider()
        token_exchanger = MockTokenExchanger()

        provider = OIDCCredentialProvider(
            oidc_provider=oidc_provider,
            token_exchanger=token_exchanger,
        )

        assert provider.oidc_provider is oidc_provider
        assert provider.token_exchanger is token_exchanger

    def test_init_with_config(self):
        """Test initialization with config object."""
        config = OIDCCredentialConfig(
            cloud_provider="aws",
            aws_role_arn="arn:aws:iam::123:role/test",
        )

        provider = OIDCCredentialProvider(
            config=config,
            oidc_provider=MockOIDCProvider(),
        )

        assert provider.cloud_provider == CloudProvider.AWS

    def test_init_with_kwargs(self):
        """Test initialization with keyword arguments."""
        provider = OIDCCredentialProvider(
            cloud_provider="gcp",
            gcp_project_number="123",
            gcp_pool_id="pool",
            oidc_provider=MockOIDCProvider(),
        )

        assert provider.cloud_provider == CloudProvider.GCP

    def test_cloud_provider_property(self):
        """Test cloud provider property."""
        provider = OIDCCredentialProvider(
            cloud_provider="azure",
            oidc_provider=MockOIDCProvider(),
        )

        assert provider.cloud_provider == CloudProvider.AZURE

    def test_is_available(self):
        """Test is_available method."""
        # Available
        provider = OIDCCredentialProvider(
            oidc_provider=MockOIDCProvider(available=True),
        )
        assert provider.is_available() is True

        # Not available
        provider = OIDCCredentialProvider(
            oidc_provider=MockOIDCProvider(available=False),
        )
        assert provider.is_available() is False

    def test_get_oidc_token(self):
        """Test get_oidc_token method."""
        provider = OIDCCredentialProvider(
            oidc_provider=MockOIDCProvider(),
        )

        token = provider.get_oidc_token()

        assert token is not None
        assert token.provider == "mock"

    def test_get_credentials(self):
        """Test get_credentials method."""
        provider = OIDCCredentialProvider(
            oidc_provider=MockOIDCProvider(),
            token_exchanger=MockTokenExchanger(),
        )

        creds = provider.get_credentials()

        assert isinstance(creds, AWSCredentials)
        assert creds.access_key_id == "AKID"

    def test_get_credentials_caching(self):
        """Test credential caching at the OIDCCredentialProvider level."""
        exchange_count = 0

        class CountingExchanger(MockTokenExchanger):
            def _exchange(self, token):
                nonlocal exchange_count
                exchange_count += 1
                return super()._exchange(token)

        # Disable caching on the exchanger so we can see exchange calls
        counting_exchanger = CountingExchanger(enable_cache=False)

        provider = OIDCCredentialProvider(
            oidc_provider=MockOIDCProvider(),
            token_exchanger=counting_exchanger,
            enable_cache=True,
        )

        # First call - triggers exchange
        creds1 = provider.get_credentials()
        first_exchange_count = exchange_count
        assert first_exchange_count >= 1

        # Second call - uses OIDCCredentialProvider's cached credentials
        creds2 = provider.get_credentials()
        # Should NOT trigger new exchange since provider caches
        assert exchange_count == first_exchange_count
        assert creds1.access_key_id == creds2.access_key_id

        # Force refresh - bypasses provider cache, triggers new exchange
        creds3 = provider.get_credentials(force_refresh=True)
        assert exchange_count > first_exchange_count

    def test_get_credentials_expired_cache(self):
        """Test that expired credentials are refreshed."""
        expired_creds = AWSCredentials(
            access_key_id="EXPIRED",
            secret_access_key="SECRET",
            session_token="TOKEN",
            expires_at=datetime.now() - timedelta(minutes=1),
        )
        fresh_creds = AWSCredentials(
            access_key_id="FRESH",
            secret_access_key="SECRET",
            session_token="TOKEN",
            expires_at=datetime.now() + timedelta(hours=1),
        )

        class RefreshingExchanger(MockTokenExchanger):
            def __init__(self):
                self._call_count = 0
                super().__init__()

            def _exchange(self, token):
                self._call_count += 1
                if self._call_count == 1:
                    return expired_creds
                return fresh_creds

        exchanger = RefreshingExchanger()
        provider = OIDCCredentialProvider(
            oidc_provider=MockOIDCProvider(),
            token_exchanger=exchanger,
        )

        # First call returns expired (but cached internally)
        creds1 = provider.get_credentials()
        assert creds1.access_key_id == "EXPIRED"

        # Second call should refresh because expired
        creds2 = provider.get_credentials()
        assert creds2.access_key_id == "FRESH"

    def test_clear_cache(self):
        """Test cache clearing."""
        provider = OIDCCredentialProvider(
            oidc_provider=MockOIDCProvider(),
            token_exchanger=MockTokenExchanger(),
        )

        # Get credentials to populate cache
        provider.get_credentials()
        assert provider._cached_credentials is not None

        # Clear cache
        provider.clear_cache()
        assert provider._cached_credentials is None

    def test_auto_detect_ci_not_available(self):
        """Test error when auto-detect finds no CI."""
        with patch.dict(os.environ, {}, clear=True):
            provider = OIDCCredentialProvider(
                auto_detect_ci=True,
            )

            with pytest.raises(OIDCProviderNotAvailableError) as exc_info:
                provider.get_credentials()

            assert "No supported CI environment" in str(exc_info.value)

    def test_auto_detect_ci_disabled(self):
        """Test error when auto-detect is disabled and no provider."""
        provider = OIDCCredentialProvider(
            auto_detect_ci=False,
        )

        with pytest.raises(OIDCError) as exc_info:
            _ = provider.oidc_provider

        assert "not configured" in str(exc_info.value)

    def test_repr(self):
        """Test repr method."""
        provider = OIDCCredentialProvider(
            cloud_provider="aws",
            oidc_provider=MockOIDCProvider(),
        )

        repr_str = repr(provider)
        assert "OIDCCredentialProvider" in repr_str
        assert "aws" in repr_str


# =============================================================================
# OIDCSecretProvider Tests
# =============================================================================


class TestOIDCSecretProvider:
    """Tests for OIDCSecretProvider."""

    def test_name(self):
        """Test provider name."""
        cred_provider = OIDCCredentialProvider(
            oidc_provider=MockOIDCProvider(),
            token_exchanger=MockTokenExchanger(),
        )
        secret_provider = OIDCSecretProvider(
            cloud_provider="aws",
            credential_provider=cred_provider,
        )

        assert secret_provider.name == "oidc-aws"

    def test_name_gcp(self):
        """Test provider name for GCP."""
        secret_provider = OIDCSecretProvider(
            cloud_provider="gcp",
            credential_provider=OIDCCredentialProvider(
                oidc_provider=MockOIDCProvider(),
                token_exchanger=MockTokenExchanger(cloud=CloudProvider.GCP),
            ),
        )

        assert secret_provider.name == "oidc-gcp"

    def test_supports_key_no_prefix(self):
        """Test supports_key without prefix."""
        secret_provider = OIDCSecretProvider(
            cloud_provider="aws",
            credential_provider=OIDCCredentialProvider(
                oidc_provider=MockOIDCProvider(),
                token_exchanger=MockTokenExchanger(),
            ),
        )

        assert secret_provider.supports_key("any-key") is True

    def test_supports_key_with_prefix(self):
        """Test supports_key with prefix."""
        secret_provider = OIDCSecretProvider(
            cloud_provider="aws",
            prefix="prod/",
            credential_provider=OIDCCredentialProvider(
                oidc_provider=MockOIDCProvider(),
                token_exchanger=MockTokenExchanger(),
            ),
        )

        assert secret_provider.supports_key("prod/secret") is True
        assert secret_provider.supports_key("dev/secret") is False

    def test_get_not_implemented(self):
        """Test that get raises NotImplementedError for each provider."""
        providers = ["aws", "gcp", "azure", "vault"]

        for cloud in providers:
            secret_provider = OIDCSecretProvider(
                cloud_provider=cloud,
                credential_provider=OIDCCredentialProvider(
                    oidc_provider=MockOIDCProvider(),
                    token_exchanger=MockTokenExchanger(cloud=CloudProvider(cloud)),
                ),
            )

            with pytest.raises(NotImplementedError):
                secret_provider.get("my-secret")


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestGetOIDCCredentials:
    """Tests for get_oidc_credentials function."""

    def test_get_oidc_credentials(self):
        """Test get_oidc_credentials function."""
        mock_creds = AWSCredentials(
            access_key_id="AKID",
            secret_access_key="SECRET",
            session_token="TOKEN",
        )

        # Mock the OIDCCredentialProvider class
        with patch(
            "truthound.secrets.oidc.credential_provider.OIDCCredentialProvider"
        ) as MockProvider:
            mock_instance = MagicMock()
            mock_instance.get_credentials.return_value = mock_creds
            MockProvider.return_value = mock_instance

            creds = get_oidc_credentials(
                cloud_provider="aws",
                aws_role_arn="arn:aws:iam::123:role/test",
            )

            assert creds.access_key_id == "AKID"
            assert creds.secret_access_key == "SECRET"
            MockProvider.assert_called_once()


class TestWithOIDCCredentials:
    """Tests for with_oidc_credentials context manager."""

    def test_with_oidc_credentials_sets_env(self):
        """Test that context manager sets environment variables."""
        mock_creds = AWSCredentials(
            access_key_id="TEST_AKID",
            secret_access_key="TEST_SECRET",
            session_token="TEST_TOKEN",
        )

        with patch(
            "truthound.secrets.oidc.credential_provider.get_oidc_credentials",
            return_value=mock_creds,
        ):
            original_env = os.environ.copy()

            with with_oidc_credentials(
                cloud_provider="aws",
                set_environment=True,
            ) as creds:
                assert creds.access_key_id == "TEST_AKID"
                assert os.environ.get("AWS_ACCESS_KEY_ID") == "TEST_AKID"
                assert os.environ.get("AWS_SECRET_ACCESS_KEY") == "TEST_SECRET"
                assert os.environ.get("AWS_SESSION_TOKEN") == "TEST_TOKEN"

            # Verify env is restored
            assert os.environ.get("AWS_ACCESS_KEY_ID") == original_env.get(
                "AWS_ACCESS_KEY_ID"
            )

    def test_with_oidc_credentials_no_env(self):
        """Test context manager without setting environment."""
        mock_creds = AWSCredentials(
            access_key_id="TEST_AKID",
            secret_access_key="TEST_SECRET",
            session_token="TEST_TOKEN",
        )

        with patch(
            "truthound.secrets.oidc.credential_provider.get_oidc_credentials",
            return_value=mock_creds,
        ):
            original_akid = os.environ.get("AWS_ACCESS_KEY_ID")

            with with_oidc_credentials(
                cloud_provider="aws",
                set_environment=False,
            ) as creds:
                assert creds.access_key_id == "TEST_AKID"
                # Environment should not be modified
                assert os.environ.get("AWS_ACCESS_KEY_ID") == original_akid

    def test_with_oidc_credentials_restores_on_exception(self):
        """Test that environment is restored even on exception."""
        mock_creds = AWSCredentials(
            access_key_id="TEST_AKID",
            secret_access_key="TEST_SECRET",
            session_token="TEST_TOKEN",
        )

        with patch(
            "truthound.secrets.oidc.credential_provider.get_oidc_credentials",
            return_value=mock_creds,
        ):
            original_akid = os.environ.get("AWS_ACCESS_KEY_ID")

            with pytest.raises(ValueError):
                with with_oidc_credentials(cloud_provider="aws") as creds:
                    assert os.environ.get("AWS_ACCESS_KEY_ID") == "TEST_AKID"
                    raise ValueError("Test exception")

            # Verify env is restored
            assert os.environ.get("AWS_ACCESS_KEY_ID") == original_akid


class TestOIDCCredentialsDecorator:
    """Tests for oidc_credentials decorator."""

    def test_decorator_injects_credentials(self):
        """Test that decorator injects credentials as first argument."""
        mock_creds = AWSCredentials(
            access_key_id="DECORATED_AKID",
            secret_access_key="SECRET",
            session_token="TOKEN",
        )

        mock_provider_instance = MagicMock()
        mock_provider_instance.get_credentials.return_value = mock_creds

        with patch(
            "truthound.secrets.oidc.credential_provider.OIDCCredentialProvider",
            return_value=mock_provider_instance,
        ):

            @oidc_credentials("aws", aws_role_arn="arn:aws:iam::123:role/test")
            def my_function(credentials, arg1, arg2):
                return credentials.access_key_id, arg1, arg2

            result = my_function("value1", "value2")

            assert result == ("DECORATED_AKID", "value1", "value2")

    def test_decorator_preserves_function_metadata(self):
        """Test that decorator preserves function metadata."""

        @oidc_credentials("aws")
        def documented_function(credentials):
            """This is the docstring."""
            pass

        assert documented_function.__name__ == "documented_function"
        assert documented_function.__doc__ == "This is the docstring."


# =============================================================================
# Token Exchanger Creation Tests
# =============================================================================


class TestTokenExchangerCreation:
    """Tests for automatic token exchanger creation."""

    def test_creates_aws_exchanger(self):
        """Test that AWS exchanger is created from config."""
        provider = OIDCCredentialProvider(
            cloud_provider="aws",
            aws_role_arn="arn:aws:iam::123:role/test",
            aws_session_name="test-session",
            aws_region="us-west-2",
            oidc_provider=MockOIDCProvider(),
        )

        exchanger = provider.token_exchanger

        assert isinstance(exchanger, AWSTokenExchanger)
        assert exchanger._config.role_arn == "arn:aws:iam::123:role/test"
        assert exchanger._config.session_name == "test-session"
        assert exchanger._config.region == "us-west-2"

    def test_creates_gcp_exchanger(self):
        """Test that GCP exchanger is created from config."""
        from truthound.secrets.oidc.exchangers import GCPTokenExchanger

        provider = OIDCCredentialProvider(
            cloud_provider="gcp",
            gcp_project_number="123456789",
            gcp_pool_id="my-pool",
            gcp_provider_id="github",
            gcp_service_account="sa@project.iam.gserviceaccount.com",
            oidc_provider=MockOIDCProvider(),
        )

        exchanger = provider.token_exchanger

        assert isinstance(exchanger, GCPTokenExchanger)
        assert exchanger._config.project_number == "123456789"
        assert exchanger._config.pool_id == "my-pool"

    def test_creates_azure_exchanger(self):
        """Test that Azure exchanger is created from config."""
        from truthound.secrets.oidc.exchangers import AzureTokenExchanger

        provider = OIDCCredentialProvider(
            cloud_provider="azure",
            azure_tenant_id="tenant-id",
            azure_client_id="client-id",
            azure_subscription_id="sub-id",
            oidc_provider=MockOIDCProvider(),
        )

        exchanger = provider.token_exchanger

        assert isinstance(exchanger, AzureTokenExchanger)
        assert exchanger._config.tenant_id == "tenant-id"

    def test_creates_vault_exchanger(self):
        """Test that Vault exchanger is created from config."""
        from truthound.secrets.oidc.exchangers import VaultTokenExchanger

        provider = OIDCCredentialProvider(
            cloud_provider="vault",
            vault_url="https://vault.example.com",
            vault_role="my-role",
            vault_jwt_path="custom-jwt",
            oidc_provider=MockOIDCProvider(),
        )

        exchanger = provider.token_exchanger

        assert isinstance(exchanger, VaultTokenExchanger)
        assert exchanger._config.vault_url == "https://vault.example.com"
        assert exchanger._config.role == "my-role"
