"""Tests for cloud token exchanger implementations.

This module tests token exchange for various cloud providers:
- AWSTokenExchanger
- GCPTokenExchanger
- AzureTokenExchanger
- VaultTokenExchanger
- Factory function
"""

from __future__ import annotations

import base64
import json
import os
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from xml.etree.ElementTree import Element, SubElement, tostring

import pytest

from truthound.secrets.oidc.exchangers import (
    AWSTokenExchanger,
    AWSTokenExchangerConfig,
    GCPTokenExchanger,
    GCPTokenExchangerConfig,
    AzureTokenExchanger,
    AzureTokenExchangerConfig,
    VaultTokenExchanger,
    VaultTokenExchangerConfig,
    VaultCredentials,
    create_token_exchanger,
)
from truthound.secrets.oidc.base import (
    OIDCToken,
    OIDCExchangeError,
    CloudProvider,
    AWSCredentials,
    GCPCredentials,
    AzureCredentials,
)


# =============================================================================
# Test Utilities
# =============================================================================


def create_jwt_token(
    claims: dict,
    header: dict | None = None,
) -> str:
    """Create a test JWT token."""
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


def create_oidc_token(exp_offset: int = 3600) -> OIDCToken:
    """Create a test OIDC token."""
    now = datetime.now()
    claims = {
        "iss": "https://token.actions.githubusercontent.com",
        "sub": "repo:owner/repo:ref:refs/heads/main",
        "aud": "sts.amazonaws.com",
        "exp": int((now + timedelta(seconds=exp_offset)).timestamp()),
        "iat": int(now.timestamp()),
        "repository": "owner/repo",
    }
    jwt = create_jwt_token(claims)
    return OIDCToken(jwt, provider="test")


def create_sts_response(
    access_key_id: str = "AKIAIOSFODNN7EXAMPLE",
    secret_access_key: str = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    session_token: str = "session-token",
    expiration: datetime | None = None,
    role_arn: str = "arn:aws:sts::123456789012:assumed-role/role/session",
) -> bytes:
    """Create a mock STS AssumeRoleWithWebIdentity response."""
    if expiration is None:
        # Use naive datetime - the code will store this as-is
        expiration = datetime.now() + timedelta(hours=1)

    # Build XML response
    root = Element("AssumeRoleWithWebIdentityResponse")
    root.set("xmlns", "https://sts.amazonaws.com/doc/2011-06-15/")

    result = SubElement(root, "AssumeRoleWithWebIdentityResult")

    creds = SubElement(result, "Credentials")
    SubElement(creds, "AccessKeyId").text = access_key_id
    SubElement(creds, "SecretAccessKey").text = secret_access_key
    SubElement(creds, "SessionToken").text = session_token
    # Use isoformat without Z suffix to keep the datetime naive when parsed
    SubElement(creds, "Expiration").text = expiration.isoformat()

    assumed_role = SubElement(result, "AssumedRoleUser")
    SubElement(assumed_role, "Arn").text = role_arn
    SubElement(assumed_role, "AssumedRoleId").text = "AROA123:session"

    return tostring(root, encoding="utf-8")


# =============================================================================
# AWS Token Exchanger Tests
# =============================================================================


class TestAWSTokenExchanger:
    """Tests for AWSTokenExchanger."""

    def test_cloud_provider(self):
        """Test cloud provider property."""
        exchanger = AWSTokenExchanger(role_arn="arn:aws:iam::123:role/test")
        assert exchanger.cloud_provider == CloudProvider.AWS

    def test_config_from_init(self):
        """Test configuration from init parameters."""
        exchanger = AWSTokenExchanger(
            role_arn="arn:aws:iam::123456789012:role/my-role",
            session_name="my-session",
            session_duration_seconds=7200,
            region="us-west-2",
        )

        assert exchanger._config.role_arn == "arn:aws:iam::123456789012:role/my-role"
        assert exchanger._config.session_name == "my-session"
        assert exchanger._config.session_duration_seconds == 7200
        assert exchanger._config.region == "us-west-2"

    def test_config_from_env(self):
        """Test configuration from environment variables."""
        with patch.dict(
            os.environ,
            {
                "AWS_ROLE_ARN": "arn:aws:iam::999:role/env-role",
                "AWS_REGION": "eu-west-1",
            },
            clear=True,
        ):
            exchanger = AWSTokenExchanger()

            assert exchanger._config.role_arn == "arn:aws:iam::999:role/env-role"
            assert exchanger._config.region == "eu-west-1"

    def test_config_object(self):
        """Test configuration from config object."""
        config = AWSTokenExchangerConfig(
            role_arn="arn:aws:iam::123:role/config-role",
            region="ap-south-1",
            sts_endpoint="https://sts.custom.amazonaws.com",
        )
        exchanger = AWSTokenExchanger(config=config)

        assert exchanger._config.role_arn == "arn:aws:iam::123:role/config-role"
        assert exchanger._config.sts_endpoint == "https://sts.custom.amazonaws.com"

    def test_get_sts_endpoint_default(self):
        """Test default STS endpoint generation."""
        exchanger = AWSTokenExchanger(
            role_arn="arn:aws:iam::123:role/test",
            region="us-west-2",
        )

        endpoint = exchanger._get_sts_endpoint()
        assert endpoint == "https://sts.us-west-2.amazonaws.com/"

    def test_get_sts_endpoint_custom(self):
        """Test custom STS endpoint."""
        config = AWSTokenExchangerConfig(
            role_arn="arn:aws:iam::123:role/test",
            sts_endpoint="https://vpce-xxx.sts.amazonaws.com",
        )
        exchanger = AWSTokenExchanger(config=config)

        endpoint = exchanger._get_sts_endpoint()
        assert endpoint == "https://vpce-xxx.sts.amazonaws.com"

    def test_exchange_success(self):
        """Test successful token exchange."""
        token = create_oidc_token()
        sts_response = create_sts_response()

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = sts_response
            mock_response.__enter__ = lambda s: s
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            exchanger = AWSTokenExchanger(
                role_arn="arn:aws:iam::123456789012:role/my-role",
            )
            creds = exchanger.exchange(token)

            assert isinstance(creds, AWSCredentials)
            assert creds.access_key_id == "AKIAIOSFODNN7EXAMPLE"
            assert creds.secret_access_key == "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
            assert creds.session_token == "session-token"

    def test_exchange_missing_role_arn(self):
        """Test exchange fails when role ARN is missing."""
        token = create_oidc_token()

        with patch.dict(os.environ, {}, clear=True):
            exchanger = AWSTokenExchanger()

            with pytest.raises(OIDCExchangeError) as exc_info:
                exchanger.exchange(token)

            assert "role_arn is required" in str(exc_info.value)

    def test_exchange_http_error(self):
        """Test exchange handles HTTP errors."""
        token = create_oidc_token()

        with patch("urllib.request.urlopen") as mock_urlopen:
            import urllib.error

            mock_urlopen.side_effect = urllib.error.HTTPError(
                url="https://sts.amazonaws.com",
                code=403,
                msg="Forbidden",
                hdrs={},
                fp=MagicMock(read=lambda: b'{"error": "access_denied"}'),
            )

            exchanger = AWSTokenExchanger(
                role_arn="arn:aws:iam::123:role/test",
            )

            with pytest.raises(OIDCExchangeError) as exc_info:
                exchanger.exchange(token)

            assert exc_info.value.cloud_provider == "aws"
            assert exc_info.value.status_code == 403

    def test_credential_caching(self):
        """Test that credentials are cached."""
        # Create a single token to reuse (same hash)
        token = create_oidc_token()
        sts_response = create_sts_response()
        call_count = 0

        with patch("urllib.request.urlopen") as mock_urlopen:

            def side_effect(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                mock_response = MagicMock()
                mock_response.read.return_value = sts_response
                mock_response.__enter__ = lambda s: s
                mock_response.__exit__ = MagicMock(return_value=False)
                return mock_response

            mock_urlopen.side_effect = side_effect

            exchanger = AWSTokenExchanger(
                role_arn="arn:aws:iam::123:role/test",
                enable_cache=True,
            )

            # First exchange - makes HTTP call
            creds1 = exchanger.exchange(token)
            first_call_count = call_count
            assert first_call_count >= 1

            # Second exchange with SAME token instance - should use cache
            # Cache key is token.hash, so we need the same token object
            creds2 = exchanger.exchange(token)
            # Call count should not increase since same token is cached
            assert call_count == first_call_count

            # Clear cache and exchange again - makes new HTTP call
            exchanger.clear_cache()
            creds3 = exchanger.exchange(token)
            assert call_count > first_call_count


# =============================================================================
# GCP Token Exchanger Tests
# =============================================================================


class TestGCPTokenExchanger:
    """Tests for GCPTokenExchanger."""

    def test_cloud_provider(self):
        """Test cloud provider property."""
        exchanger = GCPTokenExchanger(
            project_number="123",
            pool_id="pool",
            provider_id="provider",
            service_account_email="sa@project.iam.gserviceaccount.com",
        )
        assert exchanger.cloud_provider == CloudProvider.GCP

    def test_config_from_init(self):
        """Test configuration from init parameters."""
        exchanger = GCPTokenExchanger(
            project_number="123456789",
            pool_id="my-pool",
            provider_id="github",
            service_account_email="sa@project.iam.gserviceaccount.com",
        )

        assert exchanger._config.project_number == "123456789"
        assert exchanger._config.pool_id == "my-pool"
        assert exchanger._config.provider_id == "github"
        assert exchanger._config.service_account_email == "sa@project.iam.gserviceaccount.com"

    def test_get_audience(self):
        """Test audience URL generation."""
        exchanger = GCPTokenExchanger(
            project_number="123456789",
            pool_id="my-pool",
            provider_id="github",
            service_account_email="sa@project.iam.gserviceaccount.com",
        )

        audience = exchanger._get_audience()
        expected = (
            "//iam.googleapis.com/projects/123456789/"
            "locations/global/workloadIdentityPools/my-pool/"
            "providers/github"
        )
        assert audience == expected

    def test_exchange_missing_config(self):
        """Test exchange fails when config is incomplete."""
        token = create_oidc_token()
        exchanger = GCPTokenExchanger(project_number="123")  # Missing other fields

        with pytest.raises(OIDCExchangeError) as exc_info:
            exchanger.exchange(token)

        assert "required" in str(exc_info.value)

    def test_exchange_success(self):
        """Test successful token exchange."""
        token = create_oidc_token()

        # Mock responses for both API calls
        sts_response = {"access_token": "federated-token"}
        sa_response = {
            "accessToken": "final-access-token",
            "expireTime": (datetime.now() + timedelta(hours=1)).isoformat() + "Z",
        }

        call_count = 0

        with patch("urllib.request.urlopen") as mock_urlopen:

            def side_effect(request, **kwargs):
                nonlocal call_count
                call_count += 1
                mock_response = MagicMock()
                if call_count == 1:
                    mock_response.read.return_value = json.dumps(sts_response).encode()
                else:
                    mock_response.read.return_value = json.dumps(sa_response).encode()
                mock_response.__enter__ = lambda s: s
                mock_response.__exit__ = MagicMock(return_value=False)
                return mock_response

            mock_urlopen.side_effect = side_effect

            exchanger = GCPTokenExchanger(
                project_number="123456789",
                pool_id="my-pool",
                provider_id="github",
                service_account_email="sa@my-project.iam.gserviceaccount.com",
            )

            creds = exchanger.exchange(token)

            assert isinstance(creds, GCPCredentials)
            assert creds.access_token == "final-access-token"
            assert creds.service_account == "sa@my-project.iam.gserviceaccount.com"
            assert creds.project_id == "my-project"
            assert call_count == 2  # STS + SA impersonation


# =============================================================================
# Azure Token Exchanger Tests
# =============================================================================


class TestAzureTokenExchanger:
    """Tests for AzureTokenExchanger."""

    def test_cloud_provider(self):
        """Test cloud provider property."""
        exchanger = AzureTokenExchanger(
            tenant_id="tenant",
            client_id="client",
        )
        assert exchanger.cloud_provider == CloudProvider.AZURE

    def test_config_from_init(self):
        """Test configuration from init parameters."""
        exchanger = AzureTokenExchanger(
            tenant_id="12345678-1234-1234-1234-123456789012",
            client_id="87654321-4321-4321-4321-210987654321",
            subscription_id="abcdef12-3456-7890-abcd-ef1234567890",
            scope="https://storage.azure.com/.default",
        )

        assert exchanger._config.tenant_id == "12345678-1234-1234-1234-123456789012"
        assert exchanger._config.client_id == "87654321-4321-4321-4321-210987654321"
        assert exchanger._config.scope == "https://storage.azure.com/.default"

    def test_config_from_env(self):
        """Test configuration from environment variables."""
        with patch.dict(
            os.environ,
            {
                "AZURE_TENANT_ID": "env-tenant",
                "AZURE_CLIENT_ID": "env-client",
                "AZURE_SUBSCRIPTION_ID": "env-sub",
            },
            clear=True,
        ):
            exchanger = AzureTokenExchanger()

            assert exchanger._config.tenant_id == "env-tenant"
            assert exchanger._config.client_id == "env-client"
            assert exchanger._config.subscription_id == "env-sub"

    def test_exchange_missing_config(self):
        """Test exchange fails when config is incomplete."""
        token = create_oidc_token()

        with patch.dict(os.environ, {}, clear=True):
            exchanger = AzureTokenExchanger(tenant_id="tenant")  # Missing client_id

            with pytest.raises(OIDCExchangeError) as exc_info:
                exchanger.exchange(token)

            assert "required" in str(exc_info.value)

    def test_exchange_success(self):
        """Test successful token exchange."""
        token = create_oidc_token()

        azure_response = {
            "access_token": "azure-access-token",
            "token_type": "Bearer",
            "expires_in": 3600,
        }

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = json.dumps(azure_response).encode()
            mock_response.__enter__ = lambda s: s
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            exchanger = AzureTokenExchanger(
                tenant_id="tenant-id",
                client_id="client-id",
                subscription_id="sub-id",
            )

            creds = exchanger.exchange(token)

            assert isinstance(creds, AzureCredentials)
            assert creds.access_token == "azure-access-token"
            assert creds.tenant_id == "tenant-id"
            assert creds.client_id == "client-id"


# =============================================================================
# Vault Token Exchanger Tests
# =============================================================================


class TestVaultTokenExchanger:
    """Tests for VaultTokenExchanger."""

    def test_cloud_provider(self):
        """Test cloud provider property."""
        exchanger = VaultTokenExchanger(
            vault_url="https://vault.example.com",
            role="my-role",
        )
        assert exchanger.cloud_provider == CloudProvider.VAULT

    def test_config_from_init(self):
        """Test configuration from init parameters."""
        exchanger = VaultTokenExchanger(
            vault_url="https://vault.example.com",
            role="github-role",
            jwt_auth_path="custom-jwt",
            namespace="my-namespace",
        )

        assert exchanger._config.vault_url == "https://vault.example.com"
        assert exchanger._config.role == "github-role"
        assert exchanger._config.jwt_auth_path == "custom-jwt"
        assert exchanger._config.namespace == "my-namespace"

    def test_config_from_env(self):
        """Test configuration from environment variables."""
        with patch.dict(
            os.environ,
            {
                "VAULT_ADDR": "https://env-vault.example.com",
                "VAULT_NAMESPACE": "env-namespace",
            },
            clear=True,
        ):
            exchanger = VaultTokenExchanger(role="my-role")

            assert exchanger._config.vault_url == "https://env-vault.example.com"
            assert exchanger._config.namespace == "env-namespace"

    def test_exchange_missing_config(self):
        """Test exchange fails when config is incomplete."""
        token = create_oidc_token()

        with patch.dict(os.environ, {}, clear=True):
            exchanger = VaultTokenExchanger()  # Missing vault_url and role

            with pytest.raises(OIDCExchangeError) as exc_info:
                exchanger.exchange(token)

            assert "required" in str(exc_info.value)

    def test_exchange_success(self):
        """Test successful token exchange."""
        token = create_oidc_token()

        vault_response = {
            "auth": {
                "client_token": "hvs.example-token",
                "accessor": "accessor-123",
                "policies": ["default", "my-policy"],
                "renewable": True,
                "lease_duration": 3600,
            }
        }

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = json.dumps(vault_response).encode()
            mock_response.__enter__ = lambda s: s
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            exchanger = VaultTokenExchanger(
                vault_url="https://vault.example.com",
                role="github-role",
            )

            creds = exchanger.exchange(token)

            assert isinstance(creds, VaultCredentials)
            assert creds.client_token == "hvs.example-token"
            assert creds.accessor == "accessor-123"
            assert creds.policies == ["default", "my-policy"]
            assert creds.renewable is True
            assert creds.lease_duration == 3600

    def test_vault_credentials_authorization_header(self):
        """Test Vault credentials authorization header."""
        creds = VaultCredentials(
            client_token="hvs.token",
            accessor="accessor",
        )

        header = creds.get_authorization_header()
        assert header == {"X-Vault-Token": "hvs.token"}


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateTokenExchanger:
    """Tests for create_token_exchanger factory function."""

    def test_create_aws_exchanger(self):
        """Test creating AWS token exchanger."""
        exchanger = create_token_exchanger(
            "aws",
            role_arn="arn:aws:iam::123:role/test",
        )

        assert isinstance(exchanger, AWSTokenExchanger)
        assert exchanger.cloud_provider == CloudProvider.AWS

    def test_create_gcp_exchanger(self):
        """Test creating GCP token exchanger."""
        exchanger = create_token_exchanger(
            "gcp",
            project_number="123",
            pool_id="pool",
            provider_id="provider",
            service_account_email="sa@project.iam.gserviceaccount.com",
        )

        assert isinstance(exchanger, GCPTokenExchanger)
        assert exchanger.cloud_provider == CloudProvider.GCP

    def test_create_azure_exchanger(self):
        """Test creating Azure token exchanger."""
        exchanger = create_token_exchanger(
            "azure",
            tenant_id="tenant",
            client_id="client",
        )

        assert isinstance(exchanger, AzureTokenExchanger)
        assert exchanger.cloud_provider == CloudProvider.AZURE

    def test_create_vault_exchanger(self):
        """Test creating Vault token exchanger."""
        exchanger = create_token_exchanger(
            "vault",
            vault_url="https://vault.example.com",
            role="my-role",
        )

        assert isinstance(exchanger, VaultTokenExchanger)
        assert exchanger.cloud_provider == CloudProvider.VAULT

    def test_create_with_cloud_provider_enum(self):
        """Test creating exchanger with CloudProvider enum."""
        exchanger = create_token_exchanger(
            CloudProvider.AWS,
            role_arn="arn:aws:iam::123:role/test",
        )

        assert isinstance(exchanger, AWSTokenExchanger)

    def test_create_unsupported_provider(self):
        """Test error for unsupported provider."""
        with pytest.raises(ValueError) as exc_info:
            create_token_exchanger("unsupported")

        # Error message contains either "Unsupported" or "not a valid"
        error_msg = str(exc_info.value).lower()
        assert "unsupported" in error_msg or "not a valid" in error_msg
