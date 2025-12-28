"""Cloud Token Exchanger Implementations.

This module provides token exchange implementations for various cloud providers:
- AWS STS (AssumeRoleWithWebIdentity)
- Google Cloud Workload Identity
- Azure Federated Credentials
- HashiCorp Vault JWT Auth

Each exchanger converts an OIDC token to cloud-specific credentials.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any
from xml.etree import ElementTree

from truthound.secrets.oidc.base import (
    AWSCredentials,
    AzureCredentials,
    BaseTokenExchanger,
    CloudCredentials,
    CloudProvider,
    GCPCredentials,
    OIDCExchangeError,
    OIDCToken,
)

if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


# =============================================================================
# AWS Token Exchanger
# =============================================================================


@dataclass
class AWSTokenExchangerConfig:
    """Configuration for AWS token exchange.

    Attributes:
        role_arn: ARN of the IAM role to assume.
        session_name: Session name for the assumed role.
        session_duration_seconds: How long credentials should be valid.
        region: AWS region for STS endpoint.
        sts_endpoint: Custom STS endpoint (optional).
        request_timeout: HTTP request timeout.
    """

    role_arn: str = ""
    session_name: str = "truthound-oidc"
    session_duration_seconds: int = 3600
    region: str = "us-east-1"
    sts_endpoint: str | None = None
    request_timeout: float = 30.0


class AWSTokenExchanger(BaseTokenExchanger):
    """AWS STS token exchanger using AssumeRoleWithWebIdentity.

    Exchanges an OIDC token for temporary AWS credentials by calling
    the AWS STS AssumeRoleWithWebIdentity API.

    Requirements:
        - IAM role must have a trust policy allowing the OIDC provider
        - Role ARN must be configured

    Example:
        >>> exchanger = AWSTokenExchanger(
        ...     role_arn="arn:aws:iam::123456789012:role/my-role",
        ...     session_name="my-session",
        ... )
        >>> credentials = exchanger.exchange(oidc_token)
        >>> print(credentials.access_key_id)

    Trust Policy Example:
        ```json
        {
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": {
                    "Federated": "arn:aws:iam::123456789012:oidc-provider/token.actions.githubusercontent.com"
                },
                "Action": "sts:AssumeRoleWithWebIdentity",
                "Condition": {
                    "StringEquals": {
                        "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
                    },
                    "StringLike": {
                        "token.actions.githubusercontent.com:sub": "repo:owner/repo:*"
                    }
                }
            }]
        }
        ```
    """

    # Default STS endpoint
    STS_ENDPOINT_TEMPLATE = "https://sts.{region}.amazonaws.com/"

    def __init__(
        self,
        role_arn: str | None = None,
        *,
        session_name: str = "truthound-oidc",
        session_duration_seconds: int = 3600,
        region: str | None = None,
        config: AWSTokenExchangerConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize AWS token exchanger.

        Args:
            role_arn: IAM role ARN (or use AWS_ROLE_ARN env var).
            session_name: Session name.
            session_duration_seconds: Credential validity duration.
            region: AWS region (or use AWS_REGION env var).
            config: Full configuration object.
            **kwargs: Additional base class arguments.
        """
        self._config = config or AWSTokenExchangerConfig()

        # Override with explicit parameters
        if role_arn:
            self._config.role_arn = role_arn
        elif not self._config.role_arn:
            self._config.role_arn = os.environ.get("AWS_ROLE_ARN", "")

        self._config.session_name = session_name
        self._config.session_duration_seconds = session_duration_seconds

        if region:
            self._config.region = region
        elif not self._config.region or self._config.region == "us-east-1":
            self._config.region = os.environ.get("AWS_REGION", "us-east-1")

        super().__init__(**kwargs)

    @property
    def cloud_provider(self) -> CloudProvider:
        return CloudProvider.AWS

    def _get_sts_endpoint(self) -> str:
        """Get the STS endpoint URL."""
        if self._config.sts_endpoint:
            return self._config.sts_endpoint
        return self.STS_ENDPOINT_TEMPLATE.format(region=self._config.region)

    def _exchange(self, token: OIDCToken) -> AWSCredentials:
        """Exchange OIDC token for AWS credentials.

        Args:
            token: OIDC token to exchange.

        Returns:
            AWS credentials.

        Raises:
            OIDCExchangeError: If exchange fails.
        """
        if not self._config.role_arn:
            raise OIDCExchangeError(
                "role_arn is required",
                cloud_provider="aws",
            )

        # Build STS request parameters
        params = {
            "Action": "AssumeRoleWithWebIdentity",
            "Version": "2011-06-15",
            "RoleArn": self._config.role_arn,
            "RoleSessionName": self._config.session_name,
            "WebIdentityToken": token.get_token(),
            "DurationSeconds": str(self._config.session_duration_seconds),
        }

        endpoint = self._get_sts_endpoint()
        url = f"{endpoint}?{urllib.parse.urlencode(params)}"

        request = urllib.request.Request(
            url,
            method="POST",
            headers={"Accept": "application/xml"},
        )

        try:
            with urllib.request.urlopen(
                request, timeout=self._config.request_timeout
            ) as response:
                return self._parse_sts_response(response.read())

        except urllib.error.HTTPError as e:
            error_body = ""
            try:
                error_body = e.read().decode()
            except Exception:
                pass
            raise OIDCExchangeError(
                f"STS API error: {error_body or e.reason}",
                cloud_provider="aws",
                status_code=e.code,
                response=error_body,
            ) from e
        except urllib.error.URLError as e:
            raise OIDCExchangeError(
                f"Network error: {e.reason}",
                cloud_provider="aws",
            ) from e

    def _parse_sts_response(self, response_xml: bytes) -> AWSCredentials:
        """Parse STS AssumeRoleWithWebIdentity response.

        Args:
            response_xml: XML response from STS.

        Returns:
            AWS credentials.
        """
        try:
            # Parse XML response
            root = ElementTree.fromstring(response_xml)

            # Find credentials in response (namespace handling)
            ns = {"sts": "https://sts.amazonaws.com/doc/2011-06-15/"}

            # Try with namespace first, then without
            creds = root.find(".//sts:Credentials", ns)
            if creds is None:
                creds = root.find(".//Credentials")

            if creds is None:
                raise OIDCExchangeError(
                    "Credentials not found in STS response",
                    cloud_provider="aws",
                )

            # Extract credential values
            def get_text(elem: ElementTree.Element | None, tag: str) -> str:
                child = elem.find(f"sts:{tag}", ns) if elem else None
                if child is None and elem is not None:
                    child = elem.find(tag)
                return child.text if child is not None and child.text else ""

            access_key_id = get_text(creds, "AccessKeyId")
            secret_access_key = get_text(creds, "SecretAccessKey")
            session_token = get_text(creds, "SessionToken")
            expiration_str = get_text(creds, "Expiration")

            # Parse expiration time
            expires_at = None
            if expiration_str:
                try:
                    # Handle ISO format with Z suffix
                    if expiration_str.endswith("Z"):
                        expiration_str = expiration_str[:-1] + "+00:00"
                    expires_at = datetime.fromisoformat(expiration_str)
                except ValueError:
                    pass

            # Get assumed role ARN
            assumed_role = root.find(".//sts:AssumedRoleUser", ns)
            if assumed_role is None:
                assumed_role = root.find(".//AssumedRoleUser")

            assumed_role_arn = ""
            if assumed_role is not None:
                arn_elem = assumed_role.find("sts:Arn", ns)
                if arn_elem is None:
                    arn_elem = assumed_role.find("Arn")
                if arn_elem is not None and arn_elem.text:
                    assumed_role_arn = arn_elem.text

            return AWSCredentials(
                access_key_id=access_key_id,
                secret_access_key=secret_access_key,
                session_token=session_token,
                assumed_role_arn=assumed_role_arn,
                expires_at=expires_at,
            )

        except ElementTree.ParseError as e:
            raise OIDCExchangeError(
                f"Invalid STS response: {e}",
                cloud_provider="aws",
            ) from e


# =============================================================================
# GCP Token Exchanger
# =============================================================================


@dataclass
class GCPTokenExchangerConfig:
    """Configuration for GCP token exchange.

    Attributes:
        project_number: GCP project number.
        pool_id: Workload Identity Pool ID.
        provider_id: Workload Identity Provider ID.
        service_account_email: Service account to impersonate.
        token_lifetime_seconds: Access token lifetime.
        request_timeout: HTTP request timeout.
    """

    project_number: str = ""
    pool_id: str = ""
    provider_id: str = ""
    service_account_email: str = ""
    token_lifetime_seconds: int = 3600
    request_timeout: float = 30.0


class GCPTokenExchanger(BaseTokenExchanger):
    """GCP Workload Identity token exchanger.

    Exchanges an OIDC token for GCP access token through:
    1. STS Token Exchange - exchange OIDC for federated token
    2. Service Account Impersonation - get access token for SA

    Requirements:
        - Workload Identity Pool configured
        - OIDC provider registered in the pool
        - Service account with appropriate permissions

    Example:
        >>> exchanger = GCPTokenExchanger(
        ...     project_number="123456789",
        ...     pool_id="my-pool",
        ...     provider_id="github",
        ...     service_account_email="sa@project.iam.gserviceaccount.com",
        ... )
        >>> credentials = exchanger.exchange(oidc_token)
        >>> print(credentials.access_token)

    Configuration in GCP:
        1. Create Workload Identity Pool
        2. Add OIDC provider (e.g., GitHub Actions)
        3. Grant service account impersonation permission
    """

    # GCP endpoints
    STS_EXCHANGE_URL = "https://sts.googleapis.com/v1/token"
    SA_IMPERSONATE_URL = (
        "https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/"
        "{service_account}:generateAccessToken"
    )

    def __init__(
        self,
        project_number: str | None = None,
        pool_id: str | None = None,
        provider_id: str | None = None,
        service_account_email: str | None = None,
        *,
        config: GCPTokenExchangerConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize GCP token exchanger.

        Args:
            project_number: GCP project number.
            pool_id: Workload Identity Pool ID.
            provider_id: Workload Identity Provider ID.
            service_account_email: Service account to impersonate.
            config: Full configuration object.
            **kwargs: Additional base class arguments.
        """
        self._config = config or GCPTokenExchangerConfig()

        # Override with explicit parameters
        if project_number:
            self._config.project_number = project_number
        if pool_id:
            self._config.pool_id = pool_id
        if provider_id:
            self._config.provider_id = provider_id
        if service_account_email:
            self._config.service_account_email = service_account_email

        super().__init__(**kwargs)

    @property
    def cloud_provider(self) -> CloudProvider:
        return CloudProvider.GCP

    def _get_audience(self) -> str:
        """Get the Workload Identity audience URL."""
        return (
            f"//iam.googleapis.com/projects/{self._config.project_number}/"
            f"locations/global/workloadIdentityPools/{self._config.pool_id}/"
            f"providers/{self._config.provider_id}"
        )

    def _exchange(self, token: OIDCToken) -> GCPCredentials:
        """Exchange OIDC token for GCP credentials.

        Args:
            token: OIDC token to exchange.

        Returns:
            GCP credentials.

        Raises:
            OIDCExchangeError: If exchange fails.
        """
        if not all([
            self._config.project_number,
            self._config.pool_id,
            self._config.provider_id,
            self._config.service_account_email,
        ]):
            raise OIDCExchangeError(
                "project_number, pool_id, provider_id, and "
                "service_account_email are required",
                cloud_provider="gcp",
            )

        # Step 1: Exchange OIDC token for federated token
        federated_token = self._exchange_for_federated_token(token)

        # Step 2: Impersonate service account
        return self._impersonate_service_account(federated_token)

    def _exchange_for_federated_token(self, token: OIDCToken) -> str:
        """Exchange OIDC token for GCP federated token.

        Args:
            token: OIDC token.

        Returns:
            Federated access token.
        """
        audience = self._get_audience()

        request_body = {
            "grantType": "urn:ietf:params:oauth:grant-type:token-exchange",
            "audience": audience,
            "scope": "https://www.googleapis.com/auth/cloud-platform",
            "requestedTokenType": "urn:ietf:params:oauth:token-type:access_token",
            "subjectTokenType": "urn:ietf:params:oauth:token-type:jwt",
            "subjectToken": token.get_token(),
        }

        request = urllib.request.Request(
            self.STS_EXCHANGE_URL,
            data=json.dumps(request_body).encode(),
            headers={
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request, timeout=self._config.request_timeout
            ) as response:
                data = json.loads(response.read())
                return data["access_token"]

        except urllib.error.HTTPError as e:
            error_body = ""
            try:
                error_body = e.read().decode()
            except Exception:
                pass
            raise OIDCExchangeError(
                f"GCP STS exchange failed: {error_body or e.reason}",
                cloud_provider="gcp",
                status_code=e.code,
                response=error_body,
            ) from e
        except KeyError:
            raise OIDCExchangeError(
                "access_token not found in STS response",
                cloud_provider="gcp",
            )

    def _impersonate_service_account(self, federated_token: str) -> GCPCredentials:
        """Impersonate service account to get access token.

        Args:
            federated_token: Federated token from STS exchange.

        Returns:
            GCP credentials with access token.
        """
        url = self.SA_IMPERSONATE_URL.format(
            service_account=urllib.parse.quote(
                self._config.service_account_email, safe=""
            )
        )

        request_body = {
            "scope": ["https://www.googleapis.com/auth/cloud-platform"],
            "lifetime": f"{self._config.token_lifetime_seconds}s",
        }

        request = urllib.request.Request(
            url,
            data=json.dumps(request_body).encode(),
            headers={
                "Authorization": f"Bearer {federated_token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request, timeout=self._config.request_timeout
            ) as response:
                data = json.loads(response.read())

                # Parse expiration time
                expires_at = None
                expire_time = data.get("expireTime")
                if expire_time:
                    try:
                        if expire_time.endswith("Z"):
                            expire_time = expire_time[:-1] + "+00:00"
                        expires_at = datetime.fromisoformat(expire_time)
                    except ValueError:
                        pass

                # Extract project ID from service account email
                parts = self._config.service_account_email.split("@")
                project_id = ""
                if len(parts) == 2:
                    project_parts = parts[1].split(".")
                    if project_parts:
                        project_id = project_parts[0]

                return GCPCredentials(
                    access_token=data["accessToken"],
                    service_account=self._config.service_account_email,
                    project_id=project_id,
                    expires_at=expires_at,
                )

        except urllib.error.HTTPError as e:
            error_body = ""
            try:
                error_body = e.read().decode()
            except Exception:
                pass
            raise OIDCExchangeError(
                f"Service account impersonation failed: {error_body or e.reason}",
                cloud_provider="gcp",
                status_code=e.code,
                response=error_body,
            ) from e


# =============================================================================
# Azure Token Exchanger
# =============================================================================


@dataclass
class AzureTokenExchangerConfig:
    """Configuration for Azure token exchange.

    Attributes:
        tenant_id: Azure tenant ID.
        client_id: Azure client/app ID.
        subscription_id: Azure subscription ID.
        scope: Token scope (default: Azure management).
        request_timeout: HTTP request timeout.
    """

    tenant_id: str = ""
    client_id: str = ""
    subscription_id: str = ""
    scope: str = "https://management.azure.com/.default"
    request_timeout: float = 30.0


class AzureTokenExchanger(BaseTokenExchanger):
    """Azure federated credentials token exchanger.

    Exchanges an OIDC token for Azure access token using the
    client credentials flow with federated identity.

    Requirements:
        - App registration with federated credential configured
        - OIDC issuer and subject claim mapping

    Example:
        >>> exchanger = AzureTokenExchanger(
        ...     tenant_id="12345678-...",
        ...     client_id="87654321-...",
        ... )
        >>> credentials = exchanger.exchange(oidc_token)
        >>> print(credentials.access_token)

    Configuration in Azure:
        1. Create App Registration
        2. Add Federated Credential with OIDC issuer
        3. Configure subject claim matching
    """

    # Azure OAuth2 endpoint template
    TOKEN_URL_TEMPLATE = (
        "https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
    )

    def __init__(
        self,
        tenant_id: str | None = None,
        client_id: str | None = None,
        *,
        subscription_id: str | None = None,
        scope: str | None = None,
        config: AzureTokenExchangerConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize Azure token exchanger.

        Args:
            tenant_id: Azure tenant ID (or AZURE_TENANT_ID env var).
            client_id: Azure client ID (or AZURE_CLIENT_ID env var).
            subscription_id: Azure subscription ID.
            scope: Token scope.
            config: Full configuration object.
            **kwargs: Additional base class arguments.
        """
        self._config = config or AzureTokenExchangerConfig()

        # Override with explicit parameters or env vars
        if tenant_id:
            self._config.tenant_id = tenant_id
        elif not self._config.tenant_id:
            self._config.tenant_id = os.environ.get("AZURE_TENANT_ID", "")

        if client_id:
            self._config.client_id = client_id
        elif not self._config.client_id:
            self._config.client_id = os.environ.get("AZURE_CLIENT_ID", "")

        if subscription_id:
            self._config.subscription_id = subscription_id
        elif not self._config.subscription_id:
            self._config.subscription_id = os.environ.get("AZURE_SUBSCRIPTION_ID", "")

        if scope:
            self._config.scope = scope

        super().__init__(**kwargs)

    @property
    def cloud_provider(self) -> CloudProvider:
        return CloudProvider.AZURE

    def _exchange(self, token: OIDCToken) -> AzureCredentials:
        """Exchange OIDC token for Azure credentials.

        Args:
            token: OIDC token to exchange.

        Returns:
            Azure credentials.

        Raises:
            OIDCExchangeError: If exchange fails.
        """
        if not self._config.tenant_id or not self._config.client_id:
            raise OIDCExchangeError(
                "tenant_id and client_id are required",
                cloud_provider="azure",
            )

        url = self.TOKEN_URL_TEMPLATE.format(tenant_id=self._config.tenant_id)

        # Build form data for token request
        form_data = {
            "client_id": self._config.client_id,
            "client_assertion_type": (
                "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"
            ),
            "client_assertion": token.get_token(),
            "grant_type": "client_credentials",
            "scope": self._config.scope,
        }

        request = urllib.request.Request(
            url,
            data=urllib.parse.urlencode(form_data).encode(),
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request, timeout=self._config.request_timeout
            ) as response:
                data = json.loads(response.read())

                # Calculate expiration time
                expires_at = None
                expires_in = data.get("expires_in")
                if expires_in:
                    expires_at = datetime.now() + timedelta(seconds=int(expires_in))

                return AzureCredentials(
                    access_token=data["access_token"],
                    token_type=data.get("token_type", "Bearer"),
                    tenant_id=self._config.tenant_id,
                    client_id=self._config.client_id,
                    subscription_id=self._config.subscription_id,
                    expires_at=expires_at,
                )

        except urllib.error.HTTPError as e:
            error_body = ""
            try:
                error_body = e.read().decode()
            except Exception:
                pass
            raise OIDCExchangeError(
                f"Azure token exchange failed: {error_body or e.reason}",
                cloud_provider="azure",
                status_code=e.code,
                response=error_body,
            ) from e
        except KeyError:
            raise OIDCExchangeError(
                "access_token not found in Azure response",
                cloud_provider="azure",
            )


# =============================================================================
# Vault Token Exchanger
# =============================================================================


@dataclass
class VaultTokenExchangerConfig:
    """Configuration for HashiCorp Vault token exchange.

    Attributes:
        vault_url: Vault server URL.
        jwt_auth_path: Path to JWT auth backend.
        role: Vault role to use.
        namespace: Vault namespace (Enterprise only).
        request_timeout: HTTP request timeout.
    """

    vault_url: str = ""
    jwt_auth_path: str = "jwt"
    role: str = ""
    namespace: str = ""
    request_timeout: float = 30.0


@dataclass
class VaultCredentials(CloudCredentials):
    """HashiCorp Vault credentials.

    Attributes:
        client_token: Vault client token.
        accessor: Token accessor.
        policies: List of applied policies.
        renewable: Whether token is renewable.
        lease_duration: Token lease duration in seconds.
    """

    client_token: str = ""
    accessor: str = ""
    policies: list[str] = field(default_factory=list)
    renewable: bool = False
    lease_duration: int = 0

    def __post_init__(self) -> None:
        self.provider = CloudProvider.VAULT

    def get_authorization_header(self) -> dict[str, str]:
        """Get HTTP authorization header for Vault."""
        return {"X-Vault-Token": self.client_token}

    def __repr__(self) -> str:
        """Safe representation."""
        return (
            f"VaultCredentials(accessor={self.accessor}, "
            f"policies={self.policies})"
        )


class VaultTokenExchanger(BaseTokenExchanger):
    """HashiCorp Vault JWT auth token exchanger.

    Exchanges an OIDC token for Vault client token using the
    JWT authentication method.

    Requirements:
        - JWT auth backend enabled and configured
        - Role configured with appropriate policies

    Example:
        >>> exchanger = VaultTokenExchanger(
        ...     vault_url="https://vault.example.com",
        ...     role="my-github-role",
        ... )
        >>> credentials = exchanger.exchange(oidc_token)
        >>> print(credentials.client_token)

    Vault Configuration:
        ```bash
        vault auth enable jwt

        vault write auth/jwt/config \\
            oidc_discovery_url="https://token.actions.githubusercontent.com" \\
            bound_issuer="https://token.actions.githubusercontent.com"

        vault write auth/jwt/role/my-github-role \\
            role_type="jwt" \\
            bound_audiences="https://vault.example.com" \\
            bound_claims_type="glob" \\
            bound_claims='{"repository":"owner/repo"}' \\
            user_claim="repository" \\
            policies="my-policy" \\
            ttl="1h"
        ```
    """

    def __init__(
        self,
        vault_url: str | None = None,
        role: str | None = None,
        *,
        jwt_auth_path: str = "jwt",
        namespace: str | None = None,
        config: VaultTokenExchangerConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize Vault token exchanger.

        Args:
            vault_url: Vault server URL (or VAULT_ADDR env var).
            role: Vault role name.
            jwt_auth_path: Path to JWT auth backend.
            namespace: Vault namespace.
            config: Full configuration object.
            **kwargs: Additional base class arguments.
        """
        self._config = config or VaultTokenExchangerConfig()

        # Override with explicit parameters or env vars
        if vault_url:
            self._config.vault_url = vault_url
        elif not self._config.vault_url:
            self._config.vault_url = os.environ.get("VAULT_ADDR", "")

        if role:
            self._config.role = role

        self._config.jwt_auth_path = jwt_auth_path

        if namespace:
            self._config.namespace = namespace
        elif not self._config.namespace:
            self._config.namespace = os.environ.get("VAULT_NAMESPACE", "")

        super().__init__(**kwargs)

    @property
    def cloud_provider(self) -> CloudProvider:
        return CloudProvider.VAULT

    def _exchange(self, token: OIDCToken) -> VaultCredentials:
        """Exchange OIDC token for Vault credentials.

        Args:
            token: OIDC token to exchange.

        Returns:
            Vault credentials.

        Raises:
            OIDCExchangeError: If exchange fails.
        """
        if not self._config.vault_url or not self._config.role:
            raise OIDCExchangeError(
                "vault_url and role are required",
                cloud_provider="vault",
            )

        url = (
            f"{self._config.vault_url.rstrip('/')}/v1/auth/"
            f"{self._config.jwt_auth_path}/login"
        )

        request_body = {
            "jwt": token.get_token(),
            "role": self._config.role,
        }

        headers = {"Content-Type": "application/json"}
        if self._config.namespace:
            headers["X-Vault-Namespace"] = self._config.namespace

        request = urllib.request.Request(
            url,
            data=json.dumps(request_body).encode(),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request, timeout=self._config.request_timeout
            ) as response:
                data = json.loads(response.read())
                auth = data.get("auth", {})

                # Calculate expiration time
                expires_at = None
                lease_duration = auth.get("lease_duration", 0)
                if lease_duration:
                    expires_at = datetime.now() + timedelta(seconds=lease_duration)

                return VaultCredentials(
                    client_token=auth.get("client_token", ""),
                    accessor=auth.get("accessor", ""),
                    policies=auth.get("policies", []),
                    renewable=auth.get("renewable", False),
                    lease_duration=lease_duration,
                    expires_at=expires_at,
                )

        except urllib.error.HTTPError as e:
            error_body = ""
            try:
                error_body = e.read().decode()
            except Exception:
                pass
            raise OIDCExchangeError(
                f"Vault JWT login failed: {error_body or e.reason}",
                cloud_provider="vault",
                status_code=e.code,
                response=error_body,
            ) from e


# =============================================================================
# Factory Function
# =============================================================================


def create_token_exchanger(
    cloud_provider: CloudProvider | str,
    **kwargs: Any,
) -> BaseTokenExchanger:
    """Create a token exchanger for the specified cloud provider.

    Args:
        cloud_provider: Target cloud provider.
        **kwargs: Provider-specific configuration.

    Returns:
        Token exchanger instance.

    Raises:
        ValueError: If cloud provider is not supported.

    Example:
        >>> exchanger = create_token_exchanger(
        ...     "aws",
        ...     role_arn="arn:aws:iam::123456789012:role/my-role",
        ... )
    """
    if isinstance(cloud_provider, str):
        cloud_provider = CloudProvider(cloud_provider.lower())

    exchangers = {
        CloudProvider.AWS: AWSTokenExchanger,
        CloudProvider.GCP: GCPTokenExchanger,
        CloudProvider.AZURE: AzureTokenExchanger,
        CloudProvider.VAULT: VaultTokenExchanger,
    }

    exchanger_cls = exchangers.get(cloud_provider)
    if exchanger_cls is None:
        raise ValueError(f"Unsupported cloud provider: {cloud_provider}")

    return exchanger_cls(**kwargs)
