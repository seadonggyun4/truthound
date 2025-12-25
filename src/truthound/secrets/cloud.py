"""Cloud secret provider implementations.

This module provides integrations with major cloud secret management services:
- AWS Secrets Manager
- HashiCorp Vault
- Azure Key Vault
- GCP Secret Manager

Each provider follows a consistent interface while handling provider-specific
authentication and API interactions.

Design Notes:
    - Lazy client initialization to avoid import errors
    - Async-ready architecture with sync wrappers
    - Comprehensive error mapping to SecretError hierarchy
    - Built-in retry and caching support
"""

from __future__ import annotations

import json
from typing import Any, TYPE_CHECKING

from truthound.secrets.base import (
    BaseSecretProvider,
    SecretNotFoundError,
    SecretAccessError,
    SecretProviderError,
)

if TYPE_CHECKING:
    pass


# =============================================================================
# AWS Secrets Manager Provider
# =============================================================================


class AWSSecretsManagerProvider(BaseSecretProvider):
    """Secret provider for AWS Secrets Manager.

    Supports:
        - Secret retrieval by name or ARN
        - Version staging labels (AWSCURRENT, AWSPREVIOUS, custom)
        - JSON field extraction
        - Automatic credential discovery (IAM, env vars, config files)

    Authentication:
        Uses boto3's credential chain:
        1. Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
        2. Shared credentials file (~/.aws/credentials)
        3. IAM role (EC2, ECS, Lambda)
        4. Explicit credentials in constructor

    Example:
        >>> provider = AWSSecretsManagerProvider(region_name="us-east-1")
        >>> secret = provider.get("prod/database/credentials")
        >>> db_password = provider.get("prod/database/credentials", field="password")

        >>> # With explicit credentials
        >>> provider = AWSSecretsManagerProvider(
        ...     region_name="us-east-1",
        ...     aws_access_key_id="...",
        ...     aws_secret_access_key="...",
        ... )
    """

    def __init__(
        self,
        region_name: str | None = None,
        *,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        endpoint_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize AWS Secrets Manager provider.

        Args:
            region_name: AWS region. Defaults to AWS_DEFAULT_REGION or us-east-1.
            aws_access_key_id: AWS access key ID.
            aws_secret_access_key: AWS secret access key.
            aws_session_token: AWS session token for temporary credentials.
            endpoint_url: Custom endpoint URL (for LocalStack, etc.).
            **kwargs: Additional arguments passed to BaseSecretProvider.
        """
        super().__init__(**kwargs)
        self._region_name = region_name
        self._access_key_id = aws_access_key_id
        self._secret_access_key = aws_secret_access_key
        self._session_token = aws_session_token
        self._endpoint_url = endpoint_url
        self._client: Any = None

    @property
    def name(self) -> str:
        return "aws"

    def _get_client(self) -> Any:
        """Lazily initialize the boto3 client."""
        if self._client is None:
            try:
                import boto3
                from botocore.config import Config
            except ImportError:
                raise SecretProviderError(
                    self.name,
                    "boto3 is required for AWS Secrets Manager. "
                    "Install with: pip install boto3",
                )

            config = Config(
                retries={"max_attempts": 3, "mode": "standard"}
            )

            client_kwargs: dict[str, Any] = {"config": config}

            if self._region_name:
                client_kwargs["region_name"] = self._region_name
            if self._access_key_id:
                client_kwargs["aws_access_key_id"] = self._access_key_id
            if self._secret_access_key:
                client_kwargs["aws_secret_access_key"] = self._secret_access_key
            if self._session_token:
                client_kwargs["aws_session_token"] = self._session_token
            if self._endpoint_url:
                client_kwargs["endpoint_url"] = self._endpoint_url

            self._client = boto3.client("secretsmanager", **client_kwargs)

        return self._client

    def _fetch(
        self,
        key: str,
        version: str | None = None,
        field: str | None = None,
    ) -> str:
        """Fetch secret from AWS Secrets Manager."""
        try:
            client = self._get_client()

            # Build request
            request: dict[str, Any] = {"SecretId": key}

            if version:
                # Check if it's a version ID or stage label
                if version.startswith("AWSCURRENT") or version.startswith("AWSPREVIOUS"):
                    request["VersionStage"] = version
                else:
                    request["VersionId"] = version

            response = client.get_secret_value(**request)

        except Exception as e:
            error_code = getattr(e, "response", {}).get("Error", {}).get("Code", "")

            if error_code == "ResourceNotFoundException":
                raise SecretNotFoundError(key, self.name) from e
            elif error_code in (
                "AccessDeniedException",
                "UnauthorizedAccess",
                "InvalidAccessException",
            ):
                raise SecretAccessError(
                    key, "Access denied by IAM policy", self.name
                ) from e
            elif error_code == "InvalidParameterException":
                raise SecretProviderError(
                    self.name, f"Invalid parameter for secret '{key}': {e}", e
                ) from e
            else:
                raise SecretProviderError(
                    self.name, f"Failed to retrieve secret '{key}': {e}", e
                ) from e

        # Extract value
        if "SecretString" in response:
            value = response["SecretString"]
        elif "SecretBinary" in response:
            # Decode binary secrets
            value = response["SecretBinary"].decode("utf-8")
        else:
            raise SecretProviderError(
                self.name, f"Secret '{key}' has no value"
            )

        # Handle field extraction for JSON secrets
        if field:
            try:
                data = json.loads(value)
                if isinstance(data, dict) and field in data:
                    value = str(data[field])
                else:
                    raise SecretNotFoundError(f"{key}#{field}", self.name)
            except json.JSONDecodeError:
                raise SecretProviderError(
                    self.name,
                    f"Secret '{key}' is not valid JSON, cannot extract field '{field}'",
                )

        return value


# =============================================================================
# HashiCorp Vault Provider
# =============================================================================


class VaultProvider(BaseSecretProvider):
    """Secret provider for HashiCorp Vault.

    Supports:
        - KV v1 and v2 secret engines
        - Multiple auth methods (token, AppRole, AWS, Kubernetes)
        - Secret versioning (KV v2)
        - Namespace support (Enterprise)
        - TLS configuration

    Authentication Methods:
        - Token: Direct token authentication
        - AppRole: Machine-to-machine authentication
        - AWS IAM: EC2/Lambda role authentication
        - Kubernetes: Pod service account authentication

    Example:
        >>> # Token auth
        >>> provider = VaultProvider(
        ...     url="https://vault.example.com:8200",
        ...     token="hvs.xxxxx",
        ... )
        >>> secret = provider.get("secret/data/database/creds")

        >>> # AppRole auth
        >>> provider = VaultProvider(
        ...     url="https://vault.example.com:8200",
        ...     auth_method="approle",
        ...     role_id="xxxx",
        ...     secret_id="yyyy",
        ... )

        >>> # KV v2 with version
        >>> secret = provider.get("database/creds", version="3")
    """

    def __init__(
        self,
        url: str,
        *,
        token: str | None = None,
        auth_method: str = "token",
        role_id: str | None = None,
        secret_id: str | None = None,
        mount_point: str = "secret",
        kv_version: int = 2,
        namespace: str | None = None,
        verify: bool | str = True,
        timeout: int = 30,
        **kwargs: Any,
    ) -> None:
        """Initialize Vault provider.

        Args:
            url: Vault server URL.
            token: Vault token (for token auth).
            auth_method: Auth method: "token", "approle", "aws", "kubernetes".
            role_id: AppRole role ID.
            secret_id: AppRole secret ID.
            mount_point: KV engine mount point.
            kv_version: KV engine version (1 or 2).
            namespace: Vault namespace (Enterprise).
            verify: SSL verification (True, False, or path to CA cert).
            timeout: Request timeout in seconds.
            **kwargs: Additional arguments passed to BaseSecretProvider.
        """
        super().__init__(**kwargs)
        self._url = url.rstrip("/")
        self._token = token
        self._auth_method = auth_method
        self._role_id = role_id
        self._secret_id = secret_id
        self._mount_point = mount_point
        self._kv_version = kv_version
        self._namespace = namespace
        self._verify = verify
        self._timeout = timeout
        self._client: Any = None
        self._authenticated = False

    @property
    def name(self) -> str:
        return "vault"

    def _get_client(self) -> Any:
        """Lazily initialize and authenticate hvac client."""
        if self._client is None:
            try:
                import hvac
            except ImportError:
                raise SecretProviderError(
                    self.name,
                    "hvac is required for HashiCorp Vault. "
                    "Install with: pip install hvac",
                )

            self._client = hvac.Client(
                url=self._url,
                token=self._token,
                namespace=self._namespace,
                verify=self._verify,
                timeout=self._timeout,
            )

            self._authenticate()

        return self._client

    def _authenticate(self) -> None:
        """Authenticate to Vault using configured method."""
        if self._authenticated:
            return

        client = self._client

        if self._auth_method == "token":
            # Token already set in constructor
            if not self._token:
                raise SecretProviderError(
                    self.name, "Token required for token authentication"
                )

        elif self._auth_method == "approle":
            if not self._role_id or not self._secret_id:
                raise SecretProviderError(
                    self.name, "role_id and secret_id required for AppRole auth"
                )
            try:
                response = client.auth.approle.login(
                    role_id=self._role_id,
                    secret_id=self._secret_id,
                )
                client.token = response["auth"]["client_token"]
            except Exception as e:
                raise SecretAccessError(
                    "approle", f"AppRole authentication failed: {e}", self.name
                ) from e

        elif self._auth_method == "aws":
            try:
                client.auth.aws.iam_login()
            except Exception as e:
                raise SecretAccessError(
                    "aws", f"AWS IAM authentication failed: {e}", self.name
                ) from e

        elif self._auth_method == "kubernetes":
            try:
                # Read service account token
                with open("/var/run/secrets/kubernetes.io/serviceaccount/token") as f:
                    jwt = f.read()
                client.auth.kubernetes.login(role=self._role_id, jwt=jwt)
            except Exception as e:
                raise SecretAccessError(
                    "kubernetes",
                    f"Kubernetes authentication failed: {e}",
                    self.name,
                ) from e

        else:
            raise SecretProviderError(
                self.name, f"Unknown auth method: {self._auth_method}"
            )

        self._authenticated = True

    def _fetch(
        self,
        key: str,
        version: str | None = None,
        field: str | None = None,
    ) -> str:
        """Fetch secret from Vault."""
        try:
            client = self._get_client()

            if self._kv_version == 2:
                # KV v2
                kwargs: dict[str, Any] = {
                    "path": key,
                    "mount_point": self._mount_point,
                }
                if version:
                    kwargs["version"] = int(version)

                response = client.secrets.kv.v2.read_secret_version(**kwargs)
                data = response.get("data", {}).get("data", {})

            else:
                # KV v1
                response = client.secrets.kv.v1.read_secret(
                    path=key,
                    mount_point=self._mount_point,
                )
                data = response.get("data", {})

        except Exception as e:
            error_class = type(e).__name__

            if "Forbidden" in str(e) or error_class == "Forbidden":
                raise SecretAccessError(
                    key, "Access denied by Vault policy", self.name
                ) from e
            elif "InvalidPath" in str(e) or "secret not found" in str(e).lower():
                raise SecretNotFoundError(key, self.name) from e
            else:
                raise SecretProviderError(
                    self.name, f"Failed to retrieve secret '{key}': {e}", e
                ) from e

        if not data:
            raise SecretNotFoundError(key, self.name)

        # Extract specific field or return all as JSON
        if field:
            if field not in data:
                raise SecretNotFoundError(f"{key}#{field}", self.name)
            value = data[field]
        else:
            # If single key, return value; otherwise return JSON
            if len(data) == 1:
                value = list(data.values())[0]
            else:
                value = json.dumps(data)

        return str(value)


# =============================================================================
# Azure Key Vault Provider
# =============================================================================


class AzureKeyVaultProvider(BaseSecretProvider):
    """Secret provider for Azure Key Vault.

    Supports:
        - Secret retrieval by name
        - Version support
        - Multiple authentication methods via DefaultAzureCredential

    Authentication:
        Uses azure-identity's DefaultAzureCredential which tries:
        1. Environment variables (AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_TENANT_ID)
        2. Managed Identity (MSI)
        3. Azure CLI
        4. Visual Studio Code credentials

    Example:
        >>> provider = AzureKeyVaultProvider(
        ...     vault_url="https://my-vault.vault.azure.net"
        ... )
        >>> secret = provider.get("database-password")

        >>> # With specific version
        >>> secret = provider.get("api-key", version="abc123")
    """

    def __init__(
        self,
        vault_url: str,
        *,
        credential: Any = None,
        **kwargs: Any,
    ) -> None:
        """Initialize Azure Key Vault provider.

        Args:
            vault_url: Key Vault URL (https://{vault-name}.vault.azure.net).
            credential: Azure credential object. Uses DefaultAzureCredential if None.
            **kwargs: Additional arguments passed to BaseSecretProvider.
        """
        super().__init__(**kwargs)
        self._vault_url = vault_url.rstrip("/")
        self._credential = credential
        self._client: Any = None

    @property
    def name(self) -> str:
        return "azure"

    def _get_client(self) -> Any:
        """Lazily initialize the Azure client."""
        if self._client is None:
            try:
                from azure.keyvault.secrets import SecretClient
                from azure.identity import DefaultAzureCredential
            except ImportError:
                raise SecretProviderError(
                    self.name,
                    "azure-keyvault-secrets and azure-identity are required. "
                    "Install with: pip install azure-keyvault-secrets azure-identity",
                )

            credential = self._credential or DefaultAzureCredential()
            self._client = SecretClient(
                vault_url=self._vault_url,
                credential=credential,
            )

        return self._client

    def _fetch(
        self,
        key: str,
        version: str | None = None,
        field: str | None = None,
    ) -> str:
        """Fetch secret from Azure Key Vault."""
        try:
            client = self._get_client()
            secret = client.get_secret(key, version=version)
            value = secret.value

        except Exception as e:
            error_class = type(e).__name__

            if "ResourceNotFoundError" in error_class:
                raise SecretNotFoundError(key, self.name) from e
            elif "ClientAuthenticationError" in error_class:
                raise SecretAccessError(
                    key, "Authentication failed", self.name
                ) from e
            elif "HttpResponseError" in error_class and "Forbidden" in str(e):
                raise SecretAccessError(
                    key, "Access denied by Key Vault policy", self.name
                ) from e
            else:
                raise SecretProviderError(
                    self.name, f"Failed to retrieve secret '{key}': {e}", e
                ) from e

        if value is None:
            raise SecretNotFoundError(key, self.name)

        # Handle field extraction for JSON secrets
        if field:
            try:
                data = json.loads(value)
                if isinstance(data, dict) and field in data:
                    value = str(data[field])
                else:
                    raise SecretNotFoundError(f"{key}#{field}", self.name)
            except json.JSONDecodeError:
                raise SecretProviderError(
                    self.name,
                    f"Secret '{key}' is not valid JSON, cannot extract field '{field}'",
                )

        return value


# =============================================================================
# GCP Secret Manager Provider
# =============================================================================


class GCPSecretManagerProvider(BaseSecretProvider):
    """Secret provider for Google Cloud Secret Manager.

    Supports:
        - Secret retrieval by name
        - Version support (specific version or "latest")
        - Automatic credential discovery (ADC)

    Authentication:
        Uses Application Default Credentials (ADC):
        1. GOOGLE_APPLICATION_CREDENTIALS environment variable
        2. User credentials (from gcloud auth application-default login)
        3. Compute Engine service account
        4. Kubernetes workload identity

    Example:
        >>> provider = GCPSecretManagerProvider(project_id="my-project")
        >>> secret = provider.get("database-password")

        >>> # With specific version
        >>> secret = provider.get("api-key", version="5")
    """

    def __init__(
        self,
        project_id: str,
        *,
        credentials: Any = None,
        **kwargs: Any,
    ) -> None:
        """Initialize GCP Secret Manager provider.

        Args:
            project_id: GCP project ID.
            credentials: Google credentials object. Uses ADC if None.
            **kwargs: Additional arguments passed to BaseSecretProvider.
        """
        super().__init__(**kwargs)
        self._project_id = project_id
        self._credentials = credentials
        self._client: Any = None

    @property
    def name(self) -> str:
        return "gcp"

    def _get_client(self) -> Any:
        """Lazily initialize the GCP client."""
        if self._client is None:
            try:
                from google.cloud import secretmanager
            except ImportError:
                raise SecretProviderError(
                    self.name,
                    "google-cloud-secret-manager is required. "
                    "Install with: pip install google-cloud-secret-manager",
                )

            if self._credentials:
                self._client = secretmanager.SecretManagerServiceClient(
                    credentials=self._credentials
                )
            else:
                self._client = secretmanager.SecretManagerServiceClient()

        return self._client

    def _build_secret_path(self, key: str, version: str | None = None) -> str:
        """Build the full secret path."""
        version = version or "latest"
        return f"projects/{self._project_id}/secrets/{key}/versions/{version}"

    def _fetch(
        self,
        key: str,
        version: str | None = None,
        field: str | None = None,
    ) -> str:
        """Fetch secret from GCP Secret Manager."""
        try:
            client = self._get_client()
            secret_path = self._build_secret_path(key, version)
            response = client.access_secret_version(name=secret_path)
            value = response.payload.data.decode("utf-8")

        except Exception as e:
            error_class = type(e).__name__

            if "NotFound" in error_class or "NOT_FOUND" in str(e):
                raise SecretNotFoundError(key, self.name) from e
            elif "PermissionDenied" in error_class or "PERMISSION_DENIED" in str(e):
                raise SecretAccessError(
                    key, "Access denied by IAM policy", self.name
                ) from e
            else:
                raise SecretProviderError(
                    self.name, f"Failed to retrieve secret '{key}': {e}", e
                ) from e

        # Handle field extraction for JSON secrets
        if field:
            try:
                data = json.loads(value)
                if isinstance(data, dict) and field in data:
                    value = str(data[field])
                else:
                    raise SecretNotFoundError(f"{key}#{field}", self.name)
            except json.JSONDecodeError:
                raise SecretProviderError(
                    self.name,
                    f"Secret '{key}' is not valid JSON, cannot extract field '{field}'",
                )

        return value
