"""HashiCorp Vault backend for integration tests.

This module provides Vault integration testing with support for:
- Docker containers (dev mode)
- Local Vault instances
- Vault Enterprise

Features:
    - KV secrets engine
    - Transit secrets engine (encryption/decryption)
    - Token authentication
    - AppRole authentication
    - Dynamic secrets

Usage:
    >>> config = VaultConfig.from_env()
    >>> with VaultBackend(config) as backend:
    ...     backend.write_secret("secret/data/myapp", {"password": "secret"})
    ...     secret = backend.read_secret("secret/data/myapp")
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, ClassVar, TYPE_CHECKING

from tests.integration.external.base import (
    ExternalServiceBackend,
    HealthCheckResult,
    ProviderType,
    ServiceCategory,
)
from tests.integration.external.providers.docker_provider import DockerContainerConfig

if TYPE_CHECKING:
    import hvac

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class VaultConfig(DockerContainerConfig):
    """Vault-specific configuration.

    Attributes:
        token: Vault token for authentication
        role_id: AppRole role ID
        secret_id: AppRole secret ID
        namespace: Vault namespace (Enterprise)
        ssl: Use SSL/TLS
        verify_certs: Verify SSL certificates
        kv_version: KV secrets engine version (1 or 2)
    """

    token: str | None = None
    role_id: str | None = None
    secret_id: str | None = None
    namespace: str | None = None
    ssl: bool = False
    verify_certs: bool = True
    kv_version: int = 2

    def __post_init__(self) -> None:
        """Set Vault-specific defaults."""
        self.name = self.name or "vault"
        self.category = ServiceCategory.SECRETS
        self.image = self.image or "hashicorp/vault"
        self.tag = self.tag or "1.15"
        self.ports = self.ports or {"8200/tcp": None}
        self.health_cmd = self.health_cmd or "vault status"

        # Dev mode for testing
        if "VAULT_DEV_ROOT_TOKEN_ID" not in self.environment:
            self.environment["VAULT_DEV_ROOT_TOKEN_ID"] = "root-token"
        if "VAULT_DEV_LISTEN_ADDRESS" not in self.environment:
            self.environment["VAULT_DEV_LISTEN_ADDRESS"] = "0.0.0.0:8200"

        # Use dev token if no token specified
        if not self.token:
            self.token = self.environment.get("VAULT_DEV_ROOT_TOKEN_ID", "root-token")

        # Set command for dev server
        if not self.command:
            self.command = "server -dev"

    @classmethod
    def from_env(cls, name: str = "vault") -> "VaultConfig":
        """Create configuration from environment variables."""
        prefix = "TRUTHOUND_TEST_VAULT"

        return cls(
            name=name,
            category=ServiceCategory.SECRETS,
            host=os.getenv(f"{prefix}_HOST"),
            port=int(os.getenv(f"{prefix}_PORT", "0")) or None,
            token=os.getenv(f"{prefix}_TOKEN"),
            role_id=os.getenv(f"{prefix}_ROLE_ID"),
            secret_id=os.getenv(f"{prefix}_SECRET_ID"),
            namespace=os.getenv(f"{prefix}_NAMESPACE"),
            ssl=os.getenv(f"{prefix}_SSL", "false").lower() == "true",
            kv_version=int(os.getenv(f"{prefix}_KV_VERSION", "2")),
            timeout_seconds=int(os.getenv(f"{prefix}_TIMEOUT", "30")),
        )


# =============================================================================
# Vault Backend
# =============================================================================


class VaultBackend(ExternalServiceBackend[VaultConfig, "hvac.Client"]):
    """Vault test backend.

    Provides Vault connection and operations for integration testing.

    Features:
        - Automatic Docker container management (dev mode)
        - KV secrets engine operations
        - Transit engine for encryption/decryption
        - Multiple authentication methods
    """

    service_name: ClassVar[str] = "vault"
    service_category: ClassVar[ServiceCategory] = ServiceCategory.SECRETS
    default_port: ClassVar[int] = 8200
    default_image: ClassVar[str] = "hashicorp/vault:1.15"

    def __init__(
        self,
        config: VaultConfig | None = None,
        provider: Any = None,
    ) -> None:
        """Initialize Vault backend."""
        if config is None:
            config = VaultConfig.from_env()
        super().__init__(config, provider)

    def _create_client(self) -> "hvac.Client":
        """Create Vault client."""
        try:
            import hvac
        except ImportError:
            raise ImportError(
                "hvac package not installed. Run: pip install hvac"
            )

        config = self.config
        scheme = "https" if config.ssl else "http"
        url = f"{scheme}://{self.host}:{self.port}"

        client = hvac.Client(
            url=url,
            token=config.token,
            namespace=config.namespace,
            verify=config.verify_certs if config.ssl else False,
            timeout=config.timeout_seconds,
        )

        # AppRole authentication if configured
        if config.role_id and config.secret_id:
            auth_response = client.auth.approle.login(
                role_id=config.role_id,
                secret_id=config.secret_id,
            )
            client.token = auth_response["auth"]["client_token"]

        # Verify authentication
        if not client.is_authenticated():
            raise ConnectionError("Vault authentication failed")

        return client

    def _close_client(self) -> None:
        """Close Vault client (revoke token if needed)."""
        pass  # hvac client doesn't need explicit closing

    def _perform_health_check(self) -> HealthCheckResult:
        """Perform Vault health check."""
        if self._client is None:
            return HealthCheckResult.failure("Client not connected")

        try:
            health = self._client.sys.read_health_status(method="GET")

            if health.get("sealed", True):
                return HealthCheckResult.failure("Vault is sealed")

            if not health.get("initialized", False):
                return HealthCheckResult.failure("Vault is not initialized")

            return HealthCheckResult.success(
                "Vault healthy",
                version=health.get("version"),
                cluster_name=health.get("cluster_name"),
            )

        except Exception as e:
            return HealthCheckResult.failure(str(e))

    # -------------------------------------------------------------------------
    # KV Secrets Operations
    # -------------------------------------------------------------------------

    def read_secret(self, path: str) -> dict[str, Any] | None:
        """Read a secret from KV store.

        Args:
            path: Secret path (e.g., "secret/data/myapp")

        Returns:
            Secret data or None if not found
        """
        if self._client is None:
            return None

        try:
            if self.config.kv_version == 2:
                # KV v2 path format
                mount_point = path.split("/")[0]
                secret_path = "/".join(path.split("/")[2:])  # Skip "data"
                response = self._client.secrets.kv.v2.read_secret_version(
                    path=secret_path,
                    mount_point=mount_point,
                )
                return response["data"]["data"]
            else:
                # KV v1
                response = self._client.secrets.kv.v1.read_secret(path=path)
                return response["data"]
        except Exception as e:
            logger.debug(f"Failed to read secret {path}: {e}")
            return None

    def write_secret(self, path: str, data: dict[str, Any]) -> bool:
        """Write a secret to KV store.

        Args:
            path: Secret path (e.g., "secret/data/myapp")
            data: Secret data

        Returns:
            True if successful
        """
        if self._client is None:
            return False

        try:
            if self.config.kv_version == 2:
                mount_point = path.split("/")[0]
                secret_path = "/".join(path.split("/")[2:])
                self._client.secrets.kv.v2.create_or_update_secret(
                    path=secret_path,
                    secret=data,
                    mount_point=mount_point,
                )
            else:
                self._client.secrets.kv.v1.create_or_update_secret(
                    path=path,
                    secret=data,
                )
            return True
        except Exception as e:
            logger.error(f"Failed to write secret {path}: {e}")
            return False

    def delete_secret(self, path: str) -> bool:
        """Delete a secret from KV store.

        Args:
            path: Secret path

        Returns:
            True if successful
        """
        if self._client is None:
            return False

        try:
            if self.config.kv_version == 2:
                mount_point = path.split("/")[0]
                secret_path = "/".join(path.split("/")[2:])
                self._client.secrets.kv.v2.delete_metadata_and_all_versions(
                    path=secret_path,
                    mount_point=mount_point,
                )
            else:
                self._client.secrets.kv.v1.delete_secret(path=path)
            return True
        except Exception as e:
            logger.debug(f"Failed to delete secret {path}: {e}")
            return False

    def list_secrets(self, path: str) -> list[str]:
        """List secrets at path.

        Args:
            path: Path to list

        Returns:
            List of secret keys
        """
        if self._client is None:
            return []

        try:
            if self.config.kv_version == 2:
                mount_point = path.split("/")[0]
                list_path = "/".join(path.split("/")[2:]) if len(path.split("/")) > 2 else ""
                response = self._client.secrets.kv.v2.list_secrets(
                    path=list_path,
                    mount_point=mount_point,
                )
                return response["data"]["keys"]
            else:
                response = self._client.secrets.kv.v1.list_secrets(path=path)
                return response["data"]["keys"]
        except Exception:
            return []

    # -------------------------------------------------------------------------
    # Transit Engine Operations
    # -------------------------------------------------------------------------

    def enable_transit(self) -> bool:
        """Enable the transit secrets engine."""
        if self._client is None:
            return False

        try:
            self._client.sys.enable_secrets_engine(
                backend_type="transit",
                path="transit",
            )
            return True
        except Exception as e:
            if "path is already in use" in str(e).lower():
                return True
            logger.error(f"Failed to enable transit engine: {e}")
            return False

    def create_transit_key(self, name: str, key_type: str = "aes256-gcm96") -> bool:
        """Create a transit encryption key.

        Args:
            name: Key name
            key_type: Key type (aes256-gcm96, chacha20-poly1305, etc.)

        Returns:
            True if successful
        """
        if self._client is None:
            return False

        try:
            self._client.secrets.transit.create_key(name=name, key_type=key_type)
            return True
        except Exception as e:
            logger.error(f"Failed to create transit key {name}: {e}")
            return False

    def transit_encrypt(self, key_name: str, plaintext: str) -> str | None:
        """Encrypt data using transit engine.

        Args:
            key_name: Encryption key name
            plaintext: Data to encrypt

        Returns:
            Ciphertext or None on failure
        """
        if self._client is None:
            return None

        try:
            import base64
            plaintext_b64 = base64.b64encode(plaintext.encode()).decode()
            response = self._client.secrets.transit.encrypt_data(
                name=key_name,
                plaintext=plaintext_b64,
            )
            return response["data"]["ciphertext"]
        except Exception as e:
            logger.error(f"Transit encryption failed: {e}")
            return None

    def transit_decrypt(self, key_name: str, ciphertext: str) -> str | None:
        """Decrypt data using transit engine.

        Args:
            key_name: Encryption key name
            ciphertext: Data to decrypt

        Returns:
            Plaintext or None on failure
        """
        if self._client is None:
            return None

        try:
            import base64
            response = self._client.secrets.transit.decrypt_data(
                name=key_name,
                ciphertext=ciphertext,
            )
            plaintext_b64 = response["data"]["plaintext"]
            return base64.b64decode(plaintext_b64).decode()
        except Exception as e:
            logger.error(f"Transit decryption failed: {e}")
            return None


# =============================================================================
# Test Helpers
# =============================================================================


def create_vault_backend(
    provider_type: ProviderType = ProviderType.DOCKER,
) -> VaultBackend:
    """Create a Vault backend with specified provider."""
    config = VaultConfig.from_env()
    config.provider = provider_type

    from tests.integration.external.providers import get_provider
    provider = get_provider(provider_type, config)

    return VaultBackend(config, provider)
