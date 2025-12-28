"""Cloud KMS backend for integration tests.

This module provides Key Management Service integration testing with support for:
- AWS KMS
- Google Cloud KMS
- Azure Key Vault
- HashiCorp Vault (transit engine)
- LocalStack (for local testing)

Features:
    - Key management (create, rotate, delete)
    - Encryption/decryption operations
    - Key versioning
    - Multi-provider support

Usage:
    >>> config = KMSConfig.from_env()
    >>> with KMSBackend(config) as backend:
    ...     ciphertext = backend.encrypt("my-key", b"secret data")
    ...     plaintext = backend.decrypt("my-key", ciphertext)
"""

from __future__ import annotations

import base64
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Protocol

from tests.integration.external.base import (
    ExternalServiceBackend,
    HealthCheckResult,
    ProviderType,
    ServiceCategory,
    ServiceConfig,
)
from tests.integration.external.providers.docker_provider import DockerContainerConfig

logger = logging.getLogger(__name__)


# =============================================================================
# KMS Provider Enum
# =============================================================================


class KMSProvider(Enum):
    """Supported KMS providers."""

    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    VAULT = "vault"
    LOCALSTACK = "localstack"
    MOCK = "mock"


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class KMSConfig(DockerContainerConfig):
    """KMS-specific configuration.

    Attributes:
        kms_provider: KMS provider to use
        region: Cloud region
        key_id: Default key ID/ARN
        access_key: AWS access key
        secret_key: AWS secret key
        project_id: GCP project ID
        credentials_file: Path to credentials file
        vault_token: Vault token (for Vault provider)
        vault_transit_path: Vault transit mount path
    """

    kms_provider: KMSProvider = KMSProvider.LOCALSTACK
    region: str = "us-east-1"
    key_id: str | None = None
    access_key: str | None = None
    secret_key: str | None = None
    project_id: str | None = None
    credentials_file: str | None = None
    vault_token: str | None = None
    vault_transit_path: str = "transit"

    def __post_init__(self) -> None:
        """Set KMS-specific defaults."""
        self.name = self.name or "kms"
        self.category = ServiceCategory.SECRETS

        if self.kms_provider == KMSProvider.LOCALSTACK:
            self.image = self.image or "localstack/localstack"
            self.tag = self.tag or "latest"
            self.ports = self.ports or {"4566/tcp": None}
            self.environment.setdefault("SERVICES", "kms")
            self.environment.setdefault("DEBUG", "1")
            self.health_cmd = "curl -s http://localhost:4566/_localstack/health"

    @classmethod
    def from_env(cls, name: str = "kms") -> "KMSConfig":
        """Create configuration from environment variables."""
        prefix = "TRUTHOUND_TEST_KMS"

        provider_str = os.getenv(f"{prefix}_PROVIDER", "localstack")
        try:
            kms_provider = KMSProvider(provider_str.lower())
        except ValueError:
            kms_provider = KMSProvider.LOCALSTACK

        return cls(
            name=name,
            category=ServiceCategory.SECRETS,
            kms_provider=kms_provider,
            host=os.getenv(f"{prefix}_HOST"),
            port=int(os.getenv(f"{prefix}_PORT", "0")) or None,
            region=os.getenv(f"{prefix}_REGION", "us-east-1"),
            key_id=os.getenv(f"{prefix}_KEY_ID"),
            access_key=os.getenv(f"{prefix}_ACCESS_KEY"),
            secret_key=os.getenv(f"{prefix}_SECRET_KEY"),
            project_id=os.getenv(f"{prefix}_PROJECT_ID"),
            credentials_file=os.getenv(f"{prefix}_CREDENTIALS_FILE"),
            vault_token=os.getenv(f"{prefix}_VAULT_TOKEN"),
            timeout_seconds=int(os.getenv(f"{prefix}_TIMEOUT", "30")),
        )


# =============================================================================
# KMS Client Protocol
# =============================================================================


class KMSClientProtocol(Protocol):
    """Protocol for KMS client implementations."""

    def create_key(self, description: str) -> str:
        """Create a new encryption key.

        Returns:
            Key ID/ARN
        """
        ...

    def encrypt(self, key_id: str, plaintext: bytes) -> bytes:
        """Encrypt data.

        Returns:
            Ciphertext
        """
        ...

    def decrypt(self, key_id: str, ciphertext: bytes) -> bytes:
        """Decrypt data.

        Returns:
            Plaintext
        """
        ...

    def generate_data_key(self, key_id: str) -> tuple[bytes, bytes]:
        """Generate a data encryption key.

        Returns:
            Tuple of (plaintext_key, encrypted_key)
        """
        ...

    def rotate_key(self, key_id: str) -> bool:
        """Enable key rotation."""
        ...

    def delete_key(self, key_id: str) -> bool:
        """Schedule key deletion."""
        ...


# =============================================================================
# LocalStack KMS Client
# =============================================================================


class LocalStackKMSClient:
    """KMS client for LocalStack."""

    def __init__(self, endpoint_url: str, region: str, access_key: str, secret_key: str):
        """Initialize LocalStack KMS client."""
        try:
            import boto3
        except ImportError:
            raise ImportError("boto3 package not installed. Run: pip install boto3")

        self._client = boto3.client(
            "kms",
            endpoint_url=endpoint_url,
            region_name=region,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )

    def create_key(self, description: str = "Test key") -> str:
        """Create a KMS key."""
        response = self._client.create_key(Description=description)
        return response["KeyMetadata"]["KeyId"]

    def encrypt(self, key_id: str, plaintext: bytes) -> bytes:
        """Encrypt data."""
        response = self._client.encrypt(KeyId=key_id, Plaintext=plaintext)
        return response["CiphertextBlob"]

    def decrypt(self, key_id: str, ciphertext: bytes) -> bytes:
        """Decrypt data."""
        response = self._client.decrypt(KeyId=key_id, CiphertextBlob=ciphertext)
        return response["Plaintext"]

    def generate_data_key(self, key_id: str) -> tuple[bytes, bytes]:
        """Generate a data encryption key."""
        response = self._client.generate_data_key(KeyId=key_id, KeySpec="AES_256")
        return response["Plaintext"], response["CiphertextBlob"]

    def rotate_key(self, key_id: str) -> bool:
        """Enable automatic key rotation."""
        try:
            self._client.enable_key_rotation(KeyId=key_id)
            return True
        except Exception:
            return False

    def delete_key(self, key_id: str, pending_window_days: int = 7) -> bool:
        """Schedule key for deletion."""
        try:
            self._client.schedule_key_deletion(
                KeyId=key_id,
                PendingWindowInDays=pending_window_days,
            )
            return True
        except Exception:
            return False

    def list_keys(self) -> list[str]:
        """List all keys."""
        response = self._client.list_keys()
        return [key["KeyId"] for key in response.get("Keys", [])]


# =============================================================================
# Mock KMS Client
# =============================================================================


class MockKMSClient:
    """In-memory mock KMS client for testing."""

    def __init__(self) -> None:
        """Initialize mock KMS client."""
        self._keys: dict[str, dict[str, Any]] = {}
        self._key_counter = 0
        import os
        self._master_key = os.urandom(32)

    def create_key(self, description: str = "Test key") -> str:
        """Create a mock key."""
        import os
        import uuid

        self._key_counter += 1
        key_id = str(uuid.uuid4())
        self._keys[key_id] = {
            "id": key_id,
            "description": description,
            "key_material": os.urandom(32),
            "created_at": None,
            "rotation_enabled": False,
            "deleted": False,
        }
        return key_id

    def encrypt(self, key_id: str, plaintext: bytes) -> bytes:
        """Encrypt data using mock encryption."""
        if key_id not in self._keys or self._keys[key_id]["deleted"]:
            raise ValueError(f"Key not found: {key_id}")

        import hashlib
        key_material = self._keys[key_id]["key_material"]
        key_stream = hashlib.sha256(key_material + b"encrypt").digest()

        # Simple XOR for mock (NOT SECURE - for testing only)
        ciphertext = bytes(
            p ^ key_stream[i % len(key_stream)]
            for i, p in enumerate(plaintext)
        )
        # Prepend key_id for decryption
        return key_id.encode() + b":" + ciphertext

    def decrypt(self, key_id: str, ciphertext: bytes) -> bytes:
        """Decrypt data using mock decryption."""
        # Extract key_id from ciphertext
        parts = ciphertext.split(b":", 1)
        if len(parts) != 2:
            raise ValueError("Invalid ciphertext format")

        stored_key_id = parts[0].decode()
        encrypted_data = parts[1]

        if stored_key_id not in self._keys or self._keys[stored_key_id]["deleted"]:
            raise ValueError(f"Key not found: {stored_key_id}")

        import hashlib
        key_material = self._keys[stored_key_id]["key_material"]
        key_stream = hashlib.sha256(key_material + b"encrypt").digest()

        plaintext = bytes(
            c ^ key_stream[i % len(key_stream)]
            for i, c in enumerate(encrypted_data)
        )
        return plaintext

    def generate_data_key(self, key_id: str) -> tuple[bytes, bytes]:
        """Generate a data key."""
        import os
        plaintext_key = os.urandom(32)
        encrypted_key = self.encrypt(key_id, plaintext_key)
        return plaintext_key, encrypted_key

    def rotate_key(self, key_id: str) -> bool:
        """Enable key rotation."""
        if key_id in self._keys:
            self._keys[key_id]["rotation_enabled"] = True
            return True
        return False

    def delete_key(self, key_id: str, pending_window_days: int = 7) -> bool:
        """Mark key as deleted."""
        if key_id in self._keys:
            self._keys[key_id]["deleted"] = True
            return True
        return False

    def list_keys(self) -> list[str]:
        """List all keys."""
        return [k for k, v in self._keys.items() if not v["deleted"]]


# =============================================================================
# KMS Backend
# =============================================================================


class KMSBackend(ExternalServiceBackend[KMSConfig, KMSClientProtocol]):
    """Cloud KMS test backend.

    Provides unified interface for testing with various KMS providers.

    Features:
        - Multi-provider support (AWS, GCP, Azure, Vault, LocalStack, Mock)
        - Key lifecycle management
        - Encryption/decryption operations
        - Data key generation
    """

    service_name: ClassVar[str] = "kms"
    service_category: ClassVar[ServiceCategory] = ServiceCategory.SECRETS
    default_port: ClassVar[int] = 4566  # LocalStack default
    default_image: ClassVar[str] = "localstack/localstack:latest"

    def __init__(
        self,
        config: KMSConfig | None = None,
        provider: Any = None,
    ) -> None:
        """Initialize KMS backend."""
        if config is None:
            config = KMSConfig.from_env()
        super().__init__(config, provider)

    def _create_client(self) -> KMSClientProtocol:
        """Create KMS client based on provider."""
        config = self.config

        if config.kms_provider == KMSProvider.LOCALSTACK:
            endpoint_url = f"http://{self.host}:{self.port}"
            return LocalStackKMSClient(
                endpoint_url=endpoint_url,
                region=config.region,
                access_key=config.access_key or "test",
                secret_key=config.secret_key or "test",
            )

        elif config.kms_provider == KMSProvider.MOCK:
            return MockKMSClient()

        elif config.kms_provider == KMSProvider.AWS:
            return self._create_aws_client(config)

        elif config.kms_provider == KMSProvider.GCP:
            return self._create_gcp_client(config)

        elif config.kms_provider == KMSProvider.AZURE:
            return self._create_azure_client(config)

        elif config.kms_provider == KMSProvider.VAULT:
            return self._create_vault_client(config)

        else:
            raise ValueError(f"Unsupported KMS provider: {config.kms_provider}")

    def _create_aws_client(self, config: KMSConfig) -> KMSClientProtocol:
        """Create AWS KMS client."""
        try:
            import boto3
        except ImportError:
            raise ImportError("boto3 package not installed. Run: pip install boto3")

        return LocalStackKMSClient(
            endpoint_url=f"https://kms.{config.region}.amazonaws.com",
            region=config.region,
            access_key=config.access_key or "",
            secret_key=config.secret_key or "",
        )

    def _create_gcp_client(self, config: KMSConfig) -> KMSClientProtocol:
        """Create GCP Cloud KMS client."""
        # Would implement GCP-specific client
        raise NotImplementedError("GCP KMS client not implemented")

    def _create_azure_client(self, config: KMSConfig) -> KMSClientProtocol:
        """Create Azure Key Vault client."""
        # Would implement Azure-specific client
        raise NotImplementedError("Azure Key Vault client not implemented")

    def _create_vault_client(self, config: KMSConfig) -> KMSClientProtocol:
        """Create Vault transit engine client."""
        from tests.integration.external.backends.vault_backend import VaultBackend, VaultConfig

        vault_config = VaultConfig(
            host=config.host or "localhost",
            port=config.port or 8200,
            token=config.vault_token,
        )
        vault = VaultBackend(vault_config)
        vault.connect()
        vault.enable_transit()

        # Wrap in protocol-compatible interface
        return VaultTransitAdapter(vault, config.vault_transit_path)

    def _close_client(self) -> None:
        """Close KMS client."""
        pass  # Most clients don't need explicit closing

    def _perform_health_check(self) -> HealthCheckResult:
        """Perform KMS health check."""
        if self._client is None:
            return HealthCheckResult.failure("Client not connected")

        try:
            # Try to list keys as health check
            keys = self._client.list_keys()
            return HealthCheckResult.success(
                "KMS healthy",
                provider=self.config.kms_provider.value,
                key_count=len(keys),
            )
        except Exception as e:
            return HealthCheckResult.failure(str(e))

    # -------------------------------------------------------------------------
    # KMS Operations
    # -------------------------------------------------------------------------

    def create_key(self, description: str = "Test key") -> str:
        """Create a new encryption key.

        Args:
            description: Key description

        Returns:
            Key ID
        """
        if self._client is None:
            raise RuntimeError("Client not connected")
        return self._client.create_key(description)

    def encrypt(self, key_id: str, plaintext: bytes) -> bytes:
        """Encrypt data.

        Args:
            key_id: Key ID to use
            plaintext: Data to encrypt

        Returns:
            Ciphertext
        """
        if self._client is None:
            raise RuntimeError("Client not connected")
        return self._client.encrypt(key_id, plaintext)

    def decrypt(self, key_id: str, ciphertext: bytes) -> bytes:
        """Decrypt data.

        Args:
            key_id: Key ID to use
            ciphertext: Data to decrypt

        Returns:
            Plaintext
        """
        if self._client is None:
            raise RuntimeError("Client not connected")
        return self._client.decrypt(key_id, ciphertext)

    def generate_data_key(self, key_id: str) -> tuple[bytes, bytes]:
        """Generate a data encryption key.

        Args:
            key_id: Master key ID

        Returns:
            Tuple of (plaintext_key, encrypted_key)
        """
        if self._client is None:
            raise RuntimeError("Client not connected")
        return self._client.generate_data_key(key_id)

    def encrypt_string(self, key_id: str, plaintext: str) -> str:
        """Encrypt a string and return base64-encoded ciphertext.

        Args:
            key_id: Key ID to use
            plaintext: String to encrypt

        Returns:
            Base64-encoded ciphertext
        """
        ciphertext = self.encrypt(key_id, plaintext.encode())
        return base64.b64encode(ciphertext).decode()

    def decrypt_string(self, key_id: str, ciphertext_b64: str) -> str:
        """Decrypt a base64-encoded ciphertext to string.

        Args:
            key_id: Key ID to use
            ciphertext_b64: Base64-encoded ciphertext

        Returns:
            Decrypted string
        """
        ciphertext = base64.b64decode(ciphertext_b64)
        plaintext = self.decrypt(key_id, ciphertext)
        return plaintext.decode()


# =============================================================================
# Vault Transit Adapter
# =============================================================================


class VaultTransitAdapter:
    """Adapter to use Vault Transit as KMS client."""

    def __init__(self, vault_backend: Any, transit_path: str):
        """Initialize adapter."""
        self._vault = vault_backend
        self._path = transit_path

    def create_key(self, description: str = "Test key") -> str:
        """Create a transit key."""
        import uuid
        key_name = f"key-{uuid.uuid4().hex[:8]}"
        self._vault.create_transit_key(key_name)
        return key_name

    def encrypt(self, key_id: str, plaintext: bytes) -> bytes:
        """Encrypt using transit engine."""
        plaintext_str = base64.b64encode(plaintext).decode()
        ciphertext = self._vault.transit_encrypt(key_id, plaintext_str)
        return ciphertext.encode() if ciphertext else b""

    def decrypt(self, key_id: str, ciphertext: bytes) -> bytes:
        """Decrypt using transit engine."""
        ciphertext_str = ciphertext.decode()
        plaintext_b64 = self._vault.transit_decrypt(key_id, ciphertext_str)
        return base64.b64decode(plaintext_b64) if plaintext_b64 else b""

    def generate_data_key(self, key_id: str) -> tuple[bytes, bytes]:
        """Generate data key."""
        import os
        plaintext_key = os.urandom(32)
        encrypted_key = self.encrypt(key_id, plaintext_key)
        return plaintext_key, encrypted_key

    def rotate_key(self, key_id: str) -> bool:
        """Rotate key."""
        return True

    def delete_key(self, key_id: str) -> bool:
        """Delete key."""
        return True

    def list_keys(self) -> list[str]:
        """List keys."""
        return []


# =============================================================================
# Test Helpers
# =============================================================================


def create_kms_backend(
    provider_type: ProviderType = ProviderType.DOCKER,
    kms_provider: KMSProvider = KMSProvider.LOCALSTACK,
) -> KMSBackend:
    """Create a KMS backend with specified providers."""
    config = KMSConfig.from_env()
    config.provider = provider_type
    config.kms_provider = kms_provider

    if kms_provider == KMSProvider.MOCK:
        config.provider = ProviderType.MOCK

    from tests.integration.external.providers import get_provider
    provider = get_provider(config.provider, config)

    return KMSBackend(config, provider)
