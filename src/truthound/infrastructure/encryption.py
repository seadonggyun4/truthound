"""Enterprise data encryption system for Truthound.

This module extends the base encryption system with enterprise features:
- At-rest encryption for validation results
- Field-level encryption for sensitive columns
- Cloud KMS integration (AWS, GCP, Azure, Vault)
- Key rotation and lifecycle management

Architecture:
    AtRestEncryption
         |
         +---> Local encryption (AES-256-GCM)
         +---> Cloud KMS wrapping
         |
    FieldLevelEncryption
         |
         +---> Per-column encryption policies
         +---> Format-preserving encryption
         |
    KeyProvider
         |
         +---> VaultKeyProvider (HashiCorp Vault)
         +---> AwsKmsProvider (AWS KMS)
         +---> GcpKmsProvider (Google Cloud KMS)
         +---> AzureKeyVaultProvider (Azure Key Vault)

Usage:
    >>> from truthound.infrastructure.encryption import (
    ...     get_encryptor, configure_encryption,
    ...     AtRestEncryption, FieldLevelEncryption,
    ... )
    >>>
    >>> # Configure encryption
    >>> configure_encryption(
    ...     provider="aws_kms",
    ...     key_id="alias/truthound-data-key",
    ...     region="us-east-1",
    ... )
    >>>
    >>> # Encrypt data at rest
    >>> encryptor = get_encryptor()
    >>> encrypted = encryptor.encrypt(sensitive_data)
    >>>
    >>> # Field-level encryption
    >>> field_enc = FieldLevelEncryption(
    ...     policies={
    ...         "ssn": {"algorithm": "format_preserving"},
    ...         "email": {"algorithm": "aes_gcm"},
    ...     }
    ... )
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import struct
import threading
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterator, TypeVar

# Re-export base encryption components
from truthound.stores.encryption import (
    EncryptionAlgorithm,
    EncryptionConfig,
    EncryptionError,
    DecryptionError,
    generate_key,
    generate_nonce,
    get_encryptor as get_base_encryptor,
    AesGcmEncryptor,
    ChaCha20Poly1305Encryptor,
)


# =============================================================================
# Key Provider Protocol
# =============================================================================


class KeyProvider(ABC):
    """Abstract base class for key providers.

    Key providers supply encryption keys from various sources:
    - Cloud KMS (AWS, GCP, Azure)
    - HashiCorp Vault
    - Local key stores

    The provider handles key retrieval, caching, and rotation.
    """

    @abstractmethod
    def get_key(self, key_id: str) -> bytes:
        """Get encryption key by ID.

        Args:
            key_id: Key identifier.

        Returns:
            Raw key bytes.
        """
        pass

    @abstractmethod
    def encrypt_data_key(self, data_key: bytes, key_id: str) -> bytes:
        """Encrypt a data key with a master key (envelope encryption).

        Args:
            data_key: Plain data key.
            key_id: Master key ID.

        Returns:
            Encrypted data key.
        """
        pass

    @abstractmethod
    def decrypt_data_key(self, encrypted_key: bytes, key_id: str) -> bytes:
        """Decrypt a data key with a master key.

        Args:
            encrypted_key: Encrypted data key.
            key_id: Master key ID.

        Returns:
            Plain data key.
        """
        pass

    def generate_data_key(self, key_id: str, key_length: int = 32) -> tuple[bytes, bytes]:
        """Generate a new data key and encrypt it.

        Args:
            key_id: Master key ID for wrapping.
            key_length: Key length in bytes.

        Returns:
            Tuple of (plain_key, encrypted_key).
        """
        plain_key = os.urandom(key_length)
        encrypted_key = self.encrypt_data_key(plain_key, key_id)
        return plain_key, encrypted_key

    def close(self) -> None:
        """Clean up provider resources."""
        pass


# =============================================================================
# Cloud Key Providers
# =============================================================================


class VaultKeyProvider(KeyProvider):
    """HashiCorp Vault key provider.

    Uses Vault's Transit secrets engine for key management.
    """

    def __init__(
        self,
        url: str,
        *,
        token: str | None = None,
        mount_point: str = "transit",
        cache_ttl: float = 300.0,
    ) -> None:
        """Initialize Vault provider.

        Args:
            url: Vault server URL.
            token: Vault token (or VAULT_TOKEN env var).
            mount_point: Transit engine mount point.
            cache_ttl: Key cache TTL in seconds.
        """
        self._url = url.rstrip("/")
        self._token = token or os.getenv("VAULT_TOKEN", "")
        self._mount_point = mount_point
        self._cache_ttl = cache_ttl
        self._cache: dict[str, tuple[bytes, float]] = {}
        self._lock = threading.Lock()

    def get_key(self, key_id: str) -> bytes:
        """Get key from Vault (uses transit for wrapping only)."""
        # For Vault transit, we generate local data keys and use transit for wrapping
        # This returns a locally generated key that will be wrapped by Vault
        return os.urandom(32)

    def encrypt_data_key(self, data_key: bytes, key_id: str) -> bytes:
        """Encrypt data key using Vault transit."""
        try:
            import urllib.request

            # Base64 encode the data key
            plaintext_b64 = base64.b64encode(data_key).decode("utf-8")

            url = f"{self._url}/v1/{self._mount_point}/encrypt/{key_id}"
            payload = json.dumps({"plaintext": plaintext_b64}).encode("utf-8")

            request = urllib.request.Request(
                url,
                data=payload,
                headers={
                    "X-Vault-Token": self._token,
                    "Content-Type": "application/json",
                },
                method="POST",
            )

            with urllib.request.urlopen(request, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
                ciphertext = data["data"]["ciphertext"]
                return ciphertext.encode("utf-8")

        except Exception as e:
            raise EncryptionError(f"Vault encrypt failed: {e}")

    def decrypt_data_key(self, encrypted_key: bytes, key_id: str) -> bytes:
        """Decrypt data key using Vault transit."""
        try:
            import urllib.request

            ciphertext = encrypted_key.decode("utf-8")

            url = f"{self._url}/v1/{self._mount_point}/decrypt/{key_id}"
            payload = json.dumps({"ciphertext": ciphertext}).encode("utf-8")

            request = urllib.request.Request(
                url,
                data=payload,
                headers={
                    "X-Vault-Token": self._token,
                    "Content-Type": "application/json",
                },
                method="POST",
            )

            with urllib.request.urlopen(request, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
                plaintext_b64 = data["data"]["plaintext"]
                return base64.b64decode(plaintext_b64)

        except Exception as e:
            raise DecryptionError(f"Vault decrypt failed: {e}")


class AwsKmsProvider(KeyProvider):
    """AWS KMS key provider.

    Uses AWS KMS for key management and envelope encryption.
    """

    def __init__(
        self,
        key_id: str,
        *,
        region: str | None = None,
        cache_ttl: float = 300.0,
    ) -> None:
        """Initialize AWS KMS provider.

        Args:
            key_id: KMS key ID or alias (e.g., alias/my-key).
            region: AWS region.
            cache_ttl: Key cache TTL.
        """
        self._key_id = key_id
        self._region = region or os.getenv("AWS_REGION", "us-east-1")
        self._cache_ttl = cache_ttl
        self._client = None
        self._lock = threading.Lock()

    def _get_client(self) -> Any:
        """Get or create KMS client."""
        if self._client is None:
            try:
                import boto3

                self._client = boto3.client("kms", region_name=self._region)
            except ImportError:
                raise EncryptionError("boto3 not installed")
        return self._client

    def get_key(self, key_id: str) -> bytes:
        """Generate data key from KMS."""
        plain_key, _ = self.generate_data_key(key_id)
        return plain_key

    def encrypt_data_key(self, data_key: bytes, key_id: str) -> bytes:
        """Encrypt data key with KMS."""
        try:
            client = self._get_client()
            response = client.encrypt(
                KeyId=key_id or self._key_id,
                Plaintext=data_key,
            )
            return response["CiphertextBlob"]
        except Exception as e:
            raise EncryptionError(f"AWS KMS encrypt failed: {e}")

    def decrypt_data_key(self, encrypted_key: bytes, key_id: str) -> bytes:
        """Decrypt data key with KMS."""
        try:
            client = self._get_client()
            response = client.decrypt(
                KeyId=key_id or self._key_id,
                CiphertextBlob=encrypted_key,
            )
            return response["Plaintext"]
        except Exception as e:
            raise DecryptionError(f"AWS KMS decrypt failed: {e}")

    def generate_data_key(self, key_id: str, key_length: int = 32) -> tuple[bytes, bytes]:
        """Generate data key using KMS GenerateDataKey."""
        try:
            client = self._get_client()
            key_spec = "AES_256" if key_length == 32 else "AES_128"
            response = client.generate_data_key(
                KeyId=key_id or self._key_id,
                KeySpec=key_spec,
            )
            return response["Plaintext"], response["CiphertextBlob"]
        except Exception as e:
            raise EncryptionError(f"AWS KMS generate data key failed: {e}")


class GcpKmsProvider(KeyProvider):
    """Google Cloud KMS key provider.

    Uses Google Cloud KMS for key management.
    """

    def __init__(
        self,
        key_name: str,
        *,
        project_id: str | None = None,
        location: str = "global",
        key_ring: str = "truthound",
    ) -> None:
        """Initialize GCP KMS provider.

        Args:
            key_name: KMS key name.
            project_id: GCP project ID.
            location: KMS location.
            key_ring: Key ring name.
        """
        self._key_name = key_name
        self._project_id = project_id or os.getenv("GCP_PROJECT_ID", "")
        self._location = location
        self._key_ring = key_ring
        self._client = None

    def _get_client(self) -> Any:
        """Get or create KMS client."""
        if self._client is None:
            try:
                from google.cloud import kms

                self._client = kms.KeyManagementServiceClient()
            except ImportError:
                raise EncryptionError("google-cloud-kms not installed")
        return self._client

    def _get_key_path(self, key_id: str) -> str:
        """Get full key path."""
        key = key_id or self._key_name
        return f"projects/{self._project_id}/locations/{self._location}/keyRings/{self._key_ring}/cryptoKeys/{key}"

    def get_key(self, key_id: str) -> bytes:
        """Generate data key."""
        plain_key, _ = self.generate_data_key(key_id)
        return plain_key

    def encrypt_data_key(self, data_key: bytes, key_id: str) -> bytes:
        """Encrypt data key with GCP KMS."""
        try:
            client = self._get_client()
            key_path = self._get_key_path(key_id)
            response = client.encrypt(
                request={"name": key_path, "plaintext": data_key}
            )
            return response.ciphertext
        except Exception as e:
            raise EncryptionError(f"GCP KMS encrypt failed: {e}")

    def decrypt_data_key(self, encrypted_key: bytes, key_id: str) -> bytes:
        """Decrypt data key with GCP KMS."""
        try:
            client = self._get_client()
            key_path = self._get_key_path(key_id)
            response = client.decrypt(
                request={"name": key_path, "ciphertext": encrypted_key}
            )
            return response.plaintext
        except Exception as e:
            raise DecryptionError(f"GCP KMS decrypt failed: {e}")


class AzureKeyVaultProvider(KeyProvider):
    """Azure Key Vault key provider.

    Uses Azure Key Vault for key management.
    """

    def __init__(
        self,
        vault_url: str,
        key_name: str,
        *,
        credential: Any = None,
    ) -> None:
        """Initialize Azure Key Vault provider.

        Args:
            vault_url: Key Vault URL.
            key_name: Key name.
            credential: Azure credential (DefaultAzureCredential if None).
        """
        self._vault_url = vault_url
        self._key_name = key_name
        self._credential = credential
        self._client = None
        self._crypto_client = None

    def _get_clients(self) -> tuple[Any, Any]:
        """Get or create Key Vault clients."""
        if self._client is None:
            try:
                from azure.identity import DefaultAzureCredential
                from azure.keyvault.keys import KeyClient
                from azure.keyvault.keys.crypto import CryptographyClient

                credential = self._credential or DefaultAzureCredential()
                self._client = KeyClient(
                    vault_url=self._vault_url,
                    credential=credential,
                )
                key = self._client.get_key(self._key_name)
                self._crypto_client = CryptographyClient(key, credential=credential)
            except ImportError:
                raise EncryptionError("azure-keyvault-keys not installed")

        return self._client, self._crypto_client

    def get_key(self, key_id: str) -> bytes:
        """Generate data key."""
        return os.urandom(32)

    def encrypt_data_key(self, data_key: bytes, key_id: str) -> bytes:
        """Encrypt data key with Azure Key Vault."""
        try:
            from azure.keyvault.keys.crypto import EncryptionAlgorithm as AzureAlgorithm

            _, crypto_client = self._get_clients()
            result = crypto_client.encrypt(AzureAlgorithm.rsa_oaep, data_key)
            return result.ciphertext
        except Exception as e:
            raise EncryptionError(f"Azure Key Vault encrypt failed: {e}")

    def decrypt_data_key(self, encrypted_key: bytes, key_id: str) -> bytes:
        """Decrypt data key with Azure Key Vault."""
        try:
            from azure.keyvault.keys.crypto import EncryptionAlgorithm as AzureAlgorithm

            _, crypto_client = self._get_clients()
            result = crypto_client.decrypt(AzureAlgorithm.rsa_oaep, encrypted_key)
            return result.plaintext
        except Exception as e:
            raise DecryptionError(f"Azure Key Vault decrypt failed: {e}")


# =============================================================================
# Local Key Provider
# =============================================================================


class LocalKeyProvider(KeyProvider):
    """Local key provider for development/testing.

    Stores keys locally using password-based encryption.
    NOT recommended for production use.
    """

    def __init__(
        self,
        key_file: str | Path = ".truthound_keys",
        *,
        master_password: str | None = None,
    ) -> None:
        """Initialize local key provider.

        Args:
            key_file: Path to key storage file.
            master_password: Master password (or TRUTHOUND_MASTER_KEY env).
        """
        self._key_file = Path(key_file)
        self._master_password = master_password or os.getenv("TRUTHOUND_MASTER_KEY", "")
        self._keys: dict[str, bytes] = {}
        self._lock = threading.Lock()
        self._load_keys()

    def _load_keys(self) -> None:
        """Load keys from file."""
        if not self._key_file.exists():
            return

        try:
            content = self._key_file.read_bytes()
            # Simple XOR with master key hash for obfuscation (not secure!)
            key_hash = hashlib.sha256(self._master_password.encode()).digest()
            decrypted = bytes(b ^ key_hash[i % 32] for i, b in enumerate(content))
            self._keys = json.loads(decrypted.decode("utf-8"))
            # Convert hex strings back to bytes
            self._keys = {k: bytes.fromhex(v) for k, v in self._keys.items()}
        except Exception:
            self._keys = {}

    def _save_keys(self) -> None:
        """Save keys to file."""
        try:
            # Convert bytes to hex for JSON
            keys_hex = {k: v.hex() for k, v in self._keys.items()}
            content = json.dumps(keys_hex).encode("utf-8")
            # Simple XOR with master key hash
            key_hash = hashlib.sha256(self._master_password.encode()).digest()
            encrypted = bytes(b ^ key_hash[i % 32] for i, b in enumerate(content))
            self._key_file.write_bytes(encrypted)
        except Exception:
            pass

    def get_key(self, key_id: str) -> bytes:
        """Get or generate key."""
        with self._lock:
            if key_id not in self._keys:
                self._keys[key_id] = os.urandom(32)
                self._save_keys()
            return self._keys[key_id]

    def encrypt_data_key(self, data_key: bytes, key_id: str) -> bytes:
        """Encrypt data key with master key."""
        master_key = self.get_key(key_id)
        encryptor = AesGcmEncryptor()
        return encryptor.encrypt(data_key, master_key)

    def decrypt_data_key(self, encrypted_key: bytes, key_id: str) -> bytes:
        """Decrypt data key with master key."""
        master_key = self.get_key(key_id)
        encryptor = AesGcmEncryptor()
        return encryptor.decrypt(encrypted_key, master_key)


# =============================================================================
# At-Rest Encryption
# =============================================================================


@dataclass
class EncryptedData:
    """Encrypted data with metadata.

    Attributes:
        ciphertext: Encrypted data.
        encrypted_key: Wrapped data encryption key.
        key_id: Master key ID used for wrapping.
        algorithm: Encryption algorithm used.
        nonce: Nonce/IV used for encryption.
        timestamp: Encryption timestamp.
        metadata: Additional metadata.
    """

    ciphertext: bytes
    encrypted_key: bytes
    key_id: str
    algorithm: str = "AES-256-GCM"
    nonce: bytes = b""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_bytes(self) -> bytes:
        """Serialize to bytes."""
        header = {
            "key_id": self.key_id,
            "algorithm": self.algorithm,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }
        header_json = json.dumps(header).encode("utf-8")

        # Format: header_len (4) | header | nonce_len (1) | nonce | key_len (2) | encrypted_key | ciphertext
        parts = [
            struct.pack(">I", len(header_json)),
            header_json,
            struct.pack(">B", len(self.nonce)),
            self.nonce,
            struct.pack(">H", len(self.encrypted_key)),
            self.encrypted_key,
            self.ciphertext,
        ]
        return b"".join(parts)

    @classmethod
    def from_bytes(cls, data: bytes) -> "EncryptedData":
        """Deserialize from bytes."""
        offset = 0

        # Read header
        header_len = struct.unpack(">I", data[offset : offset + 4])[0]
        offset += 4
        header = json.loads(data[offset : offset + header_len].decode("utf-8"))
        offset += header_len

        # Read nonce
        nonce_len = struct.unpack(">B", data[offset : offset + 1])[0]
        offset += 1
        nonce = data[offset : offset + nonce_len]
        offset += nonce_len

        # Read encrypted key
        key_len = struct.unpack(">H", data[offset : offset + 2])[0]
        offset += 2
        encrypted_key = data[offset : offset + key_len]
        offset += key_len

        # Rest is ciphertext
        ciphertext = data[offset:]

        return cls(
            ciphertext=ciphertext,
            encrypted_key=encrypted_key,
            key_id=header["key_id"],
            algorithm=header["algorithm"],
            nonce=nonce,
            timestamp=datetime.fromisoformat(header["timestamp"]),
            metadata=header.get("metadata", {}),
        )


class AtRestEncryption:
    """At-rest encryption for data files.

    Provides envelope encryption using a cloud KMS or local key provider.
    Data is encrypted with a unique data encryption key (DEK), which is
    then wrapped with a key encryption key (KEK) from the provider.

    Example:
        >>> encryptor = AtRestEncryption(
        ...     provider=AwsKmsProvider("alias/my-key"),
        ...     key_id="alias/my-key",
        ... )
        >>>
        >>> # Encrypt
        >>> encrypted = encryptor.encrypt(b"sensitive data")
        >>>
        >>> # Decrypt
        >>> decrypted = encryptor.decrypt(encrypted)
    """

    def __init__(
        self,
        provider: KeyProvider,
        key_id: str,
        *,
        algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM,
    ) -> None:
        """Initialize at-rest encryption.

        Args:
            provider: Key provider.
            key_id: Default key ID for encryption.
            algorithm: Encryption algorithm.
        """
        self._provider = provider
        self._key_id = key_id
        self._algorithm = algorithm

        # Get encryptor for the algorithm
        if algorithm == EncryptionAlgorithm.AES_256_GCM:
            self._encryptor = AesGcmEncryptor()
        elif algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
            self._encryptor = ChaCha20Poly1305Encryptor()
        else:
            self._encryptor = AesGcmEncryptor()

    def encrypt(
        self,
        data: bytes,
        *,
        key_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EncryptedData:
        """Encrypt data.

        Args:
            data: Data to encrypt.
            key_id: Key ID (uses default if None).
            metadata: Additional metadata.

        Returns:
            EncryptedData with ciphertext and wrapped key.
        """
        key_id = key_id or self._key_id

        # Generate data key
        plain_key, encrypted_key = self._provider.generate_data_key(key_id)

        # Generate nonce
        nonce = generate_nonce(self._algorithm)

        # Encrypt data
        ciphertext = self._encryptor.encrypt(data, plain_key, nonce)

        # Clear plain key from memory
        plain_key = b"\x00" * len(plain_key)

        return EncryptedData(
            ciphertext=ciphertext,
            encrypted_key=encrypted_key,
            key_id=key_id,
            algorithm=self._algorithm.value,
            nonce=nonce,
            metadata=metadata or {},
        )

    def decrypt(self, encrypted: EncryptedData) -> bytes:
        """Decrypt data.

        Args:
            encrypted: Encrypted data.

        Returns:
            Decrypted data.
        """
        # Decrypt data key
        plain_key = self._provider.decrypt_data_key(
            encrypted.encrypted_key,
            encrypted.key_id,
        )

        # Decrypt data
        try:
            return self._encryptor.decrypt(encrypted.ciphertext, plain_key)
        finally:
            # Clear plain key from memory
            plain_key = b"\x00" * len(plain_key)

    def encrypt_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
        **kwargs: Any,
    ) -> None:
        """Encrypt a file.

        Args:
            input_path: Input file path.
            output_path: Output file path.
            **kwargs: Additional arguments for encrypt().
        """
        data = Path(input_path).read_bytes()
        encrypted = self.encrypt(data, **kwargs)
        Path(output_path).write_bytes(encrypted.to_bytes())

    def decrypt_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
    ) -> None:
        """Decrypt a file.

        Args:
            input_path: Encrypted file path.
            output_path: Output file path.
        """
        data = Path(input_path).read_bytes()
        encrypted = EncryptedData.from_bytes(data)
        decrypted = self.decrypt(encrypted)
        Path(output_path).write_bytes(decrypted)


# =============================================================================
# Field-Level Encryption
# =============================================================================


@dataclass
class FieldEncryptionPolicy:
    """Encryption policy for a field/column.

    Attributes:
        algorithm: Encryption algorithm.
        key_id: Key ID to use.
        format_preserving: Use format-preserving encryption.
        deterministic: Use deterministic encryption (for searching).
        mask_format: Masking format for display.
    """

    algorithm: str = "aes_gcm"  # aes_gcm, chacha20, format_preserving
    key_id: str = ""
    format_preserving: bool = False
    deterministic: bool = False
    mask_format: str = ""  # e.g., "***-**-{last4}" for SSN


class FieldLevelEncryption:
    """Field-level encryption for sensitive columns.

    Encrypts individual fields/columns based on policies.
    Supports format-preserving encryption for data that must
    maintain its format (e.g., credit card numbers).

    Example:
        >>> fle = FieldLevelEncryption(
        ...     provider=AwsKmsProvider("alias/my-key"),
        ...     policies={
        ...         "ssn": FieldEncryptionPolicy(format_preserving=True),
        ...         "email": FieldEncryptionPolicy(algorithm="aes_gcm"),
        ...     },
        ... )
        >>>
        >>> # Encrypt field
        >>> encrypted_ssn = fle.encrypt_field("ssn", "123-45-6789")
        >>>
        >>> # Decrypt field
        >>> ssn = fle.decrypt_field("ssn", encrypted_ssn)
    """

    def __init__(
        self,
        provider: KeyProvider,
        policies: dict[str, FieldEncryptionPolicy] | None = None,
        *,
        default_key_id: str = "",
    ) -> None:
        """Initialize field-level encryption.

        Args:
            provider: Key provider.
            policies: Field encryption policies.
            default_key_id: Default key ID.
        """
        self._provider = provider
        self._policies = policies or {}
        self._default_key_id = default_key_id
        self._encryptor = AesGcmEncryptor()

    def add_policy(self, field_name: str, policy: FieldEncryptionPolicy) -> None:
        """Add encryption policy for a field.

        Args:
            field_name: Field name.
            policy: Encryption policy.
        """
        self._policies[field_name] = policy

    def encrypt_field(self, field_name: str, value: str) -> str:
        """Encrypt a field value.

        Args:
            field_name: Field name.
            value: Value to encrypt.

        Returns:
            Encrypted value (base64 encoded).
        """
        policy = self._policies.get(field_name, FieldEncryptionPolicy())
        key_id = policy.key_id or self._default_key_id

        if policy.format_preserving:
            return self._format_preserving_encrypt(value, key_id)

        # Get key
        key = self._provider.get_key(key_id)

        if policy.deterministic:
            # Use fixed nonce derived from value hash (for searching)
            nonce = hashlib.sha256(value.encode()).digest()[:12]
        else:
            nonce = generate_nonce(EncryptionAlgorithm.AES_256_GCM)

        # Encrypt
        ciphertext = self._encryptor.encrypt(value.encode("utf-8"), key, nonce)

        # Encode as base64
        return base64.b64encode(nonce + ciphertext).decode("utf-8")

    def decrypt_field(self, field_name: str, encrypted_value: str) -> str:
        """Decrypt a field value.

        Args:
            field_name: Field name.
            encrypted_value: Encrypted value (base64 encoded).

        Returns:
            Decrypted value.
        """
        policy = self._policies.get(field_name, FieldEncryptionPolicy())
        key_id = policy.key_id or self._default_key_id

        if policy.format_preserving:
            return self._format_preserving_decrypt(encrypted_value, key_id)

        # Decode base64
        data = base64.b64decode(encrypted_value)
        nonce = data[:12]
        ciphertext = data[12:]

        # Get key
        key = self._provider.get_key(key_id)

        # Decrypt
        plaintext = self._encryptor.decrypt(ciphertext, key)
        return plaintext.decode("utf-8")

    def _format_preserving_encrypt(self, value: str, key_id: str) -> str:
        """Format-preserving encryption (simplified).

        This is a simplified FPE implementation. For production,
        use a proper FPE library like python-ff3.
        """
        # Get key
        key = self._provider.get_key(key_id)

        # Simple FPE: XOR with key-derived stream
        key_stream = hashlib.sha256(key + value.encode()).digest()
        result = []

        for i, char in enumerate(value):
            if char.isdigit():
                offset = key_stream[i % 32] % 10
                new_digit = (int(char) + offset) % 10
                result.append(str(new_digit))
            elif char.isalpha():
                offset = key_stream[i % 32] % 26
                if char.isupper():
                    new_char = chr((ord(char) - ord("A") + offset) % 26 + ord("A"))
                else:
                    new_char = chr((ord(char) - ord("a") + offset) % 26 + ord("a"))
                result.append(new_char)
            else:
                result.append(char)

        return "".join(result)

    def _format_preserving_decrypt(self, value: str, key_id: str) -> str:
        """Format-preserving decryption."""
        key = self._provider.get_key(key_id)

        # Reverse the FPE
        key_stream = hashlib.sha256(key + value.encode()).digest()
        result = []

        for i, char in enumerate(value):
            if char.isdigit():
                offset = key_stream[i % 32] % 10
                new_digit = (int(char) - offset) % 10
                result.append(str(new_digit))
            elif char.isalpha():
                offset = key_stream[i % 32] % 26
                if char.isupper():
                    new_char = chr((ord(char) - ord("A") - offset) % 26 + ord("A"))
                else:
                    new_char = chr((ord(char) - ord("a") - offset) % 26 + ord("a"))
                result.append(new_char)
            else:
                result.append(char)

        return "".join(result)

    def mask_field(self, field_name: str, value: str) -> str:
        """Mask a field value for display.

        Args:
            field_name: Field name.
            value: Value to mask.

        Returns:
            Masked value.
        """
        policy = self._policies.get(field_name, FieldEncryptionPolicy())

        if policy.mask_format:
            # Apply mask format
            if "{last4}" in policy.mask_format:
                last4 = value[-4:] if len(value) >= 4 else value
                return policy.mask_format.replace("{last4}", last4)
            return policy.mask_format

        # Default masking
        if len(value) <= 4:
            return "*" * len(value)
        return value[:2] + "*" * (len(value) - 4) + value[-2:]


# =============================================================================
# Encryption Configuration
# =============================================================================


@dataclass
class EnterpriseEncryptionConfig:
    """Enterprise encryption configuration.

    Example:
        >>> config = EnterpriseEncryptionConfig(
        ...     provider="aws_kms",
        ...     key_id="alias/truthound-data-key",
        ...     region="us-east-1",
        ... )
    """

    enabled: bool = True
    provider: str = "local"  # local, vault, aws_kms, gcp_kms, azure_keyvault

    # Provider-specific settings
    key_id: str = ""

    # Vault settings
    vault_url: str = ""
    vault_token: str = ""
    vault_mount_point: str = "transit"

    # AWS settings
    aws_region: str = ""

    # GCP settings
    gcp_project_id: str = ""
    gcp_location: str = "global"
    gcp_key_ring: str = "truthound"

    # Azure settings
    azure_vault_url: str = ""

    # Local settings
    local_key_file: str = ".truthound_keys"
    local_master_password: str = ""

    # Field-level encryption
    field_policies: dict[str, dict[str, Any]] = field(default_factory=dict)


# =============================================================================
# Global Encryption
# =============================================================================

_global_provider: KeyProvider | None = None
_global_encryptor: AtRestEncryption | None = None
_lock = threading.Lock()


def configure_encryption(
    *,
    provider: str = "local",
    key_id: str = "",
    vault_url: str = "",
    aws_region: str = "",
    gcp_project_id: str = "",
    azure_vault_url: str = "",
    **kwargs: Any,
) -> AtRestEncryption:
    """Configure global encryption.

    Args:
        provider: Key provider type.
        key_id: Default key ID.
        vault_url: Vault URL (if using Vault).
        aws_region: AWS region (if using AWS KMS).
        gcp_project_id: GCP project ID (if using GCP KMS).
        azure_vault_url: Azure Key Vault URL.
        **kwargs: Additional provider configuration.

    Returns:
        Configured AtRestEncryption instance.
    """
    global _global_provider, _global_encryptor

    with _lock:
        # Create provider
        if provider == "vault":
            _global_provider = VaultKeyProvider(
                vault_url,
                token=kwargs.get("vault_token"),
                mount_point=kwargs.get("vault_mount_point", "transit"),
            )
        elif provider == "aws_kms":
            _global_provider = AwsKmsProvider(
                key_id,
                region=aws_region,
            )
        elif provider == "gcp_kms":
            _global_provider = GcpKmsProvider(
                key_id,
                project_id=gcp_project_id,
                location=kwargs.get("gcp_location", "global"),
                key_ring=kwargs.get("gcp_key_ring", "truthound"),
            )
        elif provider == "azure_keyvault":
            _global_provider = AzureKeyVaultProvider(
                azure_vault_url,
                key_id,
            )
        else:  # local
            _global_provider = LocalKeyProvider(
                key_file=kwargs.get("local_key_file", ".truthound_keys"),
                master_password=kwargs.get("local_master_password"),
            )

        _global_encryptor = AtRestEncryption(_global_provider, key_id)
        return _global_encryptor


def get_encryptor() -> AtRestEncryption:
    """Get the global encryptor.

    Returns:
        AtRestEncryption instance.
    """
    global _global_encryptor

    with _lock:
        if _global_encryptor is None:
            _global_encryptor = configure_encryption(provider="local", key_id="default")
        return _global_encryptor
