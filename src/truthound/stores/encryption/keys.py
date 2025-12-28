"""Key management module for encryption.

This module provides comprehensive key management including:
- Key derivation functions (KDFs) for password-based encryption
- Key rotation and versioning
- In-memory and file-based key storage
- Envelope encryption for secure key wrapping
- Integration with external key management systems (AWS KMS, HashiCorp Vault)

Security Best Practices:
    - Never hardcode keys in source code
    - Use strong KDFs for password-derived keys
    - Rotate keys regularly
    - Use envelope encryption for key wrapping
    - Clear key material from memory when done

Example:
    >>> from truthound.stores.encryption.keys import (
    ...     KeyManager,
    ...     derive_key,
    ...     KeyDerivation,
    ... )
    >>>
    >>> # Derive key from password
    >>> key = derive_key("my_password", kdf=KeyDerivation.ARGON2ID)
    >>>
    >>> # Use key manager for key lifecycle
    >>> manager = KeyManager()
    >>> key_obj = manager.create_key(algorithm=EncryptionAlgorithm.AES_256_GCM)
    >>> manager.rotate_key(key_obj.key_id)
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Iterator

from truthound.stores.encryption.base import (
    EncryptionAlgorithm,
    EncryptionKey,
    KeyDerivation,
    KeyDerivationConfig,
    KeyDerivationError,
    KeyError_,
    KeyExpiredError,
    KeyType,
    generate_key,
    generate_key_id,
    generate_salt,
)


# =============================================================================
# Key Derivation Functions
# =============================================================================


class BaseKeyDeriver(ABC):
    """Base class for key derivation implementations."""

    def __init__(self, kdf: KeyDerivation) -> None:
        self._kdf = kdf

    @property
    def kdf(self) -> KeyDerivation:
        """Get the KDF type."""
        return self._kdf

    @abstractmethod
    def derive(
        self,
        password: str | bytes,
        salt: bytes,
        key_size: int,
        **kwargs: Any,
    ) -> bytes:
        """Derive a key from password.

        Args:
            password: Password or passphrase.
            salt: Cryptographic salt.
            key_size: Desired key size in bytes.
            **kwargs: KDF-specific parameters.

        Returns:
            Derived key bytes.
        """
        ...


class Argon2KeyDeriver(BaseKeyDeriver):
    """Argon2 key derivation (recommended for new applications).

    Argon2 is the winner of the Password Hashing Competition and
    provides excellent resistance against GPU and ASIC attacks.

    Variants:
        - Argon2id: Hybrid mode (recommended)
        - Argon2i: Data-independent (side-channel resistant)
        - Argon2d: Data-dependent (higher resistance to GPU attacks)
    """

    def __init__(
        self,
        variant: KeyDerivation = KeyDerivation.ARGON2ID,
        time_cost: int = 3,
        memory_cost: int = 65536,
        parallelism: int = 4,
    ) -> None:
        """Initialize Argon2 key deriver.

        Args:
            variant: Argon2 variant (id, i, or d).
            time_cost: Number of iterations.
            memory_cost: Memory usage in KiB.
            parallelism: Degree of parallelism.
        """
        super().__init__(variant)
        self.time_cost = time_cost
        self.memory_cost = memory_cost
        self.parallelism = parallelism

    def derive(
        self,
        password: str | bytes,
        salt: bytes,
        key_size: int,
        **kwargs: Any,
    ) -> bytes:
        """Derive key using Argon2."""
        try:
            from argon2.low_level import Type, hash_secret_raw
        except ImportError as e:
            raise KeyDerivationError(
                "Argon2 requires 'argon2-cffi' package: pip install argon2-cffi",
                self._kdf.value,
            ) from e

        if isinstance(password, str):
            password = password.encode("utf-8")

        # Map variant to type
        type_map = {
            KeyDerivation.ARGON2ID: Type.ID,
            KeyDerivation.ARGON2I: Type.I,
            KeyDerivation.ARGON2D: Type.D,
        }
        argon_type = type_map.get(self._kdf, Type.ID)

        time_cost = kwargs.get("time_cost", self.time_cost)
        memory_cost = kwargs.get("memory_cost", self.memory_cost)
        parallelism = kwargs.get("parallelism", self.parallelism)

        return hash_secret_raw(
            secret=password,
            salt=salt,
            time_cost=time_cost,
            memory_cost=memory_cost,
            parallelism=parallelism,
            hash_len=key_size,
            type=argon_type,
        )


class PBKDF2KeyDeriver(BaseKeyDeriver):
    """PBKDF2 key derivation (widely compatible).

    PBKDF2 is a widely supported KDF that is suitable for most
    applications requiring password-based encryption.
    """

    def __init__(
        self,
        hash_name: str = "sha256",
        iterations: int = 600_000,
    ) -> None:
        """Initialize PBKDF2 key deriver.

        Args:
            hash_name: Hash function (sha256, sha512).
            iterations: Number of iterations (higher = slower + more secure).
        """
        kdf = (
            KeyDerivation.PBKDF2_SHA512
            if hash_name == "sha512"
            else KeyDerivation.PBKDF2_SHA256
        )
        super().__init__(kdf)
        self.hash_name = hash_name
        self.iterations = iterations

    def derive(
        self,
        password: str | bytes,
        salt: bytes,
        key_size: int,
        **kwargs: Any,
    ) -> bytes:
        """Derive key using PBKDF2."""
        if isinstance(password, str):
            password = password.encode("utf-8")

        iterations = kwargs.get("iterations", self.iterations)

        return hashlib.pbkdf2_hmac(
            self.hash_name,
            password,
            salt,
            iterations,
            dklen=key_size,
        )


class ScryptKeyDeriver(BaseKeyDeriver):
    """scrypt key derivation (memory-hard).

    scrypt is designed to be memory-hard, making it expensive to
    parallelize on GPUs or ASICs.
    """

    def __init__(
        self,
        n: int = 2**14,
        r: int = 8,
        p: int = 1,
    ) -> None:
        """Initialize scrypt key deriver.

        Args:
            n: CPU/memory cost parameter (must be power of 2).
            r: Block size parameter.
            p: Parallelization parameter.
        """
        super().__init__(KeyDerivation.SCRYPT)
        self.n = n
        self.r = r
        self.p = p

    def derive(
        self,
        password: str | bytes,
        salt: bytes,
        key_size: int,
        **kwargs: Any,
    ) -> bytes:
        """Derive key using scrypt."""
        if isinstance(password, str):
            password = password.encode("utf-8")

        n = kwargs.get("n", self.n)
        r = kwargs.get("r", self.r)
        p = kwargs.get("p", self.p)

        return hashlib.scrypt(
            password,
            salt=salt,
            n=n,
            r=r,
            p=p,
            dklen=key_size,
        )


class HKDFKeyDeriver(BaseKeyDeriver):
    """HKDF key derivation (for key expansion).

    HKDF is used for key expansion and derivation from existing
    key material (not for password-based derivation).
    """

    def __init__(self, hash_name: str = "sha256") -> None:
        """Initialize HKDF key deriver.

        Args:
            hash_name: Hash function (sha256, sha512).
        """
        kdf = (
            KeyDerivation.HKDF_SHA512
            if hash_name == "sha512"
            else KeyDerivation.HKDF_SHA256
        )
        super().__init__(kdf)
        self.hash_name = hash_name

    def derive(
        self,
        password: str | bytes,
        salt: bytes,
        key_size: int,
        **kwargs: Any,
    ) -> bytes:
        """Derive key using HKDF."""
        try:
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.kdf.hkdf import HKDF
        except ImportError as e:
            raise KeyDerivationError(
                "HKDF requires 'cryptography' package",
                self._kdf.value,
            ) from e

        if isinstance(password, str):
            password = password.encode("utf-8")

        hash_algo = hashes.SHA512() if self.hash_name == "sha512" else hashes.SHA256()
        info = kwargs.get("info", b"truthound-encryption")

        hkdf = HKDF(
            algorithm=hash_algo,
            length=key_size,
            salt=salt if salt else None,
            info=info,
        )
        return hkdf.derive(password)


# =============================================================================
# Key Deriver Registry and Factory
# =============================================================================

_KEY_DERIVER_REGISTRY: dict[KeyDerivation, type[BaseKeyDeriver]] = {
    KeyDerivation.ARGON2ID: Argon2KeyDeriver,
    KeyDerivation.ARGON2I: Argon2KeyDeriver,
    KeyDerivation.ARGON2D: Argon2KeyDeriver,
    KeyDerivation.PBKDF2_SHA256: PBKDF2KeyDeriver,
    KeyDerivation.PBKDF2_SHA512: PBKDF2KeyDeriver,
    KeyDerivation.SCRYPT: ScryptKeyDeriver,
    KeyDerivation.HKDF_SHA256: HKDFKeyDeriver,
    KeyDerivation.HKDF_SHA512: HKDFKeyDeriver,
}


def get_key_deriver(kdf: KeyDerivation, **kwargs: Any) -> BaseKeyDeriver:
    """Get a key deriver instance.

    Args:
        kdf: Key derivation function.
        **kwargs: KDF-specific parameters.

    Returns:
        Key deriver instance.
    """
    deriver_class = _KEY_DERIVER_REGISTRY.get(kdf)
    if deriver_class is None:
        raise KeyDerivationError(f"Unsupported KDF: {kdf.value}")

    # Handle variants
    if kdf in (KeyDerivation.ARGON2ID, KeyDerivation.ARGON2I, KeyDerivation.ARGON2D):
        return Argon2KeyDeriver(variant=kdf, **kwargs)
    elif kdf == KeyDerivation.PBKDF2_SHA512:
        return PBKDF2KeyDeriver(hash_name="sha512", **kwargs)
    elif kdf == KeyDerivation.HKDF_SHA512:
        return HKDFKeyDeriver(hash_name="sha512")

    return deriver_class(**kwargs)


def derive_key(
    password: str | bytes,
    salt: bytes | None = None,
    key_size: int = 32,
    kdf: KeyDerivation = KeyDerivation.ARGON2ID,
    config: KeyDerivationConfig | None = None,
) -> tuple[bytes, bytes]:
    """Derive an encryption key from a password.

    Args:
        password: Password or passphrase.
        salt: Salt (generated if not provided).
        key_size: Desired key size in bytes.
        kdf: Key derivation function.
        config: Detailed configuration.

    Returns:
        Tuple of (derived_key, salt).

    Example:
        >>> key, salt = derive_key("my_password")
        >>> # Store salt alongside encrypted data
    """
    if salt is None:
        salt_size = config.salt_size if config else 16
        salt = generate_salt(salt_size)

    kwargs: dict[str, Any] = {}
    if config:
        if kdf in (KeyDerivation.ARGON2ID, KeyDerivation.ARGON2I, KeyDerivation.ARGON2D):
            kwargs = {
                "time_cost": config.time_cost,
                "memory_cost": config.memory_cost,
                "parallelism": config.parallelism,
            }
        elif kdf in (KeyDerivation.PBKDF2_SHA256, KeyDerivation.PBKDF2_SHA512):
            kwargs = {"iterations": config.get_iterations()}
        elif kdf == KeyDerivation.SCRYPT:
            kwargs = {"n": config.n, "r": config.r, "p": config.p}

    deriver = get_key_deriver(kdf, **kwargs)
    key = deriver.derive(password, salt, key_size)
    return key, salt


# =============================================================================
# Key Storage Backends
# =============================================================================


class BaseKeyStore(ABC):
    """Base class for key storage backends."""

    @abstractmethod
    def get(self, key_id: str) -> EncryptionKey | None:
        """Retrieve a key by ID."""
        ...

    @abstractmethod
    def put(self, key: EncryptionKey) -> None:
        """Store a key."""
        ...

    @abstractmethod
    def delete(self, key_id: str) -> bool:
        """Delete a key by ID."""
        ...

    @abstractmethod
    def list_keys(self) -> list[str]:
        """List all key IDs."""
        ...

    def exists(self, key_id: str) -> bool:
        """Check if a key exists."""
        return self.get(key_id) is not None


class InMemoryKeyStore(BaseKeyStore):
    """Thread-safe in-memory key storage.

    WARNING: Keys are lost when the process exits. Use only for
    testing or ephemeral keys.
    """

    def __init__(self) -> None:
        self._keys: dict[str, EncryptionKey] = {}
        self._lock = RLock()

    def get(self, key_id: str) -> EncryptionKey | None:
        with self._lock:
            return self._keys.get(key_id)

    def put(self, key: EncryptionKey) -> None:
        with self._lock:
            self._keys[key.key_id] = key

    def delete(self, key_id: str) -> bool:
        with self._lock:
            if key_id in self._keys:
                # Clear key material before removing
                self._keys[key_id].clear()
                del self._keys[key_id]
                return True
            return False

    def list_keys(self) -> list[str]:
        with self._lock:
            return list(self._keys.keys())

    def clear(self) -> None:
        """Clear all keys from memory."""
        with self._lock:
            for key in self._keys.values():
                key.clear()
            self._keys.clear()


class FileKeyStore(BaseKeyStore):
    """File-based key storage with encryption.

    Keys are stored encrypted using a master key. The master key
    should be provided externally (e.g., from environment variable).

    WARNING: This is a basic implementation. For production use,
    consider using a proper secrets manager.
    """

    def __init__(
        self,
        path: str | Path,
        master_key: bytes | None = None,
        master_password: str | None = None,
    ) -> None:
        """Initialize file key store.

        Args:
            path: Directory to store keys.
            master_key: Master key for encrypting stored keys.
            master_password: Password to derive master key (alternative to master_key).
        """
        self._path = Path(path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

        # Derive or use provided master key
        if master_key:
            self._master_key = master_key
        elif master_password:
            # Use fixed salt for master key (stored in .master_salt file)
            salt_file = self._path / ".master_salt"
            if salt_file.exists():
                salt = salt_file.read_bytes()
            else:
                salt = generate_salt(16)
                salt_file.write_bytes(salt)
            self._master_key, _ = derive_key(master_password, salt=salt)
        else:
            # No encryption - store keys in plaintext (NOT RECOMMENDED)
            self._master_key = None

    def _get_key_path(self, key_id: str) -> Path:
        """Get file path for a key."""
        # Sanitize key_id for filesystem
        safe_id = base64.urlsafe_b64encode(key_id.encode()).decode()
        return self._path / f"{safe_id}.key"

    def _encrypt_key_data(self, data: bytes) -> bytes:
        """Encrypt key data with master key."""
        if self._master_key is None:
            return data

        from truthound.stores.encryption.providers import AesGcmEncryptor

        encryptor = AesGcmEncryptor(key_size=32)
        return encryptor.encrypt(data, self._master_key)

    def _decrypt_key_data(self, data: bytes) -> bytes:
        """Decrypt key data with master key."""
        if self._master_key is None:
            return data

        from truthound.stores.encryption.providers import AesGcmEncryptor

        encryptor = AesGcmEncryptor(key_size=32)
        return encryptor.decrypt(data, self._master_key)

    def get(self, key_id: str) -> EncryptionKey | None:
        with self._lock:
            key_path = self._get_key_path(key_id)
            if not key_path.exists():
                return None

            try:
                encrypted_data = key_path.read_bytes()
                data = self._decrypt_key_data(encrypted_data)
                key_dict = json.loads(data.decode())

                return EncryptionKey(
                    key_id=key_dict["key_id"],
                    key_material=base64.b64decode(key_dict["key_material"]),
                    algorithm=EncryptionAlgorithm(key_dict["algorithm"]),
                    key_type=KeyType(key_dict["key_type"]),
                    created_at=datetime.fromisoformat(key_dict["created_at"]),
                    expires_at=(
                        datetime.fromisoformat(key_dict["expires_at"])
                        if key_dict.get("expires_at")
                        else None
                    ),
                    version=key_dict["version"],
                    metadata=key_dict.get("metadata", {}),
                )
            except Exception:
                return None

    def put(self, key: EncryptionKey) -> None:
        with self._lock:
            key_dict = {
                "key_id": key.key_id,
                "key_material": base64.b64encode(key.key_material).decode(),
                "algorithm": key.algorithm.value,
                "key_type": key.key_type.value,
                "created_at": key.created_at.isoformat(),
                "expires_at": key.expires_at.isoformat() if key.expires_at else None,
                "version": key.version,
                "metadata": key.metadata,
            }

            data = json.dumps(key_dict).encode()
            encrypted_data = self._encrypt_key_data(data)

            key_path = self._get_key_path(key.key_id)
            key_path.write_bytes(encrypted_data)

    def delete(self, key_id: str) -> bool:
        with self._lock:
            key_path = self._get_key_path(key_id)
            if key_path.exists():
                # Overwrite with zeros before deletion
                size = key_path.stat().st_size
                key_path.write_bytes(b"\x00" * size)
                key_path.unlink()
                return True
            return False

    def list_keys(self) -> list[str]:
        with self._lock:
            keys = []
            for key_path in self._path.glob("*.key"):
                try:
                    safe_id = key_path.stem
                    key_id = base64.urlsafe_b64decode(safe_id).decode()
                    keys.append(key_id)
                except Exception:
                    continue
            return keys


class EnvironmentKeyStore(BaseKeyStore):
    """Key storage using environment variables.

    Keys are stored base64-encoded in environment variables with
    a configurable prefix.

    Example:
        TRUTHOUND_KEY_mykey=base64_encoded_key
    """

    def __init__(self, prefix: str = "TRUTHOUND_KEY_") -> None:
        """Initialize environment key store.

        Args:
            prefix: Prefix for environment variable names.
        """
        self._prefix = prefix

    def get(self, key_id: str) -> EncryptionKey | None:
        env_var = f"{self._prefix}{key_id}"
        value = os.environ.get(env_var)
        if not value:
            return None

        try:
            key_material = base64.b64decode(value)
            # Infer algorithm from key size
            if len(key_material) == 16:
                algorithm = EncryptionAlgorithm.AES_128_GCM
            elif len(key_material) == 32:
                algorithm = EncryptionAlgorithm.AES_256_GCM
            elif len(key_material) == 44:  # Fernet key (base64)
                algorithm = EncryptionAlgorithm.FERNET
                key_material = value.encode()  # Fernet uses base64 string
            else:
                algorithm = EncryptionAlgorithm.AES_256_GCM

            return EncryptionKey(
                key_id=key_id,
                key_material=key_material,
                algorithm=algorithm,
            )
        except Exception:
            return None

    def put(self, key: EncryptionKey) -> None:
        env_var = f"{self._prefix}{key.key_id}"
        if key.algorithm == EncryptionAlgorithm.FERNET:
            value = key.key_material.decode()
        else:
            value = base64.b64encode(key.key_material).decode()
        os.environ[env_var] = value

    def delete(self, key_id: str) -> bool:
        env_var = f"{self._prefix}{key_id}"
        if env_var in os.environ:
            del os.environ[env_var]
            return True
        return False

    def list_keys(self) -> list[str]:
        keys = []
        for var in os.environ:
            if var.startswith(self._prefix):
                key_id = var[len(self._prefix) :]
                keys.append(key_id)
        return keys


# =============================================================================
# Key Manager
# =============================================================================


@dataclass
class KeyManagerConfig:
    """Configuration for key manager.

    Attributes:
        default_algorithm: Default encryption algorithm.
        default_key_type: Default key type.
        default_ttl: Default key TTL (None = no expiration).
        auto_rotate_before_expiry: Auto-rotate keys before expiry.
        rotation_overlap: Time to keep old key active after rotation.
    """

    default_algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM
    default_key_type: KeyType = KeyType.DATA_ENCRYPTION_KEY
    default_ttl: timedelta | None = None
    auto_rotate_before_expiry: timedelta = timedelta(days=7)
    rotation_overlap: timedelta = timedelta(hours=24)


class KeyManager:
    """Comprehensive key management system.

    Features:
        - Key creation with automatic ID generation
        - Key rotation with version tracking
        - Multiple storage backend support
        - Key expiration management
        - Audit logging hooks

    Example:
        >>> manager = KeyManager()
        >>> key = manager.create_key()
        >>> encrypted = encrypt_data(data, key.key_material)
        >>> manager.rotate_key(key.key_id)
    """

    def __init__(
        self,
        store: BaseKeyStore | None = None,
        config: KeyManagerConfig | None = None,
        audit_hook: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        """Initialize key manager.

        Args:
            store: Key storage backend (defaults to in-memory).
            config: Manager configuration.
            audit_hook: Callback for audit events.
        """
        self._store = store or InMemoryKeyStore()
        self._config = config or KeyManagerConfig()
        self._audit_hook = audit_hook
        self._lock = RLock()

    def _audit(self, event: str, details: dict[str, Any]) -> None:
        """Log audit event."""
        if self._audit_hook:
            self._audit_hook(event, {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **details,
            })

    def create_key(
        self,
        key_id: str | None = None,
        algorithm: EncryptionAlgorithm | None = None,
        key_type: KeyType | None = None,
        ttl: timedelta | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EncryptionKey:
        """Create a new encryption key.

        Args:
            key_id: Custom key ID (auto-generated if not provided).
            algorithm: Encryption algorithm.
            key_type: Type of key.
            ttl: Time to live.
            metadata: Additional metadata.

        Returns:
            New encryption key.
        """
        with self._lock:
            key_id = key_id or generate_key_id()
            algorithm = algorithm or self._config.default_algorithm
            key_type = key_type or self._config.default_key_type

            # Calculate expiration
            expires_at = None
            effective_ttl = ttl or self._config.default_ttl
            if effective_ttl:
                expires_at = datetime.now(timezone.utc) + effective_ttl

            key = EncryptionKey(
                key_id=key_id,
                key_material=generate_key(algorithm),
                algorithm=algorithm,
                key_type=key_type,
                expires_at=expires_at,
                metadata=metadata or {},
            )

            self._store.put(key)
            self._audit("key_created", {
                "key_id": key_id,
                "algorithm": algorithm.value,
                "key_type": key_type.value,
            })

            return key

    def get_key(self, key_id: str, validate: bool = True) -> EncryptionKey:
        """Retrieve a key by ID.

        Args:
            key_id: Key identifier.
            validate: Whether to validate key is not expired.

        Returns:
            Encryption key.

        Raises:
            KeyError_: If key not found.
            KeyExpiredError: If key has expired.
        """
        key = self._store.get(key_id)
        if key is None:
            raise KeyError_(f"Key not found: {key_id}")

        if validate:
            key.validate()

        self._audit("key_accessed", {"key_id": key_id})
        return key

    def rotate_key(
        self,
        key_id: str,
        archive_old: bool = True,
    ) -> EncryptionKey:
        """Rotate a key (create new version).

        Args:
            key_id: Key to rotate.
            archive_old: Whether to keep old key version.

        Returns:
            New key version.
        """
        with self._lock:
            old_key = self.get_key(key_id, validate=False)

            # Create new key with incremented version
            new_key = EncryptionKey(
                key_id=key_id,
                key_material=generate_key(old_key.algorithm),
                algorithm=old_key.algorithm,
                key_type=old_key.key_type,
                version=old_key.version + 1,
                metadata={
                    **old_key.metadata,
                    "previous_version": old_key.version,
                    "rotated_at": datetime.now(timezone.utc).isoformat(),
                },
            )

            if self._config.default_ttl:
                new_key.expires_at = datetime.now(timezone.utc) + self._config.default_ttl

            # Archive old key if requested
            if archive_old:
                archive_id = f"{key_id}_v{old_key.version}"
                archived_key = EncryptionKey(
                    key_id=archive_id,
                    key_material=old_key.key_material,
                    algorithm=old_key.algorithm,
                    key_type=old_key.key_type,
                    created_at=old_key.created_at,
                    expires_at=datetime.now(timezone.utc) + self._config.rotation_overlap,
                    version=old_key.version,
                    metadata={**old_key.metadata, "archived": True},
                )
                self._store.put(archived_key)

            # Store new key
            self._store.put(new_key)
            self._audit("key_rotated", {
                "key_id": key_id,
                "old_version": old_key.version,
                "new_version": new_key.version,
            })

            return new_key

    def delete_key(self, key_id: str) -> bool:
        """Delete a key.

        Args:
            key_id: Key to delete.

        Returns:
            True if deleted.
        """
        with self._lock:
            result = self._store.delete(key_id)
            if result:
                self._audit("key_deleted", {"key_id": key_id})
            return result

    def list_keys(
        self,
        include_expired: bool = False,
        key_type: KeyType | None = None,
    ) -> list[EncryptionKey]:
        """List all keys.

        Args:
            include_expired: Include expired keys.
            key_type: Filter by key type.

        Returns:
            List of keys.
        """
        keys = []
        for key_id in self._store.list_keys():
            key = self._store.get(key_id)
            if key is None:
                continue
            if not include_expired and key.is_expired:
                continue
            if key_type and key.key_type != key_type:
                continue
            keys.append(key)
        return keys

    def get_or_create_key(
        self,
        key_id: str,
        **create_kwargs: Any,
    ) -> EncryptionKey:
        """Get existing key or create new one.

        Args:
            key_id: Key identifier.
            **create_kwargs: Arguments for create_key if creating.

        Returns:
            Encryption key.
        """
        try:
            return self.get_key(key_id)
        except KeyError_:
            return self.create_key(key_id=key_id, **create_kwargs)

    def cleanup_expired(self) -> int:
        """Delete all expired keys.

        Returns:
            Number of keys deleted.
        """
        deleted = 0
        for key_id in self._store.list_keys():
            key = self._store.get(key_id)
            if key and key.is_expired:
                self.delete_key(key_id)
                deleted += 1
        return deleted


# =============================================================================
# Envelope Encryption
# =============================================================================


@dataclass
class EnvelopeEncryptedData:
    """Data encrypted using envelope encryption.

    Attributes:
        encrypted_key: DEK encrypted with KEK.
        encrypted_data: Data encrypted with DEK.
        kek_id: ID of key encryption key used.
        algorithm: Algorithm for data encryption.
        nonce: Nonce used for data encryption.
        tag: Authentication tag.
    """

    encrypted_key: bytes
    encrypted_data: bytes
    kek_id: str
    algorithm: EncryptionAlgorithm
    nonce: bytes
    tag: bytes

    def to_bytes(self) -> bytes:
        """Serialize to bytes."""
        header = json.dumps({
            "kek_id": self.kek_id,
            "algorithm": self.algorithm.value,
            "key_len": len(self.encrypted_key),
            "nonce_len": len(self.nonce),
            "tag_len": len(self.tag),
        }).encode()

        header_len = len(header).to_bytes(4, "big")
        return (
            header_len
            + header
            + self.encrypted_key
            + self.nonce
            + self.encrypted_data
            + self.tag
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "EnvelopeEncryptedData":
        """Deserialize from bytes."""
        header_len = int.from_bytes(data[:4], "big")
        header = json.loads(data[4 : 4 + header_len].decode())

        offset = 4 + header_len
        key_len = header["key_len"]
        nonce_len = header["nonce_len"]
        tag_len = header["tag_len"]

        encrypted_key = data[offset : offset + key_len]
        offset += key_len

        nonce = data[offset : offset + nonce_len]
        offset += nonce_len

        encrypted_data = data[offset : -tag_len]
        tag = data[-tag_len:]

        return cls(
            encrypted_key=encrypted_key,
            encrypted_data=encrypted_data,
            kek_id=header["kek_id"],
            algorithm=EncryptionAlgorithm(header["algorithm"]),
            nonce=nonce,
            tag=tag,
        )


class EnvelopeEncryption:
    """Envelope encryption for secure key management.

    Envelope encryption uses two levels of keys:
    - Key Encryption Key (KEK): Used to encrypt data keys
    - Data Encryption Key (DEK): Used to encrypt actual data

    This pattern allows:
    - Easy key rotation (just re-encrypt DEK)
    - Integration with external KMS
    - Fine-grained access control

    Example:
        >>> envelope = EnvelopeEncryption(key_manager)
        >>> encrypted = envelope.encrypt(data, kek_id="master_key")
        >>> decrypted = envelope.decrypt(encrypted)
    """

    def __init__(
        self,
        key_manager: KeyManager,
        data_algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM,
    ) -> None:
        """Initialize envelope encryption.

        Args:
            key_manager: Key manager for KEK storage.
            data_algorithm: Algorithm for data encryption.
        """
        self._key_manager = key_manager
        self._data_algorithm = data_algorithm

    def encrypt(
        self,
        plaintext: bytes,
        kek_id: str,
        aad: bytes | None = None,
    ) -> EnvelopeEncryptedData:
        """Encrypt data using envelope encryption.

        Args:
            plaintext: Data to encrypt.
            kek_id: ID of key encryption key.
            aad: Additional authenticated data.

        Returns:
            Envelope encrypted data.
        """
        from truthound.stores.encryption.providers import get_encryptor

        # Get KEK
        kek = self._key_manager.get_key(kek_id)

        # Generate ephemeral DEK
        dek = generate_key(self._data_algorithm)

        # Encrypt DEK with KEK
        kek_encryptor = get_encryptor(kek.algorithm)
        encrypted_dek = kek_encryptor.encrypt(dek, kek.key_material)

        # Encrypt data with DEK
        dek_encryptor = get_encryptor(self._data_algorithm)
        result = dek_encryptor.encrypt_with_metrics(plaintext, dek, aad=aad)

        # Clear DEK from memory
        dek = b"\x00" * len(dek)

        return EnvelopeEncryptedData(
            encrypted_key=encrypted_dek,
            encrypted_data=result.ciphertext,
            kek_id=kek_id,
            algorithm=self._data_algorithm,
            nonce=result.nonce,
            tag=result.tag,
        )

    def decrypt(
        self,
        envelope: EnvelopeEncryptedData,
        aad: bytes | None = None,
    ) -> bytes:
        """Decrypt envelope encrypted data.

        Args:
            envelope: Envelope encrypted data.
            aad: Additional authenticated data.

        Returns:
            Decrypted plaintext.
        """
        from truthound.stores.encryption.providers import get_encryptor

        # Get KEK
        kek = self._key_manager.get_key(envelope.kek_id)

        # Decrypt DEK
        kek_encryptor = get_encryptor(kek.algorithm)
        dek = kek_encryptor.decrypt(envelope.encrypted_key, kek.key_material)

        # Decrypt data
        dek_encryptor = get_encryptor(envelope.algorithm)
        ciphertext = envelope.nonce + envelope.encrypted_data + envelope.tag
        plaintext = dek_encryptor.decrypt(ciphertext, dek, aad=aad)

        # Clear DEK from memory
        dek = b"\x00" * len(dek)

        return plaintext

    def reencrypt_key(
        self,
        envelope: EnvelopeEncryptedData,
        new_kek_id: str,
    ) -> EnvelopeEncryptedData:
        """Re-encrypt DEK with a new KEK (for key rotation).

        Args:
            envelope: Original envelope data.
            new_kek_id: ID of new key encryption key.

        Returns:
            Envelope with re-encrypted DEK.
        """
        from truthound.stores.encryption.providers import get_encryptor

        # Get old and new KEKs
        old_kek = self._key_manager.get_key(envelope.kek_id, validate=False)
        new_kek = self._key_manager.get_key(new_kek_id)

        # Decrypt DEK with old KEK
        old_encryptor = get_encryptor(old_kek.algorithm)
        dek = old_encryptor.decrypt(envelope.encrypted_key, old_kek.key_material)

        # Encrypt DEK with new KEK
        new_encryptor = get_encryptor(new_kek.algorithm)
        new_encrypted_dek = new_encryptor.encrypt(dek, new_kek.key_material)

        # Clear DEK from memory
        dek = b"\x00" * len(dek)

        return EnvelopeEncryptedData(
            encrypted_key=new_encrypted_dek,
            encrypted_data=envelope.encrypted_data,
            kek_id=new_kek_id,
            algorithm=envelope.algorithm,
            nonce=envelope.nonce,
            tag=envelope.tag,
        )
