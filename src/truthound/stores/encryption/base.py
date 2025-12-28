"""Base classes, protocols, and types for encryption system.

This module defines the core abstractions that all encryption implementations
must follow. It uses Protocol-based structural typing for maximum flexibility
and follows the same architectural patterns as the compression module.

Security Considerations:
    - All symmetric encryption uses authenticated encryption (AEAD)
    - Nonces are generated cryptographically and never reused
    - Keys are derived using strong KDFs (Argon2, PBKDF2, scrypt)
    - Memory is cleared after use where possible
    - Timing-safe comparisons for authentication tags

Example:
    >>> from truthound.stores.encryption.base import (
    ...     EncryptionAlgorithm,
    ...     EncryptionConfig,
    ... )
    >>>
    >>> config = EncryptionConfig(
    ...     algorithm=EncryptionAlgorithm.AES_256_GCM,
    ...     key_derivation=KeyDerivation.ARGON2ID,
    ... )
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import (
    Any,
    BinaryIO,
    Callable,
    Iterator,
    Protocol,
    TypeVar,
    runtime_checkable,
)
import hashlib
import hmac
import os


# =============================================================================
# Exceptions
# =============================================================================


class EncryptionError(Exception):
    """Base exception for encryption errors."""

    def __init__(self, message: str, algorithm: str | None = None) -> None:
        self.algorithm = algorithm
        super().__init__(f"[{algorithm}] {message}" if algorithm else message)


class DecryptionError(EncryptionError):
    """Error during decryption (authentication failure, corrupted data)."""

    pass


class KeyError_(EncryptionError):
    """Error related to encryption keys (invalid, expired, not found)."""

    pass


class KeyDerivationError(EncryptionError):
    """Error during key derivation."""

    pass


class UnsupportedAlgorithmError(EncryptionError):
    """Requested encryption algorithm is not available."""

    def __init__(self, algorithm: str, available: list[str] | None = None) -> None:
        self.available = available or []
        msg = f"Algorithm '{algorithm}' is not supported"
        if self.available:
            msg += f". Available: {', '.join(self.available)}"
        super().__init__(msg, algorithm)


class EncryptionConfigError(EncryptionError):
    """Invalid encryption configuration."""

    pass


class NonceReuseError(EncryptionError):
    """Attempted nonce reuse detected (critical security error)."""

    pass


class KeyExpiredError(KeyError_):
    """Encryption key has expired."""

    pass


class IntegrityError(DecryptionError):
    """Data integrity verification failed."""

    pass


# =============================================================================
# Enums
# =============================================================================


class EncryptionAlgorithm(str, Enum):
    """Supported encryption algorithms.

    All algorithms use authenticated encryption (AEAD) to provide both
    confidentiality and integrity protection.
    """

    # AES-GCM variants (recommended for most use cases)
    AES_128_GCM = "aes-128-gcm"
    AES_256_GCM = "aes-256-gcm"

    # ChaCha20-Poly1305 (better for software implementations)
    CHACHA20_POLY1305 = "chacha20-poly1305"

    # Fernet (high-level, batteries-included)
    FERNET = "fernet"

    # XChaCha20-Poly1305 (extended nonce for random generation)
    XCHACHA20_POLY1305 = "xchacha20-poly1305"

    # No encryption (for testing/development)
    NONE = "none"

    @property
    def key_size(self) -> int:
        """Get key size in bytes."""
        key_sizes = {
            self.AES_128_GCM: 16,
            self.AES_256_GCM: 32,
            self.CHACHA20_POLY1305: 32,
            self.XCHACHA20_POLY1305: 32,
            self.FERNET: 32,
            self.NONE: 0,
        }
        return key_sizes.get(self, 32)

    @property
    def nonce_size(self) -> int:
        """Get nonce/IV size in bytes."""
        nonce_sizes = {
            self.AES_128_GCM: 12,
            self.AES_256_GCM: 12,
            self.CHACHA20_POLY1305: 12,
            self.XCHACHA20_POLY1305: 24,
            self.FERNET: 16,
            self.NONE: 0,
        }
        return nonce_sizes.get(self, 12)

    @property
    def tag_size(self) -> int:
        """Get authentication tag size in bytes."""
        tag_sizes = {
            self.AES_128_GCM: 16,
            self.AES_256_GCM: 16,
            self.CHACHA20_POLY1305: 16,
            self.XCHACHA20_POLY1305: 16,
            self.FERNET: 32,  # HMAC-SHA256
            self.NONE: 0,
        }
        return tag_sizes.get(self, 16)

    @property
    def is_aead(self) -> bool:
        """Check if algorithm provides authenticated encryption."""
        return self != self.NONE


class KeyDerivation(str, Enum):
    """Key derivation functions for password-based encryption."""

    # Argon2 variants (recommended)
    ARGON2ID = "argon2id"
    ARGON2I = "argon2i"
    ARGON2D = "argon2d"

    # PBKDF2 (widely compatible)
    PBKDF2_SHA256 = "pbkdf2-sha256"
    PBKDF2_SHA512 = "pbkdf2-sha512"

    # scrypt (memory-hard)
    SCRYPT = "scrypt"

    # HKDF (for key expansion, not password derivation)
    HKDF_SHA256 = "hkdf-sha256"
    HKDF_SHA512 = "hkdf-sha512"

    # Direct key (no derivation)
    NONE = "none"

    @property
    def is_password_based(self) -> bool:
        """Check if this KDF is suitable for password-based key derivation."""
        return self in (
            self.ARGON2ID,
            self.ARGON2I,
            self.ARGON2D,
            self.PBKDF2_SHA256,
            self.PBKDF2_SHA512,
            self.SCRYPT,
        )

    @property
    def default_iterations(self) -> int:
        """Get default iteration count for this KDF."""
        iterations = {
            self.PBKDF2_SHA256: 600_000,
            self.PBKDF2_SHA512: 210_000,
            self.ARGON2ID: 3,
            self.ARGON2I: 4,
            self.ARGON2D: 3,
            self.SCRYPT: 1,
        }
        return iterations.get(self, 1)


class KeyType(str, Enum):
    """Types of encryption keys."""

    # Symmetric keys
    SYMMETRIC = "symmetric"
    DATA_ENCRYPTION_KEY = "dek"
    KEY_ENCRYPTION_KEY = "kek"

    # For envelope encryption
    MASTER_KEY = "master"
    DERIVED_KEY = "derived"

    # Session keys
    SESSION = "session"
    EPHEMERAL = "ephemeral"


class EncryptionMode(Enum):
    """Encryption operation mode."""

    ENCRYPT = auto()
    DECRYPT = auto()
    BOTH = auto()


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class KeyDerivationConfig:
    """Configuration for key derivation.

    Attributes:
        kdf: Key derivation function to use.
        salt_size: Size of salt in bytes.
        iterations: Number of iterations (for PBKDF2).
        memory_cost: Memory cost in KiB (for Argon2).
        parallelism: Degree of parallelism (for Argon2).
        time_cost: Time cost / iterations (for Argon2).
        n: CPU/memory cost parameter (for scrypt).
        r: Block size parameter (for scrypt).
        p: Parallelization parameter (for scrypt).
    """

    kdf: KeyDerivation = KeyDerivation.ARGON2ID
    salt_size: int = 16
    iterations: int | None = None  # Uses KDF default if None
    memory_cost: int = 65536  # 64 MiB for Argon2
    parallelism: int = 4
    time_cost: int = 3
    n: int = 2**14  # scrypt CPU/memory cost
    r: int = 8  # scrypt block size
    p: int = 1  # scrypt parallelism

    def get_iterations(self) -> int:
        """Get effective iterations count."""
        return self.iterations or self.kdf.default_iterations

    def validate(self) -> None:
        """Validate configuration."""
        if self.salt_size < 8:
            raise EncryptionConfigError("salt_size must be at least 8 bytes")
        if self.memory_cost < 8:
            raise EncryptionConfigError("memory_cost must be at least 8 KiB")
        if self.parallelism < 1:
            raise EncryptionConfigError("parallelism must be at least 1")


@dataclass
class EncryptionConfig:
    """Configuration for encryption operations.

    Attributes:
        algorithm: Encryption algorithm to use.
        key_derivation: Key derivation configuration.
        chunk_size: Size of chunks for streaming encryption.
        include_header: Include metadata header in encrypted output.
        verify_on_decrypt: Verify integrity on decryption (always true for AEAD).
        associated_data: Additional authenticated data (AAD).
        key_id: Identifier for the encryption key (for key management).
    """

    algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM
    key_derivation: KeyDerivationConfig = field(default_factory=KeyDerivationConfig)
    chunk_size: int = 64 * 1024  # 64KB
    include_header: bool = True
    verify_on_decrypt: bool = True
    associated_data: bytes | None = None
    key_id: str | None = None

    def validate(self) -> None:
        """Validate configuration."""
        if self.chunk_size <= 0:
            raise EncryptionConfigError("chunk_size must be positive")
        self.key_derivation.validate()


@dataclass
class EncryptionKey:
    """Represents an encryption key with metadata.

    Attributes:
        key_id: Unique identifier for this key.
        key_material: The actual key bytes (should be cleared after use).
        algorithm: Algorithm this key is for.
        key_type: Type of key.
        created_at: When the key was created.
        expires_at: When the key expires (None = never).
        version: Key version for rotation tracking.
        metadata: Additional key metadata.
    """

    key_id: str
    key_material: bytes
    algorithm: EncryptionAlgorithm
    key_type: KeyType = KeyType.DATA_ENCRYPTION_KEY
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime | None = None
    version: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate key after creation."""
        expected_size = self.algorithm.key_size
        if expected_size > 0 and len(self.key_material) != expected_size:
            raise KeyError_(
                f"Key size mismatch: expected {expected_size} bytes, "
                f"got {len(self.key_material)} bytes",
                self.algorithm.value,
            )

    @property
    def is_expired(self) -> bool:
        """Check if key has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    def validate(self) -> None:
        """Validate key is usable."""
        if self.is_expired:
            raise KeyExpiredError(
                f"Key '{self.key_id}' expired at {self.expires_at}",
                self.algorithm.value,
            )

    def clear(self) -> None:
        """Clear key material from memory (best-effort)."""
        if self.key_material:
            # Overwrite with zeros (best-effort, Python may have copies)
            self.key_material = b"\x00" * len(self.key_material)

    def to_dict(self, include_key: bool = False) -> dict[str, Any]:
        """Convert to dictionary (optionally excluding key material)."""
        result = {
            "key_id": self.key_id,
            "algorithm": self.algorithm.value,
            "key_type": self.key_type.value,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "version": self.version,
            "metadata": self.metadata,
        }
        if include_key:
            import base64

            result["key_material"] = base64.b64encode(self.key_material).decode()
        return result


@dataclass
class EncryptionMetrics:
    """Metrics from an encryption operation.

    Attributes:
        plaintext_size: Size of plaintext data in bytes.
        ciphertext_size: Size of ciphertext data in bytes.
        overhead_bytes: Encryption overhead (nonce + tag).
        encryption_time_ms: Time taken to encrypt in milliseconds.
        decryption_time_ms: Time taken to decrypt in milliseconds.
        algorithm: Algorithm used.
        key_derivation_time_ms: Time for key derivation (if applicable).
    """

    plaintext_size: int = 0
    ciphertext_size: int = 0
    overhead_bytes: int = 0
    encryption_time_ms: float = 0.0
    decryption_time_ms: float = 0.0
    algorithm: EncryptionAlgorithm = EncryptionAlgorithm.NONE
    key_derivation_time_ms: float = 0.0

    @property
    def overhead_percent(self) -> float:
        """Calculate overhead percentage."""
        if self.plaintext_size == 0:
            return 0.0
        return (self.overhead_bytes / self.plaintext_size) * 100

    @property
    def throughput_encrypt_mbps(self) -> float:
        """Calculate encryption throughput in MB/s."""
        if self.encryption_time_ms == 0:
            return 0.0
        return (self.plaintext_size / 1024 / 1024) / (self.encryption_time_ms / 1000)

    @property
    def throughput_decrypt_mbps(self) -> float:
        """Calculate decryption throughput in MB/s."""
        if self.decryption_time_ms == 0:
            return 0.0
        return (self.plaintext_size / 1024 / 1024) / (self.decryption_time_ms / 1000)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "plaintext_size": self.plaintext_size,
            "ciphertext_size": self.ciphertext_size,
            "overhead_bytes": self.overhead_bytes,
            "overhead_percent": round(self.overhead_percent, 2),
            "encryption_time_ms": round(self.encryption_time_ms, 2),
            "decryption_time_ms": round(self.decryption_time_ms, 2),
            "throughput_encrypt_mbps": round(self.throughput_encrypt_mbps, 2),
            "throughput_decrypt_mbps": round(self.throughput_decrypt_mbps, 2),
            "algorithm": self.algorithm.value,
            "key_derivation_time_ms": round(self.key_derivation_time_ms, 2),
        }


@dataclass
class EncryptionResult:
    """Result of an encryption operation.

    Attributes:
        ciphertext: Encrypted data bytes.
        nonce: Nonce/IV used for encryption.
        tag: Authentication tag.
        metrics: Encryption metrics.
        header: Optional metadata header.
        key_id: ID of key used (for key management).
    """

    ciphertext: bytes
    nonce: bytes
    tag: bytes
    metrics: EncryptionMetrics
    header: dict[str, Any] = field(default_factory=dict)
    key_id: str | None = None

    def to_bytes(self, include_header: bool = False) -> bytes:
        """Serialize to bytes format.

        Format: [header_len (4 bytes)][header_json][nonce][ciphertext][tag]
        Or without header: [nonce][ciphertext][tag]
        """
        import json

        if include_header and self.header:
            header_bytes = json.dumps(self.header).encode()
            header_len = len(header_bytes).to_bytes(4, "big")
            return header_len + header_bytes + self.nonce + self.ciphertext + self.tag
        return self.nonce + self.ciphertext + self.tag

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        algorithm: EncryptionAlgorithm,
        has_header: bool = False,
    ) -> "EncryptionResult":
        """Deserialize from bytes format."""
        import json

        header: dict[str, Any] = {}
        offset = 0

        if has_header:
            header_len = int.from_bytes(data[:4], "big")
            offset = 4
            header = json.loads(data[offset : offset + header_len].decode())
            offset += header_len

        nonce_size = algorithm.nonce_size
        tag_size = algorithm.tag_size

        nonce = data[offset : offset + nonce_size]
        offset += nonce_size

        ciphertext = data[offset : -tag_size] if tag_size > 0 else data[offset:]
        tag = data[-tag_size:] if tag_size > 0 else b""

        return cls(
            ciphertext=ciphertext,
            nonce=nonce,
            tag=tag,
            metrics=EncryptionMetrics(algorithm=algorithm),
            header=header,
        )


@dataclass
class EncryptionStats:
    """Aggregated encryption statistics across multiple operations.

    Attributes:
        total_operations: Number of encryption operations.
        total_plaintext_bytes: Total bytes encrypted.
        total_ciphertext_bytes: Total bytes of ciphertext.
        total_encryption_time_ms: Total encryption time.
        total_decryption_time_ms: Total decryption time.
        algorithm_usage: Count of each algorithm used.
        errors: Number of errors encountered.
    """

    total_operations: int = 0
    total_plaintext_bytes: int = 0
    total_ciphertext_bytes: int = 0
    total_encryption_time_ms: float = 0.0
    total_decryption_time_ms: float = 0.0
    algorithm_usage: dict[str, int] = field(default_factory=dict)
    errors: int = 0

    def record(self, metrics: EncryptionMetrics) -> None:
        """Record metrics from an encryption operation."""
        self.total_operations += 1
        self.total_plaintext_bytes += metrics.plaintext_size
        self.total_ciphertext_bytes += metrics.ciphertext_size
        self.total_encryption_time_ms += metrics.encryption_time_ms
        self.total_decryption_time_ms += metrics.decryption_time_ms

        algo = metrics.algorithm.value
        self.algorithm_usage[algo] = self.algorithm_usage.get(algo, 0) + 1

    @property
    def average_overhead(self) -> float:
        """Calculate average overhead percentage."""
        if self.total_plaintext_bytes == 0:
            return 0.0
        overhead = self.total_ciphertext_bytes - self.total_plaintext_bytes
        return (overhead / self.total_plaintext_bytes) * 100

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_operations": self.total_operations,
            "total_plaintext_bytes": self.total_plaintext_bytes,
            "total_ciphertext_bytes": self.total_ciphertext_bytes,
            "average_overhead_percent": round(self.average_overhead, 2),
            "total_encryption_time_ms": round(self.total_encryption_time_ms, 2),
            "total_decryption_time_ms": round(self.total_decryption_time_ms, 2),
            "algorithm_usage": self.algorithm_usage,
            "errors": self.errors,
        }


# =============================================================================
# Header Format
# =============================================================================


@dataclass
class EncryptionHeader:
    """Encryption metadata header.

    This header is prepended to encrypted data to enable self-describing
    encryption format. The header itself is authenticated but not encrypted.

    Attributes:
        version: Header format version.
        algorithm: Encryption algorithm used.
        kdf: Key derivation function used.
        salt: Salt used for key derivation (if applicable).
        key_id: ID of the key used.
        created_at: Timestamp of encryption.
        nonce_size: Size of nonce in bytes.
        tag_size: Size of authentication tag in bytes.
        extra: Additional metadata.
    """

    version: int = 1
    algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM
    kdf: KeyDerivation = KeyDerivation.NONE
    salt: bytes = b""
    key_id: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    nonce_size: int = 12
    tag_size: int = 16
    extra: dict[str, Any] = field(default_factory=dict)

    def to_bytes(self) -> bytes:
        """Serialize header to bytes."""
        import json

        data = {
            "v": self.version,
            "alg": self.algorithm.value,
            "kdf": self.kdf.value,
            "salt": self.salt.hex() if self.salt else "",
            "kid": self.key_id,
            "ts": self.created_at.isoformat(),
            "ns": self.nonce_size,
            "ts_": self.tag_size,
            "ext": self.extra,
        }
        header_json = json.dumps(data, separators=(",", ":")).encode()
        return len(header_json).to_bytes(4, "big") + header_json

    @classmethod
    def from_bytes(cls, data: bytes) -> tuple["EncryptionHeader", int]:
        """Deserialize header from bytes.

        Returns:
            Tuple of (header, bytes_consumed).
        """
        import json

        header_len = int.from_bytes(data[:4], "big")
        header_json = data[4 : 4 + header_len]
        d = json.loads(header_json.decode())

        header = cls(
            version=d["v"],
            algorithm=EncryptionAlgorithm(d["alg"]),
            kdf=KeyDerivation(d["kdf"]),
            salt=bytes.fromhex(d["salt"]) if d["salt"] else b"",
            key_id=d.get("kid"),
            created_at=datetime.fromisoformat(d["ts"]),
            nonce_size=d["ns"],
            tag_size=d["ts_"],
            extra=d.get("ext", {}),
        )
        return header, 4 + header_len


# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class Encryptor(Protocol):
    """Protocol for encryption implementations."""

    @property
    def algorithm(self) -> EncryptionAlgorithm:
        """Get the encryption algorithm."""
        ...

    def encrypt(self, plaintext: bytes, key: bytes) -> bytes:
        """Encrypt plaintext data.

        Args:
            plaintext: Data to encrypt.
            key: Encryption key.

        Returns:
            Encrypted data (nonce + ciphertext + tag).
        """
        ...

    def encrypt_with_metrics(
        self,
        plaintext: bytes,
        key: bytes,
        aad: bytes | None = None,
    ) -> EncryptionResult:
        """Encrypt data and return detailed result with metrics.

        Args:
            plaintext: Data to encrypt.
            key: Encryption key.
            aad: Additional authenticated data.

        Returns:
            Encryption result with metrics.
        """
        ...


@runtime_checkable
class Decryptor(Protocol):
    """Protocol for decryption implementations."""

    @property
    def algorithm(self) -> EncryptionAlgorithm:
        """Get the encryption algorithm."""
        ...

    def decrypt(self, ciphertext: bytes, key: bytes) -> bytes:
        """Decrypt ciphertext data.

        Args:
            ciphertext: Data to decrypt (nonce + ciphertext + tag).
            key: Decryption key.

        Returns:
            Decrypted plaintext.

        Raises:
            DecryptionError: If decryption or authentication fails.
        """
        ...


@runtime_checkable
class KeyDeriver(Protocol):
    """Protocol for key derivation implementations."""

    def derive_key(
        self,
        password: str | bytes,
        salt: bytes,
        key_size: int,
    ) -> bytes:
        """Derive a key from password/passphrase.

        Args:
            password: Password or passphrase.
            salt: Cryptographic salt.
            key_size: Desired key size in bytes.

        Returns:
            Derived key.
        """
        ...


@runtime_checkable
class StreamingEncryptor(Protocol):
    """Protocol for streaming encryption."""

    def write(self, data: bytes) -> bytes:
        """Write data to encryption stream.

        Args:
            data: Plaintext chunk to encrypt.

        Returns:
            Encrypted data (may be buffered).
        """
        ...

    def flush(self) -> bytes:
        """Flush buffered data.

        Returns:
            Encrypted data from buffer.
        """
        ...

    def finalize(self) -> bytes:
        """Finalize encryption and get remaining data with tag.

        Returns:
            Final encrypted data including authentication tag.
        """
        ...


@runtime_checkable
class StreamingDecryptor(Protocol):
    """Protocol for streaming decryption."""

    def write(self, data: bytes) -> bytes:
        """Write encrypted data and get decrypted output.

        Args:
            data: Ciphertext chunk.

        Returns:
            Decrypted plaintext.
        """
        ...

    def finalize(self) -> bytes:
        """Finalize decryption and verify authentication.

        Returns:
            Any remaining decrypted data.

        Raises:
            IntegrityError: If authentication fails.
        """
        ...


@runtime_checkable
class KeyManager(Protocol):
    """Protocol for key management."""

    def get_key(self, key_id: str) -> EncryptionKey:
        """Retrieve a key by ID.

        Args:
            key_id: Key identifier.

        Returns:
            Encryption key.

        Raises:
            KeyError_: If key not found.
        """
        ...

    def store_key(self, key: EncryptionKey) -> None:
        """Store a key.

        Args:
            key: Key to store.
        """
        ...

    def rotate_key(self, key_id: str) -> EncryptionKey:
        """Rotate a key (create new version).

        Args:
            key_id: Key to rotate.

        Returns:
            New key version.
        """
        ...

    def delete_key(self, key_id: str) -> None:
        """Delete a key.

        Args:
            key_id: Key to delete.
        """
        ...


# =============================================================================
# Utility Functions
# =============================================================================


def generate_key(algorithm: EncryptionAlgorithm) -> bytes:
    """Generate a cryptographically secure random key.

    Args:
        algorithm: Algorithm to generate key for.

    Returns:
        Random key bytes.
    """
    return os.urandom(algorithm.key_size)


def generate_nonce(algorithm: EncryptionAlgorithm) -> bytes:
    """Generate a cryptographically secure random nonce.

    Args:
        algorithm: Algorithm to generate nonce for.

    Returns:
        Random nonce bytes.
    """
    return os.urandom(algorithm.nonce_size)


def generate_salt(size: int = 16) -> bytes:
    """Generate a cryptographically secure random salt.

    Args:
        size: Salt size in bytes.

    Returns:
        Random salt bytes.
    """
    return os.urandom(size)


def generate_key_id() -> str:
    """Generate a unique key identifier.

    Returns:
        Unique key ID string.
    """
    return os.urandom(16).hex()


def constant_time_compare(a: bytes, b: bytes) -> bool:
    """Compare two byte strings in constant time.

    This prevents timing attacks when comparing authentication tags.

    Args:
        a: First byte string.
        b: Second byte string.

    Returns:
        True if equal, False otherwise.
    """
    return hmac.compare_digest(a, b)


def secure_hash(data: bytes, algorithm: str = "sha256") -> str:
    """Compute secure hash of data.

    Args:
        data: Data to hash.
        algorithm: Hash algorithm name.

    Returns:
        Hex-encoded hash.
    """
    return hashlib.new(algorithm, data).hexdigest()


# =============================================================================
# Type Variables
# =============================================================================

T = TypeVar("T")
EncryptorT = TypeVar("EncryptorT", bound=Encryptor)
DecryptorT = TypeVar("DecryptorT", bound=Decryptor)
KeyDeriverT = TypeVar("KeyDeriverT", bound=KeyDeriver)
