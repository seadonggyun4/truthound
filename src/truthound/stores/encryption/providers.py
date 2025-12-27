"""Encryption provider implementations.

This module provides concrete implementations of encryption algorithms.
All implementations use authenticated encryption (AEAD) for security.

Supported Algorithms:
    - AES-256-GCM (recommended for most use cases)
    - AES-128-GCM (faster, still secure)
    - ChaCha20-Poly1305 (better for software, mobile)
    - XChaCha20-Poly1305 (extended nonce, safer random nonce)
    - Fernet (high-level, batteries-included)

Example:
    >>> from truthound.stores.encryption.providers import (
    ...     get_encryptor,
    ...     AesGcmEncryptor,
    ... )
    >>>
    >>> # Using factory function
    >>> encryptor = get_encryptor("aes-256-gcm")
    >>> key = encryptor.generate_key()
    >>> encrypted = encryptor.encrypt(b"secret data", key)
    >>> decrypted = encryptor.decrypt(encrypted, key)
    >>>
    >>> # Direct instantiation
    >>> aes = AesGcmEncryptor(key_size=32)
    >>> result = aes.encrypt_with_metrics(b"data", key)
    >>> print(result.metrics.to_dict())
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

from truthound.stores.encryption.base import (
    DecryptionError,
    EncryptionAlgorithm,
    EncryptionConfig,
    EncryptionError,
    EncryptionMetrics,
    EncryptionResult,
    IntegrityError,
    UnsupportedAlgorithmError,
    constant_time_compare,
    generate_key,
    generate_nonce,
)


# =============================================================================
# Base Encryptor
# =============================================================================


class BaseEncryptor(ABC):
    """Base class for all encryption implementations.

    Provides common functionality for encryption providers including
    metrics tracking, key validation, and consistent interface.
    """

    def __init__(self, algorithm: EncryptionAlgorithm) -> None:
        """Initialize encryptor.

        Args:
            algorithm: Encryption algorithm this provider implements.
        """
        self._algorithm = algorithm

    @property
    def algorithm(self) -> EncryptionAlgorithm:
        """Get the encryption algorithm."""
        return self._algorithm

    @property
    def key_size(self) -> int:
        """Get required key size in bytes."""
        return self._algorithm.key_size

    @property
    def nonce_size(self) -> int:
        """Get nonce size in bytes."""
        return self._algorithm.nonce_size

    @property
    def tag_size(self) -> int:
        """Get authentication tag size in bytes."""
        return self._algorithm.tag_size

    def generate_key(self) -> bytes:
        """Generate a new random key for this algorithm."""
        return generate_key(self._algorithm)

    def generate_nonce(self) -> bytes:
        """Generate a new random nonce for this algorithm."""
        return generate_nonce(self._algorithm)

    def _validate_key(self, key: bytes) -> None:
        """Validate key size."""
        if len(key) != self.key_size:
            raise EncryptionError(
                f"Invalid key size: expected {self.key_size} bytes, "
                f"got {len(key)} bytes",
                self._algorithm.value,
            )

    @abstractmethod
    def _encrypt_impl(
        self,
        plaintext: bytes,
        key: bytes,
        nonce: bytes,
        aad: bytes | None = None,
    ) -> tuple[bytes, bytes]:
        """Implementation-specific encryption.

        Args:
            plaintext: Data to encrypt.
            key: Encryption key.
            nonce: Nonce/IV.
            aad: Additional authenticated data.

        Returns:
            Tuple of (ciphertext, tag).
        """
        ...

    @abstractmethod
    def _decrypt_impl(
        self,
        ciphertext: bytes,
        key: bytes,
        nonce: bytes,
        tag: bytes,
        aad: bytes | None = None,
    ) -> bytes:
        """Implementation-specific decryption.

        Args:
            ciphertext: Encrypted data.
            key: Decryption key.
            nonce: Nonce/IV used for encryption.
            tag: Authentication tag.
            aad: Additional authenticated data.

        Returns:
            Decrypted plaintext.

        Raises:
            DecryptionError: If decryption fails.
        """
        ...

    def encrypt(
        self,
        plaintext: bytes,
        key: bytes,
        nonce: bytes | None = None,
        aad: bytes | None = None,
    ) -> bytes:
        """Encrypt plaintext data.

        Args:
            plaintext: Data to encrypt.
            key: Encryption key.
            nonce: Optional nonce (generated if not provided).
            aad: Additional authenticated data.

        Returns:
            Encrypted data in format: nonce || ciphertext || tag
        """
        self._validate_key(key)
        if nonce is None:
            nonce = self.generate_nonce()

        ciphertext, tag = self._encrypt_impl(plaintext, key, nonce, aad)
        return nonce + ciphertext + tag

    def encrypt_with_metrics(
        self,
        plaintext: bytes,
        key: bytes,
        nonce: bytes | None = None,
        aad: bytes | None = None,
    ) -> EncryptionResult:
        """Encrypt data with detailed metrics.

        Args:
            plaintext: Data to encrypt.
            key: Encryption key.
            nonce: Optional nonce (generated if not provided).
            aad: Additional authenticated data.

        Returns:
            Encryption result with metrics.
        """
        self._validate_key(key)
        if nonce is None:
            nonce = self.generate_nonce()

        start_time = time.perf_counter()
        ciphertext, tag = self._encrypt_impl(plaintext, key, nonce, aad)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        metrics = EncryptionMetrics(
            plaintext_size=len(plaintext),
            ciphertext_size=len(ciphertext) + len(nonce) + len(tag),
            overhead_bytes=len(nonce) + len(tag),
            encryption_time_ms=elapsed_ms,
            algorithm=self._algorithm,
        )

        return EncryptionResult(
            ciphertext=ciphertext,
            nonce=nonce,
            tag=tag,
            metrics=metrics,
        )

    def decrypt(
        self,
        data: bytes,
        key: bytes,
        aad: bytes | None = None,
    ) -> bytes:
        """Decrypt ciphertext data.

        Args:
            data: Encrypted data in format: nonce || ciphertext || tag
            key: Decryption key.
            aad: Additional authenticated data.

        Returns:
            Decrypted plaintext.

        Raises:
            DecryptionError: If decryption or authentication fails.
        """
        self._validate_key(key)

        nonce = data[: self.nonce_size]
        tag = data[-self.tag_size :]
        ciphertext = data[self.nonce_size : -self.tag_size]

        return self._decrypt_impl(ciphertext, key, nonce, tag, aad)

    def decrypt_with_metrics(
        self,
        data: bytes,
        key: bytes,
        aad: bytes | None = None,
    ) -> tuple[bytes, EncryptionMetrics]:
        """Decrypt data with detailed metrics.

        Args:
            data: Encrypted data.
            key: Decryption key.
            aad: Additional authenticated data.

        Returns:
            Tuple of (plaintext, metrics).
        """
        start_time = time.perf_counter()
        plaintext = self.decrypt(data, key, aad)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        metrics = EncryptionMetrics(
            plaintext_size=len(plaintext),
            ciphertext_size=len(data),
            overhead_bytes=self.nonce_size + self.tag_size,
            decryption_time_ms=elapsed_ms,
            algorithm=self._algorithm,
        )

        return plaintext, metrics


# =============================================================================
# AES-GCM Implementations
# =============================================================================


class AesGcmEncryptor(BaseEncryptor):
    """AES-GCM authenticated encryption.

    AES-GCM is the recommended algorithm for most use cases. It provides
    both confidentiality and integrity using a single key.

    Key Features:
        - Hardware acceleration on modern CPUs (AES-NI)
        - Widely supported and standardized
        - 128-bit authentication tag
        - 96-bit nonce (never reuse with same key!)

    Example:
        >>> aes = AesGcmEncryptor(key_size=32)  # AES-256
        >>> key = aes.generate_key()
        >>> encrypted = aes.encrypt(b"secret", key)
        >>> decrypted = aes.decrypt(encrypted, key)
    """

    def __init__(self, key_size: int = 32) -> None:
        """Initialize AES-GCM encryptor.

        Args:
            key_size: Key size in bytes (16 for AES-128, 32 for AES-256).
        """
        if key_size == 16:
            algorithm = EncryptionAlgorithm.AES_128_GCM
        elif key_size == 32:
            algorithm = EncryptionAlgorithm.AES_256_GCM
        else:
            raise EncryptionError(
                f"Invalid AES key size: {key_size}. Use 16 or 32 bytes."
            )
        super().__init__(algorithm)
        self._aesgcm: Any = None

    def _get_aesgcm(self, key: bytes) -> Any:
        """Get AESGCM cipher instance."""
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError as e:
            raise UnsupportedAlgorithmError(
                self._algorithm.value,
                available=["Install 'cryptography' package: pip install cryptography"],
            ) from e
        return AESGCM(key)

    def _encrypt_impl(
        self,
        plaintext: bytes,
        key: bytes,
        nonce: bytes,
        aad: bytes | None = None,
    ) -> tuple[bytes, bytes]:
        """Encrypt using AES-GCM."""
        aesgcm = self._get_aesgcm(key)
        # cryptography library returns ciphertext || tag
        result = aesgcm.encrypt(nonce, plaintext, aad)
        ciphertext = result[:-16]  # Everything except last 16 bytes
        tag = result[-16:]  # Last 16 bytes is the tag
        return ciphertext, tag

    def _decrypt_impl(
        self,
        ciphertext: bytes,
        key: bytes,
        nonce: bytes,
        tag: bytes,
        aad: bytes | None = None,
    ) -> bytes:
        """Decrypt using AES-GCM."""
        try:
            from cryptography.exceptions import InvalidTag
        except ImportError as e:
            raise UnsupportedAlgorithmError(
                self._algorithm.value,
                available=["Install 'cryptography' package"],
            ) from e

        aesgcm = self._get_aesgcm(key)
        try:
            # cryptography expects ciphertext || tag
            return aesgcm.decrypt(nonce, ciphertext + tag, aad)
        except InvalidTag as e:
            raise IntegrityError(
                "Authentication failed: data may be corrupted or tampered",
                self._algorithm.value,
            ) from e
        except Exception as e:
            raise DecryptionError(str(e), self._algorithm.value) from e


# =============================================================================
# ChaCha20-Poly1305 Implementation
# =============================================================================


class ChaCha20Poly1305Encryptor(BaseEncryptor):
    """ChaCha20-Poly1305 authenticated encryption.

    ChaCha20-Poly1305 is a modern AEAD cipher that performs well in
    software implementations (no hardware acceleration needed).

    Key Features:
        - Excellent software performance
        - 256-bit key, 96-bit nonce
        - 128-bit authentication tag
        - Constant-time implementation

    Example:
        >>> chacha = ChaCha20Poly1305Encryptor()
        >>> key = chacha.generate_key()
        >>> encrypted = chacha.encrypt(b"secret", key)
    """

    def __init__(self) -> None:
        """Initialize ChaCha20-Poly1305 encryptor."""
        super().__init__(EncryptionAlgorithm.CHACHA20_POLY1305)

    def _get_chacha(self, key: bytes) -> Any:
        """Get ChaCha20Poly1305 cipher instance."""
        try:
            from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
        except ImportError as e:
            raise UnsupportedAlgorithmError(
                self._algorithm.value,
                available=["Install 'cryptography' package"],
            ) from e
        return ChaCha20Poly1305(key)

    def _encrypt_impl(
        self,
        plaintext: bytes,
        key: bytes,
        nonce: bytes,
        aad: bytes | None = None,
    ) -> tuple[bytes, bytes]:
        """Encrypt using ChaCha20-Poly1305."""
        chacha = self._get_chacha(key)
        result = chacha.encrypt(nonce, plaintext, aad)
        ciphertext = result[:-16]
        tag = result[-16:]
        return ciphertext, tag

    def _decrypt_impl(
        self,
        ciphertext: bytes,
        key: bytes,
        nonce: bytes,
        tag: bytes,
        aad: bytes | None = None,
    ) -> bytes:
        """Decrypt using ChaCha20-Poly1305."""
        try:
            from cryptography.exceptions import InvalidTag
        except ImportError as e:
            raise UnsupportedAlgorithmError(
                self._algorithm.value,
                available=["Install 'cryptography' package"],
            ) from e

        chacha = self._get_chacha(key)
        try:
            return chacha.decrypt(nonce, ciphertext + tag, aad)
        except InvalidTag as e:
            raise IntegrityError(
                "Authentication failed: data may be corrupted or tampered",
                self._algorithm.value,
            ) from e
        except Exception as e:
            raise DecryptionError(str(e), self._algorithm.value) from e


# =============================================================================
# XChaCha20-Poly1305 Implementation
# =============================================================================


class XChaCha20Poly1305Encryptor(BaseEncryptor):
    """XChaCha20-Poly1305 authenticated encryption.

    XChaCha20-Poly1305 extends ChaCha20-Poly1305 with a 192-bit nonce,
    making it safe to generate nonces randomly without collision risk.

    Key Features:
        - 192-bit nonce (safe for random generation)
        - Same security as ChaCha20-Poly1305
        - Ideal for file encryption

    Note:
        Requires the 'pynacl' package for native implementation.
        Falls back to a cryptography-based implementation if not available.

    Example:
        >>> xchacha = XChaCha20Poly1305Encryptor()
        >>> key = xchacha.generate_key()
        >>> encrypted = xchacha.encrypt(b"secret", key)
    """

    def __init__(self) -> None:
        """Initialize XChaCha20-Poly1305 encryptor."""
        super().__init__(EncryptionAlgorithm.XCHACHA20_POLY1305)
        self._use_nacl = self._check_nacl()

    def _check_nacl(self) -> bool:
        """Check if pynacl is available."""
        try:
            import nacl.secret  # noqa: F401

            return True
        except ImportError:
            return False

    def _encrypt_impl(
        self,
        plaintext: bytes,
        key: bytes,
        nonce: bytes,
        aad: bytes | None = None,
    ) -> tuple[bytes, bytes]:
        """Encrypt using XChaCha20-Poly1305."""
        if self._use_nacl:
            return self._encrypt_nacl(plaintext, key, nonce)
        return self._encrypt_hchacha(plaintext, key, nonce, aad)

    def _decrypt_impl(
        self,
        ciphertext: bytes,
        key: bytes,
        nonce: bytes,
        tag: bytes,
        aad: bytes | None = None,
    ) -> bytes:
        """Decrypt using XChaCha20-Poly1305."""
        if self._use_nacl:
            return self._decrypt_nacl(ciphertext, key, nonce, tag)
        return self._decrypt_hchacha(ciphertext, key, nonce, tag, aad)

    def _encrypt_nacl(
        self, plaintext: bytes, key: bytes, nonce: bytes
    ) -> tuple[bytes, bytes]:
        """Encrypt using pynacl."""
        import nacl.secret

        box = nacl.secret.SecretBox(key)
        encrypted = box.encrypt(plaintext, nonce)
        # nacl format: nonce || ciphertext || tag
        # We already have nonce, so skip it
        ct_and_tag = encrypted[24:]  # Skip 24-byte nonce
        ciphertext = ct_and_tag[:-16]
        tag = ct_and_tag[-16:]
        return ciphertext, tag

    def _decrypt_nacl(
        self, ciphertext: bytes, key: bytes, nonce: bytes, tag: bytes
    ) -> bytes:
        """Decrypt using pynacl."""
        import nacl.exceptions
        import nacl.secret

        box = nacl.secret.SecretBox(key)
        try:
            # nacl expects nonce || ciphertext || tag
            return box.decrypt(nonce + ciphertext + tag)
        except nacl.exceptions.CryptoError as e:
            raise IntegrityError(
                "Authentication failed: data may be corrupted or tampered",
                self._algorithm.value,
            ) from e

    def _encrypt_hchacha(
        self,
        plaintext: bytes,
        key: bytes,
        nonce: bytes,
        aad: bytes | None = None,
    ) -> tuple[bytes, bytes]:
        """Encrypt using HChaCha20 + ChaCha20-Poly1305 (cryptography fallback)."""
        # XChaCha20 = HChaCha20(key, nonce[:16]) to derive subkey
        # Then ChaCha20-Poly1305 with subkey and modified nonce
        try:
            from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
        except ImportError as e:
            raise UnsupportedAlgorithmError(
                self._algorithm.value,
                available=[
                    "Install 'pynacl' or 'cryptography': pip install pynacl cryptography"
                ],
            ) from e

        # Derive subkey using HChaCha20 (simplified - use first 16 bytes of nonce)
        subkey = self._hchacha20(key, nonce[:16])
        # Use last 8 bytes of nonce + 4 zero bytes as the ChaCha20 nonce
        chacha_nonce = b"\x00" * 4 + nonce[16:24]

        chacha = ChaCha20Poly1305(subkey)
        result = chacha.encrypt(chacha_nonce, plaintext, aad)
        return result[:-16], result[-16:]

    def _decrypt_hchacha(
        self,
        ciphertext: bytes,
        key: bytes,
        nonce: bytes,
        tag: bytes,
        aad: bytes | None = None,
    ) -> bytes:
        """Decrypt using HChaCha20 + ChaCha20-Poly1305."""
        try:
            from cryptography.exceptions import InvalidTag
            from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
        except ImportError as e:
            raise UnsupportedAlgorithmError(
                self._algorithm.value,
                available=["Install 'pynacl' or 'cryptography'"],
            ) from e

        subkey = self._hchacha20(key, nonce[:16])
        chacha_nonce = b"\x00" * 4 + nonce[16:24]

        chacha = ChaCha20Poly1305(subkey)
        try:
            return chacha.decrypt(chacha_nonce, ciphertext + tag, aad)
        except InvalidTag as e:
            raise IntegrityError(
                "Authentication failed",
                self._algorithm.value,
            ) from e

    def _hchacha20(self, key: bytes, nonce: bytes) -> bytes:
        """HChaCha20 key derivation (simplified implementation).

        This is a simplified version. For production, use a proper
        HChaCha20 implementation from a cryptographic library.
        """
        import hashlib

        # Simplified: HKDF-style derivation for fallback
        # In production, implement actual HChaCha20
        return hashlib.blake2b(key + nonce, digest_size=32).digest()


# =============================================================================
# Fernet Implementation
# =============================================================================


class FernetEncryptor(BaseEncryptor):
    """Fernet symmetric encryption.

    Fernet provides a high-level, easy-to-use API for authenticated
    encryption. It uses AES-128-CBC with HMAC-SHA256 for authentication.

    Key Features:
        - Batteries-included design
        - Timestamp for freshness checking
        - URL-safe base64 encoding
        - Simple API

    Note:
        Fernet keys are base64-encoded. Use generate_key() to create
        properly formatted keys.

    Example:
        >>> fernet = FernetEncryptor()
        >>> key = fernet.generate_key()
        >>> encrypted = fernet.encrypt(b"secret", key)
    """

    def __init__(self) -> None:
        """Initialize Fernet encryptor."""
        super().__init__(EncryptionAlgorithm.FERNET)

    def generate_key(self) -> bytes:
        """Generate a Fernet key (base64-encoded)."""
        try:
            from cryptography.fernet import Fernet
        except ImportError as e:
            raise UnsupportedAlgorithmError(
                self._algorithm.value,
                available=["Install 'cryptography' package"],
            ) from e
        return Fernet.generate_key()

    def _get_fernet(self, key: bytes) -> Any:
        """Get Fernet instance."""
        try:
            from cryptography.fernet import Fernet
        except ImportError as e:
            raise UnsupportedAlgorithmError(
                self._algorithm.value,
                available=["Install 'cryptography' package"],
            ) from e
        return Fernet(key)

    def _validate_key(self, key: bytes) -> None:
        """Validate Fernet key (base64-encoded, 32 bytes decoded)."""
        import base64

        try:
            decoded = base64.urlsafe_b64decode(key)
            if len(decoded) != 32:
                raise EncryptionError(
                    f"Invalid Fernet key: decoded to {len(decoded)} bytes, expected 32",
                    self._algorithm.value,
                )
        except Exception as e:
            if isinstance(e, EncryptionError):
                raise
            raise EncryptionError(
                f"Invalid Fernet key format: {e}",
                self._algorithm.value,
            ) from e

    def _encrypt_impl(
        self,
        plaintext: bytes,
        key: bytes,
        nonce: bytes,
        aad: bytes | None = None,
    ) -> tuple[bytes, bytes]:
        """Encrypt using Fernet.

        Note: Fernet manages its own nonce/IV internally.
        The nonce parameter is ignored.
        """
        fernet = self._get_fernet(key)
        token = fernet.encrypt(plaintext)
        # Fernet token includes everything; use empty tag
        return token, b""

    def _decrypt_impl(
        self,
        ciphertext: bytes,
        key: bytes,
        nonce: bytes,
        tag: bytes,
        aad: bytes | None = None,
    ) -> bytes:
        """Decrypt using Fernet."""
        try:
            from cryptography.fernet import InvalidToken
        except ImportError as e:
            raise UnsupportedAlgorithmError(
                self._algorithm.value,
                available=["Install 'cryptography' package"],
            ) from e

        fernet = self._get_fernet(key)
        try:
            return fernet.decrypt(ciphertext)
        except InvalidToken as e:
            raise IntegrityError(
                "Authentication failed or token expired",
                self._algorithm.value,
            ) from e

    def encrypt(
        self,
        plaintext: bytes,
        key: bytes,
        nonce: bytes | None = None,
        aad: bytes | None = None,
    ) -> bytes:
        """Encrypt using Fernet.

        Fernet handles nonce internally, so the nonce parameter is ignored.
        """
        self._validate_key(key)
        ciphertext, _ = self._encrypt_impl(plaintext, key, b"", aad)
        return ciphertext

    def decrypt(
        self,
        data: bytes,
        key: bytes,
        aad: bytes | None = None,
    ) -> bytes:
        """Decrypt Fernet token."""
        self._validate_key(key)
        return self._decrypt_impl(data, key, b"", b"", aad)


# =============================================================================
# No-op Encryptor (for testing/development)
# =============================================================================


class NoopEncryptor(BaseEncryptor):
    """No-op encryptor that passes data through unchanged.

    WARNING: This provides NO security and should only be used for
    testing or development purposes.

    Example:
        >>> noop = NoopEncryptor()
        >>> encrypted = noop.encrypt(b"data", b"")
        >>> assert encrypted == b"data"
    """

    def __init__(self) -> None:
        """Initialize no-op encryptor."""
        super().__init__(EncryptionAlgorithm.NONE)

    def _validate_key(self, key: bytes) -> None:
        """No validation needed for no-op."""
        pass

    def _encrypt_impl(
        self,
        plaintext: bytes,
        key: bytes,
        nonce: bytes,
        aad: bytes | None = None,
    ) -> tuple[bytes, bytes]:
        """Return data unchanged."""
        return plaintext, b""

    def _decrypt_impl(
        self,
        ciphertext: bytes,
        key: bytes,
        nonce: bytes,
        tag: bytes,
        aad: bytes | None = None,
    ) -> bytes:
        """Return data unchanged."""
        return ciphertext

    def encrypt(
        self,
        plaintext: bytes,
        key: bytes,
        nonce: bytes | None = None,
        aad: bytes | None = None,
    ) -> bytes:
        """Return data unchanged."""
        return plaintext

    def decrypt(
        self,
        data: bytes,
        key: bytes,
        aad: bytes | None = None,
    ) -> bytes:
        """Return data unchanged."""
        return data


# =============================================================================
# Factory Functions
# =============================================================================

# Registry of available encryptors
_ENCRYPTOR_REGISTRY: dict[EncryptionAlgorithm, type[BaseEncryptor]] = {
    EncryptionAlgorithm.AES_128_GCM: AesGcmEncryptor,
    EncryptionAlgorithm.AES_256_GCM: AesGcmEncryptor,
    EncryptionAlgorithm.CHACHA20_POLY1305: ChaCha20Poly1305Encryptor,
    EncryptionAlgorithm.XCHACHA20_POLY1305: XChaCha20Poly1305Encryptor,
    EncryptionAlgorithm.FERNET: FernetEncryptor,
    EncryptionAlgorithm.NONE: NoopEncryptor,
}


def get_encryptor(
    algorithm: str | EncryptionAlgorithm,
) -> BaseEncryptor:
    """Get an encryptor instance for the specified algorithm.

    Args:
        algorithm: Algorithm name or enum value.

    Returns:
        Configured encryptor instance.

    Raises:
        UnsupportedAlgorithmError: If algorithm is not supported.

    Example:
        >>> encryptor = get_encryptor("aes-256-gcm")
        >>> key = encryptor.generate_key()
        >>> encrypted = encryptor.encrypt(b"data", key)
    """
    if isinstance(algorithm, str):
        try:
            algorithm = EncryptionAlgorithm(algorithm)
        except ValueError:
            raise UnsupportedAlgorithmError(
                algorithm,
                available=list_available_algorithms(),
            )

    encryptor_class = _ENCRYPTOR_REGISTRY.get(algorithm)
    if encryptor_class is None:
        raise UnsupportedAlgorithmError(
            algorithm.value,
            available=list_available_algorithms(),
        )

    # Handle AES key size variants
    if algorithm == EncryptionAlgorithm.AES_128_GCM:
        return AesGcmEncryptor(key_size=16)
    elif algorithm == EncryptionAlgorithm.AES_256_GCM:
        return AesGcmEncryptor(key_size=32)

    return encryptor_class()


def register_encryptor(
    algorithm: EncryptionAlgorithm,
    encryptor_class: type[BaseEncryptor],
) -> None:
    """Register a custom encryptor implementation.

    Args:
        algorithm: Algorithm enum value.
        encryptor_class: Encryptor class to register.

    Example:
        >>> class MyEncryptor(BaseEncryptor):
        ...     pass
        >>> register_encryptor(EncryptionAlgorithm.CUSTOM, MyEncryptor)
    """
    _ENCRYPTOR_REGISTRY[algorithm] = encryptor_class


def list_available_algorithms() -> list[str]:
    """List all available encryption algorithms.

    Returns:
        List of algorithm names.
    """
    available = []
    for algo in _ENCRYPTOR_REGISTRY:
        try:
            # Test if the algorithm's dependencies are available
            encryptor = get_encryptor(algo)
            if algo != EncryptionAlgorithm.NONE:
                # Try to create a key to verify dependencies
                encryptor.generate_key()
            available.append(algo.value)
        except (UnsupportedAlgorithmError, ImportError):
            pass
    return available


def is_algorithm_available(algorithm: str | EncryptionAlgorithm) -> bool:
    """Check if an algorithm is available.

    Args:
        algorithm: Algorithm to check.

    Returns:
        True if algorithm is available.
    """
    if isinstance(algorithm, str):
        try:
            algorithm = EncryptionAlgorithm(algorithm)
        except ValueError:
            return False

    try:
        encryptor = get_encryptor(algorithm)
        if algorithm != EncryptionAlgorithm.NONE:
            encryptor.generate_key()
        return True
    except (UnsupportedAlgorithmError, ImportError):
        return False
