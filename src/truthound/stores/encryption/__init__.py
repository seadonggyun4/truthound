"""Encryption module for secure validation result storage.

This module provides a comprehensive encryption system for protecting
sensitive validation results. It supports multiple encryption algorithms,
key management, and streaming encryption for large datasets.

Features:
    - Multiple AEAD algorithms (AES-256-GCM, ChaCha20-Poly1305, XChaCha20)
    - Password-based key derivation (Argon2, PBKDF2, scrypt)
    - Key management with rotation and expiration
    - Envelope encryption for secure key wrapping
    - Streaming encryption for large files
    - Compress-then-encrypt pipelines

Security Best Practices:
    - Always use authenticated encryption (AEAD)
    - Never reuse nonces with the same key
    - Use strong key derivation for passwords
    - Rotate keys regularly
    - Compress before encrypting

Quick Start:
    >>> from truthound.stores.encryption import (
    ...     get_encryptor,
    ...     generate_key,
    ...     EncryptionAlgorithm,
    ... )
    >>>
    >>> # Generate a key
    >>> key = generate_key(EncryptionAlgorithm.AES_256_GCM)
    >>>
    >>> # Encrypt data
    >>> encryptor = get_encryptor("aes-256-gcm")
    >>> encrypted = encryptor.encrypt(b"sensitive data", key)
    >>> decrypted = encryptor.decrypt(encrypted, key)

Password-Based Encryption:
    >>> from truthound.stores.encryption import derive_key, KeyDerivation
    >>>
    >>> # Derive key from password
    >>> key, salt = derive_key("my_password", kdf=KeyDerivation.ARGON2ID)
    >>>
    >>> # Store salt alongside encrypted data for decryption
    >>> encrypted = encryptor.encrypt(b"data", key)

Key Management:
    >>> from truthound.stores.encryption import KeyManager, FileKeyStore
    >>>
    >>> # Create key manager with file storage
    >>> store = FileKeyStore(".keys", master_password="master_secret")
    >>> manager = KeyManager(store=store)
    >>>
    >>> # Create and rotate keys
    >>> key = manager.create_key(key_id="my_key")
    >>> new_key = manager.rotate_key("my_key")

Streaming Encryption:
    >>> from truthound.stores.encryption import StreamingEncryptor, StreamingDecryptor
    >>>
    >>> # Encrypt large file
    >>> with StreamingEncryptor(key, "output.enc") as enc:
    ...     for chunk in read_large_file():
    ...         enc.write(chunk)
    >>>
    >>> # Decrypt
    >>> with StreamingDecryptor(key, "output.enc") as dec:
    ...     for chunk in dec:
    ...         process(chunk)

Encryption Pipeline (compress-then-encrypt):
    >>> from truthound.stores.encryption import create_secure_pipeline
    >>>
    >>> pipeline = create_secure_pipeline(key, compression="zstd")
    >>> result = pipeline.process(sensitive_data)
    >>> original = pipeline.reverse(result.data, result.header)
"""

# Base types and protocols
from truthound.stores.encryption.base import (
    # Protocols
    Encryptor,
    Decryptor,
    KeyDeriver,
    StreamingEncryptor as StreamingEncryptorProtocol,
    StreamingDecryptor as StreamingDecryptorProtocol,
    KeyManager as KeyManagerProtocol,
    # Enums
    EncryptionAlgorithm,
    KeyDerivation,
    KeyType,
    EncryptionMode,
    # Data classes
    EncryptionConfig,
    EncryptionKey,
    EncryptionMetrics,
    EncryptionResult,
    EncryptionStats,
    EncryptionHeader,
    KeyDerivationConfig,
    # Exceptions
    EncryptionError,
    DecryptionError,
    KeyError_,
    KeyDerivationError,
    KeyExpiredError,
    UnsupportedAlgorithmError,
    EncryptionConfigError,
    NonceReuseError,
    IntegrityError,
    # Utility functions
    generate_key,
    generate_nonce,
    generate_salt,
    generate_key_id,
    constant_time_compare,
    secure_hash,
)

# Encryption providers
from truthound.stores.encryption.providers import (
    # Base class
    BaseEncryptor,
    # Implementations
    AesGcmEncryptor,
    ChaCha20Poly1305Encryptor,
    XChaCha20Poly1305Encryptor,
    FernetEncryptor,
    NoopEncryptor,
    # Factory functions
    get_encryptor,
    register_encryptor,
    list_available_algorithms,
    is_algorithm_available,
)

# Key management
from truthound.stores.encryption.keys import (
    # Key derivation
    BaseKeyDeriver,
    Argon2KeyDeriver,
    PBKDF2KeyDeriver,
    ScryptKeyDeriver,
    HKDFKeyDeriver,
    get_key_deriver,
    derive_key,
    # Key storage
    BaseKeyStore,
    InMemoryKeyStore,
    FileKeyStore,
    EnvironmentKeyStore,
    # Key manager
    KeyManager,
    KeyManagerConfig,
    # Envelope encryption
    EnvelopeEncryption,
    EnvelopeEncryptedData,
)

# Streaming encryption
from truthound.stores.encryption.streaming import (
    # Metrics
    ChunkMetadata,
    StreamingMetrics,
    # Streaming
    StreamingEncryptor,
    StreamingDecryptor,
    StreamingHeader,
    # Chunked (random access)
    ChunkIndex,
    ChunkedEncryptor,
    ChunkedDecryptor,
    # Utilities
    derive_chunk_nonce,
)

# Pipeline
from truthound.stores.encryption.pipeline import (
    # Types
    StageType,
    StageMetrics,
    PipelineMetrics,
    PipelineResult,
    PipelineHeader,
    # Stages
    PipelineStage,
    CompressionStage,
    EncryptionStage,
    ChecksumStage,
    # Pipeline
    EncryptionPipeline,
    # Pre-built pipelines
    create_secure_pipeline,
    create_fast_pipeline,
    create_max_compression_pipeline,
    create_password_pipeline,
    # Streaming pipeline
    StreamingPipeline,
)


__all__ = [
    # === Base Types ===
    # Protocols
    "Encryptor",
    "Decryptor",
    "KeyDeriver",
    "StreamingEncryptorProtocol",
    "StreamingDecryptorProtocol",
    "KeyManagerProtocol",
    # Enums
    "EncryptionAlgorithm",
    "KeyDerivation",
    "KeyType",
    "EncryptionMode",
    # Data classes
    "EncryptionConfig",
    "EncryptionKey",
    "EncryptionMetrics",
    "EncryptionResult",
    "EncryptionStats",
    "EncryptionHeader",
    "KeyDerivationConfig",
    # Exceptions
    "EncryptionError",
    "DecryptionError",
    "KeyError_",
    "KeyDerivationError",
    "KeyExpiredError",
    "UnsupportedAlgorithmError",
    "EncryptionConfigError",
    "NonceReuseError",
    "IntegrityError",
    # Utility functions
    "generate_key",
    "generate_nonce",
    "generate_salt",
    "generate_key_id",
    "constant_time_compare",
    "secure_hash",
    # === Providers ===
    "BaseEncryptor",
    "AesGcmEncryptor",
    "ChaCha20Poly1305Encryptor",
    "XChaCha20Poly1305Encryptor",
    "FernetEncryptor",
    "NoopEncryptor",
    "get_encryptor",
    "register_encryptor",
    "list_available_algorithms",
    "is_algorithm_available",
    # === Key Management ===
    # Key derivation
    "BaseKeyDeriver",
    "Argon2KeyDeriver",
    "PBKDF2KeyDeriver",
    "ScryptKeyDeriver",
    "HKDFKeyDeriver",
    "get_key_deriver",
    "derive_key",
    # Key storage
    "BaseKeyStore",
    "InMemoryKeyStore",
    "FileKeyStore",
    "EnvironmentKeyStore",
    # Key manager
    "KeyManager",
    "KeyManagerConfig",
    # Envelope encryption
    "EnvelopeEncryption",
    "EnvelopeEncryptedData",
    # === Streaming ===
    "ChunkMetadata",
    "StreamingMetrics",
    "StreamingEncryptor",
    "StreamingDecryptor",
    "StreamingHeader",
    "ChunkIndex",
    "ChunkedEncryptor",
    "ChunkedDecryptor",
    "derive_chunk_nonce",
    # === Pipeline ===
    "StageType",
    "StageMetrics",
    "PipelineMetrics",
    "PipelineResult",
    "PipelineHeader",
    "PipelineStage",
    "CompressionStage",
    "EncryptionStage",
    "ChecksumStage",
    "EncryptionPipeline",
    "create_secure_pipeline",
    "create_fast_pipeline",
    "create_max_compression_pipeline",
    "create_password_pipeline",
    "StreamingPipeline",
]
