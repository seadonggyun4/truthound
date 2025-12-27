"""Tests for encryption base module."""

from datetime import datetime, timedelta, timezone

import pytest

from truthound.stores.encryption.base import (
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
    KeyExpiredError,
    UnsupportedAlgorithmError,
    EncryptionConfigError,
    IntegrityError,
    # Utilities
    generate_key,
    generate_nonce,
    generate_salt,
    generate_key_id,
    constant_time_compare,
    secure_hash,
)


class TestEncryptionAlgorithm:
    """Tests for EncryptionAlgorithm enum."""

    def test_key_sizes(self):
        """Test key size properties."""
        assert EncryptionAlgorithm.AES_128_GCM.key_size == 16
        assert EncryptionAlgorithm.AES_256_GCM.key_size == 32
        assert EncryptionAlgorithm.CHACHA20_POLY1305.key_size == 32
        assert EncryptionAlgorithm.XCHACHA20_POLY1305.key_size == 32
        assert EncryptionAlgorithm.NONE.key_size == 0

    def test_nonce_sizes(self):
        """Test nonce size properties."""
        assert EncryptionAlgorithm.AES_256_GCM.nonce_size == 12
        assert EncryptionAlgorithm.CHACHA20_POLY1305.nonce_size == 12
        assert EncryptionAlgorithm.XCHACHA20_POLY1305.nonce_size == 24
        assert EncryptionAlgorithm.NONE.nonce_size == 0

    def test_tag_sizes(self):
        """Test tag size properties."""
        assert EncryptionAlgorithm.AES_256_GCM.tag_size == 16
        assert EncryptionAlgorithm.CHACHA20_POLY1305.tag_size == 16
        assert EncryptionAlgorithm.NONE.tag_size == 0

    def test_is_aead(self):
        """Test AEAD detection."""
        assert EncryptionAlgorithm.AES_256_GCM.is_aead
        assert EncryptionAlgorithm.CHACHA20_POLY1305.is_aead
        assert not EncryptionAlgorithm.NONE.is_aead


class TestKeyDerivation:
    """Tests for KeyDerivation enum."""

    def test_is_password_based(self):
        """Test password-based KDF detection."""
        assert KeyDerivation.ARGON2ID.is_password_based
        assert KeyDerivation.PBKDF2_SHA256.is_password_based
        assert KeyDerivation.SCRYPT.is_password_based
        assert not KeyDerivation.HKDF_SHA256.is_password_based
        assert not KeyDerivation.NONE.is_password_based

    def test_default_iterations(self):
        """Test default iteration counts."""
        assert KeyDerivation.PBKDF2_SHA256.default_iterations == 600_000
        assert KeyDerivation.PBKDF2_SHA512.default_iterations == 210_000
        assert KeyDerivation.ARGON2ID.default_iterations == 3


class TestKeyDerivationConfig:
    """Tests for KeyDerivationConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = KeyDerivationConfig()
        assert config.kdf == KeyDerivation.ARGON2ID
        assert config.salt_size == 16
        assert config.memory_cost == 65536

    def test_get_iterations(self):
        """Test iterations retrieval."""
        config = KeyDerivationConfig(kdf=KeyDerivation.PBKDF2_SHA256)
        assert config.get_iterations() == 600_000

        config_custom = KeyDerivationConfig(
            kdf=KeyDerivation.PBKDF2_SHA256,
            iterations=100_000,
        )
        assert config_custom.get_iterations() == 100_000

    def test_validation(self):
        """Test configuration validation."""
        config = KeyDerivationConfig(salt_size=4)
        with pytest.raises(EncryptionConfigError):
            config.validate()

        config2 = KeyDerivationConfig(parallelism=0)
        with pytest.raises(EncryptionConfigError):
            config2.validate()


class TestEncryptionConfig:
    """Tests for EncryptionConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = EncryptionConfig()
        assert config.algorithm == EncryptionAlgorithm.AES_256_GCM
        assert config.chunk_size == 64 * 1024

    def test_validation(self):
        """Test configuration validation."""
        config = EncryptionConfig(chunk_size=0)
        with pytest.raises(EncryptionConfigError):
            config.validate()


class TestEncryptionKey:
    """Tests for EncryptionKey."""

    def test_create_key(self):
        """Test key creation."""
        key = EncryptionKey(
            key_id="test_key",
            key_material=b"x" * 32,
            algorithm=EncryptionAlgorithm.AES_256_GCM,
        )
        assert key.key_id == "test_key"
        assert key.version == 1
        assert not key.is_expired

    def test_invalid_key_size(self):
        """Test key size validation."""
        with pytest.raises(KeyError_):
            EncryptionKey(
                key_id="test",
                key_material=b"short",
                algorithm=EncryptionAlgorithm.AES_256_GCM,
            )

    def test_key_expiration(self):
        """Test key expiration."""
        expired_key = EncryptionKey(
            key_id="expired",
            key_material=b"x" * 32,
            algorithm=EncryptionAlgorithm.AES_256_GCM,
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        assert expired_key.is_expired

        with pytest.raises(KeyExpiredError):
            expired_key.validate()

    def test_key_to_dict(self):
        """Test key serialization."""
        key = EncryptionKey(
            key_id="test",
            key_material=b"x" * 32,
            algorithm=EncryptionAlgorithm.AES_256_GCM,
        )
        d = key.to_dict(include_key=False)
        assert "key_id" in d
        assert "key_material" not in d

        d_with_key = key.to_dict(include_key=True)
        assert "key_material" in d_with_key

    def test_key_clear(self):
        """Test key material clearing."""
        key = EncryptionKey(
            key_id="test",
            key_material=b"secret" * 5 + b"xx",
            algorithm=EncryptionAlgorithm.AES_256_GCM,
        )
        key.clear()
        assert key.key_material == b"\x00" * 32


class TestEncryptionMetrics:
    """Tests for EncryptionMetrics."""

    def test_overhead_calculation(self):
        """Test overhead calculation."""
        metrics = EncryptionMetrics(
            plaintext_size=1000,
            ciphertext_size=1028,
            overhead_bytes=28,
        )
        assert abs(metrics.overhead_percent - 2.8) < 0.01

    def test_throughput_calculation(self):
        """Test throughput calculation."""
        metrics = EncryptionMetrics(
            plaintext_size=1024 * 1024,  # 1 MB
            encryption_time_ms=100,  # 100 ms
        )
        assert metrics.throughput_encrypt_mbps == 10.0

    def test_to_dict(self):
        """Test metrics serialization."""
        metrics = EncryptionMetrics(
            plaintext_size=1000,
            ciphertext_size=1028,
            algorithm=EncryptionAlgorithm.AES_256_GCM,
        )
        d = metrics.to_dict()
        assert "plaintext_size" in d
        assert d["algorithm"] == "aes-256-gcm"


class TestEncryptionResult:
    """Tests for EncryptionResult."""

    def test_to_bytes_and_back(self):
        """Test serialization round-trip."""
        result = EncryptionResult(
            ciphertext=b"encrypted_data",
            nonce=b"x" * 12,
            tag=b"y" * 16,
            metrics=EncryptionMetrics(),
        )

        serialized = result.to_bytes()
        restored = EncryptionResult.from_bytes(
            serialized,
            EncryptionAlgorithm.AES_256_GCM,
        )

        assert restored.nonce == result.nonce
        assert restored.ciphertext == result.ciphertext
        assert restored.tag == result.tag


class TestEncryptionStats:
    """Tests for EncryptionStats."""

    def test_record_metrics(self):
        """Test metrics recording."""
        stats = EncryptionStats()

        metrics1 = EncryptionMetrics(
            plaintext_size=1000,
            ciphertext_size=1028,
            algorithm=EncryptionAlgorithm.AES_256_GCM,
        )
        metrics2 = EncryptionMetrics(
            plaintext_size=2000,
            ciphertext_size=2028,
            algorithm=EncryptionAlgorithm.AES_256_GCM,
        )

        stats.record(metrics1)
        stats.record(metrics2)

        assert stats.total_operations == 2
        assert stats.total_plaintext_bytes == 3000
        assert stats.algorithm_usage["aes-256-gcm"] == 2


class TestEncryptionHeader:
    """Tests for EncryptionHeader."""

    def test_header_round_trip(self):
        """Test header serialization round-trip."""
        header = EncryptionHeader(
            algorithm=EncryptionAlgorithm.AES_256_GCM,
            kdf=KeyDerivation.ARGON2ID,
            salt=b"salt" * 4,
            key_id="my_key",
        )

        serialized = header.to_bytes()
        restored, consumed = EncryptionHeader.from_bytes(serialized)

        assert restored.algorithm == header.algorithm
        assert restored.kdf == header.kdf
        assert restored.salt == header.salt
        assert restored.key_id == header.key_id


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_generate_key(self):
        """Test key generation."""
        key_128 = generate_key(EncryptionAlgorithm.AES_128_GCM)
        assert len(key_128) == 16

        key_256 = generate_key(EncryptionAlgorithm.AES_256_GCM)
        assert len(key_256) == 32

    def test_generate_nonce(self):
        """Test nonce generation."""
        nonce = generate_nonce(EncryptionAlgorithm.AES_256_GCM)
        assert len(nonce) == 12

        nonce_x = generate_nonce(EncryptionAlgorithm.XCHACHA20_POLY1305)
        assert len(nonce_x) == 24

    def test_generate_salt(self):
        """Test salt generation."""
        salt = generate_salt(16)
        assert len(salt) == 16

        # Ensure randomness
        salt2 = generate_salt(16)
        assert salt != salt2

    def test_generate_key_id(self):
        """Test key ID generation."""
        key_id = generate_key_id()
        assert len(key_id) == 32  # 16 bytes hex encoded
        assert key_id.isalnum()

    def test_constant_time_compare(self):
        """Test constant-time comparison."""
        assert constant_time_compare(b"hello", b"hello")
        assert not constant_time_compare(b"hello", b"world")
        assert not constant_time_compare(b"hello", b"hell")

    def test_secure_hash(self):
        """Test secure hashing."""
        hash1 = secure_hash(b"data")
        hash2 = secure_hash(b"data")
        hash3 = secure_hash(b"different")

        assert hash1 == hash2
        assert hash1 != hash3
        assert len(hash1) == 64  # SHA256 hex


class TestExceptions:
    """Tests for exception hierarchy."""

    def test_encryption_error(self):
        """Test EncryptionError."""
        err = EncryptionError("test error", "aes-256-gcm")
        assert "aes-256-gcm" in str(err)
        assert err.algorithm == "aes-256-gcm"

    def test_decryption_error(self):
        """Test DecryptionError inheritance."""
        err = DecryptionError("decryption failed")
        assert isinstance(err, EncryptionError)

    def test_unsupported_algorithm_error(self):
        """Test UnsupportedAlgorithmError."""
        err = UnsupportedAlgorithmError("unknown", available=["aes-256-gcm"])
        assert "unknown" in str(err)
        assert "aes-256-gcm" in str(err)

    def test_integrity_error(self):
        """Test IntegrityError."""
        err = IntegrityError("tag mismatch", "aes-256-gcm")
        assert isinstance(err, DecryptionError)
