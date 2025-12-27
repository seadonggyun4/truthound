"""Tests for encryption providers."""

import pytest

from truthound.stores.encryption.base import (
    DecryptionError,
    EncryptionAlgorithm,
    IntegrityError,
    UnsupportedAlgorithmError,
)
from truthound.stores.encryption.providers import (
    AesGcmEncryptor,
    BaseEncryptor,
    ChaCha20Poly1305Encryptor,
    FernetEncryptor,
    NoopEncryptor,
    XChaCha20Poly1305Encryptor,
    get_encryptor,
    is_algorithm_available,
    list_available_algorithms,
    register_encryptor,
)


# Check if cryptography is available
try:
    import cryptography

    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False

# Check if pynacl is available
try:
    import nacl

    HAS_NACL = True
except ImportError:
    HAS_NACL = False


@pytest.fixture
def sample_data():
    """Sample data for encryption tests."""
    return b"This is some test data for encryption!" * 10


@pytest.fixture
def aad():
    """Additional authenticated data."""
    return b"additional_authenticated_data"


class TestNoopEncryptor:
    """Tests for NoopEncryptor (always available)."""

    def test_encrypt_decrypt(self, sample_data):
        """Test no-op encryption/decryption."""
        encryptor = NoopEncryptor()
        encrypted = encryptor.encrypt(sample_data, b"")
        assert encrypted == sample_data

        decrypted = encryptor.decrypt(encrypted, b"")
        assert decrypted == sample_data

    def test_algorithm(self):
        """Test algorithm property."""
        encryptor = NoopEncryptor()
        assert encryptor.algorithm == EncryptionAlgorithm.NONE


@pytest.mark.skipif(not HAS_CRYPTOGRAPHY, reason="cryptography not installed")
class TestAesGcmEncryptor:
    """Tests for AesGcmEncryptor."""

    def test_aes_128_gcm(self, sample_data):
        """Test AES-128-GCM encryption."""
        encryptor = AesGcmEncryptor(key_size=16)
        assert encryptor.algorithm == EncryptionAlgorithm.AES_128_GCM

        key = encryptor.generate_key()
        assert len(key) == 16

        encrypted = encryptor.encrypt(sample_data, key)
        assert encrypted != sample_data

        decrypted = encryptor.decrypt(encrypted, key)
        assert decrypted == sample_data

    def test_aes_256_gcm(self, sample_data):
        """Test AES-256-GCM encryption."""
        encryptor = AesGcmEncryptor(key_size=32)
        assert encryptor.algorithm == EncryptionAlgorithm.AES_256_GCM

        key = encryptor.generate_key()
        assert len(key) == 32

        encrypted = encryptor.encrypt(sample_data, key)
        decrypted = encryptor.decrypt(encrypted, key)
        assert decrypted == sample_data

    def test_with_aad(self, sample_data, aad):
        """Test encryption with additional authenticated data."""
        encryptor = AesGcmEncryptor()
        key = encryptor.generate_key()

        encrypted = encryptor.encrypt(sample_data, key, aad=aad)

        # Should succeed with correct AAD
        decrypted = encryptor.decrypt(encrypted, key, aad=aad)
        assert decrypted == sample_data

        # Should fail with wrong AAD
        with pytest.raises(IntegrityError):
            encryptor.decrypt(encrypted, key, aad=b"wrong_aad")

    def test_tampered_data(self, sample_data):
        """Test detection of tampered ciphertext."""
        encryptor = AesGcmEncryptor()
        key = encryptor.generate_key()

        encrypted = encryptor.encrypt(sample_data, key)

        # Tamper with ciphertext
        tampered = bytearray(encrypted)
        tampered[20] ^= 0xFF
        tampered = bytes(tampered)

        with pytest.raises(IntegrityError):
            encryptor.decrypt(tampered, key)

    def test_wrong_key(self, sample_data):
        """Test decryption with wrong key."""
        encryptor = AesGcmEncryptor()
        key1 = encryptor.generate_key()
        key2 = encryptor.generate_key()

        encrypted = encryptor.encrypt(sample_data, key1)

        with pytest.raises(IntegrityError):
            encryptor.decrypt(encrypted, key2)

    def test_encrypt_with_metrics(self, sample_data):
        """Test encryption with metrics."""
        encryptor = AesGcmEncryptor()
        key = encryptor.generate_key()

        result = encryptor.encrypt_with_metrics(sample_data, key)

        assert result.ciphertext
        assert result.nonce
        assert result.tag
        assert result.metrics.plaintext_size == len(sample_data)
        assert result.metrics.algorithm == EncryptionAlgorithm.AES_256_GCM

    def test_decrypt_with_metrics(self, sample_data):
        """Test decryption with metrics."""
        encryptor = AesGcmEncryptor()
        key = encryptor.generate_key()

        encrypted = encryptor.encrypt(sample_data, key)
        decrypted, metrics = encryptor.decrypt_with_metrics(encrypted, key)

        assert decrypted == sample_data
        assert metrics.plaintext_size == len(sample_data)

    def test_invalid_key_size(self):
        """Test rejection of invalid key size."""
        with pytest.raises(Exception):
            AesGcmEncryptor(key_size=24)  # Invalid

    def test_nonce_uniqueness(self, sample_data):
        """Test that nonces are unique."""
        encryptor = AesGcmEncryptor()
        key = encryptor.generate_key()

        nonces = set()
        for _ in range(100):
            result = encryptor.encrypt_with_metrics(sample_data, key)
            nonces.add(result.nonce)

        assert len(nonces) == 100  # All unique

    def test_empty_data(self):
        """Test encryption of empty data."""
        encryptor = AesGcmEncryptor()
        key = encryptor.generate_key()

        encrypted = encryptor.encrypt(b"", key)
        decrypted = encryptor.decrypt(encrypted, key)
        assert decrypted == b""

    def test_large_data(self):
        """Test encryption of large data."""
        encryptor = AesGcmEncryptor()
        key = encryptor.generate_key()

        large_data = b"x" * (1024 * 1024)  # 1 MB
        encrypted = encryptor.encrypt(large_data, key)
        decrypted = encryptor.decrypt(encrypted, key)
        assert decrypted == large_data


@pytest.mark.skipif(not HAS_CRYPTOGRAPHY, reason="cryptography not installed")
class TestChaCha20Poly1305Encryptor:
    """Tests for ChaCha20Poly1305Encryptor."""

    def test_basic_encryption(self, sample_data):
        """Test basic encryption/decryption."""
        encryptor = ChaCha20Poly1305Encryptor()
        assert encryptor.algorithm == EncryptionAlgorithm.CHACHA20_POLY1305

        key = encryptor.generate_key()
        assert len(key) == 32

        encrypted = encryptor.encrypt(sample_data, key)
        decrypted = encryptor.decrypt(encrypted, key)
        assert decrypted == sample_data

    def test_with_aad(self, sample_data, aad):
        """Test with additional authenticated data."""
        encryptor = ChaCha20Poly1305Encryptor()
        key = encryptor.generate_key()

        encrypted = encryptor.encrypt(sample_data, key, aad=aad)
        decrypted = encryptor.decrypt(encrypted, key, aad=aad)
        assert decrypted == sample_data

    def test_tampered_detection(self, sample_data):
        """Test tamper detection."""
        encryptor = ChaCha20Poly1305Encryptor()
        key = encryptor.generate_key()

        encrypted = encryptor.encrypt(sample_data, key)
        tampered = bytes([encrypted[0] ^ 0xFF]) + encrypted[1:]

        with pytest.raises(IntegrityError):
            encryptor.decrypt(tampered, key)


@pytest.mark.skipif(
    not (HAS_CRYPTOGRAPHY or HAS_NACL),
    reason="Neither cryptography nor nacl installed",
)
class TestXChaCha20Poly1305Encryptor:
    """Tests for XChaCha20Poly1305Encryptor."""

    def test_basic_encryption(self, sample_data):
        """Test basic encryption/decryption."""
        encryptor = XChaCha20Poly1305Encryptor()
        assert encryptor.algorithm == EncryptionAlgorithm.XCHACHA20_POLY1305

        key = encryptor.generate_key()
        assert len(key) == 32

        encrypted = encryptor.encrypt(sample_data, key)
        decrypted = encryptor.decrypt(encrypted, key)
        assert decrypted == sample_data

    def test_extended_nonce(self, sample_data):
        """Test that extended nonce is used."""
        encryptor = XChaCha20Poly1305Encryptor()
        assert encryptor.nonce_size == 24  # Extended nonce


@pytest.mark.skipif(not HAS_CRYPTOGRAPHY, reason="cryptography not installed")
class TestFernetEncryptor:
    """Tests for FernetEncryptor."""

    def test_basic_encryption(self, sample_data):
        """Test basic Fernet encryption."""
        encryptor = FernetEncryptor()
        assert encryptor.algorithm == EncryptionAlgorithm.FERNET

        key = encryptor.generate_key()
        assert len(key) == 44  # Base64 encoded

        encrypted = encryptor.encrypt(sample_data, key)
        decrypted = encryptor.decrypt(encrypted, key)
        assert decrypted == sample_data

    def test_invalid_key_format(self, sample_data):
        """Test rejection of invalid Fernet key."""
        encryptor = FernetEncryptor()

        with pytest.raises(Exception):
            encryptor.encrypt(sample_data, b"invalid_key")

    def test_tampered_token(self, sample_data):
        """Test tamper detection."""
        encryptor = FernetEncryptor()
        key = encryptor.generate_key()

        encrypted = encryptor.encrypt(sample_data, key)

        # Tamper with token
        tampered = encrypted[:-1] + bytes([encrypted[-1] ^ 0xFF])

        with pytest.raises(IntegrityError):
            encryptor.decrypt(tampered, key)


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_get_encryptor_noop(self):
        """Test getting noop encryptor."""
        encryptor = get_encryptor("none")
        assert isinstance(encryptor, NoopEncryptor)

    @pytest.mark.skipif(not HAS_CRYPTOGRAPHY, reason="cryptography not installed")
    def test_get_encryptor_aes(self):
        """Test getting AES encryptor."""
        enc_128 = get_encryptor("aes-128-gcm")
        assert enc_128.key_size == 16

        enc_256 = get_encryptor("aes-256-gcm")
        assert enc_256.key_size == 32

        enc_enum = get_encryptor(EncryptionAlgorithm.AES_256_GCM)
        assert enc_enum.key_size == 32

    def test_get_encryptor_invalid(self):
        """Test getting invalid encryptor."""
        with pytest.raises(UnsupportedAlgorithmError):
            get_encryptor("invalid-algorithm")

    def test_list_available_algorithms(self):
        """Test listing available algorithms."""
        available = list_available_algorithms()
        assert "none" in available

        if HAS_CRYPTOGRAPHY:
            assert "aes-256-gcm" in available
            assert "chacha20-poly1305" in available

    def test_is_algorithm_available(self):
        """Test algorithm availability check."""
        assert is_algorithm_available("none")
        assert not is_algorithm_available("fake-algorithm")

    @pytest.mark.skipif(not HAS_CRYPTOGRAPHY, reason="cryptography not installed")
    def test_is_algorithm_available_aes(self):
        """Test AES availability."""
        assert is_algorithm_available("aes-256-gcm")
        assert is_algorithm_available(EncryptionAlgorithm.AES_256_GCM)


@pytest.mark.skipif(not HAS_CRYPTOGRAPHY, reason="cryptography not installed")
class TestCrossAlgorithmCompatibility:
    """Test data encrypted with one algorithm cannot be decrypted with another."""

    def test_cross_algorithm_fails(self, sample_data):
        """Test that cross-algorithm decryption fails."""
        aes = AesGcmEncryptor()
        chacha = ChaCha20Poly1305Encryptor()

        aes_key = aes.generate_key()
        chacha_key = chacha.generate_key()

        aes_encrypted = aes.encrypt(sample_data, aes_key)

        # Cannot decrypt AES ciphertext with ChaCha key/encryptor
        with pytest.raises(Exception):
            chacha.decrypt(aes_encrypted, chacha_key)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.skipif(not HAS_CRYPTOGRAPHY, reason="cryptography not installed")
    def test_binary_data(self):
        """Test encryption of binary data with null bytes."""
        encryptor = AesGcmEncryptor()
        key = encryptor.generate_key()

        binary_data = bytes(range(256)) * 10
        encrypted = encryptor.encrypt(binary_data, key)
        decrypted = encryptor.decrypt(encrypted, key)
        assert decrypted == binary_data

    @pytest.mark.skipif(not HAS_CRYPTOGRAPHY, reason="cryptography not installed")
    def test_unicode_as_bytes(self):
        """Test encryption of unicode content as bytes."""
        encryptor = AesGcmEncryptor()
        key = encryptor.generate_key()

        unicode_data = "Hello, 世界! 🌍".encode("utf-8")
        encrypted = encryptor.encrypt(unicode_data, key)
        decrypted = encryptor.decrypt(encrypted, key)
        assert decrypted == unicode_data
        assert decrypted.decode("utf-8") == "Hello, 世界! 🌍"

    @pytest.mark.skipif(not HAS_CRYPTOGRAPHY, reason="cryptography not installed")
    def test_repeated_encryption(self, sample_data):
        """Test that repeated encryption produces different ciphertexts."""
        encryptor = AesGcmEncryptor()
        key = encryptor.generate_key()

        ciphertexts = set()
        for _ in range(10):
            encrypted = encryptor.encrypt(sample_data, key)
            ciphertexts.add(encrypted)

        # All ciphertexts should be different (due to random nonce)
        assert len(ciphertexts) == 10
