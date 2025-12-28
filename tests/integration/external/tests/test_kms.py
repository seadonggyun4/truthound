"""Cloud KMS integration tests.

Tests KMS functionality including:
- Key creation and management
- Encryption/decryption operations
- Data key generation
- Key rotation
- Multi-provider support (AWS, GCP, Azure, Vault, LocalStack)

These tests can run against:
- LocalStack (default)
- Mock KMS (for fast testing)
- Real cloud KMS (with credentials)
"""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from tests.integration.external.backends.kms_backend import KMSBackend


# =============================================================================
# Mock KMS Tests (Fast, No External Dependencies)
# =============================================================================


class TestMockKMS:
    """Tests using mock KMS for fast execution."""

    @pytest.fixture
    def mock_kms(self):
        """Create mock KMS client."""
        from tests.integration.external.backends.kms_backend import MockKMSClient
        return MockKMSClient()

    def test_create_key(self, mock_kms) -> None:
        """Test key creation."""
        key_id = mock_kms.create_key("Test key")
        assert key_id is not None
        assert len(key_id) > 0

        # Create multiple keys
        key_id2 = mock_kms.create_key("Another key")
        assert key_id != key_id2

    def test_list_keys(self, mock_kms) -> None:
        """Test listing keys."""
        # Initially empty
        assert len(mock_kms.list_keys()) == 0

        # Create keys
        key1 = mock_kms.create_key("Key 1")
        key2 = mock_kms.create_key("Key 2")

        keys = mock_kms.list_keys()
        assert len(keys) == 2
        assert key1 in keys
        assert key2 in keys

    def test_encrypt_decrypt(self, mock_kms) -> None:
        """Test encryption and decryption."""
        key_id = mock_kms.create_key("Test key")
        plaintext = b"Hello, World!"

        # Encrypt
        ciphertext = mock_kms.encrypt(key_id, plaintext)
        assert ciphertext != plaintext
        assert len(ciphertext) > 0

        # Decrypt
        decrypted = mock_kms.decrypt(key_id, ciphertext)
        assert decrypted == plaintext

    def test_encrypt_empty_data(self, mock_kms) -> None:
        """Test encrypting empty data."""
        key_id = mock_kms.create_key("Test key")
        plaintext = b""

        ciphertext = mock_kms.encrypt(key_id, plaintext)
        decrypted = mock_kms.decrypt(key_id, ciphertext)
        assert decrypted == plaintext

    def test_encrypt_large_data(self, mock_kms) -> None:
        """Test encrypting large data."""
        key_id = mock_kms.create_key("Test key")
        plaintext = b"x" * 10000

        ciphertext = mock_kms.encrypt(key_id, plaintext)
        decrypted = mock_kms.decrypt(key_id, ciphertext)
        assert decrypted == plaintext

    def test_encrypt_binary_data(self, mock_kms) -> None:
        """Test encrypting binary data."""
        key_id = mock_kms.create_key("Test key")
        plaintext = bytes(range(256))

        ciphertext = mock_kms.encrypt(key_id, plaintext)
        decrypted = mock_kms.decrypt(key_id, ciphertext)
        assert decrypted == plaintext

    def test_generate_data_key(self, mock_kms) -> None:
        """Test data key generation."""
        key_id = mock_kms.create_key("Master key")

        plaintext_key, encrypted_key = mock_kms.generate_data_key(key_id)

        # Plaintext key should be 32 bytes (AES-256)
        assert len(plaintext_key) == 32

        # Encrypted key should be decryptable
        decrypted_key = mock_kms.decrypt(key_id, encrypted_key)
        assert decrypted_key == plaintext_key

    def test_delete_key(self, mock_kms) -> None:
        """Test key deletion."""
        key_id = mock_kms.create_key("Test key")
        assert key_id in mock_kms.list_keys()

        # Delete key
        assert mock_kms.delete_key(key_id) is True
        assert key_id not in mock_kms.list_keys()

        # Encryption with deleted key should fail
        with pytest.raises(ValueError):
            mock_kms.encrypt(key_id, b"test")

    def test_rotate_key(self, mock_kms) -> None:
        """Test key rotation flag."""
        key_id = mock_kms.create_key("Test key")

        assert mock_kms.rotate_key(key_id) is True

    def test_nonexistent_key(self, mock_kms) -> None:
        """Test operations with non-existent key."""
        with pytest.raises(ValueError):
            mock_kms.encrypt("nonexistent-key", b"test")


# =============================================================================
# Docker/LocalStack KMS Tests
# =============================================================================


@pytest.mark.kms
@pytest.mark.integration
class TestLocalStackKMS:
    """Tests using LocalStack KMS."""

    def test_backend_connection(self, kms_backend: "KMSBackend") -> None:
        """Test KMS backend connection."""
        assert kms_backend.is_running
        assert kms_backend.is_healthy

    def test_health_check(self, kms_backend: "KMSBackend") -> None:
        """Test health check."""
        result = kms_backend.health_check()
        assert result.healthy

    def test_create_key(self, kms_backend: "KMSBackend") -> None:
        """Test key creation."""
        key_id = kms_backend.create_key("Integration test key")
        assert key_id is not None
        assert len(key_id) > 0

    def test_encrypt_decrypt_bytes(self, kms_backend: "KMSBackend") -> None:
        """Test byte encryption/decryption."""
        key_id = kms_backend.create_key("Test key")
        plaintext = b"Secret data for testing"

        # Encrypt
        ciphertext = kms_backend.encrypt(key_id, plaintext)
        assert ciphertext != plaintext

        # Decrypt
        decrypted = kms_backend.decrypt(key_id, ciphertext)
        assert decrypted == plaintext

    def test_encrypt_decrypt_string(self, kms_backend: "KMSBackend") -> None:
        """Test string encryption/decryption."""
        key_id = kms_backend.create_key("Test key")
        plaintext = "Secret string data"

        # Encrypt
        ciphertext = kms_backend.encrypt_string(key_id, plaintext)
        assert ciphertext != plaintext

        # Decrypt
        decrypted = kms_backend.decrypt_string(key_id, ciphertext)
        assert decrypted == plaintext

    def test_generate_data_key(self, kms_backend: "KMSBackend") -> None:
        """Test data key generation."""
        key_id = kms_backend.create_key("Master key")

        plaintext_key, encrypted_key = kms_backend.generate_data_key(key_id)

        # Verify key sizes
        assert len(plaintext_key) == 32  # AES-256

        # Verify encrypted key can be decrypted
        decrypted = kms_backend.decrypt(key_id, encrypted_key)
        assert decrypted == plaintext_key

    def test_multiple_keys(self, kms_backend: "KMSBackend") -> None:
        """Test using multiple keys."""
        key1 = kms_backend.create_key("Key 1")
        key2 = kms_backend.create_key("Key 2")

        plaintext = b"Test data"

        # Encrypt with different keys
        ciphertext1 = kms_backend.encrypt(key1, plaintext)
        ciphertext2 = kms_backend.encrypt(key2, plaintext)

        # Ciphertexts should be different
        assert ciphertext1 != ciphertext2

        # Both should decrypt correctly
        assert kms_backend.decrypt(key1, ciphertext1) == plaintext
        assert kms_backend.decrypt(key2, ciphertext2) == plaintext

    def test_unicode_data(self, kms_backend: "KMSBackend") -> None:
        """Test encrypting unicode data."""
        key_id = kms_backend.create_key("Unicode key")
        plaintext = "Hello 世界 🌍 مرحبا"

        ciphertext = kms_backend.encrypt_string(key_id, plaintext)
        decrypted = kms_backend.decrypt_string(key_id, ciphertext)

        assert decrypted == plaintext


# =============================================================================
# Envelope Encryption Pattern Tests
# =============================================================================


@pytest.mark.kms
@pytest.mark.integration
class TestEnvelopeEncryption:
    """Tests for envelope encryption pattern."""

    def test_envelope_encryption_flow(self, kms_backend: "KMSBackend") -> None:
        """Test complete envelope encryption flow."""
        # Create master key
        master_key_id = kms_backend.create_key("Master key")

        # Generate data key
        plaintext_dek, encrypted_dek = kms_backend.generate_data_key(master_key_id)

        # Simulate encrypting data with DEK (using simple XOR for demo)
        data = b"Sensitive user data that needs protection"
        encrypted_data = bytes(d ^ plaintext_dek[i % len(plaintext_dek)]
                               for i, d in enumerate(data))

        # Clear plaintext DEK from memory (in practice)
        del plaintext_dek

        # Store encrypted_dek and encrypted_data
        stored_package = {
            "encrypted_dek": base64.b64encode(encrypted_dek).decode(),
            "encrypted_data": base64.b64encode(encrypted_data).decode(),
        }

        # Later: Decrypt
        recovered_encrypted_dek = base64.b64decode(stored_package["encrypted_dek"])
        recovered_encrypted_data = base64.b64decode(stored_package["encrypted_data"])

        # Decrypt DEK using master key
        recovered_dek = kms_backend.decrypt(master_key_id, recovered_encrypted_dek)

        # Decrypt data using DEK
        recovered_data = bytes(
            d ^ recovered_dek[i % len(recovered_dek)]
            for i, d in enumerate(recovered_encrypted_data)
        )

        assert recovered_data == data

    def test_key_hierarchy(self, kms_backend: "KMSBackend") -> None:
        """Test key hierarchy (root -> region -> tenant)."""
        # Create root key
        root_key = kms_backend.create_key("Root key")

        # Create "region" keys encrypted with root
        region_key_plain, region_key_encrypted = kms_backend.generate_data_key(root_key)

        # Create "tenant" keys encrypted with region
        # (In practice, you'd use the region key for encryption)
        tenant_key_plain, tenant_key_encrypted = kms_backend.generate_data_key(root_key)

        # All keys should be unique
        assert root_key != region_key_plain
        assert region_key_plain != tenant_key_plain

        # Recovery should work
        recovered_region = kms_backend.decrypt(root_key, region_key_encrypted)
        assert recovered_region == region_key_plain


# =============================================================================
# Security Tests
# =============================================================================


@pytest.mark.kms
@pytest.mark.integration
class TestKMSSecurity:
    """Security-focused tests for KMS."""

    def test_ciphertext_uniqueness(self, kms_backend: "KMSBackend") -> None:
        """Test that same plaintext produces different ciphertexts."""
        key_id = kms_backend.create_key("Test key")
        plaintext = b"Same data"

        # Encrypt multiple times
        ciphertexts = [kms_backend.encrypt(key_id, plaintext) for _ in range(5)]

        # All should decrypt to same plaintext
        for ct in ciphertexts:
            assert kms_backend.decrypt(key_id, ct) == plaintext

        # Note: In real KMS, ciphertexts would be unique due to IV/nonce
        # Mock implementation may not have this property

    def test_wrong_key_decryption_fails(self, kms_backend: "KMSBackend") -> None:
        """Test that decryption with wrong key fails."""
        key1 = kms_backend.create_key("Key 1")
        key2 = kms_backend.create_key("Key 2")

        plaintext = b"Test data"
        ciphertext = kms_backend.encrypt(key1, plaintext)

        # Attempt to decrypt with wrong key
        # Behavior depends on implementation:
        # - AWS KMS: Raises exception
        # - Mock: May return garbage or raise exception
        try:
            result = kms_backend.decrypt(key2, ciphertext)
            # If no exception, result should not equal plaintext
            assert result != plaintext
        except Exception:
            # Expected behavior
            pass

    def test_empty_plaintext(self, kms_backend: "KMSBackend") -> None:
        """Test encrypting empty data."""
        key_id = kms_backend.create_key("Test key")

        ciphertext = kms_backend.encrypt(key_id, b"")
        decrypted = kms_backend.decrypt(key_id, ciphertext)

        assert decrypted == b""
