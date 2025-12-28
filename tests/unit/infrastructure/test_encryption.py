"""Tests for enterprise encryption system."""

import base64
import os
import struct
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Check if optional dependencies are available
try:
    import boto3
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False

from truthound.infrastructure.encryption import (
    # Key providers
    KeyProvider,
    LocalKeyProvider,
    VaultKeyProvider,
    AwsKmsProvider,
    GcpKmsProvider,
    AzureKeyVaultProvider,
    # Encryption classes
    AtRestEncryption,
    FieldLevelEncryption,
    EncryptedData,
    FieldEncryptionPolicy,
    EnterpriseEncryptionConfig,
    # Factory functions
    get_encryptor,
    configure_encryption,
)

from truthound.stores.encryption import (
    EncryptionAlgorithm,
    EncryptionError,
    DecryptionError,
)


class TestLocalKeyProvider:
    """Tests for LocalKeyProvider."""

    def test_create_provider(self):
        """Test creating local key provider."""
        provider = LocalKeyProvider(
            key_file=".test_keys",
            master_password="test_password",
        )

        assert provider._master_password == "test_password"

    def test_get_key_generates_key(self):
        """Test that get_key generates a new key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            key_file = Path(tmpdir) / ".keys"
            provider = LocalKeyProvider(
                key_file=key_file,
                master_password="test",
            )

            key = provider.get_key("test-key")

            assert len(key) == 32
            assert isinstance(key, bytes)

    def test_get_key_returns_same_key(self):
        """Test that get_key returns the same key for the same ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            key_file = Path(tmpdir) / ".keys"
            provider = LocalKeyProvider(
                key_file=key_file,
                master_password="test",
            )

            key1 = provider.get_key("test-key")
            key2 = provider.get_key("test-key")

            assert key1 == key2

    def test_encrypt_decrypt_data_key(self):
        """Test encrypting and decrypting data keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            key_file = Path(tmpdir) / ".keys"
            provider = LocalKeyProvider(
                key_file=key_file,
                master_password="test",
            )

            data_key = b"0123456789abcdef0123456789abcdef"
            encrypted = provider.encrypt_data_key(data_key, "master-key")
            decrypted = provider.decrypt_data_key(encrypted, "master-key")

            assert decrypted == data_key

    def test_generate_data_key(self):
        """Test generating a data key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            key_file = Path(tmpdir) / ".keys"
            provider = LocalKeyProvider(
                key_file=key_file,
                master_password="test",
            )

            plain_key, encrypted_key = provider.generate_data_key("master-key")

            assert len(plain_key) == 32
            assert len(encrypted_key) > 0

            # Decrypt should return original
            decrypted = provider.decrypt_data_key(encrypted_key, "master-key")
            assert decrypted == plain_key


class TestVaultKeyProvider:
    """Tests for VaultKeyProvider."""

    def test_create_provider(self):
        """Test creating Vault provider."""
        provider = VaultKeyProvider(
            url="http://localhost:8200",
            token="test-token",
            mount_point="transit",
        )

        assert provider._url == "http://localhost:8200"
        assert provider._token == "test-token"
        assert provider._mount_point == "transit"

        provider.close()

    def test_get_key_generates_random(self):
        """Test that get_key generates random bytes."""
        provider = VaultKeyProvider(
            url="http://localhost:8200",
            token="test-token",
        )

        key1 = provider.get_key("test-key")
        key2 = provider.get_key("test-key")

        # Each call generates new random key
        assert len(key1) == 32
        assert len(key2) == 32
        assert key1 != key2  # Random each time

        provider.close()


class TestAwsKmsProvider:
    """Tests for AwsKmsProvider."""

    def test_create_provider(self):
        """Test creating AWS KMS provider."""
        provider = AwsKmsProvider(
            key_id="alias/my-key",
            region="us-east-1",
        )

        assert provider._key_id == "alias/my-key"
        assert provider._region == "us-east-1"

    @pytest.mark.skipif(not HAS_BOTO3, reason="boto3 not installed")
    @patch("boto3.client")
    def test_encrypt_data_key(self, mock_boto3_client):
        """Test encrypting data key with KMS."""
        mock_client = MagicMock()
        mock_client.encrypt.return_value = {
            "CiphertextBlob": b"encrypted-key"
        }
        mock_boto3_client.return_value = mock_client

        provider = AwsKmsProvider(
            key_id="alias/my-key",
            region="us-east-1",
        )

        encrypted = provider.encrypt_data_key(b"plain-key", "alias/my-key")

        assert encrypted == b"encrypted-key"
        mock_client.encrypt.assert_called_once()

    @pytest.mark.skipif(not HAS_BOTO3, reason="boto3 not installed")
    @patch("boto3.client")
    def test_decrypt_data_key(self, mock_boto3_client):
        """Test decrypting data key with KMS."""
        mock_client = MagicMock()
        mock_client.decrypt.return_value = {
            "Plaintext": b"plain-key"
        }
        mock_boto3_client.return_value = mock_client

        provider = AwsKmsProvider(
            key_id="alias/my-key",
            region="us-east-1",
        )

        decrypted = provider.decrypt_data_key(b"encrypted-key", "alias/my-key")

        assert decrypted == b"plain-key"
        mock_client.decrypt.assert_called_once()

    @pytest.mark.skipif(not HAS_BOTO3, reason="boto3 not installed")
    @patch("boto3.client")
    def test_generate_data_key(self, mock_boto3_client):
        """Test generating data key with KMS."""
        mock_client = MagicMock()
        mock_client.generate_data_key.return_value = {
            "Plaintext": b"0" * 32,
            "CiphertextBlob": b"encrypted-key",
        }
        mock_boto3_client.return_value = mock_client

        provider = AwsKmsProvider(
            key_id="alias/my-key",
            region="us-east-1",
        )

        plain_key, encrypted_key = provider.generate_data_key("alias/my-key")

        assert plain_key == b"0" * 32
        assert encrypted_key == b"encrypted-key"


class TestGcpKmsProvider:
    """Tests for GcpKmsProvider."""

    def test_create_provider(self):
        """Test creating GCP KMS provider."""
        provider = GcpKmsProvider(
            key_name="my-key",
            project_id="my-project",
            location="us-east1",
            key_ring="my-ring",
        )

        assert provider._key_name == "my-key"
        assert provider._project_id == "my-project"
        assert provider._location == "us-east1"
        assert provider._key_ring == "my-ring"

    def test_get_key_path(self):
        """Test key path generation."""
        provider = GcpKmsProvider(
            key_name="my-key",
            project_id="my-project",
            location="global",
            key_ring="truthound",
        )

        path = provider._get_key_path("my-key")

        assert path == "projects/my-project/locations/global/keyRings/truthound/cryptoKeys/my-key"


class TestAzureKeyVaultProvider:
    """Tests for AzureKeyVaultProvider."""

    def test_create_provider(self):
        """Test creating Azure Key Vault provider."""
        provider = AzureKeyVaultProvider(
            vault_url="https://my-vault.vault.azure.net",
            key_name="my-key",
        )

        assert provider._vault_url == "https://my-vault.vault.azure.net"
        assert provider._key_name == "my-key"


class TestEncryptedData:
    """Tests for EncryptedData."""

    def test_create_encrypted_data(self):
        """Test creating encrypted data."""
        encrypted = EncryptedData(
            ciphertext=b"encrypted-content",
            encrypted_key=b"encrypted-key",
            key_id="my-key",
            algorithm="AES-256-GCM",
            nonce=b"123456789012",
            metadata={"type": "test"},
        )

        assert encrypted.ciphertext == b"encrypted-content"
        assert encrypted.key_id == "my-key"
        assert encrypted.algorithm == "AES-256-GCM"
        assert encrypted.metadata["type"] == "test"

    def test_serialize_deserialize(self):
        """Test serialization and deserialization."""
        original = EncryptedData(
            ciphertext=b"encrypted-content",
            encrypted_key=b"encrypted-key",
            key_id="my-key",
            algorithm="AES-256-GCM",
            nonce=b"123456789012",
            metadata={"type": "test"},
        )

        # Serialize
        serialized = original.to_bytes()

        # Deserialize
        restored = EncryptedData.from_bytes(serialized)

        assert restored.ciphertext == original.ciphertext
        assert restored.encrypted_key == original.encrypted_key
        assert restored.key_id == original.key_id
        assert restored.algorithm == original.algorithm
        assert restored.nonce == original.nonce
        assert restored.metadata == original.metadata


class TestAtRestEncryption:
    """Tests for AtRestEncryption."""

    def setup_method(self):
        """Create fresh encryptor for each test."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.tmpdir = tmpdir
            self.key_file = Path(tmpdir) / ".keys"
            self.provider = LocalKeyProvider(
                key_file=self.key_file,
                master_password="test",
            )
        # Create new provider for each test
        self.provider = LocalKeyProvider(
            key_file=Path(tempfile.gettempdir()) / f".test_keys_{id(self)}",
            master_password="test",
        )
        self.encryptor = AtRestEncryption(
            provider=self.provider,
            key_id="default",
        )

    def test_encrypt_decrypt(self):
        """Test encrypting and decrypting data."""
        plaintext = b"This is sensitive data that needs encryption."

        encrypted = self.encryptor.encrypt(plaintext)
        decrypted = self.encryptor.decrypt(encrypted)

        assert decrypted == plaintext

    def test_encrypt_with_metadata(self):
        """Test encrypting with metadata."""
        plaintext = b"Sensitive data"

        encrypted = self.encryptor.encrypt(
            plaintext,
            metadata={"source": "test", "version": 1},
        )

        assert encrypted.metadata["source"] == "test"
        assert encrypted.metadata["version"] == 1

    def test_encrypt_different_keys(self):
        """Test that different data produces different ciphertext."""
        data1 = b"Data one"
        data2 = b"Data two"

        encrypted1 = self.encryptor.encrypt(data1)
        encrypted2 = self.encryptor.encrypt(data2)

        assert encrypted1.ciphertext != encrypted2.ciphertext

    def test_encrypt_file(self):
        """Test file encryption."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "plaintext.txt"
            output_file = Path(tmpdir) / "encrypted.bin"
            decrypted_file = Path(tmpdir) / "decrypted.txt"

            # Write test data
            input_file.write_bytes(b"File contents to encrypt")

            # Encrypt
            self.encryptor.encrypt_file(input_file, output_file)

            # Verify encrypted file exists and is different
            assert output_file.exists()
            assert output_file.read_bytes() != input_file.read_bytes()

            # Decrypt
            self.encryptor.decrypt_file(output_file, decrypted_file)

            # Verify decrypted content
            assert decrypted_file.read_bytes() == input_file.read_bytes()


class TestFieldLevelEncryption:
    """Tests for FieldLevelEncryption."""

    def setup_method(self):
        """Create fresh FLE for each test."""
        self.provider = LocalKeyProvider(
            key_file=Path(tempfile.gettempdir()) / f".test_keys_fle_{id(self)}",
            master_password="test",
        )
        self.fle = FieldLevelEncryption(
            provider=self.provider,
            policies={
                "ssn": FieldEncryptionPolicy(
                    format_preserving=True,
                ),
                "email": FieldEncryptionPolicy(
                    algorithm="aes_gcm",
                ),
                "credit_card": FieldEncryptionPolicy(
                    mask_format="****-****-****-{last4}",
                ),
            },
            default_key_id="default",
        )

    def test_encrypt_decrypt_field(self):
        """Test encrypting and decrypting a field."""
        value = "test@example.com"

        encrypted = self.fle.encrypt_field("email", value)
        decrypted = self.fle.decrypt_field("email", encrypted)

        assert decrypted == value

    def test_format_preserving_encryption(self):
        """Test format-preserving encryption."""
        ssn = "123-45-6789"

        encrypted = self.fle.encrypt_field("ssn", ssn)

        # Format should be preserved (XXX-XX-XXXX)
        assert len(encrypted) == len(ssn)
        assert encrypted[3] == "-"
        assert encrypted[6] == "-"

    def test_add_policy(self):
        """Test adding a policy."""
        self.fle.add_policy(
            "phone",
            FieldEncryptionPolicy(algorithm="aes_gcm"),
        )

        assert "phone" in self.fle._policies

    def test_mask_field(self):
        """Test field masking."""
        credit_card = "4111111111111234"

        masked = self.fle.mask_field("credit_card", credit_card)

        assert masked == "****-****-****-1234"

    def test_mask_field_default(self):
        """Test default masking."""
        value = "sensitive_data"

        masked = self.fle.mask_field("unknown_field", value)

        # Default masking: keep first 2 and last 2 chars
        assert masked.startswith("se")
        assert masked.endswith("ta")
        assert "*" in masked

    def test_deterministic_encryption(self):
        """Test deterministic encryption produces same output."""
        self.fle.add_policy(
            "search_field",
            FieldEncryptionPolicy(deterministic=True),
        )

        value = "searchable"

        encrypted1 = self.fle.encrypt_field("search_field", value)
        encrypted2 = self.fle.encrypt_field("search_field", value)

        # Deterministic should produce same ciphertext
        assert encrypted1 == encrypted2


class TestFieldEncryptionPolicy:
    """Tests for FieldEncryptionPolicy."""

    def test_default_policy(self):
        """Test default policy values."""
        policy = FieldEncryptionPolicy()

        assert policy.algorithm == "aes_gcm"
        assert policy.key_id == ""
        assert policy.format_preserving is False
        assert policy.deterministic is False
        assert policy.mask_format == ""

    def test_custom_policy(self):
        """Test custom policy values."""
        policy = FieldEncryptionPolicy(
            algorithm="chacha20",
            key_id="custom-key",
            format_preserving=True,
            mask_format="XXX-{last4}",
        )

        assert policy.algorithm == "chacha20"
        assert policy.key_id == "custom-key"
        assert policy.format_preserving is True
        assert policy.mask_format == "XXX-{last4}"


class TestEnterpriseEncryptionConfig:
    """Tests for EnterpriseEncryptionConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = EnterpriseEncryptionConfig()

        assert config.enabled is True
        assert config.provider == "local"

    def test_aws_config(self):
        """Test AWS KMS configuration."""
        config = EnterpriseEncryptionConfig(
            provider="aws_kms",
            key_id="alias/my-key",
            aws_region="us-west-2",
        )

        assert config.provider == "aws_kms"
        assert config.key_id == "alias/my-key"
        assert config.aws_region == "us-west-2"

    def test_vault_config(self):
        """Test Vault configuration."""
        config = EnterpriseEncryptionConfig(
            provider="vault",
            vault_url="http://vault.example.com:8200",
            vault_token="s.xxxxx",
            vault_mount_point="transit",
        )

        assert config.provider == "vault"
        assert config.vault_url == "http://vault.example.com:8200"
        assert config.vault_mount_point == "transit"


class TestGlobalEncryption:
    """Tests for global encryption functions."""

    def test_configure_encryption_local(self):
        """Test configuring local encryption."""
        encryptor = configure_encryption(
            provider="local",
            key_id="test-key",
            local_master_password="test",
        )

        assert encryptor is not None
        assert encryptor._key_id == "test-key"

    def test_get_encryptor(self):
        """Test getting global encryptor."""
        encryptor1 = get_encryptor()
        encryptor2 = get_encryptor()

        # Should return same instance
        assert encryptor1 is encryptor2

    def test_configure_encryption_vault(self):
        """Test configuring Vault encryption."""
        encryptor = configure_encryption(
            provider="vault",
            key_id="my-key",
            vault_url="http://localhost:8200",
            vault_token="test-token",
        )

        assert encryptor is not None
        assert isinstance(encryptor._provider, VaultKeyProvider)

    @pytest.mark.skipif(not HAS_BOTO3, reason="boto3 not installed")
    def test_configure_encryption_aws(self):
        """Test configuring AWS KMS encryption."""
        with patch("boto3.client"):
            encryptor = configure_encryption(
                provider="aws_kms",
                key_id="alias/my-key",
                aws_region="us-east-1",
            )

            assert encryptor is not None
            assert isinstance(encryptor._provider, AwsKmsProvider)


class TestEncryptionIntegration:
    """Integration tests for encryption system."""

    def test_end_to_end_encryption(self):
        """Test end-to-end encryption workflow."""
        # Configure
        encryptor = configure_encryption(
            provider="local",
            key_id="integration-test",
            local_master_password="integration",
        )

        # Encrypt
        plaintext = b"Integration test data"
        encrypted = encryptor.encrypt(plaintext, metadata={"test": True})

        # Serialize and deserialize
        serialized = encrypted.to_bytes()
        restored = EncryptedData.from_bytes(serialized)

        # Decrypt
        decrypted = encryptor.decrypt(restored)

        assert decrypted == plaintext

    def test_field_level_with_at_rest(self):
        """Test combining field-level and at-rest encryption."""
        provider = LocalKeyProvider(
            key_file=Path(tempfile.gettempdir()) / ".integration_keys",
            master_password="test",
        )

        # Field-level encryption
        fle = FieldLevelEncryption(
            provider=provider,
            policies={
                "ssn": FieldEncryptionPolicy(algorithm="aes_gcm"),
            },
            default_key_id="fle-key",
        )

        # Encrypt field
        ssn = "123-45-6789"
        encrypted_ssn = fle.encrypt_field("ssn", ssn)

        # At-rest encryption of the encrypted field
        at_rest = AtRestEncryption(provider=provider, key_id="at-rest-key")
        double_encrypted = at_rest.encrypt(encrypted_ssn.encode())

        # Decrypt at-rest
        decrypted_stage1 = at_rest.decrypt(double_encrypted)

        # Decrypt field
        decrypted_ssn = fle.decrypt_field("ssn", decrypted_stage1.decode())

        assert decrypted_ssn == ssn
