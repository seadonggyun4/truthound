"""Tests for key management module."""

import os
import tempfile
from datetime import timedelta
from pathlib import Path

import pytest

from truthound.stores.encryption.base import (
    EncryptionAlgorithm,
    KeyDerivation,
    KeyExpiredError,
    KeyType,
)
from truthound.stores.encryption.keys import (
    # Key derivation
    Argon2KeyDeriver,
    HKDFKeyDeriver,
    PBKDF2KeyDeriver,
    ScryptKeyDeriver,
    derive_key,
    get_key_deriver,
    # Key storage
    EnvironmentKeyStore,
    FileKeyStore,
    InMemoryKeyStore,
    # Key manager
    KeyManager,
    KeyManagerConfig,
    # Envelope encryption
    EnvelopeEncryptedData,
    EnvelopeEncryption,
)


# Check for optional dependencies
try:
    import argon2

    HAS_ARGON2 = True
except ImportError:
    HAS_ARGON2 = False

try:
    import cryptography

    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False


class TestPBKDF2KeyDeriver:
    """Tests for PBKDF2 key derivation (always available)."""

    def test_derive_key_sha256(self):
        """Test PBKDF2-SHA256 key derivation."""
        deriver = PBKDF2KeyDeriver(hash_name="sha256", iterations=1000)
        salt = b"test_salt_16byte"

        key = deriver.derive("password", salt, 32)
        assert len(key) == 32

        # Same inputs should produce same key
        key2 = deriver.derive("password", salt, 32)
        assert key == key2

    def test_derive_key_sha512(self):
        """Test PBKDF2-SHA512 key derivation."""
        deriver = PBKDF2KeyDeriver(hash_name="sha512", iterations=1000)
        salt = b"test_salt_16byte"

        key = deriver.derive("password", salt, 32)
        assert len(key) == 32

    def test_different_passwords(self):
        """Test that different passwords produce different keys."""
        deriver = PBKDF2KeyDeriver(iterations=1000)
        salt = b"test_salt_16byte"

        key1 = deriver.derive("password1", salt, 32)
        key2 = deriver.derive("password2", salt, 32)
        assert key1 != key2

    def test_different_salts(self):
        """Test that different salts produce different keys."""
        deriver = PBKDF2KeyDeriver(iterations=1000)

        key1 = deriver.derive("password", b"salt1___________", 32)
        key2 = deriver.derive("password", b"salt2___________", 32)
        assert key1 != key2

    def test_bytes_password(self):
        """Test derivation with bytes password."""
        deriver = PBKDF2KeyDeriver(iterations=1000)
        salt = b"test_salt_16byte"

        key = deriver.derive(b"password_bytes", salt, 32)
        assert len(key) == 32


class TestScryptKeyDeriver:
    """Tests for scrypt key derivation."""

    def test_derive_key(self):
        """Test scrypt key derivation."""
        deriver = ScryptKeyDeriver(n=2**10, r=8, p=1)  # Lower params for testing
        salt = b"test_salt_16byte"

        key = deriver.derive("password", salt, 32)
        assert len(key) == 32

    def test_deterministic(self):
        """Test that derivation is deterministic."""
        deriver = ScryptKeyDeriver(n=2**10, r=8, p=1)
        salt = b"test_salt_16byte"

        key1 = deriver.derive("password", salt, 32)
        key2 = deriver.derive("password", salt, 32)
        assert key1 == key2


@pytest.mark.skipif(not HAS_ARGON2, reason="argon2-cffi not installed")
class TestArgon2KeyDeriver:
    """Tests for Argon2 key derivation."""

    def test_argon2id(self):
        """Test Argon2id key derivation."""
        deriver = Argon2KeyDeriver(
            variant=KeyDerivation.ARGON2ID,
            time_cost=1,
            memory_cost=1024,
            parallelism=1,
        )
        salt = b"test_salt_16byte"

        key = deriver.derive("password", salt, 32)
        assert len(key) == 32

    def test_argon2i(self):
        """Test Argon2i key derivation."""
        deriver = Argon2KeyDeriver(
            variant=KeyDerivation.ARGON2I,
            time_cost=1,
            memory_cost=1024,
            parallelism=1,
        )
        salt = b"test_salt_16byte"

        key = deriver.derive("password", salt, 32)
        assert len(key) == 32

    def test_argon2d(self):
        """Test Argon2d key derivation."""
        deriver = Argon2KeyDeriver(
            variant=KeyDerivation.ARGON2D,
            time_cost=1,
            memory_cost=1024,
            parallelism=1,
        )
        salt = b"test_salt_16byte"

        key = deriver.derive("password", salt, 32)
        assert len(key) == 32


@pytest.mark.skipif(not HAS_CRYPTOGRAPHY, reason="cryptography not installed")
class TestHKDFKeyDeriver:
    """Tests for HKDF key derivation."""

    def test_hkdf_sha256(self):
        """Test HKDF-SHA256 key derivation."""
        deriver = HKDFKeyDeriver(hash_name="sha256")
        salt = b"test_salt_16byte"

        key = deriver.derive(b"input_key_material", salt, 32)
        assert len(key) == 32

    def test_hkdf_sha512(self):
        """Test HKDF-SHA512 key derivation."""
        deriver = HKDFKeyDeriver(hash_name="sha512")
        salt = b"test_salt_16byte"

        key = deriver.derive(b"input_key_material", salt, 64)
        assert len(key) == 64


class TestDeriveKeyFunction:
    """Tests for the derive_key convenience function."""

    def test_derive_with_generated_salt(self):
        """Test key derivation with auto-generated salt."""
        key, salt = derive_key("password", kdf=KeyDerivation.PBKDF2_SHA256)

        assert len(key) == 32
        assert len(salt) == 16

    def test_derive_with_provided_salt(self):
        """Test key derivation with provided salt."""
        salt = b"provided_salt___"
        key, returned_salt = derive_key("password", salt=salt, kdf=KeyDerivation.PBKDF2_SHA256)

        assert len(key) == 32
        assert returned_salt == salt

    def test_derive_custom_key_size(self):
        """Test derivation with custom key size."""
        key, salt = derive_key("password", key_size=64, kdf=KeyDerivation.PBKDF2_SHA256)
        assert len(key) == 64


class TestGetKeyDeriver:
    """Tests for get_key_deriver factory."""

    def test_get_pbkdf2(self):
        """Test getting PBKDF2 deriver."""
        deriver = get_key_deriver(KeyDerivation.PBKDF2_SHA256)
        assert isinstance(deriver, PBKDF2KeyDeriver)

    def test_get_scrypt(self):
        """Test getting scrypt deriver."""
        deriver = get_key_deriver(KeyDerivation.SCRYPT)
        assert isinstance(deriver, ScryptKeyDeriver)

    @pytest.mark.skipif(not HAS_ARGON2, reason="argon2-cffi not installed")
    def test_get_argon2(self):
        """Test getting Argon2 deriver."""
        deriver = get_key_deriver(KeyDerivation.ARGON2ID)
        assert isinstance(deriver, Argon2KeyDeriver)


class TestInMemoryKeyStore:
    """Tests for InMemoryKeyStore."""

    def test_put_and_get(self):
        """Test storing and retrieving keys."""
        from truthound.stores.encryption.base import EncryptionKey

        store = InMemoryKeyStore()
        key = EncryptionKey(
            key_id="test",
            key_material=b"x" * 32,
            algorithm=EncryptionAlgorithm.AES_256_GCM,
        )

        store.put(key)
        retrieved = store.get("test")

        assert retrieved is not None
        assert retrieved.key_id == "test"

    def test_get_nonexistent(self):
        """Test getting nonexistent key."""
        store = InMemoryKeyStore()
        assert store.get("nonexistent") is None

    def test_delete(self):
        """Test key deletion."""
        from truthound.stores.encryption.base import EncryptionKey

        store = InMemoryKeyStore()
        key = EncryptionKey(
            key_id="test",
            key_material=b"x" * 32,
            algorithm=EncryptionAlgorithm.AES_256_GCM,
        )

        store.put(key)
        assert store.delete("test")
        assert store.get("test") is None
        assert not store.delete("test")  # Already deleted

    def test_list_keys(self):
        """Test listing keys."""
        from truthound.stores.encryption.base import EncryptionKey

        store = InMemoryKeyStore()
        for i in range(3):
            key = EncryptionKey(
                key_id=f"key_{i}",
                key_material=b"x" * 32,
                algorithm=EncryptionAlgorithm.AES_256_GCM,
            )
            store.put(key)

        keys = store.list_keys()
        assert len(keys) == 3
        assert "key_0" in keys

    def test_exists(self):
        """Test key existence check."""
        from truthound.stores.encryption.base import EncryptionKey

        store = InMemoryKeyStore()
        key = EncryptionKey(
            key_id="test",
            key_material=b"x" * 32,
            algorithm=EncryptionAlgorithm.AES_256_GCM,
        )

        assert not store.exists("test")
        store.put(key)
        assert store.exists("test")

    def test_clear(self):
        """Test clearing all keys."""
        from truthound.stores.encryption.base import EncryptionKey

        store = InMemoryKeyStore()
        for i in range(3):
            key = EncryptionKey(
                key_id=f"key_{i}",
                key_material=b"x" * 32,
                algorithm=EncryptionAlgorithm.AES_256_GCM,
            )
            store.put(key)

        store.clear()
        assert len(store.list_keys()) == 0


@pytest.mark.skipif(not HAS_CRYPTOGRAPHY, reason="cryptography not installed")
class TestFileKeyStore:
    """Tests for FileKeyStore."""

    def test_put_and_get_with_password(self):
        """Test storing and retrieving with password protection."""
        from truthound.stores.encryption.base import EncryptionKey

        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileKeyStore(tmpdir, master_password="test_master")
            key = EncryptionKey(
                key_id="test",
                key_material=b"x" * 32,
                algorithm=EncryptionAlgorithm.AES_256_GCM,
            )

            store.put(key)
            retrieved = store.get("test")

            assert retrieved is not None
            assert retrieved.key_material == key.key_material

    def test_persistence(self):
        """Test key persistence across store instances."""
        from truthound.stores.encryption.base import EncryptionKey

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and store key
            store1 = FileKeyStore(tmpdir, master_password="test")
            key = EncryptionKey(
                key_id="persistent",
                key_material=b"x" * 32,
                algorithm=EncryptionAlgorithm.AES_256_GCM,
            )
            store1.put(key)

            # Create new store instance and retrieve
            store2 = FileKeyStore(tmpdir, master_password="test")
            retrieved = store2.get("persistent")

            assert retrieved is not None
            assert retrieved.key_material == key.key_material

    def test_delete(self):
        """Test key deletion."""
        from truthound.stores.encryption.base import EncryptionKey

        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileKeyStore(tmpdir, master_password="test")
            key = EncryptionKey(
                key_id="test",
                key_material=b"x" * 32,
                algorithm=EncryptionAlgorithm.AES_256_GCM,
            )

            store.put(key)
            assert store.delete("test")
            assert store.get("test") is None


class TestEnvironmentKeyStore:
    """Tests for EnvironmentKeyStore."""

    def test_put_and_get(self):
        """Test storing and retrieving from environment."""
        import base64

        store = EnvironmentKeyStore(prefix="TEST_KEY_")

        # Manually set environment variable
        key_material = b"x" * 32
        os.environ["TEST_KEY_mykey"] = base64.b64encode(key_material).decode()

        try:
            retrieved = store.get("mykey")
            assert retrieved is not None
            assert retrieved.key_material == key_material
        finally:
            del os.environ["TEST_KEY_mykey"]

    def test_delete(self):
        """Test deleting from environment."""
        import base64

        store = EnvironmentKeyStore(prefix="TEST_KEY_")
        os.environ["TEST_KEY_deleteme"] = base64.b64encode(b"x" * 32).decode()

        assert store.delete("deleteme")
        assert "TEST_KEY_deleteme" not in os.environ

    def test_list_keys(self):
        """Test listing environment keys."""
        import base64

        store = EnvironmentKeyStore(prefix="TEST_LISTKEY_")

        os.environ["TEST_LISTKEY_a"] = base64.b64encode(b"x" * 32).decode()
        os.environ["TEST_LISTKEY_b"] = base64.b64encode(b"x" * 32).decode()

        try:
            keys = store.list_keys()
            assert "a" in keys
            assert "b" in keys
        finally:
            del os.environ["TEST_LISTKEY_a"]
            del os.environ["TEST_LISTKEY_b"]


class TestKeyManager:
    """Tests for KeyManager."""

    def test_create_key(self):
        """Test key creation."""
        manager = KeyManager()
        key = manager.create_key(key_id="test_key")

        assert key.key_id == "test_key"
        assert len(key.key_material) == 32
        assert key.algorithm == EncryptionAlgorithm.AES_256_GCM

    def test_create_key_auto_id(self):
        """Test key creation with auto-generated ID."""
        manager = KeyManager()
        key = manager.create_key()

        assert key.key_id is not None
        assert len(key.key_id) == 32

    def test_get_key(self):
        """Test key retrieval."""
        manager = KeyManager()
        created = manager.create_key(key_id="test")
        retrieved = manager.get_key("test")

        assert retrieved.key_material == created.key_material

    def test_get_key_not_found(self):
        """Test getting nonexistent key."""
        from truthound.stores.encryption.keys import KeyError_

        manager = KeyManager()
        with pytest.raises(KeyError_):
            manager.get_key("nonexistent")

    def test_rotate_key(self):
        """Test key rotation."""
        manager = KeyManager()
        original = manager.create_key(key_id="rotate_me")
        rotated = manager.rotate_key("rotate_me")

        assert rotated.key_id == "rotate_me"
        assert rotated.version == 2
        assert rotated.key_material != original.key_material

    def test_delete_key(self):
        """Test key deletion."""
        manager = KeyManager()
        manager.create_key(key_id="delete_me")

        assert manager.delete_key("delete_me")
        with pytest.raises(Exception):
            manager.get_key("delete_me")

    def test_list_keys(self):
        """Test listing keys."""
        manager = KeyManager()
        for i in range(3):
            manager.create_key(key_id=f"key_{i}")

        keys = manager.list_keys()
        assert len(keys) == 3

    def test_get_or_create_key(self):
        """Test get_or_create_key."""
        manager = KeyManager()

        # Create new
        key1 = manager.get_or_create_key("new_key")
        assert key1 is not None

        # Get existing
        key2 = manager.get_or_create_key("new_key")
        assert key2.key_material == key1.key_material

    def test_key_with_ttl(self):
        """Test key with TTL."""
        config = KeyManagerConfig(default_ttl=timedelta(hours=1))
        manager = KeyManager(config=config)

        key = manager.create_key(key_id="expiring")
        assert key.expires_at is not None

    def test_cleanup_expired(self):
        """Test cleanup of expired keys."""
        from datetime import datetime, timezone

        from truthound.stores.encryption.base import EncryptionKey

        manager = KeyManager()

        # Create expired key directly in store
        expired_key = EncryptionKey(
            key_id="expired",
            key_material=b"x" * 32,
            algorithm=EncryptionAlgorithm.AES_256_GCM,
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        manager._store.put(expired_key)

        manager.create_key(key_id="valid")

        deleted = manager.cleanup_expired()
        assert deleted == 1
        assert len(manager.list_keys()) == 1

    def test_audit_hook(self):
        """Test audit hook callback."""
        audit_events = []

        def audit_hook(event: str, details: dict):
            audit_events.append((event, details))

        manager = KeyManager(audit_hook=audit_hook)
        manager.create_key(key_id="audited")

        assert len(audit_events) >= 1
        assert audit_events[0][0] == "key_created"


@pytest.mark.skipif(not HAS_CRYPTOGRAPHY, reason="cryptography not installed")
class TestEnvelopeEncryption:
    """Tests for envelope encryption."""

    def test_encrypt_decrypt(self):
        """Test envelope encryption round-trip."""
        manager = KeyManager()
        kek = manager.create_key(key_id="master", key_type=KeyType.KEY_ENCRYPTION_KEY)

        envelope = EnvelopeEncryption(manager)

        plaintext = b"sensitive data" * 100
        encrypted = envelope.encrypt(plaintext, "master")
        decrypted = envelope.decrypt(encrypted)

        assert decrypted == plaintext

    def test_encrypted_data_serialization(self):
        """Test EnvelopeEncryptedData serialization."""
        manager = KeyManager()
        manager.create_key(key_id="master", key_type=KeyType.KEY_ENCRYPTION_KEY)

        envelope = EnvelopeEncryption(manager)

        plaintext = b"test data"
        encrypted = envelope.encrypt(plaintext, "master")

        # Serialize and deserialize
        serialized = encrypted.to_bytes()
        restored = EnvelopeEncryptedData.from_bytes(serialized)

        assert restored.kek_id == encrypted.kek_id
        assert restored.encrypted_data == encrypted.encrypted_data

        # Can still decrypt
        decrypted = envelope.decrypt(restored)
        assert decrypted == plaintext

    def test_reencrypt_key(self):
        """Test DEK re-encryption for key rotation."""
        manager = KeyManager()
        manager.create_key(key_id="master_v1", key_type=KeyType.KEY_ENCRYPTION_KEY)
        manager.create_key(key_id="master_v2", key_type=KeyType.KEY_ENCRYPTION_KEY)

        envelope = EnvelopeEncryption(manager)

        plaintext = b"test data"
        encrypted = envelope.encrypt(plaintext, "master_v1")

        # Re-encrypt DEK with new KEK
        reencrypted = envelope.reencrypt_key(encrypted, "master_v2")

        assert reencrypted.kek_id == "master_v2"
        assert reencrypted.encrypted_data == encrypted.encrypted_data

        # Can still decrypt with new KEK
        decrypted = envelope.decrypt(reencrypted)
        assert decrypted == plaintext
