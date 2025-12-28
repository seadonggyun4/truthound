"""Tests for trust store implementation."""

import pytest
from datetime import datetime, timezone
from pathlib import Path
import tempfile

from truthound.plugins.security.signing.trust_store import (
    TrustStoreImpl,
    CertificateEntry,
)
from truthound.plugins.security.protocols import TrustLevel
from truthound.plugins.security.exceptions import CertificateRevokedError


class TestCertificateEntry:
    """Tests for CertificateEntry dataclass."""

    def test_create_entry(self):
        """Test creating a certificate entry."""
        entry = CertificateEntry(
            cert_id="abc123",
            certificate=b"cert_data",
            trust_level=TrustLevel.TRUSTED,
        )

        assert entry.cert_id == "abc123"
        assert entry.certificate == b"cert_data"
        assert entry.trust_level == TrustLevel.TRUSTED
        assert entry.added_at is not None
        assert entry.expires_at is None
        assert entry.revoked_at is None

    def test_is_revoked(self):
        """Test is_revoked property."""
        entry = CertificateEntry(
            cert_id="abc123",
            certificate=b"cert_data",
            trust_level=TrustLevel.TRUSTED,
        )

        assert entry.is_revoked is False

        # Revoke it
        revoked_entry = CertificateEntry(
            cert_id=entry.cert_id,
            certificate=entry.certificate,
            trust_level=TrustLevel.REVOKED,
            revoked_at=datetime.now(timezone.utc),
            revocation_reason="Test revocation",
        )

        assert revoked_entry.is_revoked is True

    def test_is_expired(self):
        """Test is_expired property."""
        from datetime import timedelta

        # Not expired
        entry = CertificateEntry(
            cert_id="abc123",
            certificate=b"cert_data",
            trust_level=TrustLevel.TRUSTED,
            expires_at=datetime.now(timezone.utc) + timedelta(days=30),
        )

        assert entry.is_expired is False

        # Expired
        expired_entry = CertificateEntry(
            cert_id="abc456",
            certificate=b"cert_data",
            trust_level=TrustLevel.TRUSTED,
            expires_at=datetime.now(timezone.utc) - timedelta(days=1),
        )

        assert expired_entry.is_expired is True

    def test_is_valid(self):
        """Test is_valid property."""
        entry = CertificateEntry(
            cert_id="abc123",
            certificate=b"cert_data",
            trust_level=TrustLevel.TRUSTED,
        )

        assert entry.is_valid is True

    def test_to_dict(self):
        """Test converting entry to dictionary."""
        entry = CertificateEntry(
            cert_id="abc123",
            certificate=b"cert_data",
            trust_level=TrustLevel.TRUSTED,
            metadata={"key": "value"},
        )

        data = entry.to_dict()

        assert data["cert_id"] == "abc123"
        assert data["trust_level"] == TrustLevel.TRUSTED.value
        assert data["metadata"]["key"] == "value"


class TestTrustStoreImpl:
    """Tests for TrustStoreImpl."""

    def test_create_empty_store(self):
        """Test creating an empty trust store."""
        store = TrustStoreImpl()

        assert len(store.list_certificates()) == 0
        assert len(store.list_trusted_signers()) == 0

    def test_add_trusted_certificate(self):
        """Test adding a trusted certificate."""
        store = TrustStoreImpl(auto_save=False)

        cert = b"test_certificate_data"
        cert_id = store.add_trusted_certificate(cert)

        assert cert_id is not None
        assert len(cert_id) == 16  # SHA256 hex prefix

        # Should be retrievable
        is_trusted, level = store.is_trusted(cert)
        assert is_trusted is True
        assert level == TrustLevel.TRUSTED

    def test_add_certificate_with_level(self):
        """Test adding certificate with specific trust level."""
        store = TrustStoreImpl(auto_save=False)

        cert = b"test_certificate_data"
        store.add_trusted_certificate(cert, trust_level=TrustLevel.VERIFIED)

        is_trusted, level = store.is_trusted(cert)
        assert is_trusted is True
        assert level == TrustLevel.VERIFIED

    def test_add_certificate_with_metadata(self):
        """Test adding certificate with metadata."""
        store = TrustStoreImpl(auto_save=False)

        cert = b"test_certificate_data"
        store.add_trusted_certificate(
            cert,
            metadata={"issuer": "Test CA", "purpose": "testing"},
        )

        certs = store.list_certificates()
        assert len(certs) == 1
        assert certs[0]["metadata"]["issuer"] == "Test CA"

    def test_is_trusted_unknown_certificate(self):
        """Test checking unknown certificate."""
        store = TrustStoreImpl(auto_save=False)

        is_trusted, level = store.is_trusted(b"unknown_cert")

        assert is_trusted is False
        assert level == TrustLevel.UNKNOWN


class TestTrustStoreRevocation:
    """Tests for certificate revocation."""

    def test_revoke_certificate(self):
        """Test revoking a certificate."""
        store = TrustStoreImpl(auto_save=False)

        cert = b"test_certificate"
        cert_id = store.add_trusted_certificate(cert)

        # Revoke
        result = store.revoke_certificate(cert_id, reason="Compromised key")

        assert result is True

        # Should no longer be trusted
        is_trusted, level = store.is_trusted(cert)
        assert is_trusted is False
        assert level == TrustLevel.REVOKED

    def test_revoke_nonexistent_certificate(self):
        """Test revoking nonexistent certificate."""
        store = TrustStoreImpl(auto_save=False)

        result = store.revoke_certificate("nonexistent")

        assert result is False

    def test_readd_revoked_certificate_raises(self):
        """Test re-adding revoked certificate raises error."""
        store = TrustStoreImpl(auto_save=False)

        cert = b"test_certificate"
        cert_id = store.add_trusted_certificate(cert)
        store.revoke_certificate(cert_id, reason="Test")

        with pytest.raises(CertificateRevokedError):
            store.add_trusted_certificate(cert)


class TestTrustStoreCertificateManagement:
    """Tests for certificate management."""

    def test_remove_certificate(self):
        """Test removing a certificate."""
        store = TrustStoreImpl(auto_save=False)

        cert = b"test_certificate"
        cert_id = store.add_trusted_certificate(cert)

        result = store.remove_certificate(cert_id)

        assert result is True

        is_trusted, level = store.is_trusted(cert)
        assert is_trusted is False

    def test_remove_nonexistent_certificate(self):
        """Test removing nonexistent certificate."""
        store = TrustStoreImpl(auto_save=False)

        result = store.remove_certificate("nonexistent")

        assert result is False

    def test_list_certificates(self):
        """Test listing all certificates."""
        store = TrustStoreImpl(auto_save=False)

        store.add_trusted_certificate(b"cert1")
        store.add_trusted_certificate(b"cert2")
        store.add_trusted_certificate(b"cert3")

        certs = store.list_certificates()

        assert len(certs) == 3

    def test_clear_store(self):
        """Test clearing the store."""
        store = TrustStoreImpl(auto_save=False)

        store.add_trusted_certificate(b"cert1")
        store.add_trusted_certificate(b"cert2")

        store.clear()

        assert len(store.list_certificates()) == 0


class TestTrustStoreSignerTrust:
    """Tests for signer trust management."""

    def test_set_signer_trust(self):
        """Test setting signer trust level."""
        store = TrustStoreImpl(auto_save=False)

        store.set_signer_trust("signer-1", TrustLevel.TRUSTED)

        level = store.get_trust_level("signer-1")
        assert level == TrustLevel.TRUSTED

    def test_get_unknown_signer_trust(self):
        """Test getting trust level for unknown signer."""
        store = TrustStoreImpl(auto_save=False)

        level = store.get_trust_level("unknown-signer")

        assert level == TrustLevel.UNKNOWN

    def test_list_trusted_signers(self):
        """Test listing trusted signers."""
        store = TrustStoreImpl(auto_save=False)

        store.set_signer_trust("signer-1", TrustLevel.TRUSTED)
        store.set_signer_trust("signer-2", TrustLevel.TRUSTED)
        store.set_signer_trust("signer-3", TrustLevel.VERIFIED)  # Not TRUSTED level

        signers = store.list_trusted_signers()

        assert "signer-1" in signers
        assert "signer-2" in signers
        assert "signer-3" not in signers  # Only TRUSTED level counts


class TestTrustStorePersistence:
    """Tests for trust store persistence."""

    def test_save_and_load(self):
        """Test saving and loading trust store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "trust_store.json"

            # Create and populate store
            store1 = TrustStoreImpl(store_path=store_path, auto_save=True)
            store1.set_signer_trust("signer-1", TrustLevel.TRUSTED)
            store1.set_signer_trust("signer-2", TrustLevel.VERIFIED)

            # Create new store from same path
            store2 = TrustStoreImpl(store_path=store_path)

            # Should have loaded the data
            assert store2.get_trust_level("signer-1") == TrustLevel.TRUSTED
            assert store2.get_trust_level("signer-2") == TrustLevel.VERIFIED

    def test_auto_save_off(self):
        """Test auto-save can be disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "trust_store.json"

            store = TrustStoreImpl(store_path=store_path, auto_save=False)
            store.set_signer_trust("signer-1", TrustLevel.TRUSTED)

            # File should not exist
            assert not store_path.exists()
