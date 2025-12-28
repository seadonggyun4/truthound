"""Tests for verification chain."""

import pytest
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile

from truthound.plugins.security.signing.verifier import (
    IntegrityVerifier,
    SignatureVerifier,
    TrustVerifier,
    ExpirationVerifier,
    ChainVerifier,
    create_verification_chain,
    VerificationChainBuilder,
)
from truthound.plugins.security.signing.service import SigningServiceImpl
from truthound.plugins.security.signing.trust_store import TrustStoreImpl
from truthound.plugins.security.protocols import SignatureInfo, TrustLevel


class TestIntegrityVerifier:
    """Tests for IntegrityVerifier."""

    def test_integrity_check_passes(self):
        """Test integrity check passes for unmodified file."""
        service = SigningServiceImpl()
        verifier = IntegrityVerifier(service)

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            f.write(b"plugin = True")
            f.flush()
            path = Path(f.name)

        try:
            # Create signature
            signature = service.sign(path, private_key=b"secret")

            # Verify integrity
            context: dict = {}
            result = verifier._do_verify(path, signature, context)

            assert result is None  # None means continue chain
            assert "plugin_hash" in context
        finally:
            path.unlink()

    def test_integrity_check_fails_modified(self):
        """Test integrity check fails for modified file."""
        service = SigningServiceImpl()
        verifier = IntegrityVerifier(service)

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            f.write(b"plugin = True")
            f.flush()
            path = Path(f.name)

        try:
            signature = service.sign(path, private_key=b"secret")

            # Modify file
            path.write_text("plugin = False")

            context: dict = {}
            result = verifier._do_verify(path, signature, context)

            assert result is not None
            assert result.is_valid is False
            assert "modified" in result.errors[0].lower()
        finally:
            path.unlink()

    def test_integrity_check_fails_missing_hash(self):
        """Test integrity check fails when hash is missing from signature."""
        verifier = IntegrityVerifier()

        # Create signature without hash
        signature = SignatureInfo(
            signer_id="test",
            algorithm="SHA256",
            signature=b"sig",
            timestamp=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(days=30),
            metadata={},  # No plugin_hash
        )

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            f.write(b"plugin = True")
            f.flush()
            path = Path(f.name)

        try:
            context: dict = {}
            result = verifier._do_verify(path, signature, context)

            assert result is not None
            assert result.is_valid is False
        finally:
            path.unlink()


class TestExpirationVerifier:
    """Tests for ExpirationVerifier."""

    def test_expiration_check_passes(self):
        """Test expiration check passes for valid signature."""
        verifier = ExpirationVerifier()

        signature = SignatureInfo(
            signer_id="test",
            algorithm="SHA256",
            signature=b"sig",
            timestamp=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(days=30),
        )

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            path = Path(f.name)

        try:
            context: dict = {}
            result = verifier._do_verify(path, signature, context)

            assert result is None  # Continue chain
        finally:
            path.unlink()

    def test_expiration_check_fails_expired(self):
        """Test expiration check fails for expired signature."""
        verifier = ExpirationVerifier()

        signature = SignatureInfo(
            signer_id="test",
            algorithm="SHA256",
            signature=b"sig",
            timestamp=datetime.now(timezone.utc) - timedelta(days=60),
            expires_at=datetime.now(timezone.utc) - timedelta(days=30),  # Expired
        )

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            path = Path(f.name)

        try:
            context: dict = {}
            result = verifier._do_verify(path, signature, context)

            assert result is not None
            assert result.is_valid is False
            assert "expired" in result.errors[0].lower()
        finally:
            path.unlink()

    def test_max_age_enforcement(self):
        """Test max age policy enforcement."""
        verifier = ExpirationVerifier(max_age_days=30)

        # Signature is 60 days old
        signature = SignatureInfo(
            signer_id="test",
            algorithm="SHA256",
            signature=b"sig",
            timestamp=datetime.now(timezone.utc) - timedelta(days=60),
            expires_at=datetime.now(timezone.utc) + timedelta(days=305),  # Not expired
        )

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            path = Path(f.name)

        try:
            context: dict = {}
            result = verifier._do_verify(path, signature, context)

            assert result is not None
            assert result.is_valid is False
            assert "60 days old" in result.errors[0]
        finally:
            path.unlink()


class TestTrustVerifier:
    """Tests for TrustVerifier."""

    def test_trust_check_passes_trusted(self):
        """Test trust check passes for trusted signer."""
        store = TrustStoreImpl(auto_save=False)
        store.set_signer_trust("trusted-signer", TrustLevel.TRUSTED)
        verifier = TrustVerifier(store)

        signature = SignatureInfo(
            signer_id="trusted-signer",
            algorithm="SHA256",
            signature=b"sig",
            timestamp=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(days=30),
        )

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            path = Path(f.name)

        try:
            context: dict = {}
            result = verifier._do_verify(path, signature, context)

            assert result is None
            assert context["trust_level"] == TrustLevel.TRUSTED
        finally:
            path.unlink()

    def test_trust_check_fails_revoked(self):
        """Test trust check fails for revoked signer."""
        store = TrustStoreImpl(auto_save=False)
        store.set_signer_trust("revoked-signer", TrustLevel.REVOKED)
        verifier = TrustVerifier(store)

        signature = SignatureInfo(
            signer_id="revoked-signer",
            algorithm="SHA256",
            signature=b"sig",
            timestamp=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(days=30),
        )

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            path = Path(f.name)

        try:
            context: dict = {}
            result = verifier._do_verify(path, signature, context)

            assert result is not None
            assert result.is_valid is False
            assert "revoked" in result.errors[0].lower()
        finally:
            path.unlink()

    def test_trust_check_unknown_adds_warning(self):
        """Test unknown signer adds warning but continues."""
        store = TrustStoreImpl(auto_save=False)
        verifier = TrustVerifier(store)

        signature = SignatureInfo(
            signer_id="unknown-signer",
            algorithm="SHA256",
            signature=b"sig",
            timestamp=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(days=30),
        )

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            path = Path(f.name)

        try:
            context: dict = {}
            result = verifier._do_verify(path, signature, context)

            assert result is None  # Continue chain
            assert "warnings" in context
            assert any("not in trust store" in w for w in context["warnings"])
        finally:
            path.unlink()


class TestVerificationChain:
    """Tests for verification chain."""

    def test_create_verification_chain(self):
        """Test creating a default verification chain."""
        chain = create_verification_chain()

        assert chain is not None

    def test_chain_execution_success(self):
        """Test full chain execution for valid signature."""
        store = TrustStoreImpl(auto_save=False)
        store.set_signer_trust("truthound", TrustLevel.TRUSTED)

        service = SigningServiceImpl()
        chain = create_verification_chain(
            trust_store=store,
            signing_service=service,
        )

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            f.write(b"plugin = True")
            f.flush()
            path = Path(f.name)

        try:
            signature = service.sign(path, private_key=b"secret")
            context: dict = {}
            result = chain.verify(path, signature, context)

            assert result.is_valid is True
        finally:
            path.unlink()

    def test_chain_execution_failure(self):
        """Test chain stops on first failure."""
        service = SigningServiceImpl()
        chain = create_verification_chain(signing_service=service)

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            f.write(b"plugin = True")
            f.flush()
            path = Path(f.name)

        try:
            signature = service.sign(path, private_key=b"secret")

            # Modify file to fail integrity check
            path.write_text("plugin = False")

            context: dict = {}
            result = chain.verify(path, signature, context)

            assert result.is_valid is False
            # Should fail at integrity, not continue to other checks
            assert "modified" in result.errors[0].lower()
        finally:
            path.unlink()


class TestVerificationChainBuilder:
    """Tests for VerificationChainBuilder."""

    def test_build_empty_chain_raises(self):
        """Test building empty chain raises error."""
        builder = VerificationChainBuilder()

        with pytest.raises(ValueError, match="No verification handlers"):
            builder.build()

    def test_build_custom_chain(self):
        """Test building custom verification chain."""
        store = TrustStoreImpl(auto_save=False)

        chain = (
            VerificationChainBuilder()
            .with_integrity_check()
            .with_expiration_check(max_age_days=90)
            .with_trust_check(store)
            .build()
        )

        assert chain is not None

    def test_build_minimal_chain(self):
        """Test building minimal chain."""
        chain = (
            VerificationChainBuilder()
            .with_integrity_check()
            .build()
        )

        assert chain is not None

    def test_build_chain_with_custom_handler(self):
        """Test adding custom handler to chain."""
        from truthound.plugins.security.signing.verifier import VerificationHandlerBase

        class CustomHandler(VerificationHandlerBase):
            def _do_verify(self, plugin_path, signature, context):
                context["custom_check"] = True
                return None

        chain = (
            VerificationChainBuilder()
            .with_integrity_check()
            .with_custom_handler(CustomHandler())
            .build()
        )

        # Verify custom handler is in chain
        service = SigningServiceImpl()
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            f.write(b"plugin = True")
            f.flush()
            path = Path(f.name)

        try:
            signature = service.sign(path, private_key=b"secret")
            context: dict = {}
            chain.verify(path, signature, context)

            assert context.get("custom_check") is True
        finally:
            path.unlink()
