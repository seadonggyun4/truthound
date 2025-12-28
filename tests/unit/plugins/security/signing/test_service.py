"""Tests for signing service implementation."""

import pytest
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile

from truthound.plugins.security.signing.service import (
    SigningServiceImpl,
    SignatureAlgorithm,
)
from truthound.plugins.security.protocols import TrustLevel
from truthound.plugins.security.exceptions import SignatureError


class TestSigningServiceBasics:
    """Basic tests for SigningServiceImpl."""

    def test_create_service_defaults(self):
        """Test creating service with defaults."""
        service = SigningServiceImpl()

        assert service.algorithm == SignatureAlgorithm.SHA256
        assert service.signer_id == "truthound"
        assert service.validity_days == 365

    def test_create_service_custom(self):
        """Test creating service with custom values."""
        service = SigningServiceImpl(
            algorithm=SignatureAlgorithm.HMAC_SHA256,
            signer_id="my-signer",
            validity_days=90,
        )

        assert service.algorithm == SignatureAlgorithm.HMAC_SHA256
        assert service.signer_id == "my-signer"
        assert service.validity_days == 90


class TestSignatureAlgorithm:
    """Tests for signature algorithms."""

    def test_all_algorithms_defined(self):
        """Test all algorithms are defined."""
        algorithms = [
            SignatureAlgorithm.SHA256,
            SignatureAlgorithm.SHA512,
            SignatureAlgorithm.HMAC_SHA256,
            SignatureAlgorithm.HMAC_SHA512,
            SignatureAlgorithm.RSA_SHA256,
            SignatureAlgorithm.ED25519,
        ]
        assert len(algorithms) == 6


class TestPluginHashing:
    """Tests for plugin hashing functionality."""

    def test_hash_single_file(self):
        """Test hashing a single file."""
        service = SigningServiceImpl()

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            f.write(b"def hello(): return 'world'")
            f.flush()
            path = Path(f.name)

        try:
            hash1 = service.get_plugin_hash(path)

            assert isinstance(hash1, str)
            assert len(hash1) == 64  # SHA256 hex length

            # Same content should produce same hash
            hash2 = service.get_plugin_hash(path)
            assert hash1 == hash2
        finally:
            path.unlink()

    def test_hash_directory(self):
        """Test hashing a directory."""
        service = SigningServiceImpl()

        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_dir = Path(tmpdir)

            # Create some Python files
            (plugin_dir / "main.py").write_text("def main(): pass")
            (plugin_dir / "utils.py").write_text("def util(): pass")

            hash_result = service.get_plugin_hash(plugin_dir)

            assert isinstance(hash_result, str)
            assert len(hash_result) == 64

    def test_hash_changes_with_content(self):
        """Test hash changes when content changes."""
        service = SigningServiceImpl()

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            f.write(b"version = 1")
            f.flush()
            path = Path(f.name)

        try:
            hash1 = service.get_plugin_hash(path)

            # Modify file
            path.write_text("version = 2")
            hash2 = service.get_plugin_hash(path)

            assert hash1 != hash2
        finally:
            path.unlink()

    def test_hash_nonexistent_path_raises(self):
        """Test hashing nonexistent path raises error."""
        service = SigningServiceImpl()

        with pytest.raises(SignatureError, match="does not exist"):
            service.get_plugin_hash(Path("/nonexistent/path"))


class TestPluginSigning:
    """Tests for plugin signing."""

    def test_sign_file(self):
        """Test signing a file."""
        service = SigningServiceImpl()

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            f.write(b"plugin_code = True")
            f.flush()
            path = Path(f.name)

        try:
            signature = service.sign(path, private_key=b"secret")

            assert signature.signer_id == "truthound"
            assert signature.algorithm == "SHA256"
            assert signature.signature is not None
            assert signature.timestamp is not None
            assert signature.expires_at is not None
            assert "plugin_hash" in signature.metadata
        finally:
            path.unlink()

    def test_sign_with_metadata(self):
        """Test signing with additional metadata."""
        service = SigningServiceImpl()

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            f.write(b"plugin = True")
            f.flush()
            path = Path(f.name)

        try:
            signature = service.sign(
                path,
                private_key=b"secret",
                metadata={"version": "1.0.0", "author": "test"},
            )

            assert signature.metadata["version"] == "1.0.0"
            assert signature.metadata["author"] == "test"
            assert "plugin_hash" in signature.metadata
        finally:
            path.unlink()

    def test_signature_expiration(self):
        """Test signature expiration is set correctly."""
        service = SigningServiceImpl(validity_days=30)

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            f.write(b"plugin = True")
            f.flush()
            path = Path(f.name)

        try:
            signature = service.sign(path, private_key=b"secret")

            now = datetime.now(timezone.utc)
            expected_expiry = now + timedelta(days=30)

            # Should expire around 30 days from now
            diff = abs((signature.expires_at - expected_expiry).total_seconds())
            assert diff < 60  # Within 1 minute
        finally:
            path.unlink()


class TestPluginVerification:
    """Tests for plugin verification."""

    def test_verify_valid_signature(self):
        """Test verifying a valid signature."""
        service = SigningServiceImpl()

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            f.write(b"plugin = True")
            f.flush()
            path = Path(f.name)

        try:
            # Sign the plugin
            signature = service.sign(path, private_key=b"secret")

            # Verify
            result = service.verify(path, signature)

            assert result.is_valid is True
            assert result.trust_level == TrustLevel.VERIFIED
        finally:
            path.unlink()

    def test_verify_modified_file_fails(self):
        """Test verification fails if file is modified."""
        service = SigningServiceImpl()

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            f.write(b"plugin = True")
            f.flush()
            path = Path(f.name)

        try:
            # Sign the plugin
            signature = service.sign(path, private_key=b"secret")

            # Modify the file
            path.write_text("plugin = False  # modified")

            # Verify should fail
            result = service.verify(path, signature)

            assert result.is_valid is False
            assert "modified" in result.errors[0].lower()
        finally:
            path.unlink()


class TestHMACAlgorithms:
    """Tests for HMAC-based algorithms."""

    def test_hmac_sha256_signing(self):
        """Test HMAC-SHA256 signing."""
        service = SigningServiceImpl(algorithm=SignatureAlgorithm.HMAC_SHA256)

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            f.write(b"plugin = True")
            f.flush()
            path = Path(f.name)

        try:
            signature = service.sign(path, private_key=b"secret_key")

            assert signature.algorithm == "HMAC_SHA256"
            assert len(signature.signature) == 32  # HMAC-SHA256 produces 32 bytes
        finally:
            path.unlink()

    def test_hmac_sha512_signing(self):
        """Test HMAC-SHA512 signing."""
        service = SigningServiceImpl(algorithm=SignatureAlgorithm.HMAC_SHA512)

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            f.write(b"plugin = True")
            f.flush()
            path = Path(f.name)

        try:
            signature = service.sign(path, private_key=b"secret_key")

            assert signature.algorithm == "HMAC_SHA512"
            assert len(signature.signature) == 64  # HMAC-SHA512 produces 64 bytes
        finally:
            path.unlink()


class TestSignatureVerificationMethods:
    """Tests for signature verification methods."""

    def test_verify_sha256_signature(self):
        """Test verifying SHA256 signature."""
        service = SigningServiceImpl(algorithm=SignatureAlgorithm.SHA256)

        data = b"test data"
        signature = service._create_signature(data, b"")

        result = service.verify_signature(data, signature)
        assert result is True

    def test_verify_hmac_signature(self):
        """Test verifying HMAC signature."""
        service = SigningServiceImpl(algorithm=SignatureAlgorithm.HMAC_SHA256)

        data = b"test data"
        secret = b"secret_key"
        signature = service._create_signature(data, secret)

        result = service.verify_signature(data, signature, secret=secret)
        assert result is True

    def test_verify_hmac_wrong_secret_fails(self):
        """Test HMAC verification fails with wrong secret."""
        service = SigningServiceImpl(algorithm=SignatureAlgorithm.HMAC_SHA256)

        data = b"test data"
        signature = service._create_signature(data, b"secret_key")

        result = service.verify_signature(data, signature, secret=b"wrong_key")
        assert result is False
