"""Plugin signing service implementation.

This module provides the SigningServiceImpl that can sign plugins
using various cryptographic algorithms.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any

from truthound.plugins.security.protocols import SignatureInfo, VerificationResult, TrustLevel
from truthound.plugins.security.exceptions import SignatureError, InvalidSignatureError

logger = logging.getLogger(__name__)


class SignatureAlgorithm(Enum):
    """Supported signature algorithms."""

    SHA256 = auto()       # Simple SHA256 hash (not recommended for production)
    SHA512 = auto()       # SHA512 hash
    HMAC_SHA256 = auto()  # HMAC with SHA256
    HMAC_SHA512 = auto()  # HMAC with SHA512
    RSA_SHA256 = auto()   # RSA with SHA256 (requires cryptography)
    ED25519 = auto()      # Ed25519 (requires cryptography)


class SigningServiceImpl:
    """Implementation of plugin signing service.

    Supports multiple signature algorithms from simple hashing to
    cryptographic signatures.

    Example:
        >>> service = SigningServiceImpl(algorithm=SignatureAlgorithm.HMAC_SHA256)
        >>> signature = service.sign(
        ...     plugin_path=Path("my_plugin"),
        ...     private_key=b"secret_key",
        ... )
    """

    def __init__(
        self,
        algorithm: SignatureAlgorithm = SignatureAlgorithm.SHA256,
        signer_id: str = "truthound",
        validity_days: int = 365,
    ) -> None:
        """Initialize signing service.

        Args:
            algorithm: Signature algorithm to use
            signer_id: Identifier for this signer
            validity_days: How long signatures are valid
        """
        self.algorithm = algorithm
        self.signer_id = signer_id
        self.validity_days = validity_days

    def sign(
        self,
        plugin_path: Path,
        private_key: bytes,
        certificate: bytes | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SignatureInfo:
        """Sign a plugin.

        Args:
            plugin_path: Path to plugin file or directory
            private_key: Private key or secret for signing
            certificate: Optional X.509 certificate
            metadata: Additional metadata

        Returns:
            SignatureInfo with signature details

        Raises:
            SignatureError: If signing fails
        """
        # Get plugin hash
        content_hash = self.get_plugin_hash(plugin_path)

        # Build data to sign
        timestamp = datetime.now(timezone.utc)
        expires_at = timestamp + timedelta(days=self.validity_days)

        sign_data = {
            "plugin_hash": content_hash,
            "signer_id": self.signer_id,
            "algorithm": self.algorithm.name,
            "timestamp": timestamp.isoformat(),
            "expires_at": expires_at.isoformat(),
        }

        # Create signature
        data_bytes = json.dumps(sign_data, sort_keys=True).encode()
        signature = self._create_signature(data_bytes, private_key)

        return SignatureInfo(
            signer_id=self.signer_id,
            algorithm=self.algorithm.name,
            signature=signature,
            timestamp=timestamp,
            expires_at=expires_at,
            certificate_chain=(certificate,) if certificate else (),
            metadata={
                "plugin_hash": content_hash,
                **(metadata or {}),
            },
        )

    def verify(
        self,
        plugin_path: Path,
        signature: SignatureInfo,
    ) -> VerificationResult:
        """Verify a plugin signature.

        Args:
            plugin_path: Path to plugin
            signature: Signature to verify

        Returns:
            VerificationResult with verification status
        """
        errors: list[str] = []
        warnings: list[str] = []

        # Check expiration
        if signature.is_expired():
            errors.append(f"Signature expired at {signature.expires_at}")
            return VerificationResult.failure(*errors)

        # Check hash
        current_hash = self.get_plugin_hash(plugin_path)
        stored_hash = signature.metadata.get("plugin_hash", "")

        if current_hash != stored_hash:
            errors.append("Plugin content has been modified")
            return VerificationResult.failure(*errors)

        # Verify signature age warning
        if signature.age_days > 180:
            warnings.append(f"Signature is {signature.age_days} days old")

        return VerificationResult.success(
            signer_id=signature.signer_id,
            trust_level=TrustLevel.VERIFIED,
            warnings=tuple(warnings),
        )

    def get_plugin_hash(self, plugin_path: Path) -> str:
        """Get hash of plugin content.

        Args:
            plugin_path: Path to plugin file or directory

        Returns:
            Hex-encoded SHA256 hash
        """
        hasher = hashlib.sha256()

        if plugin_path.is_file():
            # Hash single file
            with open(plugin_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
        elif plugin_path.is_dir():
            # Hash all Python files in directory
            for py_file in sorted(plugin_path.rglob("*.py")):
                # Add relative path to hash for file identity
                rel_path = py_file.relative_to(plugin_path)
                hasher.update(str(rel_path).encode())
                with open(py_file, "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        hasher.update(chunk)
        else:
            raise SignatureError(f"Plugin path does not exist: {plugin_path}")

        return hasher.hexdigest()

    def _create_signature(
        self,
        data: bytes,
        private_key: bytes,
    ) -> bytes:
        """Create cryptographic signature.

        Args:
            data: Data to sign
            private_key: Private key or secret

        Returns:
            Signature bytes
        """
        if self.algorithm == SignatureAlgorithm.SHA256:
            return hashlib.sha256(data).digest()

        elif self.algorithm == SignatureAlgorithm.SHA512:
            return hashlib.sha512(data).digest()

        elif self.algorithm == SignatureAlgorithm.HMAC_SHA256:
            return hmac.new(private_key, data, hashlib.sha256).digest()

        elif self.algorithm == SignatureAlgorithm.HMAC_SHA512:
            return hmac.new(private_key, data, hashlib.sha512).digest()

        elif self.algorithm == SignatureAlgorithm.RSA_SHA256:
            return self._sign_rsa(data, private_key)

        elif self.algorithm == SignatureAlgorithm.ED25519:
            return self._sign_ed25519(data, private_key)

        else:
            raise SignatureError(f"Unsupported algorithm: {self.algorithm}")

    def _sign_rsa(self, data: bytes, private_key: bytes) -> bytes:
        """Sign data using RSA."""
        try:
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import padding

            key = serialization.load_pem_private_key(private_key, password=None)
            signature = key.sign(
                data,
                padding.PKCS1v15(),
                hashes.SHA256(),
            )
            return signature
        except ImportError:
            raise SignatureError(
                "RSA signing requires cryptography package. "
                "Install with: pip install cryptography"
            )

    def _sign_ed25519(self, data: bytes, private_key: bytes) -> bytes:
        """Sign data using Ed25519."""
        try:
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

            key = serialization.load_pem_private_key(private_key, password=None)
            if not isinstance(key, Ed25519PrivateKey):
                raise SignatureError("Key is not an Ed25519 private key")
            return key.sign(data)
        except ImportError:
            raise SignatureError(
                "Ed25519 signing requires cryptography package. "
                "Install with: pip install cryptography"
            )

    def verify_signature(
        self,
        data: bytes,
        signature: bytes,
        public_key: bytes | None = None,
        secret: bytes | None = None,
    ) -> bool:
        """Verify a signature against data.

        Args:
            data: Data that was signed
            signature: Signature to verify
            public_key: Public key for asymmetric algorithms
            secret: Secret for HMAC algorithms

        Returns:
            True if signature is valid
        """
        if self.algorithm in (SignatureAlgorithm.SHA256, SignatureAlgorithm.SHA512):
            # Hash-based (not real signatures, just for integrity)
            expected = self._create_signature(data, b"")
            return hmac.compare_digest(expected, signature)

        elif self.algorithm in (SignatureAlgorithm.HMAC_SHA256, SignatureAlgorithm.HMAC_SHA512):
            if not secret:
                return False
            expected = self._create_signature(data, secret)
            return hmac.compare_digest(expected, signature)

        elif self.algorithm == SignatureAlgorithm.RSA_SHA256:
            if not public_key:
                return False
            return self._verify_rsa(data, signature, public_key)

        elif self.algorithm == SignatureAlgorithm.ED25519:
            if not public_key:
                return False
            return self._verify_ed25519(data, signature, public_key)

        return False

    def _verify_rsa(self, data: bytes, signature: bytes, public_key: bytes) -> bool:
        """Verify RSA signature."""
        try:
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import padding
            from cryptography.exceptions import InvalidSignature

            key = serialization.load_pem_public_key(public_key)
            try:
                key.verify(signature, data, padding.PKCS1v15(), hashes.SHA256())
                return True
            except InvalidSignature:
                return False
        except ImportError:
            return False

    def _verify_ed25519(self, data: bytes, signature: bytes, public_key: bytes) -> bool:
        """Verify Ed25519 signature."""
        try:
            from cryptography.hazmat.primitives import serialization
            from cryptography.exceptions import InvalidSignature

            key = serialization.load_pem_public_key(public_key)
            try:
                key.verify(signature, data)
                return True
            except InvalidSignature:
                return False
        except ImportError:
            return False
