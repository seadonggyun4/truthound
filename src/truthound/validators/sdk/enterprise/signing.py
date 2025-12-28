"""Code signing and verification for validators.

This module provides cryptographic signing and verification:
- GPG/PGP signing
- SHA256/SHA512 checksums
- Certificate-based signing
- Signature verification and trust chains

Example:
    from truthound.validators.sdk.enterprise.signing import (
        SignatureManager,
        SignatureConfig,
        sign_validator,
        verify_validator,
    )

    # Sign a validator
    manager = SignatureManager(config)
    signature = manager.sign_validator(MyValidator)

    # Verify a validator
    is_valid = manager.verify_validator(MyValidator, signature)
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import inspect
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class SignatureAlgorithm(Enum):
    """Supported signature algorithms."""

    SHA256 = auto()      # Simple SHA256 hash
    SHA512 = auto()      # SHA512 hash
    HMAC_SHA256 = auto() # HMAC with SHA256
    HMAC_SHA512 = auto() # HMAC with SHA512
    RSA_SHA256 = auto()  # RSA with SHA256 (requires cryptography)
    ED25519 = auto()     # Ed25519 (requires cryptography)


class SignatureVerificationError(Exception):
    """Raised when signature verification fails."""

    def __init__(
        self,
        message: str,
        validator_name: str = "",
        reason: str = "",
    ):
        self.validator_name = validator_name
        self.reason = reason
        super().__init__(message)


class SignatureExpiredError(SignatureVerificationError):
    """Raised when signature has expired."""
    pass


class SignatureTamperError(SignatureVerificationError):
    """Raised when code tampering is detected."""
    pass


@dataclass(frozen=True)
class SignatureConfig:
    """Configuration for signature operations.

    Attributes:
        algorithm: Signature algorithm to use
        secret_key: Secret key for HMAC algorithms
        private_key_path: Path to private key for asymmetric algorithms
        public_key_path: Path to public key for verification
        validity_days: How long signatures are valid
        require_timestamp: Whether to include timestamp
        trusted_signers: List of trusted signer identifiers
        revocation_list_url: URL to check for revoked signatures
    """

    algorithm: SignatureAlgorithm = SignatureAlgorithm.SHA256
    secret_key: str = ""
    private_key_path: Path | None = None
    public_key_path: Path | None = None
    validity_days: int = 365
    require_timestamp: bool = True
    trusted_signers: tuple[str, ...] = field(default_factory=tuple)
    revocation_list_url: str = ""

    @classmethod
    def development(cls) -> "SignatureConfig":
        """Create development configuration (weak security)."""
        return cls(
            algorithm=SignatureAlgorithm.SHA256,
            validity_days=30,
            require_timestamp=False,
        )

    @classmethod
    def production(cls, secret_key: str) -> "SignatureConfig":
        """Create production configuration."""
        return cls(
            algorithm=SignatureAlgorithm.HMAC_SHA256,
            secret_key=secret_key,
            validity_days=365,
            require_timestamp=True,
        )


@dataclass
class ValidatorSignature:
    """Signature for a validator.

    Attributes:
        validator_name: Name of the signed validator
        validator_version: Version of the validator
        code_hash: Hash of the validator source code
        signature: The actual signature bytes (base64 encoded)
        algorithm: Algorithm used for signing
        signer_id: Identifier of the signer
        signed_at: When the signature was created
        expires_at: When the signature expires
        metadata: Additional metadata
    """

    validator_name: str
    validator_version: str
    code_hash: str
    signature: str
    algorithm: SignatureAlgorithm
    signer_id: str = ""
    signed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if signature has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "validator_name": self.validator_name,
            "validator_version": self.validator_version,
            "code_hash": self.code_hash,
            "signature": self.signature,
            "algorithm": self.algorithm.name,
            "signer_id": self.signer_id,
            "signed_at": self.signed_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ValidatorSignature":
        """Create from dictionary."""
        return cls(
            validator_name=data["validator_name"],
            validator_version=data["validator_version"],
            code_hash=data["code_hash"],
            signature=data["signature"],
            algorithm=SignatureAlgorithm[data["algorithm"]],
            signer_id=data.get("signer_id", ""),
            signed_at=datetime.fromisoformat(data["signed_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"])
                if data.get("expires_at") else None,
            metadata=data.get("metadata", {}),
        )

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "ValidatorSignature":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


class SignatureProvider(ABC):
    """Abstract base class for signature providers."""

    @abstractmethod
    def sign(self, data: bytes) -> str:
        """Sign data and return signature."""
        pass

    @abstractmethod
    def verify(self, data: bytes, signature: str) -> bool:
        """Verify signature against data."""
        pass


class SHA256Provider(SignatureProvider):
    """Simple SHA256 hash provider (not cryptographically secure for signing)."""

    def sign(self, data: bytes) -> str:
        """Create SHA256 hash of data."""
        return hashlib.sha256(data).hexdigest()

    def verify(self, data: bytes, signature: str) -> bool:
        """Verify SHA256 hash matches."""
        return self.sign(data) == signature


class SHA512Provider(SignatureProvider):
    """SHA512 hash provider."""

    def sign(self, data: bytes) -> str:
        """Create SHA512 hash of data."""
        return hashlib.sha512(data).hexdigest()

    def verify(self, data: bytes, signature: str) -> bool:
        """Verify SHA512 hash matches."""
        return self.sign(data) == signature


class HMACProvider(SignatureProvider):
    """HMAC signature provider."""

    def __init__(self, secret_key: str, hash_algorithm: str = "sha256"):
        """Initialize HMAC provider.

        Args:
            secret_key: Secret key for HMAC
            hash_algorithm: Hash algorithm (sha256 or sha512)
        """
        if not secret_key:
            raise ValueError("Secret key is required for HMAC signing")
        self.secret_key = secret_key.encode()
        self.hash_algorithm = hash_algorithm

    def sign(self, data: bytes) -> str:
        """Create HMAC signature."""
        signature = hmac.new(
            self.secret_key,
            data,
            self.hash_algorithm,
        )
        return base64.b64encode(signature.digest()).decode()

    def verify(self, data: bytes, signature: str) -> bool:
        """Verify HMAC signature."""
        expected = self.sign(data)
        return hmac.compare_digest(expected, signature)


class RSAProvider(SignatureProvider):
    """RSA signature provider (requires cryptography package)."""

    def __init__(
        self,
        private_key_path: Path | None = None,
        public_key_path: Path | None = None,
    ):
        """Initialize RSA provider.

        Args:
            private_key_path: Path to PEM private key (for signing)
            public_key_path: Path to PEM public key (for verification)
        """
        self._private_key = None
        self._public_key = None

        try:
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.backends import default_backend

            if private_key_path and private_key_path.exists():
                with open(private_key_path, "rb") as f:
                    self._private_key = serialization.load_pem_private_key(
                        f.read(),
                        password=None,
                        backend=default_backend(),
                    )

            if public_key_path and public_key_path.exists():
                with open(public_key_path, "rb") as f:
                    self._public_key = serialization.load_pem_public_key(
                        f.read(),
                        backend=default_backend(),
                    )
        except ImportError:
            raise ImportError(
                "cryptography package is required for RSA signing. "
                "Install with: pip install cryptography"
            )

    def sign(self, data: bytes) -> str:
        """Create RSA signature."""
        if self._private_key is None:
            raise ValueError("Private key is required for signing")

        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding

        signature = self._private_key.sign(
            data,
            padding.PKCS1v15(),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode()

    def verify(self, data: bytes, signature: str) -> bool:
        """Verify RSA signature."""
        if self._public_key is None:
            raise ValueError("Public key is required for verification")

        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding
        from cryptography.exceptions import InvalidSignature

        try:
            signature_bytes = base64.b64decode(signature)
            self._public_key.verify(
                signature_bytes,
                data,
                padding.PKCS1v15(),
                hashes.SHA256(),
            )
            return True
        except InvalidSignature:
            return False


class SignatureManager:
    """Manages validator signing and verification.

    Provides a high-level API for signing validators and
    verifying their signatures.
    """

    def __init__(self, config: SignatureConfig):
        """Initialize signature manager.

        Args:
            config: Signature configuration
        """
        self.config = config
        self._provider = self._create_provider()
        self._signature_cache: dict[str, ValidatorSignature] = {}

    def _create_provider(self) -> SignatureProvider:
        """Create appropriate signature provider."""
        algorithm = self.config.algorithm

        if algorithm == SignatureAlgorithm.SHA256:
            return SHA256Provider()
        elif algorithm == SignatureAlgorithm.SHA512:
            return SHA512Provider()
        elif algorithm == SignatureAlgorithm.HMAC_SHA256:
            return HMACProvider(self.config.secret_key, "sha256")
        elif algorithm == SignatureAlgorithm.HMAC_SHA512:
            return HMACProvider(self.config.secret_key, "sha512")
        elif algorithm == SignatureAlgorithm.RSA_SHA256:
            return RSAProvider(
                self.config.private_key_path,
                self.config.public_key_path,
            )
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

    def _get_validator_source(self, validator_class: type) -> str:
        """Get source code of validator class."""
        try:
            return inspect.getsource(validator_class)
        except (OSError, TypeError):
            # Fallback to inspecting methods
            source_parts = []
            for name, method in inspect.getmembers(validator_class):
                if not name.startswith("_") or name in ("__init__", "__call__"):
                    try:
                        source_parts.append(inspect.getsource(method))
                    except (OSError, TypeError):
                        pass
            return "\n".join(source_parts)

    def _compute_code_hash(self, validator_class: type) -> str:
        """Compute hash of validator source code."""
        source = self._get_validator_source(validator_class)
        # Also include class attributes
        attrs = {
            k: str(v) for k, v in vars(validator_class).items()
            if not k.startswith("_") and not callable(v)
        }
        content = source + json.dumps(attrs, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def sign_validator(
        self,
        validator_class: type,
        signer_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> ValidatorSignature:
        """Sign a validator class.

        Args:
            validator_class: Validator class to sign
            signer_id: Identifier of the signer
            metadata: Additional metadata to include

        Returns:
            ValidatorSignature containing the signature
        """
        validator_name = getattr(validator_class, "name", validator_class.__name__)
        validator_version = getattr(validator_class, "version", "1.0.0")

        # Compute code hash
        code_hash = self._compute_code_hash(validator_class)

        # Build data to sign
        signed_at = datetime.now(timezone.utc)
        expires_at = signed_at + timedelta(days=self.config.validity_days)

        sign_data = {
            "validator_name": validator_name,
            "validator_version": validator_version,
            "code_hash": code_hash,
            "signer_id": signer_id,
            "signed_at": signed_at.isoformat(),
            "expires_at": expires_at.isoformat(),
        }

        # Create signature
        data_bytes = json.dumps(sign_data, sort_keys=True).encode()
        signature = self._provider.sign(data_bytes)

        return ValidatorSignature(
            validator_name=validator_name,
            validator_version=validator_version,
            code_hash=code_hash,
            signature=signature,
            algorithm=self.config.algorithm,
            signer_id=signer_id,
            signed_at=signed_at,
            expires_at=expires_at,
            metadata=metadata or {},
        )

    def verify_validator(
        self,
        validator_class: type,
        signature: ValidatorSignature,
        check_expiry: bool = True,
        check_signer: bool = True,
    ) -> bool:
        """Verify a validator's signature.

        Args:
            validator_class: Validator class to verify
            signature: Signature to verify against
            check_expiry: Whether to check signature expiry
            check_signer: Whether to verify signer is trusted

        Returns:
            True if signature is valid

        Raises:
            SignatureVerificationError: If verification fails
        """
        validator_name = getattr(validator_class, "name", validator_class.__name__)

        # Check expiry
        if check_expiry and signature.is_expired():
            raise SignatureExpiredError(
                f"Signature for '{validator_name}' has expired",
                validator_name=validator_name,
                reason="expired",
            )

        # Check signer
        if check_signer and self.config.trusted_signers:
            if signature.signer_id not in self.config.trusted_signers:
                raise SignatureVerificationError(
                    f"Signer '{signature.signer_id}' is not trusted",
                    validator_name=validator_name,
                    reason="untrusted_signer",
                )

        # Verify code hash
        current_hash = self._compute_code_hash(validator_class)
        if current_hash != signature.code_hash:
            raise SignatureTamperError(
                f"Code tampering detected for '{validator_name}'",
                validator_name=validator_name,
                reason="code_modified",
            )

        # Verify signature
        sign_data = {
            "validator_name": signature.validator_name,
            "validator_version": signature.validator_version,
            "code_hash": signature.code_hash,
            "signer_id": signature.signer_id,
            "signed_at": signature.signed_at.isoformat(),
            "expires_at": signature.expires_at.isoformat() if signature.expires_at else None,
        }
        data_bytes = json.dumps(sign_data, sort_keys=True).encode()

        if not self._provider.verify(data_bytes, signature.signature):
            raise SignatureVerificationError(
                f"Invalid signature for '{validator_name}'",
                validator_name=validator_name,
                reason="invalid_signature",
            )

        return True

    def save_signature(
        self,
        signature: ValidatorSignature,
        path: Path,
    ) -> None:
        """Save signature to file.

        Args:
            signature: Signature to save
            path: Path to save to
        """
        with open(path, "w") as f:
            f.write(signature.to_json())

    def load_signature(self, path: Path) -> ValidatorSignature:
        """Load signature from file.

        Args:
            path: Path to load from

        Returns:
            Loaded signature
        """
        with open(path, "r") as f:
            return ValidatorSignature.from_json(f.read())


# Convenience functions

def sign_validator(
    validator_class: type,
    secret_key: str = "",
    algorithm: SignatureAlgorithm = SignatureAlgorithm.SHA256,
    signer_id: str = "",
) -> ValidatorSignature:
    """Sign a validator with default configuration.

    Args:
        validator_class: Validator class to sign
        secret_key: Secret key for HMAC (optional)
        algorithm: Signature algorithm
        signer_id: Signer identifier

    Returns:
        ValidatorSignature
    """
    config = SignatureConfig(
        algorithm=algorithm,
        secret_key=secret_key,
    )
    manager = SignatureManager(config)
    return manager.sign_validator(validator_class, signer_id)


def verify_validator(
    validator_class: type,
    signature: ValidatorSignature,
    secret_key: str = "",
) -> bool:
    """Verify a validator signature.

    Args:
        validator_class: Validator class to verify
        signature: Signature to verify
        secret_key: Secret key for HMAC (if applicable)

    Returns:
        True if valid

    Raises:
        SignatureVerificationError: If verification fails
    """
    config = SignatureConfig(
        algorithm=signature.algorithm,
        secret_key=secret_key,
    )
    manager = SignatureManager(config)
    return manager.verify_validator(validator_class, signature)
