"""Trust store for managing trusted certificates and signers.

This module provides a TrustStore implementation for managing
which signers and certificates are trusted.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import threading
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from typing import Any

from truthound.plugins.security.exceptions import (
    CertificateRevokedError,
)
from truthound.plugins.security.protocols import TrustLevel
from truthound.stores._filesystem_io import publish

logger = logging.getLogger(__name__)


def _mutation(method):
    @wraps(method)
    def invoke(self, *args, **kwargs):
        with self._lock:
            certificates, signers = deepcopy(self._certificates), dict(self._signer_trust)
            try:
                return method(self, *args, **kwargs)
            except BaseException:
                self._certificates, self._signer_trust = certificates, signers
                raise

    return invoke


@dataclass
class CertificateEntry:
    """Entry in the trust store."""

    cert_id: str
    certificate: bytes
    trust_level: TrustLevel
    added_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime | None = None
    revoked_at: datetime | None = None
    revocation_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_revoked(self) -> bool:
        """Check if certificate is revoked."""
        return self.revoked_at is not None

    @property
    def is_expired(self) -> bool:
        """Check if certificate is expired."""
        if self.expires_at is None:
            return False
        return datetime.now(UTC) > self.expires_at

    @property
    def is_valid(self) -> bool:
        """Check if certificate is valid (not revoked and not expired)."""
        return not self.is_revoked and not self.is_expired

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cert_id": self.cert_id,
            "trust_level": self.trust_level.value,
            "added_at": self.added_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "revoked_at": self.revoked_at.isoformat() if self.revoked_at else None,
            "revocation_reason": self.revocation_reason,
            "metadata": self.metadata,
        }


class TrustStoreImpl:
    """Implementation of trust store for certificates and signers.

    Manages a collection of trusted certificates with different
    trust levels. Supports persistence to JSON file.

    Example:
        >>> store = TrustStoreImpl()
        >>> cert_id = store.add_trusted_certificate(cert_bytes)
        >>> is_trusted, level = store.is_trusted(cert_bytes)
    """

    def __init__(
        self,
        store_path: Path | None = None,
        auto_save: bool = True,
    ) -> None:
        """Initialize trust store.

        Args:
            store_path: Path to persist store (optional)
            auto_save: Whether to auto-save after modifications
        """
        self._lock = threading.RLock()
        self._store_path = Path(store_path) if store_path is not None else None
        self._auto_save = auto_save
        self._certificates: dict[str, CertificateEntry] = {}
        self._signer_trust: dict[str, TrustLevel] = {}

        # Load from file if exists
        if store_path and store_path.exists():
            self._load()

    @_mutation
    def add_trusted_certificate(
        self,
        certificate: bytes,
        trust_level: TrustLevel = TrustLevel.TRUSTED,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Add a certificate to the trust store.

        Args:
            certificate: Certificate bytes (DER or PEM)
            trust_level: Trust level to assign
            metadata: Additional metadata

        Returns:
            Certificate ID
        """
        if not isinstance(certificate, bytes) or not certificate:
            raise ValueError("Certificate must contain bytes")
        trust_level = TrustLevel(trust_level)
        metadata = deepcopy(metadata or {})
        # Generate certificate ID from hash
        cert_id = self._compute_cert_id(certificate)

        # Check if already exists
        if cert_id in self._certificates:
            existing = self._certificates[cert_id]
            if existing.is_revoked:
                raise CertificateRevokedError(
                    f"Certificate {cert_id} was revoked",
                    cert_id=cert_id,
                    revoked_at=existing.revoked_at,
                    reason=existing.revocation_reason,
                )
            # Update trust level if higher
            if trust_level == TrustLevel.TRUSTED:
                existing = CertificateEntry(
                    cert_id=existing.cert_id,
                    certificate=existing.certificate,
                    trust_level=trust_level,
                    added_at=existing.added_at,
                    expires_at=existing.expires_at,
                    metadata={**existing.metadata, **(metadata or {})},
                )
                self._certificates[cert_id] = existing
                if self._auto_save:
                    self._save()
            return cert_id

        # Create new entry
        entry = CertificateEntry(
            cert_id=cert_id,
            certificate=certificate,
            trust_level=trust_level,
            metadata=metadata or {},
        )
        self._certificates[cert_id] = entry

        # Extract signer ID from certificate if possible
        signer_id = self._extract_signer_id(certificate)
        if signer_id:
            self._signer_trust[signer_id] = trust_level
            entry.metadata["signer_id"] = signer_id

        logger.info(f"Added certificate {cert_id} with trust level {trust_level.value}")

        if self._auto_save:
            self._save()

        return cert_id

    @_mutation
    def remove_certificate(self, cert_id: str) -> bool:
        """Remove a certificate from trust store.

        Args:
            cert_id: Certificate ID

        Returns:
            True if removed, False if not found
        """
        if cert_id not in self._certificates:
            return False

        entry = self._certificates.pop(cert_id)

        # Also remove signer trust if applicable
        signer_id = entry.metadata.get("signer_id")
        if signer_id:
            self._signer_trust.pop(signer_id, None)

        logger.info(f"Removed certificate {cert_id}")

        if self._auto_save:
            self._save()

        return True

    @_mutation
    def revoke_certificate(
        self,
        cert_id: str,
        reason: str = "",
    ) -> bool:
        """Revoke a certificate.

        Args:
            cert_id: Certificate ID
            reason: Revocation reason

        Returns:
            True if revoked, False if not found
        """
        if cert_id not in self._certificates:
            return False

        entry = self._certificates[cert_id]
        self._certificates[cert_id] = CertificateEntry(
            cert_id=entry.cert_id,
            certificate=entry.certificate,
            trust_level=TrustLevel.REVOKED,
            added_at=entry.added_at,
            expires_at=entry.expires_at,
            revoked_at=datetime.now(UTC),
            revocation_reason=reason,
            metadata=entry.metadata,
        )

        # Also revoke signer trust
        signer_id = entry.metadata.get("signer_id")
        if signer_id:
            self._signer_trust[signer_id] = TrustLevel.REVOKED

        logger.info(f"Revoked certificate {cert_id}: {reason}")

        if self._auto_save:
            self._save()

        return True

    def is_trusted(self, certificate: bytes) -> tuple[bool, TrustLevel]:
        """Check if certificate is trusted.

        Args:
            certificate: Certificate to check

        Returns:
            Tuple of (is_trusted, trust_level)
        """
        cert_id = self._compute_cert_id(certificate)

        if cert_id not in self._certificates:
            return False, TrustLevel.UNKNOWN

        entry = self._certificates[cert_id]

        if entry.is_revoked:
            return False, TrustLevel.REVOKED

        if entry.is_expired:
            return False, TrustLevel.UNKNOWN

        is_trusted = entry.trust_level in (TrustLevel.TRUSTED, TrustLevel.VERIFIED)
        return is_trusted, entry.trust_level

    def get_trust_level(self, signer_id: str) -> TrustLevel:
        """Get trust level for a signer.

        Args:
            signer_id: Signer identifier

        Returns:
            Trust level of the signer
        """
        return self._signer_trust.get(signer_id, TrustLevel.UNKNOWN)

    @_mutation
    def set_signer_trust(
        self,
        signer_id: str,
        trust_level: TrustLevel,
    ) -> None:
        """Set trust level for a signer.

        Args:
            signer_id: Signer identifier
            trust_level: Trust level to set
        """
        if not isinstance(signer_id, str) or not signer_id:
            raise ValueError("Signer ID must be nonempty text")
        self._signer_trust[signer_id] = TrustLevel(trust_level)

        if self._auto_save:
            self._save()

    def list_certificates(self) -> list[dict[str, Any]]:
        """List all certificates in store.

        Returns:
            List of certificate info dicts
        """
        return deepcopy([entry.to_dict() for entry in self._certificates.values()])

    def list_trusted_signers(self) -> list[str]:
        """List all trusted signer IDs.

        Returns:
            List of trusted signer IDs
        """
        return [
            signer_id
            for signer_id, level in self._signer_trust.items()
            if level == TrustLevel.TRUSTED
        ]

    def _compute_cert_id(self, certificate: bytes) -> str:
        """Compute certificate ID from content hash."""
        return hashlib.sha256(certificate).hexdigest()[:16]

    def _extract_signer_id(self, certificate: bytes) -> str | None:
        """Try to extract signer ID from certificate."""
        try:
            from cryptography import x509

            cert = x509.load_pem_x509_certificate(certificate)
            # Use subject common name as signer ID
            for attr in cert.subject:
                if attr.oid == x509.oid.NameOID.COMMON_NAME:
                    return attr.value
        except Exception:
            pass
        return None

    def _save(self) -> None:
        """Save trust store to file."""
        if not self._store_path:
            return

        data = {
            "format_version": 2,
            "certificates": {
                cert_id: {
                    **entry.to_dict(),
                    "certificate": base64.b64encode(entry.certificate).decode("ascii"),
                }
                for cert_id, entry in self._certificates.items()
            },
            "signer_trust": {
                signer_id: level.value for signer_id, level in self._signer_trust.items()
            },
        }

        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        publish(self._store_path, json.dumps(data, indent=2).encode("utf-8"))

    def _load(self) -> None:
        """Load trust store from file."""
        if not self._store_path or not self._store_path.exists():
            return

        with self._store_path.open("rb") as stream:
            content = stream.read(8 * 1024 * 1024 + 1)
        if len(content) > 8 * 1024 * 1024:
            raise ValueError("Trust store exceeds 8 MiB")
        data = json.loads(content)
        if not isinstance(data, dict) or data.get("format_version", 1) not in (1, 2):
            raise ValueError("Unsupported trust store format")
        signers = {}
        for name, level in data.get("signer_trust", {}).items():
            if not isinstance(name, str) or not name:
                raise ValueError("Invalid signer identity")
            signers[name] = TrustLevel(level)
        certificates = {}
        for cert_id, item in data.get("certificates", {}).items():
            if "certificate" not in item:
                raise ValueError(
                    "Legacy certificate metadata must be reimported with certificate bytes"
                )
            certificate = base64.b64decode(item["certificate"], validate=True)
            if (
                not certificate
                or self._compute_cert_id(certificate) != cert_id
                or item["cert_id"] != cert_id
            ):
                raise ValueError("Certificate digest mismatch")
            dates = {}
            for key in ("added_at", "expires_at", "revoked_at"):
                value = item.get(key)
                dates[key] = datetime.fromisoformat(value) if value is not None else None
                if dates[key] is not None and dates[key].utcoffset() is None:
                    raise ValueError("Certificate timestamps must have a timezone")
            if dates["added_at"] is None or not isinstance(item.get("metadata", {}), dict):
                raise ValueError("Invalid certificate metadata")
            certificates[cert_id] = CertificateEntry(
                cert_id=cert_id,
                certificate=certificate,
                trust_level=TrustLevel(item["trust_level"]),
                revocation_reason=item.get("revocation_reason", ""),
                metadata=deepcopy(item.get("metadata", {})),
                **dates,
            )
        self._certificates, self._signer_trust = certificates, signers

    @_mutation
    def clear(self) -> None:
        """Clear all certificates and trust levels."""
        self._certificates.clear()
        self._signer_trust.clear()

        if self._auto_save and self._store_path:
            self._save()
