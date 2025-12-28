"""Security-related exceptions for plugin system.

This module defines a hierarchy of exceptions for security operations:
- SandboxError: Errors during sandbox execution
- SignatureError: Errors during signing/verification
- CertificateError: Certificate management errors

All exceptions include contextual information for debugging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


class SecurityError(Exception):
    """Base exception for all plugin security errors.

    Attributes:
        message: Error message
        plugin_id: ID of the affected plugin (if applicable)
        details: Additional error details
    """

    def __init__(
        self,
        message: str,
        plugin_id: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        self.message = message
        self.plugin_id = plugin_id
        self.details = details or {}
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "plugin_id": self.plugin_id,
            "details": self.details,
        }


# =============================================================================
# Sandbox Exceptions
# =============================================================================


class SandboxError(SecurityError):
    """Base exception for sandbox-related errors.

    Raised when sandbox creation or execution fails.
    """

    def __init__(
        self,
        message: str,
        plugin_id: str | None = None,
        sandbox_id: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        self.sandbox_id = sandbox_id
        super().__init__(message, plugin_id, details)


class SandboxTimeoutError(SandboxError):
    """Raised when sandbox execution times out.

    Attributes:
        timeout_seconds: The timeout that was exceeded
        execution_time: Actual execution time before timeout
    """

    def __init__(
        self,
        message: str,
        plugin_id: str | None = None,
        sandbox_id: str | None = None,
        timeout_seconds: float = 0.0,
        execution_time: float = 0.0,
    ):
        details = {
            "timeout_seconds": timeout_seconds,
            "execution_time": execution_time,
        }
        super().__init__(message, plugin_id, sandbox_id, details)
        self.timeout_seconds = timeout_seconds
        self.execution_time = execution_time


class SandboxResourceError(SandboxError):
    """Raised when sandbox resource limits are exceeded.

    Attributes:
        resource_type: Type of resource exceeded (memory, cpu, etc.)
        limit: The limit that was exceeded
        actual: Actual value when limit was exceeded
    """

    def __init__(
        self,
        message: str,
        plugin_id: str | None = None,
        sandbox_id: str | None = None,
        resource_type: str = "",
        limit: float = 0.0,
        actual: float = 0.0,
    ):
        details = {
            "resource_type": resource_type,
            "limit": limit,
            "actual": actual,
        }
        super().__init__(message, plugin_id, sandbox_id, details)
        self.resource_type = resource_type
        self.limit = limit
        self.actual = actual


class SandboxSecurityViolation(SandboxError):
    """Raised when plugin code violates security policy.

    Attributes:
        violation_type: Type of violation (import, syscall, network, etc.)
        attempted_action: The action that was blocked
    """

    def __init__(
        self,
        message: str,
        plugin_id: str | None = None,
        sandbox_id: str | None = None,
        violation_type: str = "",
        attempted_action: str = "",
    ):
        details = {
            "violation_type": violation_type,
            "attempted_action": attempted_action,
        }
        super().__init__(message, plugin_id, sandbox_id, details)
        self.violation_type = violation_type
        self.attempted_action = attempted_action


# =============================================================================
# Signature Exceptions
# =============================================================================


class SignatureError(SecurityError):
    """Base exception for signature-related errors.

    Raised when signing or verification fails.
    """

    def __init__(
        self,
        message: str,
        plugin_id: str | None = None,
        signer_id: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        self.signer_id = signer_id
        super().__init__(message, plugin_id, details)


class SignatureExpiredError(SignatureError):
    """Raised when a signature has expired.

    Attributes:
        expired_at: When the signature expired
        signed_at: When the signature was created
    """

    def __init__(
        self,
        message: str,
        plugin_id: str | None = None,
        signer_id: str | None = None,
        expired_at: datetime | None = None,
        signed_at: datetime | None = None,
    ):
        details = {
            "expired_at": expired_at.isoformat() if expired_at else None,
            "signed_at": signed_at.isoformat() if signed_at else None,
        }
        super().__init__(message, plugin_id, signer_id, details)
        self.expired_at = expired_at
        self.signed_at = signed_at


class SignatureTamperError(SignatureError):
    """Raised when code tampering is detected.

    Attributes:
        expected_hash: Hash at signing time
        actual_hash: Current hash
    """

    def __init__(
        self,
        message: str,
        plugin_id: str | None = None,
        signer_id: str | None = None,
        expected_hash: str = "",
        actual_hash: str = "",
    ):
        details = {
            "expected_hash": expected_hash,
            "actual_hash": actual_hash,
        }
        super().__init__(message, plugin_id, signer_id, details)
        self.expected_hash = expected_hash
        self.actual_hash = actual_hash


class UntrustedSignerError(SignatureError):
    """Raised when signer is not trusted.

    Attributes:
        trust_level: Current trust level of the signer
        required_trust_level: Required trust level
    """

    def __init__(
        self,
        message: str,
        plugin_id: str | None = None,
        signer_id: str | None = None,
        trust_level: str = "unknown",
        required_trust_level: str = "trusted",
    ):
        details = {
            "trust_level": trust_level,
            "required_trust_level": required_trust_level,
        }
        super().__init__(message, plugin_id, signer_id, details)
        self.trust_level = trust_level
        self.required_trust_level = required_trust_level


class InvalidSignatureError(SignatureError):
    """Raised when signature is cryptographically invalid."""

    def __init__(
        self,
        message: str,
        plugin_id: str | None = None,
        signer_id: str | None = None,
        algorithm: str = "",
    ):
        details = {"algorithm": algorithm}
        super().__init__(message, plugin_id, signer_id, details)
        self.algorithm = algorithm


# =============================================================================
# Certificate Exceptions
# =============================================================================


class CertificateError(SecurityError):
    """Base exception for certificate-related errors.

    Raised when certificate operations fail.
    """

    def __init__(
        self,
        message: str,
        cert_id: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        self.cert_id = cert_id
        super().__init__(message, details=details)


class CertificateExpiredError(CertificateError):
    """Raised when a certificate has expired."""

    def __init__(
        self,
        message: str,
        cert_id: str | None = None,
        expired_at: datetime | None = None,
    ):
        details = {"expired_at": expired_at.isoformat() if expired_at else None}
        super().__init__(message, cert_id, details)
        self.expired_at = expired_at


class CertificateRevokedError(CertificateError):
    """Raised when a certificate has been revoked."""

    def __init__(
        self,
        message: str,
        cert_id: str | None = None,
        revoked_at: datetime | None = None,
        reason: str = "",
    ):
        details = {
            "revoked_at": revoked_at.isoformat() if revoked_at else None,
            "reason": reason,
        }
        super().__init__(message, cert_id, details)
        self.revoked_at = revoked_at
        self.reason = reason


class CertificateNotFoundError(CertificateError):
    """Raised when a certificate is not found in trust store."""

    pass


class InvalidCertificateError(CertificateError):
    """Raised when a certificate is malformed or invalid."""

    def __init__(
        self,
        message: str,
        cert_id: str | None = None,
        parse_error: str = "",
    ):
        details = {"parse_error": parse_error}
        super().__init__(message, cert_id, details)
        self.parse_error = parse_error
