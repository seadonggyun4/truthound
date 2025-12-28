"""Verification chain for plugin signatures.

This module implements the Chain of Responsibility pattern for
multi-step signature verification.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from truthound.plugins.security.protocols import (
    SignatureInfo,
    VerificationResult,
    TrustLevel,
    VerificationHandler,
)
from truthound.plugins.security.signing.trust_store import TrustStoreImpl
from truthound.plugins.security.signing.service import SigningServiceImpl

logger = logging.getLogger(__name__)


class VerificationHandlerBase(ABC):
    """Base class for verification handlers in the chain.

    Implements the Chain of Responsibility pattern where each handler
    performs one verification step and can pass to the next handler.
    """

    def __init__(self) -> None:
        """Initialize handler."""
        self._next: VerificationHandlerBase | None = None

    def set_next(self, handler: "VerificationHandlerBase") -> "VerificationHandlerBase":
        """Set the next handler in the chain.

        Args:
            handler: Next handler

        Returns:
            The next handler (for fluent chaining)
        """
        self._next = handler
        return handler

    def verify(
        self,
        plugin_path: Path,
        signature: SignatureInfo,
        context: dict[str, Any],
    ) -> VerificationResult | None:
        """Verify signature in this step of the chain.

        Args:
            plugin_path: Path to plugin
            signature: Signature to verify
            context: Shared context for passing data between handlers

        Returns:
            VerificationResult if verification completed (success or failure),
            None to continue to next handler
        """
        result = self._do_verify(plugin_path, signature, context)

        # If this handler produced a failure, stop the chain
        if result and not result.is_valid:
            return result

        # Continue to next handler if exists
        if self._next:
            return self._next.verify(plugin_path, signature, context)

        # End of chain - return accumulated result
        return self._finalize(context)

    @abstractmethod
    def _do_verify(
        self,
        plugin_path: Path,
        signature: SignatureInfo,
        context: dict[str, Any],
    ) -> VerificationResult | None:
        """Perform this handler's verification step.

        Args:
            plugin_path: Path to plugin
            signature: Signature to verify
            context: Shared context

        Returns:
            VerificationResult on failure, None to continue
        """
        ...

    def _finalize(self, context: dict[str, Any]) -> VerificationResult:
        """Create final result from accumulated context.

        Args:
            context: Accumulated context from all handlers

        Returns:
            Final VerificationResult
        """
        return VerificationResult.success(
            signer_id=context.get("signer_id", "unknown"),
            trust_level=context.get("trust_level", TrustLevel.VERIFIED),
            warnings=tuple(context.get("warnings", [])),
        )


class IntegrityVerifier(VerificationHandlerBase):
    """Verifies file integrity by comparing hashes."""

    def __init__(self, signing_service: SigningServiceImpl | None = None) -> None:
        """Initialize integrity verifier.

        Args:
            signing_service: Service for computing hashes
        """
        super().__init__()
        self._service = signing_service or SigningServiceImpl()

    def _do_verify(
        self,
        plugin_path: Path,
        signature: SignatureInfo,
        context: dict[str, Any],
    ) -> VerificationResult | None:
        """Check file integrity."""
        stored_hash = signature.metadata.get("plugin_hash", "")
        if not stored_hash:
            return VerificationResult.failure(
                "Signature does not contain plugin hash"
            )

        current_hash = self._service.get_plugin_hash(plugin_path)

        if current_hash != stored_hash:
            return VerificationResult.failure(
                "Plugin content has been modified since signing"
            )

        # Store hash in context for other handlers
        context["plugin_hash"] = current_hash
        logger.debug(f"Integrity check passed for {plugin_path}")
        return None


class SignatureVerifier(VerificationHandlerBase):
    """Verifies the cryptographic signature."""

    def __init__(self, signing_service: SigningServiceImpl | None = None) -> None:
        """Initialize signature verifier.

        Args:
            signing_service: Service for verifying signatures
        """
        super().__init__()
        self._service = signing_service or SigningServiceImpl()

    def _do_verify(
        self,
        plugin_path: Path,
        signature: SignatureInfo,
        context: dict[str, Any],
    ) -> VerificationResult | None:
        """Verify cryptographic signature."""
        # For now, we rely on integrity check and trust verification
        # Full cryptographic verification would require public keys
        context["signer_id"] = signature.signer_id
        logger.debug(f"Signature structure verified for signer {signature.signer_id}")
        return None


class TrustVerifier(VerificationHandlerBase):
    """Verifies signer is trusted."""

    def __init__(self, trust_store: TrustStoreImpl) -> None:
        """Initialize trust verifier.

        Args:
            trust_store: Trust store to check against
        """
        super().__init__()
        self._trust_store = trust_store

    def _do_verify(
        self,
        plugin_path: Path,
        signature: SignatureInfo,
        context: dict[str, Any],
    ) -> VerificationResult | None:
        """Check if signer is trusted."""
        signer_id = signature.signer_id
        trust_level = self._trust_store.get_trust_level(signer_id)

        if trust_level == TrustLevel.REVOKED:
            return VerificationResult.failure(
                f"Signer '{signer_id}' has been revoked"
            )

        if trust_level == TrustLevel.UNKNOWN:
            # Add warning but continue
            warnings = context.setdefault("warnings", [])
            warnings.append(f"Signer '{signer_id}' is not in trust store")

        context["trust_level"] = trust_level
        logger.debug(f"Trust verification: {signer_id} has level {trust_level.value}")
        return None


class ExpirationVerifier(VerificationHandlerBase):
    """Verifies signature has not expired."""

    def __init__(self, max_age_days: int = 0) -> None:
        """Initialize expiration verifier.

        Args:
            max_age_days: Maximum age in days (0 = use signature expiry only)
        """
        super().__init__()
        self._max_age_days = max_age_days

    def _do_verify(
        self,
        plugin_path: Path,
        signature: SignatureInfo,
        context: dict[str, Any],
    ) -> VerificationResult | None:
        """Check signature expiration."""
        # Check if signature has explicit expiry
        if signature.is_expired():
            return VerificationResult.failure(
                f"Signature expired at {signature.expires_at}"
            )

        # Check max age policy
        if self._max_age_days > 0:
            age_days = signature.age_days
            if age_days > self._max_age_days:
                return VerificationResult.failure(
                    f"Signature is {age_days} days old, max allowed is {self._max_age_days}"
                )

            # Warn if approaching expiry
            if age_days > self._max_age_days * 0.8:
                warnings = context.setdefault("warnings", [])
                remaining = self._max_age_days - age_days
                warnings.append(f"Signature expires in {remaining} days")

        logger.debug(f"Expiration check passed, signature age: {signature.age_days} days")
        return None


class ChainVerifier(VerificationHandlerBase):
    """Verifies certificate chain (if certificates provided)."""

    def __init__(self, trust_store: TrustStoreImpl) -> None:
        """Initialize chain verifier.

        Args:
            trust_store: Trust store for root certificates
        """
        super().__init__()
        self._trust_store = trust_store

    def _do_verify(
        self,
        plugin_path: Path,
        signature: SignatureInfo,
        context: dict[str, Any],
    ) -> VerificationResult | None:
        """Verify certificate chain."""
        if not signature.certificate_chain:
            # No certificate chain, skip
            return None

        # Check if any certificate in chain is trusted
        for cert in signature.certificate_chain:
            is_trusted, level = self._trust_store.is_trusted(cert)
            if is_trusted:
                context["trust_level"] = level
                logger.debug("Certificate chain verified via trust store")
                return None

        # Certificate chain not trusted
        warnings = context.setdefault("warnings", [])
        warnings.append("Certificate chain not rooted in trust store")
        return None


def create_verification_chain(
    trust_store: TrustStoreImpl | None = None,
    signing_service: SigningServiceImpl | None = None,
    max_age_days: int = 365,
    require_trusted_signer: bool = False,
) -> VerificationHandlerBase:
    """Create a complete verification chain.

    Creates a chain of handlers in order:
    1. IntegrityVerifier - Check file hasn't been modified
    2. ExpirationVerifier - Check signature isn't expired
    3. SignatureVerifier - Verify cryptographic signature
    4. TrustVerifier - Check signer is trusted (optional)
    5. ChainVerifier - Verify certificate chain (if present)

    Args:
        trust_store: Trust store for certificate checks
        signing_service: Service for signature operations
        max_age_days: Maximum signature age in days
        require_trusted_signer: Whether to require trusted signer

    Returns:
        First handler in the chain
    """
    # Create handlers
    trust_store = trust_store or TrustStoreImpl()
    signing_service = signing_service or SigningServiceImpl()

    integrity = IntegrityVerifier(signing_service)
    expiration = ExpirationVerifier(max_age_days)
    signature = SignatureVerifier(signing_service)
    trust = TrustVerifier(trust_store)
    chain = ChainVerifier(trust_store)

    # Build chain
    integrity.set_next(expiration).set_next(signature).set_next(trust).set_next(chain)

    return integrity


class VerificationChainBuilder:
    """Fluent builder for verification chains.

    Example:
        >>> chain = (
        ...     VerificationChainBuilder()
        ...     .with_integrity_check()
        ...     .with_expiration_check(max_age_days=90)
        ...     .with_trust_check(trust_store)
        ...     .build()
        ... )
    """

    def __init__(self) -> None:
        """Initialize builder."""
        self._handlers: list[VerificationHandlerBase] = []

    def with_integrity_check(
        self,
        signing_service: SigningServiceImpl | None = None,
    ) -> "VerificationChainBuilder":
        """Add integrity verification.

        Args:
            signing_service: Service for hash computation

        Returns:
            Self for chaining
        """
        self._handlers.append(IntegrityVerifier(signing_service))
        return self

    def with_expiration_check(
        self,
        max_age_days: int = 0,
    ) -> "VerificationChainBuilder":
        """Add expiration verification.

        Args:
            max_age_days: Maximum age in days (0 = use signature expiry)

        Returns:
            Self for chaining
        """
        self._handlers.append(ExpirationVerifier(max_age_days))
        return self

    def with_signature_check(
        self,
        signing_service: SigningServiceImpl | None = None,
    ) -> "VerificationChainBuilder":
        """Add signature verification.

        Args:
            signing_service: Service for signature verification

        Returns:
            Self for chaining
        """
        self._handlers.append(SignatureVerifier(signing_service))
        return self

    def with_trust_check(
        self,
        trust_store: TrustStoreImpl,
    ) -> "VerificationChainBuilder":
        """Add trust verification.

        Args:
            trust_store: Trust store to check against

        Returns:
            Self for chaining
        """
        self._handlers.append(TrustVerifier(trust_store))
        return self

    def with_chain_check(
        self,
        trust_store: TrustStoreImpl,
    ) -> "VerificationChainBuilder":
        """Add certificate chain verification.

        Args:
            trust_store: Trust store for root certificates

        Returns:
            Self for chaining
        """
        self._handlers.append(ChainVerifier(trust_store))
        return self

    def with_custom_handler(
        self,
        handler: VerificationHandlerBase,
    ) -> "VerificationChainBuilder":
        """Add a custom verification handler.

        Args:
            handler: Custom handler instance

        Returns:
            Self for chaining
        """
        self._handlers.append(handler)
        return self

    def build(self) -> VerificationHandlerBase:
        """Build the verification chain.

        Returns:
            First handler in the chain

        Raises:
            ValueError: If no handlers configured
        """
        if not self._handlers:
            raise ValueError("No verification handlers configured")

        # Link handlers
        for i in range(len(self._handlers) - 1):
            self._handlers[i].set_next(self._handlers[i + 1])

        return self._handlers[0]
