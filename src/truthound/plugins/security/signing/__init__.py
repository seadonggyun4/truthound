"""Plugin signing and verification module.

This module provides cryptographic signing and verification for plugins
using the Chain of Responsibility pattern for multi-step verification.

Components:
    - SigningServiceImpl: Signs plugins with various algorithms
    - TrustStoreImpl: Manages trusted certificates
    - VerificationChain: Chain of verification handlers

Example:
    >>> from truthound.plugins.security.signing import (
    ...     SigningServiceImpl,
    ...     TrustStoreImpl,
    ...     create_verification_chain,
    ... )
    >>>
    >>> # Sign a plugin
    >>> service = SigningServiceImpl()
    >>> signature = service.sign(plugin_path, private_key)
    >>>
    >>> # Verify with trust store
    >>> trust_store = TrustStoreImpl()
    >>> chain = create_verification_chain(trust_store)
    >>> result = chain.verify(plugin_path, signature, {})
"""

from __future__ import annotations

from truthound.plugins.security.signing.service import (
    SigningServiceImpl,
    SignatureAlgorithm,
)
from truthound.plugins.security.signing.trust_store import TrustStoreImpl
from truthound.plugins.security.signing.verifier import (
    VerificationHandlerBase,
    IntegrityVerifier,
    SignatureVerifier,
    TrustVerifier,
    ExpirationVerifier,
    create_verification_chain,
)

__all__ = [
    # Service
    "SigningServiceImpl",
    "SignatureAlgorithm",
    # Trust Store
    "TrustStoreImpl",
    # Verification Chain
    "VerificationHandlerBase",
    "IntegrityVerifier",
    "SignatureVerifier",
    "TrustVerifier",
    "ExpirationVerifier",
    "create_verification_chain",
]
