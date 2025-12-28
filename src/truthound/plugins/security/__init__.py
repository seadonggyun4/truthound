"""Plugin Security Module.

This module provides comprehensive security features for Truthound plugins:
- Sandbox execution with multiple isolation levels
- Plugin signing and verification
- Trust stores and certificate management
- Security policies and presets

Architecture:
    The security module follows Protocol-first design, allowing easy extension
    and testing through dependency injection.

Example:
    >>> from truthound.plugins.security import (
    ...     SecurityPolicy,
    ...     SandboxFactory,
    ...     SigningService,
    ... )
    >>>
    >>> # Create a security policy
    >>> policy = SecurityPolicy.strict()
    >>>
    >>> # Create sandbox and execute
    >>> sandbox = SandboxFactory.create(policy.isolation_level)
    >>> result = await sandbox.execute(context, my_func)
"""

from __future__ import annotations

from truthound.plugins.security.protocols import (
    # Enums
    IsolationLevel,
    TrustLevel,
    # Resource and Policy
    ResourceLimits,
    SecurityPolicy,
    # Sandbox Protocols
    SandboxContext,
    SandboxEngine,
    # Signing Protocols
    SignatureInfo,
    VerificationResult,
    SigningService,
    TrustStore,
    VerificationHandler,
)

from truthound.plugins.security.policies import (
    SecurityPolicyPresets,
    create_policy,
)

from truthound.plugins.security.exceptions import (
    SecurityError,
    SandboxError,
    SandboxTimeoutError,
    SandboxResourceError,
    SandboxSecurityViolation,
    SignatureError,
    SignatureExpiredError,
    SignatureTamperError,
    UntrustedSignerError,
    CertificateError,
)

__all__ = [
    # Protocols and Types
    "IsolationLevel",
    "TrustLevel",
    "ResourceLimits",
    "SecurityPolicy",
    "SandboxContext",
    "SandboxEngine",
    "SignatureInfo",
    "VerificationResult",
    "SigningService",
    "TrustStore",
    "VerificationHandler",
    # Policies
    "SecurityPolicyPresets",
    "create_policy",
    # Exceptions
    "SecurityError",
    "SandboxError",
    "SandboxTimeoutError",
    "SandboxResourceError",
    "SandboxSecurityViolation",
    "SignatureError",
    "SignatureExpiredError",
    "SignatureTamperError",
    "UntrustedSignerError",
    "CertificateError",
]
