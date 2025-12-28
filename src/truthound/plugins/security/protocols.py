"""Security protocols for plugin system.

This module defines all Protocol interfaces for the plugin security system.
Following Protocol-first design, all components are defined as runtime-checkable
Protocols to enable:
- Type checking at runtime
- Easy mocking in tests
- Flexible implementations
- Clear contracts between components

Design Principles:
    1. Protocol-First: All abstractions are Protocol-based
    2. Immutable Data: Configuration uses frozen dataclasses
    3. Fail-Safe Defaults: Default values are always secure
    4. Composition: Components can be composed via dependency injection
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Protocol,
    TypeVar,
    runtime_checkable,
)

if TYPE_CHECKING:
    pass


# =============================================================================
# Enumerations
# =============================================================================


class IsolationLevel(Enum):
    """Plugin isolation level.

    Determines the level of isolation when executing plugin code.

    Levels:
        NONE: No isolation, code runs in main process (trusted plugins only)
        PROCESS: Separate process with resource limits
        CONTAINER: Docker/Podman container isolation
        WASM: WebAssembly sandbox (most restrictive)
    """

    NONE = auto()
    PROCESS = auto()
    CONTAINER = auto()
    WASM = auto()


class TrustLevel(str, Enum):
    """Trust level for signers and certificates.

    Values:
        TRUSTED: Fully trusted signer
        VERIFIED: Verified but not fully trusted
        UNKNOWN: Unknown signer
        REVOKED: Revoked certificate/signature
    """

    TRUSTED = "trusted"
    VERIFIED = "verified"
    UNKNOWN = "unknown"
    REVOKED = "revoked"


# =============================================================================
# Resource and Policy Data Classes
# =============================================================================


@dataclass(frozen=True)
class ResourceLimits:
    """Resource limits for sandboxed execution.

    All limits are enforced by the sandbox engine. Values of 0 or negative
    mean unlimited (not recommended for untrusted code).

    Attributes:
        max_memory_mb: Maximum memory in megabytes
        max_cpu_percent: Maximum CPU usage (0-100)
        max_execution_time_sec: Maximum execution time in seconds
        max_file_descriptors: Maximum open file descriptors
        allowed_paths: Paths the plugin can access (read-only by default)
        writable_paths: Paths the plugin can write to
        denied_syscalls: System calls to block (Linux-specific)
    """

    max_memory_mb: int = 512
    max_cpu_percent: float = 50.0
    max_execution_time_sec: float = 30.0
    max_file_descriptors: int = 100
    allowed_paths: tuple[str, ...] = ()
    writable_paths: tuple[str, ...] = ()
    denied_syscalls: tuple[str, ...] = (
        "fork",
        "vfork",
        "clone",
        "execve",
        "execveat",
    )

    @classmethod
    def minimal(cls) -> "ResourceLimits":
        """Create minimal resource limits for untrusted code."""
        return cls(
            max_memory_mb=128,
            max_cpu_percent=25.0,
            max_execution_time_sec=10.0,
            max_file_descriptors=10,
        )

    @classmethod
    def standard(cls) -> "ResourceLimits":
        """Create standard resource limits."""
        return cls()

    @classmethod
    def generous(cls) -> "ResourceLimits":
        """Create generous resource limits for trusted code."""
        return cls(
            max_memory_mb=2048,
            max_cpu_percent=100.0,
            max_execution_time_sec=300.0,
            max_file_descriptors=1000,
        )


@dataclass(frozen=True)
class SecurityPolicy:
    """Comprehensive security policy for plugin execution.

    This policy controls all aspects of plugin security including isolation,
    resource limits, network access, and signature requirements.

    Attributes:
        isolation_level: Level of process isolation
        resource_limits: Resource usage limits
        allow_network: Whether network access is allowed
        allow_subprocess: Whether spawning subprocesses is allowed
        allow_file_write: Whether writing to files is allowed
        allowed_modules: Python modules the plugin can import
        blocked_modules: Python modules the plugin cannot import
        required_signatures: Minimum number of valid signatures required
        require_trusted_signer: Whether signer must be in trust store
        signature_max_age_days: Maximum age of signature in days (0 = no limit)
    """

    isolation_level: IsolationLevel = IsolationLevel.PROCESS
    resource_limits: ResourceLimits = field(default_factory=ResourceLimits)
    allow_network: bool = False
    allow_subprocess: bool = False
    allow_file_write: bool = False
    allowed_modules: tuple[str, ...] = (
        "polars",
        "numpy",
        "pandas",
        "truthound",
        "json",
        "datetime",
        "dataclasses",
        "typing",
        "collections",
        "itertools",
        "functools",
        "math",
        "statistics",
        "re",
        "hashlib",
        "base64",
    )
    blocked_modules: tuple[str, ...] = (
        "os",
        "subprocess",
        "shutil",
        "socket",
        "urllib",
        "requests",
        "http",
        "ftplib",
        "smtplib",
        "telnetlib",
        "ctypes",
        "multiprocessing",
        "threading",
        "asyncio.subprocess",
        "sys",
        "importlib",
        "builtins",
        "code",
        "codeop",
        "compile",
        "eval",
        "exec",
        "pickle",
        "marshal",
    )
    required_signatures: int = 1
    require_trusted_signer: bool = True
    signature_max_age_days: int = 365

    @classmethod
    def strict(cls) -> "SecurityPolicy":
        """Create strict security policy for untrusted plugins."""
        return cls(
            isolation_level=IsolationLevel.CONTAINER,
            resource_limits=ResourceLimits.minimal(),
            allow_network=False,
            allow_subprocess=False,
            allow_file_write=False,
            required_signatures=2,
            require_trusted_signer=True,
            signature_max_age_days=90,
        )

    @classmethod
    def standard(cls) -> "SecurityPolicy":
        """Create standard security policy."""
        return cls()

    @classmethod
    def permissive(cls) -> "SecurityPolicy":
        """Create permissive policy for trusted plugins."""
        return cls(
            isolation_level=IsolationLevel.NONE,
            resource_limits=ResourceLimits.generous(),
            allow_network=True,
            allow_subprocess=False,
            allow_file_write=True,
            required_signatures=0,
            require_trusted_signer=False,
            signature_max_age_days=0,
        )

    @classmethod
    def development(cls) -> "SecurityPolicy":
        """Create development policy (no security checks)."""
        return cls(
            isolation_level=IsolationLevel.NONE,
            resource_limits=ResourceLimits.generous(),
            allow_network=True,
            allow_subprocess=True,
            allow_file_write=True,
            allowed_modules=(),  # Empty means all allowed
            blocked_modules=(),  # Empty means none blocked
            required_signatures=0,
            require_trusted_signer=False,
        )


# =============================================================================
# Sandbox Protocols
# =============================================================================


@runtime_checkable
class SandboxContext(Protocol):
    """Context for sandbox execution.

    Represents a sandboxed execution environment with its configuration
    and current state.
    """

    @property
    def plugin_id(self) -> str:
        """Unique identifier of the plugin being sandboxed."""
        ...

    @property
    def policy(self) -> SecurityPolicy:
        """Security policy for this sandbox."""
        ...

    @property
    def sandbox_id(self) -> str:
        """Unique identifier for this sandbox instance."""
        ...

    def is_alive(self) -> bool:
        """Check if sandbox is still running.

        Returns:
            True if sandbox process/container is running
        """
        ...

    def get_resource_usage(self) -> dict[str, float]:
        """Get current resource usage.

        Returns:
            Dict with keys: memory_mb, cpu_percent, execution_time_sec
        """
        ...


@runtime_checkable
class SandboxEngine(Protocol):
    """Engine for creating and managing sandboxed execution environments.

    Implementations provide different isolation mechanisms:
    - ProcessSandbox: Subprocess with resource limits
    - ContainerSandbox: Docker/Podman containers
    - WasmSandbox: WebAssembly isolation
    """

    @property
    def isolation_level(self) -> IsolationLevel:
        """The isolation level provided by this engine."""
        ...

    def create_sandbox(
        self,
        plugin_id: str,
        policy: SecurityPolicy,
    ) -> SandboxContext:
        """Create a new sandbox context for a plugin.

        Args:
            plugin_id: Unique identifier for the plugin
            policy: Security policy to apply

        Returns:
            SandboxContext for execution
        """
        ...

    async def execute(
        self,
        context: SandboxContext,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute a function within the sandbox.

        Args:
            context: Sandbox context to use
            func: Function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result of function execution

        Raises:
            SandboxTimeoutError: If execution times out
            SandboxResourceError: If resource limits exceeded
            SandboxSecurityViolation: If security policy violated
        """
        ...

    def terminate(self, context: SandboxContext) -> None:
        """Terminate a sandbox.

        Args:
            context: Sandbox to terminate
        """
        ...

    async def cleanup(self) -> None:
        """Clean up all sandbox resources."""
        ...


# =============================================================================
# Signing Protocols
# =============================================================================


@dataclass(frozen=True)
class SignatureInfo:
    """Information about a plugin signature.

    Attributes:
        signer_id: Identifier of the signer
        algorithm: Algorithm used for signing (e.g., "RSA-SHA256", "Ed25519")
        signature: Base64-encoded signature bytes
        timestamp: When the signature was created
        expires_at: When the signature expires (None = never)
        certificate_chain: Chain of certificates (issuer chain)
        metadata: Additional metadata about the signature
    """

    signer_id: str
    algorithm: str
    signature: bytes
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime | None = None
    certificate_chain: tuple[bytes, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if signature has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    @property
    def age_days(self) -> int:
        """Get age of signature in days."""
        delta = datetime.now(timezone.utc) - self.timestamp
        return delta.days


@dataclass(frozen=True)
class VerificationResult:
    """Result of signature verification.

    Attributes:
        is_valid: Whether signature is valid
        signer_id: ID of the signer (if verified)
        trust_level: Trust level of the signer
        errors: List of error messages (if any)
        warnings: List of warning messages (if any)
        verified_at: When verification was performed
    """

    is_valid: bool
    signer_id: str | None = None
    trust_level: TrustLevel = TrustLevel.UNKNOWN
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    verified_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def success(
        cls,
        signer_id: str,
        trust_level: TrustLevel = TrustLevel.TRUSTED,
        warnings: tuple[str, ...] = (),
    ) -> "VerificationResult":
        """Create successful verification result."""
        return cls(
            is_valid=True,
            signer_id=signer_id,
            trust_level=trust_level,
            warnings=warnings,
        )

    @classmethod
    def failure(cls, *errors: str) -> "VerificationResult":
        """Create failed verification result."""
        return cls(
            is_valid=False,
            errors=errors,
        )


@runtime_checkable
class SigningService(Protocol):
    """Service for signing and verifying plugins.

    Provides cryptographic signing and verification of plugin code
    to ensure integrity and authenticity.
    """

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
            private_key: PEM-encoded private key
            certificate: Optional X.509 certificate
            metadata: Additional metadata to include

        Returns:
            SignatureInfo with signature details
        """
        ...

    def verify(
        self,
        plugin_path: Path,
        signature: SignatureInfo,
    ) -> VerificationResult:
        """Verify a plugin signature.

        Args:
            plugin_path: Path to plugin file or directory
            signature: Signature to verify

        Returns:
            VerificationResult with verification status
        """
        ...

    def get_plugin_hash(self, plugin_path: Path) -> str:
        """Get hash of plugin content for signing.

        Args:
            plugin_path: Path to plugin

        Returns:
            Hex-encoded hash of plugin content
        """
        ...


@runtime_checkable
class TrustStore(Protocol):
    """Store for trusted certificates and signers.

    Manages the set of trusted certificates and signers that are
    allowed to sign plugins.
    """

    def add_trusted_certificate(
        self,
        certificate: bytes,
        trust_level: TrustLevel = TrustLevel.TRUSTED,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Add a certificate to the trust store.

        Args:
            certificate: DER or PEM encoded certificate
            trust_level: Trust level for this certificate
            metadata: Additional metadata

        Returns:
            Certificate ID
        """
        ...

    def remove_certificate(self, cert_id: str) -> bool:
        """Remove a certificate from trust store.

        Args:
            cert_id: Certificate ID

        Returns:
            True if removed, False if not found
        """
        ...

    def revoke_certificate(
        self,
        cert_id: str,
        reason: str = "",
    ) -> bool:
        """Revoke a certificate.

        Args:
            cert_id: Certificate ID
            reason: Reason for revocation

        Returns:
            True if revoked, False if not found
        """
        ...

    def is_trusted(self, certificate: bytes) -> tuple[bool, TrustLevel]:
        """Check if certificate is trusted.

        Args:
            certificate: Certificate to check

        Returns:
            Tuple of (is_trusted, trust_level)
        """
        ...

    def get_trust_level(self, signer_id: str) -> TrustLevel:
        """Get trust level for a signer.

        Args:
            signer_id: Signer identifier

        Returns:
            Trust level of the signer
        """
        ...

    def list_certificates(self) -> list[dict[str, Any]]:
        """List all certificates in store.

        Returns:
            List of certificate info dicts
        """
        ...


@runtime_checkable
class VerificationHandler(Protocol):
    """Handler in verification chain.

    Implements Chain of Responsibility pattern for multi-step
    signature verification.
    """

    def set_next(self, handler: "VerificationHandler") -> "VerificationHandler":
        """Set the next handler in the chain.

        Args:
            handler: Next handler

        Returns:
            The next handler (for chaining)
        """
        ...

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
            context: Shared context for chain

        Returns:
            VerificationResult if verification completed (success or failure),
            None to continue to next handler
        """
        ...


# =============================================================================
# Type Variables
# =============================================================================

T = TypeVar("T")
SandboxEngineT = TypeVar("SandboxEngineT", bound=SandboxEngine)
