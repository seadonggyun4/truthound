"""Security policy presets and factory functions.

This module provides pre-configured security policies for common use cases
and a factory for creating custom policies.

Usage:
    >>> from truthound.plugins.security.policies import (
    ...     SecurityPolicyPresets,
    ...     create_policy,
    ... )
    >>>
    >>> # Use a preset
    >>> policy = SecurityPolicyPresets.ENTERPRISE
    >>>
    >>> # Or create custom policy
    >>> policy = create_policy(
    ...     isolation_level="container",
    ...     allow_network=False,
    ...     required_signatures=2,
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from truthound.plugins.security.protocols import (
    IsolationLevel,
    ResourceLimits,
    SecurityPolicy,
)


class SecurityPolicyPresets(Enum):
    """Pre-configured security policy presets.

    Available presets:
        DEVELOPMENT: No restrictions, for local development
        TESTING: Minimal restrictions, for testing
        STANDARD: Balanced security for trusted plugins
        ENTERPRISE: High security for production
        STRICT: Maximum security for untrusted code
        AIRGAPPED: No network, maximum isolation
    """

    DEVELOPMENT = "development"
    TESTING = "testing"
    STANDARD = "standard"
    ENTERPRISE = "enterprise"
    STRICT = "strict"
    AIRGAPPED = "airgapped"

    def to_policy(self) -> SecurityPolicy:
        """Convert preset to SecurityPolicy instance."""
        return _PRESET_POLICIES[self]


# Policy definitions for each preset
_PRESET_POLICIES: dict[SecurityPolicyPresets, SecurityPolicy] = {
    SecurityPolicyPresets.DEVELOPMENT: SecurityPolicy(
        isolation_level=IsolationLevel.NONE,
        resource_limits=ResourceLimits(
            max_memory_mb=4096,
            max_cpu_percent=100.0,
            max_execution_time_sec=600.0,
            max_file_descriptors=10000,
        ),
        allow_network=True,
        allow_subprocess=True,
        allow_file_write=True,
        allowed_modules=(),  # All allowed
        blocked_modules=(),  # None blocked
        required_signatures=0,
        require_trusted_signer=False,
        signature_max_age_days=0,  # No limit
    ),
    SecurityPolicyPresets.TESTING: SecurityPolicy(
        isolation_level=IsolationLevel.NONE,
        resource_limits=ResourceLimits(
            max_memory_mb=2048,
            max_cpu_percent=100.0,
            max_execution_time_sec=120.0,
            max_file_descriptors=1000,
        ),
        allow_network=True,
        allow_subprocess=False,
        allow_file_write=True,
        allowed_modules=(),  # All allowed
        blocked_modules=("ctypes", "cffi"),  # Only unsafe FFI blocked
        required_signatures=0,
        require_trusted_signer=False,
        signature_max_age_days=0,
    ),
    SecurityPolicyPresets.STANDARD: SecurityPolicy(
        isolation_level=IsolationLevel.PROCESS,
        resource_limits=ResourceLimits.standard(),
        allow_network=False,
        allow_subprocess=False,
        allow_file_write=False,
        required_signatures=1,
        require_trusted_signer=False,
        signature_max_age_days=365,
    ),
    SecurityPolicyPresets.ENTERPRISE: SecurityPolicy(
        isolation_level=IsolationLevel.PROCESS,
        resource_limits=ResourceLimits(
            max_memory_mb=1024,
            max_cpu_percent=80.0,
            max_execution_time_sec=60.0,
            max_file_descriptors=100,
        ),
        allow_network=False,
        allow_subprocess=False,
        allow_file_write=False,
        required_signatures=1,
        require_trusted_signer=True,
        signature_max_age_days=180,
    ),
    SecurityPolicyPresets.STRICT: SecurityPolicy(
        isolation_level=IsolationLevel.CONTAINER,
        resource_limits=ResourceLimits.minimal(),
        allow_network=False,
        allow_subprocess=False,
        allow_file_write=False,
        required_signatures=2,
        require_trusted_signer=True,
        signature_max_age_days=90,
    ),
    SecurityPolicyPresets.AIRGAPPED: SecurityPolicy(
        isolation_level=IsolationLevel.CONTAINER,
        resource_limits=ResourceLimits(
            max_memory_mb=256,
            max_cpu_percent=50.0,
            max_execution_time_sec=30.0,
            max_file_descriptors=10,
            denied_syscalls=(
                "fork",
                "vfork",
                "clone",
                "execve",
                "execveat",
                "socket",
                "connect",
                "accept",
                "bind",
                "listen",
            ),
        ),
        allow_network=False,
        allow_subprocess=False,
        allow_file_write=False,
        blocked_modules=(
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
            "cffi",
            "multiprocessing",
            "threading",
            "asyncio",
            "sys",
            "importlib",
            "builtins",
            "code",
            "codeop",
            "pickle",
            "marshal",
            "shelve",
            "dbm",
            "sqlite3",
            "ssl",
            "tempfile",
            "pty",
            "tty",
            "fcntl",
            "mmap",
            "signal",
        ),
        required_signatures=2,
        require_trusted_signer=True,
        signature_max_age_days=30,
    ),
}


def create_policy(
    preset: str | SecurityPolicyPresets | None = None,
    isolation_level: str | IsolationLevel | None = None,
    max_memory_mb: int | None = None,
    max_cpu_percent: float | None = None,
    max_execution_time_sec: float | None = None,
    allow_network: bool | None = None,
    allow_subprocess: bool | None = None,
    allow_file_write: bool | None = None,
    allowed_modules: tuple[str, ...] | list[str] | None = None,
    blocked_modules: tuple[str, ...] | list[str] | None = None,
    required_signatures: int | None = None,
    require_trusted_signer: bool | None = None,
    signature_max_age_days: int | None = None,
) -> SecurityPolicy:
    """Create a security policy with optional customizations.

    Starts from a preset (or STANDARD if not specified) and applies
    any provided overrides.

    Args:
        preset: Base preset to start from
        isolation_level: Override isolation level
        max_memory_mb: Override memory limit
        max_cpu_percent: Override CPU limit
        max_execution_time_sec: Override time limit
        allow_network: Override network access
        allow_subprocess: Override subprocess access
        allow_file_write: Override file write access
        allowed_modules: Override allowed modules
        blocked_modules: Override blocked modules
        required_signatures: Override signature requirement
        require_trusted_signer: Override trusted signer requirement
        signature_max_age_days: Override signature age limit

    Returns:
        Configured SecurityPolicy

    Example:
        >>> # Start from enterprise preset but allow network
        >>> policy = create_policy(
        ...     preset="enterprise",
        ...     allow_network=True,
        ... )
    """
    # Get base policy
    if preset is None:
        base = SecurityPolicy.standard()
    elif isinstance(preset, SecurityPolicyPresets):
        base = preset.to_policy()
    elif isinstance(preset, str):
        try:
            preset_enum = SecurityPolicyPresets(preset.lower())
            base = preset_enum.to_policy()
        except ValueError:
            raise ValueError(
                f"Unknown preset: {preset}. "
                f"Available: {[p.value for p in SecurityPolicyPresets]}"
            )
    else:
        raise TypeError(f"preset must be str or SecurityPolicyPresets, got {type(preset)}")

    # Parse isolation level
    if isolation_level is not None:
        if isinstance(isolation_level, str):
            try:
                isolation_level = IsolationLevel[isolation_level.upper()]
            except KeyError:
                raise ValueError(
                    f"Unknown isolation level: {isolation_level}. "
                    f"Available: {[l.name.lower() for l in IsolationLevel]}"
                )
        elif not isinstance(isolation_level, IsolationLevel):
            raise TypeError(
                f"isolation_level must be str or IsolationLevel, got {type(isolation_level)}"
            )

    # Build resource limits if any resource overrides provided
    resource_overrides = {}
    if max_memory_mb is not None:
        resource_overrides["max_memory_mb"] = max_memory_mb
    if max_cpu_percent is not None:
        resource_overrides["max_cpu_percent"] = max_cpu_percent
    if max_execution_time_sec is not None:
        resource_overrides["max_execution_time_sec"] = max_execution_time_sec

    if resource_overrides:
        # Create new ResourceLimits with overrides
        resource_limits = ResourceLimits(
            max_memory_mb=resource_overrides.get(
                "max_memory_mb", base.resource_limits.max_memory_mb
            ),
            max_cpu_percent=resource_overrides.get(
                "max_cpu_percent", base.resource_limits.max_cpu_percent
            ),
            max_execution_time_sec=resource_overrides.get(
                "max_execution_time_sec", base.resource_limits.max_execution_time_sec
            ),
            max_file_descriptors=base.resource_limits.max_file_descriptors,
            allowed_paths=base.resource_limits.allowed_paths,
            writable_paths=base.resource_limits.writable_paths,
            denied_syscalls=base.resource_limits.denied_syscalls,
        )
    else:
        resource_limits = base.resource_limits

    # Convert list to tuple if needed
    if isinstance(allowed_modules, list):
        allowed_modules = tuple(allowed_modules)
    if isinstance(blocked_modules, list):
        blocked_modules = tuple(blocked_modules)

    # Create policy with overrides
    return SecurityPolicy(
        isolation_level=isolation_level if isolation_level is not None else base.isolation_level,
        resource_limits=resource_limits,
        allow_network=allow_network if allow_network is not None else base.allow_network,
        allow_subprocess=allow_subprocess if allow_subprocess is not None else base.allow_subprocess,
        allow_file_write=allow_file_write if allow_file_write is not None else base.allow_file_write,
        allowed_modules=allowed_modules if allowed_modules is not None else base.allowed_modules,
        blocked_modules=blocked_modules if blocked_modules is not None else base.blocked_modules,
        required_signatures=required_signatures if required_signatures is not None else base.required_signatures,
        require_trusted_signer=require_trusted_signer if require_trusted_signer is not None else base.require_trusted_signer,
        signature_max_age_days=signature_max_age_days if signature_max_age_days is not None else base.signature_max_age_days,
    )


def get_preset(name: str) -> SecurityPolicy:
    """Get a security policy preset by name.

    Args:
        name: Preset name (case-insensitive)

    Returns:
        SecurityPolicy for the preset

    Raises:
        ValueError: If preset name is unknown
    """
    try:
        preset = SecurityPolicyPresets(name.lower())
        return preset.to_policy()
    except ValueError:
        available = [p.value for p in SecurityPolicyPresets]
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")


def list_presets() -> list[str]:
    """List all available preset names.

    Returns:
        List of preset names
    """
    return [p.value for p in SecurityPolicyPresets]
