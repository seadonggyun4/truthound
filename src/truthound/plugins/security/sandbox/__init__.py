"""Sandbox execution module for plugin isolation.

This module provides sandbox implementations for executing plugin code
in isolated environments with resource limits and security restrictions.

Architecture:
    SandboxFactory creates the appropriate SandboxEngine based on
    the requested isolation level. Each engine implements the SandboxEngine
    protocol and provides a different level of isolation.

Available Engines:
    - NoopSandbox: No isolation (for trusted plugins)
    - ProcessSandbox: Subprocess with resource limits
    - ContainerSandbox: Docker/Podman container
    - WasmSandbox: WebAssembly sandbox (future)

Example:
    >>> from truthound.plugins.security.sandbox import (
    ...     SandboxFactory,
    ...     ProcessSandboxEngine,
    ... )
    >>> from truthound.plugins.security import SecurityPolicy
    >>>
    >>> # Create sandbox via factory
    >>> policy = SecurityPolicy.standard()
    >>> engine = SandboxFactory.create(policy.isolation_level)
    >>> context = engine.create_sandbox("my-plugin", policy)
    >>> result = await engine.execute(context, my_func, arg1, arg2)
"""

from __future__ import annotations

from truthound.plugins.security.sandbox.factory import SandboxFactory
from truthound.plugins.security.sandbox.context import SandboxContextImpl
from truthound.plugins.security.sandbox.engines.noop import NoopSandboxEngine
from truthound.plugins.security.sandbox.engines.process import ProcessSandboxEngine
from truthound.plugins.security.sandbox.engines.container import ContainerSandboxEngine

__all__ = [
    "SandboxFactory",
    "SandboxContextImpl",
    "NoopSandboxEngine",
    "ProcessSandboxEngine",
    "ContainerSandboxEngine",
]
