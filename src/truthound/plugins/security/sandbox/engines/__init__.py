"""Sandbox engine implementations.

This module contains concrete implementations of the SandboxEngine protocol:
- NoopSandboxEngine: No isolation (for trusted plugins)
- ProcessSandboxEngine: Subprocess isolation with resource limits
- ContainerSandboxEngine: Docker/Podman container isolation
"""

from __future__ import annotations

from truthound.plugins.security.sandbox.engines.noop import NoopSandboxEngine
from truthound.plugins.security.sandbox.engines.process import ProcessSandboxEngine
from truthound.plugins.security.sandbox.engines.container import ContainerSandboxEngine

__all__ = [
    "NoopSandboxEngine",
    "ProcessSandboxEngine",
    "ContainerSandboxEngine",
]
