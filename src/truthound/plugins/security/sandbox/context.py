"""Sandbox context implementation.

This module provides the concrete implementation of SandboxContext
that tracks sandbox state and resource usage.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from truthound.plugins.security.protocols import SecurityPolicy, SandboxContext


@dataclass
class SandboxContextImpl:
    """Concrete implementation of SandboxContext.

    Tracks the state of a sandbox instance including resource usage
    and lifecycle information.

    Attributes:
        plugin_id: ID of the plugin being sandboxed
        policy: Security policy applied to this sandbox
        sandbox_id: Unique ID for this sandbox instance
        created_at: When the sandbox was created
        started_at: When execution started (None if not started)
        finished_at: When execution finished (None if not finished)
    """

    plugin_id: str
    policy: SecurityPolicy
    sandbox_id: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    finished_at: datetime | None = None

    # Internal state
    _process_id: int | None = field(default=None, repr=False)
    _container_id: str | None = field(default=None, repr=False)
    _alive: bool = field(default=True, repr=False)
    _memory_used_mb: float = field(default=0.0, repr=False)
    _cpu_percent: float = field(default=0.0, repr=False)
    _start_time: float = field(default=0.0, repr=False)

    def __post_init__(self) -> None:
        """Generate sandbox ID if not provided."""
        if not self.sandbox_id:
            self.sandbox_id = self._generate_id()

    def _generate_id(self) -> str:
        """Generate unique sandbox ID."""
        content = f"{self.plugin_id}-{self.created_at.isoformat()}-{id(self)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def is_alive(self) -> bool:
        """Check if sandbox is still running."""
        return self._alive

    def get_resource_usage(self) -> dict[str, float]:
        """Get current resource usage.

        Returns:
            Dict with memory_mb, cpu_percent, and execution_time_sec
        """
        execution_time = 0.0
        if self._start_time > 0:
            if self.finished_at:
                execution_time = (self.finished_at - self.started_at).total_seconds() if self.started_at else 0.0
            else:
                execution_time = time.perf_counter() - self._start_time

        return {
            "memory_mb": self._memory_used_mb,
            "cpu_percent": self._cpu_percent,
            "execution_time_sec": execution_time,
        }

    def mark_started(self) -> None:
        """Mark sandbox as started."""
        self.started_at = datetime.now(timezone.utc)
        self._start_time = time.perf_counter()

    def mark_finished(self) -> None:
        """Mark sandbox as finished."""
        self.finished_at = datetime.now(timezone.utc)
        self._alive = False

    def mark_terminated(self) -> None:
        """Mark sandbox as terminated."""
        self._alive = False
        if not self.finished_at:
            self.finished_at = datetime.now(timezone.utc)

    def update_resource_usage(
        self,
        memory_mb: float | None = None,
        cpu_percent: float | None = None,
    ) -> None:
        """Update resource usage metrics."""
        if memory_mb is not None:
            self._memory_used_mb = memory_mb
        if cpu_percent is not None:
            self._cpu_percent = cpu_percent

    def set_process_id(self, pid: int) -> None:
        """Set the process ID for process-based sandboxes."""
        self._process_id = pid

    def set_container_id(self, container_id: str) -> None:
        """Set the container ID for container-based sandboxes."""
        self._container_id = container_id

    @property
    def process_id(self) -> int | None:
        """Get process ID if applicable."""
        return self._process_id

    @property
    def container_id(self) -> str | None:
        """Get container ID if applicable."""
        return self._container_id

    @property
    def execution_time_sec(self) -> float:
        """Get current execution time in seconds."""
        return self.get_resource_usage()["execution_time_sec"]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "plugin_id": self.plugin_id,
            "sandbox_id": self.sandbox_id,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "is_alive": self._alive,
            "resource_usage": self.get_resource_usage(),
            "process_id": self._process_id,
            "container_id": self._container_id,
        }


# Verify implementation satisfies protocol
def _verify_protocol() -> None:
    """Verify SandboxContextImpl implements SandboxContext protocol."""
    context: SandboxContext = SandboxContextImpl(
        plugin_id="test",
        policy=SecurityPolicy.standard(),
    )
    assert hasattr(context, "plugin_id")
    assert hasattr(context, "policy")
    assert hasattr(context, "sandbox_id")
    assert callable(context.is_alive)
    assert callable(context.get_resource_usage)
