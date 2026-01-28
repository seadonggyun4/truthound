"""Base classes for backend monitoring adapters.

This module provides the abstract interfaces for backend-specific monitoring.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Protocol, runtime_checkable

from truthound.profiler.distributed.monitoring.protocols import (
    MonitorEvent,
    WorkerHealth,
)


@dataclass
class BackendWorkerInfo:
    """Information about a backend worker.

    Attributes:
        worker_id: Unique worker identifier
        worker_type: Type of worker (thread, process, container)
        host: Worker hostname or IP
        port: Worker port (if applicable)
        cpu_count: Number of CPUs
        memory_mb: Total memory in MB
        started_at: When worker started
        metadata: Additional backend-specific data
    """

    worker_id: str
    worker_type: str = "unknown"
    host: str = ""
    port: int = 0
    cpu_count: int = 1
    memory_mb: float = 0.0
    started_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BackendMetrics:
    """Metrics from a backend.

    Attributes:
        timestamp: When metrics were collected
        workers_total: Total workers
        workers_active: Active workers
        tasks_running: Running tasks
        tasks_pending: Pending tasks
        memory_used_mb: Memory used in MB
        memory_available_mb: Memory available in MB
        cpu_utilization_percent: CPU usage
        network_bytes_in: Network bytes received
        network_bytes_out: Network bytes sent
        metadata: Backend-specific metrics
    """

    timestamp: datetime = field(default_factory=datetime.now)
    workers_total: int = 0
    workers_active: int = 0
    tasks_running: int = 0
    tasks_pending: int = 0
    memory_used_mb: float = 0.0
    memory_available_mb: float = 0.0
    cpu_utilization_percent: float = 0.0
    network_bytes_in: int = 0
    network_bytes_out: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class BackendMonitorProtocol(Protocol):
    """Protocol for backend monitoring adapters.

    Implement this to create custom backend monitoring integration.
    """

    @property
    def backend_name(self) -> str:
        """Get backend name."""
        ...

    def is_connected(self) -> bool:
        """Check if connected to backend."""
        ...

    def connect(self) -> None:
        """Connect to backend."""
        ...

    def disconnect(self) -> None:
        """Disconnect from backend."""
        ...

    def get_workers(self) -> list[BackendWorkerInfo]:
        """Get all workers.

        Returns:
            List of worker info
        """
        ...

    def get_worker_health(self, worker_id: str) -> WorkerHealth | None:
        """Get worker health.

        Args:
            worker_id: Worker identifier

        Returns:
            Worker health or None
        """
        ...

    def get_metrics(self) -> BackendMetrics:
        """Get backend metrics.

        Returns:
            Backend metrics
        """
        ...

    def set_event_callback(self, callback: Callable[[MonitorEvent], None]) -> None:
        """Set callback for backend events.

        Args:
            callback: Event callback
        """
        ...


class BackendMonitorAdapter(ABC):
    """Abstract base class for backend monitoring adapters.

    Provides common functionality for all backend adapters including
    connection management, event callbacks, and polling.

    Subclasses must implement the backend-specific methods.
    """

    def __init__(
        self,
        poll_interval_seconds: float = 5.0,
    ) -> None:
        """Initialize adapter.

        Args:
            poll_interval_seconds: Interval for polling metrics
        """
        self._poll_interval = poll_interval_seconds
        self._event_callback: Callable[[MonitorEvent], None] | None = None
        self._connected = False

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Get backend name."""
        pass

    def is_connected(self) -> bool:
        """Check if connected to backend."""
        return self._connected

    @abstractmethod
    def connect(self) -> None:
        """Connect to backend."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from backend."""
        pass

    @abstractmethod
    def get_workers(self) -> list[BackendWorkerInfo]:
        """Get all workers.

        Returns:
            List of worker info
        """
        pass

    @abstractmethod
    def get_worker_health(self, worker_id: str) -> WorkerHealth | None:
        """Get worker health.

        Args:
            worker_id: Worker identifier

        Returns:
            Worker health or None
        """
        pass

    @abstractmethod
    def get_metrics(self) -> BackendMetrics:
        """Get backend metrics.

        Returns:
            Backend metrics
        """
        pass

    def set_event_callback(self, callback: Callable[[MonitorEvent], None]) -> None:
        """Set callback for backend events.

        Args:
            callback: Event callback
        """
        self._event_callback = callback

    def _emit_event(self, event: MonitorEvent) -> None:
        """Emit event to callback.

        Args:
            event: Event to emit
        """
        if self._event_callback:
            try:
                self._event_callback(event)
            except Exception:
                pass

    def __enter__(self) -> "BackendMonitorAdapter":
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.disconnect()
