"""Distributed timeout coordination for multi-node validation.

This module provides coordination mechanisms for managing timeouts
across distributed validation workers. It supports:
- Distributed deadline coordination via Redis/DynamoDB
- Leader election for timeout management
- Heartbeat monitoring
- Distributed circuit breaker

Note: Actual Redis/DynamoDB integration requires external dependencies.
This module provides the interface and in-memory implementations
for testing and single-node deployments.

Example:
    # Create distributed manager
    manager = DistributedTimeoutManager(
        config=DistributedTimeoutConfig(
            coordinator_backend=CoordinatorBackend.MEMORY,
            node_id="worker-1",
        )
    )

    # Execute with distributed coordination
    async with manager:
        result = await manager.execute_distributed(
            validate_batch,
            timeout_seconds=60,
            batch_id="batch-123",
        )
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import threading
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Generator, TypeVar

from truthound.validators.timeout.deadline import (
    DeadlineContext,
    DeadlineExceededError,
    TimeoutBudget,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CoordinatorBackend(str, Enum):
    """Backend for distributed coordination."""

    MEMORY = "memory"       # In-memory (single node)
    REDIS = "redis"         # Redis-based coordination
    DYNAMODB = "dynamodb"   # DynamoDB-based coordination
    ETCD = "etcd"          # etcd-based coordination


class NodeStatus(str, Enum):
    """Status of a distributed node."""

    ACTIVE = "active"
    BUSY = "busy"
    DRAINING = "draining"
    OFFLINE = "offline"


@dataclass
class DistributedTimeoutConfig:
    """Configuration for distributed timeout management.

    Attributes:
        coordinator_backend: Backend for coordination
        node_id: Unique identifier for this node
        heartbeat_interval: Seconds between heartbeats
        heartbeat_timeout: Seconds before node considered dead
        max_retries: Maximum retry attempts
        circuit_breaker_threshold: Failures before opening circuit
        circuit_breaker_timeout: Seconds before retrying after open
    """

    coordinator_backend: CoordinatorBackend = CoordinatorBackend.MEMORY
    node_id: str = field(default_factory=lambda: f"node-{uuid.uuid4().hex[:8]}")
    heartbeat_interval: float = 5.0
    heartbeat_timeout: float = 15.0
    max_retries: int = 3
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 30.0
    connection_url: str = ""


@dataclass
class DistributedDeadline:
    """A deadline shared across distributed nodes.

    Attributes:
        deadline_id: Unique identifier for this deadline
        deadline_utc: Absolute deadline in UTC
        owner_node: Node that created the deadline
        operation_id: Operation being coordinated
        metadata: Additional metadata
        status: Current status
    """

    deadline_id: str
    deadline_utc: datetime
    owner_node: str
    operation_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    status: str = "active"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def remaining_seconds(self) -> float:
        """Get remaining time until deadline."""
        now = datetime.now(timezone.utc)
        remaining = (self.deadline_utc - now).total_seconds()
        return max(0.0, remaining)

    @property
    def is_expired(self) -> bool:
        """Check if deadline has passed."""
        return self.remaining_seconds <= 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "deadline_id": self.deadline_id,
            "deadline_utc": self.deadline_utc.isoformat(),
            "owner_node": self.owner_node,
            "operation_id": self.operation_id,
            "metadata": self.metadata,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DistributedDeadline":
        """Create from dictionary."""
        deadline_utc = datetime.fromisoformat(data["deadline_utc"])
        if deadline_utc.tzinfo is None:
            deadline_utc = deadline_utc.replace(tzinfo=timezone.utc)

        created = data.get("created_at")
        if created:
            created = datetime.fromisoformat(created)
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
        else:
            created = datetime.now(timezone.utc)

        return cls(
            deadline_id=data["deadline_id"],
            deadline_utc=deadline_utc,
            owner_node=data["owner_node"],
            operation_id=data.get("operation_id", ""),
            metadata=data.get("metadata", {}),
            status=data.get("status", "active"),
            created_at=created,
        )


class CoordinatorInterface(ABC):
    """Interface for distributed coordination backends."""

    @abstractmethod
    async def register_node(self, node_id: str, metadata: dict[str, Any]) -> bool:
        """Register a node with the coordinator."""
        pass

    @abstractmethod
    async def heartbeat(self, node_id: str) -> bool:
        """Send heartbeat for a node."""
        pass

    @abstractmethod
    async def create_deadline(
        self,
        node_id: str,
        timeout_seconds: float,
        operation_id: str = "",
    ) -> DistributedDeadline:
        """Create a distributed deadline."""
        pass

    @abstractmethod
    async def get_deadline(self, deadline_id: str) -> DistributedDeadline | None:
        """Get a deadline by ID."""
        pass

    @abstractmethod
    async def complete_deadline(
        self,
        deadline_id: str,
        status: str = "completed",
    ) -> bool:
        """Mark a deadline as completed."""
        pass

    @abstractmethod
    async def get_active_nodes(self) -> list[str]:
        """Get list of active node IDs."""
        pass


class InMemoryCoordinator(CoordinatorInterface):
    """In-memory coordinator for single-node or testing.

    This coordinator stores all state in memory and is suitable for:
    - Single-node deployments
    - Testing and development
    - Fallback when distributed backend is unavailable
    """

    def __init__(self) -> None:
        self._nodes: dict[str, dict[str, Any]] = {}
        self._deadlines: dict[str, DistributedDeadline] = {}
        self._heartbeats: dict[str, datetime] = {}
        self._lock = threading.Lock()

    async def register_node(self, node_id: str, metadata: dict[str, Any]) -> bool:
        """Register a node."""
        with self._lock:
            self._nodes[node_id] = {
                "metadata": metadata,
                "registered_at": datetime.now(timezone.utc),
                "status": NodeStatus.ACTIVE.value,
            }
            self._heartbeats[node_id] = datetime.now(timezone.utc)
        return True

    async def heartbeat(self, node_id: str) -> bool:
        """Send heartbeat."""
        with self._lock:
            if node_id not in self._nodes:
                return False
            self._heartbeats[node_id] = datetime.now(timezone.utc)
        return True

    async def create_deadline(
        self,
        node_id: str,
        timeout_seconds: float,
        operation_id: str = "",
    ) -> DistributedDeadline:
        """Create a deadline."""
        deadline_id = f"deadline-{uuid.uuid4().hex[:12]}"
        deadline_utc = datetime.now(timezone.utc) + timedelta(seconds=timeout_seconds)

        deadline = DistributedDeadline(
            deadline_id=deadline_id,
            deadline_utc=deadline_utc,
            owner_node=node_id,
            operation_id=operation_id,
        )

        with self._lock:
            self._deadlines[deadline_id] = deadline

        return deadline

    async def get_deadline(self, deadline_id: str) -> DistributedDeadline | None:
        """Get a deadline."""
        with self._lock:
            return self._deadlines.get(deadline_id)

    async def complete_deadline(
        self,
        deadline_id: str,
        status: str = "completed",
    ) -> bool:
        """Complete a deadline."""
        with self._lock:
            if deadline_id in self._deadlines:
                self._deadlines[deadline_id].status = status
                return True
        return False

    async def get_active_nodes(self) -> list[str]:
        """Get active nodes."""
        now = datetime.now(timezone.utc)
        timeout = timedelta(seconds=15)

        with self._lock:
            active = []
            for node_id, last_heartbeat in self._heartbeats.items():
                if now - last_heartbeat < timeout:
                    active.append(node_id)
            return active


@dataclass
class DistributedExecutionResult:
    """Result of a distributed execution.

    Attributes:
        success: Whether execution succeeded
        value: Result value if successful
        deadline: Deadline used for execution
        node_id: Node that executed the operation
        error: Error message if failed
        metrics: Execution metrics
    """

    success: bool
    value: Any = None
    deadline: DistributedDeadline | None = None
    node_id: str = ""
    error: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(
        cls,
        value: Any,
        deadline: DistributedDeadline | None = None,
        node_id: str = "",
    ) -> "DistributedExecutionResult":
        """Create success result."""
        return cls(success=True, value=value, deadline=deadline, node_id=node_id)

    @classmethod
    def failure(
        cls,
        error: str,
        deadline: DistributedDeadline | None = None,
        node_id: str = "",
    ) -> "DistributedExecutionResult":
        """Create failure result."""
        return cls(success=False, error=error, deadline=deadline, node_id=node_id)


class DistributedTimeoutManager:
    """Manager for distributed timeout coordination.

    This class manages timeout coordination across distributed nodes,
    providing:
    - Distributed deadline creation and tracking
    - Heartbeat monitoring
    - Automatic node registration
    - Circuit breaker for failed operations

    Example:
        manager = DistributedTimeoutManager(config)

        async with manager:
            result = await manager.execute_distributed(
                my_operation,
                timeout_seconds=60,
            )
    """

    def __init__(self, config: DistributedTimeoutConfig | None = None):
        """Initialize the manager.

        Args:
            config: Configuration for distributed timeout
        """
        self.config = config or DistributedTimeoutConfig()
        self._coordinator: CoordinatorInterface | None = None
        self._heartbeat_task: asyncio.Task | None = None
        self._running = False
        self._failure_count = 0
        self._circuit_open = False
        self._circuit_opened_at: datetime | None = None

    async def __aenter__(self) -> "DistributedTimeoutManager":
        """Start the manager."""
        await self.start()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Stop the manager."""
        await self.stop()

    async def start(self) -> None:
        """Start the manager and register with coordinator."""
        # Create coordinator based on backend
        if self.config.coordinator_backend == CoordinatorBackend.MEMORY:
            self._coordinator = InMemoryCoordinator()
        else:
            # For other backends, fall back to memory for now
            logger.warning(
                f"Backend {self.config.coordinator_backend} not implemented, "
                "using in-memory coordinator"
            )
            self._coordinator = InMemoryCoordinator()

        # Register this node
        await self._coordinator.register_node(
            self.config.node_id,
            {"started_at": datetime.now(timezone.utc).isoformat()},
        )

        # Start heartbeat
        self._running = True
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        logger.info(f"Distributed timeout manager started: {self.config.node_id}")

    async def stop(self) -> None:
        """Stop the manager."""
        self._running = False

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        logger.info(f"Distributed timeout manager stopped: {self.config.node_id}")

    async def _heartbeat_loop(self) -> None:
        """Background heartbeat loop."""
        while self._running:
            try:
                if self._coordinator:
                    await self._coordinator.heartbeat(self.config.node_id)
            except Exception as e:
                logger.warning(f"Heartbeat failed: {e}")

            await asyncio.sleep(self.config.heartbeat_interval)

    def _check_circuit(self) -> bool:
        """Check if circuit breaker allows requests.

        Returns:
            True if requests allowed
        """
        if not self._circuit_open:
            return True

        # Check if timeout has passed
        if self._circuit_opened_at:
            elapsed = (
                datetime.now(timezone.utc) - self._circuit_opened_at
            ).total_seconds()
            if elapsed > self.config.circuit_breaker_timeout:
                self._circuit_open = False
                self._failure_count = 0
                logger.info("Circuit breaker closed")
                return True

        return False

    def _record_failure(self) -> None:
        """Record a failure for circuit breaker."""
        self._failure_count += 1
        if self._failure_count >= self.config.circuit_breaker_threshold:
            self._circuit_open = True
            self._circuit_opened_at = datetime.now(timezone.utc)
            logger.warning("Circuit breaker opened")

    def _record_success(self) -> None:
        """Record a success for circuit breaker."""
        self._failure_count = 0

    async def execute_distributed(
        self,
        operation: Callable[[], T],
        timeout_seconds: float,
        operation_id: str = "",
    ) -> DistributedExecutionResult:
        """Execute an operation with distributed timeout coordination.

        Args:
            operation: Operation to execute
            timeout_seconds: Timeout in seconds
            operation_id: Operation identifier

        Returns:
            DistributedExecutionResult
        """
        # Check circuit breaker
        if not self._check_circuit():
            return DistributedExecutionResult.failure(
                "Circuit breaker open",
                node_id=self.config.node_id,
            )

        if not self._coordinator:
            return DistributedExecutionResult.failure(
                "Manager not started",
                node_id=self.config.node_id,
            )

        # Create distributed deadline
        deadline = await self._coordinator.create_deadline(
            self.config.node_id,
            timeout_seconds,
            operation_id,
        )

        start_time = time.time()

        try:
            # Execute with local timeout
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(None, operation),
                timeout=timeout_seconds,
            )

            # Mark deadline complete
            await self._coordinator.complete_deadline(deadline.deadline_id)
            self._record_success()

            return DistributedExecutionResult.ok(
                result,
                deadline=deadline,
                node_id=self.config.node_id,
            )

        except asyncio.TimeoutError:
            await self._coordinator.complete_deadline(
                deadline.deadline_id,
                status="timeout",
            )
            self._record_failure()

            return DistributedExecutionResult.failure(
                f"Operation timed out after {timeout_seconds}s",
                deadline=deadline,
                node_id=self.config.node_id,
            )

        except Exception as e:
            await self._coordinator.complete_deadline(
                deadline.deadline_id,
                status="error",
            )
            self._record_failure()

            return DistributedExecutionResult.failure(
                str(e),
                deadline=deadline,
                node_id=self.config.node_id,
            )

    async def get_active_nodes(self) -> list[str]:
        """Get list of active nodes.

        Returns:
            List of active node IDs
        """
        if self._coordinator:
            return await self._coordinator.get_active_nodes()
        return [self.config.node_id]

    def get_node_id(self) -> str:
        """Get this node's ID.

        Returns:
            Node ID
        """
        return self.config.node_id

    def is_circuit_open(self) -> bool:
        """Check if circuit breaker is open.

        Returns:
            True if circuit is open
        """
        return self._circuit_open


# Convenience function for simple distributed execution
async def execute_with_distributed_timeout(
    operation: Callable[[], T],
    timeout_seconds: float,
    config: DistributedTimeoutConfig | None = None,
) -> DistributedExecutionResult:
    """Execute an operation with distributed timeout.

    Args:
        operation: Operation to execute
        timeout_seconds: Timeout in seconds
        config: Optional configuration

    Returns:
        DistributedExecutionResult
    """
    manager = DistributedTimeoutManager(config)
    async with manager:
        return await manager.execute_distributed(
            operation,
            timeout_seconds,
        )
