"""Base types and classes for transaction management.

This module defines the core types, enums, and data structures used
throughout the transaction framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from truthound.checkpoint.actions.base import ActionResult


class TransactionState(str, Enum):
    """State of a transaction."""

    PENDING = "pending"          # Not yet started
    ACTIVE = "active"            # Currently executing
    COMMITTED = "committed"      # Successfully completed
    ROLLING_BACK = "rolling_back"  # Compensation in progress
    ROLLED_BACK = "rolled_back"  # Successfully rolled back
    FAILED = "failed"            # Failed (may be partially complete)
    COMPENSATED = "compensated"  # All compensations executed

    def __str__(self) -> str:
        return self.value

    @property
    def is_terminal(self) -> bool:
        """Check if this is a terminal state."""
        return self in (
            TransactionState.COMMITTED,
            TransactionState.ROLLED_BACK,
            TransactionState.FAILED,
            TransactionState.COMPENSATED,
        )

    @property
    def is_success(self) -> bool:
        """Check if transaction completed successfully."""
        return self == TransactionState.COMMITTED


class TransactionPhase(str, Enum):
    """Phase of transaction execution."""

    PREPARE = "prepare"          # Pre-execution checks
    EXECUTE = "execute"          # Main execution
    VERIFY = "verify"            # Post-execution verification
    COMPENSATE = "compensate"    # Rollback/compensation
    FINALIZE = "finalize"        # Cleanup

    def __str__(self) -> str:
        return self.value


class CompensationStrategy(str, Enum):
    """Strategy for handling compensation/rollback."""

    BACKWARD = "backward"        # Reverse order (Saga pattern)
    FORWARD = "forward"          # Forward recovery
    PARALLEL = "parallel"        # Parallel compensation
    CUSTOM = "custom"            # Custom orchestration
    NONE = "none"                # No compensation

    def __str__(self) -> str:
        return self.value


class IsolationLevel(str, Enum):
    """Transaction isolation level."""

    NONE = "none"                # No isolation
    READ_UNCOMMITTED = "read_uncommitted"
    READ_COMMITTED = "read_committed"
    REPEATABLE_READ = "repeatable_read"
    SERIALIZABLE = "serializable"

    def __str__(self) -> str:
        return self.value


@dataclass
class TransactionConfig:
    """Configuration for transaction behavior.

    Attributes:
        enabled: Whether transactions are enabled.
        rollback_on_failure: Automatically rollback on action failure.
        compensation_strategy: How to execute compensations.
        max_compensation_retries: Max retries for compensation.
        compensation_timeout: Timeout for each compensation.
        savepoint_enabled: Enable savepoints for partial rollback.
        isolation_level: Transaction isolation level.
        idempotency_enabled: Enable idempotency checks.
        idempotency_ttl_seconds: TTL for idempotency keys.
        audit_enabled: Log transaction events for audit.
        continue_on_compensation_failure: Continue if compensation fails.
    """

    enabled: bool = True
    rollback_on_failure: bool = True
    compensation_strategy: CompensationStrategy = CompensationStrategy.BACKWARD
    max_compensation_retries: int = 3
    compensation_timeout: float = 30.0
    savepoint_enabled: bool = True
    isolation_level: IsolationLevel = IsolationLevel.NONE
    idempotency_enabled: bool = False
    idempotency_ttl_seconds: int = 3600
    audit_enabled: bool = True
    continue_on_compensation_failure: bool = False

    def __post_init__(self) -> None:
        """Convert string enums."""
        if isinstance(self.compensation_strategy, str):
            self.compensation_strategy = CompensationStrategy(self.compensation_strategy)
        if isinstance(self.isolation_level, str):
            self.isolation_level = IsolationLevel(self.isolation_level)


@dataclass
class Savepoint:
    """Represents a transaction savepoint.

    Savepoints allow partial rollback to a specific point in the
    transaction instead of full rollback.

    Attributes:
        id: Unique savepoint identifier.
        name: Human-readable name.
        created_at: When savepoint was created.
        action_index: Index of action after which savepoint was created.
        state_snapshot: Captured state at savepoint time.
        metadata: Additional savepoint metadata.
    """

    id: str = field(default_factory=lambda: uuid4().hex[:12])
    name: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    action_index: int = 0
    state_snapshot: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"savepoint_{self.id}"


@dataclass
class TransactionContext:
    """Context passed through transaction execution.

    This context carries transaction state and provides methods for
    transaction control (savepoints, rollback requests, etc.).

    Attributes:
        transaction_id: Unique identifier for this transaction.
        started_at: When the transaction started.
        state: Current transaction state.
        phase: Current execution phase.
        savepoints: List of created savepoints.
        completed_actions: Actions that have completed.
        failed_action: The action that caused failure (if any).
        rollback_requested: Whether rollback has been requested.
        idempotency_key: Key for idempotency checks.
        metadata: Additional context data.
    """

    transaction_id: str = field(default_factory=lambda: f"txn_{uuid4().hex[:16]}")
    started_at: datetime = field(default_factory=datetime.now)
    state: TransactionState = TransactionState.PENDING
    phase: TransactionPhase = TransactionPhase.PREPARE
    savepoints: list[Savepoint] = field(default_factory=list)
    completed_actions: list[str] = field(default_factory=list)
    compensated_actions: list[str] = field(default_factory=list)
    failed_action: str | None = None
    rollback_requested: bool = False
    idempotency_key: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def create_savepoint(self, name: str = "", state_snapshot: dict | None = None) -> Savepoint:
        """Create a new savepoint.

        Args:
            name: Optional name for the savepoint.
            state_snapshot: State to capture.

        Returns:
            The created savepoint.
        """
        savepoint = Savepoint(
            name=name,
            action_index=len(self.completed_actions),
            state_snapshot=state_snapshot or {},
        )
        self.savepoints.append(savepoint)
        return savepoint

    def get_savepoint(self, name_or_id: str) -> Savepoint | None:
        """Get a savepoint by name or ID.

        Args:
            name_or_id: Savepoint name or ID.

        Returns:
            The savepoint, or None if not found.
        """
        for sp in self.savepoints:
            if sp.name == name_or_id or sp.id == name_or_id:
                return sp
        return None

    def request_rollback(self, reason: str = "") -> None:
        """Request a transaction rollback.

        Args:
            reason: Reason for rollback request.
        """
        self.rollback_requested = True
        self.metadata["rollback_reason"] = reason

    def mark_action_completed(self, action_name: str) -> None:
        """Mark an action as completed.

        Args:
            action_name: Name of the completed action.
        """
        if action_name not in self.completed_actions:
            self.completed_actions.append(action_name)

    def mark_action_compensated(self, action_name: str) -> None:
        """Mark an action as compensated.

        Args:
            action_name: Name of the compensated action.
        """
        if action_name not in self.compensated_actions:
            self.compensated_actions.append(action_name)

    @property
    def duration_ms(self) -> float:
        """Get transaction duration in milliseconds."""
        return (datetime.now() - self.started_at).total_seconds() * 1000

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "transaction_id": self.transaction_id,
            "started_at": self.started_at.isoformat(),
            "state": self.state.value,
            "phase": self.phase.value,
            "completed_actions": self.completed_actions,
            "compensated_actions": self.compensated_actions,
            "failed_action": self.failed_action,
            "rollback_requested": self.rollback_requested,
            "idempotency_key": self.idempotency_key,
            "duration_ms": self.duration_ms,
            "savepoint_count": len(self.savepoints),
            "metadata": self.metadata,
        }


@dataclass
class CompensationResult:
    """Result of a compensation (rollback) execution.

    Attributes:
        action_name: Name of the action being compensated.
        success: Whether compensation succeeded.
        started_at: When compensation started.
        completed_at: When compensation completed.
        duration_ms: Compensation duration.
        error: Error message if failed.
        details: Additional details.
    """

    action_name: str
    success: bool
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None
    duration_ms: float = 0.0
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "action_name": self.action_name,
            "success": self.success,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "details": self.details,
        }


@dataclass
class TransactionResult:
    """Complete result of a transaction.

    Attributes:
        transaction_id: Unique transaction identifier.
        state: Final transaction state.
        started_at: When transaction started.
        completed_at: When transaction completed.
        duration_ms: Total duration.
        action_results: Results from each action.
        compensation_results: Results from compensations.
        savepoints_used: Savepoints that were used.
        error: Error message if failed.
        metadata: Additional result metadata.
    """

    transaction_id: str
    state: TransactionState
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None
    duration_ms: float = 0.0
    action_results: list["ActionResult"] = field(default_factory=list)
    compensation_results: list[CompensationResult] = field(default_factory=list)
    savepoints_used: list[str] = field(default_factory=list)
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if transaction was successful."""
        return self.state.is_success

    @property
    def fully_compensated(self) -> bool:
        """Check if all compensations succeeded."""
        return (
            self.state == TransactionState.COMPENSATED
            and all(cr.success for cr in self.compensation_results)
        )

    @property
    def actions_completed(self) -> int:
        """Count of completed actions."""
        return len([r for r in self.action_results if r.success])

    @property
    def actions_failed(self) -> int:
        """Count of failed actions."""
        return len([r for r in self.action_results if not r.success])

    @property
    def compensations_completed(self) -> int:
        """Count of successful compensations."""
        return len([r for r in self.compensation_results if r.success])

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "transaction_id": self.transaction_id,
            "state": self.state.value,
            "success": self.success,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "actions_completed": self.actions_completed,
            "actions_failed": self.actions_failed,
            "compensations_completed": self.compensations_completed,
            "action_results": [r.to_dict() for r in self.action_results],
            "compensation_results": [r.to_dict() for r in self.compensation_results],
            "error": self.error,
            "metadata": self.metadata,
        }

    def summary(self) -> str:
        """Get a human-readable summary."""
        lines = [
            f"Transaction: {self.transaction_id}",
            f"State: {self.state.value}",
            f"Duration: {self.duration_ms:.1f}ms",
            f"Actions: {self.actions_completed} completed, {self.actions_failed} failed",
        ]
        if self.compensation_results:
            lines.append(f"Compensations: {self.compensations_completed} completed")
        if self.error:
            lines.append(f"Error: {self.error}")
        return "\n".join(lines)
