"""Saga State Machine Module.

This module provides a state machine implementation for managing
saga execution lifecycle with well-defined states and transitions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Callable
from uuid import uuid4

if TYPE_CHECKING:
    from truthound.checkpoint.transaction.saga.definition import SagaDefinition


logger = logging.getLogger(__name__)


class SagaState(str, Enum):
    """States in the saga lifecycle."""

    # Initial states
    CREATED = "created"  # Saga created but not started
    PENDING = "pending"  # Waiting to start

    # Execution states
    STARTING = "starting"  # Beginning execution
    EXECUTING = "executing"  # Executing steps
    STEP_EXECUTING = "step_executing"  # Executing a specific step
    STEP_COMPLETED = "step_completed"  # Step completed successfully
    STEP_FAILED = "step_failed"  # Step failed

    # Compensation states
    COMPENSATING = "compensating"  # Running compensations
    STEP_COMPENSATING = "step_compensating"  # Compensating a specific step
    STEP_COMPENSATED = "step_compensated"  # Step compensated successfully
    COMPENSATION_FAILED = "compensation_failed"  # Compensation failed

    # Terminal states
    COMPLETED = "completed"  # All steps completed successfully
    COMPENSATED = "compensated"  # All compensations completed
    FAILED = "failed"  # Saga failed (unrecoverable)
    ABORTED = "aborted"  # Saga aborted by user/system
    TIMED_OUT = "timed_out"  # Saga timed out

    # Recovery states
    SUSPENDED = "suspended"  # Execution suspended
    RESUMING = "resuming"  # Resuming from suspension

    def __str__(self) -> str:
        return self.value

    @property
    def is_terminal(self) -> bool:
        """Check if this is a terminal state."""
        return self in (
            SagaState.COMPLETED,
            SagaState.COMPENSATED,
            SagaState.FAILED,
            SagaState.ABORTED,
            SagaState.TIMED_OUT,
        )

    @property
    def is_executing(self) -> bool:
        """Check if saga is currently executing."""
        return self in (
            SagaState.STARTING,
            SagaState.EXECUTING,
            SagaState.STEP_EXECUTING,
            SagaState.STEP_COMPLETED,
        )

    @property
    def is_compensating(self) -> bool:
        """Check if saga is compensating."""
        return self in (
            SagaState.COMPENSATING,
            SagaState.STEP_COMPENSATING,
            SagaState.STEP_COMPENSATED,
        )

    @property
    def is_success(self) -> bool:
        """Check if saga completed successfully."""
        return self == SagaState.COMPLETED

    @property
    def is_recoverable(self) -> bool:
        """Check if saga can be recovered."""
        return self in (
            SagaState.STEP_FAILED,
            SagaState.COMPENSATION_FAILED,
            SagaState.SUSPENDED,
        )


class SagaEventType(str, Enum):
    """Types of events in saga execution."""

    # Lifecycle events
    SAGA_CREATED = "saga_created"
    SAGA_STARTED = "saga_started"
    SAGA_COMPLETED = "saga_completed"
    SAGA_FAILED = "saga_failed"
    SAGA_ABORTED = "saga_aborted"
    SAGA_TIMED_OUT = "saga_timed_out"

    # Step events
    STEP_STARTED = "step_started"
    STEP_COMPLETED = "step_completed"
    STEP_FAILED = "step_failed"
    STEP_SKIPPED = "step_skipped"
    STEP_RETRYING = "step_retrying"

    # Compensation events
    COMPENSATION_STARTED = "compensation_started"
    STEP_COMPENSATING = "step_compensating"
    STEP_COMPENSATED = "step_compensated"
    COMPENSATION_FAILED = "compensation_failed"
    COMPENSATION_COMPLETED = "compensation_completed"

    # Recovery events
    SAGA_SUSPENDED = "saga_suspended"
    SAGA_RESUMED = "saga_resumed"
    SAGA_RECOVERED = "saga_recovered"

    # Checkpoint events
    CHECKPOINT_CREATED = "checkpoint_created"
    CHECKPOINT_RESTORED = "checkpoint_restored"

    def __str__(self) -> str:
        return self.value


@dataclass
class SagaEvent:
    """Event representing a saga state change.

    Attributes:
        event_id: Unique event identifier.
        saga_id: ID of the saga this event belongs to.
        event_type: Type of event.
        timestamp: When the event occurred.
        source_state: State before the event.
        target_state: State after the event.
        step_id: ID of the step (if step-related event).
        data: Additional event data.
        error: Error information if applicable.
    """

    event_id: str = field(default_factory=lambda: f"evt_{uuid4().hex[:12]}")
    saga_id: str = ""
    event_type: SagaEventType = SagaEventType.SAGA_CREATED
    timestamp: datetime = field(default_factory=datetime.now)
    source_state: SagaState | None = None
    target_state: SagaState | None = None
    step_id: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "saga_id": self.saga_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "source_state": self.source_state.value if self.source_state else None,
            "target_state": self.target_state.value if self.target_state else None,
            "step_id": self.step_id,
            "data": self.data,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SagaEvent":
        """Create from dictionary."""
        return cls(
            event_id=data.get("event_id", ""),
            saga_id=data.get("saga_id", ""),
            event_type=SagaEventType(data["event_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            source_state=SagaState(data["source_state"]) if data.get("source_state") else None,
            target_state=SagaState(data["target_state"]) if data.get("target_state") else None,
            step_id=data.get("step_id"),
            data=data.get("data", {}),
            error=data.get("error"),
        )


@dataclass
class SagaTransition:
    """Represents a valid state transition.

    Attributes:
        from_state: Source state.
        to_state: Target state.
        event_type: Event that triggers this transition.
        guard: Optional condition for the transition.
        action: Optional action to execute during transition.
    """

    from_state: SagaState | tuple[SagaState, ...]
    to_state: SagaState
    event_type: SagaEventType
    guard: Callable[[SagaEvent], bool] | None = None
    action: Callable[[SagaEvent], None] | None = None

    def is_valid_from(self, state: SagaState) -> bool:
        """Check if transition is valid from given state."""
        if isinstance(self.from_state, tuple):
            return state in self.from_state
        return state == self.from_state

    def can_transition(self, event: SagaEvent) -> bool:
        """Check if transition is allowed for the event."""
        if self.guard:
            return self.guard(event)
        return True


class SagaStateMachine:
    """State machine for managing saga execution lifecycle.

    This class manages the state transitions of a saga, ensuring that
    only valid transitions occur and triggering appropriate callbacks.

    Example:
        >>> machine = SagaStateMachine(saga_id="saga_123")
        >>> machine.on_state_change(lambda old, new, evt: print(f"{old} -> {new}"))
        >>> machine.start()
        >>> machine.step_started("step_1")
        >>> machine.step_completed("step_1")
        >>> machine.complete()
    """

    # Valid transitions definition
    TRANSITIONS: list[SagaTransition] = [
        # Start transitions
        SagaTransition(SagaState.CREATED, SagaState.STARTING, SagaEventType.SAGA_STARTED),
        SagaTransition(SagaState.PENDING, SagaState.STARTING, SagaEventType.SAGA_STARTED),

        # Step execution transitions - STARTING goes directly to STEP_EXECUTING
        SagaTransition(SagaState.STARTING, SagaState.STEP_EXECUTING, SagaEventType.STEP_STARTED),
        SagaTransition(SagaState.EXECUTING, SagaState.STEP_EXECUTING, SagaEventType.STEP_STARTED),
        SagaTransition(SagaState.STEP_COMPLETED, SagaState.STEP_EXECUTING, SagaEventType.STEP_STARTED),
        # Allow parallel step starts (multiple steps starting from STEP_EXECUTING)
        SagaTransition(SagaState.STEP_EXECUTING, SagaState.STEP_EXECUTING, SagaEventType.STEP_STARTED),
        SagaTransition(SagaState.STEP_EXECUTING, SagaState.STEP_COMPLETED, SagaEventType.STEP_COMPLETED),
        SagaTransition(SagaState.STEP_EXECUTING, SagaState.STEP_FAILED, SagaEventType.STEP_FAILED),
        SagaTransition(SagaState.STEP_EXECUTING, SagaState.STEP_EXECUTING, SagaEventType.STEP_RETRYING),
        # Allow parallel step completions (multiple results coming back)
        SagaTransition(SagaState.STEP_COMPLETED, SagaState.STEP_COMPLETED, SagaEventType.STEP_COMPLETED),
        SagaTransition(SagaState.STEP_COMPLETED, SagaState.STEP_FAILED, SagaEventType.STEP_FAILED),

        # Completion transitions
        SagaTransition(SagaState.STEP_COMPLETED, SagaState.COMPLETED, SagaEventType.SAGA_COMPLETED),
        SagaTransition(SagaState.EXECUTING, SagaState.COMPLETED, SagaEventType.SAGA_COMPLETED),
        SagaTransition(SagaState.STARTING, SagaState.COMPLETED, SagaEventType.SAGA_COMPLETED),

        # Failure and compensation transitions
        SagaTransition(SagaState.STEP_FAILED, SagaState.COMPENSATING, SagaEventType.COMPENSATION_STARTED),
        SagaTransition(SagaState.COMPENSATING, SagaState.STEP_COMPENSATING, SagaEventType.STEP_COMPENSATING),
        SagaTransition(SagaState.STEP_COMPENSATED, SagaState.STEP_COMPENSATING, SagaEventType.STEP_COMPENSATING),
        SagaTransition(SagaState.STEP_COMPENSATING, SagaState.STEP_COMPENSATED, SagaEventType.STEP_COMPENSATED),
        SagaTransition(SagaState.STEP_COMPENSATING, SagaState.COMPENSATION_FAILED, SagaEventType.COMPENSATION_FAILED),

        # Compensation completion
        SagaTransition(SagaState.STEP_COMPENSATED, SagaState.COMPENSATED, SagaEventType.COMPENSATION_COMPLETED),
        SagaTransition(SagaState.COMPENSATING, SagaState.COMPENSATED, SagaEventType.COMPENSATION_COMPLETED),

        # Failure transitions
        SagaTransition(
            (SagaState.STEP_FAILED, SagaState.COMPENSATION_FAILED),
            SagaState.FAILED,
            SagaEventType.SAGA_FAILED,
        ),

        # Abort transitions (can abort from most states)
        SagaTransition(
            (
                SagaState.CREATED, SagaState.PENDING, SagaState.STARTING,
                SagaState.EXECUTING, SagaState.STEP_EXECUTING, SagaState.STEP_COMPLETED,
                SagaState.STEP_FAILED, SagaState.COMPENSATING, SagaState.STEP_COMPENSATING,
            ),
            SagaState.ABORTED,
            SagaEventType.SAGA_ABORTED,
        ),

        # Timeout transitions
        SagaTransition(
            (
                SagaState.STARTING, SagaState.EXECUTING, SagaState.STEP_EXECUTING,
                SagaState.COMPENSATING, SagaState.STEP_COMPENSATING,
            ),
            SagaState.TIMED_OUT,
            SagaEventType.SAGA_TIMED_OUT,
        ),

        # Suspend and resume transitions
        SagaTransition(
            (SagaState.EXECUTING, SagaState.STEP_EXECUTING, SagaState.STEP_COMPLETED,
             SagaState.STEP_FAILED, SagaState.COMPENSATING, SagaState.STEP_COMPENSATING,
             SagaState.STARTING),
            SagaState.SUSPENDED,
            SagaEventType.SAGA_SUSPENDED,
        ),
        SagaTransition(SagaState.SUSPENDED, SagaState.RESUMING, SagaEventType.SAGA_RESUMED),
        SagaTransition(SagaState.RESUMING, SagaState.STEP_EXECUTING, SagaEventType.STEP_STARTED),
        SagaTransition(SagaState.RESUMING, SagaState.COMPENSATING, SagaEventType.COMPENSATION_STARTED),
    ]

    def __init__(
        self,
        saga_id: str,
        initial_state: SagaState = SagaState.CREATED,
    ) -> None:
        """Initialize the state machine.

        Args:
            saga_id: Unique saga identifier.
            initial_state: Initial state.
        """
        self._saga_id = saga_id
        self._state = initial_state
        self._events: list[SagaEvent] = []
        self._callbacks: list[Callable[[SagaState, SagaState, SagaEvent], None]] = []
        self._current_step: str | None = None
        self._step_states: dict[str, SagaState] = {}
        self._metadata: dict[str, Any] = {}

    @property
    def saga_id(self) -> str:
        """Get saga ID."""
        return self._saga_id

    @property
    def state(self) -> SagaState:
        """Get current state."""
        return self._state

    @property
    def current_step(self) -> str | None:
        """Get current step ID."""
        return self._current_step

    @property
    def events(self) -> list[SagaEvent]:
        """Get all events."""
        return list(self._events)

    @property
    def is_terminal(self) -> bool:
        """Check if saga is in terminal state."""
        return self._state.is_terminal

    def on_state_change(
        self,
        callback: Callable[[SagaState, SagaState, SagaEvent], None],
    ) -> None:
        """Register a state change callback.

        Args:
            callback: Function to call on state change.
        """
        self._callbacks.append(callback)

    def _find_transition(
        self,
        event_type: SagaEventType,
        event: SagaEvent,
    ) -> SagaTransition | None:
        """Find a valid transition for the event."""
        for transition in self.TRANSITIONS:
            if transition.event_type != event_type:
                continue
            if not transition.is_valid_from(self._state):
                continue
            if not transition.can_transition(event):
                continue
            return transition
        return None

    def _transition(
        self,
        event_type: SagaEventType,
        step_id: str | None = None,
        data: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> SagaEvent:
        """Execute a state transition.

        Args:
            event_type: Type of event.
            step_id: Optional step ID.
            data: Optional event data.
            error: Optional error message.

        Returns:
            The created event.

        Raises:
            InvalidTransitionError: If transition is not valid.
        """
        event = SagaEvent(
            saga_id=self._saga_id,
            event_type=event_type,
            source_state=self._state,
            step_id=step_id,
            data=data or {},
            error=error,
        )

        transition = self._find_transition(event_type, event)
        if transition is None:
            raise InvalidTransitionError(
                f"No valid transition from {self._state} with event {event_type}"
            )

        # Execute transition action if defined
        if transition.action:
            transition.action(event)

        # Update state
        old_state = self._state
        self._state = transition.to_state
        event.target_state = self._state

        # Update step tracking
        if step_id:
            self._current_step = step_id
            self._step_states[step_id] = self._state

        # Record event
        self._events.append(event)

        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(old_state, self._state, event)
            except Exception as e:
                logger.warning(f"State change callback failed: {e}")

        logger.debug(
            f"Saga {self._saga_id}: {old_state.value} -> {self._state.value} "
            f"({event_type.value})"
        )

        return event

    # ==========================================================================
    # Public transition methods
    # ==========================================================================

    def start(self, data: dict[str, Any] | None = None) -> SagaEvent:
        """Start the saga execution.

        Args:
            data: Optional start data.

        Returns:
            The start event.
        """
        return self._transition(SagaEventType.SAGA_STARTED, data=data)

    def step_started(
        self,
        step_id: str,
        data: dict[str, Any] | None = None,
    ) -> SagaEvent:
        """Mark a step as started.

        Args:
            step_id: Step identifier.
            data: Optional step data.

        Returns:
            The step start event.
        """
        return self._transition(SagaEventType.STEP_STARTED, step_id=step_id, data=data)

    def step_completed(
        self,
        step_id: str,
        data: dict[str, Any] | None = None,
    ) -> SagaEvent:
        """Mark a step as completed.

        Args:
            step_id: Step identifier.
            data: Optional completion data.

        Returns:
            The step completion event.
        """
        return self._transition(SagaEventType.STEP_COMPLETED, step_id=step_id, data=data)

    def step_failed(
        self,
        step_id: str,
        error: str,
        data: dict[str, Any] | None = None,
    ) -> SagaEvent:
        """Mark a step as failed.

        Args:
            step_id: Step identifier.
            error: Error message.
            data: Optional failure data.

        Returns:
            The step failure event.
        """
        return self._transition(
            SagaEventType.STEP_FAILED,
            step_id=step_id,
            data=data,
            error=error,
        )

    def step_retrying(
        self,
        step_id: str,
        attempt: int,
        data: dict[str, Any] | None = None,
    ) -> SagaEvent:
        """Mark a step as retrying.

        Args:
            step_id: Step identifier.
            attempt: Retry attempt number.
            data: Optional retry data.

        Returns:
            The retry event.
        """
        event_data = data or {}
        event_data["attempt"] = attempt
        return self._transition(
            SagaEventType.STEP_RETRYING,
            step_id=step_id,
            data=event_data,
        )

    def complete(self, data: dict[str, Any] | None = None) -> SagaEvent:
        """Mark the saga as completed.

        Args:
            data: Optional completion data.

        Returns:
            The completion event.
        """
        return self._transition(SagaEventType.SAGA_COMPLETED, data=data)

    def start_compensation(self, data: dict[str, Any] | None = None) -> SagaEvent:
        """Start compensation process.

        Args:
            data: Optional compensation data.

        Returns:
            The compensation start event.
        """
        return self._transition(SagaEventType.COMPENSATION_STARTED, data=data)

    def step_compensating(
        self,
        step_id: str,
        data: dict[str, Any] | None = None,
    ) -> SagaEvent:
        """Mark a step as being compensated.

        Args:
            step_id: Step identifier.
            data: Optional compensation data.

        Returns:
            The step compensating event.
        """
        return self._transition(
            SagaEventType.STEP_COMPENSATING,
            step_id=step_id,
            data=data,
        )

    def step_compensated(
        self,
        step_id: str,
        data: dict[str, Any] | None = None,
    ) -> SagaEvent:
        """Mark a step compensation as completed.

        Args:
            step_id: Step identifier.
            data: Optional completion data.

        Returns:
            The step compensated event.
        """
        return self._transition(
            SagaEventType.STEP_COMPENSATED,
            step_id=step_id,
            data=data,
        )

    def compensation_failed(
        self,
        step_id: str,
        error: str,
        data: dict[str, Any] | None = None,
    ) -> SagaEvent:
        """Mark compensation as failed.

        Args:
            step_id: Step identifier.
            error: Error message.
            data: Optional failure data.

        Returns:
            The compensation failure event.
        """
        return self._transition(
            SagaEventType.COMPENSATION_FAILED,
            step_id=step_id,
            data=data,
            error=error,
        )

    def compensation_complete(self, data: dict[str, Any] | None = None) -> SagaEvent:
        """Mark all compensations as complete.

        Args:
            data: Optional completion data.

        Returns:
            The compensation completion event.
        """
        return self._transition(SagaEventType.COMPENSATION_COMPLETED, data=data)

    def fail(self, error: str, data: dict[str, Any] | None = None) -> SagaEvent:
        """Mark the saga as failed.

        Args:
            error: Error message.
            data: Optional failure data.

        Returns:
            The failure event.
        """
        return self._transition(SagaEventType.SAGA_FAILED, data=data, error=error)

    def abort(self, reason: str = "", data: dict[str, Any] | None = None) -> SagaEvent:
        """Abort the saga.

        Args:
            reason: Abort reason.
            data: Optional abort data.

        Returns:
            The abort event.
        """
        return self._transition(
            SagaEventType.SAGA_ABORTED,
            data=data,
            error=reason or "Saga aborted",
        )

    def timeout(self, data: dict[str, Any] | None = None) -> SagaEvent:
        """Mark the saga as timed out.

        Args:
            data: Optional timeout data.

        Returns:
            The timeout event.
        """
        return self._transition(
            SagaEventType.SAGA_TIMED_OUT,
            data=data,
            error="Saga timed out",
        )

    def suspend(self, reason: str = "", data: dict[str, Any] | None = None) -> SagaEvent:
        """Suspend the saga.

        Args:
            reason: Suspension reason.
            data: Optional suspension data.

        Returns:
            The suspension event.
        """
        event_data = data or {}
        event_data["reason"] = reason
        return self._transition(SagaEventType.SAGA_SUSPENDED, data=event_data)

    def resume(self, data: dict[str, Any] | None = None) -> SagaEvent:
        """Resume a suspended saga.

        Args:
            data: Optional resume data.

        Returns:
            The resume event.
        """
        return self._transition(SagaEventType.SAGA_RESUMED, data=data)

    # ==========================================================================
    # Query methods
    # ==========================================================================

    def get_step_state(self, step_id: str) -> SagaState | None:
        """Get the state of a specific step.

        Args:
            step_id: Step identifier.

        Returns:
            Step state or None if not tracked.
        """
        return self._step_states.get(step_id)

    def get_completed_steps(self) -> list[str]:
        """Get IDs of all completed steps."""
        return [
            step_id
            for step_id, state in self._step_states.items()
            if state == SagaState.STEP_COMPLETED
        ]

    def get_failed_steps(self) -> list[str]:
        """Get IDs of all failed steps."""
        return [
            step_id
            for step_id, state in self._step_states.items()
            if state == SagaState.STEP_FAILED
        ]

    def get_compensated_steps(self) -> list[str]:
        """Get IDs of all compensated steps."""
        return [
            step_id
            for step_id, state in self._step_states.items()
            if state == SagaState.STEP_COMPENSATED
        ]

    def to_dict(self) -> dict[str, Any]:
        """Convert state machine to dictionary for serialization."""
        return {
            "saga_id": self._saga_id,
            "state": self._state.value,
            "current_step": self._current_step,
            "step_states": {k: v.value for k, v in self._step_states.items()},
            "event_count": len(self._events),
            "metadata": self._metadata,
        }

    @classmethod
    def from_events(cls, saga_id: str, events: list[SagaEvent]) -> "SagaStateMachine":
        """Reconstruct state machine from events.

        Args:
            saga_id: Saga identifier.
            events: List of events to replay.

        Returns:
            Reconstructed state machine.
        """
        machine = cls(saga_id)

        for event in events:
            if event.target_state:
                machine._state = event.target_state
            if event.step_id:
                machine._current_step = event.step_id
                machine._step_states[event.step_id] = (
                    event.target_state or machine._state
                )
            machine._events.append(event)

        return machine


class InvalidTransitionError(Exception):
    """Raised when an invalid state transition is attempted."""

    pass
