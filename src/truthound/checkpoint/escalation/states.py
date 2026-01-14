"""Escalation State Machine.

This module implements a finite state machine for managing
escalation lifecycle and transitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable


class EscalationState(str, Enum):
    """Possible states in the escalation lifecycle.

    State Diagram:
        PENDING → ACTIVE → ESCALATING → (loop to ACTIVE)
                    ↓           ↓
               ACKNOWLEDGED  ACKNOWLEDGED
                    ↓           ↓
                 RESOLVED    RESOLVED

        Any state can transition to: CANCELLED, TIMED_OUT, FAILED
    """

    PENDING = "pending"  # Initial state, waiting to start
    ACTIVE = "active"  # Actively notifying current level
    ESCALATING = "escalating"  # In process of escalating to next level
    ACKNOWLEDGED = "acknowledged"  # Acknowledged by responder
    RESOLVED = "resolved"  # Issue resolved
    CANCELLED = "cancelled"  # Manually cancelled
    TIMED_OUT = "timed_out"  # Max escalations reached or timeout
    FAILED = "failed"  # System error during escalation

    @property
    def is_terminal(self) -> bool:
        """Check if this is a terminal state."""
        return self in (
            EscalationState.RESOLVED,
            EscalationState.CANCELLED,
            EscalationState.TIMED_OUT,
            EscalationState.FAILED,
        )

    @property
    def is_active(self) -> bool:
        """Check if escalation is still active."""
        return self in (
            EscalationState.PENDING,
            EscalationState.ACTIVE,
            EscalationState.ESCALATING,
        )

    @property
    def allows_acknowledgment(self) -> bool:
        """Check if state allows acknowledgment."""
        return self in (
            EscalationState.ACTIVE,
            EscalationState.ESCALATING,
        )

    @property
    def allows_resolution(self) -> bool:
        """Check if state allows resolution."""
        return self in (
            EscalationState.ACTIVE,
            EscalationState.ESCALATING,
            EscalationState.ACKNOWLEDGED,
        )


class EscalationEvent(str, Enum):
    """Events that trigger state transitions."""

    START = "start"  # Start escalation process
    NOTIFY = "notify"  # Send notification to current level
    NOTIFY_SUCCESS = "notify_success"  # Notification sent successfully
    NOTIFY_FAILURE = "notify_failure"  # Notification failed
    ESCALATE = "escalate"  # Escalate to next level
    ESCALATE_SUCCESS = "escalate_success"  # Escalation completed
    ACKNOWLEDGE = "acknowledge"  # Responder acknowledged
    RESOLVE = "resolve"  # Issue resolved
    CANCEL = "cancel"  # Manual cancellation
    TIMEOUT = "timeout"  # Timeout reached
    RETRY = "retry"  # Retry current operation
    MAX_LEVEL_REACHED = "max_level_reached"  # No more levels to escalate
    ERROR = "error"  # System error


@dataclass
class StateTransition:
    """Record of a state transition.

    Attributes:
        from_state: Previous state.
        to_state: New state.
        event: Event that triggered transition.
        timestamp: When transition occurred.
        actor: Who/what triggered the transition.
        details: Additional transition details.
    """

    from_state: EscalationState
    to_state: EscalationState
    event: EscalationEvent
    timestamp: datetime = field(default_factory=datetime.now)
    actor: str = "system"
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "from_state": self.from_state.value,
            "to_state": self.to_state.value,
            "event": self.event.value,
            "timestamp": self.timestamp.isoformat(),
            "actor": self.actor,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StateTransition:
        """Create from dictionary."""
        return cls(
            from_state=EscalationState(data["from_state"]),
            to_state=EscalationState(data["to_state"]),
            event=EscalationEvent(data["event"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            actor=data.get("actor", "system"),
            details=data.get("details", {}),
        )


# Type alias for transition guard functions
TransitionGuard = Callable[[EscalationState, EscalationEvent, dict[str, Any]], bool]

# Type alias for transition action functions
TransitionAction = Callable[[StateTransition, dict[str, Any]], None]


class EscalationStateMachine:
    """Finite state machine for escalation lifecycle management.

    This class manages valid state transitions and ensures
    escalation records follow the correct lifecycle.

    Example:
        >>> machine = EscalationStateMachine()
        >>> machine.can_transition(EscalationState.PENDING, EscalationEvent.START)
        True
        >>> transition = machine.transition(
        ...     EscalationState.PENDING,
        ...     EscalationEvent.START,
        ...     actor="scheduler",
        ... )
        >>> transition.to_state
        EscalationState.ACTIVE
    """

    # Valid transitions: (from_state, event) -> to_state
    TRANSITIONS: dict[tuple[EscalationState, EscalationEvent], EscalationState] = {
        # From PENDING
        (EscalationState.PENDING, EscalationEvent.START): EscalationState.ACTIVE,
        (EscalationState.PENDING, EscalationEvent.CANCEL): EscalationState.CANCELLED,
        (EscalationState.PENDING, EscalationEvent.ERROR): EscalationState.FAILED,
        # From ACTIVE
        (EscalationState.ACTIVE, EscalationEvent.NOTIFY): EscalationState.ACTIVE,
        (EscalationState.ACTIVE, EscalationEvent.NOTIFY_SUCCESS): EscalationState.ACTIVE,
        (EscalationState.ACTIVE, EscalationEvent.ESCALATE): EscalationState.ESCALATING,
        (EscalationState.ACTIVE, EscalationEvent.ACKNOWLEDGE): EscalationState.ACKNOWLEDGED,
        (EscalationState.ACTIVE, EscalationEvent.RESOLVE): EscalationState.RESOLVED,
        (EscalationState.ACTIVE, EscalationEvent.CANCEL): EscalationState.CANCELLED,
        (EscalationState.ACTIVE, EscalationEvent.TIMEOUT): EscalationState.TIMED_OUT,
        (EscalationState.ACTIVE, EscalationEvent.MAX_LEVEL_REACHED): EscalationState.TIMED_OUT,
        (EscalationState.ACTIVE, EscalationEvent.ERROR): EscalationState.FAILED,
        # From ESCALATING
        (EscalationState.ESCALATING, EscalationEvent.ESCALATE_SUCCESS): EscalationState.ACTIVE,
        (EscalationState.ESCALATING, EscalationEvent.ACKNOWLEDGE): EscalationState.ACKNOWLEDGED,
        (EscalationState.ESCALATING, EscalationEvent.RESOLVE): EscalationState.RESOLVED,
        (EscalationState.ESCALATING, EscalationEvent.CANCEL): EscalationState.CANCELLED,
        (EscalationState.ESCALATING, EscalationEvent.TIMEOUT): EscalationState.TIMED_OUT,
        (EscalationState.ESCALATING, EscalationEvent.MAX_LEVEL_REACHED): EscalationState.TIMED_OUT,
        (EscalationState.ESCALATING, EscalationEvent.ERROR): EscalationState.FAILED,
        (EscalationState.ESCALATING, EscalationEvent.RETRY): EscalationState.ACTIVE,
        # From ACKNOWLEDGED
        (EscalationState.ACKNOWLEDGED, EscalationEvent.RESOLVE): EscalationState.RESOLVED,
        (EscalationState.ACKNOWLEDGED, EscalationEvent.CANCEL): EscalationState.CANCELLED,
        (EscalationState.ACKNOWLEDGED, EscalationEvent.TIMEOUT): EscalationState.TIMED_OUT,
        (EscalationState.ACKNOWLEDGED, EscalationEvent.ERROR): EscalationState.FAILED,
        # RESOLVED, CANCELLED, TIMED_OUT, FAILED are terminal states (no transitions out)
    }

    def __init__(self) -> None:
        """Initialize state machine."""
        self._guards: dict[tuple[EscalationState, EscalationEvent], list[TransitionGuard]] = {}
        self._actions: dict[tuple[EscalationState, EscalationEvent], list[TransitionAction]] = {}
        self._on_enter: dict[EscalationState, list[TransitionAction]] = {}
        self._on_exit: dict[EscalationState, list[TransitionAction]] = {}

    def can_transition(
        self,
        current_state: EscalationState,
        event: EscalationEvent,
        context: dict[str, Any] | None = None,
    ) -> bool:
        """Check if a transition is valid.

        Args:
            current_state: Current escalation state.
            event: Event to trigger.
            context: Optional context for guard evaluation.

        Returns:
            True if transition is valid.
        """
        key = (current_state, event)

        # Check if transition exists
        if key not in self.TRANSITIONS:
            return False

        # Check guards
        guards = self._guards.get(key, [])
        ctx = context or {}

        for guard in guards:
            if not guard(current_state, event, ctx):
                return False

        return True

    def get_next_state(
        self,
        current_state: EscalationState,
        event: EscalationEvent,
    ) -> EscalationState | None:
        """Get the next state for a transition.

        Args:
            current_state: Current escalation state.
            event: Event to trigger.

        Returns:
            Next state if valid, None otherwise.
        """
        return self.TRANSITIONS.get((current_state, event))

    def transition(
        self,
        current_state: EscalationState,
        event: EscalationEvent,
        actor: str = "system",
        context: dict[str, Any] | None = None,
    ) -> StateTransition | None:
        """Execute a state transition.

        Args:
            current_state: Current escalation state.
            event: Event to trigger.
            actor: Who/what triggered the transition.
            context: Optional context for guards and actions.

        Returns:
            StateTransition if successful, None if invalid.

        Raises:
            ValueError: If transition is not valid.
        """
        ctx = context or {}

        if not self.can_transition(current_state, event, ctx):
            return None

        next_state = self.TRANSITIONS[(current_state, event)]

        transition = StateTransition(
            from_state=current_state,
            to_state=next_state,
            event=event,
            actor=actor,
            details=ctx.get("details", {}),
        )

        # Execute on_exit actions for current state
        for action in self._on_exit.get(current_state, []):
            action(transition, ctx)

        # Execute transition actions
        for action in self._actions.get((current_state, event), []):
            action(transition, ctx)

        # Execute on_enter actions for next state
        for action in self._on_enter.get(next_state, []):
            action(transition, ctx)

        return transition

    def add_guard(
        self,
        from_state: EscalationState,
        event: EscalationEvent,
        guard: TransitionGuard,
    ) -> EscalationStateMachine:
        """Add a guard function for a transition.

        Guards are evaluated to determine if a transition is allowed.

        Args:
            from_state: Starting state.
            event: Triggering event.
            guard: Guard function.

        Returns:
            Self for chaining.
        """
        key = (from_state, event)
        if key not in self._guards:
            self._guards[key] = []
        self._guards[key].append(guard)
        return self

    def add_action(
        self,
        from_state: EscalationState,
        event: EscalationEvent,
        action: TransitionAction,
    ) -> EscalationStateMachine:
        """Add an action for a transition.

        Actions are executed when a transition occurs.

        Args:
            from_state: Starting state.
            event: Triggering event.
            action: Action function.

        Returns:
            Self for chaining.
        """
        key = (from_state, event)
        if key not in self._actions:
            self._actions[key] = []
        self._actions[key].append(action)
        return self

    def on_enter(
        self,
        state: EscalationState,
        action: TransitionAction,
    ) -> EscalationStateMachine:
        """Add an action to execute when entering a state.

        Args:
            state: State to add action for.
            action: Action function.

        Returns:
            Self for chaining.
        """
        if state not in self._on_enter:
            self._on_enter[state] = []
        self._on_enter[state].append(action)
        return self

    def on_exit(
        self,
        state: EscalationState,
        action: TransitionAction,
    ) -> EscalationStateMachine:
        """Add an action to execute when exiting a state.

        Args:
            state: State to add action for.
            action: Action function.

        Returns:
            Self for chaining.
        """
        if state not in self._on_exit:
            self._on_exit[state] = []
        self._on_exit[state].append(action)
        return self

    def get_valid_events(self, state: EscalationState) -> list[EscalationEvent]:
        """Get all valid events for a state.

        Args:
            state: Current state.

        Returns:
            List of valid events.
        """
        return [event for (s, event) in self.TRANSITIONS if s == state]

    def get_reachable_states(self, state: EscalationState) -> list[EscalationState]:
        """Get all states reachable from a state.

        Args:
            state: Starting state.

        Returns:
            List of reachable states.
        """
        return [
            next_state
            for (s, _), next_state in self.TRANSITIONS.items()
            if s == state
        ]

    @classmethod
    def create_default(cls) -> EscalationStateMachine:
        """Create a state machine with default configuration.

        Returns:
            Configured state machine.
        """
        machine = cls()

        # Add default guards
        def max_escalation_guard(
            state: EscalationState,
            event: EscalationEvent,
            ctx: dict[str, Any],
        ) -> bool:
            """Guard to check max escalation limit."""
            max_level = ctx.get("max_level", float("inf"))
            current_level = ctx.get("current_level", 1)
            return current_level < max_level

        machine.add_guard(
            EscalationState.ACTIVE,
            EscalationEvent.ESCALATE,
            max_escalation_guard,
        )
        machine.add_guard(
            EscalationState.ESCALATING,
            EscalationEvent.ESCALATE_SUCCESS,
            max_escalation_guard,
        )

        return machine


class EscalationStateManager:
    """High-level manager for escalation state operations.

    Provides a convenient interface for common state operations
    with built-in validation.

    Example:
        >>> manager = EscalationStateManager()
        >>> manager.start(record)
        >>> manager.acknowledge(record, "user-123")
        >>> manager.resolve(record, "user-123")
    """

    def __init__(self, machine: EscalationStateMachine | None = None) -> None:
        """Initialize manager.

        Args:
            machine: State machine to use (default: create new).
        """
        self._machine = machine or EscalationStateMachine.create_default()

    @property
    def machine(self) -> EscalationStateMachine:
        """Get the underlying state machine."""
        return self._machine

    def get_state(self, state_str: str) -> EscalationState:
        """Parse a state string into an EscalationState.

        Args:
            state_str: State string value.

        Returns:
            Parsed EscalationState.
        """
        return EscalationState(state_str)

    def is_valid_transition(
        self,
        current_state: str | EscalationState,
        event: str | EscalationEvent,
        context: dict[str, Any] | None = None,
    ) -> bool:
        """Check if a transition is valid.

        Args:
            current_state: Current state (string or enum).
            event: Event to trigger (string or enum).
            context: Optional context for guards.

        Returns:
            True if transition is valid.
        """
        state = (
            EscalationState(current_state)
            if isinstance(current_state, str)
            else current_state
        )
        evt = EscalationEvent(event) if isinstance(event, str) else event
        return self._machine.can_transition(state, evt, context)

    def apply_event(
        self,
        record: Any,  # EscalationRecord
        event: str | EscalationEvent,
        actor: str = "system",
        context: dict[str, Any] | None = None,
    ) -> StateTransition | None:
        """Apply an event to an escalation record.

        Args:
            record: Escalation record to update.
            event: Event to apply.
            actor: Who triggered the event.
            context: Optional context.

        Returns:
            StateTransition if successful, None otherwise.
        """
        evt = EscalationEvent(event) if isinstance(event, str) else event
        state = EscalationState(record.state)

        ctx = context or {}
        ctx["record"] = record

        transition = self._machine.transition(state, evt, actor, ctx)

        if transition:
            record.state = transition.to_state.value
            record.updated_at = datetime.now()
            record.add_history_event(
                f"state_transition:{evt.value}",
                transition.to_dict(),
            )

        return transition

    def start(
        self,
        record: Any,
        actor: str = "system",
    ) -> StateTransition | None:
        """Start an escalation.

        Args:
            record: Escalation record.
            actor: Who triggered the start.

        Returns:
            StateTransition if successful.
        """
        return self.apply_event(record, EscalationEvent.START, actor)

    def notify(
        self,
        record: Any,
        success: bool = True,
        actor: str = "system",
    ) -> StateTransition | None:
        """Record notification attempt.

        Args:
            record: Escalation record.
            success: Whether notification succeeded.
            actor: Who triggered notification.

        Returns:
            StateTransition if successful.
        """
        event = EscalationEvent.NOTIFY_SUCCESS if success else EscalationEvent.NOTIFY_FAILURE
        return self.apply_event(record, event, actor)

    def escalate(
        self,
        record: Any,
        next_level: int,
        actor: str = "system",
    ) -> StateTransition | None:
        """Begin escalation to next level.

        Args:
            record: Escalation record.
            next_level: Next escalation level.
            actor: Who triggered escalation.

        Returns:
            StateTransition if successful.
        """
        transition = self.apply_event(
            record,
            EscalationEvent.ESCALATE,
            actor,
            {"next_level": next_level},
        )

        if transition:
            record.current_level = next_level
            record.escalation_count += 1

        return transition

    def complete_escalation(
        self,
        record: Any,
        actor: str = "system",
    ) -> StateTransition | None:
        """Complete an escalation to next level.

        Args:
            record: Escalation record.
            actor: Who completed escalation.

        Returns:
            StateTransition if successful.
        """
        return self.apply_event(record, EscalationEvent.ESCALATE_SUCCESS, actor)

    def acknowledge(
        self,
        record: Any,
        acknowledged_by: str,
    ) -> StateTransition | None:
        """Acknowledge an escalation.

        Args:
            record: Escalation record.
            acknowledged_by: Who acknowledged.

        Returns:
            StateTransition if successful.
        """
        transition = self.apply_event(
            record,
            EscalationEvent.ACKNOWLEDGE,
            acknowledged_by,
        )

        if transition:
            record.acknowledged_at = datetime.now()
            record.acknowledged_by = acknowledged_by

        return transition

    def resolve(
        self,
        record: Any,
        resolved_by: str,
    ) -> StateTransition | None:
        """Resolve an escalation.

        Args:
            record: Escalation record.
            resolved_by: Who resolved.

        Returns:
            StateTransition if successful.
        """
        transition = self.apply_event(
            record,
            EscalationEvent.RESOLVE,
            resolved_by,
        )

        if transition:
            record.resolved_at = datetime.now()
            record.resolved_by = resolved_by

        return transition

    def cancel(
        self,
        record: Any,
        cancelled_by: str,
        reason: str = "",
    ) -> StateTransition | None:
        """Cancel an escalation.

        Args:
            record: Escalation record.
            cancelled_by: Who cancelled.
            reason: Cancellation reason.

        Returns:
            StateTransition if successful.
        """
        return self.apply_event(
            record,
            EscalationEvent.CANCEL,
            cancelled_by,
            {"reason": reason},
        )

    def timeout(
        self,
        record: Any,
        reason: str = "max_escalations_reached",
    ) -> StateTransition | None:
        """Mark escalation as timed out.

        Args:
            record: Escalation record.
            reason: Timeout reason.

        Returns:
            StateTransition if successful.
        """
        return self.apply_event(
            record,
            EscalationEvent.TIMEOUT,
            "system",
            {"reason": reason},
        )

    def fail(
        self,
        record: Any,
        error: str,
    ) -> StateTransition | None:
        """Mark escalation as failed.

        Args:
            record: Escalation record.
            error: Error message.

        Returns:
            StateTransition if successful.
        """
        return self.apply_event(
            record,
            EscalationEvent.ERROR,
            "system",
            {"error": error},
        )
