"""Tests for escalation state machine."""

from __future__ import annotations

import pytest

from truthound.checkpoint.escalation.protocols import EscalationRecord
from truthound.checkpoint.escalation.states import (
    EscalationEvent,
    EscalationState,
    EscalationStateMachine,
    EscalationStateManager,
    StateTransition,
)


class TestEscalationState:
    """Tests for EscalationState enum."""

    def test_terminal_states(self) -> None:
        """Test terminal state checking."""
        assert EscalationState.RESOLVED.is_terminal
        assert EscalationState.CANCELLED.is_terminal
        assert EscalationState.TIMED_OUT.is_terminal
        assert EscalationState.FAILED.is_terminal

        assert not EscalationState.PENDING.is_terminal
        assert not EscalationState.ACTIVE.is_terminal

    def test_active_states(self) -> None:
        """Test active state checking."""
        assert EscalationState.PENDING.is_active
        assert EscalationState.ACTIVE.is_active
        assert EscalationState.ESCALATING.is_active

        assert not EscalationState.RESOLVED.is_active
        assert not EscalationState.CANCELLED.is_active

    def test_allows_acknowledgment(self) -> None:
        """Test acknowledgment allowed states."""
        assert EscalationState.ACTIVE.allows_acknowledgment
        assert EscalationState.ESCALATING.allows_acknowledgment

        assert not EscalationState.PENDING.allows_acknowledgment
        assert not EscalationState.RESOLVED.allows_acknowledgment

    def test_allows_resolution(self) -> None:
        """Test resolution allowed states."""
        assert EscalationState.ACTIVE.allows_resolution
        assert EscalationState.ESCALATING.allows_resolution
        assert EscalationState.ACKNOWLEDGED.allows_resolution

        assert not EscalationState.PENDING.allows_resolution
        assert not EscalationState.RESOLVED.allows_resolution


class TestStateTransition:
    """Tests for StateTransition."""

    def test_create_transition(self) -> None:
        """Test creating a state transition."""
        transition = StateTransition(
            from_state=EscalationState.PENDING,
            to_state=EscalationState.ACTIVE,
            event=EscalationEvent.START,
            actor="scheduler",
        )
        assert transition.from_state == EscalationState.PENDING
        assert transition.to_state == EscalationState.ACTIVE
        assert transition.event == EscalationEvent.START
        assert transition.actor == "scheduler"

    def test_transition_serialization(self) -> None:
        """Test transition serialization."""
        transition = StateTransition(
            from_state=EscalationState.ACTIVE,
            to_state=EscalationState.ACKNOWLEDGED,
            event=EscalationEvent.ACKNOWLEDGE,
            actor="user-123",
            details={"message": "I'm on it"},
        )
        data = transition.to_dict()

        assert data["from_state"] == "active"
        assert data["to_state"] == "acknowledged"
        assert data["event"] == "acknowledge"
        assert data["actor"] == "user-123"
        assert data["details"]["message"] == "I'm on it"

    def test_transition_from_dict(self) -> None:
        """Test transition deserialization."""
        data = {
            "from_state": "active",
            "to_state": "resolved",
            "event": "resolve",
            "timestamp": "2024-01-01T12:00:00",
            "actor": "user-123",
            "details": {},
        }
        transition = StateTransition.from_dict(data)

        assert transition.from_state == EscalationState.ACTIVE
        assert transition.to_state == EscalationState.RESOLVED
        assert transition.event == EscalationEvent.RESOLVE


class TestEscalationStateMachine:
    """Tests for EscalationStateMachine."""

    def test_valid_transitions(self) -> None:
        """Test valid transitions are allowed."""
        machine = EscalationStateMachine()

        assert machine.can_transition(
            EscalationState.PENDING, EscalationEvent.START
        )
        assert machine.can_transition(
            EscalationState.ACTIVE, EscalationEvent.ACKNOWLEDGE
        )
        assert machine.can_transition(
            EscalationState.ACTIVE, EscalationEvent.ESCALATE
        )
        assert machine.can_transition(
            EscalationState.ACKNOWLEDGED, EscalationEvent.RESOLVE
        )

    def test_invalid_transitions(self) -> None:
        """Test invalid transitions are blocked."""
        machine = EscalationStateMachine()

        # Can't acknowledge from pending
        assert not machine.can_transition(
            EscalationState.PENDING, EscalationEvent.ACKNOWLEDGE
        )

        # Can't escalate from resolved
        assert not machine.can_transition(
            EscalationState.RESOLVED, EscalationEvent.ESCALATE
        )

        # Can't start from active
        assert not machine.can_transition(
            EscalationState.ACTIVE, EscalationEvent.START
        )

    def test_get_next_state(self) -> None:
        """Test getting next state for transition."""
        machine = EscalationStateMachine()

        assert machine.get_next_state(
            EscalationState.PENDING, EscalationEvent.START
        ) == EscalationState.ACTIVE

        assert machine.get_next_state(
            EscalationState.ACTIVE, EscalationEvent.ACKNOWLEDGE
        ) == EscalationState.ACKNOWLEDGED

        assert machine.get_next_state(
            EscalationState.ACTIVE, EscalationEvent.ERROR
        ) == EscalationState.FAILED

    def test_execute_transition(self) -> None:
        """Test executing a transition."""
        machine = EscalationStateMachine()

        transition = machine.transition(
            EscalationState.PENDING,
            EscalationEvent.START,
            actor="scheduler",
        )

        assert transition is not None
        assert transition.from_state == EscalationState.PENDING
        assert transition.to_state == EscalationState.ACTIVE
        assert transition.actor == "scheduler"

    def test_invalid_transition_returns_none(self) -> None:
        """Test invalid transition returns None."""
        machine = EscalationStateMachine()

        transition = machine.transition(
            EscalationState.RESOLVED,
            EscalationEvent.ESCALATE,
        )

        assert transition is None

    def test_transition_with_guard(self) -> None:
        """Test transition with guard function."""
        machine = EscalationStateMachine()

        # Add a guard that blocks escalation after level 2
        def max_level_guard(
            state: EscalationState,
            event: EscalationEvent,
            ctx: dict,
        ) -> bool:
            return ctx.get("current_level", 1) < 2

        machine.add_guard(
            EscalationState.ACTIVE,
            EscalationEvent.ESCALATE,
            max_level_guard,
        )

        # Should allow escalation at level 1
        assert machine.can_transition(
            EscalationState.ACTIVE,
            EscalationEvent.ESCALATE,
            {"current_level": 1},
        )

        # Should block escalation at level 2
        assert not machine.can_transition(
            EscalationState.ACTIVE,
            EscalationEvent.ESCALATE,
            {"current_level": 2},
        )

    def test_transition_with_action(self) -> None:
        """Test transition with action callback."""
        machine = EscalationStateMachine()
        action_called = []

        def log_action(transition: StateTransition, ctx: dict) -> None:
            action_called.append(transition.event.value)

        machine.add_action(
            EscalationState.ACTIVE,
            EscalationEvent.ACKNOWLEDGE,
            log_action,
        )

        machine.transition(
            EscalationState.ACTIVE,
            EscalationEvent.ACKNOWLEDGE,
            actor="user-123",
        )

        assert "acknowledge" in action_called

    def test_on_enter_callback(self) -> None:
        """Test on_enter state callback."""
        machine = EscalationStateMachine()
        entered_states = []

        def on_enter_resolved(transition: StateTransition, ctx: dict) -> None:
            entered_states.append(transition.to_state.value)

        machine.on_enter(EscalationState.RESOLVED, on_enter_resolved)

        machine.transition(
            EscalationState.ACKNOWLEDGED,
            EscalationEvent.RESOLVE,
        )

        assert "resolved" in entered_states

    def test_on_exit_callback(self) -> None:
        """Test on_exit state callback."""
        machine = EscalationStateMachine()
        exited_states = []

        def on_exit_active(transition: StateTransition, ctx: dict) -> None:
            exited_states.append(transition.from_state.value)

        machine.on_exit(EscalationState.ACTIVE, on_exit_active)

        machine.transition(
            EscalationState.ACTIVE,
            EscalationEvent.ACKNOWLEDGE,
        )

        assert "active" in exited_states

    def test_get_valid_events(self) -> None:
        """Test getting valid events for a state."""
        machine = EscalationStateMachine()

        events = machine.get_valid_events(EscalationState.ACTIVE)

        assert EscalationEvent.NOTIFY in events
        assert EscalationEvent.ACKNOWLEDGE in events
        assert EscalationEvent.ESCALATE in events
        assert EscalationEvent.RESOLVE in events

    def test_get_reachable_states(self) -> None:
        """Test getting reachable states."""
        machine = EscalationStateMachine()

        reachable = machine.get_reachable_states(EscalationState.ACTIVE)

        assert EscalationState.ACKNOWLEDGED in reachable
        assert EscalationState.ESCALATING in reachable
        assert EscalationState.RESOLVED in reachable
        assert EscalationState.CANCELLED in reachable

    def test_create_default_machine(self) -> None:
        """Test creating default configured machine."""
        machine = EscalationStateMachine.create_default()

        # Should have max escalation guard
        assert machine.can_transition(
            EscalationState.ACTIVE,
            EscalationEvent.ESCALATE,
            {"current_level": 1, "max_level": 3},
        )


class TestEscalationStateManager:
    """Tests for EscalationStateManager."""

    def test_start_escalation(self) -> None:
        """Test starting an escalation."""
        manager = EscalationStateManager()
        record = EscalationRecord.create("inc-1", "policy-1")

        transition = manager.start(record)

        assert transition is not None
        assert record.state == "active"
        assert len(record.history) == 1

    def test_acknowledge_escalation(self) -> None:
        """Test acknowledging an escalation."""
        manager = EscalationStateManager()
        record = EscalationRecord.create("inc-1", "policy-1")
        record.state = "active"

        transition = manager.acknowledge(record, "user-123")

        assert transition is not None
        assert record.state == "acknowledged"
        assert record.acknowledged_by == "user-123"
        assert record.acknowledged_at is not None

    def test_resolve_escalation(self) -> None:
        """Test resolving an escalation."""
        manager = EscalationStateManager()
        record = EscalationRecord.create("inc-1", "policy-1")
        record.state = "acknowledged"

        transition = manager.resolve(record, "user-123")

        assert transition is not None
        assert record.state == "resolved"
        assert record.resolved_by == "user-123"
        assert record.resolved_at is not None

    def test_cancel_escalation(self) -> None:
        """Test cancelling an escalation."""
        manager = EscalationStateManager()
        record = EscalationRecord.create("inc-1", "policy-1")
        record.state = "active"

        transition = manager.cancel(record, "user-123", "False alarm")

        assert transition is not None
        assert record.state == "cancelled"

    def test_escalate_to_next_level(self) -> None:
        """Test escalating to next level."""
        manager = EscalationStateManager()
        record = EscalationRecord.create("inc-1", "policy-1")
        record.state = "active"
        record.current_level = 1

        transition = manager.escalate(record, next_level=2)

        assert transition is not None
        assert record.state == "escalating"
        assert record.current_level == 2
        assert record.escalation_count == 1

    def test_complete_escalation(self) -> None:
        """Test completing escalation."""
        manager = EscalationStateManager()
        record = EscalationRecord.create("inc-1", "policy-1")
        record.state = "escalating"

        transition = manager.complete_escalation(record)

        assert transition is not None
        assert record.state == "active"

    def test_timeout_escalation(self) -> None:
        """Test timing out an escalation."""
        manager = EscalationStateManager()
        record = EscalationRecord.create("inc-1", "policy-1")
        record.state = "active"

        transition = manager.timeout(record, "max_level_reached")

        assert transition is not None
        assert record.state == "timed_out"

    def test_fail_escalation(self) -> None:
        """Test failing an escalation."""
        manager = EscalationStateManager()
        record = EscalationRecord.create("inc-1", "policy-1")
        record.state = "active"

        transition = manager.fail(record, "Connection timeout")

        assert transition is not None
        assert record.state == "failed"

    def test_invalid_transition_returns_none(self) -> None:
        """Test invalid transition returns None."""
        manager = EscalationStateManager()
        record = EscalationRecord.create("inc-1", "policy-1")
        record.state = "resolved"

        transition = manager.escalate(record, next_level=2)

        assert transition is None
        assert record.state == "resolved"

    def test_is_valid_transition(self) -> None:
        """Test checking transition validity."""
        manager = EscalationStateManager()

        assert manager.is_valid_transition("pending", "start")
        assert manager.is_valid_transition(
            EscalationState.ACTIVE, EscalationEvent.ACKNOWLEDGE
        )
        assert not manager.is_valid_transition("resolved", "escalate")

    def test_get_state(self) -> None:
        """Test parsing state string."""
        manager = EscalationStateManager()

        state = manager.get_state("active")
        assert state == EscalationState.ACTIVE

        state = manager.get_state("resolved")
        assert state == EscalationState.RESOLVED
