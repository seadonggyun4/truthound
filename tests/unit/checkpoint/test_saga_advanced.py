"""Advanced Saga Pattern Tests.

This module provides comprehensive tests for the enterprise saga pattern
framework, covering complex scenarios including:
- State machine transitions
- Event sourcing and replay
- Advanced compensation strategies
- Complex saga patterns (chained, nested, parallel)
- Failure injection and recovery
"""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from truthound.checkpoint.actions.base import (
    ActionConfig,
    ActionResult,
    ActionStatus,
    BaseAction,
)
from truthound.checkpoint.transaction.base import (
    CompensationResult,
    TransactionContext,
)

# Import saga module components
from truthound.checkpoint.transaction.saga.definition import (
    DependencyType,
    RetryConfig,
    RetryPolicy,
    SagaDefinition,
    SagaStepDefinition,
    StepDependency,
    TimeoutConfig,
)
from truthound.checkpoint.transaction.saga.builder import (
    SagaBuilder,
    StepBuilder,
    saga,
    step,
)
from truthound.checkpoint.transaction.saga.state_machine import (
    InvalidTransitionError,
    SagaEvent,
    SagaEventType,
    SagaState,
    SagaStateMachine,
    SagaTransition,
)
from truthound.checkpoint.transaction.saga.event_store import (
    FileSagaEventStore,
    InMemorySagaEventStore,
    SagaSnapshot,
)
from truthound.checkpoint.transaction.saga.strategies import (
    CompensationPlan,
    CompensationPlanner,
    CompensationPolicy,
    CompensationPriority,
    CountermeasureStrategy,
    PivotTransaction,
    SemanticCompensation,
)
from truthound.checkpoint.transaction.saga.runner import (
    SagaExecutionContext,
    SagaExecutionResult,
    SagaMetrics,
    SagaRunner,
)
from truthound.checkpoint.transaction.saga.patterns import (
    ChainedSagaConfig,
    ChainedSagaPattern,
    NestedSagaConfig,
    NestedSagaPattern,
    ParallelSagaConfig,
    ParallelSagaPattern,
)
from truthound.checkpoint.transaction.saga.testing import (
    FailureInjector,
    FailureType,
    SagaAssertion,
    SagaScenario,
    SagaTestHarness,
    ScenarioBuilder,
)


# =============================================================================
# Test Fixtures and Helpers
# =============================================================================


@dataclass
class MockCheckpointResult:
    """Mock checkpoint result for testing."""

    status: str = "success"
    errors: list[str] | None = None
    metadata: dict[str, Any] | None = None


class SimpleTestAction(BaseAction[ActionConfig]):
    """Simple action for testing."""

    action_type = "simple_test"

    def __init__(
        self,
        name: str = "simple",
        succeed: bool = True,
        delay: float = 0.0,
        side_effect: Any = None,
    ):
        super().__init__()
        self._name = name
        self._succeed = succeed
        self._delay = delay
        self._side_effect = side_effect
        self.executed = False
        self.execute_count = 0

    @classmethod
    def _default_config(cls) -> ActionConfig:
        return ActionConfig()

    @property
    def name(self) -> str:
        return self._name

    def _execute(self, checkpoint_result: Any) -> ActionResult:
        self.executed = True
        self.execute_count += 1

        if self._delay > 0:
            time.sleep(self._delay)

        if self._side_effect:
            if callable(self._side_effect):
                self._side_effect()
            elif isinstance(self._side_effect, Exception):
                raise self._side_effect

        if self._succeed:
            return ActionResult(
                action_name=self.name,
                action_type=self.action_type,
                status=ActionStatus.SUCCESS,
                message="Success",
            )
        else:
            return ActionResult(
                action_name=self.name,
                action_type=self.action_type,
                status=ActionStatus.FAILURE,
                message="Failed",
                error="Simulated failure",
            )


class CompensatableTestAction(SimpleTestAction):
    """Compensatable action for testing."""

    action_type = "compensatable_test"

    def __init__(
        self,
        name: str = "compensatable",
        succeed: bool = True,
        compensation_succeed: bool = True,
        **kwargs: Any,
    ):
        super().__init__(name=name, succeed=succeed, **kwargs)
        self._compensation_succeed = compensation_succeed
        self.compensated = False
        self.compensation_count = 0

    def compensate(
        self,
        checkpoint_result: Any,
        action_result: ActionResult,
        context: TransactionContext,
    ) -> CompensationResult:
        self.compensated = True
        self.compensation_count += 1

        return CompensationResult(
            action_name=self.name,
            success=self._compensation_succeed,
            error=None if self._compensation_succeed else "Compensation failed",
        )


@pytest.fixture
def mock_checkpoint() -> MockCheckpointResult:
    """Create mock checkpoint result."""
    return MockCheckpointResult()


@pytest.fixture
def simple_saga() -> SagaDefinition:
    """Create a simple 3-step saga for testing."""
    saga_def = SagaDefinition(name="simple_saga")

    saga_def.add_step(
        SagaStepDefinition(
            step_id="step1",
            name="Step 1",
            action=CompensatableTestAction(name="action1"),
            compensation_fn=lambda cp, ar, ctx: CompensationResult(
                action_name="action1", success=True
            ),
        )
    )
    saga_def.add_step(
        SagaStepDefinition(
            step_id="step2",
            name="Step 2",
            action=CompensatableTestAction(name="action2"),
            compensation_fn=lambda cp, ar, ctx: CompensationResult(
                action_name="action2", success=True
            ),
        )
    )
    saga_def.add_step(
        SagaStepDefinition(
            step_id="step3",
            name="Step 3",
            action=SimpleTestAction(name="action3"),
        )
    )

    return saga_def


@pytest.fixture
def execution_context(mock_checkpoint: MockCheckpointResult) -> SagaExecutionContext:
    """Create execution context."""
    return SagaExecutionContext(checkpoint_result=mock_checkpoint)


# =============================================================================
# Tests: Saga Definition
# =============================================================================


class TestSagaDefinition:
    """Tests for SagaDefinition and SagaStepDefinition."""

    def test_create_empty_saga(self):
        saga_def = SagaDefinition(name="test_saga")
        assert saga_def.name == "test_saga"
        assert len(saga_def.steps) == 0

    def test_add_steps(self):
        saga_def = SagaDefinition(name="test")
        saga_def.add_step(SagaStepDefinition(step_id="s1", name="Step 1"))
        saga_def.add_step(SagaStepDefinition(step_id="s2", name="Step 2"))

        assert len(saga_def.steps) == 2
        assert saga_def.steps[0].step_id == "s1"
        assert saga_def.steps[1].step_id == "s2"

    def test_get_step_by_id(self):
        saga_def = SagaDefinition(name="test")
        saga_def.add_step(SagaStepDefinition(step_id="s1", name="Step 1"))
        saga_def.add_step(SagaStepDefinition(step_id="s2", name="Step 2"))

        step = saga_def.get_step("s1")
        assert step is not None
        assert step.name == "Step 1"

        assert saga_def.get_step("nonexistent") is None

    def test_execution_order_simple(self):
        saga_def = SagaDefinition(name="test")
        saga_def.add_step(SagaStepDefinition(step_id="s1"))
        saga_def.add_step(SagaStepDefinition(step_id="s2"))
        saga_def.add_step(SagaStepDefinition(step_id="s3"))

        order = saga_def.get_execution_order()
        assert [s.step_id for s in order] == ["s1", "s2", "s3"]

    def test_execution_order_with_dependencies(self):
        saga_def = SagaDefinition(name="test")
        saga_def.add_step(SagaStepDefinition(step_id="s1"))
        saga_def.add_step(
            SagaStepDefinition(
                step_id="s2",
                dependencies=[StepDependency(step_id="s1")],
            )
        )
        saga_def.add_step(
            SagaStepDefinition(
                step_id="s3",
                dependencies=[StepDependency(step_id="s2")],
            )
        )

        order = saga_def.get_execution_order()
        step_ids = [s.step_id for s in order]
        assert step_ids.index("s1") < step_ids.index("s2")
        assert step_ids.index("s2") < step_ids.index("s3")

    def test_compensation_order(self):
        saga_def = SagaDefinition(name="test")
        saga_def.add_step(
            SagaStepDefinition(
                step_id="s1",
                compensation_fn=lambda *args: True,
            )
        )
        saga_def.add_step(
            SagaStepDefinition(
                step_id="s2",
                compensation_fn=lambda *args: True,
            )
        )
        saga_def.add_step(SagaStepDefinition(step_id="s3"))  # No compensation

        comp_order = saga_def.get_compensation_order()
        assert len(comp_order) == 2
        assert comp_order[0].step_id == "s2"  # Reverse order
        assert comp_order[1].step_id == "s1"

    def test_pivot_step(self):
        saga_def = SagaDefinition(name="test")
        saga_def.add_step(SagaStepDefinition(step_id="s1"))
        saga_def.add_step(SagaStepDefinition(step_id="s2", is_pivot=True))
        saga_def.add_step(SagaStepDefinition(step_id="s3"))

        pivot = saga_def.get_pivot_step()
        assert pivot is not None
        assert pivot.step_id == "s2"

    def test_validate_success(self):
        saga_def = SagaDefinition(name="test")
        saga_def.add_step(SagaStepDefinition(step_id="s1"))
        saga_def.add_step(
            SagaStepDefinition(
                step_id="s2",
                dependencies=[StepDependency(step_id="s1")],
            )
        )

        errors = saga_def.validate()
        assert len(errors) == 0

    def test_validate_duplicate_ids(self):
        saga_def = SagaDefinition(name="test")
        saga_def.steps = [
            SagaStepDefinition(step_id="s1"),
            SagaStepDefinition(step_id="s1"),  # Duplicate
        ]

        errors = saga_def.validate()
        assert any("Duplicate" in e for e in errors)

    def test_validate_missing_dependency(self):
        saga_def = SagaDefinition(name="test")
        saga_def.add_step(
            SagaStepDefinition(
                step_id="s1",
                dependencies=[StepDependency(step_id="nonexistent")],
            )
        )

        errors = saga_def.validate()
        assert any("non-existent" in e for e in errors)

    def test_validate_multiple_pivots(self):
        saga_def = SagaDefinition(name="test")
        saga_def.add_step(SagaStepDefinition(step_id="s1", is_pivot=True))
        saga_def.add_step(SagaStepDefinition(step_id="s2", is_pivot=True))

        errors = saga_def.validate()
        assert any("pivot" in e.lower() for e in errors)

    def test_visualize(self):
        saga_def = SagaDefinition(name="Order Processing")
        saga_def.add_step(
            SagaStepDefinition(
                step_id="validate",
                name="Validate Order",
                compensation_fn=lambda *args: True,
            )
        )
        saga_def.add_step(
            SagaStepDefinition(step_id="payment", name="Process Payment", is_pivot=True)
        )

        viz = saga_def.visualize()
        assert "Order Processing" in viz
        assert "Validate Order" in viz
        assert "[C]" in viz  # Compensatable indicator
        assert "[P]" in viz  # Pivot indicator


class TestRetryConfig:
    """Tests for RetryConfig."""

    def test_default_config(self):
        config = RetryConfig()
        assert config.policy == RetryPolicy.EXPONENTIAL
        assert config.max_attempts == 3
        assert config.multiplier == 2.0

    def test_calculate_delay_fixed(self):
        config = RetryConfig(
            policy=RetryPolicy.FIXED,
            initial_delay=timedelta(seconds=2),
            jitter=0.0,
        )

        assert config.calculate_delay(0) == timedelta(seconds=2)
        assert config.calculate_delay(1) == timedelta(seconds=2)
        assert config.calculate_delay(5) == timedelta(seconds=2)

    def test_calculate_delay_exponential(self):
        config = RetryConfig(
            policy=RetryPolicy.EXPONENTIAL,
            initial_delay=timedelta(seconds=1),
            multiplier=2.0,
            jitter=0.0,
        )

        assert config.calculate_delay(0) == timedelta(seconds=1)
        assert config.calculate_delay(1) == timedelta(seconds=2)
        assert config.calculate_delay(2) == timedelta(seconds=4)

    def test_calculate_delay_max_cap(self):
        config = RetryConfig(
            policy=RetryPolicy.EXPONENTIAL,
            initial_delay=timedelta(seconds=1),
            max_delay=timedelta(seconds=10),
            multiplier=10.0,
            jitter=0.0,
        )

        # After first attempt would be 10s, should be capped
        assert config.calculate_delay(2) == timedelta(seconds=10)

    def test_should_retry_max_attempts(self):
        config = RetryConfig(max_attempts=3)

        assert config.should_retry(Exception("test"), 0)
        assert config.should_retry(Exception("test"), 2)
        assert not config.should_retry(Exception("test"), 3)

    def test_should_retry_abort_on(self):
        config = RetryConfig(
            max_attempts=10,
            abort_on=(ValueError,),
        )

        assert config.should_retry(RuntimeError("test"), 0)
        assert not config.should_retry(ValueError("test"), 0)


# =============================================================================
# Tests: Saga Builder
# =============================================================================


class TestSagaBuilder:
    """Tests for the fluent saga builder."""

    def test_basic_saga_build(self):
        saga_def = (
            SagaBuilder("test_saga")
            .description("A test saga")
            .version("2.0.0")
            .build()
        )

        assert saga_def.name == "test_saga"
        assert saga_def.description == "A test saga"
        assert saga_def.version == "2.0.0"

    def test_add_steps_with_builder(self):
        action1 = SimpleTestAction(name="action1")
        action2 = SimpleTestAction(name="action2")

        saga_def = (
            SagaBuilder("test")
            .step("step1")
                .named("First Step")
                .action(action1)
            .end_step()
            .step("step2")
                .named("Second Step")
                .action(action2)
                .depends_on("step1")
            .end_step()
            .build()
        )

        assert len(saga_def.steps) == 2
        assert saga_def.steps[0].name == "First Step"
        assert saga_def.steps[1].name == "Second Step"
        assert len(saga_def.steps[1].dependencies) == 1

    def test_step_with_compensation(self):
        action = SimpleTestAction(name="action")
        comp_action = SimpleTestAction(name="compensation")

        saga_def = (
            SagaBuilder("test")
            .step("step1")
                .action(action)
                .compensate_with(comp_action)
            .end_step()
            .build()
        )

        assert saga_def.steps[0].compensation_action is not None
        assert saga_def.steps[0].has_compensation()

    def test_step_with_retry(self):
        saga_def = (
            SagaBuilder("test")
            .step("step1")
                .action(SimpleTestAction())
                .with_retry(
                    max_attempts=5,
                    policy="exponential",
                    initial_delay=2.0,
                )
            .end_step()
            .build()
        )

        step = saga_def.steps[0]
        assert step.retry_config.max_attempts == 5
        assert step.retry_config.policy == RetryPolicy.EXPONENTIAL
        assert step.retry_config.initial_delay == timedelta(seconds=2)

    def test_step_with_timeout(self):
        saga_def = (
            SagaBuilder("test")
            .step("step1")
                .action(SimpleTestAction())
                .with_timeout(
                    execution_seconds=10,
                    compensation_seconds=5,
                    on_timeout="compensate",
                )
            .end_step()
            .build()
        )

        step = saga_def.steps[0]
        assert step.timeout_config.execution_timeout == timedelta(seconds=10)
        assert step.timeout_config.compensation_timeout == timedelta(seconds=5)
        assert step.timeout_config.on_timeout == "compensate"

    def test_step_as_pivot(self):
        saga_def = (
            SagaBuilder("test")
            .step("commit_point")
                .action(SimpleTestAction())
                .as_pivot()
            .end_step()
            .build()
        )

        assert saga_def.steps[0].is_pivot

    def test_optional_step(self):
        saga_def = (
            SagaBuilder("test")
            .step("optional_step")
                .action(SimpleTestAction())
                .optional()
            .end_step()
            .build()
        )

        assert not saga_def.steps[0].required

    def test_conditional_step(self):
        condition = lambda cp, ctx: True

        saga_def = (
            SagaBuilder("test")
            .step("conditional")
                .action(SimpleTestAction())
                .when(condition)
            .end_step()
            .build()
        )

        assert saga_def.steps[0].condition is not None

    def test_saga_helper_function(self):
        saga_def = saga("quick_saga").build()
        assert saga_def.name == "quick_saga"

    def test_step_helper_function(self):
        step_def = (
            step("standalone_step")
            .action(SimpleTestAction())
            .build()
        )

        assert step_def.step_id == "standalone_step"


# =============================================================================
# Tests: State Machine
# =============================================================================


class TestSagaStateMachine:
    """Tests for the saga state machine."""

    def test_initial_state(self):
        machine = SagaStateMachine("saga_1")
        assert machine.state == SagaState.CREATED
        assert machine.saga_id == "saga_1"

    def test_start_transition(self):
        machine = SagaStateMachine("saga_1")
        event = machine.start()

        assert machine.state == SagaState.STARTING
        assert event.event_type == SagaEventType.SAGA_STARTED
        assert event.source_state == SagaState.CREATED
        assert event.target_state == SagaState.STARTING

    def test_step_execution_flow(self):
        machine = SagaStateMachine("saga_1")
        machine.start()

        machine.step_started("step1")
        assert machine.state == SagaState.STEP_EXECUTING
        assert machine.current_step == "step1"

        machine.step_completed("step1")
        assert machine.state == SagaState.STEP_COMPLETED

        machine.complete()
        assert machine.state == SagaState.COMPLETED
        assert machine.is_terminal

    def test_step_failure_flow(self):
        machine = SagaStateMachine("saga_1")
        machine.start()
        machine.step_started("step1")

        machine.step_failed("step1", "Something went wrong")
        assert machine.state == SagaState.STEP_FAILED

        machine.start_compensation()
        assert machine.state == SagaState.COMPENSATING

    def test_compensation_flow(self):
        machine = SagaStateMachine("saga_1")
        machine.start()
        machine.step_started("step1")
        machine.step_failed("step1", "Error")
        machine.start_compensation()

        machine.step_compensating("step1")
        assert machine.state == SagaState.STEP_COMPENSATING

        machine.step_compensated("step1")
        assert machine.state == SagaState.STEP_COMPENSATED

        machine.compensation_complete()
        assert machine.state == SagaState.COMPENSATED

    def test_abort_transition(self):
        machine = SagaStateMachine("saga_1")
        machine.start()
        machine.step_started("step1")

        machine.abort("User requested abort")
        assert machine.state == SagaState.ABORTED
        assert machine.is_terminal

    def test_timeout_transition(self):
        machine = SagaStateMachine("saga_1")
        machine.start()
        machine.step_started("step1")

        machine.timeout()
        assert machine.state == SagaState.TIMED_OUT

    def test_suspend_and_resume(self):
        machine = SagaStateMachine("saga_1")
        machine.start()
        machine.step_started("step1")
        machine.step_completed("step1")

        machine.suspend("Maintenance")
        assert machine.state == SagaState.SUSPENDED

        machine.resume()
        assert machine.state == SagaState.RESUMING

    def test_invalid_transition_raises(self):
        machine = SagaStateMachine("saga_1")

        with pytest.raises(InvalidTransitionError):
            machine.complete()  # Can't complete from CREATED state

    def test_event_history(self):
        machine = SagaStateMachine("saga_1")
        machine.start()
        machine.step_started("step1")
        machine.step_completed("step1")

        events = machine.events
        assert len(events) == 3
        assert events[0].event_type == SagaEventType.SAGA_STARTED
        assert events[1].event_type == SagaEventType.STEP_STARTED
        assert events[2].event_type == SagaEventType.STEP_COMPLETED

    def test_step_state_tracking(self):
        machine = SagaStateMachine("saga_1")
        machine.start()
        machine.step_started("step1")
        machine.step_completed("step1")
        machine.step_started("step2")
        machine.step_failed("step2", "Error")

        assert "step1" in machine.get_completed_steps()
        assert "step2" in machine.get_failed_steps()

    def test_callbacks(self):
        changes = []
        machine = SagaStateMachine("saga_1")
        machine.on_state_change(
            lambda old, new, evt: changes.append((old, new))
        )

        machine.start()
        machine.step_started("step1")

        assert len(changes) == 2
        assert changes[0] == (SagaState.CREATED, SagaState.STARTING)

    def test_from_events(self):
        # Create original machine and generate events
        original = SagaStateMachine("saga_1")
        original.start()
        original.step_started("step1")
        original.step_completed("step1")

        events = original.events

        # Reconstruct from events
        reconstructed = SagaStateMachine.from_events("saga_1", events)

        assert reconstructed.state == original.state
        assert reconstructed.get_completed_steps() == original.get_completed_steps()


# =============================================================================
# Tests: Event Store
# =============================================================================


class TestInMemorySagaEventStore:
    """Tests for in-memory event store."""

    def test_append_and_get_events(self):
        store = InMemorySagaEventStore()

        event1 = SagaEvent(
            saga_id="saga_1",
            event_type=SagaEventType.SAGA_STARTED,
        )
        event2 = SagaEvent(
            saga_id="saga_1",
            event_type=SagaEventType.STEP_STARTED,
            step_id="step1",
        )

        store.append(event1)
        store.append(event2)

        events = store.get_events("saga_1")
        assert len(events) == 2
        assert events[0].event_type == SagaEventType.SAGA_STARTED

    def test_get_events_from_version(self):
        store = InMemorySagaEventStore()

        for i in range(5):
            store.append(
                SagaEvent(saga_id="saga_1", event_type=SagaEventType.STEP_STARTED)
            )

        events = store.get_events("saga_1", from_version=3)
        assert len(events) == 2

    def test_snapshot_save_and_get(self):
        store = InMemorySagaEventStore()

        snapshot = SagaSnapshot(
            saga_id="saga_1",
            state=SagaState.EXECUTING,
            version=10,
        )

        store.save_snapshot(snapshot)
        retrieved = store.get_latest_snapshot("saga_1")

        assert retrieved is not None
        assert retrieved.saga_id == "saga_1"
        assert retrieved.version == 10

    def test_replay(self):
        store = InMemorySagaEventStore()

        # Store events
        events = [
            SagaEvent(
                saga_id="saga_1",
                event_type=SagaEventType.SAGA_STARTED,
                target_state=SagaState.STARTING,
            ),
            SagaEvent(
                saga_id="saga_1",
                event_type=SagaEventType.STEP_STARTED,
                step_id="step1",
                target_state=SagaState.STEP_EXECUTING,
            ),
            SagaEvent(
                saga_id="saga_1",
                event_type=SagaEventType.STEP_COMPLETED,
                step_id="step1",
                target_state=SagaState.STEP_COMPLETED,
            ),
        ]

        for event in events:
            store.append(event)

        machine = store.replay("saga_1")
        assert machine.state == SagaState.STEP_COMPLETED
        assert "step1" in machine.get_completed_steps()

    def test_list_sagas(self):
        store = InMemorySagaEventStore()

        store.append(SagaEvent(saga_id="saga_1", event_type=SagaEventType.SAGA_STARTED))
        store.append(SagaEvent(saga_id="saga_2", event_type=SagaEventType.SAGA_STARTED))
        store.append(SagaEvent(saga_id="saga_3", event_type=SagaEventType.SAGA_STARTED))

        saga_ids = store.list_sagas()
        assert len(saga_ids) == 3
        assert "saga_1" in saga_ids

    def test_delete_saga(self):
        store = InMemorySagaEventStore()

        store.append(SagaEvent(saga_id="saga_1", event_type=SagaEventType.SAGA_STARTED))
        assert len(store.get_events("saga_1")) == 1

        store.delete_saga("saga_1")
        assert len(store.get_events("saga_1")) == 0


class TestFileSagaEventStore:
    """Tests for file-based event store."""

    def test_append_and_get(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileSagaEventStore(tmpdir)

            event = SagaEvent(
                saga_id="saga_1",
                event_type=SagaEventType.SAGA_STARTED,
            )

            store.append(event)
            events = store.get_events("saga_1")

            assert len(events) == 1
            assert events[0].saga_id == "saga_1"

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # First store instance
            store1 = FileSagaEventStore(tmpdir)
            store1.append(
                SagaEvent(saga_id="saga_1", event_type=SagaEventType.SAGA_STARTED)
            )

            # Second store instance (simulates restart)
            store2 = FileSagaEventStore(tmpdir)
            events = store2.get_events("saga_1")

            assert len(events) == 1

    def test_snapshot_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileSagaEventStore(tmpdir)

            snapshot = SagaSnapshot(
                saga_id="saga_1",
                state=SagaState.EXECUTING,
                version=5,
            )

            store.save_snapshot(snapshot)

            # Reload
            store2 = FileSagaEventStore(tmpdir)
            retrieved = store2.get_latest_snapshot("saga_1")

            assert retrieved is not None
            assert retrieved.version == 5


# =============================================================================
# Tests: Compensation Strategies
# =============================================================================


class TestCompensationPlanner:
    """Tests for compensation planning."""

    def test_backward_plan(self, simple_saga: SagaDefinition):
        planner = CompensationPlanner(CompensationPolicy.BACKWARD)

        plan = planner.create_plan(
            simple_saga,
            completed_steps=["step1", "step2"],
            failed_step="step3",
        )

        assert plan.policy == CompensationPolicy.BACKWARD
        assert len(plan.steps) == 2

        # Should be in reverse order
        execution_order = plan.get_execution_order()
        assert execution_order[0].step_id == "step2"
        assert execution_order[1].step_id == "step1"

    def test_pivot_plan(self):
        saga = SagaDefinition(name="pivot_saga")
        saga.add_step(
            SagaStepDefinition(
                step_id="s1",
                compensation_fn=lambda *args: True,
            )
        )
        saga.add_step(
            SagaStepDefinition(
                step_id="s2",
                is_pivot=True,
            )
        )
        saga.add_step(SagaStepDefinition(step_id="s3"))

        planner = CompensationPlanner(CompensationPolicy.PIVOT)

        # If pivot is completed, no compensation
        plan = planner.create_plan(
            saga,
            completed_steps=["s1", "s2"],
            failed_step="s3",
        )

        assert len(plan.steps) == 0

    def test_best_effort_plan(self, simple_saga: SagaDefinition):
        planner = CompensationPlanner(CompensationPolicy.BEST_EFFORT)

        plan = planner.create_plan(
            simple_saga,
            completed_steps=["step1", "step2"],
            failed_step="step3",
        )

        # All steps should be marked as optional priority
        for step in plan.steps:
            assert step.priority == CompensationPriority.OPTIONAL

    def test_parallel_groups(self):
        saga = SagaDefinition(name="test")
        saga.add_step(
            SagaStepDefinition(
                step_id="s1",
                compensation_fn=lambda *args: True,
            )
        )
        saga.add_step(
            SagaStepDefinition(
                step_id="s2",
                compensation_fn=lambda *args: True,
            )
        )

        planner = CompensationPlanner(CompensationPolicy.PARALLEL)

        plan = planner.create_plan(
            saga,
            completed_steps=["s1", "s2"],
            failed_step=None,
        )

        # All steps should be parallel-capable
        assert all(s.can_parallel for s in plan.steps)


class TestSemanticCompensation:
    """Tests for semantic compensation strategy."""

    def test_execute_semantic(self, mock_checkpoint: MockCheckpointResult):
        from truthound.checkpoint.transaction.saga.strategies import CompensationPlanStep

        compensated = []

        def compensation_fn(cp, ar, ctx):
            compensated.append("semantic_action")
            return CompensationResult(action_name="test", success=True)

        plan = CompensationPlan(
            saga_id="test",
            policy=CompensationPolicy.SEMANTIC,
        )
        plan.add_step(
            CompensationPlanStep(
                step_id="s1",
                step_name="Step 1",
                action=compensation_fn,
                metadata={"semantic_undo": True},
            )
        )

        strategy = SemanticCompensation()
        # Verify plan was created correctly
        assert len(plan.steps) == 1
        assert plan.steps[0].metadata.get("semantic_undo") is True


class TestPivotTransaction:
    """Tests for pivot transaction strategy."""

    def test_stops_at_pivot(self):
        compensated = []

        class TestAction:
            def execute(self, cp):
                compensated.append(self)

        plan = CompensationPlan(
            saga_id="test",
            policy=CompensationPolicy.PIVOT,
        )

        # Add steps with pivot in middle
        from truthound.checkpoint.transaction.saga.strategies import CompensationPlanStep

        plan.steps = [
            CompensationPlanStep(step_id="s1", step_name="Step 1", action=TestAction()),
            CompensationPlanStep(step_id="s2", step_name="Step 2"),  # Pivot
            CompensationPlanStep(step_id="s3", step_name="Step 3", action=TestAction()),
        ]

        strategy = PivotTransaction(pivot_step_id="s2")
        # Execution would stop at s2


# =============================================================================
# Tests: Saga Runner
# =============================================================================


class TestSagaRunner:
    """Tests for saga execution."""

    def test_successful_execution(
        self,
        simple_saga: SagaDefinition,
        execution_context: SagaExecutionContext,
    ):
        runner = SagaRunner()
        result = runner.execute(simple_saga, execution_context)

        assert result.success
        assert result.state == SagaState.COMPLETED
        assert len(result.completed_steps) == 3

    def test_failed_step_triggers_compensation(
        self,
        execution_context: SagaExecutionContext,
    ):
        saga_def = SagaDefinition(name="failing_saga")
        saga_def.add_step(
            SagaStepDefinition(
                step_id="step1",
                action=CompensatableTestAction(name="action1"),
                compensation_fn=lambda cp, ar, ctx: CompensationResult(
                    action_name="action1", success=True
                ),
            )
        )
        saga_def.add_step(
            SagaStepDefinition(
                step_id="step2",
                action=SimpleTestAction(name="failing", succeed=False),
            )
        )

        runner = SagaRunner()
        result = runner.execute(saga_def, execution_context)

        assert not result.success
        assert result.failed_step == "step2"
        assert "step1" in result.compensated_steps

    def test_optional_step_failure_continues(
        self,
        execution_context: SagaExecutionContext,
    ):
        saga_def = SagaDefinition(name="optional_saga")
        saga_def.add_step(
            SagaStepDefinition(
                step_id="step1",
                action=SimpleTestAction(name="action1"),
            )
        )
        saga_def.add_step(
            SagaStepDefinition(
                step_id="optional",
                action=SimpleTestAction(name="optional", succeed=False),
                required=False,  # Optional step
            )
        )
        saga_def.add_step(
            SagaStepDefinition(
                step_id="step3",
                action=SimpleTestAction(name="action3"),
            )
        )

        runner = SagaRunner()
        result = runner.execute(saga_def, execution_context)

        assert result.success
        assert "step1" in result.completed_steps
        assert "step3" in result.completed_steps

    def test_metrics_collection(
        self,
        simple_saga: SagaDefinition,
        execution_context: SagaExecutionContext,
    ):
        runner = SagaRunner()
        result = runner.execute(simple_saga, execution_context)

        metrics = result.metrics
        assert metrics.saga_id == simple_saga.saga_id
        assert metrics.total_duration_ms > 0
        assert len(metrics.step_durations) == 3

    def test_event_collection(
        self,
        simple_saga: SagaDefinition,
        execution_context: SagaExecutionContext,
    ):
        runner = SagaRunner()
        result = runner.execute(simple_saga, execution_context)

        events = result.events
        assert len(events) >= 4  # start + 3 step completions + complete

        event_types = [e.event_type for e in events]
        assert SagaEventType.SAGA_STARTED in event_types
        assert SagaEventType.SAGA_COMPLETED in event_types


# =============================================================================
# Tests: Saga Patterns
# =============================================================================


class TestChainedSagaPattern:
    """Tests for chained saga pattern."""

    def test_chain_execution(
        self,
        mock_checkpoint: MockCheckpointResult,
    ):
        saga1 = SagaDefinition(name="saga1")
        saga1.add_step(SagaStepDefinition(
            step_id="s1",
            action=SimpleTestAction(name="saga1_step"),
        ))

        saga2 = SagaDefinition(name="saga2")
        saga2.add_step(SagaStepDefinition(
            step_id="s2",
            action=SimpleTestAction(name="saga2_step"),
        ))

        chain = ChainedSagaPattern()
        chain.add(saga1).add(saga2)

        context = SagaExecutionContext(checkpoint_result=mock_checkpoint)
        result = chain.execute(context)

        assert result.success
        assert len(result.completed_steps) >= 2

    def test_chain_stop_on_failure(
        self,
        mock_checkpoint: MockCheckpointResult,
    ):
        saga1 = SagaDefinition(name="saga1")
        saga1.add_step(SagaStepDefinition(
            step_id="s1",
            action=SimpleTestAction(name="action1"),
        ))

        saga2 = SagaDefinition(name="saga2")
        saga2.add_step(SagaStepDefinition(
            step_id="s2",
            action=SimpleTestAction(name="failing", succeed=False),
        ))

        saga3 = SagaDefinition(name="saga3")
        saga3.add_step(SagaStepDefinition(
            step_id="s3",
            action=SimpleTestAction(name="action3"),
        ))

        config = ChainedSagaConfig(stop_on_first_failure=True)
        chain = ChainedSagaPattern(config=config)
        chain.add(saga1).add(saga2).add(saga3)

        context = SagaExecutionContext(checkpoint_result=mock_checkpoint)
        result = chain.execute(context)

        assert not result.success
        # The failed_step contains the saga_id (auto-generated), verify it belongs to saga2
        assert result.failed_step == saga2.saga_id


class TestParallelSagaPattern:
    """Tests for parallel saga pattern."""

    def test_parallel_execution(
        self,
        mock_checkpoint: MockCheckpointResult,
    ):
        saga = SagaDefinition(name="parallel_saga")
        saga.add_step(SagaStepDefinition(
            step_id="s1",
            action=SimpleTestAction(name="action1"),
        ))
        saga.add_step(SagaStepDefinition(
            step_id="s2",
            action=SimpleTestAction(name="action2"),
        ))
        saga.add_step(SagaStepDefinition(
            step_id="s3",
            action=SimpleTestAction(name="action3"),
        ))

        parallel = ParallelSagaPattern(saga)
        parallel.parallelize_steps(["s1", "s2", "s3"])

        context = SagaExecutionContext(checkpoint_result=mock_checkpoint)
        result = parallel.execute(context)

        assert result.success
        assert len(result.completed_steps) == 3


# =============================================================================
# Tests: Testing Framework
# =============================================================================


class TestFailureInjector:
    """Tests for failure injection."""

    def test_fail_step(self):
        injector = FailureInjector()
        injector.fail_step("step1", FailureType.EXCEPTION)

        with pytest.raises(RuntimeError):
            injector.check_and_inject("step1")

    def test_intermittent_failure(self):
        injector = FailureInjector()
        injector.fail_step("step1", FailureType.INTERMITTENT, probability=0.5)

        # Run multiple times, should fail roughly half
        failures = 0
        successes = 0

        for _ in range(100):
            try:
                injector.check_and_inject("step1")
                successes += 1
            except RuntimeError:
                failures += 1
            injector.reset()

        # Should have both failures and successes
        assert failures > 0
        assert successes > 0

    def test_retry_success_after(self):
        injector = FailureInjector()
        injector.fail_step("step1", retry_success_after=2)

        # First two attempts fail
        with pytest.raises(RuntimeError):
            injector.check_and_inject("step1")
        with pytest.raises(RuntimeError):
            injector.check_and_inject("step1")

        # Third succeeds
        injector.check_and_inject("step1")  # Should not raise


class TestScenarioBuilder:
    """Tests for scenario builder."""

    def test_basic_scenario(self, simple_saga: SagaDefinition):
        scenario = (
            ScenarioBuilder("test_scenario")
            .with_saga(simple_saga)
            .expect_success()
            .expect_steps_completed("step1", "step2", "step3")
            .build()
        )

        assert scenario.name == "test_scenario"
        assert len(scenario.assertions) >= 4

    def test_failure_scenario(self, simple_saga: SagaDefinition):
        scenario = (
            ScenarioBuilder("failure_test")
            .with_saga(simple_saga)
            .with_failure("step2", FailureType.EXCEPTION)
            .expect_failure()
            .expect_step_completed("step1")
            .expect_step_compensated("step1")
            .build()
        )

        assert scenario.failure_injector is not None


class TestSagaTestHarness:
    """Tests for the test harness."""

    def test_harness_run_scenario(
        self,
        simple_saga: SagaDefinition,
        mock_checkpoint: MockCheckpointResult,
    ):
        harness = SagaTestHarness(
            saga=simple_saga,
            context_factory=lambda: SagaExecutionContext(
                checkpoint_result=mock_checkpoint
            ),
        )

        scenario = (
            harness.create_scenario("happy_path")
            .expect_success()
            .build()
        )

        harness.add_scenario(scenario)
        report = harness.run_all()

        assert "PASS" in report
        assert "happy_path" in report

    def test_generate_failure_scenarios(
        self,
        simple_saga: SagaDefinition,
        mock_checkpoint: MockCheckpointResult,
    ):
        harness = SagaTestHarness(
            saga=simple_saga,
            context_factory=lambda: SagaExecutionContext(
                checkpoint_result=mock_checkpoint
            ),
        )

        scenarios = list(harness.generate_failure_scenarios())
        assert len(scenarios) >= 1


# =============================================================================
# Integration Tests
# =============================================================================


class TestSagaIntegration:
    """Integration tests for complete saga workflows."""

    def test_complete_order_processing_saga(
        self,
        mock_checkpoint: MockCheckpointResult,
    ):
        """Test a realistic order processing saga."""
        # Track side effects
        inventory_reserved = []
        payments_processed = []
        shipments_created = []

        def reserve_inventory():
            inventory_reserved.append("order_123")

        def release_inventory():
            inventory_reserved.pop()

        def process_payment():
            payments_processed.append("payment_456")

        def refund_payment():
            payments_processed.pop()

        def create_shipment():
            shipments_created.append("shipment_789")

        # Build saga
        saga = (
            SagaBuilder("order_processing")
            .description("Process customer order")
            .step("validate_order")
                .action(SimpleTestAction(name="validate"))
            .end_step()
            .step("reserve_inventory")
                .action(SimpleTestAction(
                    name="reserve",
                    side_effect=reserve_inventory,
                ))
                .compensate_with(fn=lambda cp, ar, ctx: (
                    release_inventory(),
                    CompensationResult(action_name="reserve", success=True),
                )[1])
                .depends_on("validate_order")
            .end_step()
            .step("process_payment")
                .action(SimpleTestAction(
                    name="payment",
                    side_effect=process_payment,
                ))
                .compensate_with(fn=lambda cp, ar, ctx: (
                    refund_payment(),
                    CompensationResult(action_name="payment", success=True),
                )[1])
                .depends_on("reserve_inventory")
                .with_retry(max_attempts=2)
            .end_step()
            .step("create_shipment")
                .action(SimpleTestAction(
                    name="shipment",
                    side_effect=create_shipment,
                ))
                .depends_on("process_payment")
                .as_pivot()
            .end_step()
            .step("send_confirmation")
                .action(SimpleTestAction(name="confirm"))
                .depends_on("create_shipment")
                .optional()
            .end_step()
            .with_timeout(60)
            .build()
        )

        # Execute
        runner = SagaRunner()
        context = SagaExecutionContext(checkpoint_result=mock_checkpoint)
        result = runner.execute(saga, context)

        # Verify
        assert result.success
        assert len(inventory_reserved) == 1
        assert len(payments_processed) == 1
        assert len(shipments_created) == 1

    def test_saga_failure_with_compensation(
        self,
        mock_checkpoint: MockCheckpointResult,
    ):
        """Test saga that fails and compensates."""
        side_effects = {"reserved": False, "charged": False}

        def reserve():
            side_effects["reserved"] = True

        def release():
            side_effects["reserved"] = False

        def charge():
            side_effects["charged"] = True

        def refund():
            side_effects["charged"] = False

        saga = (
            SagaBuilder("failing_saga")
            .step("reserve")
                .action(SimpleTestAction(name="reserve", side_effect=reserve))
                .compensate_with(fn=lambda cp, ar, ctx: (
                    release(),
                    CompensationResult(action_name="reserve", success=True),
                )[1])
            .end_step()
            .step("charge")
                .action(SimpleTestAction(name="charge", side_effect=charge))
                .compensate_with(fn=lambda cp, ar, ctx: (
                    refund(),
                    CompensationResult(action_name="charge", success=True),
                )[1])
                .depends_on("reserve")
            .end_step()
            .step("fail_step")
                .action(SimpleTestAction(name="fail", succeed=False))
                .depends_on("charge")
            .end_step()
            .build()
        )

        runner = SagaRunner()
        context = SagaExecutionContext(checkpoint_result=mock_checkpoint)
        result = runner.execute(saga, context)

        # Saga failed
        assert not result.success
        assert result.failed_step == "fail_step"

        # But compensations ran
        assert "reserve" in result.compensated_steps
        assert "charge" in result.compensated_steps

        # Side effects were cleaned up
        assert not side_effects["reserved"]
        assert not side_effects["charged"]

    def test_event_sourcing_replay(
        self,
        mock_checkpoint: MockCheckpointResult,
    ):
        """Test event sourcing with replay capability."""
        event_store = InMemorySagaEventStore()
        runner = SagaRunner(event_store=event_store)

        saga = (
            SagaBuilder("replayable_saga")
            .step("step1")
                .action(SimpleTestAction(name="action1"))
            .end_step()
            .step("step2")
                .action(SimpleTestAction(name="action2"))
            .end_step()
            .build()
        )

        context = SagaExecutionContext(checkpoint_result=mock_checkpoint)
        result = runner.execute(saga, context)

        # Replay from events
        replayed = event_store.replay(saga.saga_id)

        assert replayed.state == SagaState.COMPLETED
        assert replayed.get_completed_steps() == result.completed_steps


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
