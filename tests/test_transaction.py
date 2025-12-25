"""Tests for the Transaction Framework.

This module tests the transaction management system including:
- TransactionCoordinator and SagaOrchestrator
- Compensatable actions and CompensationWrapper
- TransactionalExecutor and TransactionBoundary
- IdempotencyStore implementations
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
from truthound.checkpoint.transaction import (
    # Core types
    TransactionState,
    TransactionPhase,
    CompensationStrategy,
    IsolationLevel,
    # Configuration
    TransactionConfig,
    # Results
    TransactionResult,
    CompensationResult,
    # Context
    TransactionContext,
    Savepoint,
    # Interfaces
    Compensatable,
    CompensatableAction,
    # Wrapper
    CompensationWrapper,
    # Decorators
    compensatable,
    with_compensation,
    # Coordinator
    TransactionCoordinator,
    SagaOrchestrator,
    # Executor
    TransactionalExecutor,
    TransactionBoundary,
    TransactionManager,
    # Idempotency
    IdempotencyKey,
    IdempotencyStore,
    InMemoryIdempotencyStore,
    FileIdempotencyStore,
    IdempotencyManager,
    IdempotencyConflictError,
    idempotent,
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


class SimpleAction(BaseAction[ActionConfig]):
    """Simple action for testing."""

    action_type = "simple"

    def __init__(self, name: str = "simple", succeed: bool = True, fail_checkpoint: bool = True):
        super().__init__()
        self._name = name
        self._succeed = succeed
        self._fail_checkpoint = fail_checkpoint
        self.executed = False

    @classmethod
    def _default_config(cls) -> ActionConfig:
        return ActionConfig()

    @property
    def config(self) -> ActionConfig:
        cfg = super().config
        cfg.fail_checkpoint_on_error = self._fail_checkpoint
        return cfg

    @property
    def name(self) -> str:
        return self._name

    def _execute(self, checkpoint_result: Any) -> ActionResult:
        self.executed = True
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


class CompensatableSimpleAction(CompensatableAction[ActionConfig]):
    """Compensatable action for testing."""

    action_type = "compensatable_simple"

    def __init__(
        self,
        name: str = "compensatable",
        succeed: bool = True,
        compensation_succeed: bool = True,
    ):
        super().__init__()
        self._name = name
        self._succeed = succeed
        self._compensation_succeed = compensation_succeed
        self.executed = False
        self.compensated = False

    @classmethod
    def _default_config(cls) -> ActionConfig:
        return ActionConfig()

    @property
    def name(self) -> str:
        return self._name

    def _execute(self, checkpoint_result: Any) -> ActionResult:
        self.executed = True
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
            )

    def _compensate(
        self,
        checkpoint_result: Any,
        action_result: ActionResult,
        context: TransactionContext,
    ) -> CompensationResult:
        self.compensated = True
        return CompensationResult(
            action_name=self.name,
            success=self._compensation_succeed,
            error=None if self._compensation_succeed else "Compensation failed",
        )


class FailingAction(BaseAction[ActionConfig]):
    """Action that always fails with exception."""

    action_type = "failing"

    def __init__(self, name: str = "failing", fail_checkpoint: bool = True):
        super().__init__()
        self._name = name
        self._fail_checkpoint = fail_checkpoint

    @classmethod
    def _default_config(cls) -> ActionConfig:
        return ActionConfig()

    @property
    def config(self) -> ActionConfig:
        cfg = super().config
        cfg.fail_checkpoint_on_error = self._fail_checkpoint
        return cfg

    @property
    def name(self) -> str:
        return self._name

    def _execute(self, checkpoint_result: Any) -> ActionResult:
        raise RuntimeError("Action execution failed")


# =============================================================================
# Tests: Transaction Base Types
# =============================================================================


class TestTransactionState:
    """Tests for TransactionState enum."""

    def test_state_values(self):
        assert TransactionState.PENDING.value == "pending"
        assert TransactionState.ACTIVE.value == "active"
        assert TransactionState.COMMITTED.value == "committed"
        assert TransactionState.ROLLING_BACK.value == "rolling_back"
        assert TransactionState.FAILED.value == "failed"
        assert TransactionState.COMPENSATED.value == "compensated"

    def test_is_terminal(self):
        assert TransactionState.COMMITTED.is_terminal
        assert TransactionState.FAILED.is_terminal
        assert TransactionState.COMPENSATED.is_terminal
        assert not TransactionState.PENDING.is_terminal
        assert not TransactionState.ACTIVE.is_terminal
        assert not TransactionState.ROLLING_BACK.is_terminal

    def test_is_success(self):
        assert TransactionState.COMMITTED.is_success
        assert not TransactionState.FAILED.is_success
        assert not TransactionState.COMPENSATED.is_success


class TestTransactionConfig:
    """Tests for TransactionConfig."""

    def test_defaults(self):
        config = TransactionConfig()
        assert config.enabled
        assert config.rollback_on_failure
        assert config.compensation_strategy == CompensationStrategy.BACKWARD
        assert config.max_compensation_retries == 3
        assert config.savepoint_enabled
        assert config.isolation_level == IsolationLevel.NONE
        assert not config.idempotency_enabled
        assert config.audit_enabled

    def test_string_enum_conversion(self):
        config = TransactionConfig(
            compensation_strategy="forward",
            isolation_level="serializable",
        )
        assert config.compensation_strategy == CompensationStrategy.FORWARD
        assert config.isolation_level == IsolationLevel.SERIALIZABLE


class TestTransactionContext:
    """Tests for TransactionContext."""

    def test_default_initialization(self):
        context = TransactionContext()
        assert context.transaction_id.startswith("txn_")
        assert context.state == TransactionState.PENDING
        assert context.phase == TransactionPhase.PREPARE
        assert not context.rollback_requested
        assert len(context.completed_actions) == 0

    def test_create_savepoint(self):
        context = TransactionContext()
        context.mark_action_completed("action1")

        savepoint = context.create_savepoint("sp1", {"key": "value"})

        assert savepoint.name == "sp1"
        assert savepoint.action_index == 1
        assert savepoint.state_snapshot == {"key": "value"}
        assert len(context.savepoints) == 1

    def test_get_savepoint(self):
        context = TransactionContext()
        sp1 = context.create_savepoint("sp1")

        assert context.get_savepoint("sp1") == sp1
        assert context.get_savepoint(sp1.id) == sp1
        assert context.get_savepoint("nonexistent") is None

    def test_request_rollback(self):
        context = TransactionContext()
        context.request_rollback("test reason")

        assert context.rollback_requested
        assert context.metadata["rollback_reason"] == "test reason"

    def test_mark_actions(self):
        context = TransactionContext()

        context.mark_action_completed("action1")
        context.mark_action_completed("action2")
        context.mark_action_compensated("action1")

        assert "action1" in context.completed_actions
        assert "action2" in context.completed_actions
        assert "action1" in context.compensated_actions
        assert "action2" not in context.compensated_actions

    def test_to_dict(self):
        context = TransactionContext()
        context.mark_action_completed("action1")

        data = context.to_dict()

        assert data["transaction_id"] == context.transaction_id
        assert data["state"] == "pending"
        assert "action1" in data["completed_actions"]


class TestSavepoint:
    """Tests for Savepoint."""

    def test_default_name(self):
        savepoint = Savepoint()
        assert savepoint.name.startswith("savepoint_")

    def test_custom_name(self):
        savepoint = Savepoint(name="custom")
        assert savepoint.name == "custom"


# =============================================================================
# Tests: Compensatable Actions
# =============================================================================


class TestCompensatable:
    """Tests for Compensatable protocol."""

    def test_compensatable_action_is_compensatable(self):
        action = CompensatableSimpleAction()
        assert isinstance(action, Compensatable)

    def test_simple_action_not_compensatable(self):
        action = SimpleAction()
        assert not isinstance(action, Compensatable)


class TestCompensatableAction:
    """Tests for CompensatableAction base class."""

    def test_can_compensate_default(self):
        action = CompensatableSimpleAction()
        result = ActionResult(
            action_name="test",
            action_type="test",
            status=ActionStatus.SUCCESS,
        )
        assert action.can_compensate(result)

    def test_can_compensate_failed_action(self):
        action = CompensatableSimpleAction()
        result = ActionResult(
            action_name="test",
            action_type="test",
            status=ActionStatus.FAILURE,
        )
        # Default config requires success
        assert not action.can_compensate(result)

    def test_compensate_with_retries(self):
        action = CompensatableSimpleAction(compensation_succeed=False)
        result = ActionResult(
            action_name="test",
            action_type="test",
            status=ActionStatus.SUCCESS,
        )
        context = TransactionContext()
        checkpoint = MockCheckpointResult()

        comp_result = action.compensate(checkpoint, result, context)

        assert not comp_result.success
        # The error message comes from _compensate, not from retry logic
        assert comp_result.error == "Compensation failed"

    def test_capture_and_get_state(self):
        action = CompensatableSimpleAction()
        action.capture_state("key1", "value1")
        action.capture_state("key2", {"nested": "data"})

        assert action.get_captured_state("key1") == "value1"
        assert action.get_captured_state("key2") == {"nested": "data"}
        assert action.get_captured_state("nonexistent", "default") == "default"

    def test_clear_state(self):
        action = CompensatableSimpleAction()
        action.capture_state("key1", "value1")
        action.clear_state()

        assert action.get_captured_state("key1") is None


class TestCompensationWrapper:
    """Tests for CompensationWrapper."""

    def test_wraps_non_compensatable_action(self):
        original = SimpleAction(name="original")

        def compensation_fn(cp, ar, ctx):
            return True

        wrapped = CompensationWrapper(
            action=original,
            compensation_fn=compensation_fn,
        )

        assert wrapped.name == "compensatable_original"
        assert isinstance(wrapped, Compensatable)

    def test_execute_delegates_to_wrapped(self):
        original = SimpleAction(name="original")
        wrapped = CompensationWrapper(
            action=original,
            compensation_fn=lambda cp, ar, ctx: True,
        )

        checkpoint = MockCheckpointResult()
        result = wrapped.execute(checkpoint)

        assert result.success
        assert original.executed

    def test_compensate_calls_fn(self):
        original = SimpleAction(name="original")
        compensation_called = []

        def compensation_fn(cp, ar, ctx):
            compensation_called.append(True)
            return True

        wrapped = CompensationWrapper(
            action=original,
            compensation_fn=compensation_fn,
        )

        checkpoint = MockCheckpointResult()
        action_result = ActionResult(
            action_name="test",
            action_type="test",
            status=ActionStatus.SUCCESS,
        )
        context = TransactionContext()

        comp_result = wrapped.compensate(checkpoint, action_result, context)

        assert len(compensation_called) == 1
        assert comp_result.success

    def test_compensate_returns_compensation_result(self):
        original = SimpleAction()

        def compensation_fn(cp, ar, ctx):
            return CompensationResult(
                action_name="test",
                success=True,
                details={"restored": True},
            )

        wrapped = CompensationWrapper(
            action=original,
            compensation_fn=compensation_fn,
        )

        result = wrapped.compensate(
            MockCheckpointResult(),
            ActionResult(action_name="t", action_type="t", status=ActionStatus.SUCCESS),
            TransactionContext(),
        )

        assert result.success
        assert result.details.get("restored")


# =============================================================================
# Tests: Transaction Coordinator
# =============================================================================


class TestTransactionCoordinator:
    """Tests for TransactionCoordinator."""

    def test_successful_transaction(self):
        coordinator = TransactionCoordinator()
        actions = [
            SimpleAction(name="action1"),
            SimpleAction(name="action2"),
            SimpleAction(name="action3"),
        ]
        checkpoint = MockCheckpointResult()

        result = coordinator.execute(actions, checkpoint)

        assert result.success
        assert result.state == TransactionState.COMMITTED
        assert len(result.action_results) == 3
        assert all(ar.success for ar in result.action_results)

    def test_failed_transaction_triggers_compensation(self):
        coordinator = TransactionCoordinator()

        action1 = CompensatableSimpleAction(name="action1")
        action2 = CompensatableSimpleAction(name="action2")
        action3 = SimpleAction(name="action3", succeed=False)  # Fails

        actions = [action1, action2, action3]
        checkpoint = MockCheckpointResult()

        result = coordinator.execute(actions, checkpoint)

        assert not result.success
        assert result.state == TransactionState.COMPENSATED
        assert action1.compensated
        assert action2.compensated

    def test_no_rollback_when_disabled(self):
        config = TransactionConfig(rollback_on_failure=False)
        coordinator = TransactionCoordinator(config=config)

        action1 = CompensatableSimpleAction(name="action1")
        action2 = SimpleAction(name="action2", succeed=False)

        actions = [action1, action2]
        checkpoint = MockCheckpointResult()

        result = coordinator.execute(actions, checkpoint)

        assert result.state == TransactionState.FAILED
        assert not action1.compensated

    def test_exception_during_execution(self):
        coordinator = TransactionCoordinator()
        actions = [
            SimpleAction(name="action1"),
            FailingAction(name="failing"),
        ]
        checkpoint = MockCheckpointResult()

        result = coordinator.execute(actions, checkpoint)

        assert not result.success
        assert "Action execution failed" in result.action_results[-1].error

    def test_callbacks_are_called(self):
        action_completions = []
        state_changes = []

        coordinator = TransactionCoordinator(
            on_action_complete=lambda e: action_completions.append(e.action.name),
            on_state_change=lambda old, new: state_changes.append((old.value, new.value)),
        )

        actions = [SimpleAction(name="action1"), SimpleAction(name="action2")]
        result = coordinator.execute(actions, MockCheckpointResult())

        assert "action1" in action_completions
        assert "action2" in action_completions
        assert ("pending", "active") in state_changes
        assert ("active", "committed") in state_changes

    def test_savepoint_creation(self):
        config = TransactionConfig(savepoint_enabled=True)
        coordinator = TransactionCoordinator(config=config)

        action1 = CompensatableSimpleAction(name="action1")
        action2 = CompensatableSimpleAction(name="action2")

        context = TransactionContext()
        result = coordinator.execute([action1, action2], MockCheckpointResult(), context)

        assert len(context.savepoints) >= 1

    def test_rollback_requested(self):
        coordinator = TransactionCoordinator()

        action1 = SimpleAction(name="action1")

        context = TransactionContext()
        context.request_rollback("User requested")

        result = coordinator.execute([action1], MockCheckpointResult(), context)

        # Should stop at first action because rollback was requested
        assert not action1.executed


class TestSagaOrchestrator:
    """Tests for SagaOrchestrator."""

    def test_add_step(self):
        saga = SagaOrchestrator(name="test_saga")
        saga.add_step("step1", SimpleAction(name="action1"))
        saga.add_step("step2", SimpleAction(name="action2"))

        assert len(saga._steps) == 2

    def test_add_step_chaining(self):
        saga = (
            SagaOrchestrator(name="test")
            .add_step("step1", SimpleAction())
            .add_step("step2", SimpleAction())
        )

        assert len(saga._steps) == 2

    def test_execute_saga(self):
        saga = SagaOrchestrator(name="test_saga")
        saga.add_step("validate", SimpleAction(name="validate"))
        saga.add_step("process", SimpleAction(name="process"))
        saga.add_step("notify", SimpleAction(name="notify"))

        result = saga.execute(MockCheckpointResult())

        assert result.success
        # saga_name is stored in context.metadata, which is in result.metadata["context"]["metadata"]
        assert result.metadata.get("context", {}).get("metadata", {}).get("saga_name") == "test_saga"

    def test_conditional_step(self):
        saga = SagaOrchestrator(name="test")

        executed_actions = []

        class TrackingAction(SimpleAction):
            def _execute(self, cp):
                executed_actions.append(self.name)
                return super()._execute(cp)

        saga.add_step("always", TrackingAction(name="always"))
        saga.add_step(
            "conditional",
            TrackingAction(name="conditional"),
            condition=lambda cp, ctx: False,  # Never execute
        )
        saga.add_step("final", TrackingAction(name="final"))

        saga.execute(MockCheckpointResult())

        assert "always" in executed_actions
        assert "conditional" not in executed_actions
        assert "final" in executed_actions

    def test_visualize(self):
        saga = SagaOrchestrator(name="test_saga")
        saga.add_step("step1", CompensatableSimpleAction(), required=True)
        saga.add_step("step2", SimpleAction(), required=False)

        viz = saga.visualize()

        assert "test_saga" in viz
        assert "step1" in viz
        assert "step2" in viz
        assert "[C]" in viz  # Compensation indicator
        assert "(required)" in viz
        assert "(optional)" in viz


# =============================================================================
# Tests: Transactional Executor
# =============================================================================


class TestTransactionalExecutor:
    """Tests for TransactionalExecutor."""

    def test_transaction_context_manager(self):
        executor = TransactionalExecutor()

        with executor.transaction("my_txn") as txn:
            txn.add_action(SimpleAction(name="action1"))
            txn.add_action(SimpleAction(name="action2"))

        assert len(executor._boundaries) == 1
        assert executor._boundaries[0].name == "my_txn"
        assert len(executor._boundaries[0].actions) == 2

    def test_execute_boundaries(self):
        executor = TransactionalExecutor()

        with executor.transaction("txn1") as txn:
            txn.add_action(SimpleAction(name="a1"))

        with executor.transaction("txn2") as txn:
            txn.add_action(SimpleAction(name="a2"))

        result = executor.execute(MockCheckpointResult())

        assert result.success
        assert result.metadata.get("boundary_count") == 2

    def test_execute_single(self):
        executor = TransactionalExecutor()

        actions = [SimpleAction(name="a1"), SimpleAction(name="a2")]
        result = executor.execute_single(actions, MockCheckpointResult())

        assert result.success
        assert len(result.action_results) == 2

    def test_callbacks(self):
        start_called = []
        complete_called = []

        executor = TransactionalExecutor(
            on_transaction_start=lambda ctx: start_called.append(ctx.transaction_id),
            on_transaction_complete=lambda r: complete_called.append(r.success),
        )

        with executor.transaction() as txn:
            txn.add_action(SimpleAction())

        executor.execute(MockCheckpointResult())

        assert len(start_called) == 1
        assert len(complete_called) == 1
        assert complete_called[0] is True


class TestTransactionBoundary:
    """Tests for TransactionBoundary."""

    def test_add_action(self):
        boundary = TransactionBoundary(name="test")
        boundary.add_action(SimpleAction())
        boundary.add_action(SimpleAction())

        assert len(boundary.actions) == 2

    def test_create_nested(self):
        parent = TransactionBoundary(name="parent")
        child = parent.create_nested("child")

        assert child.nested
        assert child.parent == parent
        assert child.config == parent.config


class TestTransactionManager:
    """Tests for TransactionManager."""

    def test_begin_transaction(self):
        manager = TransactionManager()
        txn_id = manager.begin_transaction()

        assert txn_id.startswith("txn_")
        assert manager.get_transaction(txn_id) is not None

    def test_add_action(self):
        manager = TransactionManager()
        txn_id = manager.begin_transaction()

        manager.add_action(txn_id, SimpleAction())
        manager.add_action(txn_id, SimpleAction())

        assert len(manager._action_log[txn_id]) == 2

    def test_commit(self):
        manager = TransactionManager()
        txn_id = manager.begin_transaction()
        manager.add_action(txn_id, SimpleAction())
        manager.execute_pending(txn_id, MockCheckpointResult())

        result = manager.commit(txn_id)

        assert result.success
        assert result.state == TransactionState.COMMITTED
        assert manager.get_transaction(txn_id) is None  # Cleaned up

    def test_rollback(self):
        manager = TransactionManager()
        txn_id = manager.begin_transaction()

        action = CompensatableSimpleAction()
        manager.add_action(txn_id, action)
        manager.execute_pending(txn_id, MockCheckpointResult())

        result = manager.rollback(txn_id, MockCheckpointResult())

        assert result.state == TransactionState.COMPENSATED
        assert action.compensated

    def test_history(self):
        manager = TransactionManager()

        # Create and commit two transactions
        for _ in range(2):
            txn_id = manager.begin_transaction()
            manager.add_action(txn_id, SimpleAction())
            manager.execute_pending(txn_id, MockCheckpointResult())
            manager.commit(txn_id)

        history = manager.get_history()
        assert len(history) == 2

        limited = manager.get_history(limit=1)
        assert len(limited) == 1


# =============================================================================
# Tests: Idempotency
# =============================================================================


class TestIdempotencyKey:
    """Tests for IdempotencyKey."""

    def test_default_expiration(self):
        key = IdempotencyKey(key="test-key")

        assert key.expires_at is not None
        assert key.expires_at > datetime.now()

    def test_is_expired(self):
        key = IdempotencyKey(
            key="test-key",
            expires_at=datetime.now() - timedelta(hours=1),
        )

        assert key.is_expired

    def test_is_completed(self):
        key = IdempotencyKey(key="test", status="pending")
        assert not key.is_completed

        key.status = "completed"
        key.result = TransactionResult(
            transaction_id="txn_1",
            state=TransactionState.COMMITTED,
        )
        assert key.is_completed

    def test_to_dict_and_from_dict(self):
        key = IdempotencyKey(
            key="test-key",
            request_hash="abc123",
        )

        data = key.to_dict()
        restored = IdempotencyKey.from_dict(data)

        assert restored.key == key.key
        assert restored.request_hash == key.request_hash


class TestInMemoryIdempotencyStore:
    """Tests for InMemoryIdempotencyStore."""

    def test_set_and_get(self):
        store = InMemoryIdempotencyStore()
        key = IdempotencyKey(key="test-key")

        store.set(key)
        retrieved = store.get("test-key")

        assert retrieved is not None
        assert retrieved.key == "test-key"

    def test_get_nonexistent(self):
        store = InMemoryIdempotencyStore()
        assert store.get("nonexistent") is None

    def test_delete(self):
        store = InMemoryIdempotencyStore()
        key = IdempotencyKey(key="test-key")

        store.set(key)
        assert store.delete("test-key")
        assert store.get("test-key") is None
        assert not store.delete("test-key")

    def test_cleanup_expired(self):
        store = InMemoryIdempotencyStore()

        # Add expired key
        expired = IdempotencyKey(
            key="expired",
            expires_at=datetime.now() - timedelta(hours=1),
        )
        store._store["expired"] = expired  # Direct add to bypass expiry check

        # Add valid key
        valid = IdempotencyKey(key="valid")
        store.set(valid)

        removed = store.cleanup_expired()

        assert removed == 1
        assert store.get("expired") is None
        assert store.get("valid") is not None

    def test_max_size_eviction(self):
        store = InMemoryIdempotencyStore(max_size=5)

        for i in range(10):
            store.set(IdempotencyKey(key=f"key-{i}"))

        assert store.size() <= 5

    def test_thread_safety(self):
        import threading

        store = InMemoryIdempotencyStore()
        errors = []

        def add_keys(prefix: str, count: int):
            try:
                for i in range(count):
                    store.set(IdempotencyKey(key=f"{prefix}-{i}"))
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=add_keys, args=("a", 100)),
            threading.Thread(target=add_keys, args=("b", 100)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert store.size() == 200


class TestFileIdempotencyStore:
    """Tests for FileIdempotencyStore."""

    def test_set_and_get(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileIdempotencyStore(tmpdir)
            key = IdempotencyKey(key="test-key")

            store.set(key)
            retrieved = store.get("test-key")

            assert retrieved is not None
            assert retrieved.key == "test-key"

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # First store instance
            store1 = FileIdempotencyStore(tmpdir)
            store1.set(IdempotencyKey(key="persistent-key"))

            # Second store instance (simulates restart)
            store2 = FileIdempotencyStore(tmpdir)
            retrieved = store2.get("persistent-key")

            assert retrieved is not None
            assert retrieved.key == "persistent-key"

    def test_delete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileIdempotencyStore(tmpdir)
            store.set(IdempotencyKey(key="to-delete"))

            assert store.delete("to-delete")
            assert store.get("to-delete") is None

    def test_cleanup_expired(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileIdempotencyStore(tmpdir)

            # Create expired key file manually
            expired = IdempotencyKey(
                key="expired",
                expires_at=datetime.now() - timedelta(hours=1),
            )
            store.set(expired)

            removed = store.cleanup_expired()
            assert removed == 1


class TestIdempotencyManager:
    """Tests for IdempotencyManager."""

    def test_generate_key(self):
        manager = IdempotencyManager()

        key1 = manager.generate_key("arg1", "arg2", kwarg1="val1")
        key2 = manager.generate_key("arg1", "arg2", kwarg1="val1")
        key3 = manager.generate_key("arg1", "arg3", kwarg1="val1")

        assert key1 == key2  # Same args = same key
        assert key1 != key3  # Different args = different key

    def test_execute_with_idempotency(self):
        manager = IdempotencyManager()
        call_count = [0]

        def execute_fn():
            call_count[0] += 1
            return TransactionResult(
                transaction_id="txn_1",
                state=TransactionState.COMMITTED,
            )

        result1, cached1 = manager.execute_with_idempotency("key-1", execute_fn)
        result2, cached2 = manager.execute_with_idempotency("key-1", execute_fn)

        assert not cached1
        assert cached2
        assert call_count[0] == 1
        assert result1.transaction_id == result2.transaction_id

    def test_execute_failure_removes_key(self):
        manager = IdempotencyManager()

        def failing_fn():
            raise RuntimeError("Failed")

        with pytest.raises(RuntimeError):
            manager.execute_with_idempotency("key-1", failing_fn)

        # Key should be removed, allowing retry
        assert manager.store.get("key-1") is None

    def test_invalidate(self):
        manager = IdempotencyManager()

        manager.execute_with_idempotency(
            "key-1",
            lambda: TransactionResult("txn_1", TransactionState.COMMITTED),
        )

        assert manager.invalidate("key-1")
        assert manager.store.get("key-1") is None


class TestIdempotentDecorator:
    """Tests for @idempotent decorator."""

    def test_decorator_makes_function_idempotent(self):
        call_count = [0]

        @idempotent(key_fn=lambda x: f"key-{x}")
        def my_function(x: str) -> TransactionResult:
            call_count[0] += 1
            return TransactionResult(
                transaction_id="txn_1",
                state=TransactionState.COMMITTED,
            )

        result1, cached1 = my_function("arg1")
        result2, cached2 = my_function("arg1")
        result3, cached3 = my_function("arg2")

        assert not cached1
        assert cached2
        assert not cached3
        assert call_count[0] == 2


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the transaction framework."""

    def test_full_saga_with_compensation(self):
        """Test a complete saga with failure and compensation."""
        saga = SagaOrchestrator(name="full_saga")

        action1 = CompensatableSimpleAction(name="step1")
        action2 = CompensatableSimpleAction(name="step2")
        action3 = SimpleAction(name="step3", succeed=False)  # Will fail

        saga.add_step("step1", action1)
        saga.add_step("step2", action2)
        saga.add_step("step3", action3)

        result = saga.execute(MockCheckpointResult())

        assert not result.success
        assert action1.executed and action1.compensated
        assert action2.executed and action2.compensated

    def test_transactional_executor_with_multiple_boundaries(self):
        """Test executor with multiple transaction boundaries."""
        executor = TransactionalExecutor()

        action1 = CompensatableSimpleAction(name="boundary1_action")
        action2 = SimpleAction(name="boundary2_action")

        with executor.transaction("boundary1") as txn:
            txn.add_action(action1)

        with executor.transaction("boundary2") as txn:
            txn.add_action(action2)

        result = executor.execute(MockCheckpointResult())

        assert result.success
        assert action1.executed
        assert action2.executed

    def test_idempotent_transaction(self):
        """Test combining idempotency with transactions."""
        idempotency_manager = IdempotencyManager()
        executor = TransactionalExecutor()

        execution_count = [0]

        def run_transaction():
            execution_count[0] += 1
            with executor.transaction() as txn:
                txn.add_action(SimpleAction())
            return executor.execute(MockCheckpointResult())

        # First execution
        result1, cached1 = idempotency_manager.execute_with_idempotency(
            "txn-key",
            run_transaction,
        )

        # Second execution (should be cached)
        result2, cached2 = idempotency_manager.execute_with_idempotency(
            "txn-key",
            run_transaction,
        )

        assert not cached1
        assert cached2
        assert execution_count[0] == 1

    def test_nested_transactions(self):
        """Test nested transaction boundaries."""
        executor = TransactionalExecutor()

        with executor.transaction("outer") as outer:
            outer.add_action(SimpleAction(name="outer_action"))

            inner = outer.create_nested("inner")
            inner.add_action(SimpleAction(name="inner_action"))

        # Nested boundaries are created but not auto-added
        assert len(executor._boundaries) == 1
        assert executor._boundaries[0].name == "outer"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
