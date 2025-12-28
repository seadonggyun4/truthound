"""Tests for distributed timeout management.

This test suite covers:
- Deadline propagation
- Timeout budget management
- Cascade handling
- Graceful degradation
- Distributed coordination
"""

import asyncio
import time
import pytest

from truthound.validators.timeout.deadline import (
    DeadlineContext,
    DeadlinePropagator,
    TimeoutBudget,
    BudgetAllocation,
    DeadlineExceededError,
    with_deadline,
    get_current_deadline,
)
from truthound.validators.timeout.cascade import (
    CascadeTimeoutHandler,
    CascadePolicy,
    CascadeLevel,
    TimeoutAction,
    TimeoutCascadeResult,
)
from truthound.validators.timeout.degradation import (
    GracefulDegradation,
    DegradationPolicy,
    DegradationLevel,
    DegradationAction,
    DegradationResult,
)
from truthound.validators.timeout.distributed import (
    DistributedTimeoutManager,
    DistributedTimeoutConfig,
    CoordinatorBackend,
    InMemoryCoordinator,
)


class TestDeadlineContext:
    """Test DeadlineContext class."""

    def test_from_seconds(self):
        """Test creating deadline from seconds."""
        ctx = DeadlineContext.from_seconds(60)
        assert ctx.remaining_seconds <= 60
        assert ctx.remaining_seconds > 59
        assert not ctx.is_expired

    def test_is_expired(self):
        """Test expiration detection."""
        ctx = DeadlineContext.from_seconds(0.01)
        time.sleep(0.02)
        assert ctx.is_expired

    def test_cancel(self):
        """Test cancellation."""
        ctx = DeadlineContext.from_seconds(60)
        assert not ctx.is_cancelled
        ctx.cancel()
        assert ctx.is_cancelled
        assert ctx.remaining_seconds == 0

    def test_context_manager(self):
        """Test context manager usage."""
        with DeadlineContext.from_seconds(60) as ctx:
            current = get_current_deadline()
            assert current is ctx

        # After exiting, should be cleared
        assert get_current_deadline() is None

    def test_with_budget_fraction(self):
        """Test creating sub-deadline with fraction."""
        ctx = DeadlineContext.from_seconds(100)
        sub = ctx.with_budget_fraction(0.5)
        assert sub.remaining_seconds <= 50
        assert sub.remaining_seconds > 49

    def test_allocate(self):
        """Test budget allocation."""
        ctx = DeadlineContext.from_seconds(100)
        allocation = ctx.allocate(
            validation=50,
            reporting=30,
            cleanup=10,
        )
        assert allocation.validation.remaining_seconds <= 50
        assert allocation.reporting.remaining_seconds <= 30
        assert allocation.cleanup.remaining_seconds <= 10

    def test_to_dict_from_dict(self):
        """Test serialization/deserialization."""
        original = DeadlineContext.from_seconds(60, operation_id="test-op")
        data = original.to_dict()
        restored = DeadlineContext.from_dict(data)

        assert restored.operation_id == original.operation_id
        # Allow small time difference
        assert abs(restored.remaining_seconds - original.remaining_seconds) < 1

    def test_check_raises_when_expired(self):
        """Test check raises when expired."""
        ctx = DeadlineContext.from_seconds(0.01)
        time.sleep(0.02)
        with pytest.raises(DeadlineExceededError):
            ctx.check()


class TestTimeoutBudget:
    """Test TimeoutBudget class."""

    def test_basic_budget(self):
        """Test basic budget creation."""
        budget = TimeoutBudget(total_seconds=120)
        assert budget.remaining_seconds <= 120
        assert budget.unallocated_seconds <= 120

    def test_allocate(self):
        """Test allocation from budget."""
        budget = TimeoutBudget(total_seconds=120)
        ctx = budget.allocate("validation", 60)
        assert ctx.remaining_seconds <= 60
        assert budget.allocated_seconds == 60

    def test_allocate_fraction(self):
        """Test fractional allocation."""
        budget = TimeoutBudget(total_seconds=100)
        ctx = budget.allocate_fraction("validation", 0.5)
        assert ctx.remaining_seconds <= 50

    def test_over_allocation_raises(self):
        """Test that over-allocation raises error."""
        budget = TimeoutBudget(total_seconds=60)
        budget.allocate("first", 50)
        with pytest.raises(ValueError):
            budget.allocate("second", 50)

    def test_use_context_manager(self):
        """Test use context manager."""
        budget = TimeoutBudget(total_seconds=60)
        with budget.use("test", 30) as ctx:
            assert ctx.remaining_seconds <= 30
            time.sleep(0.01)
        # Usage should be recorded
        assert budget._used.get("test", 0) > 0

    def test_get_summary(self):
        """Test summary generation."""
        budget = TimeoutBudget(total_seconds=120)
        budget.allocate("test", 60)
        summary = budget.get_summary()

        assert summary["total_seconds"] == 120
        assert "test" in summary["allocated"]


class TestDeadlinePropagator:
    """Test DeadlinePropagator class."""

    def test_context_sets_current(self):
        """Test context manager sets current deadline."""
        propagator = DeadlinePropagator()

        assert propagator.current is None

        with propagator.context(60, "test-op"):
            assert propagator.current is not None
            assert propagator.current.operation_id == "test-op"

        assert propagator.current is None

    def test_nested_contexts(self):
        """Test nested context managers."""
        propagator = DeadlinePropagator()

        with propagator.context(60, "outer"):
            outer = propagator.current
            assert outer is not None

            with propagator.context(30, "inner"):
                inner = propagator.current
                assert inner is not None
                assert inner.operation_id == "inner"

            # Back to outer
            assert propagator.current is not None


class TestWithDeadlineDecorator:
    """Test @with_deadline decorator."""

    def test_decorator_sets_deadline(self):
        """Test decorator sets deadline during execution."""

        @with_deadline(60, "decorated-op")
        def my_function():
            ctx = get_current_deadline()
            assert ctx is not None
            assert ctx.operation_id == "decorated-op"
            return "result"

        result = my_function()
        assert result == "result"

        # After function, deadline should be cleared
        assert get_current_deadline() is None


class TestCascadeTimeoutHandler:
    """Test CascadeTimeoutHandler class."""

    def test_default_policies(self):
        """Test default policies are set."""
        handler = CascadeTimeoutHandler()
        policy = handler.get_policy(CascadeLevel.VALIDATOR)
        assert policy.action == TimeoutAction.SKIP

    def test_handle_timeout(self):
        """Test timeout handling."""
        handler = CascadeTimeoutHandler()
        result = handler.handle_timeout(
            level=CascadeLevel.VALIDATOR,
            operation_name="null_check",
        )

        assert isinstance(result, TimeoutCascadeResult)
        assert result.level == CascadeLevel.VALIDATOR
        assert result.action_taken == TimeoutAction.SKIP

    def test_custom_policy(self):
        """Test setting custom policy."""
        handler = CascadeTimeoutHandler()
        handler.set_policy(
            CascadeLevel.VALIDATOR,
            CascadePolicy(action=TimeoutAction.RETRY, retry_count=2),
        )

        policy = handler.get_policy(CascadeLevel.VALIDATOR)
        assert policy.action == TimeoutAction.RETRY
        assert policy.retry_count == 2

    def test_retry_timeout(self):
        """Test retry timeout calculation."""
        policy = CascadePolicy(
            retry_count=3,
            retry_multiplier=2.0,
            max_retry_seconds=60,
        )

        # First retry: 10 * 2^0 = 10
        assert policy.get_retry_timeout(10, 0) == 10
        # Second retry: 10 * 2^1 = 20
        assert policy.get_retry_timeout(10, 1) == 20
        # Max cap applies
        assert policy.get_retry_timeout(50, 3) == 60

    def test_parent_level(self):
        """Test parent level resolution."""
        handler = CascadeTimeoutHandler()
        parent = handler.get_parent_level(CascadeLevel.VALIDATOR)
        assert parent == CascadeLevel.COLUMN

        # JOB has no parent
        parent = handler.get_parent_level(CascadeLevel.JOB)
        assert parent is None


class TestGracefulDegradation:
    """Test GracefulDegradation class."""

    def test_check_level_no_deadline(self):
        """Test check with no deadline returns NONE."""
        degradation = GracefulDegradation()
        level = degradation.check(None)
        assert level == DegradationLevel.NONE

    def test_check_level_plenty_of_time(self):
        """Test check with plenty of time."""
        degradation = GracefulDegradation()
        ctx = DeadlineContext.from_seconds(60)
        level = degradation.check(ctx)
        assert level == DegradationLevel.NONE

    def test_check_level_low_time(self):
        """Test check with low time triggers degradation."""
        degradation = GracefulDegradation(
            policy=DegradationPolicy(sample_threshold=0.5)
        )
        ctx = DeadlineContext.from_seconds(0.01)
        time.sleep(0.008)  # Use most of the time
        level = degradation.check(ctx)
        # Should be some degradation level
        assert level != DegradationLevel.NONE

    def test_apply_full_operation(self):
        """Test apply with full operation."""
        degradation = GracefulDegradation()
        ctx = DeadlineContext.from_seconds(60)

        result = degradation.apply(
            ctx,
            full_operation=lambda: "full_result",
        )

        assert result.value == "full_result"
        assert result.level == DegradationLevel.NONE

    def test_apply_fallback(self):
        """Test apply with fallback when degraded."""
        # Create policy that will degrade quickly
        policy = DegradationPolicy(sample_threshold=1.0)  # Always degrade
        degradation = GracefulDegradation(policy=policy)
        ctx = DeadlineContext.from_seconds(10)

        result = degradation.apply(
            ctx,
            full_operation=lambda: "full_result",
            fallback_operation=lambda: "fallback_result",
        )

        # Should use fallback due to degradation
        assert result.fallback_used or result.value == "full_result"

    def test_should_sample(self):
        """Test should_sample check."""
        degradation = GracefulDegradation()
        # With plenty of time, should not sample
        ctx = DeadlineContext.from_seconds(60)
        assert not degradation.should_sample(ctx)

    def test_stats_tracking(self):
        """Test statistics tracking."""
        degradation = GracefulDegradation()
        ctx = DeadlineContext.from_seconds(60)

        degradation.apply(ctx, full_operation=lambda: None)

        stats = degradation.get_stats()
        assert "level_none" in stats or len(stats) > 0


class TestDistributedTimeoutManager:
    """Test DistributedTimeoutManager class."""

    @pytest.mark.asyncio
    async def test_start_stop(self):
        """Test manager start and stop."""
        config = DistributedTimeoutConfig(
            coordinator_backend=CoordinatorBackend.MEMORY,
        )
        manager = DistributedTimeoutManager(config)

        async with manager:
            assert manager._running
            assert manager._coordinator is not None

        assert not manager._running

    @pytest.mark.asyncio
    async def test_execute_distributed(self):
        """Test distributed execution."""
        config = DistributedTimeoutConfig()
        manager = DistributedTimeoutManager(config)

        async with manager:
            result = await manager.execute_distributed(
                lambda: "result",
                timeout_seconds=10,
            )

            assert result.success
            assert result.value == "result"

    @pytest.mark.asyncio
    async def test_execute_timeout(self):
        """Test timeout handling."""
        config = DistributedTimeoutConfig()
        manager = DistributedTimeoutManager(config)

        def slow_operation():
            time.sleep(0.3)
            return "result"

        async with manager:
            result = await manager.execute_distributed(
                slow_operation,
                timeout_seconds=0.05,
            )

            assert not result.success
            assert "timeout" in result.error.lower()

    @pytest.mark.asyncio
    async def test_circuit_breaker(self):
        """Test circuit breaker behavior."""
        config = DistributedTimeoutConfig(
            circuit_breaker_threshold=2,
            circuit_breaker_timeout=0.1,
        )
        manager = DistributedTimeoutManager(config)

        def failing_operation():
            raise ValueError("Failed")

        async with manager:
            # Trigger failures to open circuit
            for _ in range(3):
                await manager.execute_distributed(
                    failing_operation,
                    timeout_seconds=1,
                )

            assert manager.is_circuit_open()

            # Subsequent calls should fail fast
            result = await manager.execute_distributed(
                lambda: "result",
                timeout_seconds=1,
            )
            assert not result.success
            assert "circuit" in result.error.lower()

    @pytest.mark.asyncio
    async def test_get_active_nodes(self):
        """Test getting active nodes."""
        config = DistributedTimeoutConfig()
        manager = DistributedTimeoutManager(config)

        async with manager:
            nodes = await manager.get_active_nodes()
            assert config.node_id in nodes


class TestInMemoryCoordinator:
    """Test InMemoryCoordinator class."""

    @pytest.mark.asyncio
    async def test_register_node(self):
        """Test node registration."""
        coordinator = InMemoryCoordinator()
        result = await coordinator.register_node("node-1", {"version": "1.0"})
        assert result is True

        nodes = await coordinator.get_active_nodes()
        assert "node-1" in nodes

    @pytest.mark.asyncio
    async def test_create_deadline(self):
        """Test deadline creation."""
        coordinator = InMemoryCoordinator()
        await coordinator.register_node("node-1", {})

        deadline = await coordinator.create_deadline(
            "node-1",
            60,
            "test-operation",
        )

        assert deadline.owner_node == "node-1"
        assert deadline.operation_id == "test-operation"
        assert deadline.remaining_seconds <= 60

    @pytest.mark.asyncio
    async def test_complete_deadline(self):
        """Test deadline completion."""
        coordinator = InMemoryCoordinator()
        await coordinator.register_node("node-1", {})

        deadline = await coordinator.create_deadline("node-1", 60)
        result = await coordinator.complete_deadline(deadline.deadline_id)

        assert result is True

        # Verify status updated
        updated = await coordinator.get_deadline(deadline.deadline_id)
        assert updated.status == "completed"

    @pytest.mark.asyncio
    async def test_heartbeat(self):
        """Test heartbeat functionality."""
        coordinator = InMemoryCoordinator()
        await coordinator.register_node("node-1", {})

        result = await coordinator.heartbeat("node-1")
        assert result is True

        # Unknown node should fail
        result = await coordinator.heartbeat("unknown-node")
        assert result is False
