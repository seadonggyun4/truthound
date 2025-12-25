"""Tests for async checkpoint implementation.

This module tests the async checkpoint, runner, and action implementations.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Configure pytest-asyncio
pytest_plugins = ('pytest_asyncio',)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_data_file(tmp_path: Path) -> Path:
    """Create sample CSV for testing."""
    content = """id,name,email,age
1,Alice,alice@example.com,30
2,Bob,bob@example.com,25
3,Charlie,,35
4,Diana,diana@example.com,-5
"""
    data_file = tmp_path / "test_data.csv"
    data_file.write_text(content)
    return data_file


@dataclass
class MockStatistics:
    """Mock validation statistics."""
    total_issues: int = 5
    critical_issues: int = 1
    high_issues: int = 2
    medium_issues: int = 1
    low_issues: int = 1
    total_records: int = 100
    pass_rate: float = 0.95


@pytest.fixture
def mock_checkpoint_result():
    """Create a mock CheckpointResult."""
    from truthound.checkpoint.checkpoint import CheckpointResult, CheckpointStatus

    stats = MockStatistics()

    validation_result = MagicMock()
    validation_result.statistics = stats

    return CheckpointResult(
        run_id="test_run_001",
        checkpoint_name="test_checkpoint",
        run_time=datetime.now(),
        status=CheckpointStatus.WARNING,
        validation_result=validation_result,
        data_asset="test_data.csv",
        duration_ms=1500.0,
    )


# =============================================================================
# AsyncBaseAction Tests
# =============================================================================


class TestAsyncBaseAction:
    """Tests for AsyncBaseAction base class."""

    @pytest.mark.asyncio
    async def test_async_action_execution(self, mock_checkpoint_result):
        """Test basic async action execution."""
        from truthound.checkpoint.async_base import AsyncBaseAction
        from truthound.checkpoint.actions.base import (
            ActionConfig,
            ActionResult,
            ActionStatus,
        )

        class TestAction(AsyncBaseAction[ActionConfig]):
            action_type = "test"

            @classmethod
            def _default_config(cls) -> ActionConfig:
                return ActionConfig()

            async def _execute_async(self, checkpoint_result) -> ActionResult:
                await asyncio.sleep(0.01)  # Simulate async work
                return ActionResult(
                    action_name=self.name,
                    action_type=self.action_type,
                    status=ActionStatus.SUCCESS,
                    message="Test completed",
                )

        action = TestAction()
        result = await action.execute_async(mock_checkpoint_result)

        assert result.status == ActionStatus.SUCCESS
        assert result.duration_ms > 0

    @pytest.mark.asyncio
    async def test_async_action_timeout(self, mock_checkpoint_result):
        """Test async action timeout handling."""
        from truthound.checkpoint.async_base import AsyncBaseAction
        from truthound.checkpoint.actions.base import (
            ActionConfig,
            ActionResult,
            ActionStatus,
        )

        class SlowAction(AsyncBaseAction[ActionConfig]):
            action_type = "slow"

            @classmethod
            def _default_config(cls) -> ActionConfig:
                return ActionConfig(timeout_seconds=0.1)

            async def _execute_async(self, checkpoint_result) -> ActionResult:
                await asyncio.sleep(10)  # Will timeout
                return ActionResult(
                    action_name=self.name,
                    action_type=self.action_type,
                    status=ActionStatus.SUCCESS,
                )

        action = SlowAction(timeout_seconds=0.1)
        result = await action.execute_async(mock_checkpoint_result)

        assert result.status == ActionStatus.ERROR
        assert "timeout" in result.error.lower() or "timed out" in result.error.lower()

    @pytest.mark.asyncio
    async def test_async_action_retry(self, mock_checkpoint_result):
        """Test async action retry logic."""
        from truthound.checkpoint.async_base import AsyncBaseAction
        from truthound.checkpoint.actions.base import (
            ActionConfig,
            ActionResult,
            ActionStatus,
        )

        attempt_count = 0

        class FailingAction(AsyncBaseAction[ActionConfig]):
            action_type = "failing"

            @classmethod
            def _default_config(cls) -> ActionConfig:
                return ActionConfig(retry_count=2, retry_delay_seconds=0.01)

            async def _execute_async(self, checkpoint_result) -> ActionResult:
                nonlocal attempt_count
                attempt_count += 1
                if attempt_count < 3:
                    raise ValueError("Temporary failure")
                return ActionResult(
                    action_name=self.name,
                    action_type=self.action_type,
                    status=ActionStatus.SUCCESS,
                )

        action = FailingAction(retry_count=2, retry_delay_seconds=0.01)
        result = await action.execute_async(mock_checkpoint_result)

        assert result.status == ActionStatus.SUCCESS
        assert attempt_count == 3

    @pytest.mark.asyncio
    async def test_async_action_skip_on_condition(self, mock_checkpoint_result):
        """Test action skip based on notify_on condition."""
        from truthound.checkpoint.async_base import AsyncBaseAction
        from truthound.checkpoint.actions.base import (
            ActionConfig,
            ActionResult,
            ActionStatus,
            NotifyCondition,
        )
        from truthound.checkpoint.checkpoint import CheckpointStatus

        class FailureOnlyAction(AsyncBaseAction[ActionConfig]):
            action_type = "failure_only"

            @classmethod
            def _default_config(cls) -> ActionConfig:
                return ActionConfig(notify_on=NotifyCondition.FAILURE)

            async def _execute_async(self, checkpoint_result) -> ActionResult:
                return ActionResult(
                    action_name=self.name,
                    action_type=self.action_type,
                    status=ActionStatus.SUCCESS,
                )

        # Mock result with SUCCESS status
        mock_checkpoint_result.status = CheckpointStatus.SUCCESS

        action = FailureOnlyAction(notify_on="failure")
        result = await action.execute_async(mock_checkpoint_result)

        assert result.status == ActionStatus.SKIPPED


# =============================================================================
# SyncActionAdapter Tests
# =============================================================================


class TestSyncActionAdapter:
    """Tests for SyncActionAdapter."""

    @pytest.mark.asyncio
    async def test_sync_to_async_adaptation(self, mock_checkpoint_result):
        """Test wrapping sync action for async context."""
        from truthound.checkpoint.async_base import SyncActionAdapter, adapt_to_async
        from truthound.checkpoint.actions.base import (
            BaseAction,
            ActionConfig,
            ActionResult,
            ActionStatus,
        )

        class SyncAction(BaseAction[ActionConfig]):
            action_type = "sync_test"

            @classmethod
            def _default_config(cls) -> ActionConfig:
                return ActionConfig()

            def _execute(self, checkpoint_result) -> ActionResult:
                return ActionResult(
                    action_name=self.name,
                    action_type=self.action_type,
                    status=ActionStatus.SUCCESS,
                    message="Sync action completed",
                )

        sync_action = SyncAction()
        async_action = adapt_to_async(sync_action)

        assert isinstance(async_action, SyncActionAdapter)

        result = await async_action.execute_async(mock_checkpoint_result)
        assert result.status == ActionStatus.SUCCESS


# =============================================================================
# Execution Strategy Tests
# =============================================================================


class TestExecutionStrategies:
    """Tests for action execution strategies."""

    @pytest.mark.asyncio
    async def test_sequential_strategy(self, mock_checkpoint_result):
        """Test sequential execution strategy."""
        from truthound.checkpoint.async_base import (
            AsyncBaseAction,
            SequentialStrategy,
        )
        from truthound.checkpoint.actions.base import (
            ActionConfig,
            ActionResult,
            ActionStatus,
        )

        execution_order = []

        class OrderedAction(AsyncBaseAction[ActionConfig]):
            action_type = "ordered"

            def __init__(self, order: int, **kwargs):
                super().__init__(**kwargs)
                self.order = order

            @classmethod
            def _default_config(cls) -> ActionConfig:
                return ActionConfig()

            async def _execute_async(self, checkpoint_result) -> ActionResult:
                execution_order.append(self.order)
                await asyncio.sleep(0.01)
                return ActionResult(
                    action_name=self.name,
                    action_type=self.action_type,
                    status=ActionStatus.SUCCESS,
                )

        actions = [OrderedAction(i) for i in range(3)]
        strategy = SequentialStrategy()

        await strategy.execute(actions, mock_checkpoint_result)

        assert execution_order == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_concurrent_strategy(self, mock_checkpoint_result):
        """Test concurrent execution strategy."""
        from truthound.checkpoint.async_base import (
            AsyncBaseAction,
            ConcurrentStrategy,
        )
        from truthound.checkpoint.actions.base import (
            ActionConfig,
            ActionResult,
            ActionStatus,
        )

        start_times = []

        class TimedAction(AsyncBaseAction[ActionConfig]):
            action_type = "timed"

            @classmethod
            def _default_config(cls) -> ActionConfig:
                return ActionConfig()

            async def _execute_async(self, checkpoint_result) -> ActionResult:
                start_times.append(asyncio.get_event_loop().time())
                await asyncio.sleep(0.1)
                return ActionResult(
                    action_name=self.name,
                    action_type=self.action_type,
                    status=ActionStatus.SUCCESS,
                )

        actions = [TimedAction() for _ in range(3)]
        strategy = ConcurrentStrategy(max_concurrency=3)

        await strategy.execute(actions, mock_checkpoint_result)

        # All should start at approximately the same time
        time_spread = max(start_times) - min(start_times)
        assert time_spread < 0.05  # Less than 50ms spread

    @pytest.mark.asyncio
    async def test_concurrent_strategy_with_limit(self, mock_checkpoint_result):
        """Test concurrent strategy with concurrency limit."""
        from truthound.checkpoint.async_base import (
            AsyncBaseAction,
            ConcurrentStrategy,
        )
        from truthound.checkpoint.actions.base import (
            ActionConfig,
            ActionResult,
            ActionStatus,
        )

        active_count = 0
        max_active = 0

        class CountingAction(AsyncBaseAction[ActionConfig]):
            action_type = "counting"

            @classmethod
            def _default_config(cls) -> ActionConfig:
                return ActionConfig()

            async def _execute_async(self, checkpoint_result) -> ActionResult:
                nonlocal active_count, max_active
                active_count += 1
                max_active = max(max_active, active_count)
                await asyncio.sleep(0.05)
                active_count -= 1
                return ActionResult(
                    action_name=self.name,
                    action_type=self.action_type,
                    status=ActionStatus.SUCCESS,
                )

        actions = [CountingAction() for _ in range(5)]
        strategy = ConcurrentStrategy(max_concurrency=2)

        await strategy.execute(actions, mock_checkpoint_result)

        assert max_active <= 2


# =============================================================================
# AsyncCheckpoint Tests
# =============================================================================


class TestAsyncCheckpoint:
    """Tests for AsyncCheckpoint class."""

    @pytest.mark.asyncio
    async def test_async_checkpoint_run(self, sample_data_file):
        """Test basic async checkpoint execution."""
        from truthound.checkpoint import AsyncCheckpoint

        checkpoint = AsyncCheckpoint(
            name="test_async",
            data_source=str(sample_data_file),
            validators=["null"],
        )

        result = await checkpoint.run_async()

        assert result.checkpoint_name == "test_async"
        assert result.run_id is not None
        assert result.duration_ms > 0

    @pytest.mark.asyncio
    async def test_async_checkpoint_with_actions(self, sample_data_file, tmp_path):
        """Test async checkpoint with actions."""
        from truthound.checkpoint import AsyncCheckpoint
        from truthound.checkpoint.async_actions import AsyncStoreValidationResult

        store_path = tmp_path / "results"

        checkpoint = AsyncCheckpoint(
            name="test_with_actions",
            data_source=str(sample_data_file),
            validators=["null"],
            actions=[
                AsyncStoreValidationResult(store_path=str(store_path)),
            ],
        )

        result = await checkpoint.run_async()

        assert len(result.action_results) == 1
        assert store_path.exists()

    @pytest.mark.asyncio
    async def test_async_checkpoint_concurrent_actions(self, sample_data_file):
        """Test concurrent action execution in checkpoint."""
        from truthound.checkpoint import AsyncCheckpoint
        from truthound.checkpoint.async_actions import AsyncCustomAction

        execution_times = []

        async def timed_handler(checkpoint_result):
            start = asyncio.get_event_loop().time()
            await asyncio.sleep(0.1)
            execution_times.append(asyncio.get_event_loop().time() - start)
            return {"executed": True}

        checkpoint = AsyncCheckpoint(
            name="concurrent_test",
            data_source=str(sample_data_file),
            validators=["null"],
            actions=[
                AsyncCustomAction(callback=timed_handler),
                AsyncCustomAction(callback=timed_handler),
                AsyncCustomAction(callback=timed_handler),
            ],
            execution_strategy="concurrent",
        )

        start = asyncio.get_event_loop().time()
        result = await checkpoint.run_async()
        total_time = asyncio.get_event_loop().time() - start

        # Actions should run concurrently, so total time < 3 * 0.1s
        assert len(result.action_results) == 3
        # Allow some overhead, but should be faster than sequential
        assert total_time < 0.5

    @pytest.mark.asyncio
    async def test_async_checkpoint_callbacks(self, sample_data_file):
        """Test async checkpoint callbacks."""
        from truthound.checkpoint import AsyncCheckpoint

        callback_called = False
        received_result = None

        async def on_complete(result):
            nonlocal callback_called, received_result
            callback_called = True
            received_result = result

        checkpoint = AsyncCheckpoint(
            name="callback_test",
            data_source=str(sample_data_file),
            validators=["null"],
            on_complete=on_complete,
        )

        result = await checkpoint.run_async()

        assert callback_called
        assert received_result is result

    @pytest.mark.asyncio
    async def test_to_async_checkpoint_conversion(self, sample_data_file):
        """Test converting sync Checkpoint to AsyncCheckpoint."""
        from truthound.checkpoint import Checkpoint, to_async_checkpoint

        sync_checkpoint = Checkpoint(
            name="sync_to_async",
            data_source=str(sample_data_file),
            validators=["null"],
        )

        async_checkpoint = to_async_checkpoint(sync_checkpoint)

        result = await async_checkpoint.run_async()

        assert result.checkpoint_name == "sync_to_async"


# =============================================================================
# AsyncCheckpointRunner Tests
# =============================================================================


class TestAsyncCheckpointRunner:
    """Tests for AsyncCheckpointRunner."""

    @pytest.mark.asyncio
    async def test_runner_run_once(self, sample_data_file):
        """Test running single checkpoint."""
        from truthound.checkpoint import AsyncCheckpoint, AsyncCheckpointRunner

        checkpoint = AsyncCheckpoint(
            name="runner_test",
            data_source=str(sample_data_file),
            validators=["null"],
        )

        runner = AsyncCheckpointRunner()
        runner.add_checkpoint(checkpoint)

        result = await runner.run_once_async("runner_test")

        assert result.checkpoint_name == "runner_test"

    @pytest.mark.asyncio
    async def test_runner_run_all(self, sample_data_file):
        """Test running all checkpoints."""
        from truthound.checkpoint import AsyncCheckpoint, AsyncCheckpointRunner

        runner = AsyncCheckpointRunner()

        for i in range(3):
            runner.add_checkpoint(AsyncCheckpoint(
                name=f"test_{i}",
                data_source=str(sample_data_file),
                validators=["null"],
            ))

        results = await runner.run_all_async()

        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_runner_result_callback(self, sample_data_file):
        """Test runner result callback."""
        from truthound.checkpoint import AsyncCheckpoint, AsyncCheckpointRunner

        results_received = []

        async def on_result(result):
            results_received.append(result)

        runner = AsyncCheckpointRunner(result_callback=on_result)
        runner.add_checkpoint(AsyncCheckpoint(
            name="callback_test",
            data_source=str(sample_data_file),
            validators=["null"],
        ))

        await runner.run_once_async("callback_test")

        assert len(results_received) == 1


# =============================================================================
# Async Actions Tests
# =============================================================================


class TestAsyncWebhookAction:
    """Tests for AsyncWebhookAction."""

    @pytest.mark.asyncio
    async def test_webhook_validation(self):
        """Test webhook config validation."""
        from truthound.checkpoint.async_actions import AsyncWebhookAction

        action = AsyncWebhookAction(url="")
        errors = action.validate_config()

        assert len(errors) > 0
        assert any("url" in e.lower() for e in errors)

    @pytest.mark.asyncio
    async def test_webhook_payload_building(self, mock_checkpoint_result):
        """Test webhook payload construction."""
        from truthound.checkpoint.async_actions import AsyncWebhookAction

        action = AsyncWebhookAction(
            url="https://example.com/webhook",
            include_full_result=False,
        )

        payload = action._build_payload(mock_checkpoint_result)

        assert payload["checkpoint"] == "test_checkpoint"
        assert payload["status"] == "warning"
        assert "summary" in payload
        assert "full_result" not in payload


class TestAsyncSlackNotification:
    """Tests for AsyncSlackNotification."""

    @pytest.mark.asyncio
    async def test_slack_payload_building(self, mock_checkpoint_result):
        """Test Slack message payload construction."""
        from truthound.checkpoint.async_actions import AsyncSlackNotification

        action = AsyncSlackNotification(
            webhook_url="https://hooks.slack.com/test",
            channel="#data-quality",
            include_details=True,
        )

        payload = action._build_payload(mock_checkpoint_result)

        assert payload["username"] == "Truthound"
        assert payload["channel"] == "#data-quality"
        assert "attachments" in payload


class TestAsyncStoreResult:
    """Tests for AsyncStoreValidationResult."""

    @pytest.mark.asyncio
    async def test_store_to_file(self, mock_checkpoint_result, tmp_path):
        """Test storing results to file."""
        from truthound.checkpoint.async_actions import AsyncStoreValidationResult

        store_path = tmp_path / "results"

        action = AsyncStoreValidationResult(
            store_path=str(store_path),
            format="json",
        )

        result = await action.execute_async(mock_checkpoint_result)

        assert result.status.value == "success"
        assert store_path.exists()

        # Verify JSON content
        json_files = list(store_path.rglob("*.json"))
        assert len(json_files) >= 1


# =============================================================================
# CheckpointPool Tests
# =============================================================================


class TestCheckpointPool:
    """Tests for CheckpointPool."""

    @pytest.mark.asyncio
    async def test_pool_submit(self, sample_data_file):
        """Test pool checkpoint submission."""
        from truthound.checkpoint import AsyncCheckpoint, CheckpointPool

        checkpoint = AsyncCheckpoint(
            name="pool_test",
            data_source=str(sample_data_file),
            validators=["null"],
        )

        async with CheckpointPool(workers=2) as pool:
            result = await pool.submit(checkpoint)

        assert result.checkpoint_name == "pool_test"

    @pytest.mark.asyncio
    async def test_pool_submit_many(self, sample_data_file):
        """Test pool batch submission."""
        from truthound.checkpoint import AsyncCheckpoint, CheckpointPool

        checkpoints = [
            AsyncCheckpoint(
                name=f"batch_{i}",
                data_source=str(sample_data_file),
                validators=["null"],
            )
            for i in range(3)
        ]

        async with CheckpointPool(workers=3) as pool:
            results = await pool.submit_many(checkpoints)

        assert len(results) == 3
        assert all(r.checkpoint_name.startswith("batch_") for r in results)


# =============================================================================
# Decorator Tests
# =============================================================================


class TestAsyncDecorators:
    """Tests for async decorators."""

    @pytest.mark.asyncio
    async def test_with_retry_decorator(self):
        """Test retry decorator."""
        from truthound.checkpoint.async_base import with_retry

        call_count = 0

        @with_retry(max_retries=2, delay=0.01)
        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary error")
            return "success"

        result = await flaky_func()

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_with_timeout_decorator(self):
        """Test timeout decorator."""
        from truthound.checkpoint.async_base import with_timeout

        @with_timeout(seconds=0.1)
        async def slow_func():
            await asyncio.sleep(10)
            return "done"

        with pytest.raises(asyncio.TimeoutError):
            await slow_func()

    @pytest.mark.asyncio
    async def test_with_semaphore_decorator(self):
        """Test semaphore decorator."""
        from truthound.checkpoint.async_base import with_semaphore

        semaphore = asyncio.Semaphore(2)
        active_count = 0
        max_active = 0

        @with_semaphore(semaphore)
        async def limited_func():
            nonlocal active_count, max_active
            active_count += 1
            max_active = max(max_active, active_count)
            await asyncio.sleep(0.05)
            active_count -= 1

        await asyncio.gather(*[limited_func() for _ in range(5)])

        assert max_active <= 2


# =============================================================================
# Integration Tests
# =============================================================================


class TestAsyncIntegration:
    """Integration tests for async checkpoint system."""

    @pytest.mark.asyncio
    async def test_run_checkpoints_parallel(self, sample_data_file):
        """Test parallel checkpoint execution."""
        from truthound.checkpoint import (
            AsyncCheckpoint,
            run_checkpoints_parallel,
        )

        checkpoints = [
            AsyncCheckpoint(
                name=f"parallel_{i}",
                data_source=str(sample_data_file),
                validators=["null"],
            )
            for i in range(5)
        ]

        start = asyncio.get_event_loop().time()
        results = await run_checkpoints_parallel(
            checkpoints, max_concurrent=5
        )
        duration = asyncio.get_event_loop().time() - start

        assert len(results) == 5
        # Should complete faster than sequential
        assert duration < 10  # Very generous timeout

    @pytest.mark.asyncio
    async def test_mixed_sync_async_actions(self, sample_data_file):
        """Test checkpoint with mixed sync/async actions."""
        from truthound.checkpoint import AsyncCheckpoint
        from truthound.checkpoint.actions import StoreValidationResult
        from truthound.checkpoint.async_actions import AsyncCustomAction

        async def async_handler(result):
            return {"async": True}

        checkpoint = AsyncCheckpoint(
            name="mixed_actions",
            data_source=str(sample_data_file),
            validators=["null"],
            actions=[
                # Sync action (will be adapted)
                StoreValidationResult(store_path="/tmp/truthound_test"),
                # Native async action
                AsyncCustomAction(callback=async_handler),
            ],
        )

        result = await checkpoint.run_async()

        assert len(result.action_results) == 2

    @pytest.mark.asyncio
    async def test_checkpoint_context_manager(self, sample_data_file):
        """Test async checkpoint as context manager."""
        from truthound.checkpoint import AsyncCheckpoint

        async with AsyncCheckpoint(
            name="context_test",
            data_source=str(sample_data_file),
            validators=["null"],
        ) as checkpoint:
            result = await checkpoint.run_async()

        assert result.checkpoint_name == "context_test"
