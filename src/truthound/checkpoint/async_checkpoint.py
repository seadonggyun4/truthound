"""Async Checkpoint implementation.

This module provides AsyncCheckpoint, an async-native version of Checkpoint
that supports non-blocking validation pipelines.

Design Goals:
    1. API Compatibility: Mirror sync Checkpoint API where possible
    2. Flexibility: Support mixed sync/async actions
    3. Performance: Enable concurrent action execution
    4. Observability: Async-friendly hooks and callbacks
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Awaitable
from uuid import uuid4

from truthound.checkpoint.checkpoint import (
    Checkpoint,
    CheckpointConfig,
    CheckpointResult,
    CheckpointStatus,
)
from truthound.checkpoint.async_base import (
    AsyncBaseAction,
    AsyncExecutionContext,
    ConcurrentStrategy,
    ExecutionStrategy,
    SequentialStrategy,
    SyncActionAdapter,
    adapt_to_async,
)
from truthound.checkpoint.actions.base import (
    ActionResult,
    ActionStatus,
    BaseAction,
)

if TYPE_CHECKING:
    from truthound.checkpoint.triggers.base import BaseTrigger
    from truthound.datasources.base import BaseDataSource
    from truthound.validators.base import Validator


# Type aliases
ActionType = BaseAction[Any] | AsyncBaseAction[Any]
AsyncCallback = Callable[[CheckpointResult], Awaitable[None]]
SyncCallback = Callable[[CheckpointResult], None]
CallbackType = AsyncCallback | SyncCallback


@dataclass
class AsyncCheckpointConfig(CheckpointConfig):
    """Extended configuration for async checkpoints.

    Attributes:
        max_concurrent_actions: Max actions to run in parallel.
        action_timeout: Default timeout for each action.
        execution_strategy: How to execute actions (sequential, concurrent, pipeline).
        cancel_on_first_error: Stop remaining actions on first error.
        executor_workers: Number of threads for sync action execution.
    """

    max_concurrent_actions: int = 10
    action_timeout: float = 30.0
    execution_strategy: str = "concurrent"  # sequential, concurrent, pipeline
    cancel_on_first_error: bool = False
    executor_workers: int = 4


class AsyncCheckpoint:
    """Async-native checkpoint for non-blocking validation pipelines.

    AsyncCheckpoint provides the same functionality as Checkpoint but with
    native async/await support for high-throughput scenarios.

    Key Features:
        - Non-blocking validation execution
        - Concurrent action execution with configurable strategies
        - Mixed sync/async action support
        - Async callbacks and hooks
        - Graceful cancellation

    Example:
        >>> from truthound.checkpoint import AsyncCheckpoint
        >>> from truthound.checkpoint.async_actions import AsyncSlackNotification
        >>>
        >>> checkpoint = AsyncCheckpoint(
        ...     name="async_validation",
        ...     data_source="large_dataset.parquet",
        ...     actions=[
        ...         AsyncSlackNotification(webhook_url="..."),
        ...         AsyncWebhookAction(url="..."),
        ...     ],
        ...     max_concurrent_actions=5,
        ... )
        >>>
        >>> # Run asynchronously
        >>> result = await checkpoint.run_async()
        >>>
        >>> # Or run multiple checkpoints concurrently
        >>> results = await asyncio.gather(
        ...     checkpoint1.run_async(),
        ...     checkpoint2.run_async(),
        ...     checkpoint3.run_async(),
        ... )
    """

    def __init__(
        self,
        name: str | None = None,
        config: AsyncCheckpointConfig | None = None,
        data_source: str | "BaseDataSource" | None = None,
        validators: list[str | "Validator"] | None = None,
        actions: list[ActionType] | None = None,
        triggers: list["BaseTrigger[Any]"] | None = None,
        execution_strategy: ExecutionStrategy | None = None,
        on_complete: CallbackType | None = None,
        on_error: CallbackType | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize async checkpoint.

        Args:
            name: Checkpoint name.
            config: Checkpoint configuration.
            data_source: Data source to validate.
            validators: List of validators.
            actions: Actions to execute after validation.
            triggers: Triggers for automated execution.
            execution_strategy: Custom action execution strategy.
            on_complete: Callback on successful completion.
            on_error: Callback on error.
            **kwargs: Additional config options.
        """
        # Build config
        if config:
            self._config = config
        else:
            self._config = AsyncCheckpointConfig()

        if name:
            self._config.name = name
        if data_source is not None:
            self._config.data_source = data_source
        if validators:
            self._config.validators = validators

        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)

        self._actions = actions or []
        self._triggers = triggers or []

        # Handle execution_strategy - can be ExecutionStrategy instance or string
        if isinstance(execution_strategy, ExecutionStrategy):
            self._execution_strategy = execution_strategy
        elif isinstance(execution_strategy, str):
            # String strategy goes to config, will be resolved in _get_execution_strategy
            self._config.execution_strategy = execution_strategy
            self._execution_strategy = None
        else:
            self._execution_strategy = execution_strategy
        self._on_complete = on_complete
        self._on_error = on_error

        # Execution context (created per run)
        self._executor: ThreadPoolExecutor | None = None

        # Attach triggers
        for trigger in self._triggers:
            trigger.attach(self)

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def config(self) -> AsyncCheckpointConfig:
        return self._config

    @property
    def actions(self) -> list[ActionType]:
        return self._actions

    @property
    def triggers(self) -> list["BaseTrigger[Any]"]:
        return self._triggers

    def add_action(self, action: ActionType) -> "AsyncCheckpoint":
        """Add an action."""
        self._actions.append(action)
        return self

    def add_trigger(self, trigger: "BaseTrigger[Any]") -> "AsyncCheckpoint":
        """Add a trigger."""
        trigger.attach(self)
        self._triggers.append(trigger)
        return self

    def _generate_run_id(self) -> str:
        """Generate unique run ID."""
        timestamp = datetime.now().strftime(self._config.run_name_template)
        unique_id = uuid4().hex[:8]
        return f"{self._config.name}_{timestamp}_{unique_id}"

    def _resolve_data_source(self) -> tuple[Any, str]:
        """Resolve data source to usable format."""
        from truthound.datasources.base import BaseDataSource

        source = self._config.data_source

        if isinstance(source, BaseDataSource):
            return source, source.name or str(source)

        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.exists():
                return str(path), str(path)
            return source, str(source)

        return source, type(source).__name__

    async def run_async(
        self,
        run_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> CheckpointResult:
        """Run the checkpoint asynchronously.

        This is the main entry point for async checkpoint execution.

        Args:
            run_id: Optional custom run ID.
            context: Additional context for the run.

        Returns:
            CheckpointResult with validation and action results.
        """
        run_id = run_id or self._generate_run_id()
        run_time = datetime.now()
        start_time = time.time()

        # Resolve data source
        try:
            data_source, data_asset = self._resolve_data_source()
        except Exception as e:
            result = CheckpointResult(
                run_id=run_id,
                checkpoint_name=self.name,
                run_time=run_time,
                status=CheckpointStatus.ERROR,
                data_asset="",
                duration_ms=(time.time() - start_time) * 1000,
                error=f"Failed to resolve data source: {e}",
            )
            await self._call_error_callback(result)
            return result

        # Run validation (sync, in executor)
        try:
            validation_result = await self._run_validation_async(
                data_source, data_asset, run_id, start_time
            )

            # Determine status
            if validation_result is None:
                status = CheckpointStatus.ERROR
            elif (
                validation_result.statistics.critical_issues > 0
                and self._config.fail_on_critical
            ):
                status = CheckpointStatus.FAILURE
            elif (
                validation_result.statistics.high_issues > 0
                and self._config.fail_on_high
            ):
                status = CheckpointStatus.FAILURE
            elif validation_result.statistics.total_issues > 0:
                status = CheckpointStatus.WARNING
            else:
                status = CheckpointStatus.SUCCESS

        except Exception as e:
            validation_result = None
            status = CheckpointStatus.ERROR

            result = CheckpointResult(
                run_id=run_id,
                checkpoint_name=self.name,
                run_time=run_time,
                status=status,
                validation_result=validation_result,
                data_asset=data_asset,
                duration_ms=(time.time() - start_time) * 1000,
                error=str(e),
                metadata=context or {},
            )

            # Still run actions (they might want to notify about errors)
            await self._execute_actions_async(result)
            await self._call_error_callback(result)

            return result

        # Create checkpoint result
        checkpoint_result = CheckpointResult(
            run_id=run_id,
            checkpoint_name=self.name,
            run_time=run_time,
            status=status,
            validation_result=validation_result,
            data_asset=data_asset,
            duration_ms=(time.time() - start_time) * 1000,
            metadata=context or {},
        )

        # Execute actions asynchronously
        await self._execute_actions_async(checkpoint_result)

        # Update duration
        checkpoint_result.duration_ms = (time.time() - start_time) * 1000

        # Call completion callback
        if status != CheckpointStatus.ERROR:
            await self._call_complete_callback(checkpoint_result)
        else:
            await self._call_error_callback(checkpoint_result)

        return checkpoint_result

    async def _run_validation_async(
        self,
        data_source: Any,
        data_asset: str,
        run_id: str,
        start_time: float,
    ) -> Any:
        """Run validation in executor to avoid blocking."""
        from truthound.api import check
        from truthound.stores.results import ValidationResult
        from truthound.datasources.base import BaseDataSource

        loop = asyncio.get_running_loop()

        def run_check() -> ValidationResult:
            if isinstance(data_source, BaseDataSource):
                if self._config.sample_size and data_source.needs_sampling():
                    sampled = data_source.sample(n=self._config.sample_size)
                    report = check(
                        source=sampled,
                        validators=self._config.validators,
                        min_severity=self._config.min_severity,
                        schema=self._config.schema,
                        auto_schema=self._config.auto_schema,
                    )
                else:
                    report = check(
                        source=data_source,
                        validators=self._config.validators,
                        min_severity=self._config.min_severity,
                        schema=self._config.schema,
                        auto_schema=self._config.auto_schema,
                    )
            else:
                report = check(
                    data=data_source,
                    validators=self._config.validators,
                    min_severity=self._config.min_severity,
                    schema=self._config.schema,
                    auto_schema=self._config.auto_schema,
                )

            return ValidationResult.from_report(
                report=report,
                data_asset=data_asset,
                run_id=run_id,
                tags=self._config.tags,
                metadata=self._config.metadata,
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        # Run in executor to avoid blocking
        executor = self._get_executor()
        return await loop.run_in_executor(executor, run_check)

    async def _execute_actions_async(
        self, checkpoint_result: CheckpointResult
    ) -> None:
        """Execute all actions asynchronously."""
        if not self._actions:
            return

        # Convert sync actions to async
        executor = self._get_executor()
        async_actions = [
            adapt_to_async(action, executor) for action in self._actions
        ]

        # Create execution context
        context = AsyncExecutionContext(
            executor=executor,
            semaphore=asyncio.Semaphore(self._config.max_concurrent_actions),
            timeout=self._config.action_timeout,
            cancel_on_first_error=self._config.cancel_on_first_error,
        )

        # Get execution strategy
        strategy = self._get_execution_strategy()

        try:
            action_results = await strategy.execute(
                async_actions, checkpoint_result, context
            )
            checkpoint_result.action_results.extend(action_results)

            # Check if any action failure should fail checkpoint
            for result in action_results:
                if result.status == ActionStatus.ERROR:
                    # Find corresponding action config
                    for action in self._actions:
                        if action.name == result.action_name:
                            if hasattr(action, 'config') and action.config.fail_checkpoint_on_error:
                                checkpoint_result.status = CheckpointStatus.ERROR
                                if not checkpoint_result.error:
                                    checkpoint_result.error = f"Action failed: {action.name}"
                            break

        except Exception as e:
            error_result = ActionResult(
                action_name="action_execution",
                action_type="system",
                status=ActionStatus.ERROR,
                message="Action execution failed",
                error=str(e),
            )
            checkpoint_result.action_results.append(error_result)

    def _get_executor(self) -> ThreadPoolExecutor:
        """Get or create thread pool executor."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self._config.executor_workers
            )
        return self._executor

    def _get_execution_strategy(self) -> ExecutionStrategy:
        """Get action execution strategy."""
        if self._execution_strategy:
            return self._execution_strategy

        strategy_name = self._config.execution_strategy

        if strategy_name == "sequential":
            return SequentialStrategy()
        elif strategy_name == "concurrent":
            return ConcurrentStrategy(self._config.max_concurrent_actions)
        else:
            return ConcurrentStrategy(self._config.max_concurrent_actions)

    async def _call_complete_callback(
        self, result: CheckpointResult
    ) -> None:
        """Call completion callback."""
        if self._on_complete:
            if asyncio.iscoroutinefunction(self._on_complete):
                await self._on_complete(result)
            else:
                self._on_complete(result)

    async def _call_error_callback(self, result: CheckpointResult) -> None:
        """Call error callback."""
        if self._on_error:
            if asyncio.iscoroutinefunction(self._on_error):
                await self._on_error(result)
            else:
                self._on_error(result)

    def run(
        self,
        run_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> CheckpointResult:
        """Sync wrapper for run_async.

        Allows AsyncCheckpoint to be used in sync contexts.
        """
        return asyncio.get_event_loop().run_until_complete(
            self.run_async(run_id, context)
        )

    def validate(self) -> list[str]:
        """Validate checkpoint configuration."""
        errors = []

        if not self._config.name:
            errors.append("Checkpoint name is required")

        if not self._config.data_source:
            errors.append("Data source is required")

        for action in self._actions:
            action_errors = action.validate_config()
            for err in action_errors:
                errors.append(f"Action '{action.name}': {err}")

        for trigger in self._triggers:
            trigger_errors = trigger.validate_config()
            for err in trigger_errors:
                errors.append(f"Trigger '{trigger.name}': {err}")

        return errors

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self._config.name,
            "data_source": str(self._config.data_source),
            "validators": self._config.validators,
            "async": True,
            "max_concurrent_actions": self._config.max_concurrent_actions,
            "execution_strategy": self._config.execution_strategy,
            "actions": [a.action_type for a in self._actions],
            "triggers": [t.trigger_type for t in self._triggers],
        }

    async def __aenter__(self) -> "AsyncCheckpoint":
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        if self._executor:
            self._executor.shutdown(wait=False)
            self._executor = None

    def __repr__(self) -> str:
        return (
            f"AsyncCheckpoint(name={self.name!r}, "
            f"actions={len(self._actions)}, "
            f"strategy={self._config.execution_strategy!r})"
        )


# =============================================================================
# Conversion Utilities
# =============================================================================


def to_async_checkpoint(checkpoint: Checkpoint) -> AsyncCheckpoint:
    """Convert a sync Checkpoint to AsyncCheckpoint.

    Args:
        checkpoint: Sync checkpoint to convert.

    Returns:
        Async-compatible checkpoint.
    """
    return AsyncCheckpoint(
        name=checkpoint.name,
        config=AsyncCheckpointConfig(
            name=checkpoint.config.name,
            data_source=checkpoint.config.data_source,
            validators=checkpoint.config.validators,
            min_severity=checkpoint.config.min_severity,
            schema=checkpoint.config.schema,
            auto_schema=checkpoint.config.auto_schema,
            run_name_template=checkpoint.config.run_name_template,
            tags=checkpoint.config.tags,
            metadata=checkpoint.config.metadata,
            fail_on_critical=checkpoint.config.fail_on_critical,
            fail_on_high=checkpoint.config.fail_on_high,
            timeout_seconds=checkpoint.config.timeout_seconds,
            sample_size=checkpoint.config.sample_size,
        ),
        actions=checkpoint.actions,
        triggers=checkpoint.triggers,
    )


async def run_checkpoints_async(
    checkpoints: list[AsyncCheckpoint | Checkpoint],
    max_concurrent: int = 5,
    context: dict[str, Any] | None = None,
) -> list[CheckpointResult]:
    """Run multiple checkpoints concurrently.

    Args:
        checkpoints: List of checkpoints to run.
        max_concurrent: Maximum concurrent checkpoints.
        context: Shared context for all runs.

    Returns:
        List of results in same order as input.
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def run_one(
        checkpoint: AsyncCheckpoint | Checkpoint,
    ) -> CheckpointResult:
        async with semaphore:
            if isinstance(checkpoint, AsyncCheckpoint):
                return await checkpoint.run_async(context=context)
            else:
                # Convert and run
                async_cp = to_async_checkpoint(checkpoint)
                return await async_cp.run_async(context=context)

    tasks = [run_one(cp) for cp in checkpoints]
    return await asyncio.gather(*tasks)
