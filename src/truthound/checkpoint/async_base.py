"""Async base classes for checkpoint actions and triggers.

This module provides async-compatible base classes that enable non-blocking
execution of checkpoint pipelines, supporting high-throughput enterprise
workloads.

Design Principles:
    1. Backward Compatibility: Sync actions work in async context via run_in_executor
    2. Gradual Migration: Mixed sync/async actions in same checkpoint
    3. Protocol-Based: Uses Protocol for duck typing flexibility
    4. Composable: Async middleware/decorator support
"""

from __future__ import annotations

import asyncio
import functools
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Generic,
    Protocol,
    TypeVar,
    Union,
    runtime_checkable,
)

from truthound.checkpoint.actions.base import (
    ActionConfig,
    ActionResult,
    ActionStatus,
    BaseAction,
    NotifyCondition,
)
from truthound.checkpoint.triggers.base import (
    BaseTrigger,
    TriggerConfig,
    TriggerResult,
)

if TYPE_CHECKING:
    from truthound.checkpoint.checkpoint import CheckpointResult


# Type variables
ConfigT = TypeVar("ConfigT", bound=ActionConfig)
TriggerConfigT = TypeVar("TriggerConfigT", bound=TriggerConfig)
T = TypeVar("T")


# =============================================================================
# Protocols for Duck Typing
# =============================================================================


@runtime_checkable
class AsyncExecutable(Protocol):
    """Protocol for async-executable actions."""

    async def execute_async(
        self, checkpoint_result: "CheckpointResult"
    ) -> ActionResult:
        """Execute asynchronously."""
        ...


@runtime_checkable
class SyncExecutable(Protocol):
    """Protocol for sync-executable actions."""

    def execute(self, checkpoint_result: "CheckpointResult") -> ActionResult:
        """Execute synchronously."""
        ...


# =============================================================================
# Async Action Base Class
# =============================================================================


class AsyncBaseAction(ABC, Generic[ConfigT]):
    """Async-native base class for checkpoint actions.

    This class provides the same interface as BaseAction but with async/await
    support for non-blocking I/O operations like HTTP requests, database queries,
    and file operations.

    Key Features:
        - Native async/await support
        - Automatic timeout handling via asyncio.wait_for
        - Retry with exponential backoff
        - Graceful cancellation support
        - Mixed execution: can be used alongside sync actions

    Example:
        >>> class AsyncSlackNotification(AsyncBaseAction[SlackConfig]):
        ...     action_type = "async_slack"
        ...
        ...     async def _execute_async(self, result: CheckpointResult) -> ActionResult:
        ...         async with aiohttp.ClientSession() as session:
        ...             await session.post(self._config.webhook_url, json=payload)
        ...         return ActionResult(status=ActionStatus.SUCCESS, ...)

    Attributes:
        action_type: String identifier for this action type.
    """

    action_type: str = "async_base"

    def __init__(self, config: ConfigT | None = None, **kwargs: Any) -> None:
        """Initialize the async action.

        Args:
            config: Action configuration. If None, uses default.
            **kwargs: Additional config options.
        """
        self._config = config or self._default_config()

        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)

    @classmethod
    @abstractmethod
    def _default_config(cls) -> ConfigT:
        """Create default configuration."""
        pass

    @property
    def config(self) -> ConfigT:
        """Get action configuration."""
        return self._config

    @property
    def name(self) -> str:
        """Get action name."""
        return self._config.name or f"{self.action_type}_{id(self)}"

    @property
    def enabled(self) -> bool:
        """Check if action is enabled."""
        return self._config.enabled

    def should_run(self, result_status: str) -> bool:
        """Check if action should run based on status."""
        if not self.enabled:
            return False

        notify_condition = self._config.notify_on
        if isinstance(notify_condition, str):
            notify_condition = NotifyCondition(notify_condition.lower())

        return notify_condition.should_notify(result_status)

    async def execute_async(
        self, checkpoint_result: "CheckpointResult"
    ) -> ActionResult:
        """Execute the action asynchronously with timeout and retry.

        This method wraps _execute_async with:
        - Condition checking (notify_on)
        - Timeout handling
        - Retry logic with exponential backoff
        - Timing measurement
        - Error handling

        Args:
            checkpoint_result: The checkpoint result to process.

        Returns:
            ActionResult with execution outcome.
        """
        started_at = datetime.now()
        result_status = (
            checkpoint_result.status.value
            if hasattr(checkpoint_result.status, "value")
            else str(checkpoint_result.status)
        )

        # Check if should run
        if not self.should_run(result_status):
            return ActionResult(
                action_name=self.name,
                action_type=self.action_type,
                status=ActionStatus.SKIPPED,
                message=f"Skipped: notify_on={self._config.notify_on}, status={result_status}",
                started_at=started_at,
                completed_at=datetime.now(),
            )

        # Execute with retries
        last_error: Exception | None = None
        retry_delay = self._config.retry_delay_seconds

        for attempt in range(self._config.retry_count + 1):
            try:
                # Apply timeout
                result = await asyncio.wait_for(
                    self._execute_async(checkpoint_result),
                    timeout=self._config.timeout_seconds,
                )
                result.started_at = started_at
                result.completed_at = datetime.now()
                result.duration_ms = (
                    result.completed_at - started_at
                ).total_seconds() * 1000
                return result

            except asyncio.TimeoutError:
                last_error = asyncio.TimeoutError(
                    f"Action timed out after {self._config.timeout_seconds}s"
                )
                if attempt < self._config.retry_count:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff

            except asyncio.CancelledError:
                # Propagate cancellation
                raise

            except Exception as e:
                last_error = e
                if attempt < self._config.retry_count:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2

        # All retries failed
        completed_at = datetime.now()
        return ActionResult(
            action_name=self.name,
            action_type=self.action_type,
            status=ActionStatus.ERROR,
            message=f"Failed after {self._config.retry_count + 1} attempts",
            started_at=started_at,
            completed_at=completed_at,
            duration_ms=(completed_at - started_at).total_seconds() * 1000,
            error=str(last_error),
        )

    @abstractmethod
    async def _execute_async(
        self, checkpoint_result: "CheckpointResult"
    ) -> ActionResult:
        """Execute the action implementation asynchronously.

        Subclasses must implement this method.

        Args:
            checkpoint_result: The checkpoint result to process.

        Returns:
            ActionResult with execution outcome.
        """
        pass

    def validate_config(self) -> list[str]:
        """Validate configuration."""
        return []

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


# =============================================================================
# Async Trigger Base Class
# =============================================================================


class AsyncBaseTrigger(ABC, Generic[TriggerConfigT]):
    """Async-native base class for triggers.

    Supports async trigger checking for event-based and polling triggers
    that need non-blocking I/O.

    Example:
        >>> class AsyncKafkaTrigger(AsyncBaseTrigger[KafkaConfig]):
        ...     async def should_trigger_async(self) -> TriggerResult:
        ...         message = await self._consumer.poll()
        ...         return TriggerResult(should_run=message is not None)
    """

    trigger_type: str = "async_base"

    def __init__(
        self, config: TriggerConfigT | None = None, **kwargs: Any
    ) -> None:
        self._config = config or self._default_config()

        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)

        self._status = "stopped"
        self._run_count = 0
        self._last_run: datetime | None = None
        self._checkpoint: Any = None

    @classmethod
    @abstractmethod
    def _default_config(cls) -> TriggerConfigT:
        """Create default configuration."""
        pass

    @property
    def config(self) -> TriggerConfigT:
        return self._config

    @property
    def name(self) -> str:
        return self._config.name or f"{self.trigger_type}_{id(self)}"

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    def attach(self, checkpoint: Any) -> None:
        self._checkpoint = checkpoint

    @abstractmethod
    async def should_trigger_async(self) -> TriggerResult:
        """Check trigger condition asynchronously."""
        pass

    def should_trigger(self) -> TriggerResult:
        """Sync wrapper for compatibility."""
        return asyncio.get_event_loop().run_until_complete(
            self.should_trigger_async()
        )

    async def start_async(self) -> None:
        """Start the trigger asynchronously."""
        if not self.enabled:
            return
        self._status = "active"
        await self._on_start_async()

    async def stop_async(self) -> None:
        """Stop the trigger asynchronously."""
        self._status = "stopped"
        await self._on_stop_async()

    async def _on_start_async(self) -> None:
        """Called when trigger starts. Override for custom behavior."""
        pass

    async def _on_stop_async(self) -> None:
        """Called when trigger stops. Override for custom behavior."""
        pass

    def record_run(self) -> None:
        self._run_count += 1
        self._last_run = datetime.now()

    def validate_config(self) -> list[str]:
        return []


# =============================================================================
# Adapter: Sync to Async
# =============================================================================


class SyncActionAdapter(AsyncBaseAction[ConfigT]):
    """Adapter to run sync actions in async context.

    Wraps synchronous BaseAction instances to work seamlessly in async
    checkpoint pipelines by running them in a thread pool executor.

    Example:
        >>> sync_action = SlackNotification(webhook_url="...")
        >>> async_action = SyncActionAdapter(sync_action)
        >>> await async_action.execute_async(result)
    """

    action_type = "sync_adapter"

    def __init__(
        self,
        wrapped_action: BaseAction[ConfigT],
        executor: ThreadPoolExecutor | None = None,
    ) -> None:
        """Initialize the adapter.

        Args:
            wrapped_action: The sync action to wrap.
            executor: Optional thread pool executor. If None, uses default.
        """
        self._wrapped = wrapped_action
        self._executor = executor
        # Use wrapped action's config
        self._config = wrapped_action.config

    @classmethod
    def _default_config(cls) -> ConfigT:
        # Not used, config comes from wrapped action
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self._wrapped.name

    @property
    def action_type(self) -> str:
        return f"async_{self._wrapped.action_type}"

    def should_run(self, result_status: str) -> bool:
        return self._wrapped.should_run(result_status)

    async def _execute_async(
        self, checkpoint_result: "CheckpointResult"
    ) -> ActionResult:
        """Run sync action in thread pool."""
        loop = asyncio.get_running_loop()

        # Run sync execute in executor
        result = await loop.run_in_executor(
            self._executor,
            self._wrapped.execute,
            checkpoint_result,
        )

        return result

    def validate_config(self) -> list[str]:
        return self._wrapped.validate_config()


def adapt_to_async(
    action: BaseAction[ConfigT] | AsyncBaseAction[ConfigT],
    executor: ThreadPoolExecutor | None = None,
) -> AsyncBaseAction[ConfigT]:
    """Adapt any action to async interface.

    Args:
        action: Sync or async action.
        executor: Optional thread pool for sync actions.

    Returns:
        Async-compatible action.

    Raises:
        TypeError: If action is not a valid action type.
    """
    # Check if already async
    if isinstance(action, AsyncBaseAction):
        return action

    # Check if it's a sync action
    if isinstance(action, BaseAction):
        return SyncActionAdapter(action, executor)

    # Check if it has the required interface (duck typing)
    if hasattr(action, "execute") and callable(getattr(action, "execute")):
        return SyncActionAdapter(action, executor)  # type: ignore

    raise TypeError(
        f"Cannot adapt {type(action).__name__} to async. "
        f"Expected BaseAction or AsyncBaseAction, got {type(action)}"
    )


# =============================================================================
# Async Execution Context
# =============================================================================


@dataclass
class AsyncExecutionContext:
    """Context for async checkpoint execution.

    Provides shared resources and configuration for async execution.

    Attributes:
        executor: Thread pool for sync action execution.
        semaphore: Concurrency limiter.
        timeout: Default timeout for actions.
        cancel_on_first_error: Stop remaining actions on first error.
    """

    executor: ThreadPoolExecutor | None = None
    semaphore: asyncio.Semaphore | None = None
    timeout: float = 30.0
    cancel_on_first_error: bool = False
    _owned_executor: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        if self.executor is None:
            self.executor = ThreadPoolExecutor(max_workers=4)
            self._owned_executor = True

    async def __aenter__(self) -> "AsyncExecutionContext":
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._owned_executor and self.executor:
            self.executor.shutdown(wait=False)


# =============================================================================
# Action Execution Strategies
# =============================================================================


class ExecutionStrategy(ABC):
    """Base class for action execution strategies."""

    @abstractmethod
    async def execute(
        self,
        actions: list[AsyncBaseAction[Any]],
        checkpoint_result: "CheckpointResult",
        context: AsyncExecutionContext | None = None,
    ) -> list[ActionResult]:
        """Execute actions according to strategy."""
        pass


class SequentialStrategy(ExecutionStrategy):
    """Execute actions one after another.

    Use when actions have dependencies or order matters.
    """

    async def execute(
        self,
        actions: list[AsyncBaseAction[Any]],
        checkpoint_result: "CheckpointResult",
        context: AsyncExecutionContext | None = None,
    ) -> list[ActionResult]:
        results = []
        for action in actions:
            result = await action.execute_async(checkpoint_result)
            results.append(result)

            if context and context.cancel_on_first_error:
                if result.status == ActionStatus.ERROR:
                    break

        return results


class ConcurrentStrategy(ExecutionStrategy):
    """Execute all actions concurrently.

    Use when actions are independent and can run in parallel.
    """

    def __init__(self, max_concurrency: int | None = None) -> None:
        self.max_concurrency = max_concurrency

    async def execute(
        self,
        actions: list[AsyncBaseAction[Any]],
        checkpoint_result: "CheckpointResult",
        context: AsyncExecutionContext | None = None,
    ) -> list[ActionResult]:
        semaphore = None
        if self.max_concurrency:
            semaphore = asyncio.Semaphore(self.max_concurrency)
        elif context and context.semaphore:
            semaphore = context.semaphore

        async def run_with_semaphore(
            action: AsyncBaseAction[Any],
        ) -> ActionResult:
            if semaphore:
                async with semaphore:
                    return await action.execute_async(checkpoint_result)
            return await action.execute_async(checkpoint_result)

        tasks = [run_with_semaphore(action) for action in actions]
        return await asyncio.gather(*tasks, return_exceptions=False)


class PipelineStrategy(ExecutionStrategy):
    """Execute actions in pipeline stages.

    Actions are grouped into stages. Within a stage, actions run concurrently.
    Stages execute sequentially.
    """

    def __init__(self, stages: list[list[int]]) -> None:
        """Initialize pipeline.

        Args:
            stages: List of action index groups.
                    e.g., [[0, 1], [2], [3, 4]] = 3 stages
        """
        self.stages = stages

    async def execute(
        self,
        actions: list[AsyncBaseAction[Any]],
        checkpoint_result: "CheckpointResult",
        context: AsyncExecutionContext | None = None,
    ) -> list[ActionResult]:
        all_results: list[ActionResult | None] = [None] * len(actions)

        for stage_indices in self.stages:
            stage_actions = [
                (i, actions[i]) for i in stage_indices if i < len(actions)
            ]

            tasks = [
                action.execute_async(checkpoint_result)
                for _, action in stage_actions
            ]
            stage_results = await asyncio.gather(*tasks)

            for (idx, _), result in zip(stage_actions, stage_results):
                all_results[idx] = result

        return [r for r in all_results if r is not None]


# =============================================================================
# Async Action Decorators
# =============================================================================


def with_retry(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[
    [Callable[..., Awaitable[T]]],
    Callable[..., Awaitable[T]],
]:
    """Decorator for async retry logic.

    Args:
        max_retries: Maximum retry attempts.
        delay: Initial delay between retries.
        backoff: Multiplier for delay after each retry.
        exceptions: Exception types to retry on.
    """

    def decorator(
        func: Callable[..., Awaitable[T]],
    ) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            current_delay = delay
            last_exception: Exception | None = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff

            raise last_exception or Exception("Retry failed")

        return wrapper

    return decorator


def with_timeout(
    seconds: float,
) -> Callable[
    [Callable[..., Awaitable[T]]],
    Callable[..., Awaitable[T]],
]:
    """Decorator for async timeout."""

    def decorator(
        func: Callable[..., Awaitable[T]],
    ) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)

        return wrapper

    return decorator


def with_semaphore(
    semaphore: asyncio.Semaphore,
) -> Callable[
    [Callable[..., Awaitable[T]]],
    Callable[..., Awaitable[T]],
]:
    """Decorator to limit concurrency."""

    def decorator(
        func: Callable[..., Awaitable[T]],
    ) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            async with semaphore:
                return await func(*args, **kwargs)

        return wrapper

    return decorator
