"""Async Checkpoint Runner.

This module provides AsyncCheckpointRunner for managing async checkpoint
execution with triggers, scheduling, and result streaming.

Key Features:
    - Non-blocking checkpoint execution
    - Concurrent trigger monitoring
    - Async result streaming via AsyncIterator
    - Graceful shutdown with cancellation support
    - Backpressure handling
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
)

from truthound.checkpoint.async_checkpoint import AsyncCheckpoint, to_async_checkpoint
from truthound.checkpoint.checkpoint import Checkpoint, CheckpointResult

if TYPE_CHECKING:
    from truthound.checkpoint.triggers.base import BaseTrigger


# Type aliases
AsyncResultCallback = Callable[[CheckpointResult], Awaitable[None]]
SyncResultCallback = Callable[[CheckpointResult], None]
ResultCallback = AsyncResultCallback | SyncResultCallback

AsyncErrorCallback = Callable[[Exception], Awaitable[None]]
SyncErrorCallback = Callable[[Exception], None]
ErrorCallback = AsyncErrorCallback | SyncErrorCallback


@dataclass
class AsyncRunnerConfig:
    """Configuration for async checkpoint runner.

    Attributes:
        max_concurrent_checkpoints: Max checkpoints running simultaneously.
        trigger_poll_interval: Seconds between trigger checks.
        result_queue_size: Size of result queue (0 = unlimited).
        stop_on_error: Stop runner on first checkpoint error.
        max_consecutive_failures: Max failures before stopping.
        graceful_shutdown_timeout: Timeout for graceful shutdown.
    """

    max_concurrent_checkpoints: int = 10
    trigger_poll_interval: float = 1.0
    result_queue_size: int = 1000
    stop_on_error: bool = False
    max_consecutive_failures: int = 10
    graceful_shutdown_timeout: float = 30.0


class AsyncCheckpointRunner:
    """Async runner for checkpoint execution.

    Manages async checkpoint execution with trigger support,
    concurrent execution, and result streaming.

    Example:
        >>> runner = AsyncCheckpointRunner()
        >>> runner.add_checkpoint(checkpoint1)
        >>> runner.add_checkpoint(checkpoint2)
        >>>
        >>> # Run in background
        >>> await runner.start_async()
        >>>
        >>> # Or iterate results as they complete
        >>> async for result in runner.iter_results_async():
        ...     print(f"Completed: {result.checkpoint_name}")
        >>>
        >>> # Graceful shutdown
        >>> await runner.stop_async()
    """

    def __init__(
        self,
        config: AsyncRunnerConfig | None = None,
        result_callback: ResultCallback | None = None,
        error_callback: ErrorCallback | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize async runner.

        Args:
            config: Runner configuration.
            result_callback: Called on each checkpoint completion.
            error_callback: Called on errors.
            **kwargs: Additional config options.
        """
        self._config = config or AsyncRunnerConfig()

        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)

        self._result_callback = result_callback
        self._error_callback = error_callback

        self._checkpoints: dict[str, AsyncCheckpoint] = {}
        self._running = False
        self._result_queue: asyncio.Queue[CheckpointResult] = asyncio.Queue(
            maxsize=self._config.result_queue_size or 0
        )
        self._semaphore = asyncio.Semaphore(
            self._config.max_concurrent_checkpoints
        )
        self._consecutive_failures = 0
        self._tasks: set[asyncio.Task[Any]] = set()
        self._shutdown_event = asyncio.Event()

    @property
    def running(self) -> bool:
        """Check if runner is active."""
        return self._running

    @property
    def checkpoints(self) -> dict[str, AsyncCheckpoint]:
        """Get registered checkpoints."""
        return self._checkpoints.copy()

    def add_checkpoint(
        self, checkpoint: AsyncCheckpoint | Checkpoint
    ) -> "AsyncCheckpointRunner":
        """Register a checkpoint.

        Args:
            checkpoint: Checkpoint to register.

        Returns:
            Self for chaining.
        """
        if isinstance(checkpoint, Checkpoint):
            checkpoint = to_async_checkpoint(checkpoint)

        self._checkpoints[checkpoint.name] = checkpoint
        return self

    def remove_checkpoint(self, name: str) -> bool:
        """Remove a checkpoint.

        Args:
            name: Checkpoint name.

        Returns:
            True if removed.
        """
        if name in self._checkpoints:
            del self._checkpoints[name]
            return True
        return False

    async def run_once_async(
        self,
        checkpoint: AsyncCheckpoint | Checkpoint | str,
        context: dict[str, Any] | None = None,
    ) -> CheckpointResult:
        """Run a checkpoint once.

        Args:
            checkpoint: Checkpoint or name to run.
            context: Additional context.

        Returns:
            Checkpoint result.
        """
        if isinstance(checkpoint, str):
            if checkpoint not in self._checkpoints:
                raise ValueError(f"Checkpoint not found: {checkpoint}")
            checkpoint = self._checkpoints[checkpoint]
        elif isinstance(checkpoint, Checkpoint):
            checkpoint = to_async_checkpoint(checkpoint)

        async with self._semaphore:
            result = await checkpoint.run_async(context=context)

        await self._handle_result(result)
        return result

    async def run_all_async(
        self,
        context: dict[str, Any] | None = None,
    ) -> list[CheckpointResult]:
        """Run all registered checkpoints.

        Args:
            context: Shared context for all runs.

        Returns:
            List of results.
        """
        async def run_one(cp: AsyncCheckpoint) -> CheckpointResult:
            async with self._semaphore:
                return await cp.run_async(context=context)

        tasks = [run_one(cp) for cp in self._checkpoints.values()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions and convert to results
        final_results: list[CheckpointResult] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Create error result
                cp_name = list(self._checkpoints.keys())[i]
                error_result = CheckpointResult(
                    run_id=f"{cp_name}_error",
                    checkpoint_name=cp_name,
                    run_time=datetime.now(),
                    status="error",
                    error=str(result),
                )
                final_results.append(error_result)
                await self._handle_error(result)
            else:
                final_results.append(result)
                await self._handle_result(result)

        return final_results

    async def start_async(self) -> None:
        """Start the runner.

        Begins monitoring triggers and executing checkpoints.
        """
        if self._running:
            return

        self._running = True
        self._shutdown_event.clear()

        # Start triggers
        for checkpoint in self._checkpoints.values():
            for trigger in checkpoint.triggers:
                trigger.start()

        # Create monitoring task
        monitor_task = asyncio.create_task(self._trigger_loop())
        self._tasks.add(monitor_task)
        monitor_task.add_done_callback(self._tasks.discard)

    async def stop_async(self, wait: bool = True) -> None:
        """Stop the runner gracefully.

        Args:
            wait: Wait for pending tasks to complete.
        """
        self._running = False
        self._shutdown_event.set()

        # Stop triggers
        for checkpoint in self._checkpoints.values():
            for trigger in checkpoint.triggers:
                trigger.stop()

        if wait and self._tasks:
            # Wait for tasks with timeout
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._tasks, return_exceptions=True),
                    timeout=self._config.graceful_shutdown_timeout,
                )
            except asyncio.TimeoutError:
                # Cancel remaining tasks
                for task in self._tasks:
                    task.cancel()

    async def _trigger_loop(self) -> None:
        """Main trigger monitoring loop."""
        while self._running:
            try:
                await self._check_triggers()
                await asyncio.sleep(self._config.trigger_poll_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._consecutive_failures += 1
                await self._handle_error(e)

                if self._config.stop_on_error:
                    self._running = False
                    break

                if (
                    self._consecutive_failures
                    >= self._config.max_consecutive_failures
                ):
                    self._running = False
                    break

    async def _check_triggers(self) -> None:
        """Check all triggers and schedule checkpoint runs."""
        for checkpoint in list(self._checkpoints.values()):
            for trigger in checkpoint.triggers:
                if not trigger.enabled or not trigger.should_continue():
                    continue

                # Check trigger (may be async in future)
                result = trigger.should_trigger()

                if result.should_run:
                    # Schedule checkpoint run
                    task = asyncio.create_task(
                        self._run_triggered_checkpoint(
                            checkpoint,
                            trigger,
                            result.context,
                        )
                    )
                    self._tasks.add(task)
                    task.add_done_callback(self._tasks.discard)

    async def _run_triggered_checkpoint(
        self,
        checkpoint: AsyncCheckpoint,
        trigger: "BaseTrigger[Any]",
        trigger_context: dict[str, Any],
    ) -> None:
        """Run a checkpoint triggered by a trigger."""
        try:
            async with self._semaphore:
                result = await checkpoint.run_async(
                    context={"trigger": trigger.name, **trigger_context}
                )

            trigger.record_run()
            self._consecutive_failures = 0

            await self._handle_result(result)

        except Exception as e:
            self._consecutive_failures += 1
            await self._handle_error(e)

    async def _handle_result(self, result: CheckpointResult) -> None:
        """Handle checkpoint result."""
        # Add to queue (with backpressure)
        try:
            self._result_queue.put_nowait(result)
        except asyncio.QueueFull:
            # Drop oldest result if queue is full
            try:
                self._result_queue.get_nowait()
                self._result_queue.put_nowait(result)
            except asyncio.QueueEmpty:
                pass

        # Call callback
        if self._result_callback:
            try:
                if asyncio.iscoroutinefunction(self._result_callback):
                    await self._result_callback(result)
                else:
                    self._result_callback(result)
            except Exception:
                pass  # Don't let callback errors affect runner

    async def _handle_error(self, error: Exception) -> None:
        """Handle errors."""
        if self._error_callback:
            try:
                if asyncio.iscoroutinefunction(self._error_callback):
                    await self._error_callback(error)
                else:
                    self._error_callback(error)
            except Exception:
                pass

    async def get_results_async(
        self,
        timeout: float | None = None,
        max_results: int = 100,
    ) -> list[CheckpointResult]:
        """Get completed results from queue.

        Args:
            timeout: Max time to wait.
            max_results: Max results to return.

        Returns:
            List of results.
        """
        results = []

        try:
            for _ in range(max_results):
                result = await asyncio.wait_for(
                    self._result_queue.get(),
                    timeout=timeout or 0.1,
                )
                results.append(result)
        except asyncio.TimeoutError:
            pass

        return results

    async def iter_results_async(
        self,
        timeout: float = 1.0,
    ) -> AsyncIterator[CheckpointResult]:
        """Iterate over results as they complete.

        Args:
            timeout: Time to wait between checks.

        Yields:
            CheckpointResults as they complete.
        """
        while self._running or not self._result_queue.empty():
            try:
                result = await asyncio.wait_for(
                    self._result_queue.get(),
                    timeout=timeout,
                )
                yield result
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    async def wait_for_completion_async(
        self,
        timeout: float | None = None,
    ) -> bool:
        """Wait for runner to stop.

        Args:
            timeout: Max time to wait.

        Returns:
            True if stopped cleanly.
        """
        if not self._tasks:
            return True

        try:
            await asyncio.wait_for(
                self._shutdown_event.wait(),
                timeout=timeout,
            )
            return True
        except asyncio.TimeoutError:
            return False

    def get_stats(self) -> dict[str, Any]:
        """Get runner statistics."""
        return {
            "running": self._running,
            "checkpoints": len(self._checkpoints),
            "pending_tasks": len(self._tasks),
            "queued_results": self._result_queue.qsize(),
            "consecutive_failures": self._consecutive_failures,
        }


# =============================================================================
# Convenience Functions
# =============================================================================


async def run_checkpoint_async(
    checkpoint: AsyncCheckpoint | Checkpoint | str,
    context: dict[str, Any] | None = None,
) -> CheckpointResult:
    """Convenience function to run a checkpoint asynchronously.

    Args:
        checkpoint: Checkpoint or name.
        context: Additional context.

    Returns:
        Checkpoint result.
    """
    from truthound.checkpoint.registry import get_checkpoint

    if isinstance(checkpoint, str):
        checkpoint = get_checkpoint(checkpoint)

    if isinstance(checkpoint, Checkpoint):
        checkpoint = to_async_checkpoint(checkpoint)

    return await checkpoint.run_async(context=context)


async def run_checkpoints_parallel(
    checkpoints: list[AsyncCheckpoint | Checkpoint],
    max_concurrent: int = 5,
    context: dict[str, Any] | None = None,
    on_complete: ResultCallback | None = None,
) -> list[CheckpointResult]:
    """Run multiple checkpoints in parallel.

    Args:
        checkpoints: Checkpoints to run.
        max_concurrent: Max concurrent runs.
        context: Shared context.
        on_complete: Callback for each completion.

    Returns:
        Results in input order.
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def run_one(
        cp: AsyncCheckpoint | Checkpoint,
    ) -> CheckpointResult:
        async with semaphore:
            if isinstance(cp, Checkpoint):
                cp = to_async_checkpoint(cp)
            result = await cp.run_async(context=context)

            if on_complete:
                if asyncio.iscoroutinefunction(on_complete):
                    await on_complete(result)
                else:
                    on_complete(result)

            return result

    tasks = [run_one(cp) for cp in checkpoints]
    return await asyncio.gather(*tasks)


class CheckpointPool:
    """Pool for efficient checkpoint execution.

    Maintains a pool of workers for checkpoint execution,
    suitable for high-throughput scenarios.

    Example:
        >>> async with CheckpointPool(workers=10) as pool:
        ...     results = await pool.submit_many(checkpoints)
    """

    def __init__(
        self,
        workers: int = 5,
        result_callback: ResultCallback | None = None,
    ) -> None:
        self._workers = workers
        self._result_callback = result_callback
        self._queue: asyncio.Queue[
            tuple[AsyncCheckpoint, asyncio.Future[CheckpointResult]]
        ] = asyncio.Queue()
        self._worker_tasks: list[asyncio.Task[None]] = []
        self._running = False

    async def __aenter__(self) -> "CheckpointPool":
        await self.start()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.stop()

    async def start(self) -> None:
        """Start worker pool."""
        self._running = True
        for _ in range(self._workers):
            task = asyncio.create_task(self._worker())
            self._worker_tasks.append(task)

    async def stop(self) -> None:
        """Stop worker pool."""
        self._running = False

        # Send stop signals
        for _ in range(self._workers):
            await self._queue.put((None, None))  # type: ignore

        # Wait for workers
        await asyncio.gather(*self._worker_tasks, return_exceptions=True)

    async def _worker(self) -> None:
        """Worker coroutine."""
        while self._running:
            item = await self._queue.get()
            checkpoint, future = item

            if checkpoint is None:
                break

            try:
                result = await checkpoint.run_async()
                future.set_result(result)

                if self._result_callback:
                    if asyncio.iscoroutinefunction(self._result_callback):
                        await self._result_callback(result)
                    else:
                        self._result_callback(result)

            except Exception as e:
                future.set_exception(e)

    async def submit(
        self,
        checkpoint: AsyncCheckpoint | Checkpoint,
    ) -> CheckpointResult:
        """Submit a checkpoint for execution.

        Args:
            checkpoint: Checkpoint to run.

        Returns:
            Checkpoint result.
        """
        if isinstance(checkpoint, Checkpoint):
            checkpoint = to_async_checkpoint(checkpoint)

        loop = asyncio.get_running_loop()
        future: asyncio.Future[CheckpointResult] = loop.create_future()

        await self._queue.put((checkpoint, future))
        return await future

    async def submit_many(
        self,
        checkpoints: list[AsyncCheckpoint | Checkpoint],
    ) -> list[CheckpointResult]:
        """Submit multiple checkpoints.

        Args:
            checkpoints: Checkpoints to run.

        Returns:
            Results in input order.
        """
        tasks = [self.submit(cp) for cp in checkpoints]
        return await asyncio.gather(*tasks)
