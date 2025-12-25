"""Checkpoint runner for executing validation pipelines.

This module provides the CheckpointRunner for managing checkpoint execution,
including scheduled runs, concurrent execution, and result handling.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Iterator
from queue import Queue, Empty

if TYPE_CHECKING:
    from truthound.checkpoint.checkpoint import Checkpoint, CheckpointResult
    from truthound.checkpoint.triggers.base import BaseTrigger


@dataclass
class RunnerConfig:
    """Configuration for the checkpoint runner.

    Attributes:
        max_workers: Maximum concurrent checkpoint runs.
        poll_interval_seconds: Interval for checking triggers.
        result_callback: Callback for completed runs.
        error_callback: Callback for failed runs.
        stop_on_error: Stop runner on first error.
        max_failures: Maximum consecutive failures before stopping.
        result_queue_size: Size of the result queue.
    """

    max_workers: int = 4
    poll_interval_seconds: float = 1.0
    result_callback: Callable[["CheckpointResult"], None] | None = None
    error_callback: Callable[[Exception], None] | None = None
    stop_on_error: bool = False
    max_failures: int = 10
    result_queue_size: int = 1000


class CheckpointRunner:
    """Manages execution of checkpoints with triggers.

    The runner monitors triggers and executes checkpoints when
    trigger conditions are met. It supports concurrent execution,
    result callbacks, and graceful shutdown.

    Example:
        >>> from truthound.checkpoint import Checkpoint, CheckpointRunner
        >>> from truthound.checkpoint.triggers import ScheduleTrigger
        >>>
        >>> checkpoint = Checkpoint(
        ...     name="hourly_check",
        ...     data_source="data.csv",
        ... ).add_trigger(ScheduleTrigger(interval_hours=1))
        >>>
        >>> runner = CheckpointRunner()
        >>> runner.add_checkpoint(checkpoint)
        >>>
        >>> # Run in background
        >>> runner.start()
        >>>
        >>> # Or run once
        >>> result = runner.run_once(checkpoint)
    """

    def __init__(self, config: RunnerConfig | None = None, **kwargs: Any) -> None:
        """Initialize the runner.

        Args:
            config: Runner configuration.
            **kwargs: Additional config options.
        """
        self._config = config or RunnerConfig()

        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)

        self._checkpoints: dict[str, "Checkpoint"] = {}
        self._running = False
        self._thread: threading.Thread | None = None
        self._result_queue: Queue["CheckpointResult"] = Queue(
            maxsize=self._config.result_queue_size
        )
        self._lock = threading.RLock()
        self._consecutive_failures = 0

    @property
    def running(self) -> bool:
        """Check if runner is active."""
        return self._running

    @property
    def checkpoints(self) -> dict[str, "Checkpoint"]:
        """Get registered checkpoints."""
        return self._checkpoints.copy()

    def add_checkpoint(self, checkpoint: "Checkpoint") -> "CheckpointRunner":
        """Register a checkpoint with the runner.

        Args:
            checkpoint: Checkpoint to register.

        Returns:
            Self for chaining.
        """
        with self._lock:
            self._checkpoints[checkpoint.name] = checkpoint
        return self

    def remove_checkpoint(self, name: str) -> bool:
        """Remove a checkpoint from the runner.

        Args:
            name: Name of checkpoint to remove.

        Returns:
            True if removed, False if not found.
        """
        with self._lock:
            if name in self._checkpoints:
                del self._checkpoints[name]
                return True
        return False

    def run_once(
        self,
        checkpoint: "Checkpoint | str",
        context: dict[str, Any] | None = None,
    ) -> "CheckpointResult":
        """Run a checkpoint once synchronously.

        Args:
            checkpoint: Checkpoint or checkpoint name to run.
            context: Additional context for the run.

        Returns:
            CheckpointResult from the run.
        """
        if isinstance(checkpoint, str):
            with self._lock:
                if checkpoint not in self._checkpoints:
                    raise ValueError(f"Checkpoint not found: {checkpoint}")
                checkpoint = self._checkpoints[checkpoint]

        result = checkpoint.run(context=context)

        # Call callback
        if self._config.result_callback:
            self._config.result_callback(result)

        return result

    def run_all(
        self,
        context: dict[str, Any] | None = None,
    ) -> list["CheckpointResult"]:
        """Run all registered checkpoints once.

        Args:
            context: Additional context for all runs.

        Returns:
            List of CheckpointResults.
        """
        results = []
        with self._lock:
            checkpoints = list(self._checkpoints.values())

        for checkpoint in checkpoints:
            result = checkpoint.run(context=context)
            results.append(result)

            if self._config.result_callback:
                self._config.result_callback(result)

        return results

    def start(self, blocking: bool = False) -> None:
        """Start the runner.

        Args:
            blocking: If True, block until stopped.
        """
        if self._running:
            return

        self._running = True

        # Start triggers
        with self._lock:
            for checkpoint in self._checkpoints.values():
                for trigger in checkpoint.triggers:
                    trigger.start()

        if blocking:
            self._run_loop()
        else:
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()

    def stop(self, wait: bool = True, timeout: float = 30.0) -> None:
        """Stop the runner.

        Args:
            wait: Wait for runner to stop.
            timeout: Maximum time to wait.
        """
        self._running = False

        # Stop triggers
        with self._lock:
            for checkpoint in self._checkpoints.values():
                for trigger in checkpoint.triggers:
                    trigger.stop()

        if wait and self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def _run_loop(self) -> None:
        """Main runner loop."""
        while self._running:
            try:
                self._check_triggers()
                time.sleep(self._config.poll_interval_seconds)
            except Exception as e:
                self._consecutive_failures += 1

                if self._config.error_callback:
                    self._config.error_callback(e)

                if self._config.stop_on_error:
                    self._running = False
                    break

                if self._consecutive_failures >= self._config.max_failures:
                    self._running = False
                    break

    def _check_triggers(self) -> None:
        """Check all triggers and run checkpoints as needed."""
        with self._lock:
            checkpoints = list(self._checkpoints.values())

        for checkpoint in checkpoints:
            for trigger in checkpoint.triggers:
                if not trigger.enabled or not trigger.should_continue():
                    continue

                result = trigger.should_trigger()
                if result.should_run:
                    # Run checkpoint
                    try:
                        cp_result = checkpoint.run(
                            context={"trigger": trigger.name, **result.context}
                        )
                        trigger.record_run()
                        self._consecutive_failures = 0

                        # Queue result
                        try:
                            self._result_queue.put_nowait(cp_result)
                        except Exception:
                            pass  # Queue full

                        # Callback
                        if self._config.result_callback:
                            self._config.result_callback(cp_result)

                    except Exception as e:
                        self._consecutive_failures += 1
                        if self._config.error_callback:
                            self._config.error_callback(e)

    def get_results(
        self,
        timeout: float | None = None,
        max_results: int = 100,
    ) -> list["CheckpointResult"]:
        """Get completed results from the queue.

        Args:
            timeout: Time to wait for results.
            max_results: Maximum results to return.

        Returns:
            List of CheckpointResults.
        """
        results = []

        try:
            for _ in range(max_results):
                result = self._result_queue.get(timeout=timeout or 0.1)
                results.append(result)
        except Empty:
            pass

        return results

    def iter_results(
        self,
        timeout: float = 1.0,
    ) -> Iterator["CheckpointResult"]:
        """Iterate over results as they complete.

        Args:
            timeout: Time to wait between checks.

        Yields:
            CheckpointResults as they complete.
        """
        while self._running or not self._result_queue.empty():
            try:
                result = self._result_queue.get(timeout=timeout)
                yield result
            except Empty:
                continue

    def wait_for_completion(self, timeout: float | None = None) -> bool:
        """Wait for the runner to stop.

        Args:
            timeout: Maximum time to wait.

        Returns:
            True if stopped cleanly, False if timeout.
        """
        if self._thread:
            self._thread.join(timeout=timeout)
            return not self._thread.is_alive()
        return True


def run_checkpoint(
    checkpoint: "Checkpoint | str",
    context: dict[str, Any] | None = None,
) -> "CheckpointResult":
    """Convenience function to run a checkpoint once.

    Args:
        checkpoint: Checkpoint or checkpoint name.
        context: Additional context.

    Returns:
        CheckpointResult.
    """
    from truthound.checkpoint.registry import get_checkpoint

    if isinstance(checkpoint, str):
        checkpoint = get_checkpoint(checkpoint)

    return checkpoint.run(context=context)
