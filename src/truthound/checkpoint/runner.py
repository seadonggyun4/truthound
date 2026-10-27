"""Checkpoint runner for executing validation pipelines.

This module provides the CheckpointRunner for managing checkpoint execution,
including scheduled runs, concurrent execution, and result handling.
"""

from __future__ import annotations

import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from queue import Empty, Full, Queue
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from truthound.checkpoint.checkpoint import Checkpoint, CheckpointResult


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
    result_callback: Callable[[CheckpointResult], None] | None = None
    error_callback: Callable[[Exception], None] | None = None
    stop_on_error: bool = False
    max_failures: int = 10
    result_queue_size: int = 1000
    journal_path: str | None = None


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

        self._checkpoints: dict[str, Checkpoint] = {}
        self._running = False
        self._thread: threading.Thread | None = None
        self._result_queue: Queue[CheckpointResult] = Queue(maxsize=self._config.result_queue_size)
        self._lock = threading.RLock()
        self._execution_lock = threading.RLock()
        self._active_runs = 0
        self._consecutive_failures = 0
        self._execution_guard = None
        self._durable = None
        self._closed = False
        if self._config.journal_path:
            from truthound.checkpoint.durable_runner import DurableRunnerState

            self._durable = DurableRunnerState(self._config.journal_path)

    def bind_recovery_scope(self, scope: str) -> None:
        """Pin durable recovery to the current operator authorization."""
        if self._durable:
            self._durable.bind_scope(scope)

    def set_execution_guard(self, guard) -> None:
        """Check current operator authorization before every native action run."""
        self._execution_guard = guard

    def _ensure_owner(self):
        if self._closed:
            raise RuntimeError("Runner is closed")
        if self._execution_guard:
            self._execution_guard()
        if self._durable:
            self._durable.acquire_owner()

    def close(self):
        """Drain execution before releasing durable ownership."""
        self.stop()
        with self._lock:
            if self._active_runs or (self._thread and self._thread.is_alive()):
                raise RuntimeError("Runner execution has not stopped")
            if self._durable and not self._closed:
                self._durable.close()
            self._closed = True

    def recovery_status(self, *, limit=100, offset=0):
        if not 1 <= limit <= 1000 or offset < 0:
            raise ValueError("Invalid recovery page")
        with self._lock:
            rows = self._durable.attempts(limit=limit + 1, offset=offset) if self._durable else []
            return {
                "enabled": self._durable is not None,
                "attempts": [
                    {"identity": identity, "state": state} for identity, state, _ in rows[:limit]
                ],
                "next_offset": offset + limit if len(rows) > limit else None,
            }

    def recovery_resolve(self, identity, resolution):
        """Confirm or discard an uncertain run without executing its actions."""
        with self._lock:
            self._ensure_owner()
            if (
                self._running
                or self._active_runs
                or (self._thread and self._thread.is_alive())
                or not self._durable
            ):
                raise RuntimeError("Recovery requires a quiescent durable runner")
            journal = self._durable
            row = journal.connection.execute(
                "SELECT payload FROM execution_attempts WHERE identity=?", (identity,)
            ).fetchone()
            if not row or not identity.startswith("runner:"):
                raise ValueError("Unknown runner attempt")
            import json

            from truthound.checkpoint.durable_runner import configuration

            payload = json.loads(row[0])
            checkpoint = self._checkpoints.get(payload["checkpoint"])
            with journal.transaction():
                journal.resolve_unknown(identity, resolution)
                if payload["trigger"] is not None:
                    if checkpoint is None:
                        raise ValueError("Missing checkpoint")
                    triggers = [t for t in checkpoint.triggers if t.name == payload["trigger"]]
                    if len(triggers) != 1 or configuration(triggers[0]) != payload["configuration"]:
                        raise ValueError("Trigger definition changed")
                    trigger = triggers[0]
                    journal.restore_trigger(checkpoint, trigger)
                    trigger.record_run()
                    journal.save_trigger(checkpoint, trigger)
            return {"identity": identity, "state": resolution}

    def recovery_forget(self, identities):
        with self._lock:
            self._ensure_owner()
            if (
                self._running
                or self._active_runs
                or (self._thread and self._thread.is_alive())
                or not self._durable
            ):
                raise RuntimeError("Retention requires a quiescent durable runner")
            if any(not identity.startswith("runner:") for identity in identities):
                raise ValueError("Invalid runner attempt identity")
            return {"removed": self._durable.forget_terminal(identities)}

    def _run_reserved(self, checkpoint, context=None, *, trigger=None, identity=None):
        with self._execution_lock:
            with self._lock:
                self._ensure_owner()
                journal = self._durable
                if journal:
                    if journal.blocked(checkpoint, trigger):
                        raise RuntimeError("Checkpoint has an unresolved execution outcome")
                    identity = identity or journal.reserve(checkpoint, trigger, context)
                    if not journal.claim(identity):
                        raise RuntimeError("Checkpoint execution already claimed")
                self._active_runs += 1
            try:
                result = checkpoint.run(context=context)
                self._dispatch_result_callback(
                    result, suppress_errors=trigger is not None and journal is None
                )
                with self._lock:
                    if journal:
                        with journal.transaction():
                            if trigger:
                                trigger.record_run()
                                journal.save_trigger(checkpoint, trigger)
                            journal.finish(identity, "completed")
                    elif trigger:
                        trigger.record_run()
                return result
            except BaseException:
                if journal:
                    with suppress(Exception):
                        journal.finish(identity, "unknown")
                raise
            finally:
                with self._lock:
                    self._active_runs -= 1

    @property
    def running(self) -> bool:
        """Check if runner is active."""
        return self._running

    @property
    def checkpoints(self) -> dict[str, Checkpoint]:
        """Get registered checkpoints."""
        return self._checkpoints.copy()

    def add_checkpoint(self, checkpoint: Checkpoint) -> CheckpointRunner:
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

    def edit_checkpoint(self, checkpoint_name: str, changes: dict[str, Any]):
        """Edit declarative validation settings while all execution is quiescent.

        Data sources, actions, routers and callbacks keep their existing owners.
        Native callers may supply native validator objects in ``validators``.
        """
        from dataclasses import replace

        from truthound.checkpoint._validation import validate_checkpoint_config

        fields = {
            "validators",
            "validator_config",
            "min_severity",
            "auto_schema",
            "run_name_template",
            "tags",
            "metadata",
            "fail_on_critical",
            "fail_on_high",
            "timeout_seconds",
            "sample_size",
        }
        if not changes or set(changes) - fields:
            raise ValueError("Unsupported checkpoint definition setting")
        with self._lock:
            self._ensure_owner()
            if self._running or self._active_runs or (self._thread and self._thread.is_alive()):
                raise RuntimeError("Runner must be quiescent before editing checkpoints")
            checkpoint = self._checkpoints[checkpoint_name]
            candidate = replace(checkpoint.config, **changes)
            if candidate.timeout_seconds <= 0 or (
                candidate.sample_size is not None and candidate.sample_size <= 0
            ):
                raise ValueError("Checkpoint limits must be positive")
            errors = validate_checkpoint_config(
                candidate, actions=checkpoint.actions, triggers=checkpoint.triggers
            )
            if errors:
                raise ValueError("; ".join(errors))
            checkpoint._config = candidate
            return checkpoint

    def edit_schedule(self, checkpoint_name: str, trigger_name: str, changes: dict[str, Any]):
        """Edit interval, cron or shared provider settings, retaining history.

        Stopping is not sufficient if the worker has not actually exited. Refuse
        that case rather than racing an in-flight action or resetting its clock.
        Custom trigger implementations remain owned by their provider.
        """
        from dataclasses import replace

        from truthound.checkpoint.triggers.schedule import CronTrigger, ScheduleTrigger

        with self._lock:
            if (
                self._running
                or self._active_runs
                or (self._thread is not None and self._thread.is_alive())
            ):
                raise RuntimeError("Runner must be quiescent before editing schedules")
            checkpoint = self._checkpoints[checkpoint_name]
            matches = [trigger for trigger in checkpoint.triggers if trigger.name == trigger_name]
            if len(matches) != 1:
                raise ValueError("Expected one named trigger")
            trigger = matches[0]
            if not changes or "name" in changes:
                raise ValueError("Supply configuration changes without renaming the trigger")
            self._ensure_owner()
            from dataclasses import fields as config_fields

            from truthound.checkpoint.triggers.base import TriggerConfig

            if type(trigger) not in {ScheduleTrigger, CronTrigger}:
                common = {item.name for item in config_fields(TriggerConfig)} - {"name"}
                if set(changes) - common:
                    raise ValueError("Custom trigger edits accept shared settings only")
                # Do not reconstruct providers, open connections or replace callbacks.
                from copy import copy

                candidate = copy(trigger)
                candidate._config = replace(trigger.config, **changes)
            else:
                candidate = type(trigger)(replace(trigger.config, **changes))
            errors = candidate.validate_config()
            if errors:
                raise ValueError("; ".join(errors))
            # Preserve the same trigger, run count, last run and attachment. A
            # replacement would look like a first run and could repeat actions.
            if self._durable:
                from truthound.checkpoint.durable_runner import configuration

                self._durable.restore_trigger(
                    checkpoint, trigger, compatible_configuration=configuration(candidate)
                )
                old_config = trigger.config
                try:
                    with self._durable.transaction():
                        trigger._config = candidate.config
                        self._durable.save_trigger(checkpoint, trigger)
                except BaseException:
                    trigger._config = old_config
                    raise
            else:
                trigger._config = candidate.config
            if type(trigger) in {ScheduleTrigger, CronTrigger}:
                trigger._next_run = None
            if type(trigger) is CronTrigger:
                trigger._parsed_cron = None
            return trigger

    def run_once(
        self,
        checkpoint: Checkpoint | str,
        context: dict[str, Any] | None = None,
    ) -> CheckpointResult:
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

        return self._run_reserved(checkpoint, context)

    def run_all(
        self,
        context: dict[str, Any] | None = None,
    ) -> list[CheckpointResult]:
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
            results.append(self._run_reserved(checkpoint, context))

        return results

    def start(self, blocking: bool = False) -> None:
        """Start the runner.

        Args:
            blocking: If True, block until stopped.
        """
        # Start triggers
        with self._lock:
            if self._running:
                return
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("Previous runner thread has not stopped")
            self._ensure_owner()
            if self._durable:
                from truthound.checkpoint.durable_runner import configuration

                for checkpoint in self._checkpoints.values():
                    names = [trigger.name for trigger in checkpoint.triggers]
                    if len(names) != len(set(names)):
                        raise ValueError("Durable trigger names must be unique")
                    for trigger in checkpoint.triggers:
                        configuration(trigger)
                        self._durable.restore_trigger(checkpoint, trigger)
            self._running = True
            for checkpoint in self._checkpoints.values():
                for trigger in checkpoint.triggers:
                    trigger.start()
                    if self._durable:
                        # start() hooks cannot reset already persisted execution history.
                        self._durable.restore_trigger(checkpoint, trigger)
                        with self._durable.transaction():
                            self._durable.save_trigger(checkpoint, trigger)

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

                self._dispatch_error_callback(e, suppress_errors=True)

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

        # Replay only reservations committed before their execution claim.
        if self._durable:
            self._ensure_owner()
            for identity, payload in self._durable.pending():
                if not identity.startswith("runner:"):
                    continue
                checkpoint = self._checkpoints.get(payload["checkpoint"])
                if checkpoint is None:
                    continue
                matches = [t for t in checkpoint.triggers if t.name == payload["trigger"]]
                trigger = matches[0] if len(matches) == 1 else None
                if payload["trigger"] is not None:
                    from truthound.checkpoint.durable_runner import configuration

                    if (
                        trigger is None
                        or not trigger.enabled
                        or configuration(trigger) != payload["configuration"]
                    ):
                        continue
                result = self._run_reserved(
                    checkpoint, payload["context"], trigger=trigger, identity=identity
                )
                self._enqueue_result(result)

        for checkpoint in checkpoints:
            for trigger in checkpoint.triggers:
                if not trigger.enabled or not trigger.should_continue():
                    continue
                if self._durable and self._durable.blocked(checkpoint, trigger):
                    continue
                result = trigger.should_trigger()
                if result.should_run:
                    try:
                        cp_result = self._run_reserved(
                            checkpoint, {"trigger": trigger.name, **result.context}, trigger=trigger
                        )
                        self._consecutive_failures = 0
                        self._enqueue_result(cp_result)
                    except Exception as error:
                        self._consecutive_failures += 1
                        self._dispatch_error_callback(error, suppress_errors=True)

    def _enqueue_result(self, result: CheckpointResult) -> None:
        """Queue a completed result when capacity is available."""
        with suppress(Full):
            self._result_queue.put_nowait(result)

    def _dispatch_result_callback(
        self,
        result: CheckpointResult,
        *,
        suppress_errors: bool = False,
    ) -> None:
        """Invoke result callback with optional non-fatal suppression."""
        callback = self._config.result_callback
        if callback is None:
            return
        if suppress_errors:
            with suppress(Exception):
                callback(result)
            return
        callback(result)

    def _dispatch_error_callback(
        self,
        error: Exception,
        *,
        suppress_errors: bool = False,
    ) -> None:
        """Invoke error callback with optional non-fatal suppression."""
        callback = self._config.error_callback
        if callback is None:
            return
        if suppress_errors:
            with suppress(Exception):
                callback(error)
            return
        callback(error)

    def get_results(
        self,
        timeout: float | None = None,
        max_results: int = 100,
    ) -> list[CheckpointResult]:
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
    ) -> Iterator[CheckpointResult]:
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
    checkpoint: Checkpoint | str,
    context: dict[str, Any] | None = None,
) -> CheckpointResult:
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
