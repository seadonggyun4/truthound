"""No-op sandbox engine for trusted plugins.

This engine provides no isolation. Deadlines stop awaiting results; trusted
synchronous work cannot be preempted and cleanup waits for its completion.
Use only for trusted plugins that don't require sandboxing.
"""

from __future__ import annotations

import asyncio
import logging
import math
from copy import deepcopy
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

from truthound.plugins.security.exceptions import SandboxTimeoutError
from truthound.plugins.security.protocols import (
    IsolationLevel,
    SandboxContext,
    SecurityPolicy,
)
from truthound.plugins.security.sandbox.context import SandboxContextImpl

logger = logging.getLogger(__name__)


class NoopSandboxEngine:
    """No-isolation sandbox engine.

    Executes code directly in the current process with only timeout
    enforcement. Suitable for trusted plugins that don't need isolation.

    Warning:
        This engine provides NO security isolation. Only use for
        fully trusted plugins in development or controlled environments.
    """

    @property
    def isolation_level(self) -> IsolationLevel:
        """Return the isolation level provided by this engine."""
        return IsolationLevel.NONE

    def __init__(self) -> None:
        """Initialize the no-op sandbox engine."""
        self._contexts: dict[str, SandboxContextImpl] = {}
        self._workers = set()
        self._executions = {}

    def create_sandbox(
        self,
        plugin_id: str,
        policy: SecurityPolicy,
    ) -> SandboxContext:
        """Create a sandbox context (no actual sandbox created).

        Args:
            plugin_id: Plugin identifier
            policy: Security policy (only timeout is enforced)

        Returns:
            SandboxContext for tracking
        """
        timeout = policy.resource_limits.max_execution_time_sec
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("Timeout must be finite and positive")
        context = SandboxContextImpl(
            plugin_id=plugin_id,
            policy=deepcopy(policy),
        )
        self._contexts[context.sandbox_id] = context
        logger.debug(f"Created no-op sandbox context for {plugin_id}")
        return context

    async def execute(
        self,
        context: SandboxContext,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute function with timeout enforcement only.

        Args:
            context: Sandbox context
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            SandboxTimeoutError: If execution times out
        """
        impl = self._get_impl(context)
        if self._contexts.get(context.sandbox_id) is not context:
            raise ValueError("Sandbox context is not owned by this engine")
        if impl.started_at is not None or not impl.is_alive():
            raise ValueError("Sandbox context is single-use")
        impl.mark_started()
        self._executions[context.sandbox_id] = asyncio.current_task()
        timeout = impl.policy.resource_limits.max_execution_time_sec
        asynchronous = asyncio.iscoroutinefunction(func)
        worker = None
        try:
            if asynchronous:
                worker = asyncio.create_task(func(*args, **kwargs))
            else:
                worker = asyncio.get_running_loop().run_in_executor(
                    None, lambda: func(*args, **kwargs)
                )
            self._workers.add(worker)

            def completed(future):
                self._workers.discard(future)
                if not future.cancelled():
                    future.exception()  # Retrieve errors from timed-out thread work.

            worker.add_done_callback(completed)
            return await asyncio.wait_for(asyncio.shield(worker), timeout=timeout)
        except TimeoutError:
            raise SandboxTimeoutError(
                f"Execution timed out after {timeout}s",
                plugin_id=context.plugin_id,
                sandbox_id=context.sandbox_id,
                timeout_seconds=timeout,
                execution_time=impl.execution_time_sec,
            ) from None
        finally:
            if asynchronous and worker is not None and not worker.done():
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)
            impl.mark_finished()
            self._executions.pop(context.sandbox_id, None)

    def terminate(self, context: SandboxContext) -> None:
        """Terminate sandbox (no-op for this engine).

        Args:
            context: Sandbox to terminate
        """
        impl = self._get_impl(context)
        impl.mark_terminated()
        execution = self._executions.get(context.sandbox_id)
        if self._contexts.get(context.sandbox_id) is context and execution is not None:
            execution.cancel()

    async def cleanup(self) -> None:
        """Clean up all contexts."""
        executions = list(self._executions.values())
        for execution in executions:
            execution.cancel()
        await asyncio.gather(*executions, return_exceptions=True)
        await asyncio.gather(
            *(asyncio.shield(worker) for worker in list(self._workers)), return_exceptions=True
        )
        for context in self._contexts.values():
            context.mark_terminated()
        self._contexts.clear()

    def _get_impl(self, context: SandboxContext) -> SandboxContextImpl:
        """Get implementation from context."""
        if isinstance(context, SandboxContextImpl):
            return context
        # Fallback: look up by sandbox_id
        impl = self._contexts.get(context.sandbox_id)
        if impl is None:
            raise ValueError(f"Unknown sandbox context: {context.sandbox_id}")
        return impl
