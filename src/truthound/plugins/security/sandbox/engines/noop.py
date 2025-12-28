"""No-op sandbox engine for trusted plugins.

This engine provides no isolation but still enforces timeout limits.
Use only for trusted plugins that don't require sandboxing.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable

from truthound.plugins.security.protocols import (
    IsolationLevel,
    SecurityPolicy,
    SandboxContext,
)
from truthound.plugins.security.sandbox.context import SandboxContextImpl
from truthound.plugins.security.exceptions import SandboxTimeoutError

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
        context = SandboxContextImpl(
            plugin_id=plugin_id,
            policy=policy,
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
        impl.mark_started()
        timeout = context.policy.resource_limits.max_execution_time_sec

        try:
            # Check if function is a coroutine
            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=timeout,
                )
            else:
                # Run sync function in executor with timeout
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: func(*args, **kwargs)),
                    timeout=timeout,
                )

            impl.mark_finished()
            return result

        except asyncio.TimeoutError:
            impl.mark_terminated()
            raise SandboxTimeoutError(
                f"Execution timed out after {timeout}s",
                plugin_id=context.plugin_id,
                sandbox_id=context.sandbox_id,
                timeout_seconds=timeout,
                execution_time=impl.execution_time_sec,
            )

    def terminate(self, context: SandboxContext) -> None:
        """Terminate sandbox (no-op for this engine).

        Args:
            context: Sandbox to terminate
        """
        impl = self._get_impl(context)
        impl.mark_terminated()
        self._contexts.pop(context.sandbox_id, None)

    async def cleanup(self) -> None:
        """Clean up all contexts."""
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
