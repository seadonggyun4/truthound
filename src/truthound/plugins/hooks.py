"""Hook system for event-driven plugin architecture.

This module provides a powerful hook system that allows plugins to
respond to events throughout the Truthound lifecycle.

Available Hooks:
    - before_validation: Called before validation starts
    - after_validation: Called after validation completes
    - before_profile: Called before profiling starts
    - after_profile: Called after profiling completes
    - on_report_generate: Called when a report is generated
    - on_issue_found: Called when a validation issue is found
    - on_datasource_connect: Called when a datasource connects
    - on_error: Called when an error occurs

Example:
    >>> from truthound.plugins.hooks import HookManager, before_validation
    >>>
    >>> hooks = HookManager()
    >>>
    >>> @before_validation
    ... def log_start(datasource, validators):
    ...     print(f"Starting validation on {datasource.name}")
    ...
    >>> hooks.register_decorated(log_start)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, TypeVar, Generic
from functools import wraps
import logging
import asyncio
from collections import defaultdict

logger = logging.getLogger(__name__)


# =============================================================================
# Hook Types
# =============================================================================


class HookType(str, Enum):
    """Types of hooks available in Truthound."""

    # Validation lifecycle
    BEFORE_VALIDATION = "before_validation"
    AFTER_VALIDATION = "after_validation"
    ON_ISSUE_FOUND = "on_issue_found"

    # Profiling lifecycle
    BEFORE_PROFILE = "before_profile"
    AFTER_PROFILE = "after_profile"

    # Reporting lifecycle
    ON_REPORT_GENERATE = "on_report_generate"
    ON_REPORT_WRITE = "on_report_write"

    # Data source lifecycle
    ON_DATASOURCE_CONNECT = "on_datasource_connect"
    ON_DATASOURCE_DISCONNECT = "on_datasource_disconnect"

    # Error handling
    ON_ERROR = "on_error"

    # Plugin lifecycle
    ON_PLUGIN_LOAD = "on_plugin_load"
    ON_PLUGIN_UNLOAD = "on_plugin_unload"

    # Custom/generic
    CUSTOM = "custom"


@dataclass
class Hook:
    """Represents a registered hook handler.

    Attributes:
        hook_type: Type of hook.
        handler: The callback function.
        priority: Execution order (lower = earlier).
        source: Plugin or module that registered this hook.
        async_handler: Whether handler is async.
        enabled: Whether hook is currently enabled.
    """

    hook_type: HookType | str
    handler: Callable[..., Any]
    priority: int = 100
    source: str = "unknown"
    async_handler: bool = False
    enabled: bool = True

    def __post_init__(self) -> None:
        """Detect if handler is async."""
        self.async_handler = asyncio.iscoroutinefunction(self.handler)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the hook handler."""
        if not self.enabled:
            return None
        return self.handler(*args, **kwargs)


# =============================================================================
# Hook Manager
# =============================================================================


class HookManager:
    """Manages hook registration and execution.

    The HookManager provides a central registry for hooks and handles
    their execution in priority order.

    Example:
        >>> manager = HookManager()
        >>>
        >>> def my_handler(data):
        ...     print(f"Got data: {data}")
        ...
        >>> manager.register(HookType.BEFORE_VALIDATION, my_handler)
        >>> manager.trigger(HookType.BEFORE_VALIDATION, data="test")
        Got data: test
    """

    def __init__(self) -> None:
        """Initialize the hook manager."""
        self._hooks: dict[str, list[Hook]] = defaultdict(list)
        self._hook_results: dict[str, list[Any]] = {}

    def register(
        self,
        hook_type: HookType | str,
        handler: Callable[..., Any],
        priority: int = 100,
        source: str = "unknown",
    ) -> Hook:
        """Register a hook handler.

        Args:
            hook_type: Type of hook to register for.
            handler: Callback function.
            priority: Execution order (lower = earlier).
            source: Identifier for the registering plugin.

        Returns:
            The created Hook instance.

        Example:
            >>> def on_start(datasource):
            ...     print(f"Validating {datasource.name}")
            >>> hook = manager.register(HookType.BEFORE_VALIDATION, on_start)
        """
        hook_key = hook_type.value if isinstance(hook_type, HookType) else hook_type

        hook = Hook(
            hook_type=hook_type,
            handler=handler,
            priority=priority,
            source=source,
        )

        self._hooks[hook_key].append(hook)
        # Keep hooks sorted by priority
        self._hooks[hook_key].sort(key=lambda h: h.priority)

        logger.debug(
            f"Registered hook: {hook_key} from {source} (priority={priority})"
        )
        return hook

    def register_decorated(self, func: Callable[..., Any]) -> Hook:
        """Register a hook from a decorated function.

        The function must have been decorated with one of the hook
        decorators (@before_validation, @after_validation, etc.)

        Args:
            func: Decorated function.

        Returns:
            The created Hook instance.
        """
        hook_info = getattr(func, "_truthound_hook", None)
        if not hook_info:
            raise ValueError(
                f"Function {func.__name__} is not decorated with a hook decorator"
            )

        return self.register(
            hook_type=hook_info["hook_type"],
            handler=func,
            priority=hook_info.get("priority", 100),
            source=hook_info.get("source", func.__module__),
        )

    def unregister(
        self,
        hook_type: HookType | str,
        handler: Callable[..., Any] | None = None,
        source: str | None = None,
    ) -> int:
        """Unregister hook handlers.

        Args:
            hook_type: Type of hook.
            handler: Specific handler to remove (if None, uses source).
            source: Remove all hooks from this source.

        Returns:
            Number of hooks removed.
        """
        hook_key = hook_type.value if isinstance(hook_type, HookType) else hook_type

        if hook_key not in self._hooks:
            return 0

        original_count = len(self._hooks[hook_key])

        if handler:
            self._hooks[hook_key] = [
                h for h in self._hooks[hook_key] if h.handler != handler
            ]
        elif source:
            self._hooks[hook_key] = [
                h for h in self._hooks[hook_key] if h.source != source
            ]

        removed = original_count - len(self._hooks[hook_key])
        logger.debug(f"Unregistered {removed} hook(s) from {hook_key}")
        return removed

    def trigger(
        self,
        hook_type: HookType | str,
        *args: Any,
        **kwargs: Any,
    ) -> list[Any]:
        """Trigger all handlers for a hook type.

        Handlers are executed in priority order (lowest first).
        Results from all handlers are collected and returned.

        Args:
            hook_type: Type of hook to trigger.
            *args: Positional arguments for handlers.
            **kwargs: Keyword arguments for handlers.

        Returns:
            List of results from all handlers.

        Example:
            >>> results = manager.trigger(
            ...     HookType.BEFORE_VALIDATION,
            ...     datasource=my_source,
            ...     validators=["null", "range"]
            ... )
        """
        hook_key = hook_type.value if isinstance(hook_type, HookType) else hook_type

        results: list[Any] = []
        for hook in self._hooks.get(hook_key, []):
            if not hook.enabled:
                continue

            try:
                result = hook.handler(*args, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(
                    f"Error in hook handler {hook.handler.__name__} "
                    f"from {hook.source}: {e}"
                )
                # Store error but continue with other handlers
                results.append(None)

        return results

    async def trigger_async(
        self,
        hook_type: HookType | str,
        *args: Any,
        **kwargs: Any,
    ) -> list[Any]:
        """Trigger all handlers for a hook type asynchronously.

        Async handlers are awaited, sync handlers are executed normally.
        All handlers run in priority order (not parallel).

        Args:
            hook_type: Type of hook to trigger.
            *args: Positional arguments for handlers.
            **kwargs: Keyword arguments for handlers.

        Returns:
            List of results from all handlers.
        """
        hook_key = hook_type.value if isinstance(hook_type, HookType) else hook_type

        results: list[Any] = []
        for hook in self._hooks.get(hook_key, []):
            if not hook.enabled:
                continue

            try:
                if hook.async_handler:
                    result = await hook.handler(*args, **kwargs)
                else:
                    result = hook.handler(*args, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(
                    f"Error in async hook handler {hook.handler.__name__} "
                    f"from {hook.source}: {e}"
                )
                results.append(None)

        return results

    def get_hooks(self, hook_type: HookType | str | None = None) -> list[Hook]:
        """Get registered hooks.

        Args:
            hook_type: Filter by hook type (None for all).

        Returns:
            List of Hook instances.
        """
        if hook_type is None:
            return [hook for hooks in self._hooks.values() for hook in hooks]

        hook_key = hook_type.value if isinstance(hook_type, HookType) else hook_type
        return list(self._hooks.get(hook_key, []))

    def clear(self, hook_type: HookType | str | None = None) -> None:
        """Clear registered hooks.

        Args:
            hook_type: Type to clear (None for all).
        """
        if hook_type is None:
            self._hooks.clear()
        else:
            hook_key = hook_type.value if isinstance(hook_type, HookType) else hook_type
            self._hooks.pop(hook_key, None)

    def enable(
        self,
        hook_type: HookType | str | None = None,
        source: str | None = None,
    ) -> int:
        """Enable hooks.

        Args:
            hook_type: Type to enable (None for all).
            source: Enable only hooks from this source.

        Returns:
            Number of hooks enabled.
        """
        return self._set_enabled(True, hook_type, source)

    def disable(
        self,
        hook_type: HookType | str | None = None,
        source: str | None = None,
    ) -> int:
        """Disable hooks.

        Args:
            hook_type: Type to disable (None for all).
            source: Disable only hooks from this source.

        Returns:
            Number of hooks disabled.
        """
        return self._set_enabled(False, hook_type, source)

    def _set_enabled(
        self,
        enabled: bool,
        hook_type: HookType | str | None,
        source: str | None,
    ) -> int:
        """Set enabled state for hooks."""
        count = 0
        hooks = self.get_hooks(hook_type)

        for hook in hooks:
            if source is None or hook.source == source:
                hook.enabled = enabled
                count += 1

        return count

    def __len__(self) -> int:
        """Return total number of registered hooks."""
        return sum(len(hooks) for hooks in self._hooks.values())

    def __repr__(self) -> str:
        hook_counts = {k: len(v) for k, v in self._hooks.items()}
        return f"<HookManager hooks={hook_counts}>"


# =============================================================================
# Hook Decorators
# =============================================================================


def hook(
    hook_type: HookType | str,
    priority: int = 100,
    source: str | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Generic decorator to mark a function as a hook handler.

    Args:
        hook_type: Type of hook.
        priority: Execution order (lower = earlier).
        source: Identifier for the source (default: module name).

    Returns:
        Decorator function.

    Example:
        >>> @hook(HookType.BEFORE_VALIDATION, priority=50)
        ... def my_handler(datasource, validators):
        ...     print("Before validation")
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        func._truthound_hook = {  # type: ignore
            "hook_type": hook_type,
            "priority": priority,
            "source": source or func.__module__,
        }

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        wrapper._truthound_hook = func._truthound_hook  # type: ignore
        return wrapper

    return decorator


def before_validation(
    priority: int = 100,
    source: str | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator for before_validation hooks.

    Handler signature: (datasource, validators, **kwargs) -> None

    Example:
        >>> @before_validation()
        ... def log_start(datasource, validators):
        ...     print(f"Validating {datasource.name} with {len(validators)} validators")
    """
    return hook(HookType.BEFORE_VALIDATION, priority, source)


def after_validation(
    priority: int = 100,
    source: str | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator for after_validation hooks.

    Handler signature: (datasource, result, issues, **kwargs) -> None

    Example:
        >>> @after_validation()
        ... def log_result(datasource, result, issues):
        ...     print(f"Found {len(issues)} issues in {datasource.name}")
    """
    return hook(HookType.AFTER_VALIDATION, priority, source)


def before_profile(
    priority: int = 100,
    source: str | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator for before_profile hooks.

    Handler signature: (datasource, config, **kwargs) -> None
    """
    return hook(HookType.BEFORE_PROFILE, priority, source)


def after_profile(
    priority: int = 100,
    source: str | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator for after_profile hooks.

    Handler signature: (datasource, profile, **kwargs) -> None
    """
    return hook(HookType.AFTER_PROFILE, priority, source)


def on_report_generate(
    priority: int = 100,
    source: str | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator for on_report_generate hooks.

    Handler signature: (report, format, **kwargs) -> report (can modify)
    """
    return hook(HookType.ON_REPORT_GENERATE, priority, source)


def on_issue_found(
    priority: int = 100,
    source: str | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator for on_issue_found hooks.

    Handler signature: (issue, validator, **kwargs) -> None
    """
    return hook(HookType.ON_ISSUE_FOUND, priority, source)


def on_error(
    priority: int = 100,
    source: str | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator for on_error hooks.

    Handler signature: (error, context, **kwargs) -> None
    """
    return hook(HookType.ON_ERROR, priority, source)


# =============================================================================
# Global Hook Manager
# =============================================================================

# Global hook manager instance
_global_hooks: HookManager | None = None


def get_hook_manager() -> HookManager:
    """Get the global hook manager instance."""
    global _global_hooks
    if _global_hooks is None:
        _global_hooks = HookManager()
    return _global_hooks


def reset_hook_manager() -> None:
    """Reset the global hook manager (mainly for testing)."""
    global _global_hooks
    _global_hooks = None
