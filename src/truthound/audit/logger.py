"""Main audit logger implementation.

This module provides the primary AuditLogger class that combines
storage, formatters, filters, and processors into a unified interface.
"""

from __future__ import annotations

import functools
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Iterator, TypeVar

from truthound.audit.core import (
    AuditEvent,
    AuditEventType,
    AuditSeverity,
    AuditCategory,
    AuditOutcome,
    AuditActor,
    AuditResource,
    AuditContext,
    AuditConfig,
    AuditStorage,
    AuditFilter,
    AuditProcessor,
    AuditEventBuilder,
    current_timestamp,
)
from truthound.audit.storage import MemoryAuditStorage, create_storage
from truthound.audit.filters import (
    create_filter_from_config,
    create_processor_from_config,
    CompositeFilter,
    CompositeProcessor,
)


F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# Thread-Local Context
# =============================================================================


class _AuditContext:
    """Thread-local audit context storage."""

    _local = threading.local()

    @classmethod
    def get_actor(cls) -> AuditActor | None:
        """Get current actor."""
        return getattr(cls._local, "actor", None)

    @classmethod
    def set_actor(cls, actor: AuditActor | None) -> None:
        """Set current actor."""
        cls._local.actor = actor

    @classmethod
    def get_context(cls) -> AuditContext:
        """Get current context."""
        return getattr(cls._local, "context", AuditContext())

    @classmethod
    def set_context(cls, context: AuditContext) -> None:
        """Set current context."""
        cls._local.context = context

    @classmethod
    def get_request_id(cls) -> str:
        """Get current request ID."""
        return getattr(cls._local, "request_id", "")

    @classmethod
    def set_request_id(cls, request_id: str) -> None:
        """Set current request ID."""
        cls._local.request_id = request_id

    @classmethod
    def clear(cls) -> None:
        """Clear all context."""
        cls._local.actor = None
        cls._local.context = AuditContext()
        cls._local.request_id = ""


# =============================================================================
# Audit Logger
# =============================================================================


class AuditLogger:
    """Main audit logger class.

    Provides a unified interface for logging audit events with support for
    multiple storage backends, filters, processors, and formatters.

    Example:
        >>> logger = AuditLogger(
        ...     config=AuditConfig.production("my-service"),
        ... )
        >>>
        >>> # Log an event
        >>> logger.log(
        ...     event_type=AuditEventType.UPDATE,
        ...     action="update_user",
        ...     actor=AuditActor(id="user:123"),
        ...     resource=AuditResource(id="user:456", type="user"),
        ...     outcome=AuditOutcome.SUCCESS,
        ... )
        >>>
        >>> # Or use the builder
        >>> logger.log_event(
        ...     AuditEventBuilder()
        ...     .set_type(AuditEventType.LOGIN)
        ...     .set_actor(id="user:123")
        ...     .set_outcome(AuditOutcome.SUCCESS)
        ...     .build()
        ... )
    """

    def __init__(
        self,
        config: AuditConfig | None = None,
        *,
        storage: AuditStorage | None = None,
        filters: list[AuditFilter] | None = None,
        processors: list[AuditProcessor] | None = None,
        name: str = "default",
    ) -> None:
        """Initialize audit logger.

        Args:
            config: Audit configuration.
            storage: Storage backend (uses config if not provided).
            filters: Additional filters.
            processors: Additional processors.
            name: Logger name.
        """
        self._name = name
        self._config = config or AuditConfig()
        self._lock = threading.Lock()

        # Initialize storage
        if storage:
            self._storage = storage
        else:
            self._storage = create_storage(
                self._config.storage_backend,
                **self._config.storage_config,
            )

        # Build filter chain
        config_filter = create_filter_from_config(self._config)
        if filters:
            self._filter = CompositeFilter([config_filter, *filters], mode="all")
        else:
            self._filter = config_filter

        # Build processor chain
        config_processor = create_processor_from_config(self._config)
        if processors:
            self._processor = CompositeProcessor([config_processor, *processors])
        else:
            self._processor = config_processor

    @property
    def name(self) -> str:
        """Get logger name."""
        return self._name

    @property
    def config(self) -> AuditConfig:
        """Get configuration."""
        return self._config

    @property
    def storage(self) -> AuditStorage:
        """Get storage backend."""
        return self._storage

    def log(
        self,
        event_type: AuditEventType,
        action: str = "",
        *,
        actor: AuditActor | None = None,
        resource: AuditResource | None = None,
        target: AuditResource | None = None,
        outcome: AuditOutcome = AuditOutcome.SUCCESS,
        message: str = "",
        reason: str = "",
        severity: AuditSeverity = AuditSeverity.INFO,
        category: AuditCategory | None = None,
        data: dict[str, Any] | None = None,
        old_value: Any = None,
        new_value: Any = None,
        tags: list[str] | None = None,
        duration_ms: float | None = None,
    ) -> AuditEvent | None:
        """Log an audit event.

        Args:
            event_type: Type of event.
            action: Action name.
            actor: Actor performing the action.
            resource: Resource being acted upon.
            target: Target resource (for copy/move operations).
            outcome: Outcome of the action.
            message: Human-readable message.
            reason: Reason for the action/outcome.
            severity: Event severity.
            category: Event category.
            data: Additional event data.
            old_value: Previous value (for changes).
            new_value: New value (for changes).
            tags: Event tags.
            duration_ms: Duration in milliseconds.

        Returns:
            Logged event or None if filtered.
        """
        if not self._config.enabled:
            return None

        # Use thread-local context if actor not provided
        if actor is None:
            actor = _AuditContext.get_actor() or AuditActor.system()

        # Get context
        context = _AuditContext.get_context()
        if _AuditContext.get_request_id():
            context.request_id = _AuditContext.get_request_id()

        # Infer category if not provided
        if category is None:
            category = self._infer_category(event_type)

        # Build event
        event = AuditEvent(
            event_type=event_type,
            category=category,
            severity=severity,
            action=action,
            outcome=outcome,
            message=message,
            reason=reason,
            actor=actor,
            resource=resource,
            target=target,
            context=context,
            old_value=old_value,
            new_value=new_value,
            data=data or {},
            tags=tags or [],
            duration_ms=duration_ms,
        )

        return self.log_event(event)

    def log_event(self, event: AuditEvent) -> AuditEvent | None:
        """Log a pre-built audit event.

        Args:
            event: Event to log.

        Returns:
            Logged event or None if filtered.
        """
        if not self._config.enabled:
            return None

        # Apply filter
        if not self._filter.should_log(event):
            return None

        # Apply processors
        event = self._processor.process(event)

        # Write to storage
        try:
            self._storage.write(event)
        except Exception as e:
            # Log to stderr as fallback
            import sys
            print(f"Audit log write failed: {e}", file=sys.stderr)
            return None

        return event

    def log_login(
        self,
        actor: AuditActor,
        *,
        outcome: AuditOutcome = AuditOutcome.SUCCESS,
        reason: str = "",
        ip_address: str = "",
        **kwargs: Any,
    ) -> AuditEvent | None:
        """Log a login event."""
        if ip_address and actor:
            actor.ip_address = ip_address

        return self.log(
            event_type=AuditEventType.LOGIN if outcome == AuditOutcome.SUCCESS else AuditEventType.LOGIN_FAILED,
            action="login",
            actor=actor,
            outcome=outcome,
            reason=reason,
            category=AuditCategory.AUTHENTICATION,
            **kwargs,
        )

    def log_logout(
        self,
        actor: AuditActor,
        **kwargs: Any,
    ) -> AuditEvent | None:
        """Log a logout event."""
        return self.log(
            event_type=AuditEventType.LOGOUT,
            action="logout",
            actor=actor,
            outcome=AuditOutcome.SUCCESS,
            category=AuditCategory.AUTHENTICATION,
            **kwargs,
        )

    def log_access(
        self,
        resource: AuditResource,
        *,
        actor: AuditActor | None = None,
        outcome: AuditOutcome = AuditOutcome.SUCCESS,
        **kwargs: Any,
    ) -> AuditEvent | None:
        """Log a resource access event."""
        return self.log(
            event_type=AuditEventType.READ,
            action="access",
            actor=actor,
            resource=resource,
            outcome=outcome,
            category=AuditCategory.DATA_ACCESS,
            **kwargs,
        )

    def log_create(
        self,
        resource: AuditResource,
        *,
        actor: AuditActor | None = None,
        new_value: Any = None,
        **kwargs: Any,
    ) -> AuditEvent | None:
        """Log a resource creation event."""
        return self.log(
            event_type=AuditEventType.CREATE,
            action="create",
            actor=actor,
            resource=resource,
            outcome=AuditOutcome.SUCCESS,
            category=AuditCategory.DATA_MODIFICATION,
            new_value=new_value,
            **kwargs,
        )

    def log_update(
        self,
        resource: AuditResource,
        *,
        actor: AuditActor | None = None,
        old_value: Any = None,
        new_value: Any = None,
        **kwargs: Any,
    ) -> AuditEvent | None:
        """Log a resource update event."""
        return self.log(
            event_type=AuditEventType.UPDATE,
            action="update",
            actor=actor,
            resource=resource,
            outcome=AuditOutcome.SUCCESS,
            category=AuditCategory.DATA_MODIFICATION,
            old_value=old_value,
            new_value=new_value,
            **kwargs,
        )

    def log_delete(
        self,
        resource: AuditResource,
        *,
        actor: AuditActor | None = None,
        old_value: Any = None,
        **kwargs: Any,
    ) -> AuditEvent | None:
        """Log a resource deletion event."""
        return self.log(
            event_type=AuditEventType.DELETE,
            action="delete",
            actor=actor,
            resource=resource,
            outcome=AuditOutcome.SUCCESS,
            category=AuditCategory.DATA_MODIFICATION,
            old_value=old_value,
            **kwargs,
        )

    def log_security(
        self,
        action: str,
        *,
        actor: AuditActor | None = None,
        message: str = "",
        severity: AuditSeverity = AuditSeverity.WARNING,
        **kwargs: Any,
    ) -> AuditEvent | None:
        """Log a security event."""
        return self.log(
            event_type=AuditEventType.SECURITY_ALERT,
            action=action,
            actor=actor,
            message=message,
            severity=severity,
            category=AuditCategory.SECURITY,
            **kwargs,
        )

    def log_validation(
        self,
        action: str,
        *,
        resource: AuditResource | None = None,
        outcome: AuditOutcome = AuditOutcome.SUCCESS,
        data: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AuditEvent | None:
        """Log a validation event."""
        event_type = (
            AuditEventType.VALIDATION_COMPLETE
            if outcome == AuditOutcome.SUCCESS
            else AuditEventType.VALIDATION_FAILED
        )
        return self.log(
            event_type=event_type,
            action=action,
            resource=resource,
            outcome=outcome,
            category=AuditCategory.VALIDATION,
            data=data,
            **kwargs,
        )

    def query(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        event_types: list[AuditEventType] | None = None,
        actor_id: str | None = None,
        resource_id: str | None = None,
        outcome: AuditOutcome | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AuditEvent]:
        """Query audit events."""
        return self._storage.query(
            start_time=start_time,
            end_time=end_time,
            event_types=event_types,
            actor_id=actor_id,
            resource_id=resource_id,
            outcome=outcome,
            limit=limit,
            offset=offset,
        )

    def get_event(self, event_id: str) -> AuditEvent | None:
        """Get a single event by ID."""
        return self._storage.read(event_id)

    def count(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        event_types: list[AuditEventType] | None = None,
    ) -> int:
        """Count matching events."""
        return self._storage.count(start_time, end_time, event_types)

    def flush(self) -> None:
        """Flush buffered events."""
        self._storage.flush()

    def close(self) -> None:
        """Close logger and storage."""
        self._storage.close()

    def _infer_category(self, event_type: AuditEventType) -> AuditCategory:
        """Infer category from event type."""
        mapping = {
            AuditEventType.LOGIN: AuditCategory.AUTHENTICATION,
            AuditEventType.LOGOUT: AuditCategory.AUTHENTICATION,
            AuditEventType.LOGIN_FAILED: AuditCategory.AUTHENTICATION,
            AuditEventType.PASSWORD_CHANGE: AuditCategory.AUTHENTICATION,
            AuditEventType.PERMISSION_CHANGE: AuditCategory.AUTHORIZATION,
            AuditEventType.ACCESS_DENIED: AuditCategory.AUTHORIZATION,
            AuditEventType.CREATE: AuditCategory.DATA_MODIFICATION,
            AuditEventType.UPDATE: AuditCategory.DATA_MODIFICATION,
            AuditEventType.DELETE: AuditCategory.DATA_MODIFICATION,
            AuditEventType.READ: AuditCategory.DATA_ACCESS,
            AuditEventType.EXPORT: AuditCategory.DATA_ACCESS,
            AuditEventType.IMPORT: AuditCategory.DATA_MODIFICATION,
            AuditEventType.CONFIG_CHANGE: AuditCategory.SYSTEM,
            AuditEventType.SYSTEM_START: AuditCategory.SYSTEM,
            AuditEventType.SYSTEM_STOP: AuditCategory.SYSTEM,
            AuditEventType.BACKUP: AuditCategory.SYSTEM,
            AuditEventType.RESTORE: AuditCategory.SYSTEM,
            AuditEventType.VALIDATION_START: AuditCategory.VALIDATION,
            AuditEventType.VALIDATION_COMPLETE: AuditCategory.VALIDATION,
            AuditEventType.VALIDATION_FAILED: AuditCategory.VALIDATION,
            AuditEventType.CHECKPOINT_RUN: AuditCategory.VALIDATION,
            AuditEventType.SECURITY_ALERT: AuditCategory.SECURITY,
            AuditEventType.RATE_LIMIT_EXCEEDED: AuditCategory.SECURITY,
            AuditEventType.SUSPICIOUS_ACTIVITY: AuditCategory.SECURITY,
        }
        return mapping.get(event_type, AuditCategory.CUSTOM)


# =============================================================================
# Logger Registry
# =============================================================================


class AuditLoggerRegistry:
    """Registry for managing multiple audit loggers.

    Singleton pattern for global access.

    Example:
        >>> registry = AuditLoggerRegistry()
        >>> registry.register("api", api_logger)
        >>> registry.register("security", security_logger)
        >>> logger = registry.get("api")
    """

    _instance: "AuditLoggerRegistry | None" = None
    _lock = threading.Lock()

    def __new__(cls) -> "AuditLoggerRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._loggers = {}
                    cls._instance._default = None
        return cls._instance

    def register(
        self,
        name: str,
        logger: AuditLogger,
        *,
        set_default: bool = False,
    ) -> None:
        """Register a logger."""
        self._loggers[name] = logger
        if set_default:
            self._default = name

    def get(self, name: str | None = None) -> AuditLogger | None:
        """Get a logger by name."""
        if name is None:
            name = self._default
        if name is None:
            return None
        return self._loggers.get(name)

    def get_or_create(
        self,
        name: str,
        config: AuditConfig | None = None,
    ) -> AuditLogger:
        """Get or create a logger."""
        if name not in self._loggers:
            self._loggers[name] = AuditLogger(config=config, name=name)
        return self._loggers[name]

    def unregister(self, name: str) -> bool:
        """Unregister a logger."""
        if name in self._loggers:
            del self._loggers[name]
            if self._default == name:
                self._default = None
            return True
        return False

    def list_all(self) -> list[str]:
        """List all registered logger names."""
        return list(self._loggers.keys())

    def close_all(self) -> None:
        """Close all loggers."""
        for logger in self._loggers.values():
            logger.close()
        self._loggers.clear()
        self._default = None

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None


# =============================================================================
# Convenience Functions
# =============================================================================


_default_logger: AuditLogger | None = None
_default_lock = threading.Lock()


def get_audit_logger(name: str | None = None) -> AuditLogger:
    """Get an audit logger.

    Args:
        name: Logger name (uses default if None).

    Returns:
        AuditLogger instance.
    """
    registry = AuditLoggerRegistry()
    logger = registry.get(name)

    if logger is None:
        global _default_logger
        with _default_lock:
            if _default_logger is None:
                _default_logger = AuditLogger(name="global_default")
            return _default_logger

    return logger


def configure_audit(
    name: str = "default",
    *,
    config: AuditConfig | None = None,
    storage: AuditStorage | None = None,
    set_default: bool = True,
    **kwargs: Any,
) -> AuditLogger:
    """Configure and register an audit logger.

    Args:
        name: Logger name.
        config: Audit configuration.
        storage: Storage backend.
        set_default: Set as default logger.
        **kwargs: Additional config options.

    Returns:
        Configured AuditLogger.
    """
    if config is None:
        config = AuditConfig(**kwargs)

    logger = AuditLogger(config=config, storage=storage, name=name)

    registry = AuditLoggerRegistry()
    registry.register(name, logger, set_default=set_default)

    return logger


# =============================================================================
# Context Managers
# =============================================================================


@contextmanager
def audit_context(
    actor: AuditActor | None = None,
    request_id: str = "",
    context: AuditContext | None = None,
) -> Iterator[None]:
    """Context manager for setting audit context.

    Example:
        >>> with audit_context(actor=current_user, request_id="req-123"):
        ...     # All audit logs in this block will use this context
        ...     do_something()
    """
    # Save previous state
    prev_actor = _AuditContext.get_actor()
    prev_context = _AuditContext.get_context()
    prev_request_id = _AuditContext.get_request_id()

    # Set new state
    if actor is not None:
        _AuditContext.set_actor(actor)
    if context is not None:
        _AuditContext.set_context(context)
    if request_id:
        _AuditContext.set_request_id(request_id)

    try:
        yield
    finally:
        # Restore previous state
        _AuditContext.set_actor(prev_actor)
        _AuditContext.set_context(prev_context)
        _AuditContext.set_request_id(prev_request_id)


@contextmanager
def audit_operation(
    action: str,
    *,
    logger: AuditLogger | None = None,
    event_type: AuditEventType = AuditEventType.CUSTOM,
    resource: AuditResource | None = None,
    **kwargs: Any,
) -> Iterator[AuditEventBuilder]:
    """Context manager for auditing an operation.

    Automatically logs start/end and captures duration.

    Example:
        >>> with audit_operation("process_data", resource=my_resource) as builder:
        ...     result = process_data()
        ...     builder.set_outcome(AuditOutcome.SUCCESS)
        ...     builder.add_data("records_processed", 100)
    """
    if logger is None:
        logger = get_audit_logger()

    builder = AuditEventBuilder()
    builder.set_type(event_type)
    builder.set_action(action)
    if resource:
        builder.set_resource_object(resource)

    start_time = time.time()
    outcome = AuditOutcome.SUCCESS

    try:
        yield builder
    except Exception:
        outcome = AuditOutcome.FAILURE
        raise
    finally:
        duration_ms = (time.time() - start_time) * 1000
        builder.set_duration(duration_ms)

        if builder._outcome == AuditOutcome.UNKNOWN:
            builder.set_outcome(outcome)

        event = builder.build()
        logger.log_event(event)


# =============================================================================
# Decorators
# =============================================================================


def audited(
    action: str | None = None,
    *,
    event_type: AuditEventType = AuditEventType.CUSTOM,
    logger: AuditLogger | str | None = None,
    resource_arg: str | None = None,
    include_args: bool = False,
    include_result: bool = False,
) -> Callable[[F], F]:
    """Decorator for auditing function calls.

    Example:
        >>> @audited(action="create_user", event_type=AuditEventType.CREATE)
        ... def create_user(user_data):
        ...     return User.create(user_data)
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Resolve logger
            audit_logger = logger
            if isinstance(audit_logger, str):
                audit_logger = get_audit_logger(audit_logger)
            if audit_logger is None:
                audit_logger = get_audit_logger()

            # Determine action name
            func_action = action or func.__name__

            # Build event data
            data: dict[str, Any] = {}
            if include_args:
                data["args"] = args
                data["kwargs"] = kwargs

            # Extract resource if specified
            resource = None
            if resource_arg and resource_arg in kwargs:
                res_value = kwargs[resource_arg]
                if isinstance(res_value, AuditResource):
                    resource = res_value
                else:
                    resource = AuditResource(id=str(res_value), type="argument")

            start_time = time.time()
            outcome = AuditOutcome.SUCCESS
            result = None
            reason = ""

            try:
                result = func(*args, **kwargs)
                if include_result:
                    data["result"] = result
                return result
            except Exception as e:
                outcome = AuditOutcome.FAILURE
                reason = str(e)
                raise
            finally:
                duration_ms = (time.time() - start_time) * 1000

                audit_logger.log(
                    event_type=event_type,
                    action=func_action,
                    resource=resource,
                    outcome=outcome,
                    reason=reason,
                    data=data,
                    duration_ms=duration_ms,
                )

        return wrapper  # type: ignore

    return decorator


def audited_async(
    action: str | None = None,
    *,
    event_type: AuditEventType = AuditEventType.CUSTOM,
    logger: AuditLogger | str | None = None,
) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
    """Async decorator for auditing async function calls.

    Example:
        >>> @audited_async(action="fetch_data")
        ... async def fetch_data(url):
        ...     return await http_client.get(url)
    """
    def decorator(
        func: Callable[..., Awaitable[Any]],
    ) -> Callable[..., Awaitable[Any]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Resolve logger
            audit_logger = logger
            if isinstance(audit_logger, str):
                audit_logger = get_audit_logger(audit_logger)
            if audit_logger is None:
                audit_logger = get_audit_logger()

            func_action = action or func.__name__

            start_time = time.time()
            outcome = AuditOutcome.SUCCESS
            reason = ""

            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                outcome = AuditOutcome.FAILURE
                reason = str(e)
                raise
            finally:
                duration_ms = (time.time() - start_time) * 1000

                audit_logger.log(
                    event_type=event_type,
                    action=func_action,
                    outcome=outcome,
                    reason=reason,
                    duration_ms=duration_ms,
                )

        return wrapper

    return decorator
