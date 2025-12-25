"""Integration utilities for rate limiting with other Truthound components.

This module provides integration with:
- Checkpoint module for rate-limited validations
- Observability module for metrics and tracing
"""

from __future__ import annotations

import functools
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

from truthound.ratelimit.core import (
    RateLimitConfig,
    RateLimitResult,
    RateLimitExceeded,
    RateLimitAction,
    current_time,
)
from truthound.ratelimit.limiter import RateLimiter, get_limiter
from truthound.ratelimit.middleware import RateLimitMetrics


F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# Checkpoint Integration
# =============================================================================


@dataclass
class RateLimitedCheckpointConfig:
    """Configuration for rate-limited checkpoint execution.

    Controls how rate limiting is applied to checkpoint operations.
    """

    # Rate limiting configuration
    limiter: RateLimiter | None = None
    limiter_name: str = ""

    # Key generation
    key_prefix: str = "checkpoint:"
    include_checkpoint_name: bool = True

    # Behavior
    action_on_exceeded: RateLimitAction = RateLimitAction.REJECT
    wait_for_token: bool = False
    wait_timeout: float | None = None

    # Metrics
    track_metrics: bool = True


class RateLimitedCheckpointMixin:
    """Mixin for adding rate limiting to checkpoints.

    Use this mixin with checkpoint classes to add rate limiting capability.

    Example:
        >>> class MyCheckpoint(RateLimitedCheckpointMixin, Checkpoint):
        ...     pass
        >>>
        >>> checkpoint = MyCheckpoint(
        ...     name="my_checkpoint",
        ...     rate_limit_config=RateLimitedCheckpointConfig(
        ...         limiter=RateLimiter(RateLimitConfig(requests_per_second=1)),
        ...     ),
        ... )
    """

    _rate_limit_config: RateLimitedCheckpointConfig | None = None
    _rate_limit_metrics: RateLimitMetrics | None = None

    def configure_rate_limit(
        self,
        config: RateLimitedCheckpointConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Configure rate limiting for this checkpoint.

        Args:
            config: Rate limit configuration.
            **kwargs: Override config options.
        """
        if config is None:
            config = RateLimitedCheckpointConfig(**kwargs)
        else:
            # Apply overrides
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        self._rate_limit_config = config

        if config.track_metrics:
            self._rate_limit_metrics = RateLimitMetrics()

    def _get_rate_limit_key(self) -> str:
        """Get rate limit key for this checkpoint."""
        config = self._rate_limit_config
        if config is None:
            return "checkpoint:unknown"

        parts = [config.key_prefix.rstrip(":")]

        if config.include_checkpoint_name and hasattr(self, "name"):
            parts.append(getattr(self, "name"))

        return ":".join(parts)

    def _check_rate_limit(self) -> RateLimitResult:
        """Check rate limit before execution.

        Returns:
            Rate limit result.

        Raises:
            RateLimitExceeded: If limit exceeded and action is REJECT.
        """
        config = self._rate_limit_config
        if config is None:
            return RateLimitResult(
                allowed=True,
                remaining=0,
                limit=0,
                reset_at=current_time(),
            )

        # Get limiter
        limiter = config.limiter
        if limiter is None and config.limiter_name:
            limiter = get_limiter(config.limiter_name)
        if limiter is None:
            limiter = get_limiter()

        # Acquire token
        key = self._get_rate_limit_key()
        result = limiter.acquire(
            key,
            wait=config.wait_for_token,
            timeout=config.wait_timeout,
        )

        # Track metrics
        if self._rate_limit_metrics:
            self._rate_limit_metrics.record(result)

        # Handle result
        if not result.allowed:
            if config.action_on_exceeded == RateLimitAction.REJECT:
                raise RateLimitExceeded(
                    f"Rate limit exceeded for checkpoint: {key}",
                    result=result,
                )

        return result

    def get_rate_limit_metrics(self) -> RateLimitMetrics | None:
        """Get rate limit metrics for this checkpoint."""
        return self._rate_limit_metrics


def rate_limited_checkpoint(
    config: RateLimitedCheckpointConfig | None = None,
    **kwargs: Any,
) -> Callable[[type], type]:
    """Class decorator to add rate limiting to a checkpoint class.

    Example:
        >>> @rate_limited_checkpoint(
        ...     limiter=RateLimiter(RateLimitConfig(requests_per_second=1))
        ... )
        ... class MyCheckpoint(Checkpoint):
        ...     pass
    """
    if config is None:
        config = RateLimitedCheckpointConfig(**kwargs)

    def decorator(cls: type) -> type:
        # Store original run method
        original_run = cls.run if hasattr(cls, "run") else None

        def rate_limited_run(self: Any, *args: Any, **kw: Any) -> Any:
            # Check rate limit
            if hasattr(self, "_check_rate_limit"):
                self._check_rate_limit()

            # Call original run
            if original_run:
                return original_run(self, *args, **kw)
            return None

        # Add mixin methods
        for name in dir(RateLimitedCheckpointMixin):
            if not name.startswith("_") or name in ("_get_rate_limit_key", "_check_rate_limit"):
                setattr(cls, name, getattr(RateLimitedCheckpointMixin, name))

        # Set config
        cls._rate_limit_config = config

        # Replace run method
        if original_run:
            cls.run = rate_limited_run

        return cls

    return decorator


# =============================================================================
# Observability Integration
# =============================================================================


class RateLimitObserver:
    """Observer for rate limit events with observability integration.

    Integrates rate limiting with the observability module for:
    - Structured logging
    - Metrics collection
    - Distributed tracing

    Example:
        >>> observer = RateLimitObserver(
        ...     service_name="my-service",
        ...     enable_tracing=True,
        ... )
        >>> observer.on_acquire("user:123", result)
    """

    def __init__(
        self,
        service_name: str = "truthound",
        *,
        enable_logging: bool = True,
        enable_metrics: bool = True,
        enable_tracing: bool = False,
        log_level: str = "info",
    ) -> None:
        """Initialize observer.

        Args:
            service_name: Service name for observability.
            enable_logging: Enable structured logging.
            enable_metrics: Enable metrics collection.
            enable_tracing: Enable distributed tracing.
            log_level: Default log level.
        """
        self._service_name = service_name
        self._enable_logging = enable_logging
        self._enable_metrics = enable_metrics
        self._enable_tracing = enable_tracing
        self._log_level = log_level

        # Lazy load observability components
        self._logger = None
        self._metrics = None
        self._tracer = None

    def _get_logger(self) -> Any:
        """Get or create logger."""
        if self._logger is None and self._enable_logging:
            try:
                from truthound.observability import get_logger
                self._logger = get_logger(f"{self._service_name}.ratelimit")
            except ImportError:
                pass
        return self._logger

    def _get_metrics(self) -> Any:
        """Get or create metrics collector."""
        if self._metrics is None and self._enable_metrics:
            try:
                from truthound.observability import get_metrics
                self._metrics = get_metrics()
            except ImportError:
                pass
        return self._metrics

    def _get_tracer(self) -> Any:
        """Get or create tracer."""
        if self._tracer is None and self._enable_tracing:
            try:
                from truthound.observability import get_tracer
                self._tracer = get_tracer(f"{self._service_name}.ratelimit")
            except ImportError:
                pass
        return self._tracer

    def on_acquire(
        self,
        key: str,
        result: RateLimitResult,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Handle rate limit acquire event.

        Args:
            key: Rate limit key.
            result: Rate limit result.
            context: Additional context.
        """
        context = context or {}

        # Log
        logger = self._get_logger()
        if logger:
            log_data = {
                "key": key,
                "allowed": result.allowed,
                "remaining": result.remaining,
                "limit": result.limit,
                "utilization": result.utilization,
                **context,
            }

            if result.allowed:
                logger.debug("Rate limit check passed", **log_data)
            else:
                log_data["retry_after"] = result.retry_after
                logger.warning("Rate limit exceeded", **log_data)

        # Metrics
        metrics = self._get_metrics()
        if metrics:
            labels = {"key": key, "allowed": str(result.allowed).lower()}

            # Counter for requests
            try:
                counter = metrics.counter(
                    "ratelimit_requests_total",
                    "Total rate limit requests",
                    labels=list(labels.keys()),
                )
                counter.inc(**labels)
            except Exception:
                pass

            # Gauge for remaining
            try:
                gauge = metrics.gauge(
                    "ratelimit_remaining",
                    "Remaining rate limit tokens",
                    labels=["key"],
                )
                gauge.set(result.remaining, key=key)
            except Exception:
                pass

    def on_exceeded(
        self,
        key: str,
        result: RateLimitResult,
        action: RateLimitAction,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Handle rate limit exceeded event.

        Args:
            key: Rate limit key.
            result: Rate limit result.
            action: Action taken.
            context: Additional context.
        """
        context = context or {}

        logger = self._get_logger()
        if logger:
            logger.warning(
                "Rate limit action taken",
                key=key,
                action=action.value,
                retry_after=result.retry_after,
                **context,
            )

        metrics = self._get_metrics()
        if metrics:
            try:
                counter = metrics.counter(
                    "ratelimit_exceeded_total",
                    "Total rate limit exceeded events",
                    labels=["key", "action"],
                )
                counter.inc(key=key, action=action.value)
            except Exception:
                pass


# =============================================================================
# Instrumented Rate Limiter
# =============================================================================


class InstrumentedRateLimiter(RateLimiter):
    """Rate limiter with built-in observability.

    Automatically integrates with the observability module for
    logging, metrics, and tracing.

    Example:
        >>> limiter = InstrumentedRateLimiter(
        ...     config=RateLimitConfig(requests_per_second=10),
        ...     service_name="my-service",
        ... )
    """

    def __init__(
        self,
        config: RateLimitConfig | None = None,
        *,
        service_name: str = "truthound",
        enable_tracing: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize instrumented rate limiter.

        Args:
            config: Rate limit configuration.
            service_name: Service name for observability.
            enable_tracing: Enable distributed tracing.
            **kwargs: Additional RateLimiter arguments.
        """
        super().__init__(config=config, **kwargs)

        self._observer = RateLimitObserver(
            service_name=service_name,
            enable_tracing=enable_tracing,
        )

    def acquire(
        self,
        key: str | None = None,
        *,
        tokens: int = 1,
        wait: bool = False,
        timeout: float | None = None,
        context: Any = None,
    ) -> RateLimitResult:
        """Acquire with instrumentation."""
        result = super().acquire(
            key,
            tokens=tokens,
            wait=wait,
            timeout=timeout,
            context=context,
        )

        # Observe
        actual_key = result.bucket_key or key or "unknown"
        self._observer.on_acquire(actual_key, result)

        if not result.allowed:
            self._observer.on_exceeded(
                actual_key,
                result,
                self.config.action,
            )

        return result


# =============================================================================
# Action Decorator for Rate Limiting
# =============================================================================


def rate_limit_action(
    limiter: RateLimiter | str | None = None,
    *,
    key: str | Callable[..., str] | None = None,
    tokens: int = 1,
    on_exceeded: RateLimitAction = RateLimitAction.REJECT,
) -> Callable[[type], type]:
    """Decorator for rate limiting checkpoint actions.

    Apply to action classes to limit their execution rate.

    Example:
        >>> @rate_limit_action(
        ...     key=lambda self, ctx: f"action:{ctx.checkpoint_name}",
        ...     limiter="default",
        ... )
        ... class MyAction(BaseAction):
        ...     def execute(self, context):
        ...         # Rate limited execution
        ...         pass
    """
    def decorator(cls: type) -> type:
        original_execute = cls.execute if hasattr(cls, "execute") else None

        def rate_limited_execute(self: Any, *args: Any, **kwargs: Any) -> Any:
            # Resolve limiter
            rl = limiter
            if isinstance(rl, str):
                rl = get_limiter(rl)
            if rl is None:
                rl = get_limiter()

            # Resolve key
            if callable(key):
                rate_key = key(self, *args, **kwargs)
            elif key is not None:
                rate_key = key
            else:
                rate_key = f"action:{cls.__name__}"

            # Acquire
            result = rl.acquire(rate_key, tokens=tokens)

            if not result.allowed:
                if on_exceeded == RateLimitAction.REJECT:
                    raise RateLimitExceeded(
                        f"Rate limit exceeded for action: {rate_key}",
                        result=result,
                    )

            # Execute original
            if original_execute:
                return original_execute(self, *args, **kwargs)
            return None

        if original_execute:
            cls.execute = rate_limited_execute

        return cls

    return decorator
