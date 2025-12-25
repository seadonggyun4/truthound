"""Main rate limiter implementation.

This module provides the primary RateLimiter class that combines
algorithms, storage, and policies into a unified interface.
"""

from __future__ import annotations

import functools
import threading
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, TypeVar

from truthound.ratelimit.core import (
    RateLimitAlgorithm,
    RateLimitConfig,
    RateLimitResult,
    RateLimitStorage,
    RateLimitExceeded,
    RateLimitStrategy,
    RateLimitScope,
    RateLimitAction,
    KeyExtractor,
    GlobalKeyExtractor,
    CallableKeyExtractor,
    current_time,
)
from truthound.ratelimit.algorithms import create_algorithm
from truthound.ratelimit.policy import (
    RateLimitPolicy,
    PolicyRegistry,
    PolicyMatch,
    QuotaManager,
)


F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# Rate Limiter
# =============================================================================


class RateLimiter:
    """Main rate limiter class.

    Provides a unified interface for rate limiting with support for
    multiple algorithms, storage backends, and policies.

    Example:
        >>> limiter = RateLimiter(
        ...     config=RateLimitConfig(requests_per_second=10),
        ... )
        >>> result = limiter.acquire("user:123")
        >>> if result.allowed:
        ...     process_request()
        ... else:
        ...     raise RateLimitExceeded(retry_after=result.retry_after)
    """

    def __init__(
        self,
        config: RateLimitConfig | None = None,
        *,
        algorithm: RateLimitAlgorithm | None = None,
        storage: RateLimitStorage | None = None,
        key_extractor: KeyExtractor | None = None,
        policy_registry: PolicyRegistry | None = None,
        quota_manager: QuotaManager | None = None,
        name: str = "default",
    ) -> None:
        """Initialize rate limiter.

        Args:
            config: Rate limit configuration.
            algorithm: Pre-configured algorithm (overrides config).
            storage: Storage backend for state.
            key_extractor: Key extraction strategy.
            policy_registry: Policy registry for dynamic configuration.
            quota_manager: Quota manager for long-term limits.
            name: Limiter name for identification.
        """
        self._name = name
        self._config = config or RateLimitConfig()
        self._storage = storage
        self._policy_registry = policy_registry
        self._quota_manager = quota_manager

        # Set up key extractor
        if key_extractor:
            self._key_extractor = key_extractor
        elif self._config.key_func:
            self._key_extractor = CallableKeyExtractor(self._config.key_func)
        else:
            self._key_extractor = GlobalKeyExtractor()

        # Set up algorithm
        if algorithm:
            self._algorithm = algorithm
        else:
            self._algorithm = create_algorithm(self._config, self._storage)

        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        """Get limiter name."""
        return self._name

    @property
    def config(self) -> RateLimitConfig:
        """Get current configuration."""
        return self._config

    def acquire(
        self,
        key: str | None = None,
        *,
        tokens: int = 1,
        wait: bool = False,
        timeout: float | None = None,
        context: Any = None,
    ) -> RateLimitResult:
        """Acquire tokens from the rate limiter.

        Args:
            key: Explicit key (or extracted from context).
            tokens: Number of tokens to acquire.
            wait: Wait for tokens if not available.
            timeout: Maximum wait time.
            context: Request context for key extraction and policy lookup.

        Returns:
            RateLimitResult with the decision.
        """
        # Resolve key
        if key is None:
            if context is not None:
                key = self._key_extractor.extract(context)
            else:
                key = self._key_extractor.extract({})

        # Check if we have a policy registry for dynamic config
        if self._policy_registry and context is not None:
            return self._acquire_with_policy(key, tokens, wait, timeout, context)

        return self._algorithm.acquire(key, tokens, wait=wait, timeout=timeout)

    def _acquire_with_policy(
        self,
        key: str,
        tokens: int,
        wait: bool,
        timeout: float | None,
        context: Any,
    ) -> RateLimitResult:
        """Acquire with dynamic policy resolution."""
        assert self._policy_registry is not None

        match = self._policy_registry.resolve(context)
        if not match.matched or match.config is None:
            # No matching policy, use default algorithm
            return self._algorithm.acquire(key, tokens, wait=wait, timeout=timeout)

        # Create algorithm for this policy's config
        algorithm = create_algorithm(match.config, self._storage)
        result = algorithm.acquire(key, tokens, wait=wait, timeout=timeout)

        # Add policy metadata to result
        return RateLimitResult(
            allowed=result.allowed,
            remaining=result.remaining,
            limit=result.limit,
            reset_at=result.reset_at,
            retry_after=result.retry_after,
            bucket_key=result.bucket_key,
            metadata={**result.metadata, "policy": match.policy_name, **match.metadata},
        )

    def peek(
        self,
        key: str | None = None,
        *,
        context: Any = None,
    ) -> RateLimitResult:
        """Check rate limit without consuming tokens.

        Args:
            key: Explicit key.
            context: Request context.

        Returns:
            RateLimitResult with current state.
        """
        if key is None:
            if context is not None:
                key = self._key_extractor.extract(context)
            else:
                key = self._key_extractor.extract({})

        return self._algorithm.peek(key)

    def reset(self, key: str) -> None:
        """Reset rate limit for a key.

        Args:
            key: Key to reset.
        """
        self._algorithm.reset(key)

    def check_quota(
        self,
        key: str,
        quota_name: str,
        amount: int = 1,
    ) -> RateLimitResult:
        """Check quota without consuming.

        Args:
            key: User/entity key.
            quota_name: Quota name.
            amount: Amount to check.

        Returns:
            RateLimitResult.
        """
        if self._quota_manager is None:
            return RateLimitResult(
                allowed=True,
                remaining=0,
                limit=0,
                reset_at=current_time(),
            )

        return self._quota_manager.check(key, quota_name, amount)

    def consume_quota(
        self,
        key: str,
        quota_name: str,
        amount: int = 1,
    ) -> None:
        """Consume quota.

        Args:
            key: User/entity key.
            quota_name: Quota name.
            amount: Amount to consume.

        Raises:
            QuotaExceeded: If quota exceeded.
        """
        if self._quota_manager:
            self._quota_manager.consume(key, quota_name, amount)


# =============================================================================
# Rate Limiter Registry
# =============================================================================


class RateLimiterRegistry:
    """Registry for managing multiple rate limiters.

    Allows centralized management and lookup of rate limiters.

    Example:
        >>> registry = RateLimiterRegistry()
        >>> registry.register("api", RateLimiter(config=api_config))
        >>> registry.register("search", RateLimiter(config=search_config))
        >>> limiter = registry.get("api")
    """

    _instance: "RateLimiterRegistry | None" = None
    _lock = threading.Lock()

    def __new__(cls) -> "RateLimiterRegistry":
        """Singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._limiters = {}
                    cls._instance._default = None
        return cls._instance

    def register(
        self,
        name: str,
        limiter: RateLimiter,
        *,
        set_default: bool = False,
    ) -> None:
        """Register a rate limiter.

        Args:
            name: Limiter name.
            limiter: Rate limiter instance.
            set_default: Set as default limiter.
        """
        self._limiters[name] = limiter
        if set_default:
            self._default = name

    def get(self, name: str | None = None) -> RateLimiter | None:
        """Get a rate limiter by name.

        Args:
            name: Limiter name (uses default if None).

        Returns:
            RateLimiter or None.
        """
        if name is None:
            name = self._default

        if name is None:
            return None

        return self._limiters.get(name)

    def get_or_create(
        self,
        name: str,
        config: RateLimitConfig | None = None,
    ) -> RateLimiter:
        """Get or create a rate limiter.

        Args:
            name: Limiter name.
            config: Configuration for new limiter.

        Returns:
            RateLimiter.
        """
        if name not in self._limiters:
            self._limiters[name] = RateLimiter(
                config=config or RateLimitConfig(),
                name=name,
            )
        return self._limiters[name]

    def unregister(self, name: str) -> bool:
        """Unregister a rate limiter.

        Args:
            name: Limiter name.

        Returns:
            True if removed.
        """
        if name in self._limiters:
            del self._limiters[name]
            if self._default == name:
                self._default = None
            return True
        return False

    def list_all(self) -> list[str]:
        """List all registered limiter names."""
        return list(self._limiters.keys())

    def clear(self) -> None:
        """Clear all registered limiters."""
        self._limiters.clear()
        self._default = None

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        cls._instance = None


# =============================================================================
# Convenience Functions
# =============================================================================


_default_limiter: RateLimiter | None = None
_default_lock = threading.Lock()


def get_limiter(name: str | None = None) -> RateLimiter:
    """Get a rate limiter from the registry.

    Args:
        name: Limiter name.

    Returns:
        RateLimiter.

    Raises:
        ValueError: If limiter not found.
    """
    registry = RateLimiterRegistry()
    limiter = registry.get(name)

    if limiter is None:
        global _default_limiter
        with _default_lock:
            if _default_limiter is None:
                _default_limiter = RateLimiter(name="global_default")
            return _default_limiter

    return limiter


def configure_rate_limit(
    name: str = "default",
    *,
    requests_per_second: float = 10.0,
    burst_size: int | None = None,
    strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET,
    set_default: bool = True,
    **kwargs: Any,
) -> RateLimiter:
    """Configure and register a rate limiter.

    Args:
        name: Limiter name.
        requests_per_second: Rate limit.
        burst_size: Burst capacity.
        strategy: Algorithm strategy.
        set_default: Set as default.
        **kwargs: Additional config options.

    Returns:
        Configured RateLimiter.
    """
    config = RateLimitConfig(
        requests_per_second=requests_per_second,
        burst_size=burst_size,
        strategy=strategy,
        **kwargs,
    )

    limiter = RateLimiter(config=config, name=name)

    registry = RateLimiterRegistry()
    registry.register(name, limiter, set_default=set_default)

    return limiter


# =============================================================================
# Decorator
# =============================================================================


def rate_limit(
    key: str | Callable[..., str] | None = None,
    *,
    limiter: str | RateLimiter | None = None,
    tokens: int = 1,
    on_exceeded: RateLimitAction = RateLimitAction.REJECT,
    wait: bool = False,
    timeout: float | None = None,
) -> Callable[[F], F]:
    """Decorator for rate limiting functions.

    Args:
        key: Rate limit key or function to extract key.
        limiter: Limiter name or instance.
        tokens: Tokens to consume.
        on_exceeded: Action when limit exceeded.
        wait: Wait for tokens.
        timeout: Wait timeout.

    Returns:
        Decorated function.

    Example:
        >>> @rate_limit(key=lambda user_id: f"user:{user_id}")
        ... def api_call(user_id: str):
        ...     return process(user_id)
        >>>
        >>> @rate_limit(limiter="search", tokens=5)
        ... def search(query: str):
        ...     return search_engine(query)
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Resolve limiter
            if isinstance(limiter, RateLimiter):
                rl = limiter
            else:
                rl = get_limiter(limiter)

            # Resolve key
            if callable(key):
                rate_key = key(*args, **kwargs)
            elif key is not None:
                rate_key = key
            else:
                rate_key = func.__name__

            # Acquire
            result = rl.acquire(
                rate_key,
                tokens=tokens,
                wait=wait,
                timeout=timeout,
            )

            if not result.allowed:
                if on_exceeded == RateLimitAction.REJECT:
                    raise RateLimitExceeded(
                        f"Rate limit exceeded for {rate_key}",
                        result=result,
                    )
                elif on_exceeded == RateLimitAction.LOG_ONLY:
                    # Just log and continue
                    pass
                # QUEUE and THROTTLE would need async handling

            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def rate_limit_async(
    key: str | Callable[..., str] | None = None,
    *,
    limiter: str | RateLimiter | None = None,
    tokens: int = 1,
    on_exceeded: RateLimitAction = RateLimitAction.REJECT,
) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
    """Async decorator for rate limiting.

    Args:
        key: Rate limit key or function.
        limiter: Limiter name or instance.
        tokens: Tokens to consume.
        on_exceeded: Action when exceeded.

    Returns:
        Decorated async function.

    Example:
        >>> @rate_limit_async(key="api")
        ... async def async_api():
        ...     return await fetch_data()
    """
    def decorator(
        func: Callable[..., Awaitable[Any]],
    ) -> Callable[..., Awaitable[Any]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Resolve limiter
            if isinstance(limiter, RateLimiter):
                rl = limiter
            else:
                rl = get_limiter(limiter)

            # Resolve key
            if callable(key):
                rate_key = key(*args, **kwargs)
            elif key is not None:
                rate_key = key
            else:
                rate_key = func.__name__

            # Acquire (sync for now, could be async with async storage)
            result = rl.acquire(rate_key, tokens=tokens)

            if not result.allowed:
                if on_exceeded == RateLimitAction.REJECT:
                    raise RateLimitExceeded(
                        f"Rate limit exceeded for {rate_key}",
                        result=result,
                    )

            return await func(*args, **kwargs)

        return wrapper

    return decorator


# =============================================================================
# Context Manager
# =============================================================================


class RateLimitContext:
    """Context manager for rate limiting.

    Example:
        >>> with RateLimitContext("user:123", limiter=my_limiter) as result:
        ...     if result.allowed:
        ...         do_work()
    """

    def __init__(
        self,
        key: str,
        *,
        limiter: RateLimiter | None = None,
        tokens: int = 1,
        raise_on_exceeded: bool = True,
    ) -> None:
        self._key = key
        self._limiter = limiter or get_limiter()
        self._tokens = tokens
        self._raise_on_exceeded = raise_on_exceeded
        self._result: RateLimitResult | None = None

    def __enter__(self) -> RateLimitResult:
        self._result = self._limiter.acquire(self._key, tokens=self._tokens)

        if not self._result.allowed and self._raise_on_exceeded:
            raise RateLimitExceeded(
                f"Rate limit exceeded for {self._key}",
                result=self._result,
            )

        return self._result

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass


class AsyncRateLimitContext:
    """Async context manager for rate limiting.

    Example:
        >>> async with AsyncRateLimitContext("user:123") as result:
        ...     if result.allowed:
        ...         await do_work()
    """

    def __init__(
        self,
        key: str,
        *,
        limiter: RateLimiter | None = None,
        tokens: int = 1,
        raise_on_exceeded: bool = True,
    ) -> None:
        self._key = key
        self._limiter = limiter or get_limiter()
        self._tokens = tokens
        self._raise_on_exceeded = raise_on_exceeded
        self._result: RateLimitResult | None = None

    async def __aenter__(self) -> RateLimitResult:
        # For now, sync acquire (async storage would make this fully async)
        self._result = self._limiter.acquire(self._key, tokens=self._tokens)

        if not self._result.allowed and self._raise_on_exceeded:
            raise RateLimitExceeded(
                f"Rate limit exceeded for {self._key}",
                result=self._result,
            )

        return self._result

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass
