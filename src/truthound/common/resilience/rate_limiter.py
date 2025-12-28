"""Rate limiter implementations.

This module provides rate limiting to control the rate of requests
and prevent overload.
"""

from __future__ import annotations

import functools
import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from contextlib import contextmanager
from typing import Any, Callable, Generator, TypeVar

from truthound.common.resilience.config import RateLimiterConfig
from truthound.common.resilience.protocols import RateLimiterProtocol

logger = logging.getLogger(__name__)

R = TypeVar("R")


class RateLimitExceededError(Exception):
    """Raised when rate limit is exceeded."""

    def __init__(self, name: str, wait_time: float):
        self.limiter_name = name
        self.wait_time = wait_time
        super().__init__(
            f"Rate limit exceeded for '{name}'. Retry after {wait_time:.2f}s"
        )


class RateLimiter(RateLimiterProtocol, ABC):
    """Abstract base class for rate limiter implementations."""

    def __init__(self, name: str, config: RateLimiterConfig | None = None):
        """Initialize rate limiter.

        Args:
            name: Unique name for this rate limiter.
            config: Configuration options.
        """
        self._name = name
        self._config = config or RateLimiterConfig()
        self._total_acquired = 0
        self._total_rejected = 0
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        """Get rate limiter name."""
        return self._name

    @property
    def config(self) -> RateLimiterConfig:
        """Get configuration."""
        return self._config

    def get_metrics(self) -> dict[str, Any]:
        """Get rate limiter metrics."""
        with self._lock:
            return {
                "name": self._name,
                "rate": self._config.rate,
                "period_seconds": self._config.period_seconds,
                "total_acquired": self._total_acquired,
                "total_rejected": self._total_rejected,
            }

    @contextmanager
    def limit(self, permits: int = 1) -> Generator[None, None, None]:
        """Context manager for rate-limited execution."""
        if not self.acquire(permits):
            wait_time = self.get_wait_time(permits)
            with self._lock:
                self._total_rejected += 1
            raise RateLimitExceededError(self._name, wait_time)

        with self._lock:
            self._total_acquired += permits

        yield

    def __call__(self, func: Callable[..., R]) -> Callable[..., R]:
        """Decorator for rate-limited execution."""
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> R:
            with self.limit():
                return func(*args, **kwargs)
        return wrapper


class TokenBucketRateLimiter(RateLimiter):
    """Token bucket rate limiter.

    Allows bursts up to bucket capacity, then refills at a constant rate.

    Example:
        limiter = TokenBucketRateLimiter(
            "api",
            RateLimiterConfig(rate=100, period_seconds=1.0, burst_size=150)
        )

        @limiter
        def api_call():
            return requests.get(...)
    """

    def __init__(self, name: str, config: RateLimiterConfig | None = None):
        """Initialize token bucket rate limiter."""
        super().__init__(name, config)
        self._tokens = float(self._config.effective_burst_size)
        self._last_refill = time.time()
        self._refill_rate = self._config.rate / self._config.period_seconds

    def acquire(self, permits: int = 1, timeout: float | None = None) -> bool:
        """Acquire permits, blocking if necessary."""
        with self._lock:
            self._refill()

            if self._tokens >= permits:
                self._tokens -= permits
                return True

            if timeout is None or timeout <= 0:
                return False

            # Calculate wait time
            tokens_needed = permits - self._tokens
            wait_time = tokens_needed / self._refill_rate

            if wait_time > timeout:
                return False

        # Wait and retry
        time.sleep(wait_time)

        with self._lock:
            self._refill()
            if self._tokens >= permits:
                self._tokens -= permits
                return True
            return False

    def try_acquire(self, permits: int = 1) -> bool:
        """Try to acquire permits without blocking."""
        with self._lock:
            self._refill()
            if self._tokens >= permits:
                self._tokens -= permits
                return True
            return False

    def get_wait_time(self, permits: int = 1) -> float:
        """Get time to wait before permits are available."""
        with self._lock:
            self._refill()
            if self._tokens >= permits:
                return 0.0
            tokens_needed = permits - self._tokens
            return tokens_needed / self._refill_rate

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self._last_refill
        tokens_to_add = elapsed * self._refill_rate
        self._tokens = min(
            self._tokens + tokens_to_add,
            float(self._config.effective_burst_size),
        )
        self._last_refill = now

    def reset(self) -> None:
        """Reset rate limiter state."""
        with self._lock:
            self._tokens = float(self._config.effective_burst_size)
            self._last_refill = time.time()
            self._total_acquired = 0
            self._total_rejected = 0


class SlidingWindowRateLimiter(RateLimiter):
    """Sliding window rate limiter.

    Tracks requests in a sliding time window for smooth rate limiting.

    Example:
        limiter = SlidingWindowRateLimiter(
            "api",
            RateLimiterConfig(rate=100, period_seconds=60.0)  # 100 per minute
        )

        @limiter
        def api_call():
            return requests.get(...)
    """

    def __init__(self, name: str, config: RateLimiterConfig | None = None):
        """Initialize sliding window rate limiter."""
        super().__init__(name, config)
        self._timestamps: deque[float] = deque()

    def acquire(self, permits: int = 1, timeout: float | None = None) -> bool:
        """Acquire permits, blocking if necessary."""
        with self._lock:
            self._cleanup_old_entries()

            if len(self._timestamps) + permits <= self._config.rate:
                now = time.time()
                for _ in range(permits):
                    self._timestamps.append(now)
                return True

            if timeout is None or timeout <= 0:
                return False

        # Wait for oldest entries to expire
        wait_time = self.get_wait_time(permits)
        if wait_time > timeout:
            return False

        time.sleep(wait_time)

        with self._lock:
            self._cleanup_old_entries()
            if len(self._timestamps) + permits <= self._config.rate:
                now = time.time()
                for _ in range(permits):
                    self._timestamps.append(now)
                return True
            return False

    def try_acquire(self, permits: int = 1) -> bool:
        """Try to acquire permits without blocking."""
        with self._lock:
            self._cleanup_old_entries()
            if len(self._timestamps) + permits <= self._config.rate:
                now = time.time()
                for _ in range(permits):
                    self._timestamps.append(now)
                return True
            return False

    def get_wait_time(self, permits: int = 1) -> float:
        """Get time to wait before permits are available."""
        with self._lock:
            self._cleanup_old_entries()

            if len(self._timestamps) + permits <= self._config.rate:
                return 0.0

            # How many requests need to expire?
            excess = len(self._timestamps) + permits - self._config.rate

            if excess <= 0:
                return 0.0

            if len(self._timestamps) < excess:
                return self._config.period_seconds

            # Time until oldest required entries expire
            oldest_to_expire = sorted(self._timestamps)[:excess]
            now = time.time()
            expire_times = [
                (ts + self._config.period_seconds) - now
                for ts in oldest_to_expire
            ]
            return max(0.0, max(expire_times))

    def _cleanup_old_entries(self) -> None:
        """Remove entries outside the current window."""
        now = time.time()
        cutoff = now - self._config.period_seconds
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()

    def reset(self) -> None:
        """Reset rate limiter state."""
        with self._lock:
            self._timestamps.clear()
            self._total_acquired = 0
            self._total_rejected = 0


class FixedWindowRateLimiter(RateLimiter):
    """Fixed window rate limiter.

    Resets counter at fixed intervals. Simpler but can have edge effects.

    Example:
        limiter = FixedWindowRateLimiter(
            "api",
            RateLimiterConfig(rate=100, period_seconds=60.0)  # 100 per minute
        )

        @limiter
        def api_call():
            return requests.get(...)
    """

    def __init__(self, name: str, config: RateLimiterConfig | None = None):
        """Initialize fixed window rate limiter."""
        super().__init__(name, config)
        self._count = 0
        self._window_start = time.time()

    def acquire(self, permits: int = 1, timeout: float | None = None) -> bool:
        """Acquire permits, blocking if necessary."""
        with self._lock:
            self._check_reset_window()

            if self._count + permits <= self._config.rate:
                self._count += permits
                return True

            if timeout is None or timeout <= 0:
                return False

        # Wait for window reset
        wait_time = self.get_wait_time(permits)
        if wait_time > timeout:
            return False

        time.sleep(wait_time)

        with self._lock:
            self._check_reset_window()
            if self._count + permits <= self._config.rate:
                self._count += permits
                return True
            return False

    def try_acquire(self, permits: int = 1) -> bool:
        """Try to acquire permits without blocking."""
        with self._lock:
            self._check_reset_window()
            if self._count + permits <= self._config.rate:
                self._count += permits
                return True
            return False

    def get_wait_time(self, permits: int = 1) -> float:
        """Get time to wait before permits are available."""
        with self._lock:
            self._check_reset_window()

            if self._count + permits <= self._config.rate:
                return 0.0

            # Wait until window resets
            now = time.time()
            window_end = self._window_start + self._config.period_seconds
            return max(0.0, window_end - now)

    def _check_reset_window(self) -> None:
        """Reset counter if window has passed."""
        now = time.time()
        if now - self._window_start >= self._config.period_seconds:
            self._count = 0
            self._window_start = now

    def reset(self) -> None:
        """Reset rate limiter state."""
        with self._lock:
            self._count = 0
            self._window_start = time.time()
            self._total_acquired = 0
            self._total_rejected = 0


def with_rate_limit(
    rate: int = 100,
    period_seconds: float = 1.0,
    algorithm: str = "token_bucket",
) -> Callable[[Callable[..., R]], Callable[..., R]]:
    """Convenience decorator factory for rate limiting.

    Example:
        @with_rate_limit(rate=10, period_seconds=1.0)
        def limited_operation():
            return api_call()
    """
    config = RateLimiterConfig(
        rate=rate,
        period_seconds=period_seconds,
        algorithm=algorithm,
    )

    if algorithm == "token_bucket":
        limiter = TokenBucketRateLimiter(f"inline-{id(config)}", config)
    elif algorithm == "sliding_window":
        limiter = SlidingWindowRateLimiter(f"inline-{id(config)}", config)
    else:
        limiter = FixedWindowRateLimiter(f"inline-{id(config)}", config)

    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        return limiter(func)

    return decorator
