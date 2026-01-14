"""Throttler Implementations.

This module provides concrete throttler implementations:
- TokenBucketThrottler: Token bucket algorithm with burst support
- SlidingWindowThrottler: Sliding window rate limiting
- FixedWindowThrottler: Fixed window rate limiting
- CompositeThrottler: Combines multiple throttlers for multi-level limits
- NoOpThrottler: Pass-through for testing/disabling
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from truthound.checkpoint.throttling.protocols import (
    BaseThrottler,
    RateLimit,
    ThrottleResult,
    ThrottleStatus,
    ThrottlingKey,
    ThrottlingStats,
)

logger = logging.getLogger(__name__)


@dataclass
class TokenBucketState:
    """State for a token bucket."""

    tokens: float
    last_refill: float
    total_allowed: int = 0
    total_throttled: int = 0


class TokenBucketThrottler(BaseThrottler):
    """Token bucket rate limiter.

    Allows bursts up to bucket capacity, then refills at a constant rate.
    Suitable for scenarios where occasional bursts are acceptable.

    Example:
        >>> throttler = TokenBucketThrottler("api")
        >>> limit = RateLimit.per_minute(10, burst_multiplier=1.5)
        >>> key = ThrottlingKey.for_global(TimeUnit.MINUTE)
        >>>
        >>> result = throttler.acquire(key, limit)
        >>> if result.allowed:
        ...     send_notification()
    """

    def __init__(self, name: str = "token_bucket"):
        """Initialize token bucket throttler."""
        super().__init__(name)
        self._buckets: dict[str, TokenBucketState] = {}
        self._lock = threading.RLock()

    def check(
        self,
        key: ThrottlingKey,
        limit: RateLimit,
        permits: int = 1,
    ) -> ThrottleResult:
        """Check if permits are available without consuming."""
        with self._lock:
            bucket_key = f"{key.key}:{limit.time_unit.value}"
            state = self._get_or_create_bucket(bucket_key, limit)
            self._refill(state, limit)

            if state.tokens >= permits:
                # Don't subtract permits - just checking
                remaining = int(state.tokens)
                is_burst = state.tokens > limit.limit
                return ThrottleResult.allowed_result(
                    key=key,
                    remaining=remaining,
                    limit=limit,
                    is_burst=is_burst,
                )
            else:
                tokens_needed = permits - state.tokens
                wait_time = tokens_needed / limit.tokens_per_second
                return ThrottleResult.throttled_result(
                    key=key,
                    retry_after=wait_time,
                    limit=limit,
                )

    def acquire(
        self,
        key: ThrottlingKey,
        limit: RateLimit,
        permits: int = 1,
    ) -> ThrottleResult:
        """Acquire permits (check and consume)."""
        with self._lock:
            bucket_key = f"{key.key}:{limit.time_unit.value}"
            state = self._get_or_create_bucket(bucket_key, limit)
            self._refill(state, limit)

            if state.tokens >= permits:
                state.tokens -= permits
                state.total_allowed += 1
                remaining = int(state.tokens)
                is_burst = remaining > limit.limit - permits

                result = ThrottleResult.allowed_result(
                    key=key,
                    remaining=remaining,
                    limit=limit,
                    is_burst=is_burst,
                )
            else:
                tokens_needed = permits - state.tokens
                wait_time = tokens_needed / limit.tokens_per_second
                state.total_throttled += 1

                result = ThrottleResult.throttled_result(
                    key=key,
                    retry_after=wait_time,
                    limit=limit,
                )

            self._update_stats(result)
            return result

    def _get_or_create_bucket(
        self,
        bucket_key: str,
        limit: RateLimit,
    ) -> TokenBucketState:
        """Get or create a token bucket."""
        if bucket_key not in self._buckets:
            self._buckets[bucket_key] = TokenBucketState(
                tokens=float(limit.burst_limit),
                last_refill=time.time(),
            )
            self._stats.buckets_active = len(self._buckets)
            self._stats.newest_bucket = datetime.now()
            if self._stats.oldest_bucket is None:
                self._stats.oldest_bucket = datetime.now()

        return self._buckets[bucket_key]

    def _refill(self, state: TokenBucketState, limit: RateLimit) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - state.last_refill
        tokens_to_add = elapsed * limit.tokens_per_second
        state.tokens = min(state.tokens + tokens_to_add, float(limit.burst_limit))
        state.last_refill = now

    def reset(self, key: ThrottlingKey | None = None) -> None:
        """Reset throttler state."""
        with self._lock:
            if key is None:
                self._buckets.clear()
                self._stats = ThrottlingStats()
            else:
                # Remove all buckets for this key across time units
                keys_to_remove = [k for k in self._buckets if k.startswith(key.key)]
                for k in keys_to_remove:
                    del self._buckets[k]
                self._stats.buckets_active = len(self._buckets)

    def get_bucket_state(self, key: ThrottlingKey, limit: RateLimit) -> dict[str, Any]:
        """Get current bucket state (for debugging)."""
        with self._lock:
            bucket_key = f"{key.key}:{limit.time_unit.value}"
            if bucket_key in self._buckets:
                state = self._buckets[bucket_key]
                return {
                    "tokens": state.tokens,
                    "last_refill": state.last_refill,
                    "total_allowed": state.total_allowed,
                    "total_throttled": state.total_throttled,
                }
            return {}


@dataclass
class WindowEntry:
    """Entry in a sliding window."""

    timestamp: float
    permits: int = 1


class SlidingWindowThrottler(BaseThrottler):
    """Sliding window rate limiter.

    Tracks requests in a sliding time window for smooth rate limiting.
    Provides more consistent rate limiting than fixed windows.

    Example:
        >>> throttler = SlidingWindowThrottler("api")
        >>> limit = RateLimit.per_hour(100)
        >>> key = ThrottlingKey.for_action("slack", TimeUnit.HOUR)
        >>>
        >>> result = throttler.acquire(key, limit)
    """

    def __init__(self, name: str = "sliding_window"):
        """Initialize sliding window throttler."""
        super().__init__(name)
        self._windows: dict[str, deque[WindowEntry]] = {}
        self._window_stats: dict[str, dict[str, int]] = {}
        self._lock = threading.RLock()

    def check(
        self,
        key: ThrottlingKey,
        limit: RateLimit,
        permits: int = 1,
    ) -> ThrottleResult:
        """Check if permits are available without consuming."""
        with self._lock:
            window_key = f"{key.key}:{limit.time_unit.value}"
            self._cleanup_window(window_key, limit)

            window = self._windows.get(window_key, deque())
            current_count = sum(e.permits for e in window)

            if current_count + permits <= limit.limit:
                remaining = limit.limit - current_count - permits
                return ThrottleResult.allowed_result(
                    key=key,
                    remaining=remaining,
                    limit=limit,
                )
            else:
                # Calculate wait time until oldest entries expire
                wait_time = self._calculate_wait_time(window, limit, permits)
                return ThrottleResult.throttled_result(
                    key=key,
                    retry_after=wait_time,
                    limit=limit,
                )

    def acquire(
        self,
        key: ThrottlingKey,
        limit: RateLimit,
        permits: int = 1,
    ) -> ThrottleResult:
        """Acquire permits (check and consume)."""
        with self._lock:
            window_key = f"{key.key}:{limit.time_unit.value}"
            self._cleanup_window(window_key, limit)

            if window_key not in self._windows:
                self._windows[window_key] = deque()
                self._window_stats[window_key] = {"allowed": 0, "throttled": 0}
                self._stats.buckets_active = len(self._windows)
                self._stats.newest_bucket = datetime.now()
                if self._stats.oldest_bucket is None:
                    self._stats.oldest_bucket = datetime.now()

            window = self._windows[window_key]
            current_count = sum(e.permits for e in window)

            if current_count + permits <= limit.limit:
                window.append(WindowEntry(timestamp=time.time(), permits=permits))
                remaining = limit.limit - current_count - permits
                self._window_stats[window_key]["allowed"] += 1

                result = ThrottleResult.allowed_result(
                    key=key,
                    remaining=remaining,
                    limit=limit,
                )
            else:
                wait_time = self._calculate_wait_time(window, limit, permits)
                self._window_stats[window_key]["throttled"] += 1

                result = ThrottleResult.throttled_result(
                    key=key,
                    retry_after=wait_time,
                    limit=limit,
                )

            self._update_stats(result)
            return result

    def _cleanup_window(self, window_key: str, limit: RateLimit) -> None:
        """Remove entries outside the current window."""
        if window_key not in self._windows:
            return

        window = self._windows[window_key]
        now = time.time()
        cutoff = now - limit.window_seconds

        while window and window[0].timestamp < cutoff:
            window.popleft()

    def _calculate_wait_time(
        self,
        window: deque[WindowEntry],
        limit: RateLimit,
        permits: int,
    ) -> float:
        """Calculate time to wait until permits are available."""
        if not window:
            return 0.0

        current_count = sum(e.permits for e in window)
        excess = current_count + permits - limit.limit

        if excess <= 0:
            return 0.0

        # Find when enough entries will expire
        now = time.time()
        permits_to_expire = 0
        for entry in window:
            permits_to_expire += entry.permits
            if permits_to_expire >= excess:
                expire_time = entry.timestamp + limit.window_seconds
                return max(0.0, expire_time - now)

        return float(limit.window_seconds)

    def reset(self, key: ThrottlingKey | None = None) -> None:
        """Reset throttler state."""
        with self._lock:
            if key is None:
                self._windows.clear()
                self._window_stats.clear()
                self._stats = ThrottlingStats()
            else:
                keys_to_remove = [k for k in self._windows if k.startswith(key.key)]
                for k in keys_to_remove:
                    del self._windows[k]
                    self._window_stats.pop(k, None)
                self._stats.buckets_active = len(self._windows)


class FixedWindowThrottler(BaseThrottler):
    """Fixed window rate limiter.

    Resets counter at fixed intervals. Simpler but can have edge effects
    where requests at window boundaries can effectively double the rate.

    Example:
        >>> throttler = FixedWindowThrottler("api")
        >>> limit = RateLimit.per_minute(10)
        >>> key = ThrottlingKey.for_global(TimeUnit.MINUTE)
        >>>
        >>> result = throttler.acquire(key, limit)
    """

    def __init__(self, name: str = "fixed_window"):
        """Initialize fixed window throttler."""
        super().__init__(name)
        self._counters: dict[str, tuple[int, float]] = {}  # (count, window_start)
        self._counter_stats: dict[str, dict[str, int]] = {}
        self._lock = threading.RLock()

    def check(
        self,
        key: ThrottlingKey,
        limit: RateLimit,
        permits: int = 1,
    ) -> ThrottleResult:
        """Check if permits are available without consuming."""
        with self._lock:
            counter_key = f"{key.key}:{limit.time_unit.value}"
            count, window_start = self._get_or_create_counter(counter_key, limit)

            if count + permits <= limit.limit:
                remaining = limit.limit - count - permits
                return ThrottleResult.allowed_result(
                    key=key,
                    remaining=remaining,
                    limit=limit,
                )
            else:
                now = time.time()
                window_end = window_start + limit.window_seconds
                wait_time = max(0.0, window_end - now)
                return ThrottleResult.throttled_result(
                    key=key,
                    retry_after=wait_time,
                    limit=limit,
                )

    def acquire(
        self,
        key: ThrottlingKey,
        limit: RateLimit,
        permits: int = 1,
    ) -> ThrottleResult:
        """Acquire permits (check and consume)."""
        with self._lock:
            counter_key = f"{key.key}:{limit.time_unit.value}"
            count, window_start = self._get_or_create_counter(counter_key, limit)

            if counter_key not in self._counter_stats:
                self._counter_stats[counter_key] = {"allowed": 0, "throttled": 0}
                self._stats.buckets_active = len(self._counters)
                self._stats.newest_bucket = datetime.now()
                if self._stats.oldest_bucket is None:
                    self._stats.oldest_bucket = datetime.now()

            if count + permits <= limit.limit:
                self._counters[counter_key] = (count + permits, window_start)
                remaining = limit.limit - count - permits
                self._counter_stats[counter_key]["allowed"] += 1

                result = ThrottleResult.allowed_result(
                    key=key,
                    remaining=remaining,
                    limit=limit,
                )
            else:
                now = time.time()
                window_end = window_start + limit.window_seconds
                wait_time = max(0.0, window_end - now)
                self._counter_stats[counter_key]["throttled"] += 1

                result = ThrottleResult.throttled_result(
                    key=key,
                    retry_after=wait_time,
                    limit=limit,
                )

            self._update_stats(result)
            return result

    def _get_or_create_counter(
        self,
        counter_key: str,
        limit: RateLimit,
    ) -> tuple[int, float]:
        """Get or create a counter, resetting if window has passed."""
        now = time.time()

        if counter_key in self._counters:
            count, window_start = self._counters[counter_key]
            if now - window_start >= limit.window_seconds:
                # Window has passed, reset
                self._counters[counter_key] = (0, now)
                return (0, now)
            return (count, window_start)
        else:
            self._counters[counter_key] = (0, now)
            return (0, now)

    def reset(self, key: ThrottlingKey | None = None) -> None:
        """Reset throttler state."""
        with self._lock:
            if key is None:
                self._counters.clear()
                self._counter_stats.clear()
                self._stats = ThrottlingStats()
            else:
                keys_to_remove = [k for k in self._counters if k.startswith(key.key)]
                for k in keys_to_remove:
                    del self._counters[k]
                    self._counter_stats.pop(k, None)
                self._stats.buckets_active = len(self._counters)


@dataclass
class CompositeResult:
    """Result from composite throttler check."""

    allowed: bool
    results: list[ThrottleResult] = field(default_factory=list)
    blocking_result: ThrottleResult | None = None


class CompositeThrottler(BaseThrottler):
    """Composite throttler combining multiple limits.

    Checks multiple rate limits and only allows if all pass.
    Ideal for multi-level limits (per-minute + per-hour + per-day).

    Example:
        >>> throttler = CompositeThrottler("multi-level")
        >>> throttler.add_limit(RateLimit.per_minute(10))
        >>> throttler.add_limit(RateLimit.per_hour(100))
        >>> throttler.add_limit(RateLimit.per_day(500))
        >>>
        >>> key = ThrottlingKey.for_global(TimeUnit.MINUTE)
        >>> result = throttler.acquire(key)  # Checks all limits
    """

    def __init__(
        self,
        name: str = "composite",
        algorithm: str = "token_bucket",
    ):
        """Initialize composite throttler.

        Args:
            name: Throttler name.
            algorithm: Algorithm to use for each limit.
        """
        super().__init__(name)
        self._algorithm = algorithm
        self._limits: list[RateLimit] = []
        self._throttler = self._create_throttler()

    def _create_throttler(self) -> BaseThrottler:
        """Create the underlying throttler based on algorithm."""
        if self._algorithm == "sliding_window":
            return SlidingWindowThrottler(f"{self._name}_impl")
        elif self._algorithm == "fixed_window":
            return FixedWindowThrottler(f"{self._name}_impl")
        else:
            return TokenBucketThrottler(f"{self._name}_impl")

    def add_limit(self, limit: RateLimit) -> CompositeThrottler:
        """Add a rate limit to check.

        Args:
            limit: Rate limit to add.

        Returns:
            Self for chaining.
        """
        self._limits.append(limit)
        return self

    def with_limits(self, limits: list[RateLimit]) -> CompositeThrottler:
        """Set all rate limits.

        Args:
            limits: Rate limits to set.

        Returns:
            Self for chaining.
        """
        self._limits = limits
        return self

    def check(
        self,
        key: ThrottlingKey,
        limit: RateLimit | None = None,
        permits: int = 1,
    ) -> ThrottleResult:
        """Check if permits are available against all limits.

        Args:
            key: Throttling key.
            limit: Ignored (uses configured limits).
            permits: Number of permits to check.

        Returns:
            ThrottleResult (throttled if any limit fails).
        """
        composite = self._check_all(key, permits, consume=False)

        if composite.blocking_result:
            return composite.blocking_result

        # All passed - return the result with minimum remaining
        min_remaining = min(r.remaining for r in composite.results) if composite.results else 0
        return ThrottleResult.allowed_result(
            key=key,
            remaining=min_remaining,
            limit=None,
        )

    def acquire(
        self,
        key: ThrottlingKey,
        limit: RateLimit | None = None,
        permits: int = 1,
    ) -> ThrottleResult:
        """Acquire permits from all limits.

        Args:
            key: Throttling key.
            limit: Ignored (uses configured limits).
            permits: Number of permits to acquire.

        Returns:
            ThrottleResult (throttled if any limit fails).
        """
        # First check all limits without consuming
        composite = self._check_all(key, permits, consume=False)

        if composite.blocking_result:
            self._update_stats(composite.blocking_result)
            return composite.blocking_result

        # All checks passed, now consume from all
        results: list[ThrottleResult] = []
        for rate_limit in self._limits:
            limit_key = ThrottlingKey(
                scope=key.scope,
                action_type=key.action_type,
                checkpoint_name=key.checkpoint_name,
                severity=key.severity,
                data_asset=key.data_asset,
                custom_key=key.custom_key,
                time_unit=rate_limit.time_unit,
            )
            result = self._throttler.acquire(limit_key, rate_limit, permits)
            results.append(result)

        # Return combined result
        min_remaining = min(r.remaining for r in results) if results else 0
        final_result = ThrottleResult.allowed_result(
            key=key,
            remaining=min_remaining,
            limit=None,
            is_burst=any(r.status == ThrottleStatus.BURST_ALLOWED for r in results),
        )
        final_result.metadata["limits_checked"] = len(results)

        self._update_stats(final_result)
        return final_result

    def _check_all(
        self,
        key: ThrottlingKey,
        permits: int,
        consume: bool,
    ) -> CompositeResult:
        """Check all limits.

        Args:
            key: Throttling key.
            permits: Number of permits.
            consume: Whether to consume permits.

        Returns:
            Composite result.
        """
        results: list[ThrottleResult] = []

        for rate_limit in self._limits:
            # Create key for this specific time unit
            limit_key = ThrottlingKey(
                scope=key.scope,
                action_type=key.action_type,
                checkpoint_name=key.checkpoint_name,
                severity=key.severity,
                data_asset=key.data_asset,
                custom_key=key.custom_key,
                time_unit=rate_limit.time_unit,
            )

            if consume:
                result = self._throttler.acquire(limit_key, rate_limit, permits)
            else:
                result = self._throttler.check(limit_key, rate_limit, permits)

            results.append(result)

            if not result.allowed:
                return CompositeResult(
                    allowed=False,
                    results=results,
                    blocking_result=result,
                )

        return CompositeResult(allowed=True, results=results)

    def reset(self, key: ThrottlingKey | None = None) -> None:
        """Reset throttler state."""
        self._throttler.reset(key)
        if key is None:
            self._stats = ThrottlingStats()

    def get_stats(self) -> ThrottlingStats:
        """Get combined statistics."""
        impl_stats = self._throttler.get_stats()
        self._stats.buckets_active = impl_stats.buckets_active
        self._stats.oldest_bucket = impl_stats.oldest_bucket
        self._stats.newest_bucket = impl_stats.newest_bucket
        return self._stats


class NoOpThrottler(BaseThrottler):
    """No-op throttler that always allows.

    Useful for testing or when throttling should be disabled.

    Example:
        >>> throttler = NoOpThrottler()
        >>> result = throttler.acquire(key, limit)
        >>> assert result.allowed
    """

    def __init__(self, name: str = "noop"):
        """Initialize no-op throttler."""
        super().__init__(name)

    def check(
        self,
        key: ThrottlingKey,
        limit: RateLimit,
        permits: int = 1,
    ) -> ThrottleResult:
        """Always returns allowed."""
        return ThrottleResult.allowed_result(
            key=key,
            remaining=limit.limit,
            limit=limit,
        )

    def acquire(
        self,
        key: ThrottlingKey,
        limit: RateLimit,
        permits: int = 1,
    ) -> ThrottleResult:
        """Always returns allowed."""
        result = ThrottleResult.allowed_result(
            key=key,
            remaining=limit.limit,
            limit=limit,
        )
        self._update_stats(result)
        return result

    def reset(self, key: ThrottlingKey | None = None) -> None:
        """Reset (no-op)."""
        if key is None:
            self._stats = ThrottlingStats()
