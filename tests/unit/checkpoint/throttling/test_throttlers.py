"""Tests for throttler implementations."""

from __future__ import annotations

import time
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from truthound.checkpoint.throttling.protocols import (
    RateLimit,
    ThrottleStatus,
    ThrottlingKey,
    TimeUnit,
)
from truthound.checkpoint.throttling.throttlers import (
    CompositeThrottler,
    FixedWindowThrottler,
    NoOpThrottler,
    SlidingWindowThrottler,
    TokenBucketThrottler,
)


class TestTokenBucketThrottler:
    """Tests for TokenBucketThrottler."""

    def test_basic_acquire(self) -> None:
        """Test basic permit acquisition."""
        throttler = TokenBucketThrottler("test")
        key = ThrottlingKey.for_global(TimeUnit.MINUTE)
        limit = RateLimit.per_minute(10)

        result = throttler.acquire(key, limit)

        assert result.allowed is True
        assert result.status == ThrottleStatus.ALLOWED
        assert result.remaining == 9

    def test_check_does_not_consume(self) -> None:
        """Test check does not consume tokens."""
        throttler = TokenBucketThrottler("test")
        key = ThrottlingKey.for_global(TimeUnit.MINUTE)
        limit = RateLimit.per_minute(10)

        # Check multiple times
        for _ in range(5):
            result = throttler.check(key, limit)
            assert result.allowed is True
            assert result.remaining == 10  # Should stay at 10

    def test_exhaust_tokens(self) -> None:
        """Test exhausting all tokens."""
        throttler = TokenBucketThrottler("test")
        key = ThrottlingKey.for_global(TimeUnit.MINUTE)
        limit = RateLimit.per_minute(5)

        # Acquire all tokens
        for i in range(5):
            result = throttler.acquire(key, limit)
            assert result.allowed is True

        # Next should be throttled
        result = throttler.acquire(key, limit)
        assert result.allowed is False
        assert result.status == ThrottleStatus.THROTTLED
        assert result.retry_after > 0

    def test_burst_capacity(self) -> None:
        """Test burst capacity with multiplier."""
        throttler = TokenBucketThrottler("test")
        key = ThrottlingKey.for_global(TimeUnit.MINUTE)
        limit = RateLimit.per_minute(10, burst_multiplier=1.5)

        # Should allow up to 15 (10 * 1.5)
        for i in range(15):
            result = throttler.acquire(key, limit)
            assert result.allowed is True

        # 16th should be throttled
        result = throttler.acquire(key, limit)
        assert result.allowed is False

    def test_token_refill(self) -> None:
        """Test token refill over time."""
        throttler = TokenBucketThrottler("test")
        key = ThrottlingKey.for_global(TimeUnit.SECOND)
        # 10 tokens per second = 1 token per 0.1 second
        limit = RateLimit(limit=10, time_unit=TimeUnit.SECOND)

        # Exhaust tokens
        for _ in range(10):
            throttler.acquire(key, limit)

        result = throttler.acquire(key, limit)
        assert result.allowed is False

        # Wait for refill (0.2 seconds should give ~2 tokens)
        time.sleep(0.25)

        result = throttler.acquire(key, limit)
        assert result.allowed is True

    def test_reset(self) -> None:
        """Test reset functionality."""
        throttler = TokenBucketThrottler("test")
        key = ThrottlingKey.for_global(TimeUnit.MINUTE)
        limit = RateLimit.per_minute(5)

        # Exhaust tokens
        for _ in range(5):
            throttler.acquire(key, limit)

        result = throttler.acquire(key, limit)
        assert result.allowed is False

        # Reset
        throttler.reset()

        # Should allow again
        result = throttler.acquire(key, limit)
        assert result.allowed is True

    def test_stats_tracking(self) -> None:
        """Test statistics tracking."""
        throttler = TokenBucketThrottler("test")
        key = ThrottlingKey.for_global(TimeUnit.MINUTE)
        limit = RateLimit.per_minute(3)

        # 3 allowed, 2 throttled
        for _ in range(3):
            throttler.acquire(key, limit)
        for _ in range(2):
            throttler.acquire(key, limit)

        stats = throttler.get_stats()
        assert stats.total_checked == 5
        assert stats.total_allowed == 3
        assert stats.total_throttled == 2

    def test_multiple_permits(self) -> None:
        """Test acquiring multiple permits at once."""
        throttler = TokenBucketThrottler("test")
        key = ThrottlingKey.for_global(TimeUnit.MINUTE)
        limit = RateLimit.per_minute(10)

        result = throttler.acquire(key, limit, permits=5)
        assert result.allowed is True
        assert result.remaining == 5

        result = throttler.acquire(key, limit, permits=6)
        assert result.allowed is False

    def test_different_keys(self) -> None:
        """Test different keys have separate buckets."""
        throttler = TokenBucketThrottler("test")
        key1 = ThrottlingKey.for_action("slack", TimeUnit.MINUTE)
        key2 = ThrottlingKey.for_action("email", TimeUnit.MINUTE)
        limit = RateLimit.per_minute(5)

        # Exhaust key1
        for _ in range(5):
            throttler.acquire(key1, limit)

        result = throttler.acquire(key1, limit)
        assert result.allowed is False

        # key2 should still have tokens
        result = throttler.acquire(key2, limit)
        assert result.allowed is True

    def test_thread_safety(self) -> None:
        """Test thread safety of token bucket."""
        throttler = TokenBucketThrottler("test")
        key = ThrottlingKey.for_global(TimeUnit.MINUTE)
        limit = RateLimit.per_minute(100)

        results: list[bool] = []
        lock = threading.Lock()

        def acquire() -> None:
            result = throttler.acquire(key, limit)
            with lock:
                results.append(result.allowed)

        # Run 100 concurrent acquisitions
        with ThreadPoolExecutor(max_workers=10) as executor:
            list(executor.map(lambda _: acquire(), range(100)))

        # All 100 should be allowed (burst = 100)
        assert len(results) == 100
        assert sum(results) == 100  # All True


class TestSlidingWindowThrottler:
    """Tests for SlidingWindowThrottler."""

    def test_basic_acquire(self) -> None:
        """Test basic permit acquisition."""
        throttler = SlidingWindowThrottler("test")
        key = ThrottlingKey.for_global(TimeUnit.MINUTE)
        limit = RateLimit.per_minute(10)

        result = throttler.acquire(key, limit)

        assert result.allowed is True
        assert result.remaining == 9

    def test_exhaust_limit(self) -> None:
        """Test exhausting rate limit."""
        throttler = SlidingWindowThrottler("test")
        key = ThrottlingKey.for_global(TimeUnit.MINUTE)
        limit = RateLimit.per_minute(5)

        for i in range(5):
            result = throttler.acquire(key, limit)
            assert result.allowed is True

        result = throttler.acquire(key, limit)
        assert result.allowed is False
        assert result.status == ThrottleStatus.THROTTLED

    def test_window_expiry(self) -> None:
        """Test entries expire after window."""
        throttler = SlidingWindowThrottler("test")
        key = ThrottlingKey.for_global(TimeUnit.SECOND)
        limit = RateLimit(limit=5, time_unit=TimeUnit.SECOND)

        # Exhaust limit
        for _ in range(5):
            throttler.acquire(key, limit)

        result = throttler.acquire(key, limit)
        assert result.allowed is False

        # Wait for window to expire
        time.sleep(1.1)

        # Should allow again
        result = throttler.acquire(key, limit)
        assert result.allowed is True

    def test_sliding_behavior(self) -> None:
        """Test sliding window behavior."""
        throttler = SlidingWindowThrottler("test")
        key = ThrottlingKey.for_global(TimeUnit.SECOND)
        limit = RateLimit(limit=2, time_unit=TimeUnit.SECOND)

        # Acquire 2
        throttler.acquire(key, limit)
        throttler.acquire(key, limit)

        result = throttler.acquire(key, limit)
        assert result.allowed is False

        # Wait half window
        time.sleep(0.6)

        # Acquire 2 more (first 2 still in window)
        result = throttler.acquire(key, limit)
        assert result.allowed is False

        # Wait more for first to expire
        time.sleep(0.6)

        # Now should allow
        result = throttler.acquire(key, limit)
        assert result.allowed is True


class TestFixedWindowThrottler:
    """Tests for FixedWindowThrottler."""

    def test_basic_acquire(self) -> None:
        """Test basic permit acquisition."""
        throttler = FixedWindowThrottler("test")
        key = ThrottlingKey.for_global(TimeUnit.MINUTE)
        limit = RateLimit.per_minute(10)

        result = throttler.acquire(key, limit)

        assert result.allowed is True
        assert result.remaining == 9

    def test_exhaust_limit(self) -> None:
        """Test exhausting rate limit."""
        throttler = FixedWindowThrottler("test")
        key = ThrottlingKey.for_global(TimeUnit.MINUTE)
        limit = RateLimit.per_minute(5)

        for i in range(5):
            result = throttler.acquire(key, limit)
            assert result.allowed is True

        result = throttler.acquire(key, limit)
        assert result.allowed is False

    def test_window_reset(self) -> None:
        """Test counter resets after window."""
        throttler = FixedWindowThrottler("test")
        key = ThrottlingKey.for_global(TimeUnit.SECOND)
        limit = RateLimit(limit=5, time_unit=TimeUnit.SECOND)

        # Exhaust limit
        for _ in range(5):
            throttler.acquire(key, limit)

        result = throttler.acquire(key, limit)
        assert result.allowed is False

        # Wait for window reset
        time.sleep(1.1)

        # Counter should reset
        result = throttler.acquire(key, limit)
        assert result.allowed is True
        assert result.remaining == 4  # Full window available


class TestCompositeThrottler:
    """Tests for CompositeThrottler."""

    def test_single_limit(self) -> None:
        """Test with single limit."""
        throttler = CompositeThrottler("test")
        throttler.add_limit(RateLimit.per_minute(10))

        key = ThrottlingKey.for_global(TimeUnit.MINUTE)
        result = throttler.acquire(key)

        assert result.allowed is True

    def test_multiple_limits(self) -> None:
        """Test with multiple limits."""
        throttler = CompositeThrottler("test")
        throttler.add_limit(RateLimit.per_minute(5))
        throttler.add_limit(RateLimit.per_hour(20))

        key = ThrottlingKey.for_global(TimeUnit.MINUTE)

        # Should check all limits
        for i in range(5):
            result = throttler.acquire(key)
            assert result.allowed is True

        # Per-minute limit exhausted
        result = throttler.acquire(key)
        assert result.allowed is False

    def test_limits_chaining(self) -> None:
        """Test fluent limit addition."""
        throttler = (
            CompositeThrottler("test")
            .add_limit(RateLimit.per_minute(10))
            .add_limit(RateLimit.per_hour(100))
            .add_limit(RateLimit.per_day(500))
        )

        assert len(throttler._limits) == 3

    def test_with_limits(self) -> None:
        """Test with_limits method."""
        limits = [
            RateLimit.per_minute(5),
            RateLimit.per_hour(50),
        ]
        throttler = CompositeThrottler("test").with_limits(limits)

        assert len(throttler._limits) == 2

    def test_all_limits_checked(self) -> None:
        """Test all limits are checked."""
        throttler = CompositeThrottler("test")
        # Per-minute: 10, Per-hour: 5 (stricter)
        throttler.add_limit(RateLimit.per_minute(10))
        throttler.add_limit(RateLimit.per_hour(5))

        key = ThrottlingKey.for_global(TimeUnit.MINUTE)

        # Per-hour limit should block at 5
        for i in range(5):
            result = throttler.acquire(key)
            assert result.allowed is True

        # Should be blocked by per-hour limit
        result = throttler.acquire(key)
        assert result.allowed is False

    def test_algorithm_selection(self) -> None:
        """Test different algorithms."""
        for algo in ["token_bucket", "sliding_window", "fixed_window"]:
            throttler = CompositeThrottler("test", algorithm=algo)
            throttler.add_limit(RateLimit.per_minute(10))

            key = ThrottlingKey.for_global(TimeUnit.MINUTE)
            result = throttler.acquire(key)
            assert result.allowed is True


class TestNoOpThrottler:
    """Tests for NoOpThrottler."""

    def test_always_allows(self) -> None:
        """Test NoOp always allows."""
        throttler = NoOpThrottler()
        key = ThrottlingKey.for_global(TimeUnit.MINUTE)
        limit = RateLimit.per_minute(1)

        # Should always allow
        for _ in range(1000):
            result = throttler.acquire(key, limit)
            assert result.allowed is True
            assert result.remaining == 1  # Always reports full limit

    def test_check_always_allows(self) -> None:
        """Test check always allows."""
        throttler = NoOpThrottler()
        key = ThrottlingKey.for_global(TimeUnit.MINUTE)
        limit = RateLimit.per_minute(5)

        for _ in range(100):
            result = throttler.check(key, limit)
            assert result.allowed is True

    def test_stats_tracking(self) -> None:
        """Test NoOp still tracks stats."""
        throttler = NoOpThrottler()
        key = ThrottlingKey.for_global(TimeUnit.MINUTE)
        limit = RateLimit.per_minute(10)

        for _ in range(10):
            throttler.acquire(key, limit)

        stats = throttler.get_stats()
        assert stats.total_checked == 10
        assert stats.total_allowed == 10
        assert stats.total_throttled == 0
