"""Rate limiting algorithms implementation.

This module provides various rate limiting algorithms:
- Token Bucket: Smooth rate limiting with burst support
- Sliding Window: Accurate rate limiting with window precision
- Fixed Window: Simple time-windowed counting
- Leaky Bucket: Constant rate output with queue
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from truthound.ratelimit.core import (
    RateLimitAlgorithm,
    RateLimitConfig,
    RateLimitResult,
    RateLimitStorage,
    TokenBucketState,
    WindowState,
    current_time,
    calculate_retry_after,
)


# =============================================================================
# Token Bucket Algorithm
# =============================================================================


class TokenBucketAlgorithm(RateLimitAlgorithm):
    """Token Bucket rate limiting algorithm.

    Tokens are added to the bucket at a constant rate (refill_rate).
    Each request consumes tokens from the bucket.
    Requests are allowed if there are enough tokens.

    Advantages:
        - Smooth rate limiting
        - Supports bursts up to bucket capacity
        - Simple to understand and implement

    Example:
        >>> config = RateLimitConfig(requests_per_second=10, burst_size=20)
        >>> bucket = TokenBucketAlgorithm(config)
        >>> result = bucket.acquire("user:123")
        >>> print(result.allowed, result.remaining)
    """

    def __init__(
        self,
        config: RateLimitConfig,
        storage: RateLimitStorage[TokenBucketState] | None = None,
    ) -> None:
        """Initialize token bucket.

        Args:
            config: Rate limit configuration.
            storage: Optional external storage for distributed rate limiting.
        """
        self._config = config
        self._storage = storage
        self._local_buckets: dict[str, _LocalTokenBucket] = {}
        self._lock = threading.Lock()

    @property
    def config(self) -> RateLimitConfig:
        return self._config

    def acquire(
        self,
        key: str,
        tokens: int = 1,
        *,
        wait: bool = False,
        timeout: float | None = None,
    ) -> RateLimitResult:
        """Acquire tokens from the bucket."""
        if self._storage:
            return self._acquire_distributed(key, tokens, wait, timeout)
        return self._acquire_local(key, tokens, wait, timeout)

    def _acquire_local(
        self,
        key: str,
        tokens: int,
        wait: bool,
        timeout: float | None,
    ) -> RateLimitResult:
        """Acquire tokens from local bucket."""
        with self._lock:
            bucket = self._get_or_create_bucket(key)

        return bucket.acquire(tokens, wait=wait, timeout=timeout)

    def _acquire_distributed(
        self,
        key: str,
        tokens: int,
        wait: bool,
        timeout: float | None,
    ) -> RateLimitResult:
        """Acquire tokens from distributed storage."""
        assert self._storage is not None

        storage_key = f"tb:{key}"
        state, lock_token = self._storage.get_with_lock(storage_key)

        now = current_time()
        capacity = self._config.burst_size or int(self._config.requests_per_second)
        refill_rate = self._config.refill_rate

        if state is None:
            # Initialize new bucket
            current_tokens = float(capacity)
            last_update = now
        else:
            # Refill tokens
            elapsed = now - state.last_update
            current_tokens = min(
                capacity,
                state.tokens + elapsed * refill_rate,
            )
            last_update = now

        # Check if we can acquire
        if current_tokens >= tokens:
            new_tokens = current_tokens - tokens
            new_state = TokenBucketState(
                tokens=new_tokens,
                last_update=last_update,
                bucket_key=key,
            )

            ttl = capacity / refill_rate * 2  # Keep state for 2x fill time
            self._storage.set_with_lock(storage_key, new_state, lock_token, ttl)

            return RateLimitResult(
                allowed=True,
                remaining=int(new_tokens),
                limit=capacity,
                reset_at=now + (capacity - new_tokens) / refill_rate,
                bucket_key=key,
            )

        # Not enough tokens
        retry_after = calculate_retry_after(
            tokens - current_tokens,
            refill_rate,
            self._config.max_retry_wait,
        )

        if wait and (timeout is None or retry_after <= timeout):
            time.sleep(retry_after)
            return self._acquire_distributed(key, tokens, False, None)

        return RateLimitResult(
            allowed=False,
            remaining=int(current_tokens),
            limit=capacity,
            reset_at=now + (capacity - current_tokens) / refill_rate,
            retry_after=retry_after,
            bucket_key=key,
        )

    def peek(self, key: str) -> RateLimitResult:
        """Check bucket state without consuming tokens."""
        if self._storage:
            return self._peek_distributed(key)
        return self._peek_local(key)

    def _peek_local(self, key: str) -> RateLimitResult:
        """Peek at local bucket."""
        with self._lock:
            bucket = self._local_buckets.get(key)
            if bucket is None:
                capacity = self._config.burst_size or int(self._config.requests_per_second)
                return RateLimitResult(
                    allowed=True,
                    remaining=capacity,
                    limit=capacity,
                    reset_at=current_time(),
                    bucket_key=key,
                )
            return bucket.peek()

    def _peek_distributed(self, key: str) -> RateLimitResult:
        """Peek at distributed bucket."""
        assert self._storage is not None

        storage_key = f"tb:{key}"
        state = self._storage.get(storage_key)

        now = current_time()
        capacity = self._config.burst_size or int(self._config.requests_per_second)
        refill_rate = self._config.refill_rate

        if state is None:
            return RateLimitResult(
                allowed=True,
                remaining=capacity,
                limit=capacity,
                reset_at=now,
                bucket_key=key,
            )

        # Calculate current tokens
        elapsed = now - state.last_update
        current_tokens = min(capacity, state.tokens + elapsed * refill_rate)

        return RateLimitResult(
            allowed=current_tokens >= 1,
            remaining=int(current_tokens),
            limit=capacity,
            reset_at=now + (capacity - current_tokens) / refill_rate,
            bucket_key=key,
        )

    def reset(self, key: str) -> None:
        """Reset bucket to full capacity."""
        if self._storage:
            self._storage.delete(f"tb:{key}")
        else:
            with self._lock:
                if key in self._local_buckets:
                    del self._local_buckets[key]

    def _get_or_create_bucket(self, key: str) -> "_LocalTokenBucket":
        """Get or create a local bucket."""
        if key not in self._local_buckets:
            self._local_buckets[key] = _LocalTokenBucket(
                capacity=self._config.burst_size or int(self._config.requests_per_second),
                refill_rate=self._config.refill_rate,
                key=key,
                max_retry_wait=self._config.max_retry_wait,
            )
        return self._local_buckets[key]


@dataclass
class _LocalTokenBucket:
    """Thread-safe local token bucket implementation."""

    capacity: int
    refill_rate: float
    key: str
    max_retry_wait: float = 60.0
    tokens: float = field(init=False)
    last_update: float = field(init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        self.tokens = float(self.capacity)
        self.last_update = current_time()

    def acquire(
        self,
        tokens: int = 1,
        *,
        wait: bool = False,
        timeout: float | None = None,
    ) -> RateLimitResult:
        """Attempt to acquire tokens."""
        with self._lock:
            self._refill()

            if self.tokens >= tokens:
                self.tokens -= tokens
                return RateLimitResult(
                    allowed=True,
                    remaining=int(self.tokens),
                    limit=self.capacity,
                    reset_at=self._calculate_reset_time(),
                    bucket_key=self.key,
                )

            # Calculate retry time
            retry_after = calculate_retry_after(
                tokens - self.tokens,
                self.refill_rate,
                self.max_retry_wait,
            )

        # Wait outside lock if requested
        if wait and (timeout is None or retry_after <= timeout):
            time.sleep(retry_after)
            return self.acquire(tokens, wait=False)

        return RateLimitResult(
            allowed=False,
            remaining=int(self.tokens),
            limit=self.capacity,
            reset_at=self._calculate_reset_time(),
            retry_after=retry_after,
            bucket_key=self.key,
        )

    def peek(self) -> RateLimitResult:
        """Check state without consuming."""
        with self._lock:
            self._refill()
            return RateLimitResult(
                allowed=self.tokens >= 1,
                remaining=int(self.tokens),
                limit=self.capacity,
                reset_at=self._calculate_reset_time(),
                bucket_key=self.key,
            )

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = current_time()
        elapsed = now - self.last_update
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_update = now

    def _calculate_reset_time(self) -> float:
        """Calculate when bucket will be full."""
        tokens_needed = self.capacity - self.tokens
        if tokens_needed <= 0:
            return self.last_update
        return self.last_update + tokens_needed / self.refill_rate


# =============================================================================
# Sliding Window Algorithm
# =============================================================================


class SlidingWindowAlgorithm(RateLimitAlgorithm):
    """Sliding Window rate limiting algorithm.

    Divides time into sub-windows and counts requests in each.
    The current count is calculated by summing sub-windows within
    the sliding window.

    Advantages:
        - More accurate than fixed window
        - Prevents burst at window boundaries
        - Smooth rate limiting

    Example:
        >>> config = RateLimitConfig(
        ...     requests_per_second=10,
        ...     window_size_seconds=60,
        ...     sub_window_count=10,
        ... )
        >>> limiter = SlidingWindowAlgorithm(config)
    """

    def __init__(
        self,
        config: RateLimitConfig,
        storage: RateLimitStorage[WindowState] | None = None,
    ) -> None:
        self._config = config
        self._storage = storage
        self._local_windows: dict[str, _LocalSlidingWindow] = {}
        self._lock = threading.Lock()

    @property
    def config(self) -> RateLimitConfig:
        return self._config

    @property
    def limit(self) -> int:
        """Get the request limit for the window."""
        return int(self._config.requests_per_second * self._config.window_size_seconds)

    def acquire(
        self,
        key: str,
        tokens: int = 1,
        *,
        wait: bool = False,
        timeout: float | None = None,
    ) -> RateLimitResult:
        """Acquire tokens using sliding window."""
        if self._storage:
            return self._acquire_distributed(key, tokens, wait, timeout)
        return self._acquire_local(key, tokens, wait, timeout)

    def _acquire_local(
        self,
        key: str,
        tokens: int,
        wait: bool,
        timeout: float | None,
    ) -> RateLimitResult:
        """Acquire using local window."""
        with self._lock:
            window = self._get_or_create_window(key)

        return window.acquire(tokens, wait=wait, timeout=timeout)

    def _acquire_distributed(
        self,
        key: str,
        tokens: int,
        wait: bool,
        timeout: float | None,
    ) -> RateLimitResult:
        """Acquire using distributed storage."""
        assert self._storage is not None

        now = current_time()
        window_size = self._config.window_size_seconds
        sub_window_size = window_size / self._config.sub_window_count
        current_sub_window = int(now / sub_window_size)
        limit = self.limit

        # Get current count from sub-windows
        total_count = 0
        oldest_window = current_sub_window - self._config.sub_window_count + 1

        for sub_idx in range(oldest_window, current_sub_window + 1):
            sub_key = f"sw:{key}:{sub_idx}"
            state = self._storage.get(sub_key)
            if state and isinstance(state, WindowState):
                total_count += state.count

        # Check limit
        if total_count + tokens <= limit:
            # Increment current sub-window
            sub_key = f"sw:{key}:{current_sub_window}"
            self._storage.increment(sub_key, tokens, ttl=window_size * 2)

            window_end = (current_sub_window + 1) * sub_window_size

            return RateLimitResult(
                allowed=True,
                remaining=limit - total_count - tokens,
                limit=limit,
                reset_at=window_end,
                bucket_key=key,
            )

        # Calculate retry time
        retry_after = min(sub_window_size, self._config.max_retry_wait)

        if wait and (timeout is None or retry_after <= timeout):
            time.sleep(retry_after)
            return self._acquire_distributed(key, tokens, False, None)

        window_end = (current_sub_window + 1) * sub_window_size

        return RateLimitResult(
            allowed=False,
            remaining=max(0, limit - total_count),
            limit=limit,
            reset_at=window_end,
            retry_after=retry_after,
            bucket_key=key,
        )

    def peek(self, key: str) -> RateLimitResult:
        """Peek at current window state."""
        if self._storage:
            return self._peek_distributed(key)
        return self._peek_local(key)

    def _peek_local(self, key: str) -> RateLimitResult:
        """Peek at local window."""
        with self._lock:
            window = self._local_windows.get(key)
            if window is None:
                return RateLimitResult(
                    allowed=True,
                    remaining=self.limit,
                    limit=self.limit,
                    reset_at=current_time() + self._config.window_size_seconds,
                    bucket_key=key,
                )
            return window.peek()

    def _peek_distributed(self, key: str) -> RateLimitResult:
        """Peek at distributed window."""
        assert self._storage is not None

        now = current_time()
        window_size = self._config.window_size_seconds
        sub_window_size = window_size / self._config.sub_window_count
        current_sub_window = int(now / sub_window_size)
        limit = self.limit

        total_count = 0
        oldest_window = current_sub_window - self._config.sub_window_count + 1

        for sub_idx in range(oldest_window, current_sub_window + 1):
            sub_key = f"sw:{key}:{sub_idx}"
            state = self._storage.get(sub_key)
            if state and isinstance(state, WindowState):
                total_count += state.count

        window_end = (current_sub_window + 1) * sub_window_size

        return RateLimitResult(
            allowed=total_count < limit,
            remaining=max(0, limit - total_count),
            limit=limit,
            reset_at=window_end,
            bucket_key=key,
        )

    def reset(self, key: str) -> None:
        """Reset window."""
        if self._storage:
            now = current_time()
            sub_window_size = self._config.window_size_seconds / self._config.sub_window_count
            current_sub_window = int(now / sub_window_size)

            for sub_idx in range(
                current_sub_window - self._config.sub_window_count,
                current_sub_window + 1,
            ):
                self._storage.delete(f"sw:{key}:{sub_idx}")
        else:
            with self._lock:
                if key in self._local_windows:
                    del self._local_windows[key]

    def _get_or_create_window(self, key: str) -> "_LocalSlidingWindow":
        """Get or create local window."""
        if key not in self._local_windows:
            self._local_windows[key] = _LocalSlidingWindow(
                limit=self.limit,
                window_size=self._config.window_size_seconds,
                sub_window_count=self._config.sub_window_count,
                key=key,
                max_retry_wait=self._config.max_retry_wait,
            )
        return self._local_windows[key]


@dataclass
class _LocalSlidingWindow:
    """Thread-safe local sliding window implementation."""

    limit: int
    window_size: float
    sub_window_count: int
    key: str
    max_retry_wait: float = 60.0
    _sub_windows: dict[int, int] = field(default_factory=dict, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    @property
    def sub_window_size(self) -> float:
        return self.window_size / self.sub_window_count

    def acquire(
        self,
        tokens: int = 1,
        *,
        wait: bool = False,
        timeout: float | None = None,
    ) -> RateLimitResult:
        """Attempt to acquire tokens."""
        with self._lock:
            self._cleanup()
            count = self._current_count()

            if count + tokens <= self.limit:
                now = current_time()
                current_sub = int(now / self.sub_window_size)
                self._sub_windows[current_sub] = self._sub_windows.get(current_sub, 0) + tokens

                return RateLimitResult(
                    allowed=True,
                    remaining=self.limit - count - tokens,
                    limit=self.limit,
                    reset_at=self._next_reset_time(),
                    bucket_key=self.key,
                )

            retry_after = min(self.sub_window_size, self.max_retry_wait)

        if wait and (timeout is None or retry_after <= timeout):
            time.sleep(retry_after)
            return self.acquire(tokens, wait=False)

        return RateLimitResult(
            allowed=False,
            remaining=max(0, self.limit - count),
            limit=self.limit,
            reset_at=self._next_reset_time(),
            retry_after=retry_after,
            bucket_key=self.key,
        )

    def peek(self) -> RateLimitResult:
        """Check state without consuming."""
        with self._lock:
            self._cleanup()
            count = self._current_count()

            return RateLimitResult(
                allowed=count < self.limit,
                remaining=max(0, self.limit - count),
                limit=self.limit,
                reset_at=self._next_reset_time(),
                bucket_key=self.key,
            )

    def _current_count(self) -> int:
        """Get current count across sub-windows."""
        return sum(self._sub_windows.values())

    def _cleanup(self) -> None:
        """Remove expired sub-windows."""
        now = current_time()
        current_sub = int(now / self.sub_window_size)
        oldest_valid = current_sub - self.sub_window_count + 1

        expired = [k for k in self._sub_windows if k < oldest_valid]
        for k in expired:
            del self._sub_windows[k]

    def _next_reset_time(self) -> float:
        """Get next sub-window boundary."""
        now = current_time()
        current_sub = int(now / self.sub_window_size)
        return (current_sub + 1) * self.sub_window_size


# =============================================================================
# Fixed Window Algorithm
# =============================================================================


class FixedWindowAlgorithm(RateLimitAlgorithm):
    """Fixed Window rate limiting algorithm.

    Counts requests in fixed time windows. Simple but can allow
    burst at window boundaries (2x limit).

    Advantages:
        - Very simple and memory efficient
        - Easy to understand

    Disadvantages:
        - Can allow 2x burst at window boundaries

    Example:
        >>> config = RateLimitConfig(
        ...     requests_per_second=10,
        ...     window_size_seconds=60,
        ... )
        >>> limiter = FixedWindowAlgorithm(config)
    """

    def __init__(
        self,
        config: RateLimitConfig,
        storage: RateLimitStorage[WindowState] | None = None,
    ) -> None:
        self._config = config
        self._storage = storage
        self._local_windows: dict[str, _LocalFixedWindow] = {}
        self._lock = threading.Lock()

    @property
    def config(self) -> RateLimitConfig:
        return self._config

    @property
    def limit(self) -> int:
        """Get the request limit for the window."""
        return int(self._config.requests_per_second * self._config.window_size_seconds)

    def acquire(
        self,
        key: str,
        tokens: int = 1,
        *,
        wait: bool = False,
        timeout: float | None = None,
    ) -> RateLimitResult:
        """Acquire tokens using fixed window."""
        if self._storage:
            return self._acquire_distributed(key, tokens, wait, timeout)
        return self._acquire_local(key, tokens, wait, timeout)

    def _acquire_local(
        self,
        key: str,
        tokens: int,
        wait: bool,
        timeout: float | None,
    ) -> RateLimitResult:
        """Acquire using local window."""
        with self._lock:
            window = self._get_or_create_window(key)

        return window.acquire(tokens, wait=wait, timeout=timeout)

    def _acquire_distributed(
        self,
        key: str,
        tokens: int,
        wait: bool,
        timeout: float | None,
    ) -> RateLimitResult:
        """Acquire using distributed storage."""
        assert self._storage is not None

        now = current_time()
        window_size = self._config.window_size_seconds
        window_start = int(now / window_size) * window_size
        window_end = window_start + window_size
        limit = self.limit

        storage_key = f"fw:{key}:{int(window_start)}"
        state = self._storage.get(storage_key)
        current_count = state.count if state else 0

        if current_count + tokens <= limit:
            new_count = self._storage.increment(
                storage_key,
                tokens,
                ttl=window_size * 2,
            )

            return RateLimitResult(
                allowed=True,
                remaining=limit - new_count,
                limit=limit,
                reset_at=window_end,
                bucket_key=key,
            )

        retry_after = min(window_end - now, self._config.max_retry_wait)

        if wait and (timeout is None or retry_after <= timeout):
            time.sleep(retry_after)
            return self._acquire_distributed(key, tokens, False, None)

        return RateLimitResult(
            allowed=False,
            remaining=max(0, limit - current_count),
            limit=limit,
            reset_at=window_end,
            retry_after=retry_after,
            bucket_key=key,
        )

    def peek(self, key: str) -> RateLimitResult:
        """Peek at current window state."""
        if self._storage:
            return self._peek_distributed(key)
        return self._peek_local(key)

    def _peek_local(self, key: str) -> RateLimitResult:
        """Peek at local window."""
        with self._lock:
            window = self._local_windows.get(key)
            if window is None:
                now = current_time()
                window_end = (int(now / self._config.window_size_seconds) + 1) * self._config.window_size_seconds
                return RateLimitResult(
                    allowed=True,
                    remaining=self.limit,
                    limit=self.limit,
                    reset_at=window_end,
                    bucket_key=key,
                )
            return window.peek()

    def _peek_distributed(self, key: str) -> RateLimitResult:
        """Peek at distributed window."""
        assert self._storage is not None

        now = current_time()
        window_size = self._config.window_size_seconds
        window_start = int(now / window_size) * window_size
        window_end = window_start + window_size
        limit = self.limit

        storage_key = f"fw:{key}:{int(window_start)}"
        state = self._storage.get(storage_key)
        current_count = state.count if state else 0

        return RateLimitResult(
            allowed=current_count < limit,
            remaining=max(0, limit - current_count),
            limit=limit,
            reset_at=window_end,
            bucket_key=key,
        )

    def reset(self, key: str) -> None:
        """Reset window."""
        if self._storage:
            now = current_time()
            window_start = int(now / self._config.window_size_seconds) * self._config.window_size_seconds
            self._storage.delete(f"fw:{key}:{int(window_start)}")
        else:
            with self._lock:
                if key in self._local_windows:
                    del self._local_windows[key]

    def _get_or_create_window(self, key: str) -> "_LocalFixedWindow":
        """Get or create local window."""
        if key not in self._local_windows:
            self._local_windows[key] = _LocalFixedWindow(
                limit=self.limit,
                window_size=self._config.window_size_seconds,
                key=key,
                max_retry_wait=self._config.max_retry_wait,
            )
        return self._local_windows[key]


@dataclass
class _LocalFixedWindow:
    """Thread-safe local fixed window implementation."""

    limit: int
    window_size: float
    key: str
    max_retry_wait: float = 60.0
    count: int = field(default=0, init=False)
    window_start: float = field(default=0.0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        now = current_time()
        self.window_start = int(now / self.window_size) * self.window_size

    def acquire(
        self,
        tokens: int = 1,
        *,
        wait: bool = False,
        timeout: float | None = None,
    ) -> RateLimitResult:
        """Attempt to acquire tokens."""
        with self._lock:
            self._maybe_reset_window()

            if self.count + tokens <= self.limit:
                self.count += tokens
                return RateLimitResult(
                    allowed=True,
                    remaining=self.limit - self.count,
                    limit=self.limit,
                    reset_at=self.window_start + self.window_size,
                    bucket_key=self.key,
                )

            window_end = self.window_start + self.window_size
            now = current_time()
            retry_after = min(window_end - now, self.max_retry_wait)

        if wait and (timeout is None or retry_after <= timeout):
            time.sleep(retry_after)
            return self.acquire(tokens, wait=False)

        return RateLimitResult(
            allowed=False,
            remaining=max(0, self.limit - self.count),
            limit=self.limit,
            reset_at=window_end,
            retry_after=retry_after,
            bucket_key=self.key,
        )

    def peek(self) -> RateLimitResult:
        """Check state without consuming."""
        with self._lock:
            self._maybe_reset_window()
            return RateLimitResult(
                allowed=self.count < self.limit,
                remaining=max(0, self.limit - self.count),
                limit=self.limit,
                reset_at=self.window_start + self.window_size,
                bucket_key=self.key,
            )

    def _maybe_reset_window(self) -> None:
        """Reset window if expired."""
        now = current_time()
        current_window_start = int(now / self.window_size) * self.window_size

        if current_window_start > self.window_start:
            self.window_start = current_window_start
            self.count = 0


# =============================================================================
# Leaky Bucket Algorithm
# =============================================================================


class LeakyBucketAlgorithm(RateLimitAlgorithm):
    """Leaky Bucket rate limiting algorithm.

    Requests are added to a queue (bucket) and processed at a
    constant rate (leak rate). The bucket has a maximum capacity.

    Advantages:
        - Outputs at constant rate
        - Smooths out bursts

    Disadvantages:
        - Adds latency (requests wait in queue)
        - More complex to implement

    Example:
        >>> config = RateLimitConfig(
        ...     requests_per_second=10,
        ...     burst_size=50,  # Queue capacity
        ... )
        >>> limiter = LeakyBucketAlgorithm(config)
    """

    def __init__(
        self,
        config: RateLimitConfig,
        storage: RateLimitStorage[Any] | None = None,
    ) -> None:
        self._config = config
        self._storage = storage
        self._local_buckets: dict[str, _LocalLeakyBucket] = {}
        self._lock = threading.Lock()

    @property
    def config(self) -> RateLimitConfig:
        return self._config

    def acquire(
        self,
        key: str,
        tokens: int = 1,
        *,
        wait: bool = False,
        timeout: float | None = None,
    ) -> RateLimitResult:
        """Acquire slot in leaky bucket."""
        # For distributed, fall back to local (leaky bucket is complex to distribute)
        return self._acquire_local(key, tokens, wait, timeout)

    def _acquire_local(
        self,
        key: str,
        tokens: int,
        wait: bool,
        timeout: float | None,
    ) -> RateLimitResult:
        """Acquire using local bucket."""
        with self._lock:
            bucket = self._get_or_create_bucket(key)

        return bucket.acquire(tokens, wait=wait, timeout=timeout)

    def peek(self, key: str) -> RateLimitResult:
        """Peek at bucket state."""
        with self._lock:
            bucket = self._local_buckets.get(key)
            if bucket is None:
                capacity = self._config.burst_size or int(self._config.requests_per_second)
                return RateLimitResult(
                    allowed=True,
                    remaining=capacity,
                    limit=capacity,
                    reset_at=current_time(),
                    bucket_key=key,
                )
            return bucket.peek()

    def reset(self, key: str) -> None:
        """Reset bucket."""
        with self._lock:
            if key in self._local_buckets:
                del self._local_buckets[key]

    def _get_or_create_bucket(self, key: str) -> "_LocalLeakyBucket":
        """Get or create local bucket."""
        if key not in self._local_buckets:
            self._local_buckets[key] = _LocalLeakyBucket(
                capacity=self._config.burst_size or int(self._config.requests_per_second),
                leak_rate=self._config.requests_per_second,
                key=key,
                max_retry_wait=self._config.max_retry_wait,
            )
        return self._local_buckets[key]


@dataclass
class _LocalLeakyBucket:
    """Thread-safe local leaky bucket implementation."""

    capacity: int
    leak_rate: float
    key: str
    max_retry_wait: float = 60.0
    _queue: deque[float] = field(default_factory=deque, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def acquire(
        self,
        tokens: int = 1,
        *,
        wait: bool = False,
        timeout: float | None = None,
    ) -> RateLimitResult:
        """Attempt to add to bucket."""
        with self._lock:
            self._leak()

            if len(self._queue) + tokens <= self.capacity:
                # Add tokens to queue
                now = current_time()
                for _ in range(tokens):
                    self._queue.append(now)

                return RateLimitResult(
                    allowed=True,
                    remaining=self.capacity - len(self._queue),
                    limit=self.capacity,
                    reset_at=self._next_leak_time(),
                    bucket_key=self.key,
                )

            # Calculate retry time
            retry_after = min(1.0 / self.leak_rate, self.max_retry_wait)

        if wait and (timeout is None or retry_after <= timeout):
            time.sleep(retry_after)
            return self.acquire(tokens, wait=False)

        return RateLimitResult(
            allowed=False,
            remaining=max(0, self.capacity - len(self._queue)),
            limit=self.capacity,
            reset_at=self._next_leak_time(),
            retry_after=retry_after,
            bucket_key=self.key,
        )

    def peek(self) -> RateLimitResult:
        """Check state without consuming."""
        with self._lock:
            self._leak()
            return RateLimitResult(
                allowed=len(self._queue) < self.capacity,
                remaining=max(0, self.capacity - len(self._queue)),
                limit=self.capacity,
                reset_at=self._next_leak_time(),
                bucket_key=self.key,
            )

    def _leak(self) -> None:
        """Remove leaked items from queue."""
        now = current_time()
        leak_interval = 1.0 / self.leak_rate

        while self._queue:
            oldest = self._queue[0]
            if now - oldest >= leak_interval:
                self._queue.popleft()
            else:
                break

    def _next_leak_time(self) -> float:
        """Get time of next leak."""
        if not self._queue:
            return current_time()

        oldest = self._queue[0]
        leak_interval = 1.0 / self.leak_rate
        return oldest + leak_interval


# =============================================================================
# Algorithm Factory
# =============================================================================


def create_algorithm(
    config: RateLimitConfig,
    storage: RateLimitStorage | None = None,
) -> RateLimitAlgorithm:
    """Create rate limiting algorithm from config.

    Args:
        config: Rate limit configuration.
        storage: Optional storage backend.

    Returns:
        Rate limiting algorithm instance.
    """
    from truthound.ratelimit.core import RateLimitStrategy

    if config.strategy == RateLimitStrategy.TOKEN_BUCKET:
        return TokenBucketAlgorithm(config, storage)
    elif config.strategy == RateLimitStrategy.SLIDING_WINDOW:
        return SlidingWindowAlgorithm(config, storage)
    elif config.strategy == RateLimitStrategy.FIXED_WINDOW:
        return FixedWindowAlgorithm(config, storage)
    elif config.strategy == RateLimitStrategy.LEAKY_BUCKET:
        return LeakyBucketAlgorithm(config, storage)
    else:
        raise ValueError(f"Unknown strategy: {config.strategy}")
