"""Core types, configuration, and exceptions for rate limiting.

This module provides the foundational types and interfaces for the
rate limiting system.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Generic, TypeVar


# =============================================================================
# Enums
# =============================================================================


class RateLimitStrategy(Enum):
    """Rate limiting algorithm strategy."""

    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    LEAKY_BUCKET = "leaky_bucket"


class RateLimitScope(Enum):
    """Scope for rate limit application."""

    GLOBAL = "global"  # Single limit for all requests
    PER_KEY = "per_key"  # Separate limit per key (user, IP, etc.)
    PER_ENDPOINT = "per_endpoint"  # Separate limit per endpoint
    CUSTOM = "custom"  # Custom grouping logic


class RateLimitAction(Enum):
    """Action to take when rate limit is exceeded."""

    REJECT = "reject"  # Immediately reject
    QUEUE = "queue"  # Queue for later processing
    THROTTLE = "throttle"  # Slow down processing
    LOG_ONLY = "log_only"  # Log but allow through


# =============================================================================
# Result Types
# =============================================================================


@dataclass(frozen=True)
class RateLimitResult:
    """Result of a rate limit check.

    Attributes:
        allowed: Whether the request is allowed.
        remaining: Number of remaining requests in the window.
        limit: Total limit for the window.
        reset_at: Unix timestamp when the limit resets.
        retry_after: Seconds to wait before retrying (if not allowed).
        bucket_key: The key used for rate limiting.
        metadata: Additional metadata about the decision.
    """

    allowed: bool
    remaining: int
    limit: int
    reset_at: float
    retry_after: float = 0.0
    bucket_key: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def utilization(self) -> float:
        """Get utilization percentage (0.0 to 1.0)."""
        if self.limit == 0:
            return 0.0
        return 1.0 - (self.remaining / self.limit)

    def to_headers(self) -> dict[str, str]:
        """Convert to HTTP rate limit headers.

        Returns headers following the IETF draft standard:
        - RateLimit-Limit: Total limit
        - RateLimit-Remaining: Remaining requests
        - RateLimit-Reset: Reset timestamp
        - Retry-After: Seconds to wait (only if not allowed)
        """
        headers = {
            "RateLimit-Limit": str(self.limit),
            "RateLimit-Remaining": str(max(0, self.remaining)),
            "RateLimit-Reset": str(int(self.reset_at)),
        }

        if not self.allowed and self.retry_after > 0:
            headers["Retry-After"] = str(int(self.retry_after))

        return headers


@dataclass(frozen=True)
class TokenBucketState:
    """State of a token bucket.

    Used for persistence and distributed rate limiting.
    """

    tokens: float
    last_update: float
    bucket_key: str = ""


@dataclass(frozen=True)
class WindowState:
    """State of a rate limit window.

    Used for sliding/fixed window algorithms.
    """

    count: int
    window_start: float
    bucket_key: str = ""
    sub_windows: dict[int, int] = field(default_factory=dict)  # For sliding window


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting.

    Example:
        >>> config = RateLimitConfig(
        ...     requests_per_second=100,
        ...     burst_size=200,
        ...     strategy=RateLimitStrategy.TOKEN_BUCKET,
        ... )
    """

    # Basic limits
    requests_per_second: float = 10.0
    requests_per_minute: float | None = None
    requests_per_hour: float | None = None
    requests_per_day: float | None = None

    # Burst configuration
    burst_size: int | None = None  # Max burst (defaults to requests_per_second)

    # Algorithm settings
    strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET
    window_size_seconds: float = 60.0  # For window-based algorithms
    sub_window_count: int = 10  # For sliding window precision

    # Behavior settings
    scope: RateLimitScope = RateLimitScope.GLOBAL
    action: RateLimitAction = RateLimitAction.REJECT

    # Retry settings
    max_retry_wait: float = 60.0  # Max seconds to suggest waiting

    # Key extraction
    key_func: Callable[[Any], str] | None = None

    # Warm-up settings (gradual limit increase)
    warm_up_period: float = 0.0  # Seconds to reach full rate
    warm_up_tokens: int = 0  # Starting tokens during warm-up

    def __post_init__(self) -> None:
        """Validate and normalize configuration."""
        if self.burst_size is None:
            self.burst_size = max(1, int(self.requests_per_second))

        # Validate ranges
        if self.requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        if self.burst_size <= 0:
            raise ValueError("burst_size must be positive")
        if self.window_size_seconds <= 0:
            raise ValueError("window_size_seconds must be positive")

    @property
    def refill_rate(self) -> float:
        """Get token refill rate per second."""
        return self.requests_per_second

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RateLimitConfig":
        """Create config from dictionary."""
        # Convert string enums
        if "strategy" in data and isinstance(data["strategy"], str):
            data["strategy"] = RateLimitStrategy(data["strategy"])
        if "scope" in data and isinstance(data["scope"], str):
            data["scope"] = RateLimitScope(data["scope"])
        if "action" in data and isinstance(data["action"], str):
            data["action"] = RateLimitAction(data["action"])

        return cls(**data)


@dataclass
class QuotaConfig:
    """Configuration for quota limits (longer-term limits).

    Quotas are typically applied over longer periods (day, month)
    and are often tied to billing or subscription tiers.

    Example:
        >>> quota = QuotaConfig(
        ...     name="api_calls",
        ...     limit=10000,
        ...     period_seconds=86400,  # Daily
        ...     tier="premium",
        ... )
    """

    name: str
    limit: int
    period_seconds: float
    tier: str = "default"
    soft_limit: int | None = None  # Warning threshold
    hard_limit: int | None = None  # Absolute maximum
    overage_allowed: bool = False  # Allow exceeding with penalty
    overage_rate: float = 0.0  # Cost per overage request

    def __post_init__(self) -> None:
        """Set defaults."""
        if self.hard_limit is None:
            self.hard_limit = self.limit
        if self.soft_limit is None:
            self.soft_limit = int(self.limit * 0.8)


# =============================================================================
# Exceptions
# =============================================================================


class RateLimitError(Exception):
    """Base exception for rate limiting errors."""

    def __init__(
        self,
        message: str,
        result: RateLimitResult | None = None,
    ) -> None:
        super().__init__(message)
        self.result = result


class RateLimitExceeded(RateLimitError):
    """Exception raised when rate limit is exceeded.

    Contains information about the limit and when to retry.
    """

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        result: RateLimitResult | None = None,
        retry_after: float = 0.0,
    ) -> None:
        super().__init__(message, result)
        self._retry_after = retry_after

    @property
    def retry_after(self) -> float:
        """Get seconds to wait before retrying."""
        if self.result:
            return self.result.retry_after
        return self._retry_after


class QuotaExceeded(RateLimitError):
    """Exception raised when quota is exceeded."""

    def __init__(
        self,
        message: str = "Quota exceeded",
        quota_name: str = "",
        result: RateLimitResult | None = None,
    ) -> None:
        super().__init__(message, result)
        self.quota_name = quota_name


class RateLimitConfigError(RateLimitError):
    """Exception for configuration errors."""
    pass


# =============================================================================
# Abstract Interfaces
# =============================================================================


class RateLimitAlgorithm(ABC):
    """Abstract base class for rate limiting algorithms.

    Implementations should be thread-safe.
    """

    @abstractmethod
    def acquire(
        self,
        key: str,
        tokens: int = 1,
        *,
        wait: bool = False,
        timeout: float | None = None,
    ) -> RateLimitResult:
        """Attempt to acquire tokens from the rate limiter.

        Args:
            key: Bucket key for rate limiting.
            tokens: Number of tokens to acquire.
            wait: Whether to wait for tokens if not available.
            timeout: Maximum time to wait (if wait=True).

        Returns:
            RateLimitResult with the decision.
        """
        pass

    @abstractmethod
    def peek(self, key: str) -> RateLimitResult:
        """Check rate limit without consuming tokens.

        Args:
            key: Bucket key to check.

        Returns:
            RateLimitResult with current state.
        """
        pass

    @abstractmethod
    def reset(self, key: str) -> None:
        """Reset the rate limit for a key.

        Args:
            key: Bucket key to reset.
        """
        pass

    @property
    @abstractmethod
    def config(self) -> RateLimitConfig:
        """Get the rate limit configuration."""
        pass


T = TypeVar("T")


class RateLimitStorage(ABC, Generic[T]):
    """Abstract base class for rate limit state storage.

    Implementations provide persistence for distributed rate limiting.
    Type parameter T represents the state type (TokenBucketState, WindowState, etc.)
    """

    @abstractmethod
    def get(self, key: str) -> T | None:
        """Get state for a key.

        Args:
            key: Storage key.

        Returns:
            State or None if not found.
        """
        pass

    @abstractmethod
    def set(self, key: str, state: T, ttl: float | None = None) -> None:
        """Set state for a key.

        Args:
            key: Storage key.
            state: State to store.
            ttl: Time-to-live in seconds.
        """
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete state for a key.

        Args:
            key: Storage key.

        Returns:
            True if deleted, False if not found.
        """
        pass

    @abstractmethod
    def increment(
        self,
        key: str,
        amount: int = 1,
        ttl: float | None = None,
    ) -> int:
        """Atomically increment a counter.

        Used for window-based algorithms.

        Args:
            key: Storage key.
            amount: Amount to increment.
            ttl: Time-to-live in seconds.

        Returns:
            New counter value.
        """
        pass

    @abstractmethod
    def get_with_lock(
        self,
        key: str,
        timeout: float = 1.0,
    ) -> tuple[T | None, Any]:
        """Get state with a lock for atomic operations.

        Args:
            key: Storage key.
            timeout: Lock timeout.

        Returns:
            Tuple of (state, lock_token).
        """
        pass

    @abstractmethod
    def set_with_lock(
        self,
        key: str,
        state: T,
        lock_token: Any,
        ttl: float | None = None,
    ) -> bool:
        """Set state with a lock.

        Args:
            key: Storage key.
            state: State to store.
            lock_token: Lock token from get_with_lock.
            ttl: Time-to-live.

        Returns:
            True if successful, False if lock expired.
        """
        pass


class KeyExtractor(ABC):
    """Abstract base class for extracting rate limit keys.

    Key extractors determine how requests are grouped for rate limiting.
    """

    @abstractmethod
    def extract(self, context: Any) -> str:
        """Extract a rate limit key from the context.

        Args:
            context: Request context (varies by framework).

        Returns:
            Rate limit key string.
        """
        pass


# =============================================================================
# Built-in Key Extractors
# =============================================================================


class GlobalKeyExtractor(KeyExtractor):
    """Extracts a global key (same for all requests)."""

    def __init__(self, key: str = "global") -> None:
        self._key = key

    def extract(self, context: Any) -> str:
        return self._key


class AttributeKeyExtractor(KeyExtractor):
    """Extracts key from an attribute of the context."""

    def __init__(
        self,
        attribute: str,
        prefix: str = "",
        default: str = "unknown",
    ) -> None:
        self._attribute = attribute
        self._prefix = prefix
        self._default = default

    def extract(self, context: Any) -> str:
        if isinstance(context, dict):
            value = context.get(self._attribute, self._default)
        else:
            value = getattr(context, self._attribute, self._default)

        return f"{self._prefix}{value}" if self._prefix else str(value)


class CompositeKeyExtractor(KeyExtractor):
    """Combines multiple extractors into a single key."""

    def __init__(
        self,
        extractors: list[KeyExtractor],
        separator: str = ":",
    ) -> None:
        self._extractors = extractors
        self._separator = separator

    def extract(self, context: Any) -> str:
        parts = [e.extract(context) for e in self._extractors]
        return self._separator.join(parts)


class CallableKeyExtractor(KeyExtractor):
    """Wraps a callable as a key extractor."""

    def __init__(self, func: Callable[[Any], str]) -> None:
        self._func = func

    def extract(self, context: Any) -> str:
        return self._func(context)


# =============================================================================
# Utility Functions
# =============================================================================


def current_time() -> float:
    """Get current time in seconds (monotonic if available)."""
    return time.time()


def calculate_retry_after(
    tokens_needed: float,
    refill_rate: float,
    max_wait: float = 60.0,
) -> float:
    """Calculate time to wait for tokens.

    Args:
        tokens_needed: Number of tokens needed.
        refill_rate: Tokens refilled per second.
        max_wait: Maximum wait time.

    Returns:
        Seconds to wait.
    """
    if refill_rate <= 0:
        return max_wait

    wait_time = tokens_needed / refill_rate
    return min(wait_time, max_wait)
