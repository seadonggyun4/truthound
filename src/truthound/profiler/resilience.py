"""Resilience patterns for cache backends.

This module provides fault-tolerant caching with:
- Circuit breaker pattern for failure detection
- Automatic fallback to alternative backends
- Retry logic with exponential backoff
- Health monitoring and auto-recovery
- Comprehensive logging and observability

Key classes:
- CircuitBreaker: Prevents cascade failures
- RetryPolicy: Configurable retry with backoff
- ResilientCacheBackend: Wrapper with resilience patterns
- FallbackChain: Multi-backend fallback chain
- HealthMonitor: Continuous backend health checking

Example:
    from truthound.profiler.resilience import (
        ResilientCacheBackend,
        FallbackChain,
        CircuitBreakerConfig,
    )

    # Create resilient Redis with memory fallback
    cache = ResilientCacheBackend(
        primary=RedisCacheBackend(host="redis.example.com"),
        fallback=MemoryCacheBackend(),
        circuit_breaker=CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=30,
        ),
    )
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Callable, Generic, TypeVar

from truthound.profiler.caching import (
    CacheBackend,
    CacheEntry,
    MemoryCacheBackend,
    FileCacheBackend,
)


# Set up logging
logger = logging.getLogger("truthound.cache.resilience")


# =============================================================================
# Types
# =============================================================================

T = TypeVar("T")


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


class BackendHealth(str, Enum):
    """Backend health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class FailureType(str, Enum):
    """Types of failures for classification."""

    CONNECTION = "connection"
    TIMEOUT = "timeout"
    SERIALIZATION = "serialization"
    UNKNOWN = "unknown"


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker.

    Attributes:
        failure_threshold: Number of failures before opening circuit
        success_threshold: Number of successes to close circuit
        recovery_timeout: Seconds to wait before trying half-open
        failure_window: Window in seconds to count failures
        excluded_exceptions: Exceptions that don't count as failures
    """

    failure_threshold: int = 5
    success_threshold: int = 2
    recovery_timeout: float = 30.0
    failure_window: float = 60.0
    excluded_exceptions: tuple[type[Exception], ...] = ()

    @classmethod
    def aggressive(cls) -> "CircuitBreakerConfig":
        """Aggressive config - opens quickly, recovers slowly."""
        return cls(
            failure_threshold=3,
            success_threshold=3,
            recovery_timeout=60.0,
        )

    @classmethod
    def lenient(cls) -> "CircuitBreakerConfig":
        """Lenient config - tolerates more failures."""
        return cls(
            failure_threshold=10,
            success_threshold=1,
            recovery_timeout=15.0,
        )

    @classmethod
    def disabled(cls) -> "CircuitBreakerConfig":
        """Effectively disabled circuit breaker."""
        return cls(
            failure_threshold=1000000,
            recovery_timeout=0.1,
        )


@dataclass
class RetryConfig:
    """Configuration for retry logic.

    Attributes:
        max_attempts: Maximum retry attempts (1 = no retry)
        base_delay: Base delay in seconds
        max_delay: Maximum delay cap
        exponential_base: Multiplier for exponential backoff
        jitter: Add random jitter to delays
        retryable_exceptions: Exceptions that trigger retry
    """

    max_attempts: int = 3
    base_delay: float = 0.1
    max_delay: float = 10.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple[type[Exception], ...] = (
        ConnectionError,
        TimeoutError,
        OSError,
    )

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        delay = self.base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)

        if self.jitter:
            import random
            delay *= (0.5 + random.random())

        return delay

    @classmethod
    def no_retry(cls) -> "RetryConfig":
        """No retry configuration."""
        return cls(max_attempts=1)

    @classmethod
    def quick(cls) -> "RetryConfig":
        """Quick retry for transient failures."""
        return cls(
            max_attempts=3,
            base_delay=0.05,
            max_delay=1.0,
        )

    @classmethod
    def persistent(cls) -> "RetryConfig":
        """Persistent retry for important operations."""
        return cls(
            max_attempts=5,
            base_delay=0.5,
            max_delay=30.0,
        )


@dataclass
class HealthCheckConfig:
    """Configuration for health monitoring.

    Attributes:
        check_interval: Seconds between health checks
        timeout: Timeout for health check operations
        healthy_threshold: Consecutive successes to mark healthy
        unhealthy_threshold: Consecutive failures to mark unhealthy
    """

    check_interval: float = 30.0
    timeout: float = 5.0
    healthy_threshold: int = 2
    unhealthy_threshold: int = 3
    enabled: bool = True


# =============================================================================
# Circuit Breaker
# =============================================================================


@dataclass
class FailureRecord:
    """Record of a failure event."""

    timestamp: datetime
    exception_type: str
    failure_type: FailureType
    message: str


class CircuitBreaker:
    """Circuit breaker pattern implementation.

    Prevents cascade failures by tracking failure rates and
    temporarily disabling operations when failure threshold is reached.

    Example:
        breaker = CircuitBreaker(CircuitBreakerConfig(failure_threshold=5))

        @breaker.protect
        def risky_operation():
            return redis_client.get("key")
    """

    def __init__(self, config: CircuitBreakerConfig | None = None):
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failures: list[FailureRecord] = []
        self._successes_in_half_open = 0
        self._last_failure_time: datetime | None = None
        self._opened_at: datetime | None = None
        self._lock = threading.RLock()

        # Statistics
        self._total_calls = 0
        self._total_failures = 0
        self._total_rejections = 0
        self._state_changes: list[tuple[datetime, CircuitState, CircuitState]] = []

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            self._check_recovery()
            return self._state

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (rejecting requests)."""
        return self.state == CircuitState.OPEN

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self.state == CircuitState.CLOSED

    def _check_recovery(self) -> None:
        """Check if circuit should transition to half-open."""
        if self._state != CircuitState.OPEN:
            return

        if self._opened_at is None:
            return

        elapsed = (datetime.now() - self._opened_at).total_seconds()
        if elapsed >= self.config.recovery_timeout:
            self._transition_to(CircuitState.HALF_OPEN)

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self._state
        self._state = new_state
        self._state_changes.append((datetime.now(), old_state, new_state))

        logger.info(
            f"Circuit breaker state change: {old_state.value} -> {new_state.value}"
        )

        if new_state == CircuitState.OPEN:
            self._opened_at = datetime.now()
        elif new_state == CircuitState.CLOSED:
            self._failures.clear()
            self._successes_in_half_open = 0

    def _count_recent_failures(self) -> int:
        """Count failures within the failure window."""
        cutoff = datetime.now() - timedelta(seconds=self.config.failure_window)
        return sum(1 for f in self._failures if f.timestamp > cutoff)

    def _classify_failure(self, exc: Exception) -> FailureType:
        """Classify the type of failure."""
        if isinstance(exc, (ConnectionError, ConnectionRefusedError)):
            return FailureType.CONNECTION
        elif isinstance(exc, TimeoutError):
            return FailureType.TIMEOUT
        elif isinstance(exc, (TypeError, ValueError)):
            return FailureType.SERIALIZATION
        else:
            return FailureType.UNKNOWN

    def record_success(self) -> None:
        """Record a successful operation."""
        with self._lock:
            self._total_calls += 1

            if self._state == CircuitState.HALF_OPEN:
                self._successes_in_half_open += 1
                if self._successes_in_half_open >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)

    def record_failure(self, exc: Exception) -> None:
        """Record a failed operation."""
        with self._lock:
            self._total_calls += 1
            self._total_failures += 1

            # Check if this exception should be excluded
            if isinstance(exc, self.config.excluded_exceptions):
                return

            failure = FailureRecord(
                timestamp=datetime.now(),
                exception_type=type(exc).__name__,
                failure_type=self._classify_failure(exc),
                message=str(exc),
            )
            self._failures.append(failure)
            self._last_failure_time = datetime.now()

            # Check state transitions
            if self._state == CircuitState.HALF_OPEN:
                self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.CLOSED:
                if self._count_recent_failures() >= self.config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)

    def can_execute(self) -> bool:
        """Check if operation can be executed."""
        with self._lock:
            self._check_recovery()

            if self._state == CircuitState.OPEN:
                self._total_rejections += 1
                return False

            return True

    def protect(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator to protect a function with circuit breaker.

        Args:
            func: Function to protect

        Returns:
            Wrapped function

        Raises:
            CircuitOpenError: If circuit is open
        """
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            if not self.can_execute():
                raise CircuitOpenError(
                    f"Circuit breaker is open. "
                    f"Recovery in {self.time_until_recovery:.1f}s"
                )

            try:
                result = func(*args, **kwargs)
                self.record_success()
                return result
            except Exception as e:
                self.record_failure(e)
                raise

        return wrapper

    @property
    def time_until_recovery(self) -> float:
        """Time in seconds until circuit might recover."""
        if self._state != CircuitState.OPEN or self._opened_at is None:
            return 0.0

        elapsed = (datetime.now() - self._opened_at).total_seconds()
        remaining = self.config.recovery_timeout - elapsed
        return max(0.0, remaining)

    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        with self._lock:
            self._transition_to(CircuitState.CLOSED)
            self._failures.clear()
            self._successes_in_half_open = 0

    def get_stats(self) -> dict[str, Any]:
        """Get circuit breaker statistics."""
        with self._lock:
            return {
                "state": self._state.value,
                "total_calls": self._total_calls,
                "total_failures": self._total_failures,
                "total_rejections": self._total_rejections,
                "recent_failures": self._count_recent_failures(),
                "failure_threshold": self.config.failure_threshold,
                "time_until_recovery": self.time_until_recovery,
                "last_failure": self._last_failure_time.isoformat() if self._last_failure_time else None,
            }


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""

    pass


# =============================================================================
# Retry Logic
# =============================================================================


class RetryPolicy:
    """Retry policy with exponential backoff.

    Example:
        policy = RetryPolicy(RetryConfig(max_attempts=3))

        @policy.retry
        def flaky_operation():
            return http_call()
    """

    def __init__(self, config: RetryConfig | None = None):
        self.config = config or RetryConfig()
        self._total_attempts = 0
        self._total_retries = 0
        self._lock = threading.Lock()

    def should_retry(self, exc: Exception, attempt: int) -> bool:
        """Determine if operation should be retried."""
        if attempt >= self.config.max_attempts:
            return False

        return isinstance(exc, self.config.retryable_exceptions)

    def execute_with_retry(
        self,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute function with retry logic.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            Last exception if all retries fail
        """
        last_exception: Exception | None = None

        for attempt in range(self.config.max_attempts):
            with self._lock:
                self._total_attempts += 1
                if attempt > 0:
                    self._total_retries += 1

            try:
                return func(*args, **kwargs)

            except Exception as e:
                last_exception = e

                if not self.should_retry(e, attempt + 1):
                    logger.debug(
                        f"Not retrying {func.__name__}: {type(e).__name__} "
                        f"is not retryable or max attempts reached"
                    )
                    raise

                delay = self.config.calculate_delay(attempt)
                logger.warning(
                    f"Retry {attempt + 1}/{self.config.max_attempts} for "
                    f"{func.__name__} after {delay:.2f}s: {e}"
                )
                time.sleep(delay)

        if last_exception:
            raise last_exception

        raise RuntimeError("Unexpected state in retry logic")

    def retry(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator to add retry logic to a function."""
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return self.execute_with_retry(func, *args, **kwargs)

        return wrapper

    def get_stats(self) -> dict[str, Any]:
        """Get retry statistics."""
        with self._lock:
            return {
                "total_attempts": self._total_attempts,
                "total_retries": self._total_retries,
                "retry_rate": (
                    self._total_retries / self._total_attempts
                    if self._total_attempts > 0 else 0.0
                ),
            }


# =============================================================================
# Health Monitor
# =============================================================================


class HealthMonitor:
    """Monitors backend health with periodic checks.

    Example:
        monitor = HealthMonitor(backend, HealthCheckConfig())
        monitor.start()

        if monitor.is_healthy:
            backend.get(key)

        monitor.stop()
    """

    def __init__(
        self,
        backend: CacheBackend,
        config: HealthCheckConfig | None = None,
        name: str = "cache",
    ):
        self.backend = backend
        self.config = config or HealthCheckConfig()
        self.name = name

        self._health = BackendHealth.UNKNOWN
        self._consecutive_successes = 0
        self._consecutive_failures = 0
        self._last_check: datetime | None = None
        self._last_check_duration: float = 0.0
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()

        # Statistics
        self._total_checks = 0
        self._total_failures = 0

    @property
    def health(self) -> BackendHealth:
        """Get current health status."""
        with self._lock:
            return self._health

    @property
    def is_healthy(self) -> bool:
        """Check if backend is healthy."""
        return self.health == BackendHealth.HEALTHY

    def check_health(self) -> bool:
        """Perform a health check.

        Returns:
            True if healthy
        """
        start = time.perf_counter()

        try:
            # Try a simple operation
            test_key = f"__health_check_{self.name}__"
            self.backend.exists(test_key)

            with self._lock:
                self._total_checks += 1
                self._consecutive_successes += 1
                self._consecutive_failures = 0
                self._last_check = datetime.now()
                self._last_check_duration = time.perf_counter() - start

                if self._consecutive_successes >= self.config.healthy_threshold:
                    self._health = BackendHealth.HEALTHY
                elif self._health == BackendHealth.UNHEALTHY:
                    self._health = BackendHealth.DEGRADED

            logger.debug(
                f"Health check passed for {self.name}: "
                f"{self._last_check_duration*1000:.1f}ms"
            )
            return True

        except Exception as e:
            with self._lock:
                self._total_checks += 1
                self._total_failures += 1
                self._consecutive_failures += 1
                self._consecutive_successes = 0
                self._last_check = datetime.now()
                self._last_check_duration = time.perf_counter() - start

                if self._consecutive_failures >= self.config.unhealthy_threshold:
                    self._health = BackendHealth.UNHEALTHY
                else:
                    self._health = BackendHealth.DEGRADED

            logger.warning(f"Health check failed for {self.name}: {e}")
            return False

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                self.check_health()
            except Exception as e:
                logger.error(f"Health monitor error: {e}")

            time.sleep(self.config.check_interval)

    def start(self) -> None:
        """Start background health monitoring."""
        if not self.config.enabled:
            return

        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name=f"health-monitor-{self.name}",
        )
        self._thread.start()
        logger.info(f"Started health monitor for {self.name}")

    def stop(self) -> None:
        """Stop background health monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info(f"Stopped health monitor for {self.name}")

    def get_stats(self) -> dict[str, Any]:
        """Get health monitor statistics."""
        with self._lock:
            return {
                "name": self.name,
                "health": self._health.value,
                "is_healthy": self.is_healthy,
                "consecutive_successes": self._consecutive_successes,
                "consecutive_failures": self._consecutive_failures,
                "total_checks": self._total_checks,
                "total_failures": self._total_failures,
                "last_check": self._last_check.isoformat() if self._last_check else None,
                "last_check_duration_ms": self._last_check_duration * 1000,
                "running": self._running,
            }


# =============================================================================
# Resilient Cache Backend
# =============================================================================


@dataclass
class ResilienceConfig:
    """Configuration for resilient cache backend."""

    circuit_breaker: CircuitBreakerConfig = field(
        default_factory=CircuitBreakerConfig
    )
    retry: RetryConfig = field(default_factory=RetryConfig)
    health_check: HealthCheckConfig = field(default_factory=HealthCheckConfig)
    fallback_on_error: bool = True
    log_failures: bool = True

    @classmethod
    def default(cls) -> "ResilienceConfig":
        """Default resilience configuration."""
        return cls()

    @classmethod
    def high_availability(cls) -> "ResilienceConfig":
        """High availability configuration."""
        return cls(
            circuit_breaker=CircuitBreakerConfig.lenient(),
            retry=RetryConfig.persistent(),
            health_check=HealthCheckConfig(
                check_interval=10.0,
                healthy_threshold=1,
            ),
        )

    @classmethod
    def low_latency(cls) -> "ResilienceConfig":
        """Low latency configuration - fail fast."""
        return cls(
            circuit_breaker=CircuitBreakerConfig.aggressive(),
            retry=RetryConfig.no_retry(),
            health_check=HealthCheckConfig(check_interval=60.0),
        )


class ResilientCacheBackend(CacheBackend):
    """Cache backend wrapper with resilience patterns.

    Wraps a primary backend with circuit breaker, retry logic,
    and optional fallback to a secondary backend.

    Example:
        primary = RedisCacheBackend(host="redis.example.com")
        fallback = MemoryCacheBackend()

        cache = ResilientCacheBackend(
            primary=primary,
            fallback=fallback,
            config=ResilienceConfig.high_availability(),
        )

        # Automatically falls back to memory on Redis failure
        entry = cache.get("my-key")
    """

    def __init__(
        self,
        primary: CacheBackend,
        fallback: CacheBackend | None = None,
        config: ResilienceConfig | None = None,
        name: str = "resilient-cache",
    ):
        self.primary = primary
        self.fallback = fallback or MemoryCacheBackend()
        self.config = config or ResilienceConfig.default()
        self.name = name

        # Initialize components
        self._circuit_breaker = CircuitBreaker(self.config.circuit_breaker)
        self._retry_policy = RetryPolicy(self.config.retry)
        self._health_monitor = HealthMonitor(
            primary,
            self.config.health_check,
            name=f"{name}-primary",
        )

        # Statistics
        self._primary_calls = 0
        self._fallback_calls = 0
        self._lock = threading.Lock()

        # Start health monitoring
        self._health_monitor.start()

    def _execute_with_resilience(
        self,
        primary_fn: Callable[[], T],
        fallback_fn: Callable[[], T] | None = None,
        operation_name: str = "operation",
    ) -> T:
        """Execute an operation with full resilience patterns.

        Args:
            primary_fn: Primary operation
            fallback_fn: Fallback operation
            operation_name: Name for logging

        Returns:
            Operation result
        """
        # Check circuit breaker
        if not self._circuit_breaker.can_execute():
            if fallback_fn and self.config.fallback_on_error:
                logger.debug(f"Circuit open, using fallback for {operation_name}")
                with self._lock:
                    self._fallback_calls += 1
                return fallback_fn()
            raise CircuitOpenError(f"Circuit is open for {operation_name}")

        # Try primary with retry
        try:
            result = self._retry_policy.execute_with_retry(primary_fn)
            self._circuit_breaker.record_success()
            with self._lock:
                self._primary_calls += 1
            return result

        except Exception as e:
            self._circuit_breaker.record_failure(e)

            if self.config.log_failures:
                logger.warning(
                    f"Primary cache failed for {operation_name}: {e}"
                )

            # Try fallback
            if fallback_fn and self.config.fallback_on_error:
                logger.info(f"Using fallback for {operation_name}")
                with self._lock:
                    self._fallback_calls += 1
                return fallback_fn()

            raise

    def get(self, key: str) -> CacheEntry | None:
        """Get from cache with resilience."""
        return self._execute_with_resilience(
            primary_fn=lambda: self.primary.get(key),
            fallback_fn=lambda: self.fallback.get(key),
            operation_name=f"get:{key[:20]}",
        )

    def set(
        self,
        key: str,
        entry: CacheEntry,
        ttl: timedelta | None = None,
    ) -> None:
        """Set in cache with resilience."""
        def primary_set() -> None:
            self.primary.set(key, entry, ttl)

        def fallback_set() -> None:
            self.fallback.set(key, entry, ttl)

        self._execute_with_resilience(
            primary_fn=primary_set,
            fallback_fn=fallback_set,
            operation_name=f"set:{key[:20]}",
        )

    def delete(self, key: str) -> bool:
        """Delete from cache with resilience."""
        try:
            result = self._execute_with_resilience(
                primary_fn=lambda: self.primary.delete(key),
                fallback_fn=lambda: self.fallback.delete(key),
                operation_name=f"delete:{key[:20]}",
            )
            return result
        except Exception:
            return False

    def clear(self) -> int:
        """Clear cache."""
        count = 0
        try:
            count += self.primary.clear()
        except Exception as e:
            logger.warning(f"Failed to clear primary cache: {e}")

        try:
            count += self.fallback.clear()
        except Exception as e:
            logger.warning(f"Failed to clear fallback cache: {e}")

        return count

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        return self._execute_with_resilience(
            primary_fn=lambda: self.primary.exists(key),
            fallback_fn=lambda: self.fallback.exists(key),
            operation_name=f"exists:{key[:20]}",
        )

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive statistics."""
        with self._lock:
            total_calls = self._primary_calls + self._fallback_calls
            return {
                "type": "resilient",
                "name": self.name,
                "primary_calls": self._primary_calls,
                "fallback_calls": self._fallback_calls,
                "fallback_rate": (
                    self._fallback_calls / total_calls
                    if total_calls > 0 else 0.0
                ),
                "circuit_breaker": self._circuit_breaker.get_stats(),
                "retry": self._retry_policy.get_stats(),
                "health": self._health_monitor.get_stats(),
                "primary": self.primary.get_stats(),
                "fallback": self.fallback.get_stats(),
            }

    def is_primary_healthy(self) -> bool:
        """Check if primary backend is healthy."""
        return self._health_monitor.is_healthy

    def force_primary_check(self) -> bool:
        """Force an immediate health check on primary."""
        return self._health_monitor.check_health()

    def reset_circuit(self) -> None:
        """Manually reset the circuit breaker."""
        self._circuit_breaker.reset()
        logger.info(f"Circuit breaker reset for {self.name}")

    def shutdown(self) -> None:
        """Shutdown the resilient backend."""
        self._health_monitor.stop()


# =============================================================================
# Fallback Chain
# =============================================================================


class FallbackChain(CacheBackend):
    """Chain of cache backends with automatic fallback.

    Tries backends in order until one succeeds.

    Example:
        chain = FallbackChain([
            RedisCacheBackend(host="primary-redis"),
            RedisCacheBackend(host="secondary-redis"),
            FileCacheBackend(cache_dir=".cache"),
            MemoryCacheBackend(),
        ])

        # Will try each backend in order until success
        entry = chain.get("key")
    """

    def __init__(
        self,
        backends: list[CacheBackend],
        retry_config: RetryConfig | None = None,
    ):
        if not backends:
            raise ValueError("At least one backend is required")

        self.backends = backends
        self._retry_policy = RetryPolicy(retry_config or RetryConfig.quick())
        self._lock = threading.Lock()

        # Track which backends are working
        self._backend_health: dict[int, bool] = {
            i: True for i in range(len(backends))
        }
        self._calls_per_backend: dict[int, int] = {
            i: 0 for i in range(len(backends))
        }

    def _try_backends(
        self,
        operation: Callable[[CacheBackend], T],
        operation_name: str = "operation",
    ) -> T:
        """Try operation on backends in order."""
        last_exception: Exception | None = None

        for i, backend in enumerate(self.backends):
            # Skip unhealthy backends (but try last one regardless)
            if not self._backend_health[i] and i < len(self.backends) - 1:
                continue

            try:
                result = operation(backend)

                with self._lock:
                    self._calls_per_backend[i] += 1
                    self._backend_health[i] = True

                if i > 0:
                    logger.debug(
                        f"Fallback to backend {i} succeeded for {operation_name}"
                    )

                return result

            except Exception as e:
                last_exception = e
                with self._lock:
                    self._backend_health[i] = False

                logger.warning(
                    f"Backend {i} failed for {operation_name}: {e}"
                )

        if last_exception:
            raise last_exception

        raise RuntimeError("No backends available")

    def get(self, key: str) -> CacheEntry | None:
        try:
            return self._try_backends(
                lambda b: b.get(key),
                f"get:{key[:20]}",
            )
        except Exception:
            return None

    def set(
        self,
        key: str,
        entry: CacheEntry,
        ttl: timedelta | None = None,
    ) -> None:
        self._try_backends(
            lambda b: b.set(key, entry, ttl),
            f"set:{key[:20]}",
        )

    def delete(self, key: str) -> bool:
        try:
            return self._try_backends(
                lambda b: b.delete(key),
                f"delete:{key[:20]}",
            )
        except Exception:
            return False

    def clear(self) -> int:
        total = 0
        for backend in self.backends:
            try:
                total += backend.clear()
            except Exception as e:
                logger.warning(f"Failed to clear backend: {e}")
        return total

    def exists(self, key: str) -> bool:
        try:
            return self._try_backends(
                lambda b: b.exists(key),
                f"exists:{key[:20]}",
            )
        except Exception:
            return False

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "type": "fallback_chain",
                "backend_count": len(self.backends),
                "backend_health": dict(self._backend_health),
                "calls_per_backend": dict(self._calls_per_backend),
                "backends": [b.get_stats() for b in self.backends],
            }


# =============================================================================
# Factory Functions
# =============================================================================


def create_resilient_redis_backend(
    host: str = "localhost",
    port: int = 6379,
    db: int = 0,
    password: str | None = None,
    fallback_to_memory: bool = True,
    fallback_to_file: bool = False,
    file_cache_dir: str = ".truthound_cache",
    config: ResilienceConfig | None = None,
    **redis_kwargs: Any,
) -> CacheBackend:
    """Create a resilient Redis backend with fallback.

    This is the recommended way to create a Redis-backed cache
    with automatic fallback.

    Args:
        host: Redis host
        port: Redis port
        db: Redis database
        password: Redis password
        fallback_to_memory: Use memory as fallback
        fallback_to_file: Use file as fallback
        file_cache_dir: Directory for file cache
        config: Resilience configuration
        **redis_kwargs: Additional Redis options

    Returns:
        Configured resilient cache backend
    """
    try:
        from truthound.profiler.caching import RedisCacheBackend
        primary = RedisCacheBackend(
            host=host,
            port=port,
            db=db,
            password=password,
            **redis_kwargs,
        )
    except ImportError:
        logger.warning(
            "Redis package not installed. Using memory cache."
        )
        return MemoryCacheBackend()

    # Create fallback chain
    fallbacks: list[CacheBackend] = []
    if fallback_to_file:
        fallbacks.append(FileCacheBackend(cache_dir=file_cache_dir))
    if fallback_to_memory:
        fallbacks.append(MemoryCacheBackend())

    if not fallbacks:
        fallbacks.append(MemoryCacheBackend())

    # Use first fallback as the direct fallback
    fallback = fallbacks[0] if len(fallbacks) == 1 else FallbackChain(fallbacks)

    return ResilientCacheBackend(
        primary=primary,
        fallback=fallback,
        config=config or ResilienceConfig.default(),
        name="resilient-redis",
    )


def create_high_availability_cache(
    primary_host: str,
    secondary_host: str | None = None,
    port: int = 6379,
    **kwargs: Any,
) -> CacheBackend:
    """Create a high-availability cache with multiple Redis instances.

    Args:
        primary_host: Primary Redis host
        secondary_host: Secondary Redis host (optional)
        port: Redis port
        **kwargs: Additional options

    Returns:
        High-availability cache backend
    """
    backends: list[CacheBackend] = []

    try:
        from truthound.profiler.caching import RedisCacheBackend

        backends.append(RedisCacheBackend(host=primary_host, port=port))

        if secondary_host:
            backends.append(RedisCacheBackend(host=secondary_host, port=port))
    except ImportError:
        logger.warning("Redis package not installed.")

    backends.append(FileCacheBackend())
    backends.append(MemoryCacheBackend())

    return FallbackChain(backends)
