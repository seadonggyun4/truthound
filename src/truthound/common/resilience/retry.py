"""Retry policy implementations.

This module provides configurable retry logic with various
backoff strategies for handling transient failures.
"""

from __future__ import annotations

import functools
import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

from truthound.common.resilience.config import RetryConfig
from truthound.common.resilience.protocols import RetryPolicyProtocol

logger = logging.getLogger(__name__)

R = TypeVar("R")


class RetryExhaustedError(Exception):
    """Raised when all retry attempts are exhausted."""

    def __init__(
        self,
        message: str,
        attempts: int,
        last_error: Exception | None = None,
    ):
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(message)


class BackoffStrategy(ABC):
    """Abstract base class for backoff strategies."""

    @abstractmethod
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt (0-indexed)."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset the strategy state."""
        ...


@dataclass
class ExponentialBackoff(BackoffStrategy):
    """Exponential backoff strategy.

    Delay = base_delay * (multiplier ^ attempt)

    Example:
        backoff = ExponentialBackoff(base_delay=0.1, multiplier=2.0)
        # Attempt 0: 0.1s
        # Attempt 1: 0.2s
        # Attempt 2: 0.4s
        # Attempt 3: 0.8s
    """

    base_delay: float = 0.1
    multiplier: float = 2.0
    max_delay: float = 30.0

    def get_delay(self, attempt: int) -> float:
        """Calculate exponential delay."""
        delay = self.base_delay * (self.multiplier ** attempt)
        return min(delay, self.max_delay)

    def reset(self) -> None:
        """No state to reset."""
        pass


@dataclass
class LinearBackoff(BackoffStrategy):
    """Linear backoff strategy.

    Delay = base_delay + (increment * attempt)

    Example:
        backoff = LinearBackoff(base_delay=0.1, increment=0.5)
        # Attempt 0: 0.1s
        # Attempt 1: 0.6s
        # Attempt 2: 1.1s
    """

    base_delay: float = 0.1
    increment: float = 0.5
    max_delay: float = 30.0

    def get_delay(self, attempt: int) -> float:
        """Calculate linear delay."""
        delay = self.base_delay + (self.increment * attempt)
        return min(delay, self.max_delay)

    def reset(self) -> None:
        """No state to reset."""
        pass


@dataclass
class ConstantBackoff(BackoffStrategy):
    """Constant backoff strategy.

    Always returns the same delay.

    Example:
        backoff = ConstantBackoff(delay=1.0)
        # All attempts: 1.0s
    """

    delay: float = 1.0

    def get_delay(self, attempt: int) -> float:
        """Return constant delay."""
        return self.delay

    def reset(self) -> None:
        """No state to reset."""
        pass


@dataclass
class JitteredBackoff(BackoffStrategy):
    """Backoff with jitter to prevent thundering herd.

    Wraps another backoff strategy and adds random jitter.

    Example:
        backoff = JitteredBackoff(
            base=ExponentialBackoff(),
            jitter_factor=0.5,
        )
    """

    base: BackoffStrategy
    jitter_factor: float = 0.5
    jitter_mode: str = "full"  # "full", "equal", "decorrelated"

    def get_delay(self, attempt: int) -> float:
        """Calculate jittered delay."""
        base_delay = self.base.get_delay(attempt)

        if self.jitter_mode == "full":
            # Full jitter: random between 0 and base_delay
            return random.uniform(0, base_delay)
        elif self.jitter_mode == "equal":
            # Equal jitter: base_delay/2 + random(0, base_delay/2)
            half = base_delay / 2
            return half + random.uniform(0, half)
        else:  # decorrelated
            # Decorrelated jitter
            jitter_range = base_delay * self.jitter_factor
            return base_delay + random.uniform(-jitter_range, jitter_range)

    def reset(self) -> None:
        """Reset base strategy."""
        self.base.reset()


class RetryPolicy(RetryPolicyProtocol):
    """Configurable retry policy implementation.

    Provides retry logic with pluggable backoff strategies and
    configurable exception handling.

    Example:
        retry = RetryPolicy(RetryConfig.exponential())

        # As decorator
        @retry
        def flaky_operation():
            return unreliable_service.call()

        # Direct execution
        result = retry.execute(flaky_operation, arg1, arg2)
    """

    def __init__(
        self,
        config: RetryConfig | None = None,
        backoff: BackoffStrategy | None = None,
        on_retry: Callable[[int, Exception, float], None] | None = None,
    ):
        """Initialize retry policy.

        Args:
            config: Retry configuration.
            backoff: Custom backoff strategy (overrides config-based backoff).
            on_retry: Callback called before each retry (attempt, error, delay).
        """
        self._config = config or RetryConfig()
        self._backoff = backoff or ExponentialBackoff(
            base_delay=self._config.base_delay,
            multiplier=self._config.exponential_base,
            max_delay=self._config.max_delay,
        )
        self._on_retry = on_retry
        self._name = f"retry-{id(self)}"

        # Metrics
        self._total_attempts = 0
        self._successful_attempts = 0
        self._failed_attempts = 0

    @property
    def name(self) -> str:
        """Get policy name."""
        return self._name

    @property
    def config(self) -> RetryConfig:
        """Get configuration."""
        return self._config

    def should_retry(self, attempt: int, error: Exception) -> bool:
        """Check if operation should be retried."""
        if attempt >= self._config.max_attempts - 1:
            return False
        return self._config.is_retryable(error)

    def get_delay(self, attempt: int) -> float:
        """Get delay before next retry."""
        delay = self._backoff.get_delay(attempt)

        if self._config.jitter:
            jitter_range = delay * self._config.jitter_factor
            delay = delay + random.uniform(-jitter_range, jitter_range)
            delay = max(0, delay)

        return delay

    def execute(self, func: Callable[..., R], *args: Any, **kwargs: Any) -> R:
        """Execute function with retry policy."""
        last_error: Exception | None = None

        for attempt in range(self._config.max_attempts):
            self._total_attempts += 1

            try:
                result = func(*args, **kwargs)
                self._successful_attempts += 1
                return result

            except Exception as e:
                last_error = e
                self._failed_attempts += 1

                # Check if exception is retryable at all
                if not self._config.is_retryable(e):
                    logger.warning(
                        f"Retry policy: non-retryable error on attempt {attempt + 1}: {e}"
                    )
                    raise

                # If this was the last attempt, don't retry
                if attempt >= self._config.max_attempts - 1:
                    break

                delay = self.get_delay(attempt)
                logger.info(
                    f"Retry policy: attempt {attempt + 1} failed, "
                    f"retrying in {delay:.2f}s: {e}"
                )

                if self._on_retry:
                    try:
                        self._on_retry(attempt, e, delay)
                    except Exception as callback_error:
                        logger.warning(f"Error in retry callback: {callback_error}")

                time.sleep(delay)

        raise RetryExhaustedError(
            f"All {self._config.max_attempts} retry attempts exhausted",
            attempts=self._config.max_attempts,
            last_error=last_error,
        )

    def reset(self) -> None:
        """Reset the policy state."""
        self._backoff.reset()
        self._total_attempts = 0
        self._successful_attempts = 0
        self._failed_attempts = 0

    def get_metrics(self) -> dict[str, Any]:
        """Get retry metrics."""
        return {
            "total_attempts": self._total_attempts,
            "successful_attempts": self._successful_attempts,
            "failed_attempts": self._failed_attempts,
            "success_rate": (
                self._successful_attempts / self._total_attempts * 100
                if self._total_attempts > 0
                else 0.0
            ),
        }

    def __call__(self, func: Callable[..., R]) -> Callable[..., R]:
        """Decorator for retryable execution."""
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> R:
            return self.execute(func, *args, **kwargs)
        return wrapper


def with_retry(
    max_attempts: int = 3,
    base_delay: float = 0.1,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    retryable_exceptions: tuple[type[Exception], ...] | None = None,
) -> Callable[[Callable[..., R]], Callable[..., R]]:
    """Convenience decorator factory for retry.

    Example:
        @with_retry(max_attempts=3, base_delay=0.1)
        def flaky_operation():
            return unreliable_service.call()
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        retryable_exceptions=retryable_exceptions or (Exception,),
    )
    policy = RetryPolicy(config)

    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        return policy(func)

    return decorator
