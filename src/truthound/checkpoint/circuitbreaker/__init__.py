"""Circuit Breaker Pattern Implementation for Truthound.

This module provides a robust Circuit Breaker implementation with support for:
- Multiple failure detection strategies
- Configurable state transitions
- Event-driven state change notifications
- Metrics and monitoring
- Thread-safe operation

Architecture:
    The Circuit Breaker pattern prevents cascading failures by wrapping calls
    to external services or unreliable operations. It monitors for failures
    and "trips" (opens) when a threshold is exceeded, failing fast instead
    of waiting for timeouts.

States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Failure threshold exceeded, requests fail immediately
    - HALF_OPEN: Testing if the service has recovered

Example:
    >>> from truthound.checkpoint.circuitbreaker import (
    ...     CircuitBreaker,
    ...     CircuitBreakerConfig,
    ...     circuit_breaker,
    ... )
    >>>
    >>> # Create circuit breaker
    >>> breaker = CircuitBreaker(
    ...     name="external_api",
    ...     config=CircuitBreakerConfig(
    ...         failure_threshold=5,
    ...         recovery_timeout=30.0,
    ...         half_open_max_calls=3,
    ...     ),
    ... )
    >>>
    >>> # Use with context manager
    >>> with breaker:
    ...     response = call_external_api()
    >>>
    >>> # Or with decorator
    >>> @circuit_breaker(name="db_calls", failure_threshold=3)
    ... def query_database():
    ...     return db.execute(query)

Failure Detection Strategies:
    1. COUNT: Trip after N consecutive failures
    2. PERCENTAGE: Trip when failure rate exceeds threshold
    3. TIME_WINDOW: Trip based on failures within time window
    4. COMPOSITE: Combine multiple strategies
"""

from truthound.checkpoint.circuitbreaker.core import (
    # Enums
    CircuitState,
    FailureDetectionStrategy,
    # Configuration
    CircuitBreakerConfig,
    # Exceptions
    CircuitBreakerError,
    CircuitOpenError,
    CircuitHalfOpenError,
    # Results
    CallResult,
    CircuitBreakerMetrics,
    StateChangeEvent,
)

from truthound.checkpoint.circuitbreaker.detection import (
    # Base
    FailureDetector,
    # Implementations
    ConsecutiveFailureDetector,
    PercentageFailureDetector,
    TimeWindowFailureDetector,
    CompositeFailureDetector,
    # Factory
    create_detector,
)

from truthound.checkpoint.circuitbreaker.breaker import (
    CircuitBreaker,
    CircuitBreakerStateMachine,
)

from truthound.checkpoint.circuitbreaker.registry import (
    CircuitBreakerRegistry,
    get_registry,
    get_breaker,
    register_breaker,
)

from truthound.checkpoint.circuitbreaker.middleware import (
    CircuitBreakerMiddleware,
    circuit_breaker,
    with_circuit_breaker,
)

__all__ = [
    # Enums
    "CircuitState",
    "FailureDetectionStrategy",
    # Configuration
    "CircuitBreakerConfig",
    # Exceptions
    "CircuitBreakerError",
    "CircuitOpenError",
    "CircuitHalfOpenError",
    # Results
    "CallResult",
    "CircuitBreakerMetrics",
    "StateChangeEvent",
    # Detection
    "FailureDetector",
    "ConsecutiveFailureDetector",
    "PercentageFailureDetector",
    "TimeWindowFailureDetector",
    "CompositeFailureDetector",
    "create_detector",
    # Breaker
    "CircuitBreaker",
    "CircuitBreakerStateMachine",
    # Registry
    "CircuitBreakerRegistry",
    "get_registry",
    "get_breaker",
    "register_breaker",
    # Middleware
    "CircuitBreakerMiddleware",
    "circuit_breaker",
    "with_circuit_breaker",
]
