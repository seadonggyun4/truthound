# Resilience Patterns Configuration

Truthound provides resilience patterns to protect against failures in external dependencies like databases, APIs, and message queues.

## Overview

| Pattern | Purpose | Use Case |
|---------|---------|----------|
| Circuit Breaker | Prevent cascading failures | External APIs, databases |
| Retry | Recover from transient failures | Network issues, timeouts |
| Bulkhead | Isolate resources | Connection pools, thread pools |
| Rate Limiter | Control request rate | API rate limits, quotas |

## CircuitBreakerConfig

Prevents cascading failures by monitoring error rates and temporarily stopping requests to failing services.

### Configuration

```python
from truthound.common.resilience import CircuitBreakerConfig

config = CircuitBreakerConfig(
    failure_threshold=5,           # Failures to open circuit
    success_threshold=3,           # Successes to close circuit
    timeout_seconds=30.0,          # Time before half-open
    half_open_max_calls=3,         # Test calls in half-open state
    failure_rate_threshold=50.0,   # Failure rate % to open
    slow_call_threshold_ms=1000.0, # Slow call definition
    slow_call_rate_threshold=50.0, # Slow call % to open
    window_size=100,               # Measurement window
    excluded_exceptions=(),        # Exceptions that don't count as failures
    record_slow_calls=True,        # Whether to track slow calls
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `failure_threshold` | `int` | `5` | Number of failures before opening circuit |
| `success_threshold` | `int` | `3` | Number of successes in half-open to close circuit |
| `timeout_seconds` | `float` | `30.0` | Time in open state before transitioning to half-open |
| `half_open_max_calls` | `int` | `3` | Maximum calls allowed in half-open state |
| `failure_rate_threshold` | `float` | `50.0` | Failure rate percentage (0-100) to trigger open |
| `slow_call_threshold_ms` | `float` | `1000.0` | Latency threshold in ms for slow calls |
| `slow_call_rate_threshold` | `float` | `50.0` | Slow call rate percentage (0-100) to trigger open |
| `window_size` | `int` | `100` | Number of calls to track for rate calculations |
| `excluded_exceptions` | `tuple` | `()` | Exceptions that don't count as failures |
| `record_slow_calls` | `bool` | `True` | Whether to track slow calls |

### Circuit States

```
     ┌──────────────────────────────────────┐
     │                                      │
     ▼                                      │
 ┌───────┐    failure_threshold    ┌───────┐
 │ CLOSED │ ─────────────────────► │  OPEN │
 └───────┘                         └───────┘
     ▲                                  │
     │                                  │ timeout_seconds
     │                                  ▼
     │    success_threshold      ┌───────────┐
     └─────────────────────────  │ HALF-OPEN │
                                 └───────────┘
```

### Presets

```python
# Aggressive - opens quickly, recovers slowly
# failure_threshold=3, success_threshold=3, timeout_seconds=60.0, failure_rate_threshold=30.0
config = CircuitBreakerConfig.aggressive()

# Lenient - tolerates more failures
# failure_threshold=10, success_threshold=1, timeout_seconds=15.0, failure_rate_threshold=80.0
config = CircuitBreakerConfig.lenient()

# Disabled - effectively disabled (high threshold)
# failure_threshold=1_000_000, timeout_seconds=0.1
config = CircuitBreakerConfig.disabled()

# Database optimized
# failure_threshold=5, success_threshold=2, timeout_seconds=30.0, slow_call_threshold_ms=5000.0
config = CircuitBreakerConfig.for_database()

# External API optimized
# failure_threshold=3, success_threshold=2, timeout_seconds=60.0, slow_call_threshold_ms=2000.0
config = CircuitBreakerConfig.for_external_api()
```

## RetryConfig

Automatically retries failed operations with configurable backoff strategies.

### Configuration

```python
from truthound.common.resilience import RetryConfig

config = RetryConfig(
    max_attempts=3,                # Maximum number of attempts (1 = no retry)
    base_delay=0.1,                # Base delay in seconds
    max_delay=30.0,                # Maximum delay cap in seconds
    exponential_base=2.0,          # Multiplier for exponential backoff
    jitter=True,                   # Whether to add random jitter
    jitter_factor=0.5,             # Maximum jitter as a fraction (0.0-1.0)
    retryable_exceptions=(ConnectionError, TimeoutError, OSError),
    non_retryable_exceptions=(ValueError, TypeError, KeyError),
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_attempts` | `int` | `3` | Maximum number of attempts (1 = no retry) |
| `base_delay` | `float` | `0.1` | Base delay in seconds |
| `max_delay` | `float` | `30.0` | Maximum delay cap in seconds |
| `exponential_base` | `float` | `2.0` | Multiplier for exponential backoff |
| `jitter` | `bool` | `True` | Whether to add random jitter to delays |
| `jitter_factor` | `float` | `0.5` | Maximum jitter as a fraction (0.0-1.0) |
| `retryable_exceptions` | `tuple` | `(ConnectionError, TimeoutError, OSError)` | Exceptions that trigger retry |
| `non_retryable_exceptions` | `tuple` | `(ValueError, TypeError, KeyError)` | Exceptions that should not be retried |

### Delay Calculation

The delay for attempt `n` (0-indexed) is calculated as:

```
delay = min(base_delay * (exponential_base ** n), max_delay)

# With jitter:
jitter_range = delay * jitter_factor
delay = delay + random.uniform(-jitter_range, jitter_range)
```

Example with default settings:
- Attempt 0: ~0.1s
- Attempt 1: ~0.2s
- Attempt 2: ~0.4s
- Attempt 3: ~0.8s

### Presets

```python
# No retry - fail immediately
# max_attempts=1
config = RetryConfig.no_retry()

# Quick retry for transient failures
# max_attempts=3, base_delay=0.05, max_delay=1.0
config = RetryConfig.quick()

# Persistent retry for important operations
# max_attempts=5, base_delay=0.5, max_delay=30.0
config = RetryConfig.persistent()

# Standard exponential backoff
# max_attempts=4, base_delay=0.1, max_delay=10.0, exponential_base=2.0
config = RetryConfig.exponential()
```

### Helper Methods

```python
config = RetryConfig()

# Calculate delay for a specific attempt
delay = config.calculate_delay(attempt=2)  # Returns delay in seconds

# Check if an exception should trigger retry
should_retry = config.is_retryable(ConnectionError())  # True
should_retry = config.is_retryable(ValueError())       # False
```

## BulkheadConfig

Isolates resources to prevent one component from consuming all available resources.

### Configuration

```python
from truthound.common.resilience import BulkheadConfig

config = BulkheadConfig(
    max_concurrent=10,             # Maximum concurrent executions
    max_wait_time=0.0,             # Maximum time to wait for a slot (0 = fail immediately)
    fairness=True,                 # FIFO ordering for waiting requests
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_concurrent` | `int` | `10` | Maximum concurrent executions |
| `max_wait_time` | `float` | `0.0` | Maximum time to wait for a slot in seconds (0 = fail immediately) |
| `fairness` | `bool` | `True` | Whether to use fair (FIFO) ordering for waiting requests |

### Presets

```python
# Small bulkhead for limited resources
# max_concurrent=5
config = BulkheadConfig.small()

# Medium bulkhead for moderate concurrency
# max_concurrent=20
config = BulkheadConfig.medium()

# Large bulkhead for high concurrency
# max_concurrent=50
config = BulkheadConfig.large()

# Database optimized (with wait time)
# max_concurrent=10, max_wait_time=5.0
config = BulkheadConfig.for_database()
```

## RateLimiterConfig

Controls the rate of requests to prevent overwhelming services or exceeding quotas.

### Configuration

```python
from truthound.common.resilience import RateLimiterConfig

config = RateLimiterConfig(
    rate=100,                      # Number of permits per period
    period_seconds=1.0,            # Period duration in seconds
    burst_size=None,               # Maximum burst size (defaults to rate)
    algorithm="token_bucket",      # Rate limiting algorithm
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rate` | `int` | `100` | Number of permits per period |
| `period_seconds` | `float` | `1.0` | Period duration in seconds |
| `burst_size` | `int \| None` | `None` | Maximum burst size (defaults to `rate` if not set) |
| `algorithm` | `str` | `"token_bucket"` | Rate limiting algorithm |

### Algorithms

| Algorithm | Description |
|-----------|-------------|
| `token_bucket` | Tokens refill at a steady rate, allows bursting up to burst_size |
| `sliding_window` | Counts requests in a sliding time window |
| `fixed_window` | Counts requests in fixed time intervals |

### Presets

```python
# N requests per second
config = RateLimiterConfig.per_second(rate=100, burst=150)

# N requests per minute
config = RateLimiterConfig.per_minute(rate=1000, burst=1200)

# N requests per hour
config = RateLimiterConfig.per_hour(rate=10000, burst=12000)
```

### Properties

```python
config = RateLimiterConfig(rate=100, burst_size=None)

# Get effective burst size (defaults to rate if not set)
burst = config.effective_burst_size  # Returns 100
```

## Combining Patterns

Use the `ResilienceBuilder` to combine multiple patterns:

```python
from truthound.common.resilience import (
    ResilienceBuilder,
    CircuitBreakerConfig,
    RetryConfig,
    BulkheadConfig,
    RateLimiterConfig,
)

wrapper = (
    ResilienceBuilder("my-service")
    .with_circuit_breaker(CircuitBreakerConfig.for_external_api())
    .with_retry(RetryConfig.exponential())
    .with_bulkhead(BulkheadConfig.medium())
    .with_rate_limit(RateLimiterConfig.per_second(100))
    .build()
)

# Execute with all resilience patterns
result = wrapper.execute(my_function, args)

# Or use as decorator
@wrapper
def risky_operation():
    return external_service.call()
```

### Pattern Execution Order

When combined, patterns are applied in this order (outer to inner):

1. **Rate Limiter** - Controls request rate
2. **Bulkhead** - Limits concurrent executions
3. **Circuit Breaker** - Monitors failures and opens circuit
4. **Retry** - Retries failed operations

```
Request → Rate Limiter → Bulkhead → Circuit Breaker → Retry → Actual Call
```

## Use Case Examples

### Database Connection

```python
db_config = (
    ResilienceBuilder("database")
    .with_circuit_breaker(CircuitBreakerConfig.for_database())
    .with_retry(RetryConfig(
        max_attempts=3,
        base_delay=0.5,
        retryable_exceptions=(ConnectionError, TimeoutError),
    ))
    .with_bulkhead(BulkheadConfig.for_database())
    .build()
)
```

### External API

```python
api_config = (
    ResilienceBuilder("external-api")
    .with_circuit_breaker(CircuitBreakerConfig.for_external_api())
    .with_retry(RetryConfig.exponential())
    .with_rate_limit(RateLimiterConfig.per_second(100))
    .build()
)
```

### Message Queue Consumer

```python
queue_config = (
    ResilienceBuilder("message-queue")
    .with_circuit_breaker(CircuitBreakerConfig.lenient())
    .with_retry(RetryConfig.persistent())
    .with_bulkhead(BulkheadConfig(max_concurrent=50, max_wait_time=10.0))
    .build()
)
```

## Validation

All configuration classes validate their parameters on initialization:

```python
# Raises ValueError: failure_threshold must be positive
CircuitBreakerConfig(failure_threshold=0)

# Raises ValueError: max_attempts must be at least 1
RetryConfig(max_attempts=0)

# Raises ValueError: max_concurrent must be positive
BulkheadConfig(max_concurrent=0)

# Raises ValueError: algorithm must be one of {'token_bucket', 'sliding_window', 'fixed_window'}
RateLimiterConfig(algorithm="invalid")
```
