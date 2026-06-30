# 복원력 Patterns 설정

실무 운영 가이드에서 Truthound, API, APIs을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 개요

| 실무 운영 가이드에서 Pattern을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Purpose을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Case을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|---------|---------|----------|
| 실무 운영 가이드에서 Circuit, Breaker을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Prevent cascading 실패 | 실무 운영 가이드에서 API, External, APIs을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 재시도 | Recover from transient 실패 | 실무 운영 가이드에서 Network을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Bulkhead을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Isolate을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Connection을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Rate, Limiter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Control을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 API을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

## CircuitBreakerConfig

실무 운영 가이드에서 Prevents을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### 설정

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

| 실무 운영 가이드에서 Parameter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Type을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Default을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-----------|------|---------|-------------|
| 실무 운영 가이드에서 `failure_threshold`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `int`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `5`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Number을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `success_threshold`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `int`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `3`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Number을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `timeout_seconds`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `float`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `30.0`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Time을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `half_open_max_calls`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `int`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `3`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Maximum을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `failure_rate_threshold`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `float`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `50.0`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Failure을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `slow_call_threshold_ms`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `float`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `1000.0`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Latency을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `slow_call_rate_threshold`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `float`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `50.0`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Slow을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `window_size`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `int`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `100`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Number을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `excluded_exceptions`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `tuple`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `()`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Exceptions을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `record_slow_calls`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `bool`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `True`, True을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Whether을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

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

실무 운영 가이드에서 Automatically을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### 설정

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

| 실무 운영 가이드에서 Parameter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Type을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Default을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-----------|------|---------|-------------|
| 실무 운영 가이드에서 `max_attempts`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `int`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `3`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Maximum number of attempts (1 = no 재시도) |
| 실무 운영 가이드에서 `base_delay`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `float`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `0.1`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Base을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `max_delay`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `float`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `30.0`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Maximum을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `exponential_base`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `float`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `2.0`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Multiplier을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `jitter`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `bool`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `True`, True을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Whether을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `jitter_factor`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `float`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `0.5`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Maximum을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `retryable_exceptions`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `tuple`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `(ConnectionError, TimeoutError, OSError)`, ConnectionError, TimeoutError, OSError을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Exceptions that trigger 재시도 |
| 실무 운영 가이드에서 `non_retryable_exceptions`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `tuple`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `(ValueError, TypeError, KeyError)`, ValueError, TypeError, KeyError을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Exceptions을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### Delay Calculation

실무 운영 가이드에서 `n`을(를) 다루는 항목입니다:

```
delay = min(base_delay * (exponential_base ** n), max_delay)

# With jitter:
jitter_range = delay * jitter_factor
delay = delay + random.uniform(-jitter_range, jitter_range)
```

실무 운영 가이드에서 Example을(를) 다루는 항목입니다:
- 실무 운영 가이드에서 Attempt을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Attempt을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Attempt을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Attempt을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

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

실무 운영 가이드에서 Isolates을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### 설정

```python
from truthound.common.resilience import BulkheadConfig

config = BulkheadConfig(
    max_concurrent=10,             # Maximum concurrent executions
    max_wait_time=0.0,             # Maximum time to wait for a slot (0 = fail immediately)
    fairness=True,                 # FIFO ordering for waiting requests
)
```

### Parameters

| 실무 운영 가이드에서 Parameter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Type을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Default을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-----------|------|---------|-------------|
| 실무 운영 가이드에서 `max_concurrent`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `int`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `10`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Maximum을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `max_wait_time`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `float`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `0.0`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Maximum을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `fairness`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `bool`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `True`, True을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Whether, FIFO을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

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

실무 운영 가이드에서 Controls을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### 설정

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

| 실무 운영 가이드에서 Parameter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Type을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Default을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-----------|------|---------|-------------|
| 실무 운영 가이드에서 `rate`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `int`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `100`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Number을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `period_seconds`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `float`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `1.0`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Period을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `burst_size`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `None`, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `rate`, Maximum을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `algorithm`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `str`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `"token_bucket"`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Rate을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### Algorithms

| 실무 운영 가이드에서 Algorithm을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-----------|-------------|
| 실무 운영 가이드에서 `token_bucket`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Tokens을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `sliding_window`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Counts을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `fixed_window`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Counts을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

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

실무 운영 가이드에서 `ResilienceBuilder`, ResilienceBuilder을(를) 다루는 항목입니다:

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

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 다루는 항목입니다:

1. 실무 운영 가이드에서 Rate, Limiter, Controls을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
2. 실무 운영 가이드에서 Bulkhead, Limits을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
3. 실무 운영 가이드에서 Circuit, Breaker, Monitors을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
4. **재시도** - Retries failed operations

```
Request → Rate Limiter → Bulkhead → Circuit Breaker → Retry → Actual Call
```

## Use Case 예시

### 데이터베이스 Connection

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

## 검증 파이프라인 통합 (VE-5)

실무 운영 가이드에서 Truthound, `ValidationResiliencePolicy`, ValidationResiliencePolicy을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### 검증복원력Policy

```python
from truthound.validators.resilience_bridge import (
    ValidationResiliencePolicy,
    create_default_policy,
    create_strict_policy,
)

# Default: lenient circuit breaker (10 failures to open, 15s timeout)
policy = create_default_policy(max_retries=2)

# Strict: no retries, aggressive circuit breaker
policy = create_strict_policy()

# Custom policy
from truthound.common.resilience import CircuitBreakerConfig

policy = ValidationResiliencePolicy(
    circuit_breaker_config=CircuitBreakerConfig(
        failure_threshold=5,
        success_threshold=2,
        timeout_seconds=30.0,
    ),
    max_retries=3,
    on_retry=lambda attempt, exc, delay: logger.warning(
        f"Retry {attempt} after {delay:.1f}s: {exc}"
    ),
)
```

### Execution 플로우

```
policy.execute(validator, lf)
    │
    ├── Circuit breaker check → OPEN? → return SKIPPED result
    │
    ├── _validate_safe(validator, lf, max_retries=policy.max_retries)
    │       │
    │       ├── Attempt 1 → success? → record success → return
    │       ├── Attempt 2 → exponential backoff (0.1s × 2^n, cap 5s)
    │       └── Attempt N → max_retries exceeded → record failure
    │
    └── Circuit breaker update → failure count → maybe OPEN
```

### Per-검증기 Circuit States

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 다루는 항목입니다:

```python
# Check circuit state for a specific validator
state = policy.get_circuit_state("null_check")  # "CLOSED" | "OPEN" | "HALF_OPEN"

# Reset a specific validator's circuit
policy.reset("null_check")

# Reset all circuits
policy.reset()
```

### Exception Classification

실무 운영 가이드에서 `ExceptionInfo.from_exception()`, ExceptionInfo.from_exception을(를) 다루는 항목입니다:

| 실무 운영 가이드에서 Category을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Retryable을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Exception, Types을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|----------|-----------|----------------|
| 실무 운영 가이드에서 `transient`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Yes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `TimeoutError`, `ConnectionError`, `OSError`, `ValidationTimeoutError`, TimeoutError, ConnectionError, OSError, ValidationTimeoutError을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `configuration`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `ValueError`, `TypeError`, `KeyError`, `ColumnNotFoundError`, ValueError, TypeError, KeyError, ColumnNotFoundError을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `data`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Polars, `ComputeError`, `SchemaError`, ComputeError, SchemaError을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `permanent`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

실무 운영 가이드에서 `transient`, Retry을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### 통합 with th.check()

실무 운영 가이드에서 `catch_exceptions`, `max_retries`, `th.check()`을(를) 다루는 항목입니다:

```python
import truthound as th

# All validators wrapped with exception isolation and retry
run = th.check(
    "data.csv",
    catch_exceptions=True,   # Default: True
    max_retries=3,            # Retry transient errors
)

# Access execution issues captured on the ValidationRunResult
if run.execution_issues:
    print(f"Exceptions: {len(run.execution_issues)}")
    categories: dict[str, int] = {}
    for execution_issue in run.execution_issues:
        category = execution_issue.failure_category or "unknown"
        categories[category] = categories.get(category, 0) + 1
    print(f"By category: {categories}")
```

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 검증

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 다루는 항목입니다:

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
