---
title: Retry
---

# Retry

Provides retry decorators and various backoff strategies for handling transient failures.

## Basic Usage

```python
from common.retry import retry, RetryConfig, RetryStrategy

# Simple decorator usage
@retry(max_attempts=3, exceptions=(ConnectionError, TimeoutError))
def fetch_data():
    return api.get("/data")

# Detailed control with configuration object
config = RetryConfig(
    max_attempts=5,
    base_delay_seconds=1.0,
    max_delay_seconds=60.0,
    strategy=RetryStrategy.EXPONENTIAL,
    jitter=True,
)

@retry(config=config)
async def async_fetch():
    return await api.async_get("/data")
```

## Preset Configurations

```python
from common.retry import (
    DEFAULT_RETRY_CONFIG,      # Default: 3 attempts, exponential backoff
    AGGRESSIVE_RETRY_CONFIG,   # Aggressive: 10 attempts, 0.5s start
    CONSERVATIVE_RETRY_CONFIG, # Conservative: 3 attempts, 5s start
    NO_DELAY_RETRY_CONFIG,     # No delay: for testing
)

@retry(config=AGGRESSIVE_RETRY_CONFIG)
def unreliable_operation():
    return external_service.call()
```

## Retry Strategies

| Strategy | Description | Example (base=1s) |
|----------|-------------|-------------------|
| `FIXED` | Fixed delay | 1s, 1s, 1s |
| `EXPONENTIAL` | Exponential backoff | 1s, 2s, 4s, 8s |
| `LINEAR` | Linear increase | 1s, 2s, 3s, 4s |
| `FIBONACCI` | Fibonacci sequence | 1s, 1s, 2s, 3s, 5s |

## Builder Pattern

Fluently modify immutable configuration objects:

```python
from common.retry import RetryConfig, RetryStrategy

config = RetryConfig()
config = config.with_max_attempts(5)
config = config.with_delays(base_delay_seconds=2.0, max_delay_seconds=120.0)
config = config.with_strategy(RetryStrategy.LINEAR)
config = config.with_exceptions(
    exceptions=(ValueError, ConnectionError),
    non_retryable=(KeyError,),
)
```

## Hooks

Monitor retry events:

```python
from common.retry import retry, LoggingRetryHook, MetricsRetryHook

# Logging hook: automatic retry event logging
logging_hook = LoggingRetryHook()

# Metrics hook: retry statistics collection
metrics_hook = MetricsRetryHook()

@retry(max_attempts=3, hooks=[logging_hook, metrics_hook])
def monitored_operation():
    return do_something()

# Query metrics
print(metrics_hook.total_retries)
print(metrics_hook.successful_retries)
print(metrics_hook.failed_operations)
```

## Usage Without Decorator

```python
from common.retry import retry_call, retry_call_async, RetryExecutor

# Synchronous function
result = retry_call(
    external_api.fetch,
    endpoint="/data",
    config=RetryConfig(max_attempts=3),
)

# Asynchronous function
result = await retry_call_async(
    async_api.fetch,
    endpoint="/data",
    config=RetryConfig(max_attempts=3),
)

# Using RetryExecutor
executor = RetryExecutor(config)
result = executor.execute(function, *args, **kwargs)
```

## Delay Calculators

| Calculator | Description |
|------------|-------------|
| `FixedDelayCalculator` | Fixed delay |
| `ExponentialDelayCalculator` | Exponential backoff |
| `LinearDelayCalculator` | Linear increase |
| `FibonacciDelayCalculator` | Fibonacci sequence |

## Exception Filters

| Filter | Description |
|--------|-------------|
| `TypeBasedExceptionFilter` | Exception type-based filtering |
| `CallableExceptionFilter` | Function-based filtering |
| `CompositeExceptionFilter` | Composite filter |
