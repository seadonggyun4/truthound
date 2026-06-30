---
title: 재시도
---

# 재시도

오케스트레이션 실행에서 Provides을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

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

## Preset 설정s

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

## 재시도 Strategies

| 오케스트레이션 실행에서 Strategy을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 Example을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|----------|-------------|-------------------|
| 오케스트레이션 실행에서 `FIXED`, FIXED을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 Fixed을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 오케스트레이션 실행에서 `EXPONENTIAL`, EXPONENTIAL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 Exponential을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 오케스트레이션 실행에서 `LINEAR`, LINEAR을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 Linear을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 오케스트레이션 실행에서 `FIBONACCI`, FIBONACCI을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 Fibonacci을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

## Builder Pattern

오케스트레이션 실행에서 Fluently을(를) 다루는 항목입니다:

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

Monitor 재시도 events:

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

| 오케스트레이션 실행에서 Calculator을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|------------|-------------|
| 오케스트레이션 실행에서 `FixedDelayCalculator`, FixedDelayCalculator을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 Fixed을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 오케스트레이션 실행에서 `ExponentialDelayCalculator`, ExponentialDelayCalculator을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 Exponential을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 오케스트레이션 실행에서 `LinearDelayCalculator`, LinearDelayCalculator을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 Linear을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 오케스트레이션 실행에서 `FibonacciDelayCalculator`, FibonacciDelayCalculator을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 Fibonacci을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

## Exception Filters

| 오케스트레이션 실행에서 Filter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|--------|-------------|
| 오케스트레이션 실행에서 `TypeBasedExceptionFilter`, TypeBasedExceptionFilter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 Exception을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 오케스트레이션 실행에서 `CallableExceptionFilter`, CallableExceptionFilter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 Function-based을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 오케스트레이션 실행에서 `CompositeExceptionFilter`, CompositeExceptionFilter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 Composite을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
