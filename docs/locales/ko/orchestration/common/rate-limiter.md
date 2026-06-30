---
title: Rate Limiting
---

# Rate Limiting

오케스트레이션 실행에서 API, Controls을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Basic Usage

```python
from common.rate_limiter import rate_limit, RateLimitConfig, RateLimitAlgorithm

# Simple decorator usage
@rate_limit(max_requests=100, window_seconds=60.0)
def call_api():
    return api.get("/data")

# Detailed control with configuration object
config = RateLimitConfig(
    max_requests=100,
    window_seconds=60.0,
    algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
    burst_size=20,
)

@rate_limit(config=config)
async def async_call():
    return await api.async_get("/data")
```

## Preset 설정s

```python
from common.rate_limiter import (
    DEFAULT_RATE_LIMIT_CONFIG,    # Default: 100/60s, Token Bucket
    STRICT_RATE_LIMIT_CONFIG,     # Strict: 10/60s, Sliding Window
    LENIENT_RATE_LIMIT_CONFIG,    # Lenient: 1000/60s, Wait on limit
    BURST_RATE_LIMIT_CONFIG,      # Burst: 100/60s, burst=50
    API_RATE_LIMIT_CONFIG,        # API: 100/1s
)

@rate_limit(config=STRICT_RATE_LIMIT_CONFIG)
def critical_operation():
    return payment_service.process()
```

## Rate Limiting Algorithms

| 오케스트레이션 실행에서 Algorithm을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 Case을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-----------|-------------|----------|
| 오케스트레이션 실행에서 `TOKEN_BUCKET`, TOKEN_BUCKET을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 Allows을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 API, General을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 오케스트레이션 실행에서 `SLIDING_WINDOW`, SLIDING_WINDOW을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 Precise을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 Strict을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 오케스트레이션 실행에서 `FIXED_WINDOW`, FIXED_WINDOW을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 Fixed을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 Simple을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 오케스트레이션 실행에서 `LEAKY_BUCKET`, LEAKY_BUCKET을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 Constant을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 Traffic을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

```python
from common.rate_limiter import (
    TokenBucketRateLimiter,
    SlidingWindowRateLimiter,
    FixedWindowRateLimiter,
    LeakyBucketRateLimiter,
)
```

## Builder Pattern

```python
from common.rate_limiter import RateLimitConfig, RateLimitAction

config = RateLimitConfig()
config = config.with_max_requests(50)
config = config.with_window(30.0)
config = config.with_algorithm(RateLimitAlgorithm.SLIDING_WINDOW)
config = config.with_burst_size(10)
config = config.with_on_limit(RateLimitAction.WAIT)
config = config.with_name("external_api")
```

## Rate Limit Actions

| 오케스트레이션 실행에서 Action을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|--------|-------------|
| 오케스트레이션 실행에서 `REJECT`, REJECT을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 Raise을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 오케스트레이션 실행에서 `WAIT`, WAIT을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 Wait을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 오케스트레이션 실행에서 `WARN`, WARN을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 Log을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

## Key Extractor (Multi-tenant)

```python
from common.rate_limiter import (
    rate_limit,
    DefaultKeyExtractor,
    ArgumentKeyExtractor,
    CallableKeyExtractor,
)

# Argument-based key extraction (per-user rate limiting)
@rate_limit(
    max_requests=10,
    window_seconds=60.0,
    key_extractor=ArgumentKeyExtractor(arg_name="user_id", prefix="user"),
)
def user_action(user_id: str):
    return process(user_id)

# Custom key extraction
def custom_extractor(func, args, kwargs):
    return f"tenant:{kwargs.get('tenant_id', 'default')}"

@rate_limit(
    max_requests=100,
    key_extractor=CallableKeyExtractor(custom_extractor),
)
def tenant_action(tenant_id: str):
    return process(tenant_id)
```

## Hooks

```python
from common.rate_limiter import (
    rate_limit,
    LoggingRateLimitHook,
    MetricsRateLimitHook,
)

# Logging hook
logging_hook = LoggingRateLimitHook()

# Metrics hook
metrics_hook = MetricsRateLimitHook()

@rate_limit(max_requests=100, hooks=[logging_hook, metrics_hook])
def monitored_api_call():
    return external_service.call()

# Query metrics
print(metrics_hook.acquired_count)     # Acquired token count
print(metrics_hook.rejected_count)     # Rejected request count
print(metrics_hook.waited_count)       # Waited request count
print(metrics_hook.average_wait_time)  # Average wait time
```

## Registry Usage

```python
from common.rate_limiter import get_rate_limiter, RateLimiterRegistry

# Get Rate Limiter from global registry
limiter = get_rate_limiter("external_api", config=RateLimitConfig())
if limiter.acquire("user_123"):
    process_request()

# Registry management
registry = RateLimiterRegistry()
registry.reset_all()
stats = registry.get_all_stats()
```

## Exception Handling

```python
from common.rate_limiter import rate_limit, RateLimitExceededError

@rate_limit(max_requests=10, window_seconds=60.0)
def api_call():
    return requests.get("/data")

try:
    result = api_call()
except RateLimitExceededError as e:
    print(f"Rate limit exceeded: retry after {e.retry_after_seconds} seconds")
    result = get_cached_data()
```
