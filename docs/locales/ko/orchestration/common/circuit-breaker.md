---
title: Circuit Breaker
---

# Circuit Breaker

오케스트레이션 실행에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Basic Usage

```python
from common.circuit_breaker import circuit_breaker, CircuitBreakerConfig

# Simple decorator usage
@circuit_breaker(failure_threshold=5, recovery_timeout_seconds=30.0)
def call_external_api():
    return api.get("/data")

# Detailed control with configuration object
config = CircuitBreakerConfig(
    failure_threshold=5,      # Circuit opens after 5 failures
    success_threshold=2,      # Circuit closes after 2 successes
    recovery_timeout_seconds=30.0,  # Half-open attempt after 30 seconds
)

@circuit_breaker(config=config)
async def async_call():
    return await api.async_get("/data")
```

## Preset 설정s

```python
from common.circuit_breaker import (
    DEFAULT_CIRCUIT_BREAKER_CONFIG,     # Default: 5 failures, 30s recovery
    SENSITIVE_CIRCUIT_BREAKER_CONFIG,   # Sensitive: 3 failures, 60s recovery
    RESILIENT_CIRCUIT_BREAKER_CONFIG,   # Resilient: 10 failures, 15s recovery
    AGGRESSIVE_CIRCUIT_BREAKER_CONFIG,  # Aggressive: 2 failures, 120s recovery
)

@circuit_breaker(config=SENSITIVE_CIRCUIT_BREAKER_CONFIG)
def critical_operation():
    return payment_service.process()
```

## States and Behavior

| 오케스트레이션 실행에서 State을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-------|-------------|
| 오케스트레이션 실행에서 `CLOSED`, CLOSED을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 Normal을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 오케스트레이션 실행에서 `OPEN`, OPEN을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 Failure을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 오케스트레이션 실행에서 `HALF_OPEN`, HALF_OPEN을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 Recovery을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

```
CLOSED --[failure threshold reached]--> OPEN --[recovery time elapsed]--> HALF_OPEN
   ^                                                                          |
   |                                                                          v
   +---[success threshold reached]--<--------<--------<-----------------------+
                                                                              |
   +---[failure occurred]--<--------<--------<--------------------------------+
   |
   v
  OPEN
```

## Builder Pattern

```python
from common.circuit_breaker import CircuitBreakerConfig

config = CircuitBreakerConfig()
config = config.with_failure_threshold(3)
config = config.with_recovery_timeout(60.0)
config = config.with_exceptions(
    exceptions=(ConnectionError, TimeoutError),
    ignored=(ValueError,),  # Exceptions to ignore
)
config = config.with_name("payment_api")
```

## Hooks

```python
from common.circuit_breaker import (
    circuit_breaker,
    LoggingCircuitBreakerHook,
    MetricsCircuitBreakerHook,
)

# Logging hook: automatic state change logging
logging_hook = LoggingCircuitBreakerHook()

# Metrics hook: statistics collection
metrics_hook = MetricsCircuitBreakerHook()

@circuit_breaker(failure_threshold=5, hooks=[logging_hook, metrics_hook])
def monitored_api_call():
    return external_service.call()

# Query metrics
print(metrics_hook.times_opened)    # Times circuit opened
print(metrics_hook.rejected_count)  # Rejected request count
print(metrics_hook.success_count)   # Success count
```

## Registry Usage

```python
from common.circuit_breaker import get_circuit_breaker, CircuitBreakerRegistry

# Get circuit breaker from global registry (or create)
cb = get_circuit_breaker("external_api", config=CircuitBreakerConfig())
result = cb.call(api_function)

# Registry management
registry = CircuitBreakerRegistry()
registry.reset_all()                   # Reset all circuits
states = registry.get_all_states()     # Query all states
```

## Exception Handling

```python
from common.circuit_breaker import circuit_breaker, CircuitOpenError

@circuit_breaker(failure_threshold=3)
def api_call():
    return requests.get("/data")

try:
    result = api_call()
except CircuitOpenError as e:
    print(f"Circuit open: retry possible after {e.remaining_seconds} seconds")
    # Execute fallback logic
    result = get_cached_data()
```

## 실패 Detectors

| 오케스트레이션 실행에서 Detector을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|----------|-------------|
| 오케스트레이션 실행에서 `TypeBasedFailureDetector`, TypeBasedFailureDetector을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 Exception을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 오케스트레이션 실행에서 `CallableFailureDetector`, CallableFailureDetector을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 Function-based을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 오케스트레이션 실행에서 `CompositeFailureDetector`, CompositeFailureDetector을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 오케스트레이션 실행에서 Composite을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
