---
title: Common API Reference
---

# Common Module API Reference

## base.py

### Enums

```python
class CheckStatus(Enum):
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"
    ERROR = "error"

class Severity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class FailureAction(Enum):
    RAISE = "raise"
    WARN = "warn"
    LOG = "log"
    CONTINUE = "continue"
```

### CheckConfig

```python
@dataclass(frozen=True)
class CheckConfig:
    rules: tuple[dict[str, Any], ...] = ()
    fail_on_error: bool = True
    timeout_seconds: float | None = None
    tags: frozenset[str] = frozenset()
```

### CheckResult

```python
@dataclass(frozen=True)
class CheckResult:
    status: CheckStatus
    passed_count: int = 0
    failed_count: int = 0
    warning_count: int = 0
    skipped_count: int = 0
    failures: tuple[ValidationFailure, ...] = ()
    execution_time_ms: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckResult: ...
```

### ProfileResult

```python
@dataclass(frozen=True)
class ProfileResult:
    status: ProfileStatus = ProfileStatus.COMPLETED
    columns: tuple[ColumnProfile, ...] = ()
    row_count: int = 0
    execution_time_ms: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
```

### LearnResult

```python
@dataclass(frozen=True)
class LearnResult:
    status: LearnStatus = LearnStatus.COMPLETED
    rules: tuple[LearnedRule, ...] = ()
    execution_time_ms: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
```

## logging.py

### TruthoundLogger

```python
class TruthoundLogger:
    def __init__(self, name: str): ...
    def debug(self, message: str, **kwargs): ...
    def info(self, message: str, **kwargs): ...
    def warning(self, message: str, **kwargs): ...
    def error(self, message: str, **kwargs): ...
    def critical(self, message: str, **kwargs): ...
```

### LogContext

```python
class LogContext:
    def __init__(self, **context): ...
    def __enter__(self): ...
    def __exit__(self, *args): ...
```

### PerformanceLogger

```python
class PerformanceLogger:
    def timed(self, name: str, **context): ...
    def timed_decorator(self): ...
```

## retry.py

### RetryConfig

```python
@dataclass(frozen=True)
class RetryConfig:
    max_attempts: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    jitter: bool = True
    exceptions: tuple[type[Exception], ...] = (Exception,)
    non_retryable: tuple[type[Exception], ...] = ()
```

### RetryStrategy

```python
class RetryStrategy(Enum):
    FIXED = "fixed"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIBONACCI = "fibonacci"
```

## circuit_breaker.py

### CircuitBreakerConfig

```python
@dataclass(frozen=True)
class CircuitBreakerConfig:
    failure_threshold: int = 5
    success_threshold: int = 2
    recovery_timeout_seconds: float = 30.0
    exceptions: tuple[type[Exception], ...] = (Exception,)
    ignored: tuple[type[Exception], ...] = ()
    name: str | None = None
```

### CircuitState

```python
class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"
```

## health.py

### HealthStatus

```python
class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
```

### HealthCheckResult

```python
@dataclass(frozen=True)
class HealthCheckResult:
    name: str
    status: HealthStatus
    message: str | None = None
    duration_ms: float | None = None
    details: Mapping[str, Any] = field(default_factory=dict)
```

## metrics.py

### MetricType

```python
class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
```

### Counter

```python
class Counter:
    def inc(self, value: float = 1, labels: dict | None = None): ...
```

### Gauge

```python
class Gauge:
    def set(self, value: float, labels: dict | None = None): ...
    def inc(self, value: float = 1, labels: dict | None = None): ...
    def dec(self, value: float = 1, labels: dict | None = None): ...
```

### Histogram

```python
class Histogram:
    def observe(self, value: float, labels: dict | None = None): ...
    def time(self): ...  # Context manager
```

### Summary

```python
class Summary:
    def observe(self, value: float, labels: dict | None = None): ...
```

## rate_limiter.py

### RateLimitConfig

```python
@dataclass(frozen=True)
class RateLimitConfig:
    max_requests: int = 100
    window_seconds: float = 60.0
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.TOKEN_BUCKET
    burst_size: int | None = None
    on_limit: RateLimitAction = RateLimitAction.REJECT
```

### RateLimitAlgorithm

```python
class RateLimitAlgorithm(Enum):
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    LEAKY_BUCKET = "leaky_bucket"
```

## cache.py

### CacheConfig

```python
@dataclass(frozen=True)
class CacheConfig:
    max_size: int = 1000
    ttl_seconds: float | None = None
    eviction_policy: EvictionPolicy = EvictionPolicy.LRU
    namespace: str | None = None
```

### EvictionPolicy

```python
class EvictionPolicy(Enum):
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    FIFO = "fifo"
    NONE = "none"
```
