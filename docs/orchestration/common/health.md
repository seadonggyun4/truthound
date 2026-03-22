---
title: Health Check
---

# Health Check

Monitors component status and verifies system health.

## Basic Usage

```python
from common.health import health_check, HealthCheckConfig, HealthCheckResult

# Simple decorator usage
@health_check(name="database", timeout_seconds=5.0)
def check_database():
    return db.ping()

# Direct result return
@health_check(name="cache")
def check_cache():
    if cache.is_connected():
        return HealthCheckResult.healthy("cache", message="Cache is ready")
    return HealthCheckResult.unhealthy("cache", message="Cache unavailable")
```

## Preset Configurations

```python
from common.health import (
    DEFAULT_HEALTH_CHECK_CONFIG,    # Default: 5s timeout, no cache
    FAST_HEALTH_CHECK_CONFIG,       # Fast: 1s timeout, 10s cache
    THOROUGH_HEALTH_CHECK_CONFIG,   # Thorough: 30s timeout
    CACHED_HEALTH_CHECK_CONFIG,     # Cached: 60s cache
)
```

## Status Values

| Status | Description |
|--------|-------------|
| `HEALTHY` | Fully operational |
| `DEGRADED` | Operational but with reduced performance |
| `UNHEALTHY` | Not operational |
| `UNKNOWN` | Status cannot be determined |

## Composite Health Check

Aggregates the status of multiple components:

```python
from common.health import (
    SimpleHealthChecker,
    CompositeHealthChecker,
    AggregationStrategy,
)

# Create individual checkers
db_checker = SimpleHealthChecker("database", lambda: db.ping())
cache_checker = SimpleHealthChecker("cache", lambda: cache.ping())
api_checker = SimpleHealthChecker("api", lambda: api.health())

# Create composite checker
system_checker = CompositeHealthChecker(
    name="system",
    checkers=[db_checker, cache_checker, api_checker],
    strategy=AggregationStrategy.WORST,  # Use worst status
    parallel=True,  # Parallel execution
)

result = system_checker.check()
print(f"System status: {result.status.name}")
print(f"Healthy: {result.details['healthy_count']}/{result.details['total_count']}")
```

## Aggregation Strategies

| Strategy | Description |
|----------|-------------|
| `WORST` | Use worst status |
| `BEST` | Use best status |
| `MAJORITY` | Majority vote |
| `ALL_HEALTHY` | HEALTHY only if all are healthy |
| `ANY_HEALTHY` | HEALTHY if any is healthy |

## Registry Usage

```python
from common.health import (
    register_health_check,
    check_health,
    check_all_health,
    HealthCheckRegistry,
)

# Register with global registry
register_health_check("database", lambda: db.ping())
register_health_check("cache", lambda: cache.ping())

# Individual check
result = check_health("database")

# Full check
result = check_all_health(parallel=True)
```

## Hooks

```python
from common.health import (
    health_check,
    LoggingHealthCheckHook,
    MetricsHealthCheckHook,
)

# Logging hook: automatic check event logging
logging_hook = LoggingHealthCheckHook()

# Metrics hook: statistics collection
metrics_hook = MetricsHealthCheckHook()

@health_check(name="api", hooks=[logging_hook, metrics_hook])
def check_api():
    return api.health()

# Query metrics
print(metrics_hook.get_check_count("api"))
print(metrics_hook.get_average_duration_ms("api"))
print(metrics_hook.get_status_counts("api"))
```

## Asynchronous Health Check

```python
from common.health import AsyncSimpleHealthChecker, AsyncHealthChecker

async_checker = AsyncSimpleHealthChecker(
    name="async_db",
    check_fn=lambda: db.async_ping(),
)

result = await async_checker.check()
```
