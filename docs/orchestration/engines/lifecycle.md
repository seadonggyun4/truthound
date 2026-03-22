---
title: Lifecycle Management
---

# Lifecycle Management

Long-running orchestration hosts need more than a bare `check()` call. The engine layer
includes lifecycle helpers for startup, shutdown, health checks, and hook-based
instrumentation.

## Main Components

- `ManagedEngineMixin`
- `EngineLifecycleManager`
- `EngineHealthChecker`
- lifecycle hooks such as `LoggingLifecycleHook` and `MetricsLifecycleHook`

## When This Matters

Lifecycle management is especially important in:

- Airflow workers and sensors
- Dagster or Prefect processes with reusable resources or blocks
- services that hold engine instances across multiple requests

## Basic Pattern

```python
from common.engines import EngineLifecycleManager, TruthoundEngine

engine = TruthoundEngine()
manager = EngineLifecycleManager(engine)
manager.start()
health = manager.health_check()
manager.stop()
```

## Production Guidance

- use context managers where possible
- keep health checks separate from normal validation output
- wire lifecycle hooks into the same logging and metrics systems used by the host

## Related Pages

- [Batch Processing](batch.md)
- [Shared Runtime Overview](../common/index.md)
