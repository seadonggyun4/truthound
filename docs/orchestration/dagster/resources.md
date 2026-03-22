---
title: Dagster Resources
---

# Dagster Resources

`DataQualityResource` is the canonical Dagster boundary for Truthound orchestration. It is the best choice when you want consistent engine behavior across assets, ops, and asset checks without repeating configuration everywhere.

## DataQualityResource

```python
from truthound_dagster.resources import DataQualityResource

resource = DataQualityResource(
    engine_name="truthound",
    timeout_seconds=300.0,
)
```

The resource exposes the shared runtime through host-native methods:

| Method | Purpose |
|--------|---------|
| `check(...)` | validation |
| `profile(...)` | profiling |
| `learn(...)` | rule learning |
| `stream_check(...)` | bounded-memory streaming validation |

## Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `engine_name` | str | `"truthound"` | engine choice |
| `timeout_seconds` | float | `300.0` | default operation timeout |
| `fail_on_error` | bool | `True` | hard-fail check behavior |
| `warning_threshold` | float \| None | `None` | warning threshold instead of fail |
| `parallel` | bool | `False` | parallel Truthound execution |
| `max_workers` | int \| None | `None` | worker limit for parallel execution |
| `observability` | dict \| None | `None` | shared observability settings |

## When To Use EngineResource

`EngineResource` is the lower-level boundary when you want more direct control of engine lifecycle or configuration than the higher-level `DataQualityResource` surface.

## Preset Configurations

Use preset configuration constants when you want a named configuration style instead of repeating field values.

## Resource Methods

```python
result = resource.check(data, rules=[{"column": "id", "type": "not_null"}])
profile = resource.profile(data)
learned = resource.learn(data)
```

## Lifecycle

The resource initializes the underlying engine during setup and stops it during teardown. That keeps lifecycle behavior aligned with Dagster's resource model instead of hiding it inside random user code.

## Usage in Definitions

```python
from dagster import Definitions
from truthound_dagster.resources import DataQualityResource

defs = Definitions(
    resources={"data_quality": DataQualityResource()},
    assets=[users_asset],
)
```

## Related Reading

- [Dagster Overview](index.md)
- [Ops](ops.md)
- [Assets and Asset Checks](assets.md)
