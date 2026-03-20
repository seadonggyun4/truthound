---
title: API Reference
---

# API Reference

## Shared Runtime APIs

### Engine Creation

```python
from common.engines import EngineCreationRequest, create_engine
```

Use `create_engine(...)` with either a string engine name or an `EngineCreationRequest`.

### Runtime Context

```python
from common.engines import PlatformRuntimeContext, AutoConfigPolicy, normalize_runtime_context
```

Use runtime context objects to describe the platform, connection hints, source descriptors, and zero-config policy.

### Source Resolution

```python
from common.engines import resolve_data_source
```

Supported normalized source kinds:

- dataframe
- local path
- remote URI
- SQL
- callable
- generic object

### Compatibility And Preflight

```python
from common.engines import build_compatibility_report, run_preflight
```

These APIs validate engine resolution, serializer readiness, and whether a source needs credentials that have not been supplied.

## Result Serialization

```python
from common.serializers import serialize_result_wire
```

The shared wire format is the contract that platform packages should build on.
