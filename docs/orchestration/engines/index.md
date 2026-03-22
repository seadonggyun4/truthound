---
title: Engines Overview
---

# Data Quality Engines

The orchestration adapters in this repository are host-specific, but the engine layer
is host-agnostic. It lets the same orchestration surface run on top of Truthound,
Pandera, Great Expectations, and custom engine implementations.

## What The Engine Layer Does

- normalizes validation capabilities behind shared protocols
- resolves and creates engines through a registry and resolver
- runs preflight compatibility checks before execution
- exposes optional advanced features such as batch execution, lifecycle hooks, and
  capability-aware routing

## Core Concepts

### Protocols

The shared contract starts with `DataQualityEngine` and related capability metadata in
`common/engines/base.py`.

### Registry And Resolver

Use the registry when you need to list, fetch, or override engines by name:

```python
from common.engines import get_engine, list_engines, register_engine

engine = get_engine("truthound")
available = list_engines()
```

Use the resolver when you need a runtime-aware engine creation path:

```python
from common.engines import EngineCreationRequest, create_engine, run_preflight
from common.runtime import normalize_runtime_context

runtime_context = normalize_runtime_context(platform="prefect")
request = EngineCreationRequest(engine_name="truthound", runtime_context=runtime_context)
preflight = run_preflight(request)

if preflight.compatible:
    engine = create_engine(request)
```

## When To Change Engines

Most teams should stay on the default Truthound engine unless they already have a
strong Pandera or Great Expectations investment. Switching engines is valuable when:

- a team wants to preserve existing schema assets
- a project needs a stricter contract-first workflow
- a migration is happening gradually across multiple validation stacks

## Advanced Capabilities

The engine layer also includes:

- [Truthound Engine](truthound.md)
- [Pandera](pandera.md)
- [Great Expectations](great-expectations.md)
- [Batch Processing](batch.md)
- [Streaming Validation](streaming.md)
- [Engine Chains](chain.md)
- [Lifecycle Management](lifecycle.md)
- [Drift Detection](drift-detection.md)
- [Anomaly Detection](anomaly-detection.md)

## Production Guidance

- treat preflight failures as configuration or compatibility failures, not just runtime
  noise
- keep engine selection explicit in production environments
- use host-level docs for orchestration patterns and engine docs for validation
  semantics

## Related Pages

- [Shared Runtime Overview](../common/index.md)
- [Preflight and Compatibility](../common/preflight-compatibility.md)
