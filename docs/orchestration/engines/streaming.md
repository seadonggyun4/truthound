---
title: Streaming Validation
---

# Streaming Validation

Streaming validation is the capability-aware path for validating data in batches or
incremental chunks without treating the entire input as one eager dataset.

## When To Use It

Use streaming validation when:

- data arrives incrementally
- memory pressure makes eager validation unattractive
- operators want early failure or incremental visibility

## Capability Model

Not every engine supports streaming. Check capabilities before designing a streaming
workflow around a specific engine:

```python
from common.engines import get_engine

engine = get_engine("truthound")
supports_streaming = engine.get_capabilities().supports_streaming
```

## Operational Pattern

- normalize the source first
- validate each chunk or batch
- decide whether to fail fast or aggregate later
- emit metrics and summaries at the host level

## Shared Guidance

Streaming works best when paired with:

- shared result serialization
- host-native retries and backoff
- explicit observability around chunk failures and lag

## Related Pages

- [Batch Processing](batch.md)
- [Observability and Resilience](../common/observability-resilience.md)
