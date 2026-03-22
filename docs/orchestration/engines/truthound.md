---
title: Truthound Engine
---

# Truthound Engine

`TruthoundEngine` is the default engine for the orchestration stack. It is the best
fit for teams that want Truthound-first zero-config behavior and the broadest alignment
with the rest of the adapter family in this repository.

## When To Use It

Use `TruthoundEngine` when:

- you want the default and best-supported path
- you are onboarding to Truthound 3.x for the first time
- you want rule vocabulary and runtime semantics to match the shipped adapters closely

## Basic Pattern

```python
from common.engines import TruthoundEngine

with TruthoundEngine() as engine:
    result = engine.check(data, auto_schema=True)
```

## Main Operations

- `check`: run rules or auto-schema-backed validation
- `profile`: inspect shape and quality characteristics
- `learn`: infer rules or baseline expectations from data

## Why It Is The Recommended Default

- closest alignment with the shared runtime layer
- strong zero-config behavior
- best coverage in the repository's adapter examples and CI
- natural fit for teams using Truthound as the primary quality system rather than as an
  adapter for an older investment

## Production Pattern

- use `auto_schema=True` for fast onboarding or exploratory flows
- promote learned or explicit rules for datasets with production contracts
- keep lifecycle and metrics hooks enabled in long-running orchestrators

## Failure Modes

- preflight failures usually mean runtime compatibility issues, not engine bugs
- inconsistent source shapes often belong in source resolution or serialization, not in
  the engine itself

## Related Pages

- [Batch Processing](batch.md)
- [Lifecycle Management](lifecycle.md)
