---
title: Great Expectations Adapter
---

# Great Expectations Adapter

`GreatExpectationsAdapter` exists for teams that need to preserve prior Great
Expectations investment while moving orchestration and runtime handling toward the
Truthound 3.x model.

## When To Use It

Use the Great Expectations adapter when:

- expectations already exist and cannot be replaced immediately
- the organization wants shared orchestration semantics before fully migrating rule
  assets

## Strengths

- lets existing expectation work participate in the same host integrations
- creates a migration bridge instead of forcing a hard cutover

## Tradeoffs

- adapter behavior may not expose every Truthound-native capability the same way
- teams should expect some conceptual mismatch between expectation-first and
  Truthound-first validation styles

## Production Guidance

- use it as a managed transition path, not only as a permanent abstraction layer
- keep shared runtime behavior explicit so operators understand which parts come from
  the host adapter and which come from the underlying engine

## Related Pages

- [Truthound Engine](truthound.md)
- [Pandera Adapter](pandera.md)
