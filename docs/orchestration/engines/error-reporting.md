---
title: Engine Error Reporting and Diagnostics
---

# Engine Error Reporting and Diagnostics

Engine failures become much easier to operate when teams distinguish among
capability errors, source errors, execution failures, and serialization drift.

## Who This Is For

- operators reading engine-level failures in CI or production
- contributors adding diagnostics to adapters or engines

## When To Use It

Use this page when an engine-related failure does not clearly belong to the
host itself.

## Diagnostic Layers

| Layer | Typical Question |
|-------|------------------|
| registry | did we create the engine we expected? |
| versioning | is this engine version compatible with the supported tuple? |
| capability | does the engine support the requested operation? |
| execution | did the engine fail while running the operation? |
| serialization | did the result survive transport back into the host? |

## Practical Reading Order

1. confirm engine selection and runtime context
2. inspect the compatibility or preflight report
3. inspect host-native logs and emitted result metadata
4. compare the failure against the shared [Failure Catalog](../common/failure-catalog.md)

## Production Pattern

- keep engine diagnostics visible in host-native logs
- treat preflight failures as configuration issues, not as runtime flukes
- keep result consumers on the shared serializer contract

## Related Pages

- [Failure Catalog](../common/failure-catalog.md)
- [Capability Matrix](capability-matrix.md)
- [Lifecycle](lifecycle.md)
