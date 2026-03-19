# ADR 003: Result Model

## Status

Accepted

## Context

Legacy reporting mixed presentation concerns with execution state, which made it difficult to evolve reporters and runtime behavior independently.

## Decision

Make `ValidationRunResult` the canonical runtime output and separate:

- per-check verdicts through `CheckResult`
- issue evidence through `ValidationIssue`
- runtime failures through `ExecutionIssue`

The legacy `Report` remains as an adapter.

## Consequences

- reporters can move toward a stable structured model
- runtime failures are no longer hidden inside presentation types
- compatibility remains available during migration
