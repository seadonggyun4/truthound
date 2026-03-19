# ADR 001: Validation Kernel

## Status

Accepted

## Context

Validation orchestration had become concentrated in API and validator base layers, which increased coupling and made execution strategy changes expensive.

## Decision

Introduce `truthound.core` and fix the internal boundary at five packages:

- `contracts`
- `suite`
- `planning`
- `runtime`
- `results`

`th.check()` remains as a compatibility facade over this kernel.

## Consequences

- execution concerns become testable without the full public facade
- planning and runtime can evolve independently
- extension points become clearer for plugins and future backends
