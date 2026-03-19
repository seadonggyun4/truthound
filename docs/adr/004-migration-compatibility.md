# ADR 004: Migration Compatibility

## Status

Accepted

## Context

Truthound needed a structural redesign without forcing every user and integration to migrate immediately.

## Decision

Preserve the top-level facade while internally redirecting execution to the new kernel. Keep legacy reporting available through adapters and expose the structured runtime model through `report.validation_run`.

## Consequences

- existing user code keeps working in the common path
- advanced integrations gain immediate access to the new model
- deeper subsystem migrations can proceed incrementally
