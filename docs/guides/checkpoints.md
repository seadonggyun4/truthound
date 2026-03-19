# Checkpoints

Truthound 3.0 checkpoints execute validation through the core result model and keep `CheckpointResult.validation_run` as the only canonical runtime result field.

## Runtime Model

- `CheckpointResult.validation_run` holds the canonical `ValidationRunResult`
- `CheckpointResult.validation_view` is the compatibility projection used by legacy action formatting code
- `CheckpointResult.validation_result` is removed in 3.0

## Store Boundary

Persistence actions such as `StoreValidationResult` may still serialize a storage-oriented DTO, but that conversion happens only at the persistence edge. Checkpoint execution, routing, docs updates, CI summaries, and notification actions should work from `validation_run` or `validation_view`.

## Generated Docs

Validation docs are generated from `ValidationRunResult` directly. Checkpoint actions that publish docs should not reconstruct a second report model.

## Operational Guidance

- treat checkpoint runs as orchestration over the kernel, not a second validation engine
- keep routing, throttling, escalation, and notifications on canonical checkpoint metadata plus `validation_run`
- if a consumer needs summary-oriented issue counts, prefer `validation_view` rather than ad hoc transformations

Use this guide as the canonical landing page for checkpoint orchestration, store boundaries, and generated validation docs.

Related references:

- [Architecture](../concepts/architecture.md)
- [Zero-Config Context](../concepts/zero-config.md)
- [Migration to 3.0](migration-3.0.md)
- [Legacy Archive](../legacy/index.md)
