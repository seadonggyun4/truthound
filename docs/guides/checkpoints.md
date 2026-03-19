# Checkpoints

Truthound 2.0 checkpoints execute validation through the core result model and expose `CheckpointResult.validation_run` as the canonical runtime result.

Compatibility notes:

- `CheckpointResult.validation_result` remains available for one migration cycle as a deprecated compatibility view.
- Persistence actions such as `StoreValidationResult` convert `validation_run` into the storage DTO only at the persistence boundary.
- Data docs and reporter integrations should consume `validation_run` or `ValidationRunResult` directly.

Use this guide as the canonical landing page for checkpoint orchestration, store boundaries, and generated validation docs.

Related references:

- [Architecture](../concepts/architecture.md)
- [Migration to 2.0](migration-2.0.md)
- [Legacy Archive](../legacy/index.md)
