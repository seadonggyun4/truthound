# Checkpoints

Truthound checkpoints are the bridge between local validation and operational workflows. Use this page as the entry point for CI/CD runs, scheduled validation, routing, escalation, notifications, and production-oriented checkpoint design.

## Who This Guide Is For

Use the checkpoint docs when you need to:

- run repeatable validation pipelines in CI or scheduled jobs
- connect validation outcomes to routing, throttling, escalation, and notifications
- publish docs or persist results from checkpoint execution
- understand how the checkpoint layer relates to the 3.0 kernel and result model

## The Checkpoint Mental Model

Treat checkpoints as orchestration over the kernel, not as a second validation engine.

- validation still resolves to the canonical `ValidationRunResult`
- checkpoint metadata adds automation context such as run status, actions, and routing state
- persistence, notifications, CI summaries, and docs updates should consume `validation_run` or `validation_view`, not reconstruct a parallel result model

## Start Here

| Task | Best page |
|------|-----------|
| Understand the checkpoint system end to end | [Checkpoint Family Overview](checkpoint/index.md) |
| Build your first checkpoint | [Basics](checkpoint/basics.md) |
| Add scheduled or event-driven execution | [Triggers](checkpoint/triggers.md) |
| Route issues to different teams or workflows | [Routing](checkpoint/routing.md) |
| Control notification volume or duplicate runs | [Throttling](checkpoint/throttling.md) and [Deduplication](checkpoint/deduplication.md) |
| Add multi-step operational handling | [Escalation](checkpoint/escalation.md) |
| Understand async or higher-throughput execution | [Async Execution](checkpoint/async.md) |
| Integrate with CI vendors | [CI Platforms](checkpoint/ci-platforms.md) and [CLI checkpoint commands](../cli/checkpoint/index.md) |

## Recommended Reading Path

1. [Checkpoint Family Overview](checkpoint/index.md)
2. [Basics](checkpoint/basics.md)
3. [Triggers](checkpoint/triggers.md)
4. [Routing](checkpoint/routing.md)
5. [Escalation](checkpoint/escalation.md)
6. [CI Platforms](checkpoint/ci-platforms.md)

## Runtime Model

- `CheckpointResult.validation_run` holds the canonical `ValidationRunResult`
- `CheckpointResult.validation_view` is the compatibility summary surface for legacy formatting paths
- `CheckpointResult.validation_result` is removed in 3.0

## Operational Boundaries

### Persistence boundary

Storage-facing DTO conversion is allowed at the persistence edge, but checkpoint orchestration should remain centered on canonical runtime results.

### Reporting boundary

Validation docs and reporters should build from `ValidationRunResult` directly rather than inventing a second report model.

### Automation boundary

Routing, throttling, escalation, and notifications should work from checkpoint metadata plus `validation_run`/`validation_view`.

## Related Reading

- [Checkpoint Family Overview](checkpoint/index.md)
- [Configuration Guide](configuration/index.md)
- [Architecture](../concepts/architecture.md)
- [Zero-Config Context](../concepts/zero-config.md)
- [Migration to 3.0](migration-3.0.md)
