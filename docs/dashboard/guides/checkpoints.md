# Checkpoints

Schedules, notifications, alerts, throttling, escalation, and trigger
monitoring are treated as one checkpoint family.

## Truthound mapping

- `truthound.checkpoint.Checkpoint`
- `truthound.checkpoint.AsyncCheckpoint`
- `truthound.checkpoint.monitoring`
- `truthound.checkpoint.ci`
- `truthound.checkpoint.throttling`
- `truthound.checkpoint.escalation`

## Result handling

Checkpoint views normalize Truthound 3.0 checkpoint results through:

- `validation_run`
- `validation_view`

This replaces older `validation_result` assumptions.
