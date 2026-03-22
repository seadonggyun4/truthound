---
title: Prefect Tasks
---

# Prefect Tasks

Tasks are the easiest Prefect execution surface to adopt. Use them when a flow already exists and you want to insert validation without committing to persisted block configuration yet.

## Pre-built Tasks

### `data_quality_check_task`

Use it for standard validation inside a flow.

### `data_quality_profile_task`

Use it when the flow should profile a dataset.

### `data_quality_learn_task`

Use it to learn candidate rules from baseline data.

## Specialized Tasks

Other useful helpers include:

- strict and lenient check presets
- auto-schema check helpers
- profile presets
- streaming tasks

## Task Factories

Use `create_check_task`, `create_profile_task`, and related factory helpers when you want reusable task construction without rewriting shared logic.

## Task Configuration

`CheckTaskConfig`, `ProfileTaskConfig`, and `LearnTaskConfig` expose task-level policy choices while keeping task wiring consistent.

## Task Chaining

Prefect tasks compose well when you separate load, validate, profile, notify, and persist steps.

## Using With Blocks

Pass a `DataQualityBlock` when the flow should consume saved configuration instead of ephemeral defaults.

## Retry Configuration

Keep business logic retries and platform retries understandable. Do not hide configuration mistakes behind excessive retry settings.

## Related Reading

- [Blocks](blocks.md)
- [Flows](flows.md)
- [Deployment Patterns](deployment-patterns.md)
