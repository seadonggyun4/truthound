---
title: Dagster Ops
---

# Dagster Ops

Prebuilt ops are the right choice when you want Dagster jobs to compose explicit validation stages instead of putting all behavior inside assets.

## Pre-built Ops

### `data_quality_check_op`

Use it for standard validation steps inside jobs and graphs.

### `data_quality_profile_op`

Use it when the job needs descriptive profile output.

### `data_quality_learn_op`

Use it when the job should infer candidate rules from baseline data.

## Op Factories

Use `create_check_op`, `create_profile_op`, and `create_learn_op` when you want preconfigured ops without rewriting boilerplate.

## Op Configuration

`CheckOpConfig`, `ProfileOpConfig`, and `LearnOpConfig` expose operation-level configuration surfaces for explicit job composition.

## Preset Configurations

`STRICT_CHECK_CONFIG` and `LENIENT_CHECK_CONFIG` are useful when teams want named policy modes inside job code.

## Op Chaining

Ops are a strong fit when you want separate stages for:

- load
- validate
- profile
- decide
- notify or persist

## Inputs And Outputs

The op outputs follow the same shared result contract used across the repository, with Dagster-native wrapping where appropriate.

## Related Reading

- [Resources](resources.md)
- [Assets and Asset Checks](assets.md)
- [Recipes](recipes.md)
