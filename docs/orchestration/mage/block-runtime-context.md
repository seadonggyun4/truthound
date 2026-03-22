---
title: Mage Block Types and Runtime Context
---

# Mage Block Types and Runtime Context

Truthound's Mage package is organized around Mage-native execution units: transformers, sensors, and conditions. The key runtime bridge is `BlockExecutionContext`, which carries the metadata and behavior needed to keep validation consistent across blocks.

## Who This Is For

- Mage authors deciding which block type should own validation
- teams standardizing block-level metadata and execution context
- operators debugging runtime differences between blocks

## When To Use It

Use this page when:

- a pipeline needs a transformer, sensor, or condition quality boundary
- you want to understand what `BlockExecutionContext` contributes
- runtime metadata should be stable across multiple Mage block types

## Prerequisites

- `truthound-orchestration[mage]` installed
- a Mage project with pipeline blocks
- familiarity with transformers, sensors, and conditions in Mage

## Minimal Quickstart

Use a transformer when the block should actively execute validation:

```python
from truthound_mage import CheckBlockConfig, CheckTransformer

transformer = CheckTransformer(
    config=CheckBlockConfig(
        rules=[{"column": "id", "check": "not_null"}],
    )
)

result = transformer.execute(dataframe)
```

Use an explicit execution context when metadata or environment hints matter:

```python
from truthound_mage import BlockExecutionContext

context = BlockExecutionContext()
result = transformer.execute(dataframe, context=context)
```

## Production Pattern

Use this decision table:

| Block Type | Best Fit |
|-----------|----------|
| `CheckTransformer` / `ProfileTransformer` / `LearnTransformer` | active execution inside a pipeline step |
| `DataQualitySensor` | waiting or gating behavior before later steps |
| `DataQualityCondition` | branching and pass/fail control logic |

Recommended context policy:

- use `BlockExecutionContext` whenever execution metadata matters
- keep block config focused on rules and runtime behavior, not unrelated pipeline concerns
- use one block per quality responsibility instead of mixing check, profile, and branch logic together

## Failure Modes and Troubleshooting

| Symptom | Likely Cause | What To Do |
|--------|--------------|------------|
| quality behavior differs across blocks | context metadata is missing or inconsistent | standardize `BlockExecutionContext` usage |
| pipelines become hard to reason about | one block handles validation and branching together | separate execution blocks from condition blocks |
| operators cannot tell which step owns the failure | block naming and metadata are vague | align block names with dataset and operation |

## Related Pages

- [Mage Overview](index.md)
- [Project Layout](project-layout.md)
- [Mage Recipes](recipes.md)
