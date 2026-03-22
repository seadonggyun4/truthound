---
title: Mage Overview
---

# Truthound for Mage

Truthound's Mage integration is designed for teams that want pipeline-native data
quality blocks without introducing a separate orchestration control plane. The package
leans on Mage conventions such as transformers, sensors, conditions, and `io_config`
discovery instead of forcing a custom runtime model.

## Who This Is For

- Mage teams building pipeline-local quality checks
- platform engineers standardizing data source discovery and output handling
- operators who want SLA-aware gating inside Mage blocks

## When To Use Mage

Choose Mage when:

- validation belongs directly in pipeline blocks
- your team already relies on Mage project conventions such as `io_config.yaml`
- you want a simple handoff from raw extract to quality gate to downstream load

Choose Airflow, Dagster, or Prefect when you need richer scheduler semantics, broader
deployment control, or host-native orchestration abstractions.

## What The Package Provides

- transformers for check, profile, and learn workflows
- sensor and condition blocks for gate-style branching
- `io_config.yaml` discovery and source normalization
- shared runtime preflight, observability, and serialization behavior
- SLA monitoring hooks and presets

## Minimal Quickstart

```python
from truthound_mage import CheckBlockConfig, CheckTransformer

transformer = CheckTransformer(
    config=CheckBlockConfig(
        rules=[
            {"column": "id", "check": "not_null"},
            {"column": "email", "check": "email_format"},
        ]
    )
)

result = transformer.execute(dataframe)
```

## Production Pattern

The most maintainable Mage layout is:

- extract or load data with the project's normal Mage loaders
- run `CheckTransformer` or `ProfileTransformer` in a dedicated transformer block
- gate downstream work with `DataQualitySensor` or `DataQualityCondition`
- source connection details from `io_config.yaml` or environment-backed values
- enable SLA hooks for operator visibility

## Shared Runtime Behavior

Mage inherits the same shared runtime guarantees as the other adapters:

- source resolution rules
- preflight compatibility checks
- normalized result serialization
- observability and resilience helpers

Start with:

- [Shared Runtime Overview](../common/index.md)
- [Source Resolution](../common/source-resolution.md)
- [Observability and Resilience](../common/observability-resilience.md)

## Recommended Reading Order

- [Project Layout](project-layout.md)
- [`io_config.yaml`](io-config.md)
- [Recipes](recipes.md)
- [Troubleshooting](troubleshooting.md)
